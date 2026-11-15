//! Dynamics per bin: a compressor above its threshold and a downward expander
//! below theirs, applied to each FFT bin's magnitude instead of the waveform.
//!
//! The curves are the same ones the time-domain effects use
//! ([crate::tools::gain_computer]), so a bin and a sample agree on what four to
//! one means. What the page draws is a pair of curve pages; what the audio
//! thread reads is a dense table rebuilt whenever a page or the morph moves.
//!
//! The level a bin reports is its FFT magnitude divided by the window size, so
//! a full-scale sine sits near 0 dB whatever the FFT length is.

use std::sync::Arc;

use i_am_dsp_derive::Parameters;

use crate::prelude::{FrequencyMapper, page_position};
use crate::tools::curve_pages::CurvePages;
use crate::tools::gain_computer::{CompanderCurve, db_to_linear, linear_to_db};
use crate::tools::page::{blend, segment_bend};

/// How far a curve point at 0 or 1 moves its bin's threshold, in dB.
const CURVE_RANGE_DB: f32 = 36.0;
/// The bottom of the curve's frequency axis.
const MIN_HZ: f32 = 20.0;
/// The top of the curve's frequency axis.
const MAX_HZ: f32 = 20_000.0;

/// One breakpoint of a curve page.
///
/// `time` is a position on the page's logarithmic frequency axis, 0 at 20 Hz and
/// 1 at 20 kHz. `value` is read by the curve it belongs to: a threshold curve
/// treats 0.5 as "no offset", a ratio curve treats 1 as "the full ratio".
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CurvePoint {
	/// Position across the frequency axis, 0..1.
	pub time: f32,
	/// The value at that position, 0..1.
	pub value: f32,
	/// How the segment that leaves this point bends, as the editor draws it.
	#[serde(default)]
	pub bend: f32,
}

impl CurvePoint {
	/// A point on the straight line between its neighbours.
	pub fn new(time: f32, value: f32) -> Self {
		Self {
			time,
			value,
			bend: 0.0,
		}
	}
}

/// The pages a page editor writes: one set for the threshold, one for the ratio.
///
/// Both are shared with the effect, so a page edit reaches the audio thread
/// without a lock. The default pages are flat: a threshold page at 0.5 offsets
/// nothing, and a ratio page at 1 applies the master ratio to every bin.
pub struct DynamicsCurves {
	threshold: CurvePages<CurvePoint>,
	ratio: CurvePages<CurvePoint>,
}

impl DynamicsCurves {
	/// Flat pages: no threshold tilt, and the master ratio everywhere.
	pub fn new() -> Arc<Self> {
		Arc::new(Self {
			threshold: CurvePages::new(flat(0.5)),
			ratio: CurvePages::new(flat(1.0)),
		})
	}

	/// The threshold pages.
	pub fn threshold(&self) -> &CurvePages<CurvePoint> {
		&self.threshold
	}

	/// The ratio pages.
	pub fn ratio(&self) -> &CurvePages<CurvePoint> {
		&self.ratio
	}

	/// Replaces the threshold pages, normalising them first.
	pub fn set_threshold(&self, pages: Vec<Vec<CurvePoint>>) {
		self.threshold.replace(normalize(pages, 0.5));
	}

	/// Replaces the ratio pages, normalising them first.
	pub fn set_ratio(&self, pages: Vec<Vec<CurvePoint>>) {
		self.ratio.replace(normalize(pages, 1.0));
	}

	/// Both page lists, for the page route and the state save.
	pub fn pages(&self) -> (Vec<Vec<CurvePoint>>, Vec<Vec<CurvePoint>>) {
		(self.threshold.pages(), self.ratio.pages())
	}
}

/// A flat page at one value.
fn flat(value: f32) -> Vec<CurvePoint> {
	vec![CurvePoint::new(0.0, value), CurvePoint::new(1.0, value)]
}

/// Clamps, sorts and fills pages so the audio side can index them without a check.
fn normalize(mut pages: Vec<Vec<CurvePoint>>, default_value: f32) -> Vec<Vec<CurvePoint>> {
	if pages.is_empty() {
		pages.push(flat(default_value));
	}

	for page in &mut pages {
		if page.len() < 2 {
			*page = flat(default_value);
		}

		for point in page.iter_mut() {
			point.time = point.time.clamp(0.0, 1.0);
			point.value = point.value.clamp(0.0, 1.0);
		}

		page.sort_by(|left, right| {
			left.time
				.partial_cmp(&right.time)
				.unwrap_or(core::cmp::Ordering::Equal)
		});
	}

	pages
}

/// The value a page has at a position on its axis.
fn sample_curve(page: &[CurvePoint], position: f32) -> f32 {
	let Some(first) = page.first() else {
		return 0.5;
	};

	if position <= first.time {
		return first.value;
	}

	for pair in page.windows(2) {
		let (from, to) = (pair[0], pair[1]);

		if position <= to.time {
			let span = to.time - from.time;
			let fraction = if span > 1e-9 {
				(position - from.time) / span
			} else {
				1.0
			};

			return blend(
				from.value,
				to.value,
				segment_bend(fraction.clamp(0.0, 1.0), from.bend),
			);
		}
	}

	page[page.len() - 1].value
}

/// The bin's centre frequency on the curve axis, 0..1.
fn position_of(frequency: f32, sample_rate: f32) -> f32 {
	let frequency = frequency.max(MIN_HZ);
	let nyquist = (sample_rate * 0.5).max(MAX_HZ);
	let frequency = frequency.min(nyquist);

	(frequency / MIN_HZ).ln() / (MAX_HZ / MIN_HZ).ln()
}

/// A compressor and a downward expander on every FFT bin.
#[derive(Parameters)]
pub struct SpectralDynamics {
	/// The level above which a bin is compressed.
	#[range(min = -60.0, max = 0.0)]
	#[unit(decibels)]
	#[default(-24.0)]
	#[id(display = "Compress")]
	#[layout(group = "Dynamics", order = 0, widget = knob, hero = "spectraldynamics-spectrum")]
	pub compress_threshold: f32,

	/// How much a bin above the threshold is turned down; 1 is off.
	#[range(min = 1.0, max = 20.0)]
	#[default(4.0)]
	#[id(display = "Ratio")]
	#[layout(group = "Curve", order = 0, widget = knob)]
	pub compress_ratio: f32,

	/// The level below which a bin is pushed further down.
	#[range(min = -80.0, max = -6.0)]
	#[unit(decibels)]
	#[default(-48.0)]
	#[id(display = "Expand")]
	#[layout(group = "Dynamics", order = 2, widget = knob)]
	pub expand_threshold: f32,

	/// How much a bin below the threshold is pushed down; 1 is off.
	#[range(min = 1.0, max = 20.0)]
	#[default(2.0)]
	#[id(display = "Expand Ratio")]
	#[layout(group = "Curve", order = 1, widget = knob)]
	pub expand_ratio: f32,

	/// The deepest attenuation the expander may apply.
	#[range(min = 0.0, max = 60.0)]
	#[unit(decibels)]
	#[default(24.0)]
	#[id(display = "Range")]
	#[layout(group = "Dynamics", order = 4, widget = knob)]
	pub range: f32,

	/// Gain after the two stages.
	#[range(min = -24.0, max = 24.0)]
	#[unit(decibels)]
	#[default(0.0)]
	#[id(display = "Makeup")]
	#[layout(group = "Dynamics", order = 5, widget = knob)]
	pub makeup: f32,

	/// How quickly a bin's gain falls.
	#[range(min = 0.1, max = 200.0)]
	#[unit(milliseconds)]
	#[default(10.0)]
	#[id(display = "Attack")]
	#[layout(group = "Dynamics", order = 6, widget = knob)]
	pub attack: f32,

	/// How quickly a bin's gain comes back.
	#[range(min = 1.0, max = 2_000.0)]
	#[unit(milliseconds)]
	#[default(120.0)]
	#[id(display = "Release")]
	#[layout(group = "Dynamics", order = 7, widget = knob)]
	pub release: f32,

	/// Which pair of pages the curves are read between.
	#[range(min = 0.0, max = 1.0)]
	#[default(0.0)]
	#[id(display = "Morph")]
	#[layout(group = "Curve", order = 2, widget = knob, hero = "spectraldynamics-curve")]
	pub morph: f32,

	/// How far the threshold page moves a bin's threshold.
	#[range(min = 0.0, max = 1.0)]
	#[default(1.0)]
	#[id(display = "Curve Depth")]
	#[layout(group = "Curve", order = 3, widget = knob)]
	pub curve_depth: f32,

	/// How much of the processed spectrum is heard.
	#[range(min = 0.0, max = 1.0)]
	#[default(1.0)]
	#[id(display = "Mix")]
	#[layout(group = "Dynamics", order = 8, widget = knob)]
	pub mix: f32,

	#[skip]
	/// The sample rate the analysis runs at.
	pub sample_rate: usize,

	#[skip]
	curves: Arc<DynamicsCurves>,
	#[skip]
	window_size: usize,
	#[skip]
	reference: f32,
	#[skip]
	level: Vec<f32>,
	#[skip]
	compress_gain: Vec<f32>,
	#[skip]
	expand_gain: Vec<f32>,
	#[skip]
	threshold_offset: Vec<f32>,
	#[skip]
	curve_ratio: Vec<f32>,
	#[skip]
	scratch_first: Vec<f32>,
	#[skip]
	scratch_second: Vec<f32>,
	#[skip]
	seen: (u64, u64, f32, f32),
	#[skip]
	seen_attack: f32,
	#[skip]
	seen_release: f32,
	#[skip]
	attack_coefficient: f32,
	#[skip]
	release_coefficient: f32,
}

impl SpectralDynamics {
	/// A dynamics stage over `curves`, ready for a vocoder of the given rate.
	pub fn new(curves: Arc<DynamicsCurves>, sample_rate: usize) -> Self {
		Self {
			compress_threshold: -24.0,
			compress_ratio: 4.0,
			expand_threshold: -48.0,
			expand_ratio: 2.0,
			range: 24.0,
			makeup: 0.0,
			attack: 10.0,
			release: 120.0,
			morph: 0.0,
			curve_depth: 1.0,
			mix: 1.0,
			sample_rate,
			curves,
			window_size: 0,
			reference: 1.0,
			level: Vec::new(),
			compress_gain: Vec::new(),
			expand_gain: Vec::new(),
			threshold_offset: Vec::new(),
			curve_ratio: Vec::new(),
			scratch_first: Vec::new(),
			scratch_second: Vec::new(),
			seen: (u64::MAX, u64::MAX, f32::NAN, f32::NAN),
			seen_attack: f32::NAN,
			seen_release: f32::NAN,
			attack_coefficient: 0.0,
			release_coefficient: 0.0,
		}
	}

	/// The curves this stage reads.
	pub fn curves(&self) -> Arc<DynamicsCurves> {
		self.curves.clone()
	}

	/// The per-bin tables, sized for the analysis in use.
	fn resize(&mut self, window_size: usize, window_factor: f32) {
		let bins = window_size / 2 + 1;

		self.window_size = window_size;
		// A sine at a bin's centre leaves half the window's sum in that bin, so
		// dividing by that is what puts a full-scale sine at 0 dB rather than at
		// whatever the FFT length happens to be.
		self.reference = (0.25 * window_factor * window_size as f32).max(1.0);
		self.level = vec![0.0; bins];
		self.compress_gain = vec![0.0; bins];
		self.expand_gain = vec![0.0; bins];
		self.threshold_offset = vec![0.0; bins];
		self.curve_ratio = vec![1.0; bins];
		self.scratch_first = vec![0.0; bins];
		self.scratch_second = vec![0.0; bins];
		self.seen = (u64::MAX, u64::MAX, f32::NAN, f32::NAN);
	}

	/// Rebuilds the dense tables the audio thread reads.
	fn rebuild(&mut self) {
		let threshold = self.curves.threshold().snapshot();
		let ratio = self.curves.ratio().snapshot();
		let (threshold_from, threshold_to, threshold_blend) =
			page_position(self.morph, threshold.pages().len());
		let (ratio_from, ratio_to, ratio_blend) = page_position(self.morph, ratio.pages().len());
		let threshold_pages = threshold.pages();
		let ratio_pages = ratio.pages();
		let bins = self.threshold_offset.len();

		for bin in 0..bins {
			let frequency = bin as f32 * self.sample_rate as f32 / self.window_size as f32;
			let position = position_of(frequency, self.sample_rate as f32);
			self.scratch_first[bin] = sample_curve(&threshold_pages[threshold_from], position);
			self.scratch_second[bin] = sample_curve(&threshold_pages[threshold_to], position);
		}

		for bin in 0..bins {
			let value = blend(
				self.scratch_first[bin],
				self.scratch_second[bin],
				threshold_blend,
			);
			self.threshold_offset[bin] = (value - 0.5) * 2.0 * CURVE_RANGE_DB * self.curve_depth;
		}

		for bin in 0..bins {
			let frequency = bin as f32 * self.sample_rate as f32 / self.window_size as f32;
			let position = position_of(frequency, self.sample_rate as f32);
			self.scratch_first[bin] = sample_curve(&ratio_pages[ratio_from], position);
			self.scratch_second[bin] = sample_curve(&ratio_pages[ratio_to], position);
		}

		for bin in 0..bins {
			self.curve_ratio[bin] = blend(
				self.scratch_first[bin],
				self.scratch_second[bin],
				ratio_blend,
			);
		}

		self.seen = (
			threshold.generation(),
			ratio.generation(),
			self.morph,
			self.curve_depth,
		);
	}

	/// The gain this bin's level asks for, in dB.
	fn gain_db(&self, bin: usize, level_db: f32) -> (f32, f32) {
		let offset = self.threshold_offset[bin];
		let curve = self.curve_ratio[bin];
		let depth = self.curve_depth;
		let curve = CompanderCurve {
			compress_threshold_db: self.compress_threshold + offset,
			compress_ratio: 1.0 + curve * (self.compress_ratio - 1.0) * depth,
			expand_threshold_db: self.expand_threshold + offset,
			expand_ratio: 1.0 + curve * (self.expand_ratio - 1.0) * depth,
			range_db: self.range,
		};

		curve.gains_db(level_db)
	}
}

impl FrequencyMapper for SpectralDynamics {
	fn prepare(&mut self, sample_rate: usize, window_size: usize, window_factor: f32) {
		self.sample_rate = sample_rate;
		self.resize(window_size, window_factor);
	}

	fn map_frequency(&mut self, frequency: f32, amplitude: f32) -> (f32, f32) {
		let bins = self.threshold_offset.len();

		if bins == 0 || self.window_size == 0 {
			return (frequency, amplitude);
		}

		if self.attack != self.seen_attack || self.release != self.seen_release {
			let rate = self.sample_rate.max(1) as f32;
			self.attack_coefficient = (-1.0 / (self.attack.max(0.01) * 0.001 * rate)).exp();
			self.release_coefficient = (-1.0 / (self.release.max(0.01) * 0.001 * rate)).exp();
			self.seen_attack = self.attack;
			self.seen_release = self.release;
		}

		let threshold_generation = self.curves.threshold().generation();
		let ratio_generation = self.curves.ratio().generation();

		if self.seen
			!= (
				threshold_generation,
				ratio_generation,
				self.morph,
				self.curve_depth,
			) {
			self.rebuild();
		}

		let bin = ((frequency * self.window_size as f32 / self.sample_rate.max(1) as f32).round()
			as usize)
			.min(bins - 1);

		let level_db = linear_to_db(amplitude / self.reference);
		let (compress_target, expand_target) = self.gain_db(bin, level_db);

		let follow = |current: f32, target: f32, attack: f32, release: f32| {
			let coefficient = if target < current { attack } else { release };
			coefficient * current + (1.0 - coefficient) * target
		};

		self.compress_gain[bin] = follow(
			self.compress_gain[bin],
			compress_target,
			self.attack_coefficient,
			self.release_coefficient,
		);
		self.expand_gain[bin] = follow(
			self.expand_gain[bin],
			expand_target,
			self.attack_coefficient,
			self.release_coefficient,
		);

		let gain_db = (self.compress_gain[bin] + self.expand_gain[bin] + self.makeup)
			.clamp(-self.range, self.makeup.max(0.0) + 24.0);
		let gain = db_to_linear(gain_db * self.mix);

		(frequency, amplitude * gain)
	}

	fn synthesis_phase(
		&mut self,
		_frequency: f32,
		analysis_phase: f32,
		_propagated_phase: f32,
	) -> f32 {
		// Only the magnitudes change, so the analysis phase is what makes the
		// overlapping frames add back up.
		analysis_phase
	}
}
#[cfg(test)]
mod tests {
	use super::*;

	const RATE: usize = 48_000;
	const WINDOW: usize = 2_048;

	/// A mapper with fast ballistics, so a test reaches its settled gain quickly.
	fn mapper() -> SpectralDynamics {
		let mut mapper = SpectralDynamics::new(DynamicsCurves::new(), RATE);
		FrequencyMapper::prepare(&mut mapper, RATE, WINDOW, 0.54);

		mapper.attack = 1.0;
		mapper.release = 2.0;
		mapper
	}

	/// The FFT magnitude a sine at `level_db` leaves in its bin.
	fn magnitude(level_db: f32) -> f32 {
		db_to_linear(level_db) * 0.25 * 0.54 * WINDOW as f32
	}

	/// Runs one bin at a steady level until the gain stops moving, and returns it.
	fn settled(mapper: &mut SpectralDynamics, frequency: f32, level_db: f32) -> f32 {
		let input = magnitude(level_db);
		let mut output = input;

		for _ in 0..20_000 {
			output = mapper.map_frequency(frequency, input).1;
		}

		output / input
	}

	/// Ratio 1 is a bypass: a stage that still shaves a level is a bug.
	#[test]
	fn one_to_one_leaves_every_level_alone() {
		let mut mapper = mapper();
		mapper.compress_ratio = 1.0;
		mapper.expand_ratio = 1.0;

		for level in [-60.0, -30.0, -6.0, 0.0] {
			let gain = settled(&mut mapper, 1_000.0, level);

			assert!((gain - 1.0).abs() < 1e-3, "level {level} gain {gain}");
		}
	}

	/// Four to one takes 20 dB above the threshold down to 5, so a bin at -6 dB
	/// with a -24 dB threshold settles at -13.5 dB.
	#[test]
	fn a_loud_bin_is_compressed_by_the_curve() {
		let mut mapper = mapper();
		mapper.compress_ratio = 4.0;
		let gain = settled(&mut mapper, 1_000.0, -6.0);
		let expected = db_to_linear(-13.5);

		assert!(
			(gain - expected).abs() < expected * 0.02,
			"gain {gain}, expected {expected}"
		);
	}

	/// Twelve dB under the expand threshold at two to one is a 12 dB cut.
	#[test]
	fn a_quiet_bin_is_pushed_down_by_the_expander() {
		let mut mapper = mapper();
		let gain = settled(&mut mapper, 1_000.0, -60.0);
		let expected = db_to_linear(-12.0);

		assert!(
			(gain - expected).abs() < expected * 0.02,
			"gain {gain}, expected {expected}"
		);
	}

	/// The range, not the slope, decides the deepest attenuation.
	#[test]
	fn the_range_caps_the_expander() {
		let mut mapper = mapper();
		mapper.expand_ratio = 20.0;
		let gain = settled(&mut mapper, 1_000.0, -80.0);
		let expected = db_to_linear(-24.0);

		assert!(
			(gain - expected).abs() < expected * 0.02,
			"gain {gain}, expected {expected}"
		);
	}

	/// A threshold page can move one end of the spectrum and not the other: a
	/// -30 dB bin is untouched by the flat page and compressed under the ramp.
	#[test]
	fn the_threshold_page_tilts_what_counts_as_loud() {
		let mut flat = mapper();
		flat.compress_ratio = 4.0;
		let untouched = settled(&mut flat, 100.0, -30.0);

		let mut ramp = mapper();
		ramp.compress_ratio = 4.0;
		ramp.curves.set_threshold(vec![vec![
			CurvePoint::new(0.0, 0.0),
			CurvePoint::new(1.0, 1.0),
		]]);
		let compressed = settled(&mut ramp, 100.0, -30.0);

		assert!((untouched - 1.0).abs() < 1e-3, "flat {untouched}");
		assert!(compressed < 0.6, "ramp {compressed}");
	}

	/// The morph blends two pages, so the same level lands between them.
	#[test]
	fn the_morph_walks_between_the_pages() {
		let make = |morph: f32| {
			let mut mapper = mapper();
			mapper.compress_ratio = 4.0;
			mapper.compress_threshold = -12.0;
			mapper.curves.set_threshold(vec![
				vec![CurvePoint::new(0.0, 0.5), CurvePoint::new(1.0, 0.5)],
				vec![CurvePoint::new(0.0, 0.0), CurvePoint::new(1.0, 0.0)],
			]);
			mapper.morph = morph;
			mapper
		};

		let first = settled(&mut make(0.0), 1_000.0, -6.0);
		let last = settled(&mut make(1.0), 1_000.0, -6.0);
		let middle = settled(&mut make(0.5), 1_000.0, -6.0);

		assert!(first > last, "first {first}, last {last}");
		assert!(
			middle < first && middle > last,
			"middle {middle} between {last} and {first}"
		);
	}

	/// A bend bulges the segment the editor draws, so the same page samples
	/// differently in the middle of it.
	#[test]
	fn a_bend_moves_the_middle_of_a_segment() {
		let straight = [CurvePoint::new(0.0, 0.0), CurvePoint::new(1.0, 1.0)];
		let bent = [
			CurvePoint {
				time: 0.0,
				value: 0.0,
				bend: 2.0,
			},
			CurvePoint::new(1.0, 1.0),
		];

		assert!((sample_curve(&straight, 0.5) - 0.5).abs() < 1e-6);
		assert!(
			sample_curve(&bent, 0.5) < 0.4,
			"bent {:?}",
			sample_curve(&bent, 0.5)
		);
	}

	/// A page edit reaches the audio side without the mapper being rebuilt.
	#[test]
	fn an_edit_is_read_on_the_next_bin() {
		let mut mapper = mapper();
		mapper.compress_ratio = 4.0;
		let before = settled(&mut mapper, 1_000.0, -6.0);

		// A page that drops every bin's threshold by 36 dB compresses the same
		// level harder, so the edit has to reach the audio side.
		mapper.curves.set_threshold(vec![vec![
			CurvePoint::new(0.0, 0.0),
			CurvePoint::new(1.0, 0.0),
		]]);
		let after = settled(&mut mapper, 1_000.0, -6.0);

		assert!(after < before, "before {before}, after {after}");
	}

	/// A smaller window resizes the per-bin tables and keeps working.
	#[test]
	fn a_window_change_resizes_the_tables() {
		let mut mapper = mapper();
		FrequencyMapper::prepare(&mut mapper, RATE, 1_024, 0.54);

		assert_eq!(mapper.threshold_offset.len(), 1_024 / 2 + 1);

		let gain = settled(&mut mapper, 1_000.0, -6.0);

		assert!(gain.is_finite() && gain > 0.0, "gain {gain}");
	}

	/// Only the magnitude changes, so the analysis phase is what adds back up.
	#[test]
	fn the_synthesis_phase_is_the_analysis_phase() {
		let mut mapper = mapper();

		assert_eq!(mapper.synthesis_phase(440.0, 0.5, 2.0), 0.5);
	}
	/// What the analyser actually hands a mapper, per bin.
	#[test]
	fn measure_the_bin_levels() {
		use crate::Effect;
		use crate::prelude::{Parameter, SetValue};

		struct Spy {
			seen: Vec<(f32, f32)>,
		}

		impl crate::prelude::Parameters for Spy {
			fn get_parameters(&self) -> Vec<Parameter> {
				Vec::new()
			}

			fn set_parameter(&mut self, _: &str, _: SetValue) -> bool {
				false
			}
		}

		impl FrequencyMapper for Spy {
			fn map_frequency(&mut self, frequency: f32, amplitude: f32) -> (f32, f32) {
				self.seen.push((frequency, amplitude));
				(frequency, amplitude)
			}
		}

		let mut effect = crate::prelude::PhaseVocoder::new(Spy { seen: Vec::new() }, 2_048, 48_000);
		let mut context: Box<dyn crate::ProcessContext> = Box::new(());
		let mut out = Vec::new();

		for index in 0..8_192 {
			let phase = std::f32::consts::TAU * 1_000.0 * index as f32 / 48_000.0;
			let mut frame = [phase.sin()];
			effect.process(&mut frame, &[], &mut context);
			out.push(frame[0]);
		}

		let seen = &effect.mapper.seen;
		let frame = 1024;
		let last = &seen[seen.len() - frame..];
		let peak = last.iter().fold(
			(0.0f32, 0.0f32),
			|best, pair| if pair.1 > best.1 { *pair } else { best },
		);
		let mut magnitudes = last.iter().map(|pair| pair.1).collect::<Vec<_>>();
		magnitudes.sort_by(|a, b| a.partial_cmp(b).unwrap());
		let median = magnitudes[magnitudes.len() / 2];
		let rms = (magnitudes.iter().map(|m| m * m).sum::<f32>() / magnitudes.len() as f32).sqrt();
		let db = |value: f32| 20.0 * value.max(1e-9).log10();

		let reference = 0.25 * 0.54 * 2_048.0;
		let peak_db = db(peak.1 / reference);

		eprintln!(
			"frames {} | peak {:.1} at {:.0} Hz -> {:.1} dBFS | median {} ({:.1} dBFS) | rms {:.1} ({:.1} dBFS)",
			seen.len() / frame,
			peak.1,
			peak.0,
			peak_db,
			median,
			db(median / reference),
			rms,
			db(rms / reference),
		);

		assert!(
			peak_db.abs() < 2.0,
			"a full-scale sine has to read as 0 dBFS, not {peak_db}"
		);
	}
}
