//! The OTT effect: a three-band upward/downward compressor in the spirit of
//! Xfer's OTT, ported from the LV2 reimplementation by Rui-727.
//!
//! The constants and the shape of the algorithm follow Rui-727/OTT, an MIT
//! licensed LV2 reimplementation of the Xfer design.

use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::Arc;

use i_am_dsp::prelude::{Effect, ProcessContext};
use i_am_dsp_derive::Parameters;


/// The low/mid crossover.
const XO_LOW: f32 = 88.3;
/// The mid/high crossover.
const XO_HIGH: f32 = 2500.0;
/// Attack at time = 1.
const ATTACK_MS: f32 = 5.0;
/// Release at time = 1.
const RELEASE_MS: f32 = 50.0;
/// RMS detector window.
const RMS_MS: f32 = 2.5;
/// Soft knee width, in dB.
const KNEE_DB: f32 = 6.0;
/// Upward ratio.
const UP_RATIO: f32 = 4.0;
/// Downward ratio for the low and mid bands; the high band is a limiter.
const DOWN_RATIO_LM: f32 = 66.7;
/// Floor for the dB conversion.
const DB_FLOOR: f32 = 1e-10;
/// Denormals below this are flushed to zero.
const DENORMAL: f32 = 1e-20;
/// The most upward boost, in dB.
const GR_MIN: f32 = -24.0;
/// The most downward reduction, in dB.
const GR_MAX: f32 = 60.0;

/// The controls, in the form the core wants them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OttSettings {
	/// How much of the effect is mixed in, 0..1.
	pub depth: f32,
	/// The average time constant, as a fraction of the band attack/release.
	pub time: f32,
	/// The upward compression amount.
	pub upward: f32,
	/// The downward compression amount.
	pub downward: f32,
	/// Input gain in dB.
	pub input_gain: f32,
	/// Output gain in dB.
	pub output_gain: f32,
	/// Per-band downward thresholds in dB.
	pub band_threshold: [f32; 3],
	/// Per-band makeup gains in dB.
	pub band_gain: [f32; 3],
}

impl Default for OttSettings {
	fn default() -> Self {
		Self {
			depth: 1.0,
			time: 0.5,
			upward: 1.0,
			downward: 1.0,
			input_gain: 5.2,
			output_gain: 0.0,
			band_threshold: [-30.0; 3],
			band_gain: [10.3, 5.7, 10.3],
		}
	}
}

fn db_to_lin(db: f32) -> f32 {
	10.0_f32.powf(db / 20.0)
}

fn flush(value: f32) -> f32 {
	if value.abs() < DENORMAL { 0.0 } else { value }
}

fn clamp_gain(gain: f32) -> f32 {
	if !gain.is_finite() {
		return 0.0;
	}

	gain.clamp(1e-6, 100.0)
}

fn clamp_gr(gr: f32) -> f32 {
	if !gr.is_finite() {
		return 0.0;
	}

	gr.clamp(GR_MIN, GR_MAX)
}

#[derive(Clone, Copy, Default)]
struct Biquad {
	b0: f32,
	b1: f32,
	b2: f32,
	a1: f32,
	a2: f32,
}

impl Biquad {
	fn lowpass(frequency: f32, sample_rate: f32) -> Self {
		let omega = 2.0 * std::f32::consts::PI * frequency / sample_rate;
		let cos = omega.cos();
		let alpha = omega.sin() * std::f32::consts::FRAC_1_SQRT_2;
		let a0 = 1.0 + alpha;

		Self {
			b0: (1.0 - cos) * 0.5 / a0,
			b1: (1.0 - cos) / a0,
			b2: (1.0 - cos) * 0.5 / a0,
			a1: -2.0 * cos / a0,
			a2: (1.0 - alpha) / a0,
		}
	}

	fn highpass(frequency: f32, sample_rate: f32) -> Self {
		let omega = 2.0 * std::f32::consts::PI * frequency / sample_rate;
		let cos = omega.cos();
		let alpha = omega.sin() * std::f32::consts::FRAC_1_SQRT_2;
		let a0 = 1.0 + alpha;

		Self {
			b0: (1.0 + cos) * 0.5 / a0,
			b1: -(1.0 + cos) / a0,
			b2: (1.0 + cos) * 0.5 / a0,
			a1: -2.0 * cos / a0,
			a2: (1.0 - alpha) / a0,
		}
	}
}

#[derive(Clone, Copy, Default)]
struct BiquadState {
	x1: f32,
	x2: f32,
	y1: f32,
	y2: f32,
}

impl BiquadState {
	fn process(&mut self, coef: Biquad, x: f32) -> f32 {
		let y = coef.b0 * x + coef.b1 * self.x1 + coef.b2 * self.x2
			- coef.a1 * self.y1 - coef.a2 * self.y2;

		self.x2 = self.x1;
		self.x1 = x;
		self.y2 = self.y1;
		self.y1 = y;

		y
	}
}

struct Lr4 {
	coef: Biquad,
	first: BiquadState,
	second: BiquadState,
}

impl Lr4 {
	fn new(coef: Biquad) -> Self {
		Self {
			coef,
			first: BiquadState::default(),
			second: BiquadState::default(),
		}
	}

	fn process(&mut self, x: f32) -> f32 {
		let y = self.first.process(self.coef, x);
		self.second.process(self.coef, y)
	}

	fn reset(&mut self) {
		self.first = BiquadState::default();
		self.second = BiquadState::default();
	}
}

#[derive(Clone, Copy, Default)]
struct BandState {
	rms_down_sq: f32,
	rms_down_right_sq: f32,
	env_down_ac: f32,
	env_down_ra: f32,
	rms_up_sq: f32,
	rms_up_right_sq: f32,
	env_up_ac: f32,
	env_up_ra: f32,
}

fn gain_down(level_db: f32, threshold: f32, slope: f32, width: f32) -> f32 {
	let overshoot = level_db - threshold;
	let knee = width * 0.5;

	if overshoot <= -knee {
		return 0.0;
	}

	if overshoot < knee {
		let t = overshoot + knee;
		return 0.5 * slope * t * t / width;
	}

	slope * overshoot
}

fn gain_up(level_db: f32, threshold: f32, slope: f32, width: f32) -> f32 {
	let undershoot = threshold - level_db;

	if undershoot <= 0.0 || undershoot > 30.0 {
		return 0.0;
	}

	let knee = width * 0.5;

	if undershoot < knee {
		let t = undershoot + knee;
		return -0.5 * slope * t * t / width;
	}

	-slope * undershoot
}

fn ballistics(target: f32, state1: &mut f32, state2: &mut f32, attack: f32, release: f32) -> f32 {
	let released = release * *state2 + (1.0 - release) * target;
	*state2 = if target > released { target } else { released };
	*state1 = attack * *state1 + (1.0 - attack) * *state2;
	*state1
}

/// The three-band compressor core.
pub struct OttCore {
	sample_rate: f32,
	settings: OttSettings,
	lp_low: Lr4,
	hp_low: Lr4,
	lp_mid: Lr4,
	hp_high: Lr4,
	lp_low_right: Lr4,
	hp_low_right: Lr4,
	lp_mid_right: Lr4,
	hp_high_right: Lr4,
	bands: [BandState; 3],
	alpha_attack: f32,
	alpha_release: f32,
	alpha_rms: f32,
	input_gain_lin: f32,
	output_gain_lin: f32,
	band_gain_lin: [f32; 3],
	slope_down: [f32; 3],
	slope_up: [f32; 3],
	meters: [f32; 3],
}

impl OttCore {
	/// A core for a sample rate, with the default settings.
	pub fn new(sample_rate: f32) -> Self {
		let settings = OttSettings::default();
		let mut core = Self {
			sample_rate,
			settings,
			lp_low: Lr4::new(Biquad::lowpass(XO_LOW, sample_rate)),
			hp_low: Lr4::new(Biquad::highpass(XO_LOW, sample_rate)),
			lp_mid: Lr4::new(Biquad::lowpass(XO_HIGH, sample_rate)),
			hp_high: Lr4::new(Biquad::highpass(XO_HIGH, sample_rate)),
			lp_low_right: Lr4::new(Biquad::lowpass(XO_LOW, sample_rate)),
			hp_low_right: Lr4::new(Biquad::highpass(XO_LOW, sample_rate)),
			lp_mid_right: Lr4::new(Biquad::lowpass(XO_HIGH, sample_rate)),
			hp_high_right: Lr4::new(Biquad::highpass(XO_HIGH, sample_rate)),
			bands: [BandState::default(); 3],
			alpha_attack: 0.0,
			alpha_release: 0.0,
			alpha_rms: 0.0,
			input_gain_lin: 1.0,
			output_gain_lin: 1.0,
			band_gain_lin: [1.0; 3],
			slope_down: [0.0; 3],
			slope_up: [0.0; 3],
			meters: [0.0; 3],
		};

		core.set_settings(settings);
		core.reset();
		core
	}

	/// Applies new settings.
	pub fn set_settings(&mut self, settings: OttSettings) {
		self.settings = settings;

		let time = settings.time.clamp(0.0, 1.0);
		let time_scale = 0.1 + 1.9 * time;
		let tau_attack = ATTACK_MS * 1e-3 * time_scale;
		let tau_release = RELEASE_MS * 1e-3 * time_scale;
		let tau_rms = RMS_MS * 1e-3;
		let fs = self.sample_rate;

		self.alpha_attack = (-1.0 / (tau_attack * fs)).exp();
		self.alpha_release = (-1.0 / (tau_release * fs)).exp();
		self.alpha_rms = (-1.0 / (tau_rms * fs)).exp();

		self.input_gain_lin = db_to_lin(settings.input_gain);
		self.output_gain_lin = db_to_lin(settings.output_gain);
		for band in 0..3 {
			self.band_gain_lin[band] = db_to_lin(settings.band_gain[band]);
		}

		let down = settings.downward.clamp(0.0, 1.0);
		let up = settings.upward.clamp(0.0, 1.0);
		let slope_down_lm = 1.0 - 1.0 / DOWN_RATIO_LM;
		let slope_up = 1.0 - 1.0 / UP_RATIO;

		self.slope_down = [slope_down_lm * down, slope_down_lm * down, down];
		self.slope_up = [slope_up * up; 3];
	}

	/// Clears the filters and puts the detectors at the threshold.
	pub fn reset(&mut self) {
		self.lp_low.reset();
		self.hp_low.reset();
		self.lp_mid.reset();
		self.hp_high.reset();
		self.lp_low_right.reset();
		self.hp_low_right.reset();
		self.lp_mid_right.reset();
		self.hp_high_right.reset();

		let threshold = db_to_lin(self.settings.band_threshold[0]);
		let threshold_sq = threshold * threshold;

		for band in &mut self.bands {
			*band = BandState::default();
			band.rms_down_sq = threshold_sq;
			band.rms_down_right_sq = threshold_sq;
			band.rms_up_sq = threshold_sq;
			band.rms_up_right_sq = threshold_sq;
		}

		self.meters = [0.0; 3];
	}

	/// Splits one sample into low, mid and high, input gain included.
	#[cfg(test)]
	pub fn crossover_split(&mut self, input: f32) -> [f32; 3] {
		let x = flush(input) * self.input_gain_lin;
		let low = self.lp_low.process(x);
		let mid_high = self.hp_low.process(x);
		let mid = self.lp_mid.process(mid_high);
		let high = self.hp_high.process(mid_high);

		[low, mid, high]
	}

	/// Processes one stereo frame.
	pub fn process(&mut self, left: f32, right: f32) -> (f32, f32) {
		let settings = self.settings;
		let depth = settings.depth.clamp(0.0, 1.0);

		if depth <= 0.0 {
			return (left, right);
		}

		let input_gain = self.input_gain_lin;
		let output_gain = self.output_gain_lin;
		let alpha_attack = self.alpha_attack;
		let alpha_release = self.alpha_release;
		let alpha_rms = self.alpha_rms;
		let slopes_down = self.slope_down;
		let slopes_up = self.slope_up;
		let gains = self.band_gain_lin;

		let x_left = flush(left) * input_gain;
		let x_right = flush(right) * input_gain;

		let low_left = self.lp_low.process(x_left);
		let mid_high_left = self.hp_low.process(x_left);
		let mid_left = self.lp_mid.process(mid_high_left);
		let high_left = self.hp_high.process(mid_high_left);

		let low_right = self.lp_low_right.process(x_right);
		let mid_high_right = self.hp_low_right.process(x_right);
		let mid_right = self.lp_mid_right.process(mid_high_right);
		let high_right = self.hp_high_right.process(mid_high_right);

		let bands_left = [low_left, mid_left, high_left];
		let bands_right = [low_right, mid_right, high_right];

		let mut wet_left = 0.0;
		let mut wet_right = 0.0;

		for band in 0..3 {
			let mut band_left = bands_left[band];
			let mut band_right = bands_right[band];
			let state = &mut self.bands[band];
			let threshold = settings.band_threshold[band];

			// Downward, with the louder channel driving the linked detector.
			state.rms_down_sq = alpha_rms * state.rms_down_sq
				+ (1.0 - alpha_rms) * (band_left * band_left);
			state.rms_down_right_sq = alpha_rms * state.rms_down_right_sq
				+ (1.0 - alpha_rms) * (band_right * band_right);

			let level_left = state.rms_down_sq.sqrt();
			let level_right = state.rms_down_right_sq.sqrt();
			let level = level_left.max(level_right);
			let level_db = 20.0 * (level + DB_FLOOR).log10();
			let gr_raw = gain_down(level_db, threshold, slopes_down[band], KNEE_DB);
			let gr_down = ballistics(
				gr_raw,
				&mut state.env_down_ac,
				&mut state.env_down_ra,
				alpha_attack,
				alpha_release,
			);
			let gr_down = clamp_gr(gr_down);
			let gain = clamp_gain((-gr_down * std::f32::consts::LN_10 / 20.0).exp());
			band_left *= gain;
			band_right *= gain;

			// Upward, on the post-downward signal.
			state.rms_up_sq = alpha_rms * state.rms_up_sq
				+ (1.0 - alpha_rms) * (band_left * band_left);
			state.rms_up_right_sq = alpha_rms * state.rms_up_right_sq
				+ (1.0 - alpha_rms) * (band_right * band_right);

			let level_left = state.rms_up_sq.sqrt();
			let level_right = state.rms_up_right_sq.sqrt();
			let level = level_left.max(level_right);
			let level_db = 20.0 * (level + DB_FLOOR).log10();
			let gr_raw = gain_up(level_db, threshold, slopes_up[band], KNEE_DB);
			let gr_up = ballistics(
				gr_raw,
				&mut state.env_up_ac,
				&mut state.env_up_ra,
				alpha_attack,
				alpha_release,
			);
			let gr_up = clamp_gr(gr_up);
			let gain = clamp_gain((-gr_up * std::f32::consts::LN_10 / 20.0).exp());
			band_left *= gain;
			band_right *= gain;

			self.meters[band] = gr_down + gr_up;
			wet_left += band_left * gains[band];
			wet_right += band_right * gains[band];
		}

		wet_left *= output_gain;
		wet_right *= output_gain;

		let out_left = left * (1.0 - depth) + wet_left * depth;
		let out_right = right * (1.0 - depth) + wet_right * depth;

		(finish(out_left), finish(out_right))
	}

	/// The net gain in dB per band, for the meter.
	pub fn meters(&self) -> [f32; 3] {
		self.meters
	}
}

fn finish(value: f32) -> f32 {
	if value.is_finite() { flush(value) } else { 0.0 }
}

/// The three band meters the page reads.
pub struct OttMeters {
	bands: [AtomicI32; 3],
}

impl Default for OttMeters {
	fn default() -> Self {
		Self {
			bands: [AtomicI32::new(0), AtomicI32::new(0), AtomicI32::new(0)],
		}
	}
}

impl OttMeters {
	/// Stores the net gain in dB, in hundredths.
	pub fn set(&self, bands: [f32; 3]) {
		for (meter, value) in self.bands.iter().zip(bands) {
			meter.store((value.clamp(GR_MIN, GR_MAX) * 100.0) as i32, Ordering::Relaxed);
		}
	}

	/// The last stored values, in dB.
	pub fn get(&self) -> [f32; 3] {
		[
			self.bands[0].load(Ordering::Relaxed) as f32 / 100.0,
			self.bands[1].load(Ordering::Relaxed) as f32 / 100.0,
			self.bands[2].load(Ordering::Relaxed) as f32 / 100.0,
		]
	}
}

/// The product's DSP.
#[derive(Parameters)]
pub struct OttEffect {
	/// The drive into the compressor.
	#[range(min = -24.0, max = 24.0)]
	#[unit(db)]
	#[default(5.2)]
	#[id(display = "Input")]
	#[layout(group = "OTT", order = 0, widget = knob, size = large, hero = "ott")]
	pub input_gain: f32,

	/// The level after the compressor.
	#[range(min = -24.0, max = 24.0)]
	#[unit(db)]
	#[default(0.0)]
	#[id(display = "Output")]
	#[layout(group = "OTT", order = 1, widget = knob)]
	pub output_gain: f32,

	/// The wet/dry mix.
	#[range(min = 0.0, max = 1.0)]
	#[unit(percent)]
	#[default(1.0)]
	#[id(display = "Depth")]
	#[layout(group = "OTT", order = 2, widget = knob)]
	pub depth: f32,

	/// The attack/release scale.
	#[range(min = 0.0, max = 1.0)]
	#[unit(percent)]
	#[default(0.5)]
	#[id(display = "Time")]
	#[layout(group = "OTT", order = 3, widget = knob)]
	pub time: f32,

	/// How much upward compression.
	#[range(min = 0.0, max = 1.0)]
	#[unit(percent)]
	#[default(1.0)]
	#[id(display = "Upward")]
	#[layout(group = "OTT", order = 4, widget = knob)]
	pub upward: f32,

	/// How much downward compression.
	#[range(min = 0.0, max = 1.0)]
	#[unit(percent)]
	#[default(1.0)]
	#[id(display = "Downward")]
	#[layout(group = "OTT", order = 5, widget = knob)]
	pub downward: f32,

	/// The low band threshold.
	#[range(min = -60.0, max = 0.0)]
	#[unit(db)]
	#[default(-30.0)]
	#[id(display = "Low Thresh")]
	#[layout(group = "Bands", order = 0, widget = knob)]
	pub band1_threshold: f32,

	/// The low band output gain.
	#[range(min = -24.0, max = 24.0)]
	#[unit(db)]
	#[default(10.3)]
	#[id(display = "Low Gain")]
	#[layout(group = "Bands", order = 1, widget = knob)]
	pub band1_gain: f32,

	/// The mid band threshold.
	#[range(min = -60.0, max = 0.0)]
	#[unit(db)]
	#[default(-30.0)]
	#[id(display = "Mid Thresh")]
	#[layout(group = "Bands", order = 2, widget = knob)]
	pub band2_threshold: f32,

	/// The mid band output gain.
	#[range(min = -24.0, max = 24.0)]
	#[unit(db)]
	#[default(5.7)]
	#[id(display = "Mid Gain")]
	#[layout(group = "Bands", order = 3, widget = knob)]
	pub band2_gain: f32,

	/// The high band threshold.
	#[range(min = -60.0, max = 0.0)]
	#[unit(db)]
	#[default(-30.0)]
	#[id(display = "High Thresh")]
	#[layout(group = "Bands", order = 4, widget = knob)]
	pub band3_threshold: f32,

	/// The high band output gain.
	#[range(min = -24.0, max = 24.0)]
	#[unit(db)]
	#[default(10.3)]
	#[id(display = "High Gain")]
	#[layout(group = "Bands", order = 5, widget = knob)]
	pub band3_gain: f32,

	#[skip]
	core: OttCore,
	#[skip]
	meters: Arc<OttMeters>,
	#[skip]
	last: OttSettings,
}

impl OttEffect {
	/// A compressor for a sample rate.
	pub fn new(sample_rate: usize, meters: Arc<OttMeters>) -> Self {
		let core = OttCore::new(sample_rate as f32);
		let settings = OttSettings::default();

		Self {
			input_gain: settings.input_gain,
			output_gain: settings.output_gain,
			depth: settings.depth,
			time: settings.time,
			upward: settings.upward,
			downward: settings.downward,
			band1_threshold: settings.band_threshold[0],
			band1_gain: settings.band_gain[0],
			band2_threshold: settings.band_threshold[1],
			band2_gain: settings.band_gain[1],
			band3_threshold: settings.band_threshold[2],
			band3_gain: settings.band_gain[2],
			core,
			meters,
			last: settings,
		}
	}

	fn settings(&self) -> OttSettings {
		OttSettings {
			depth: self.depth,
			time: self.time,
			upward: self.upward,
			downward: self.downward,
			input_gain: self.input_gain,
			output_gain: self.output_gain,
			band_threshold: [
				self.band1_threshold,
				self.band2_threshold,
				self.band3_threshold,
			],
			band_gain: [self.band1_gain, self.band2_gain, self.band3_gain],
		}
	}
}

impl Effect<2> for OttEffect {
	fn delay(&self) -> usize {
		0
	}

	fn process(
		&mut self,
		samples: &mut [f32; 2],
		_other: &[&[f32; 2]],
		_context: &mut Box<dyn ProcessContext>,
	) {
		let settings = self.settings();

		if settings != self.last {
			self.core.set_settings(settings);
			self.last = settings;
		}

		let (left, right) = self.core.process(samples[0], samples[1]);
		self.meters.set(self.core.meters());
		samples[0] = left;
		samples[1] = right;
	}
}

#[cfg(test)]
mod moved_tests {
	use super::*;

	#[test]
	fn the_crossover_sums_flat() {
		let mut core = OttCore::new(44_100.0);
		core.set_settings(OttSettings {
			input_gain: 0.0,
			..Default::default()
		});

		let mut input_sq = 0.0_f64;
		let mut sum_sq = 0.0_f64;
		let mut state = 0xdead_beef_u32;

		for _ in 0..65_536 {
			state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
			let x = (state >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0;
			let bands = core.crossover_split(x);
			let sum = bands[0] + bands[1] + bands[2];
			input_sq += (x as f64) * (x as f64);
			sum_sq += (sum as f64) * (sum as f64);
		}

		let ratio = (sum_sq / input_sq).sqrt();
		assert!(ratio > 0.99 && ratio < 1.01, "{ratio}");
	}
}
