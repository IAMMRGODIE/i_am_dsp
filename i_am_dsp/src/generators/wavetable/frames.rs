//! The frames the shapes oscillator plays: the four classic shapes, or a table
//! the user brought.
//!
//! A frame is one cycle, and every frame is cut into band limited levels at build
//! time, so a note only ever plays the partials below its own Nyquist: a bright
//! table stops aliasing once the pitch rises, the way Serum and Vital handle
//! their tables. The imported table lives in a shared slot as well as in the
//! parameters, because the editor's upload route runs on the UI thread and a
//! serialized parameter cannot travel through the parameter map; a save reads
//! the slot, and a load writes it back.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use rustfft::FftPlanner;
use rustfft::num_complex::Complex;

use crate::prelude::{
	Oscillator, Parameter, ParameterMeta, Parameters, SetValue, Unit, Value, from_binary, to_binary,
};

/// The samples in one frame of a table.
pub const WAVE_FRAME_LEN: usize = 2048;
/// The most frames a table keeps; a longer file is sampled down to this.
pub const MAX_WAVE_FRAMES: usize = 256;
/// The shortest band limited level a frame is cut into.
const MIN_LEVEL_SIZE: usize = 4;

/// A table the user brought, saved with the plug-in state.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ImportedWave {
	/// The file name, for the interface.
	pub name: String,
	/// One cycle per frame, each one [`WAVE_FRAME_LEN`] long.
	pub frames: Vec<Vec<f32>>,
}

impl ImportedWave {
	/// A table from its frames.
	pub fn new(name: impl Into<String>, frames: Vec<Vec<f32>>) -> Self {
		Self {
			name: name.into(),
			frames: frames.into_iter().take(MAX_WAVE_FRAMES).collect(),
		}
	}

	/// How many frames the table holds.
	pub fn frames(&self) -> usize {
		self.frames.len()
	}

	/// The table with every frame played backwards.
	pub fn reversed(&self) -> Self {
		Self {
			name: self.name.clone(),
			frames: self
				.frames
				.iter()
				.map(|frame| frame.iter().rev().copied().collect())
				.collect(),
		}
	}

	/// The table with its polarity flipped.
	pub fn inverted(&self) -> Self {
		Self {
			name: self.name.clone(),
			frames: self
				.frames
				.iter()
				.map(|frame| frame.iter().map(|value| -value).collect())
				.collect(),
		}
	}
}

/// The four classic shapes, one cycle each.
fn classic_frames() -> Vec<Vec<f32>> {
	use std::f32::consts::TAU;

	let frame = |shape: fn(f32) -> f32| -> Vec<f32> {
		(0..WAVE_FRAME_LEN)
			.map(|index| shape(index as f32 / WAVE_FRAME_LEN as f32))
			.collect()
	};

	vec![
		frame(|phase| (phase * TAU).sin()),
		frame(|phase| 1.0 - 4.0 * (phase - 0.5).abs()),
		frame(|phase| 2.0 * phase - 1.0),
		frame(|phase| if phase < 0.5 { 1.0 } else { -1.0 }),
	]
}

/// One frame at every band limit: level k keeps the partials up to 1024 >> k
/// and is stored with 2048 >> k samples.
struct FrameLevels {
	levels: Vec<Vec<f32>>,
}

impl FrameLevels {
	fn build(frame: &[f32]) -> Self {
		// Level zero is the frame itself: it holds every partial it can.
		let mut levels = vec![frame.to_vec()];
		let mut planner = FftPlanner::new();
		let mut spectrum: Vec<Complex<f32>> = frame
			.iter()
			.map(|value| Complex::new(*value, 0.0))
			.collect();
		planner
			.plan_fft_forward(WAVE_FRAME_LEN)
			.process(&mut spectrum);

		// The unnormalized inverse transform sums the bins, so the coefficients
		// are divided by the length to get the wave back.
		let scale = 1.0 / WAVE_FRAME_LEN as f32;
		let mut size = WAVE_FRAME_LEN / 2;

		while size >= MIN_LEVEL_SIZE {
			let limit = size / 2;
			let mut level = vec![Complex::ZERO; size];

			for bin in 0..=limit {
				level[bin] = spectrum[bin] * scale;
			}

			for bin in 1..limit {
				level[size - bin] = level[bin].conj();
			}

			planner.plan_fft_inverse(size).process(&mut level);
			levels.push(level.iter().map(|bin| bin.re).collect());

			size /= 2;
		}

		Self { levels }
	}

	/// Reads a level at the given phase with a cubic interpolation.
	fn read(&self, level: usize, phase: f32) -> f32 {
		let table = &self.levels[level];
		let size = table.len();
		let position = (phase - phase.floor()) * size as f32;
		let index = position as usize;
		let mask = size - 1;
		let before = table[index.wrapping_sub(1) & mask];
		let now = table[index & mask];
		let next = table[(index + 1) & mask];
		let after = table[(index + 2) & mask];

		crate::tools::interpolate::cubic_interpolate(
			position - index as f32,
			[before, now, next, after],
		)
	}

	/// Reads the sum of the partials below the given ratio.
	fn play(&self, phase: f32, max_harmonic: f32) -> f32 {
		let (level, coarser, fade) = self.select(max_harmonic);
		let value = self.read(level, phase);

		if fade <= 0.0 {
			return value;
		}

		value + (self.read(coarser, phase) - value) * fade
	}

	/// Picks the pair of levels that keeps every partial below the given ratio,
	/// and how much of the coarser one to mix in. The mix hands over exactly as
	/// the finer level would start to alias, the same way the additive wavetable
	/// fades between its levels.
	fn select(&self, max_harmonic: f32) -> (usize, usize, f32) {
		let last = self.levels.len() - 1;
		let max_ratio = (WAVE_FRAME_LEN / 2) as f32;

		if max_harmonic.is_nan() || max_harmonic >= max_ratio {
			return (0, 0, 0.0);
		}

		if max_harmonic < 1.0 {
			return (last, last, 0.0);
		}

		let bits = (max_ratio / max_harmonic).to_bits();
		let exponent = ((bits >> 23) & 0xff) as i32 - 127;
		let mantissa = bits & 0x007f_ffff;
		let level = (if mantissa == 0 {
			exponent
		} else {
			exponent + 1
		})
		.clamp(0, last as i32) as usize;

		if level >= last {
			return (last, last, 0.0);
		}

		let log2 = f32::from_bits(mantissa | 0x3f80_0000) - 1.0 + exponent as f32;
		let fade = (1.0 + log2 - level as f32).clamp(0.0, 1.0);

		(level, level + 1, fade * fade * fade)
	}
}

/// The next bank's identity: a fresh number per bank, so a page that caches the
/// table it drew notices a reload even when the allocator reuses an address.
static NEXT_BANK: AtomicU64 = AtomicU64::new(1);

/// A table's frames with their band limited levels, built once.
pub struct WaveBank {
	id: u64,
	name: Option<String>,
	frames: Vec<Vec<f32>>,
	levels: Vec<FrameLevels>,
}

impl WaveBank {
	/// Builds a bank from frames, resampling any frame that is not a full cycle.
	pub fn new(name: Option<String>, frames: Vec<Vec<f32>>) -> Arc<Self> {
		let frames: Vec<Vec<f32>> = frames
			.into_iter()
			.take(MAX_WAVE_FRAMES)
			.map(|frame| resample(&frame, WAVE_FRAME_LEN))
			.collect();
		let levels = frames
			.iter()
			.map(|frame| FrameLevels::build(frame))
			.collect();

		Arc::new(Self {
			id: NEXT_BANK.fetch_add(1, Ordering::Relaxed),
			name,
			frames,
			levels,
		})
	}

	/// The bank's identity, for a page that caches what it drew.
	pub fn id(&self) -> u64 {
		self.id
	}

	/// The name of the file the table came from.
	pub fn name(&self) -> Option<&str> {
		self.name.as_deref()
	}

	/// The frames themselves.
	pub fn frames(&self) -> &[Vec<f32>] {
		&self.frames
	}

	/// How many frames the table holds.
	pub fn len(&self) -> usize {
		self.frames.len()
	}

	/// Whether the table holds no frames at all.
	pub fn is_empty(&self) -> bool {
		self.frames.is_empty()
	}

	/// The table as the parameter form: the frames and the name.
	pub fn imported(&self) -> ImportedWave {
		ImportedWave {
			name: self.name.clone().unwrap_or_default(),
			frames: self.frames.clone(),
		}
	}

	/// Reads the morph between two frames at the given phase, with every partial
	/// above the note's Nyquist dropped.
	pub fn sample(
		&self,
		phase: f32,
		from: usize,
		to: usize,
		blend: f32,
		max_harmonic: f32,
		fade: f32,
	) -> f32 {
		let first = self.sample_frame(from, phase, max_harmonic, fade);

		if blend <= 0.0 || from == to {
			return first;
		}

		let second = self.sample_frame(to, phase, max_harmonic, fade);

		first + (second - first) * blend
	}

	/// Reads one frame, band limited to the note.
	///
	/// `fade` is the edge crossfade, 0 to 1: the frame's last quarter is pulled
	/// towards its first samples, so the wrap from the end back to the start stops
	/// being a jump. Doing it here rather than rebuilding the table keeps a
	/// parameter the host re-sends every block from rebuilding anything.
	pub fn sample_frame(&self, frame: usize, phase: f32, max_harmonic: f32, fade: f32) -> f32 {
		if max_harmonic < 1.0 {
			return 0.0;
		}

		let levels = &self.levels[frame.min(self.levels.len() - 1)];
		let value = levels.play(phase, max_harmonic);
		let width = (fade * 0.25).clamp(0.0, 0.25);

		if width <= 0.0 {
			return value;
		}

		let p = phase - phase.floor();

		if p < 1.0 - width {
			return value;
		}

		let samples = width * WAVE_FRAME_LEN as f32;
		let position = (p - (1.0 - width)) * WAVE_FRAME_LEN as f32;
		let blend = ((position + 1.0) / (samples + 1.0)).clamp(0.0, 1.0);
		let head = levels.play(position / WAVE_FRAME_LEN as f32, max_harmonic);

		value * (1.0 - blend) + head * blend
	}
}

/// Reads a frame at a new length with linear interpolation.
fn resample(frame: &[f32], len: usize) -> Vec<f32> {
	if frame.len() == len {
		return frame.to_vec();
	}

	let mut output = vec![0.0; len];

	if frame.is_empty() {
		return output;
	}

	if frame.len() == 1 {
		output.fill(frame[0]);
		return output;
	}

	for (index, value) in output.iter_mut().enumerate() {
		let position = index as f32 / len as f32 * frame.len() as f32;
		let start = position.floor() as usize;
		let blend = position - start as f32;
		let before = frame[start.min(frame.len() - 1)];
		let after = frame[(start + 1).min(frame.len() - 1)];

		*value = before + (after - before) * blend;
	}

	output
}

fn morph_parameter(value: f32) -> Parameter {
	let meta = ParameterMeta {
		default: Some(0.0),
		unit: Unit::None,
		display_name: Some("Morph".into()),
		choices: None,
		#[cfg(feature = "layout")]
		layout: crate::prelude::ParamLayout::new()
			.group("Shape")
			.order(0)
			.widget(crate::prelude::ParamWidget::Knob)
			.hero("shape"),
	};

	Parameter::new(
		"smooth_factor",
		Value::Float {
			value,
			range: 0.0..=1.0,
			logarithmic: false,
		},
	)
	.with_meta(meta)
}

/// What the editor and the voice share: the imported table.
#[derive(Default)]
pub struct WaveSlot {
	/// The table, once something has been imported.
	pub bank: Option<Arc<WaveBank>>,
}

fn x_fade_parameter(value: f32) -> Parameter {
	let meta = ParameterMeta {
		default: Some(0.0),
		unit: Unit::None,
		display_name: Some("X-Fade".into()),
		choices: None,
		#[cfg(feature = "layout")]
		layout: crate::prelude::ParamLayout::new()
			.group("Shape")
			.order(1)
			.widget(crate::prelude::ParamWidget::Knob),
	};

	Parameter::new(
		"x_fade",
		Value::Float {
			value,
			range: 0.0..=1.0,
			logarithmic: false,
		},
	)
	.with_meta(meta)
}

/// The morphing wavetable behind I Am Shapes.
///
/// The band limit of a note is fixed by its pitch: the driver calls
/// [`Oscillator::set_pitch`] before a note, and every frame then plays from the
/// level that keeps its partials below the note's Nyquist.
pub struct WaveFramesOsc {
	classic: Arc<WaveBank>,
	slot: Arc<Mutex<WaveSlot>>,
	smooth_factor: f32,
	x_fade: f32,
	max_harmonic: f32,
}

impl Default for WaveFramesOsc {
	fn default() -> Self {
		Self::new()
	}
}

impl WaveFramesOsc {
	/// An oscillator with nothing imported: the four classic shapes.
	pub fn new() -> Self {
		Self::with_slot(Arc::new(Mutex::new(WaveSlot::default())))
	}

	/// An oscillator over a slot the editor and the voice share.
	pub fn with_slot(slot: Arc<Mutex<WaveSlot>>) -> Self {
		Self {
			classic: WaveBank::new(None, classic_frames()),
			slot,
			smooth_factor: 0.0,
			x_fade: 0.0,
			max_harmonic: f32::INFINITY,
		}
	}

	/// The slot the imported table lives in.
	pub fn slot(&self) -> Arc<Mutex<WaveSlot>> {
		self.slot.clone()
	}

	/// The table in play: the imported one, or the four classic shapes.
	pub fn bank(&self) -> Arc<WaveBank> {
		match self
			.slot
			.lock()
			.unwrap_or_else(|error| error.into_inner())
			.bank
			.as_ref()
		{
			Some(bank) => bank.clone(),
			None => self.classic.clone(),
		}
	}

	/// The imported table, if one is loaded.
	pub fn imported(&self) -> Option<ImportedWave> {
		self.slot
			.lock()
			.unwrap_or_else(|error| error.into_inner())
			.bank
			.as_ref()
			.map(|bank| bank.imported())
	}

	/// Replaces the imported table, or clears it.
	pub fn set_imported(&self, wave: Option<ImportedWave>) {
		let mut slot = self.slot.lock().unwrap_or_else(|error| error.into_inner());
		let bank = wave.map(|wave| WaveBank::new(Some(wave.name), wave.frames));

		slot.bank = bank;
	}

	/// How much of the frame the edge crossfade covers, 0 for none.
	pub fn x_fade(&self) -> f32 {
		self.x_fade
	}

	/// Sets the edge crossfade and rebuilds the table's levels for it.
	pub fn set_x_fade(&mut self, value: f32) {
		// Nothing to rebuild: the crossfade is applied as the frames are read,
		// which is what keeps a parameter the host re-sends every block from
		// rebuilding the table every block.
		self.x_fade = value.clamp(0.0, 1.0);
	}

	/// Where the morph sits, 0 for the first frame.
	pub fn smooth_factor(&self) -> f32 {
		self.smooth_factor
	}

	/// Moves the morph.
	pub fn set_smooth_factor(&mut self, value: f32) {
		self.smooth_factor = value.clamp(0.0, 1.0);
	}

	/// The frame pair the morph sits between, and how far into it.
	fn frames_in_play(&self, bank: &WaveBank) -> (usize, usize, f32) {
		crate::prelude::page_position(self.smooth_factor, bank.len())
	}
}

impl Parameters for WaveFramesOsc {
	fn get_parameters(&self) -> Vec<Parameter> {
		let mut parameters = vec![
			morph_parameter(self.smooth_factor),
			x_fade_parameter(self.x_fade),
		];

		if let Ok(binary) = to_binary(&self.imported()) {
			parameters.push(Parameter::new("imported", Value::Serialized(binary)));
		}

		parameters
	}

	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		match identifier {
			"smooth_factor" => {
				let SetValue::Float(value) = value else {
					return false;
				};

				self.set_smooth_factor(value);
				true
			}
			"x_fade" => {
				let SetValue::Float(value) = value else {
					return false;
				};

				self.set_x_fade(value);
				true
			}
			"imported" => {
				let SetValue::Serialized(data) = value else {
					return false;
				};

				match from_binary::<Option<ImportedWave>>(data) {
					Ok(wave) => {
						self.set_imported(wave);
						true
					}
					Err(_) => false,
				}
			}
			_ => false,
		}
	}
}

impl Oscillator<2> for WaveFramesOsc {
	fn play_at(&self, frequency: f32, time: f32, phase: [f32; 2]) -> [f32; 2] {
		let bank = self.bank();
		let (from, to, blend) = self.frames_in_play(&bank);
		let mut output = [0.0; 2];

		for (channel, value) in output.iter_mut().enumerate() {
			let t = (frequency * time + phase[channel]).rem_euclid(1.0);

			*value = bank.sample(t, from, to, blend, self.max_harmonic, self.x_fade);
		}

		output
	}

	fn set_pitch(&mut self, frequency: f32, sample_rate: usize) {
		self.max_harmonic = if frequency.abs() < 1e-6 {
			f32::INFINITY
		} else {
			sample_rate as f32 * 0.5 / frequency.abs()
		};
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, _: &mut egui::Ui, _: String) {}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn the_classic_shapes_morph() {
		let mut table = WaveFramesOsc::new();

		// Low enough that every partial fits: the raw frames, no band limit.
		Oscillator::<2>::set_pitch(&mut table, 20.0, 48_000);

		let rms = |table: &WaveFramesOsc| {
			let mut sum = 0.0;

			for index in 0..1024 {
				let value = table.play_at(20.0, index as f32 / 1024.0, [0.0, 0.0])[0];

				sum += value * value;
			}

			(sum / 1024.0).sqrt()
		};

		table.set_smooth_factor(0.0);
		let sine = rms(&table);
		table.set_smooth_factor(1.0);
		let square = rms(&table);

		assert!((sine - 0.707).abs() < 0.02, "{sine}");
		assert!((square - 1.0).abs() < 0.02, "{square}");
	}

	#[test]
	fn an_import_replaces_the_frames() {
		let mut table = WaveFramesOsc::new();
		table.set_imported(Some(ImportedWave::new(
			"test",
			vec![vec![0.5; WAVE_FRAME_LEN], vec![-0.5; WAVE_FRAME_LEN]],
		)));
		Oscillator::<2>::set_pitch(&mut table, 110.0, 48_000);

		assert_eq!(table.imported().map(|wave| wave.name), Some("test".into()));

		table.set_smooth_factor(0.0);
		assert!((table.play_at(110.0, 0.0, [0.0, 0.0])[0] - 0.5).abs() < 1e-6);
		table.set_smooth_factor(0.5);
		assert!(table.play_at(110.0, 0.0, [0.0, 0.0])[0].abs() < 1e-6);
		table.set_smooth_factor(1.0);
		assert!((table.play_at(110.0, 0.0, [0.0, 0.0])[0] + 0.5).abs() < 1e-6);
	}

	/// A bright frame played high loses its top partials instead of folding them
	/// back: the wave gets smooth, so neighbouring samples stop jumping.
	#[test]
	fn a_bright_frame_is_band_limited() {
		let square: Vec<f32> = (0..WAVE_FRAME_LEN)
			.map(|index| {
				if (index as f32 / WAVE_FRAME_LEN as f32) < 0.5 {
					1.0
				} else {
					-1.0
				}
			})
			.collect();
		let bank = WaveBank::new(Some("square".into()), vec![square]);
		let mut worst: f32 = 0.0;
		let mut previous = bank.sample(0.0, 0, 0, 0.0, 4.0, 0.0);

		for index in 1..1024 {
			let value = bank.sample(index as f32 / 1024.0, 0, 0, 0.0, 4.0, 0.0);

			worst = worst.max((value - previous).abs());
			previous = value;
		}

		assert!(worst < 0.5, "the wave jumps by {worst}");
	}

	/// The edge crossfade pulls the end of a frame towards its beginning, so the
	/// jump at the loop point shrinks.
	#[test]
	fn the_edge_crossfade_closes_the_loop() {
		let saw: Vec<f32> = (0..WAVE_FRAME_LEN)
			.map(|index| 2.0 * index as f32 / WAVE_FRAME_LEN as f32 - 1.0)
			.collect();
		let bank = WaveBank::new(Some("saw".into()), vec![saw]);

		let jump = |fade: f32| {
			let first = bank.sample(0.0, 0, 0, 0.0, 1024.0, fade);
			let last = bank.sample(0.999, 0, 0, 0.0, 1024.0, fade);

			(last - first).abs()
		};

		assert!(jump(0.0) > 1.5, "{}", jump(0.0));
		assert!(
			jump(0.5) < jump(0.0) * 0.5,
			"{} vs {}",
			jump(0.5),
			jump(0.0)
		);
	}

	/// Moving the crossfade must not rebuild the table: the host re-sends every
	/// parameter on every block, and a rebuild there would starve the audio thread.
	#[test]
	fn the_crossfade_does_not_rebuild_the_table() {
		let saw: Vec<f32> = (0..WAVE_FRAME_LEN)
			.map(|index| 2.0 * index as f32 / WAVE_FRAME_LEN as f32 - 1.0)
			.collect();
		let mut table = WaveFramesOsc::new();
		table.set_imported(Some(ImportedWave::new("saw", vec![saw])));

		let before = table.bank().id();
		table.set_x_fade(1.0);

		assert!((table.x_fade() - 1.0).abs() < 1e-6);
		assert_eq!(table.bank().id(), before);
	}

	/// The table's two flips: backwards in time, and upside down.
	#[test]
	fn a_table_can_be_reversed_and_inverted() {
		let wave = ImportedWave::new("ramp", vec![vec![0.0, 0.5, 1.0], vec![-1.0, 0.0, 1.0]]);

		assert_eq!(wave.reversed().frames[0], vec![1.0, 0.5, 0.0]);
		assert_eq!(wave.reversed().frames[1], vec![1.0, 0.0, -1.0]);
		assert_eq!(wave.inverted().frames[1], vec![1.0, 0.0, -1.0]);
		assert_eq!(wave.inverted().name, "ramp");
	}

	#[test]
	fn the_table_is_state_not_a_control() {
		let table = WaveFramesOsc::new();
		table.set_imported(Some(ImportedWave::new(
			"saved",
			vec![vec![0.25; WAVE_FRAME_LEN]],
		)));

		let parameters = table.get_parameters();
		let imported = parameters
			.iter()
			.find(|parameter| parameter.identifier == "imported")
			.expect("the imported table");
		let Value::Serialized(data) = imported.value.clone() else {
			panic!("the table is not serialized")
		};

		let mut restored = WaveFramesOsc::new();
		assert!(restored.set_parameter("imported", SetValue::Serialized(data)));
		assert_eq!(
			restored.imported().map(|wave| wave.name),
			Some("saved".into())
		);
	}
}
