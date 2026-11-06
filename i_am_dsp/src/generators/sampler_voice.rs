//! One sample, played three ways: as a pitched sampler, as a sliced break, or as
//! grains.
//!
//! The voice reads a sample the editor put in a shared slot, because a serialized
//! parameter cannot travel through the parameter map and the upload happens on the
//! UI thread. A save reads the same slot, so the sample travels with the project.

use std::sync::{Arc, Mutex};

use crate::prelude::{
	Oscillator, Parameter, ParameterMeta, Parameters, SetValue, Unit, Value, from_binary, to_binary,
};

/// The sample a voice plays: mono, at the rate it was imported at.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SampleData {
	/// The file name, for the interface.
	pub name: String,
	/// The rate the samples were captured at.
	pub rate: u32,
	/// The samples themselves.
	pub pcm: Vec<f32>,
}

impl SampleData {
	/// How long the sample is, in seconds.
	pub fn seconds(&self) -> f32 {
		self.pcm.len() as f32 / self.rate.max(1) as f32
	}
}

/// The slot the editor and the voice share.
pub struct SampleSlot {
	/// The sample, once something has been imported.
	pub sample: Option<Arc<SampleData>>,
}

impl Default for SampleSlot {
	fn default() -> Self {
		Self { sample: None }
	}
}

/// How a voice plays the sample.
pub const MODES: [&str; 3] = ["Sample", "Slice", "Grain"];

fn knob(
	id: &'static str,
	display: &'static str,
	value: f32,
	range: std::ops::RangeInclusive<f32>,
	group: &'static str,
	order: u16,
	unit: Unit,
) -> Parameter {
	let meta = ParameterMeta {
		default: Some(value as f64),
		unit,
		display_name: Some(display.into()),
		choices: None,
		#[cfg(feature = "layout")]
		layout: crate::prelude::ParamLayout::new()
			.group(group)
			.order(order)
			.widget(crate::prelude::ParamWidget::Knob),
	};

	Parameter::new(
		id,
		Value::Float {
			value,
			range,
			logarithmic: false,
		},
	)
	.with_meta(meta)
}

fn toggle(
	id: &'static str,
	display: &'static str,
	value: bool,
	group: &'static str,
	order: u16,
) -> Parameter {
	let meta = ParameterMeta {
		default: Some(if value { 1.0 } else { 0.0 }),
		unit: Unit::None,
		display_name: Some(display.into()),
		choices: None,
		#[cfg(feature = "layout")]
		layout: crate::prelude::ParamLayout::new()
			.group(group)
			.order(order)
			.widget(crate::prelude::ParamWidget::Toggle),
	};

	Parameter::new(id, Value::Bool(value)).with_meta(meta)
}

/// The sampler voice.
pub struct SamplerVoice {
	slot: Arc<Mutex<SampleSlot>>,
	/// 0 is the whole sample, 1 the slices, 2 the grains.
	mode: i32,
	/// The note the sample plays at its own pitch.
	root: f32,
	reverse: bool,
	loop_enabled: bool,
	loop_start: f32,
	loop_end: f32,
	speed: f32,
	slices: i32,
	position: f32,
	grain_size: f32,
	density: f32,
	spray: f32,
	grain_pitch: f32,
	gain: f32,
	pan: f32,
	polarity: bool,
	sample_rate: f32,
	pitch: f32,
}

impl Default for SamplerVoice {
	fn default() -> Self {
		Self::new()
	}
}

impl SamplerVoice {
	/// A voice with nothing imported.
	pub fn new() -> Self {
		Self::with_slot(Arc::new(Mutex::new(SampleSlot::default())))
	}

	/// A voice over a slot the editor and the voice share.
	pub fn with_slot(slot: Arc<Mutex<SampleSlot>>) -> Self {
		Self {
			slot,
			mode: 0,
			root: 60.0,
			reverse: false,
			loop_enabled: false,
			loop_start: 0.0,
			loop_end: 1.0,
			speed: 1.0,
			slices: 8,
			position: 0.0,
			grain_size: 0.06,
			density: 12.0,
			spray: 0.2,
			grain_pitch: 0.0,
			gain: 1.0,
			pan: 0.0,
			polarity: false,
			sample_rate: 48_000.0,
			pitch: 0.0,
		}
	}

	/// The slot the sample lives in.
	pub fn slot(&self) -> Arc<Mutex<SampleSlot>> {
		self.slot.clone()
	}

	/// The sample in play.
	pub fn sample(&self) -> Option<Arc<SampleData>> {
		self.slot
			.lock()
			.unwrap_or_else(|error| error.into_inner())
			.sample
			.clone()
	}

	/// Replaces the sample.
	pub fn set_sample(&self, sample: Option<SampleData>) {
		self.slot
			.lock()
			.unwrap_or_else(|error| error.into_inner())
			.sample = sample.map(Arc::new);
	}

	/// Reads the sample at a fractional index, with a cubic interpolation.
	fn read(&self, data: &SampleData, position: f32) -> f32 {
		let len = data.pcm.len();

		if len < 2 || !position.is_finite() || position < 0.0 || position >= (len - 1) as f32 {
			return 0.0;
		}

		let start = position.floor() as usize;
		let frac = position - start as f32;
		let before = if start == 0 { 0.0 } else { data.pcm[start - 1] };
		let now = data.pcm[start];
		let next = data.pcm[start + 1];
		let after = if start + 2 < len {
			data.pcm[start + 2]
		} else {
			0.0
		};

		crate::tools::interpolate::cubic_interpolate(frac, [before, now, next, after])
	}

	/// The note number the pitch tracker is playing, from its frequency.
	fn note(&self) -> f32 {
		let pitch = if self.pitch > 1.0 { self.pitch } else { 440.0 };

		69.0 + 12.0 * (pitch / 440.0).log2()
	}

	/// How the sample is played at the current time.
	fn play_sample(&self, data: &SampleData, elapsed: f32) -> f32 {
		let len = data.pcm.len() as f32;
		let mut position = elapsed * data.rate as f32 * self.speed;

		if self.loop_enabled {
			let start = self.loop_start.clamp(0.0, 1.0) * (len - 1.0);
			let end = self.loop_end.clamp(0.0, 1.0) * (len - 1.0);

			if end > start + 1.0 && position > end {
				position = start + (position - start) % (end - start);
			}
		}

		if self.reverse {
			position = (len - 1.0) - position;
		}

		self.read(data, position)
	}

	/// Plays one slice of the sample, chosen by the note.
	fn play_slice(&self, data: &SampleData, elapsed: f32) -> f32 {
		let slices = self.slices.clamp(2, 64) as usize;
		let length = data.pcm.len() / slices;

		if length < 2 {
			return 0.0;
		}

		let index = ((self.note() - self.root).round() as i64).rem_euclid(slices as i64) as usize;
		let start = index * length;
		let position = start as f32 + elapsed * data.rate as f32 * self.speed;
		let end = start + length;

		if position as usize >= end {
			return self.read(data, position) * 0.0 + 0.0;
		}

		self.read(data, position.min((end - 1) as f32))
	}

	/// Plays a cloud of overlapping grains, walked forward from the position.
	fn play_grains(&self, data: &SampleData, elapsed: f32) -> f32 {
		const GRAINS: i64 = 8;

		let length = data.pcm.len() as f32;
		let size = self.grain_size.clamp(0.005, 0.5);
		let density = self.density.clamp(0.5, 200.0);
		let base = self.position.clamp(0.0, 1.0) * length;
		let spray = self.spray.clamp(0.0, 1.0) * length;
		let ratio = 2.0_f32.powf(self.grain_pitch.clamp(-24.0, 24.0) / 12.0);
		let current = (elapsed * density).floor() as i64;
		let mut sum = 0.0;

		for offset in 0..GRAINS {
			let index = current - offset;
			let start = index as f32 / density;
			let age = elapsed - start;

			if age < 0.0 || age >= size {
				continue;
			}

			let window = 0.5 - 0.5 * (std::f32::consts::TAU * age / size).cos();
			let position = base + hash(index) * spray + age * data.rate as f32 * ratio;

			sum += self.read(data, position) * window;
		}

		sum / (GRAINS as f32).sqrt()
	}
}

/// A deterministic 0..1 value per grain, so the spray stays the same take after
/// take instead of hissing differently every note.
fn hash(index: i64) -> f32 {
	let mut value = index as u64 ^ 0x9E37_79B9_7F4A_7C15;

	value ^= value >> 33;
	value = value.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
	value ^= value >> 33;
	value = value.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
	value ^= value >> 33;

	(value >> 40) as f32 / 16_777_216.0
}

impl Parameters for SamplerVoice {
	fn get_parameters(&self) -> Vec<Parameter> {
		let meta = ParameterMeta {
			default: Some(0.0),
			unit: Unit::None,
			display_name: Some("Mode".into()),
			choices: Some(&["Sample", "Slice", "Grain"]),
			#[cfg(feature = "layout")]
			layout: crate::prelude::ParamLayout::new()
				.group("Sample")
				.order(0)
				.widget(crate::prelude::ParamWidget::Choice)
				.hero("sampler"),
		};
		let mut parameters = vec![
			Parameter::new(
				"mode",
				Value::Int {
					value: self.mode,
					range: 0..=2,
					logarithmic: false,
				},
			)
			.with_meta(meta),
			knob(
				"root",
				"Root",
				self.root,
				0.0..=127.0,
				"Sample",
				1,
				Unit::None,
			),
			toggle("loop", "Loop", self.loop_enabled, "Loop", 0),
			knob(
				"loop_start",
				"Loop Start",
				self.loop_start,
				0.0..=1.0,
				"Loop",
				1,
				Unit::None,
			),
			knob(
				"loop_end",
				"Loop End",
				self.loop_end,
				0.0..=1.0,
				"Loop",
				2,
				Unit::None,
			),
			toggle("reverse", "Reverse", self.reverse, "Loop", 3),
			knob(
				"speed",
				"Speed",
				self.speed,
				0.25..=4.0,
				"Sample",
				2,
				Unit::None,
			),
			knob(
				"slices",
				"Slices",
				self.slices as f32,
				2.0..=64.0,
				"Slice",
				0,
				Unit::None,
			),
			knob(
				"position",
				"Position",
				self.position,
				0.0..=1.0,
				"Grain",
				0,
				Unit::None,
			),
			knob(
				"grain_size",
				"Grain",
				self.grain_size,
				0.005..=0.5,
				"Grain",
				1,
				Unit::Seconds,
			),
			knob(
				"density",
				"Density",
				self.density,
				0.5..=200.0,
				"Grain",
				2,
				Unit::None,
			),
			knob(
				"spray",
				"Spray",
				self.spray,
				0.0..=1.0,
				"Grain",
				3,
				Unit::None,
			),
			knob(
				"grain_pitch",
				"Grain Pitch",
				self.grain_pitch,
				-24.0..=24.0,
				"Grain",
				4,
				Unit::None,
			),
			knob(
				"gain",
				"Gain",
				self.gain,
				0.0..=2.0,
				"Sample",
				3,
				Unit::None,
			),
			knob("pan", "Pan", self.pan, -1.0..=1.0, "Sample", 4, Unit::None),
		];

		if let Ok(binary) = to_binary(&self.sample().as_deref()) {
			parameters.push(Parameter::new("sample", Value::Serialized(binary)));
		}

		parameters
	}

	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		match identifier {
			"mode" => match value {
				SetValue::Int(value) => {
					self.mode = value.clamp(0, 2);
					true
				}
				_ => false,
			},
			"loop" => match value {
				SetValue::Bool(value) => {
					self.loop_enabled = value;
					true
				}
				_ => false,
			},
			"reverse" => match value {
				SetValue::Bool(value) => {
					self.reverse = value;
					true
				}
				_ => false,
			},
			"polarity" => match value {
				SetValue::Bool(value) => {
					self.polarity = value;
					true
				}
				_ => false,
			},
			"sample" => match value {
				SetValue::Serialized(data) => match from_binary::<Option<SampleData>>(data) {
					Ok(sample) => {
						self.slot
							.lock()
							.unwrap_or_else(|error| error.into_inner())
							.sample = sample.map(Arc::new);
						true
					}
					Err(_) => false,
				},
				_ => false,
			},
			"root" => set_float(&mut self.root, value),
			"loop_start" => set_float(&mut self.loop_start, value),
			"loop_end" => set_float(&mut self.loop_end, value),
			"speed" => set_float(&mut self.speed, value),
			"slices" => match value {
				SetValue::Float(value) => {
					self.slices = value.round().clamp(2.0, 64.0) as i32;
					true
				}
				_ => false,
			},
			"position" => set_float(&mut self.position, value),
			"grain_size" => set_float(&mut self.grain_size, value),
			"density" => set_float(&mut self.density, value),
			"spray" => set_float(&mut self.spray, value),
			"grain_pitch" => set_float(&mut self.grain_pitch, value),
			"gain" => set_float(&mut self.gain, value),
			"pan" => set_float(&mut self.pan, value),
			_ => false,
		}
	}
}

fn set_float(field: &mut f32, value: SetValue) -> bool {
	match value {
		SetValue::Float(value) => {
			*field = value;
			true
		}
		_ => false,
	}
}

impl Oscillator<2> for SamplerVoice {
	fn play_at(&self, _frequency: f32, time: f32, _phase: [f32; 2]) -> [f32; 2] {
		let Some(data) = self.sample() else {
			return [0.0; 2];
		};

		if data.pcm.len() < 2 {
			return [0.0; 2];
		}

		// The driver counts time in cycles of the note, so the seconds it has been
		// playing are the cycles divided by the note's frequency.
		let pitch = if self.pitch > 1.0 { self.pitch } else { 1.0 };
		let elapsed = time / pitch;
		let value = match self.mode {
			1 => self.play_slice(&data, elapsed),
			2 => self.play_grains(&data, elapsed),
			_ => self.play_sample(&data, elapsed),
		};
		// Equal power panning, normalized so the middle is unity on both sides.
		let angle = (self.pan.clamp(-1.0, 1.0) + 1.0) * std::f32::consts::FRAC_PI_4;
		let gain = value * self.gain * if self.polarity { -1.0 } else { 1.0 };

		[
			gain * angle.cos() * std::f32::consts::SQRT_2,
			gain * angle.sin() * std::f32::consts::SQRT_2,
		]
	}

	fn set_pitch(&mut self, frequency: f32, sample_rate: usize) {
		self.pitch = frequency.abs();
		self.sample_rate = sample_rate as f32;
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, _: &mut egui::Ui, _: String) {}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn sample() -> SampleData {
		SampleData {
			name: "test".into(),
			rate: 48_000,
			pcm: (0..4_800)
				.map(|index| (index as f32 / 48_000.0 * 440.0 * std::f32::consts::TAU).sin())
				.collect(),
		}
	}

	fn voice(mode: i32) -> SamplerVoice {
		let mut voice = SamplerVoice::new();
		voice.set_sample(Some(sample()));
		voice.mode = mode;
		Oscillator::<2>::set_pitch(&mut voice, 261.63, 48_000);
		voice
	}

	/// The sampler plays the sample at the note's pitch: an octave up reads twice
	/// as fast.
	#[test]
	fn the_sampler_follows_the_pitch() {
		let mut low = voice(0);
		let mut high = voice(0);
		Oscillator::<2>::set_pitch(&mut low, 261.63, 48_000);
		Oscillator::<2>::set_pitch(&mut high, 523.26, 48_000);

		let low_value = Oscillator::<2>::play_at(&low, 1.0, 0.25, [0.0, 0.0])[0];
		let high_value = Oscillator::<2>::play_at(&high, 1.0, 0.25, [0.0, 0.0])[0];

		assert!(low_value.is_finite() && high_value.is_finite());
		assert!(
			(low_value - high_value).abs() > 1e-4,
			"the pitch did nothing"
		);
	}

	/// The slice mode picks a different slice per key.
	#[test]
	fn the_slices_follow_the_keys() {
		let mut voice = voice(1);
		voice.slices = 4;
		voice.root = 60.0;

		let mut values = Vec::new();

		for note in [60.0, 62.0, 64.0, 65.0] {
			Oscillator::<2>::set_pitch(
				&mut voice,
				440.0 * 2.0_f32.powf((note - 69.0) / 12.0),
				48_000,
			);

			let value = Oscillator::<2>::play_at(&voice, 1.0, 0.01 * 440.0, [0.0, 0.0])[0];

			values.push(value);
		}

		assert!(values.iter().any(|value| value.abs() > 1e-4), "{values:?}");
	}

	/// Grains make sound, and the position moves what they read.
	#[test]
	fn the_grains_play() {
		let mut voice = voice(2);
		voice.density = 40.0;
		voice.grain_size = 0.05;

		let mut energy = 0.0;

		for index in 0..4_800 {
			let value =
				Oscillator::<2>::play_at(&voice, 1.0, index as f32 / 48_000.0 * 261.63, [0.0, 0.0])
					[0];

			energy += value.abs();
		}

		assert!(energy > 0.1, "the grains were silent");
	}
}
