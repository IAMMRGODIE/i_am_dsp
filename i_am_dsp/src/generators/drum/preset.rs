use std::sync::Arc;

use crate::generators::sampler_voice::SampleData;

use super::Drum;
use super::curve::{Curve, Point};

/// The click sample a host imports, shared between the page and the voices.
#[derive(Default)]
pub struct ClickSlot {
	data: std::sync::Mutex<Option<SampleData>>,
	generation: std::sync::atomic::AtomicU64,
}

impl ClickSlot {
	/// An empty slot.
	pub fn new() -> Arc<Self> {
		Arc::new(Self::default())
	}

	/// The sample currently loaded.
	pub fn load(&self) -> Option<SampleData> {
		self.data.lock().ok()?.clone()
	}

	/// How many times the slot has changed.
	pub fn generation(&self) -> u64 {
		self.generation.load(std::sync::atomic::Ordering::Acquire)
	}

	/// Installs a sample and bumps the generation.
	pub fn set(&self, data: SampleData) {
		if let Ok(mut current) = self.data.lock() {
			*current = Some(data);
			self.generation
				.fetch_add(1, std::sync::atomic::Ordering::Release);
		}
	}

	/// Forgets the sample, for a restored kit that had none.
	pub fn clear(&self) {
		if let Ok(mut current) = self.data.lock() {
			*current = None;
			self.generation
				.fetch_add(1, std::sync::atomic::Ordering::Release);
		}
	}

	/// What the page draws.
	pub fn json(&self) -> String {
		match self.load() {
			Some(data) => format!(
				"{{\"name\":\"{}\",\"rate\":{},\"samples\":{}}}",
				data.name.replace('"', ""),
				data.rate,
				data.pcm.len()
			),
			None => "{\"name\":null,\"rate\":0,\"samples\":0}".to_string(),
		}
	}
}

/// The default amplitude shape of each drum, in seconds.
///
/// The curve is the envelope and the length both: the last point is when the
/// voice stops. There is no Decay knob to fight with the shape.
pub fn default_amp_templates() -> [Curve; 6] {
	let simple = |end: f32| {
		Curve::new(vec![
			Point {
				time: 0.0,
				value: 1.0,
				bend: -2.5,
			},
			Point {
				time: end,
				value: 0.0,
				bend: 0.0,
			},
		])
	};

	[
		simple(0.45),
		simple(0.18),
		Curve::new(vec![
			Point {
				time: 0.0,
				value: 1.0,
				bend: 0.0,
			},
			Point {
				time: 0.065,
				value: 0.0,
				bend: 0.0,
			},
		]),
		Curve::new(
			[
				(0.0, 1.0, -3.0),
				(0.009, 0.0, 0.0),
				(0.012, 0.9, -3.0),
				(0.021, 0.0, 0.0),
				(0.024, 0.8, -2.5),
				(0.033, 0.0, 0.0),
				(0.036, 0.7, -1.5),
				(0.060, 0.32, 0.0),
				(0.35, 0.0, 0.0),
			]
			.into_iter()
			.map(|(time, value, bend)| Point { time, value, bend })
			.collect(),
		),
		simple(0.5),
		Curve::new(vec![
			Point {
				time: 0.0,
				value: 1.0,
				bend: 0.0,
			},
			Point {
				time: 6.0,
				value: 0.0,
				bend: 0.0,
			},
		]),
	]
}

/// A fitted drum: the modal partials and noise bands an analysis produced.
///
/// This is the shape the offline fitters write and the page posts to `/preset`:
/// the engine does not care whether the numbers came from a spectrogram or from
/// a hand, only that they describe a bank.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct DrumPreset {
	/// A name for the page.
	#[serde(default)]
	pub name: String,
	/// How long the bank rings, in seconds.
	#[serde(default = "default_preset_decay")]
	pub decay: f32,
	/// The decaying partials.
	#[serde(default)]
	pub partials: Vec<PresetPartial>,
	/// The noise bands.
	#[serde(default)]
	pub noise: Vec<PresetNoise>,
}

/// One fitted partial.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct PresetPartial {
	/// Frequency in Hz.
	pub frequency: f32,
	/// Linear amplitude.
	#[serde(default = "default_preset_amplitude")]
	pub amplitude: f32,
	/// Time to -60 dB, in seconds.
	#[serde(default = "default_preset_decay")]
	pub t60: f32,
}

/// One fitted noise band.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct PresetNoise {
	/// Centre frequency in Hz.
	pub frequency: f32,
	/// Resonance.
	#[serde(default = "default_preset_q")]
	pub q: f32,
	/// Linear amplitude.
	#[serde(default = "default_preset_amplitude")]
	pub gain: f32,
	/// Time to -60 dB, in seconds.
	#[serde(default = "default_preset_decay")]
	pub t60: f32,
}

fn default_preset_decay() -> f32 {
	0.5
}

fn default_preset_amplitude() -> f32 {
	1.0
}

fn default_preset_q() -> f32 {
	0.8
}

/// The fitted preset a host posts, shared between the page and the voices.
#[derive(Default)]
pub struct FitSlot {
	data: std::sync::Mutex<Option<DrumPreset>>,
	generation: std::sync::atomic::AtomicU64,
}

impl FitSlot {
	/// An empty slot.
	pub fn new() -> Arc<Self> {
		Arc::new(Self::default())
	}

	/// The preset currently fitted.
	pub fn load(&self) -> Option<DrumPreset> {
		self.data.lock().ok()?.clone()
	}

	/// How many times the slot has changed.
	pub fn generation(&self) -> u64 {
		self.generation.load(std::sync::atomic::Ordering::Acquire)
	}

	/// Installs a preset and bumps the generation.
	pub fn set(&self, preset: DrumPreset) {
		if let Ok(mut current) = self.data.lock() {
			*current = Some(preset);
			self.generation
				.fetch_add(1, std::sync::atomic::Ordering::Release);
		}
	}

	/// A summary for the page.
	pub fn json(&self) -> String {
		match self.load() {
			Some(preset) => format!(
				"{{\"name\":\"{}\",\"partials\":{},\"noise\":{},\"decay\":{}}}",
				escape_name(&preset.name),
				preset.partials.len(),
				preset.noise.len(),
				preset.decay
			),
			None => "{\"name\":null,\"partials\":0,\"noise\":0,\"decay\":0}".to_string(),
		}
	}
}

/// The six one-shot slots, one per drum: a sample replaces the synthesis for
/// that drum until it is cleared.
#[derive(Default)]
pub struct Samples {
	slots: std::sync::Mutex<[Option<SampleData>; 6]>,
	generation: std::sync::atomic::AtomicU64,
}

impl Samples {
	/// An empty set of slots.
	pub fn new() -> Arc<Self> {
		Arc::new(Self::default())
	}

	/// The sample in one slot.
	pub fn get(&self, drum: Drum) -> Option<SampleData> {
		self.slots
			.lock()
			.ok()
			.and_then(|slots| slots[drum.index()].clone())
	}

	/// Installs a sample and bumps the generation.
	pub fn set(&self, drum: Drum, data: Option<SampleData>) {
		if let Ok(mut slots) = self.slots.lock() {
			slots[drum.index()] = data;
			self.generation
				.fetch_add(1, std::sync::atomic::Ordering::Release);
		}
	}

	/// The six slots as they are, for a save.
	pub fn snapshot(&self) -> [Option<SampleData>; 6] {
		self.slots
			.lock()
			.map(|slots| slots.clone())
			.unwrap_or_default()
	}

	/// Puts six saved slots back and bumps the generation.
	pub fn restore(&self, slots: [Option<SampleData>; 6]) {
		if let Ok(mut current) = self.slots.lock() {
			*current = slots;
			self.generation
				.fetch_add(1, std::sync::atomic::Ordering::Release);
		}
	}

	/// How many times the slots have changed.
	pub fn generation(&self) -> u64 {
		self.generation.load(std::sync::atomic::Ordering::Acquire)
	}

	/// What the page draws: one entry per drum, in GM order.
	pub fn json(&self) -> String {
		let slots = self
			.slots
			.lock()
			.map(|slots| slots.clone())
			.unwrap_or_default();
		let mut out = String::from("[");

		for (index, slot) in slots.iter().enumerate() {
			if index > 0 {
				out.push(',');
			}

			match slot {
				Some(data) => {
					let _ = std::fmt::Write::write_fmt(
						&mut out,
						format_args!(
							"{{\"name\":\"{}\",\"rate\":{},\"samples\":{},\"peaks\":[{}]}}",
							escape_name(&data.name),
							data.rate,
							data.pcm.len(),
							peaks_json(&data.pcm)
						),
					);
				}
				None => out.push_str("null"),
			}
		}

		out.push(']');
		out
	}
}

/// One slot index per drum: the order the page and the mixer use.
impl Drum {
	pub(super) fn index(self) -> usize {
		match self {
			Drum::Kick => 0,
			Drum::Snare => 1,
			Drum::HatClosed | Drum::HatPedal | Drum::HatOpen => 2,
			Drum::Clap => 3,
			Drum::Tom => 4,
			Drum::Crash | Drum::Ride => 5,
		}
	}
}

/// The slot name the page posts, mapped to the drum it fills.
pub fn slot_drum(name: &str) -> Option<Drum> {
	Some(match name {
		"kick" => Drum::Kick,
		"snare" => Drum::Snare,
		"hat" => Drum::HatClosed,
		"clap" => Drum::Clap,
		"tom" => Drum::Tom,
		"cymbal" => Drum::Crash,
		_ => return None,
	})
}

/// The sample's shape in a few numbers, for the page to draw.
fn peaks_json(pcm: &[f32]) -> String {
	const COLUMNS: usize = 32;

	let mut out = String::new();

	for column in 0..COLUMNS {
		let start = column * pcm.len() / COLUMNS;
		let end = ((column + 1) * pcm.len() / COLUMNS).max(start + 1);
		let peak = pcm[start.min(pcm.len())..end.min(pcm.len())]
			.iter()
			.fold(0.0f32, |peak, sample| peak.max(sample.abs()));

		if column > 0 {
			out.push(',');
		}

		let _ = std::fmt::Write::write_fmt(&mut out, format_args!("{peak:.3}"));
	}

	out
}

/// A name with the characters JSON cannot carry removed.
fn escape_name(name: &str) -> String {
	name.replace(['"', '\\'], "")
}

/// Decodes a WAV into a mono one-shot, at the rate it was recorded at.
pub fn decode_click(bytes: &[u8]) -> Result<SampleData, String> {
	let pcm = crate::tools::pcm_data::load_from_bytes::<2>(bytes.to_vec())
		.map_err(|error| error.to_string())?;
	let [left, right] = pcm.pcm_data;
	let mono: Vec<f32> = if right.is_empty() {
		left
	} else {
		left.iter()
			.zip(&right)
			.map(|(left, right)| 0.5 * (left + right))
			.collect()
	};

	Ok(SampleData {
		right: None,
		name: "click".to_string(),
		rate: pcm.sample_rate as u32,
		pcm: mono,
	})
}

/// The impulse response of the cymbal tail: a quarter second of a real crash,
/// captured at 48 kHz, embedded so a hit never has to look for a file.
pub(super) fn load_cymbal_ir(rate: usize) -> Vec<f32> {
	let bytes = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/cymbal-ir.f32"));
	let mut ir: Vec<f32> = bytes
		.as_chunks::<4>()
		.0
		.iter()
		.map(|bytes| f32::from_le_bytes(*bytes))
		.collect();

	if rate != 48_000
		&& !ir.is_empty()
		&& let Ok([resampled]) = crate::tools::pcm_data::resample([ir.clone()], 48_000, rate)
	{
		ir = resampled;
	}

	ir
}

/// The impulse response of the hat's colour: twenty milliseconds of a real
/// closed hat, embedded so a hit never has to look for a file.
pub(super) fn load_hat_ir(rate: usize) -> Vec<f32> {
	let bytes = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/hat-ir.f32"));
	let mut ir: Vec<f32> = bytes
		.as_chunks::<4>()
		.0
		.iter()
		.map(|bytes| f32::from_le_bytes(*bytes))
		.collect();

	if rate != 48_000
		&& !ir.is_empty()
		&& let Ok([resampled]) = crate::tools::pcm_data::resample([ir.clone()], 48_000, rate)
	{
		ir = resampled;
	}

	ir
}

/// The values each kit preset writes, in parameter order.
///
/// The knobs stay editable after a preset is chosen: this is a starting point,
/// not a mode.
pub(super) fn kit_values(index: i32) -> &'static [(&'static str, f32)] {
	const STUDIO: &[(&str, f32)] = &[
		("tune", 52.0),
		("decay", 0.45),
		("click", 0.5),
		("click_tone", 1_800.0),
		("snare_tune", 180.0),
		("snare_decay", 0.18),
		("snare_wires", 0.5),
		("snare_click", 0.4),
		("hat_decay", 0.05),
		("hat_open", 0.45),
		("hat_cutoff", 6_500.0),
		("clap_tone", 1_600.0),
		("clap_decay", 0.35),
		("tom_decay", 0.5),
		("cymbal_decay", 2.5),
		("cymbal_fm", 0.7),
		("humanize", 0.15),
		("velocity_brightness", 0.35),
	];
	const EIGHTY_EIGHT: &[(&str, f32)] = &[
		("tune", 48.0),
		("decay", 0.85),
		("click", 0.3),
		("click_tone", 900.0),
		("snare_tune", 200.0),
		("snare_decay", 0.25),
		("snare_wires", 0.75),
		("snare_click", 0.25),
		("hat_decay", 0.07),
		("hat_open", 0.6),
		("hat_cutoff", 7_000.0),
		("clap_tone", 1_100.0),
		("clap_decay", 0.5),
		("tom_decay", 0.4),
		("cymbal_decay", 3.5),
		("cymbal_fm", 0.45),
		("humanize", 0.1),
		("velocity_brightness", 0.5),
	];

	if index == 1 { EIGHTY_EIGHT } else { STUDIO }
}

/// The snare's three shapes: the shell's level, the wires' level, and the
/// shell's pitch in octaves (0 is 1x, 1 is 2x, ...).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SnareCurves {
	/// The shell's amplitude, in seconds.
	pub base: Curve,
	/// The wires' amplitude, in seconds.
	pub noise: Curve,
	/// The shell's pitch, in octaves.
	pub pitch: Curve,
}

impl Default for SnareCurves {
	fn default() -> Self {
		Self {
			base: Curve::from_triples([(0.0, 1.0, -2.5), (0.18, 0.0, 0.0)]),
			noise: Curve::from_triples([(0.0, 1.0, -2.5), (0.16, 0.0, 0.0)]),
			pitch: Curve::from_triples([(0.0, 0.678, -1.0), (0.015, 0.0, 0.0)]),
		}
	}
}
