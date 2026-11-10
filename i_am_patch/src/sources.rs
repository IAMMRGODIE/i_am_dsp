//! The modulation sources a patch can use.
//!
//! These are the reusable half of the rack's modulation system: an LFO, an
//! envelope follower, a smoothed random source, a wavetable oscillator, a pitch
//! tracker and the last MIDI note. The runtime that drives them is in
//! [`modulation`](crate::modulation).

use serde::{Deserialize, Serialize};

/// The knobs of the four modulation sources.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SourceSettings {
	/// LFO rate in hertz.
	pub lfo_rate: f32,
	/// 0 sine, 1 triangle, 2 saw, 3 square, 4 custom.
	pub lfo_shape: i32,
	/// The EditableLfo points: (time 0..1, value 0..1, bend).
	#[serde(default = "default_lfo_points")]
	pub lfo_points: Vec<(f32, f32, f32)>,
	/// Envelope attack in milliseconds.
	pub env_attack: f32,
	/// Envelope release in milliseconds.
	pub env_release: f32,
	/// Random sample-and-hold rate in hertz.
	pub random_rate: f32,
	/// Random smoothing in milliseconds.
	pub random_smooth: f32,
	/// Wavetable rate in hertz.
	pub table_rate: f32,
	/// 0 sine, 1 triangle, 2 saw, 3 square.
	pub table_shape: i32,
	/// How far the table morphs toward the next shape, 0..1.
	pub table_morph: f32,
	/// Whether the LFO rate is musical rather than hertz.
	pub lfo_sync: bool,
	/// The LFO period in beats when synced.
	pub lfo_beats: f32,
	/// Whether the random rate is musical rather than hertz.
	pub random_sync: bool,
	/// The random period in beats when synced.
	pub random_beats: f32,
	/// Whether the table rate is musical rather than hertz.
	pub table_sync: bool,
	/// The table period in beats when synced.
	pub table_beats: f32,
	/// Whether each source is bypassed.
	pub lfo_bypass: bool,
	/// Whether the envelope is bypassed.
	pub env_bypass: bool,
	/// Whether random is bypassed.
	pub random_bypass: bool,
	/// Whether the table is bypassed.
	pub table_bypass: bool,
	/// Pitch clarity gate, 0..1.
	pub pitch_clarity: f32,
	/// Pitch glide in milliseconds.
	pub pitch_glide: f32,
	/// 0 note, 1 velocity.
	pub midi_mode: i32,
	/// MIDI note transposition in semitones, -24..24.
	pub midi_transpose: i32,
	/// MIDI note fine tune in cents, -100..100.
	pub midi_fine: f32,
	/// Note input -> output curve, x/y in 0..1.
	#[serde(default = "default_midi_curve")]
	pub midi_note_curve: Vec<(f32, f32)>,
	/// Velocity input -> output curve, x/y in 0..1.
	#[serde(default = "default_midi_curve")]
	pub midi_velocity_curve: Vec<(f32, f32)>,
}

impl Default for SourceSettings {
	fn default() -> Self {
		Self {
			lfo_rate: 1.0,
			lfo_shape: 0,
			lfo_points: lfo_preset(0),
			env_attack: 10.0,
			env_release: 100.0,
			random_rate: 4.0,
			random_smooth: 80.0,
			table_rate: 1.0,
			table_shape: 0,
			table_morph: 0.0,
			lfo_sync: false,
			lfo_beats: 1.0,
			random_sync: false,
			random_beats: 1.0,
			table_sync: false,
			table_beats: 1.0,
			lfo_bypass: false,
			env_bypass: false,
			random_bypass: false,
			table_bypass: false,
			pitch_clarity: 0.5,
			pitch_glide: 40.0,
			midi_mode: 0,
			midi_transpose: 0,
			midi_fine: 0.0,
			midi_note_curve: default_midi_curve(),
			midi_velocity_curve: default_midi_curve(),
		}
	}
}

/// What a modulation source does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SourceKind {
	/// An editable low-frequency oscillator.
	#[default]
	Lfo,
	/// An envelope follower driven by the input.
	Envelope,
	/// A smoothed sample-and-hold random source.
	Random,
	/// A wavetable oscillator.
	Table,
	/// A monophonic pitch tracker on the input.
	Pitch,
	/// The last MIDI note or velocity.
	Midi,
}

impl SourceKind {
	/// The name the page shows.
	pub fn name(self) -> &'static str {
		match self {
			Self::Lfo => "LFO",
			Self::Envelope => "Envelope",
			Self::Random => "Random",
			Self::Table => "Table",
			Self::Pitch => "Pitch",
			Self::Midi => "MIDI",
		}
	}
}

/// One modulation-source slot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceSlot {
	/// Stable within a state; assignments address it by this id.
	pub id: u32,
	/// What this slot generates.
	pub kind: SourceKind,
	/// Whether the source output is forced to zero.
	pub bypass: bool,
	/// The settings for every kind; only the fields of `kind` are used.
	pub settings: SourceSettings,
}

impl SourceSlot {
	/// A slot of one kind with its default settings.
	pub fn new(id: u32, kind: SourceKind) -> Self {
		Self {
			id,
			kind,
			bypass: false,
			settings: SourceSettings::default(),
		}
	}
}

/// The four sources a fresh rack starts with.
pub fn default_source_slots() -> Vec<SourceSlot> {
	vec![
		SourceSlot::new(0, SourceKind::Lfo),
		SourceSlot::new(1, SourceKind::Envelope),
		SourceSlot::new(2, SourceKind::Random),
		SourceSlot::new(3, SourceKind::Table),
	]
}

fn default_midi_curve() -> Vec<(f32, f32)> {
	vec![(0.0, 0.0), (1.0, 1.0)]
}

fn default_lfo_points() -> Vec<(f32, f32, f32)> {
	lfo_preset(0)
}

/// The EditableLfo points for one of the four preset shapes.
pub fn lfo_preset(shape: i32) -> Vec<(f32, f32, f32)> {
	let points: &[(f32, f32)] = match shape.rem_euclid(4) {
		1 => &[(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)],
		2 => &[(0.0, 0.0), (1.0, 1.0)],
		3 => &[(0.0, 1.0), (0.5, 1.0), (0.5001, 0.0), (1.0, 0.0)],
		_ => &[(0.0, 0.5), (0.25, 1.0), (0.5, 0.5), (0.75, 0.0), (1.0, 0.5)],
	};

	points
		.iter()
		.map(|(time, value)| (*time, *value, 0.0))
		.collect()
}
