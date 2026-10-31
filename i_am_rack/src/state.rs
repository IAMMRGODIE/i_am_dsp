//! The rack's own state, and the binary form a host saves.
//!
//! M2b splits the chain into three bands and keeps four modulation sources plus
//! eight assignments. Every value is stored as f64 and converted to the
//! parameter's own type by the engine; the host never sees these as parameters,
//! the whole struct travels inside one serialized parameter.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::module::{self, ModuleKind};

/// The version this build writes.
pub const STATE_VERSION: u32 = 5;

/// The default low/mid crossover.
pub const DEFAULT_SPLIT_LOW: f32 = 120.0;
/// The default mid/high crossover.
pub const DEFAULT_SPLIT_HIGH: f32 = 2_000.0;

/// Everything a rack remembers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RackState {
	/// The layout version, for a later migration.
	pub version: u32,
	/// The low/mid crossover in hertz.
	pub split_low: f32,
	/// The mid/high crossover in hertz.
	pub split_high: f32,
	/// Low, mid and high, in that order.
	pub bands: [BandState; 3],
	/// The four modulation sources: LFO, envelope, random and table.
	#[serde(default)]
	pub sources: SourceSettings,
	/// Eight modulation assignments; empty slots are off.
	#[serde(default)]
	pub assignments: [Option<Assignment>; 8],
	/// Eight macro values, 0..1; assignments can use them as sources.
	#[serde(default)]
	pub macros: [f32; 8],
}

/// The knobs of the four modulation sources.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
		}
	}
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
		_ => &[
			(0.0, 0.5),
			(0.25, 1.0),
			(0.5, 0.5),
			(0.75, 0.0),
			(1.0, 0.5),
		],
	};

	points
		.iter()
		.map(|(time, value)| (*time, *value, 0.0))
		.collect()
}

/// How an assignment lays its depth over the parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AssignmentMode {
	/// The source moves around the current value, ± depth.
	#[default]
	Centered,
	/// The current value is the start; the source moves from there to one side.
	FromBase,
}

/// One source-to-target modulation assignment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Assignment {
	/// 0 LFO, 1 envelope, 2 random, 3 table, 4..=11 macro 1..8.
	pub source: usize,
	/// What it moves; an assignment with no target is inactive.
	pub target: Option<Target>,
	/// -1..1, applied over the target's normalized range.
	pub depth: f32,
	/// Whether depth is centred on the value or starts at it.
	#[serde(default)]
	pub mode: AssignmentMode,
}

impl Default for Assignment {
	fn default() -> Self {
		Self {
			source: 0,
			target: None,
			depth: 0.5,
			mode: AssignmentMode::Centered,
		}
	}
}

/// What an assignment moves.
#[derive(
	Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum Target {
	/// One float parameter of one module.
	Module {
		/// The module id.
		module: u32,
		/// The parameter identifier inside that module.
		parameter: String,
	},
	/// A band's linear gain.
	BandGain(usize),
	/// A band's pan.
	BandPan(usize),
}

/// One band: its gain stage and the two slots in front of it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandState {
	/// Linear gain, 0..2.
	pub gain: f64,
	/// -1 left, 0 centre, 1 right.
	pub pan: f64,
	/// Silences the band.
	pub mute: bool,
	/// Solos the band; any solo silences every band that is not soloed.
	pub solo: bool,
	/// The two module slots, in processing order.
	pub slots: [Option<ModuleState>; 2],
}

/// One module in a slot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleState {
	/// Stable within a state file; the page addresses commands by it.
	pub id: u32,
	/// What the module is.
	pub kind: ModuleKind,
	/// Whether its processing is skipped.
	pub bypass: bool,
	/// Its parameters, by identifier. Floats, integers and booleans all have a
	/// numeric form here; the parameter map knows which one to store.
	pub params: BTreeMap<String, f64>,
}

impl ModuleState {
	/// A module with the defaults its effect declares.
	pub fn new(id: u32, kind: ModuleKind) -> Self {
		Self {
			id,
			kind,
			bypass: false,
			params: module::default_params(kind),
		}
	}
}

impl BandState {
	/// A band with unity gain, centred, and no modules.
	pub fn new() -> Self {
		Self {
			gain: 1.0,
			pan: 0.0,
			mute: false,
			solo: false,
			slots: [None, None],
		}
	}

	/// A band whose first slot holds a module.
	pub fn with(id: u32, kind: ModuleKind) -> Self {
		let mut band = Self::new();
		band.slots[0] = Some(ModuleState::new(id, kind));
		band
	}
}

impl Default for BandState {
	fn default() -> Self {
		Self::new()
	}
}

impl RackState {
	/// The next unused module id, so a new slot keeps every existing id stable.
	pub fn next_id(&self) -> u32 {
		self.bands
			.iter()
			.flat_map(|band| band.slots.iter().flatten())
			.map(|module| module.id)
			.max()
			.map_or(0, |id| id + 1)
	}

	/// The module with this id, whichever band and slot holds it.
	pub fn module(&self, id: u32) -> Option<&ModuleState> {
		self.bands
			.iter()
			.flat_map(|band| band.slots.iter())
			.flatten()
			.find(|module| module.id == id)
	}

	/// The module with this id, whichever band and slot holds it.
	pub fn module_mut(&mut self, id: u32) -> Option<&mut ModuleState> {
		self.bands
			.iter_mut()
			.flat_map(|band| band.slots.iter_mut())
			.flatten()
			.find(|module| module.id == id)
	}

	/// The chain a fresh instance starts with: one product in each band's first
	/// slot, the second slot empty.
	pub fn initial() -> Self {
		Self {
			version: STATE_VERSION,
			split_low: DEFAULT_SPLIT_LOW,
			split_high: DEFAULT_SPLIT_HIGH,
			bands: [
				BandState::with(0, ModuleKind::Filter),
				BandState::with(2, ModuleKind::Distortion),
				BandState::with(4, ModuleKind::Gain),
			],
			sources: SourceSettings::default(),
			assignments: std::array::from_fn(|_| None),
			macros: [0.0; 8],
		}
	}
}
