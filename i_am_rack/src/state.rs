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
pub const STATE_VERSION: u32 = 6;

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
	/// The legacy flat source settings, kept so version 5 files still load.
	#[serde(default)]
	pub sources: SourceSettings,
	/// The live modulation-source slots.
	#[serde(default)]
	pub source_slots: Vec<SourceSlot>,
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
}

impl SourceKind {
	/// The name the page shows.
	pub fn name(self) -> &'static str {
		match self {
			Self::Lfo => "LFO",
			Self::Envelope => "Envelope",
			Self::Random => "Random",
			Self::Table => "Table",
		}
	}
}

/// One modulation-source slot.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
	/// The next free source id, 0..8, or `None` when all slots are used.
	pub fn next_source_id(&self) -> Option<u32> {
		(0..8).find(|id| {
			!self.source_slots.iter().any(|slot| slot.id == *id)
		})
	}

	/// The source slot with this id.
	pub fn source_slot(&self, id: u32) -> Option<&SourceSlot> {
		self.source_slots.iter().find(|slot| slot.id == id)
	}

	/// The source slot with this id, mutably.
	pub fn source_slot_mut(&mut self, id: u32) -> Option<&mut SourceSlot> {
		self.source_slots.iter_mut().find(|slot| slot.id == id)
	}

	/// Brings a version-5 file into the source-slot layout.
	pub fn migrate(&mut self) {
		if self.source_slots.is_empty() {
			let legacy = self.sources.clone();
			self.source_slots = default_source_slots();

			for slot in &mut self.source_slots {
				slot.settings = legacy.clone();
				slot.bypass = match slot.kind {
					SourceKind::Lfo => legacy.lfo_bypass,
					SourceKind::Envelope => legacy.env_bypass,
					SourceKind::Random => legacy.random_bypass,
					SourceKind::Table => legacy.table_bypass,
				};
			}

			for assignment in self.assignments.iter_mut().flatten() {
				if assignment.source >= 4 && assignment.source < 100 {
					assignment.source = 100 + assignment.source - 4;
				}
			}
		}

		self.source_slots.truncate(8);
		self.version = STATE_VERSION;
	}

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
			source_slots: default_source_slots(),
			assignments: std::array::from_fn(|_| None),
			macros: [0.0; 8],
		}
	}
}
