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
pub const STATE_VERSION: u32 = 9;
/// Assignment source id offset for the eight macros.
pub const MACRO_BASE: usize = 1_000_000_000;
/// How many fixed automation parameters the host exposes.
pub const AUTOMATION_SLOTS: usize = 32;

/// The default low/mid crossover.
pub const DEFAULT_SPLIT_LOW: f32 = 120.0;
/// The default mid/high crossover.
pub const DEFAULT_SPLIT_HIGH: f32 = 2_000.0;

/// Everything a rack remembers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RackState {
	/// The layout version, for a later migration.
	pub version: u32,
	/// The low/mid crossover in hertz.
	pub split_low: f32,
	/// The mid/high crossover in hertz.
	pub split_high: f32,
	/// Low, mid and high, in that order.
	pub bands: [BandState; 3],
	/// A fourth, full-range lane run beside the three crossover bands.
	#[serde(default)]
	pub parallel: BandState,
	/// The legacy flat source settings, kept so version 5 files still load.
	#[serde(default)]
	pub sources: SourceSettings,
	/// The live modulation-source slots.
	#[serde(default)]
	pub source_slots: Vec<SourceSlot>,
	/// Modulation-matrix rows; empty slots are off.
	#[serde(default)]
	pub assignments: Vec<Option<Assignment>>,
	/// Eight macro values, 0..1; assignments can use them as sources.
	#[serde(default)]
	pub macros: [f32; 8],
	/// What each macro is called; empty means the page shows "M1".."M8".
	#[serde(default = "default_macro_names")]
	pub macro_names: [String; 8],
	/// The other side of the A/B comparison.
	#[serde(default)]
	pub ab: AbState,
	/// Fixed host automation bindings.
	#[serde(default = "default_automation")]
	pub automation: [Option<Target>; AUTOMATION_SLOTS],
	/// The host values for those slots, 0..1.
	#[serde(default)]
	pub automation_values: [f32; AUTOMATION_SLOTS],
}

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

/// The A/B comparison: the live rack is one side, this holds the other.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct AbState {
	/// 0 is A, 1 is B.
	pub active: u8,
	/// The other side, always with its own comparison cleared.
	#[serde(default)]
	pub other: Option<Box<RackState>>,
}

impl AbState {
	/// The letter the page shows for the live side.
	pub fn letter(&self) -> &'static str {
		if self.active == 0 { "A" } else { "B" }
	}
}

/// The default, unnamed macro labels.
fn default_macro_names() -> [String; 8] {
	std::array::from_fn(|_| String::new())
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

fn default_automation() -> [Option<Target>; AUTOMATION_SLOTS] {
	std::array::from_fn(|_| None)
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
	Centered,
	/// The current value is the start; the source moves from there to one side.
	#[default]
	FromBase,
}

/// One source-to-target modulation assignment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
			mode: AssignmentMode::FromBase,
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BandState {
	/// Linear gain, 0..2.
	pub gain: f64,
	/// -1 left, 0 centre, 1 right.
	pub pan: f64,
	/// Silences the band.
	pub mute: bool,
	/// Solos the band; any solo silences every band that is not soloed.
	pub solo: bool,
	/// The module slots, in processing order.
	pub slots: Vec<Option<ModuleState>>,
}

/// One module in a slot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
			slots: vec![None, None],
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
		let mut id = 0;

		while self.source_slots.iter().any(|slot| slot.id == id) {
			id = id.checked_add(1)?;
		}

		Some(id)
	}

	/// The source slot with this id.
	pub fn source_slot(&self, id: u32) -> Option<&SourceSlot> {
		self.source_slots.iter().find(|slot| slot.id == id)
	}

	/// The source slot with this id, mutably.
	pub fn source_slot_mut(&mut self, id: u32) -> Option<&mut SourceSlot> {
		self.source_slots.iter_mut().find(|slot| slot.id == id)
	}

	/// Keeps every module contiguous and leaves a single empty slot at the end.
	pub fn normalize_slots(&mut self) {
		for band in &mut self.bands {
			compact_slots(band);
		}

		compact_slots(&mut self.parallel);
	}

	/// Brings a version-5 file into the source-slot layout.
	pub fn migrate(&mut self) {
		let previous = self.version;

		if previous < 6 && self.source_slots.is_empty() {
			let legacy = self.sources.clone();
			self.source_slots = default_source_slots();

			for slot in &mut self.source_slots {
				slot.settings = legacy.clone();
				slot.bypass = match slot.kind {
					SourceKind::Lfo => legacy.lfo_bypass,
					SourceKind::Envelope => legacy.env_bypass,
					SourceKind::Random => legacy.random_bypass,
					SourceKind::Table => legacy.table_bypass,
					SourceKind::Pitch | SourceKind::Midi => false,
				};
			}
		}

		if previous < 7 {
			for assignment in self.assignments.iter_mut().flatten() {
				if assignment.source >= 4 && assignment.source < 100 {
					assignment.source = MACRO_BASE + assignment.source - 4;
				} else if assignment.source >= 100 && assignment.source < MACRO_BASE {
					assignment.source = MACRO_BASE + assignment.source - 100;
				}
			}
		}

		self.ab.active = self.ab.active.min(1);

		// The other A/B side is a full state, but never a nested comparison.
		if let Some(other) = self.ab.other.as_deref_mut() {
			other.ab = AbState::default();
			other.version = STATE_VERSION;
		}

		self.normalize_slots();
		self.version = STATE_VERSION;
	}

	/// The next unused module id, so a new slot keeps every existing id stable.
	pub fn next_id(&self) -> u32 {
		self.bands
			.iter()
			.chain(std::iter::once(&self.parallel))
			.flat_map(|band| band.slots.iter().flatten())
			.map(|module| module.id)
			.max()
			.map_or(0, |id| id + 1)
	}

	/// The module with this id, whichever band and slot holds it.
	pub fn module(&self, id: u32) -> Option<&ModuleState> {
		self.bands
			.iter()
			.chain(std::iter::once(&self.parallel))
			.flat_map(|band| band.slots.iter())
			.flatten()
			.find(|module| module.id == id)
	}

	/// The module with this id, whichever band and slot holds it.
	pub fn module_mut(&mut self, id: u32) -> Option<&mut ModuleState> {
		let Self { bands, parallel, .. } = self;

		bands
			.iter_mut()
			.chain(std::iter::once(parallel))
			.flat_map(|band| band.slots.iter_mut())
			.flatten()
			.find(|module| module.id == id)
	}

	/// The chain a fresh instance starts with: one product in each band's first
	/// slot, the second slot empty.
	pub fn initial() -> Self {
		let mut state = Self {
			version: STATE_VERSION,
			split_low: DEFAULT_SPLIT_LOW,
			split_high: DEFAULT_SPLIT_HIGH,
			bands: [
				BandState::with(0, ModuleKind::Filter),
				BandState::with(2, ModuleKind::Distortion),
				BandState::with(4, ModuleKind::Gain),
			],
			parallel: BandState::new(),
			sources: SourceSettings::default(),
			source_slots: default_source_slots(),
			assignments: vec![None; 8],
			macros: [0.0; 8],
			macro_names: default_macro_names(),
			ab: AbState::default(),
			automation: default_automation(),
			automation_values: [0.0; AUTOMATION_SLOTS],
		};

		state.normalize_slots();
		state
	}

	/// Switches the live state with the other A/B side.
	///
	/// The first switch copies the live rack into the other side; later ones
	/// swap, so each side keeps its own edits. Both sides are plain states, so a
	/// project save carries the comparison with it.
	pub fn toggle_ab(&mut self) {
		let current = self.side_snapshot();
		let active = 1 - self.ab.active;

		match self.ab.other.take() {
			Some(other) => {
				let mut incoming = *other;
				incoming.ab = AbState {
					active,
					other: Some(Box::new(current)),
				};
				*self = incoming;
			},
			None => {
				self.ab.other = Some(Box::new(current));
				self.ab.active = active;
			},
		}
	}

	/// A copy of the live side with comparison metadata cleared, so the other
	/// side never nests a second comparison.
	fn side_snapshot(&self) -> Self {
		let mut copy = self.clone();
		copy.ab = AbState::default();
		copy
	}
}

/// Keeps every module contiguous and leaves a single empty slot at the end.
fn compact_slots(band: &mut BandState) {
	let mut modules = band
		.slots
		.drain(..)
		.flatten()
		.map(Some)
		.collect::<Vec<_>>();
	modules.push(None);
	band.slots = modules;
}
