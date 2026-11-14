//! The rack's own state, and the binary form a host saves.
//!
//! M2b splits the chain into three bands and keeps four modulation sources plus
//! eight assignments. Every value is stored as f64 and converted to the
//! parameter's own type by the engine; the host never sees these as parameters,
//! the whole struct travels inside one serialized parameter.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

pub use i_am_patch::sources::{
	SourceKind, SourceSettings, SourceSlot, default_source_slots, lfo_preset,
};

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

fn default_automation() -> [Option<Target>; AUTOMATION_SLOTS] {
	std::array::from_fn(|_| None)
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
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
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
	/// The group it belongs to, when it is grouped with a neighbour.
	#[serde(default)]
	pub group: Option<u32>,
	/// Its parameters, by identifier. Floats, integers and booleans all have a
	/// numeric form here; the parameter map knows which one to store.
	pub params: BTreeMap<String, f64>,
	/// The impulse response a convolver was loaded with, so a preset keeps its
	/// room. Absent for every other kind.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub ir: Option<i_am_effects::IrData>,
	/// The source a scripted module runs, so a preset keeps its script. Absent
	/// for every kind that does not run one.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub script: Option<String>,
	/// The History pages a History module holds, so a preset keeps them.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub history: Option<Vec<i_am_dsp::prelude::HistoryPage>>,
	/// The Filter Table pages a Filter Table module holds.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub filter_pages: Option<Vec<Vec<i_am_dsp::prelude::FilterPoint>>>,
}

impl ModuleState {
	/// A module with the defaults its effect declares.
	pub fn new(id: u32, kind: ModuleKind) -> Self {
		Self {
			id,
			kind,
			bypass: false,
			group: None,
			params: module::default_params(kind),
			ir: None,
			script: None,
			history: None,
			filter_pages: None,
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

	/// The next free group id.
	pub fn next_group_id(&self) -> u32 {
		self.bands
			.iter()
			.chain(std::iter::once(&self.parallel))
			.flat_map(|band| band.slots.iter().flatten())
			.filter_map(|module| module.group)
			.max()
			.map_or(0, |id| id + 1)
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
		let Self {
			bands, parallel, ..
		} = self;

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
			}
			None => {
				self.ab.other = Some(Box::new(current));
				self.ab.active = active;
			}
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
	let mut modules = band.slots.drain(..).flatten().map(Some).collect::<Vec<_>>();
	modules.push(None);
	band.slots = modules;

	clean_groups(band);
}

/// Drops any group that no longer has two adjacent modules in this lane.
fn clean_groups(band: &mut BandState) {
	let mut positions: BTreeMap<u32, Vec<usize>> = BTreeMap::new();

	for (index, module) in band.slots.iter().enumerate() {
		if let Some(module) = module
			&& let Some(group) = module.group
		{
			positions.entry(group).or_default().push(index);
		}
	}

	let mut broken = Vec::new();

	for (group, slots) in positions {
		let contiguous = slots
			.last()
			.zip(slots.first())
			.is_some_and(|(last, first)| *last - *first + 1 == slots.len());

		if slots.len() < 2 || !contiguous {
			broken.push(group);
		}
	}

	if broken.is_empty() {
		return;
	}

	for module in band.slots.iter_mut().flatten() {
		if module.group.is_some_and(|group| broken.contains(&group)) {
			module.group = None;
		}
	}
}
