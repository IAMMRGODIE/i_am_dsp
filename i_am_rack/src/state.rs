//! The rack's own state, and the binary form a host saves.
//!
//! M1 splits the chain into three bands. Every value is stored as f64 and
//! converted to the parameter's own type by the engine; the host never sees
//! these as parameters, the whole struct travels inside one serialized
//! parameter.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::module::{self, ModuleKind};

/// The version this build writes.
pub const STATE_VERSION: u32 = 3;

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
		}
	}
}
