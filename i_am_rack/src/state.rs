//! The rack's own state, and the binary form a host saves.
//!
//! M0 keeps this deliberately plain: a version, a list of modules, and the float
//! parameters each module holds. The host never sees these as parameters; the
//! whole struct travels inside one serialized parameter.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::module::{self, ModuleKind};

/// The version this build writes.
pub const STATE_VERSION: u32 = 1;

/// Everything a rack remembers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RackState {
	/// The layout version, for a later migration.
	pub version: u32,
	/// The module chain, in processing order.
	pub modules: Vec<ModuleState>,
}

/// One module in the chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleState {
	/// Stable within a state file; the page addresses commands by it.
	pub id: u32,
	/// What the module is.
	pub kind: ModuleKind,
	/// Whether its processing is skipped.
	pub bypass: bool,
	/// Its float parameters, by identifier.
	pub params: BTreeMap<String, f32>,
}

impl ModuleState {
	/// A module with the defaults its effect declares.
	fn new(id: u32, kind: ModuleKind) -> Self {
		Self {
			id,
			kind,
			bypass: false,
			params: module::default_params(kind),
		}
	}
}

impl RackState {
	/// The chain a fresh instance starts with.
	pub fn initial() -> Self {
		Self {
			version: STATE_VERSION,
			modules: vec![
				ModuleState::new(0, ModuleKind::Filter),
				ModuleState::new(1, ModuleKind::Drive),
				ModuleState::new(2, ModuleKind::Gain),
			],
		}
	}
}
