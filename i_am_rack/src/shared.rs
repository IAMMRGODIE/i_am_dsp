//! What the editor and the audio thread both touch.
//!
//! The state is behind a mutex because a page command can land between two
//! audio blocks. The audio thread only takes it when the dirty flag says the page
//! changed something, which in M0 is a rare event; the steady-state path is one
//! atomic swap and no lock.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize};

use i_am_dsp::prelude::ParamMap;
use i_am_web_gui::Control;

use crate::module::ModuleKind;
use crate::state::RackState;

pub(crate) struct RackShared {
	/// The editable truth. The engine catches up when dirty is set.
	pub state: Mutex<RackState>,
	/// The controls of each kind, built once from the effect metadata.
	pub schemas: BTreeMap<ModuleKind, Vec<Control>>,
	/// Set by a page command, cleared by the audio thread after applying it.
	pub dirty: AtomicBool,
	/// The rate the host announced, for the page readout.
	pub rate: AtomicUsize,
	/// The latest live source values/phases.
	pub live: Mutex<Vec<(u32, f32, f32)>>,
	/// The host-facing parameter map, kept in sync with the state.
	pub params: ParamMap,
}

impl RackShared {
	/// Locks the state, ignoring a poisoned lock: a panic in the page path must
	/// not take the audio thread's next block with it.
	pub fn lock(&self) -> std::sync::MutexGuard<'_, RackState> {
		self.state
			.lock()
			.unwrap_or_else(|poisoned| poisoned.into_inner())
	}

	/// Mirrors page-driven macro/automation edits into the host map.
	pub fn sync_params(&self, state: &RackState) {
		for (index, value) in state.macros.iter().enumerate() {
			self.params.set(
				&format!("macro{}", index + 1),
				*value,
				std::sync::atomic::Ordering::SeqCst,
			);
		}

		for (index, value) in state.automation_values.iter().enumerate() {
			self.params.set(
				&format!("automation{}", index + 1),
				*value,
				std::sync::atomic::Ordering::SeqCst,
			);
		}
	}
}
