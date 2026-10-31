//! What the editor and the audio thread both touch.
//!
//! The state is behind a mutex because a page command can land between two
//! audio blocks. The audio thread only takes it when the dirty flag says the page
//! changed something, which in M0 is a rare event; the steady-state path is one
//! atomic swap and no lock.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize};

use i_am_dsp::prelude::{ParamMap, SetValue};
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
	///
	/// Only values that actually changed are stored: `ParamMap::set` marks a
	/// parameter as changed, and the host reports every changed parameter, so
	/// writing the whole table here would spam all 40 controls at once.
	pub fn sync_params(&self, state: &RackState) {
		for (index, value) in state.macros.iter().enumerate() {
			self.set_if_changed(&format!("macro{}", index + 1), *value);
		}

		for (index, value) in state.automation_values.iter().enumerate() {
			self.set_if_changed(&format!("automation{}", index + 1), *value);
		}
	}

	/// Pulls host-written macro/automation values into the state.
	///
	/// Host parameter modulation changes the map without calling
	/// `set_parameter`, so the page path and the audio path both call this
	/// before applying their own edits.
	pub fn pull_params(&self, state: &mut RackState) -> bool {
		let mut changed = false;

		for (index, value) in state.macros.iter_mut().enumerate() {
			if let Some(param) = self.params.get(&format!("macro{}", index + 1))
				&& let SetValue::Float(host) =
					param.load(std::sync::atomic::Ordering::Relaxed)
				&& host != *value
			{
				*value = host;
				changed = true;
			}
		}

		for (index, value) in state.automation_values.iter_mut().enumerate() {
			if let Some(param) = self.params.get(&format!("automation{}", index + 1))
				&& let SetValue::Float(host) =
					param.load(std::sync::atomic::Ordering::Relaxed)
				&& host != *value
			{
				*value = host;
				changed = true;
			}
		}

		changed
	}

	fn set_if_changed(&self, identifier: &str, value: f32) {
		let Some(param) = self.params.get(identifier) else {
			return;
		};

		if let SetValue::Float(current) = param.load(std::sync::atomic::Ordering::Relaxed)
			&& current == value
		{
			return;
		}

		self.params
			.set(identifier, value, std::sync::atomic::Ordering::SeqCst);
	}
}
