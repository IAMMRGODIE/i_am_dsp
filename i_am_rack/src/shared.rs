//! What the editor and the audio thread both touch.
//!
//! The state is behind a mutex because a page command can land between two
//! audio blocks. The audio thread only takes it when the dirty flag says the page
//! changed something, which in M0 is a rare event; the steady-state path is one
//! atomic swap and no lock.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicBool, AtomicUsize};

use i_am_dsp::prelude::{ParamMap, SetValue};
use i_am_web_gui::Control;

use crate::module::ModuleKind;
use crate::state::RackState;

/// How many edits the rack remembers, one entry per page command batch.
pub const HISTORY_LIMIT: usize = 64;

/// The undo and redo stacks.
///
/// A page command edits the whole state under one lock, so a snapshot taken
/// before the batch is an exact undo step. Redo is cleared as soon as a new
/// edit lands, the way a document editor behaves.
/// Edits to the same control inside this window are one undo step, so a knob
/// drag does not leave sixty-four one-pixel undo states behind.
const COALESCE_WINDOW: Duration = Duration::from_millis(600);

#[derive(Default)]
pub(crate) struct EditHistory {
	undo: Vec<RackState>,
	redo: Vec<RackState>,
	/// The controls the last recorded edit touched.
	last_signature: Option<String>,
	/// When that edit landed.
	last_at: Option<Instant>,
}

impl EditHistory {
	/// Remembers the state before an edit that actually changed something.
	///
	/// Consecutive edits to the same controls within COALESCE_WINDOW keep the
	/// oldest snapshot, which makes one undo return to before the drag.
	pub fn record(&mut self, before: RackState, signature: String) {
		let now = Instant::now();
		let coalesce = self.last_signature.as_deref() == Some(signature.as_str())
			&& self
				.last_at
				.is_some_and(|at| now.duration_since(at) < COALESCE_WINDOW);

		self.redo.clear();

		if !coalesce {
			self.undo.push(before);

			if self.undo.len() > HISTORY_LIMIT {
				self.undo.remove(0);
			}
		}

		self.last_signature = Some(signature);
		self.last_at = Some(now);
	}

	/// Records a wholesale replacement (a preset) as its own step.
	pub fn record_replacement(&mut self, before: RackState) {
		self.record(before, String::new());
		self.last_signature = None;
		self.last_at = None;
	}

	/// Swaps the live state with the previous one; returns false when empty.
	pub fn undo(&mut self, current: &mut RackState) -> bool {
		let Some(previous) = self.undo.pop() else {
			return false;
		};

		self.redo.push(current.clone());
		*current = previous;
		self.last_signature = None;
		self.last_at = None;
		true
	}

	/// Swaps the live state with the one undo replaced; returns false when empty.
	pub fn redo(&mut self, current: &mut RackState) -> bool {
		let Some(next) = self.redo.pop() else {
			return false;
		};

		self.undo.push(current.clone());
		*current = next;
		self.last_signature = None;
		self.last_at = None;
		true
	}

	/// Whether there is anything to undo.
	pub fn can_undo(&self) -> bool {
		!self.undo.is_empty()
	}

	/// Whether there is anything to redo.
	pub fn can_redo(&self) -> bool {
		!self.redo.is_empty()
	}

	/// Forgets every step, e.g. when the host loads another project.
	pub fn clear(&mut self) {
		self.undo.clear();
		self.redo.clear();
	}
}

pub(crate) struct RackShared {
	/// The editable truth. The engine catches up when dirty is set.
	pub state: Mutex<RackState>,
	/// Undo/redo snapshots of the state above.
	pub history: Mutex<EditHistory>,
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

	/// Forgets every step, e.g. when the host loads a project or an A/B
	/// switch makes the other side's timeline meaningless.
	pub fn clear_history(&self) {
		self.history
			.lock()
			.unwrap_or_else(|poisoned| poisoned.into_inner())
			.clear();
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
