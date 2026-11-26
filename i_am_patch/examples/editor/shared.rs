//! What the page and the audio thread share.

use super::*;

/// The watched nodes and their rings.
pub(super) type Taps = Arc<Mutex<Vec<(u32, Arc<Scope>, Arc<Scope>)>>>;

/// What the server and the demo audio thread share.
#[derive(Clone)]
pub(super) struct Shared {
	pub(super) graph: Arc<Mutex<Graph>>,
	pub(super) generation: Arc<AtomicU64>,
	pub(super) rate: usize,
	pub(super) taps: Taps,
	pub(super) phases: Arc<Mutex<Vec<(u32, f32)>>>,
	pub(super) midi_inputs: Arc<Mutex<Vec<(u32, f32)>>>,
	pub(super) error: Arc<Mutex<Option<String>>>,
	/// What the page handed the nodes; the demo graph has none.
	pub(super) slots: i_am_patch::Slots,
	/// The control edits that reach the running engine without a rebuild.
	pub(super) edits: Arc<Mutex<Vec<i_am_patch::edit::Edit>>>,
}

impl Shared {
	/// The rings for a watched node: output and input.
	pub(super) fn tap_for(&self, node: u32) -> (Arc<Scope>, Arc<Scope>) {
		let mut taps = self.taps.lock().expect("taps");

		if let Some((_, scope, input)) = taps.iter().find(|(id, _, _)| *id == node) {
			return (scope.clone(), input.clone());
		}

		let scope = Arc::new(Scope::new());
		let input = Arc::new(Scope::new());
		taps.push((node, scope.clone(), input.clone()));
		(scope, input)
	}

	pub(super) fn tap_snapshot_into(&self, out: &mut Vec<(u32, Arc<Scope>, Arc<Scope>)>) {
		out.clear();

		if let Ok(taps) = self.taps.try_lock() {
			out.extend(taps.iter().cloned());
		}
	}

	pub(super) fn set_phase(&self, node: u32, phase: f32) {
		let Ok(mut phases) = self.phases.try_lock() else {
			return;
		};

		if let Some(entry) = phases.iter_mut().find(|(id, _)| *id == node) {
			entry.1 = phase;
		} else {
			phases.push((node, phase));
		}
	}

	pub(super) fn phase_for(&self, node: u32) -> Option<f32> {
		self.phases
			.lock()
			.expect("phases")
			.iter()
			.find(|(id, _)| *id == node)
			.map(|(_, phase)| *phase)
	}

	pub(super) fn set_midi_input(&self, node: u32, position: f32) {
		let Ok(mut inputs) = self.midi_inputs.try_lock() else {
			return;
		};

		if let Some(entry) = inputs.iter_mut().find(|(id, _)| *id == node) {
			entry.1 = position;
		} else {
			inputs.push((node, position));
		}
	}

	pub(super) fn midi_input_for(&self, node: u32) -> Option<f32> {
		self.midi_inputs
			.lock()
			.expect("midi inputs")
			.iter()
			.find(|(id, _)| *id == node)
			.map(|(_, position)| *position)
	}
}
