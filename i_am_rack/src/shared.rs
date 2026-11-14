//! What the editor and the audio thread both touch.
//!
//! The state is behind a mutex because a page command can land between two
//! audio blocks. The audio thread only takes it when the dirty flag says the page
//! changed something, which in M0 is a rare event; the steady-state path is one
//! atomic swap and no lock.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use i_am_dsp::prelude::{ParamMap, SetValue};
use i_am_effects::{HistoryHandle, IrSlot};

use crate::script::ScriptSlot;
use i_am_web_gui::{Channels, Control, Scope};

use crate::module::ModuleKind;
use crate::state::RackState;

/// The most bands a module meter can hold. The vocoder's bank tops out at 128.
pub(crate) const MAX_METER_BANDS: usize = 128;

/// The live band meter of one module, for its card: the OTT's three gain
/// reductions, or a vocoder's input and output level per band.
pub(crate) struct ModuleMeters {
	bands: AtomicUsize,
	/// Whether the values are the OTT's gain reduction rather than levels.
	gains: AtomicBool,
	input: [AtomicU32; MAX_METER_BANDS],
	output: [AtomicU32; MAX_METER_BANDS],
}

impl Default for ModuleMeters {
	fn default() -> Self {
		Self {
			bands: AtomicUsize::new(0),
			gains: AtomicBool::new(false),
			input: core::array::from_fn(|_| AtomicU32::new(0)),
			output: core::array::from_fn(|_| AtomicU32::new(0)),
		}
	}
}

impl ModuleMeters {
	/// Publishes the OTT's three gain-reduction values.
	pub fn publish_gains(&self, gains: [f32; 3]) {
		self.bands.store(3, Ordering::Relaxed);
		self.gains.store(true, Ordering::Relaxed);

		for (meter, value) in self.input.iter().zip(gains) {
			meter.store(value.to_bits(), Ordering::Relaxed);
		}
	}

	/// Publishes a vocoder's band levels.
	pub fn publish_bands(&self, bands: usize, input: &[f32], output: &[f32]) {
		let bands = bands.min(MAX_METER_BANDS);
		self.bands.store(bands, Ordering::Relaxed);
		self.gains.store(false, Ordering::Relaxed);

		for band in 0..bands {
			self.input[band].store(
				input.get(band).copied().unwrap_or(0.0).to_bits(),
				Ordering::Relaxed,
			);
			self.output[band].store(
				output.get(band).copied().unwrap_or(0.0).to_bits(),
				Ordering::Relaxed,
			);
		}
	}

	/// The current meter: how many bands, whether it is gain reduction, and the
	/// two rows of values.
	pub fn snapshot(&self) -> (usize, bool, Vec<f32>, Vec<f32>) {
		let bands = self.bands.load(Ordering::Relaxed).min(MAX_METER_BANDS);
		let row = |values: &[AtomicU32]| {
			(0..bands)
				.map(|band| f32::from_bits(values[band].load(Ordering::Relaxed)))
				.collect::<Vec<_>>()
		};

		(
			bands,
			self.gains.load(Ordering::Relaxed),
			row(&self.input),
			row(&self.output),
		)
	}
}
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
	/// The identifiers of that map, in order: macros, then automation slots.
	pub param_names: Vec<String>,
	/// One scope per module id: a card shows the audio that entered and left
	/// that module, not the rack's own input and output.
	pub scopes: Mutex<HashMap<u32, Arc<Scope>>>,
	/// One meter per module id, for the cards that draw band levels.
	pub meters: Mutex<HashMap<u32, Arc<ModuleMeters>>>,
	/// One impulse response slot per module id, for the convolver's card.
	pub ir: Mutex<HashMap<u32, Arc<IrSlot>>>,
	/// One script worker per module id, for the scripted cards.
	pub scripts: Mutex<HashMap<u32, Arc<ScriptSlot>>>,
	/// One History page handle per module id, for the History card.
	pub histories: Mutex<HashMap<u32, HistoryHandle>>,
}

impl RackShared {
	/// The scope of a module, created the first time the engine or a page asks
	/// for it, with the rings that reader needs. Both sides get the same one.
	pub fn scope(&self, id: u32, channels: Channels) -> Arc<Scope> {
		let mut scopes = self
			.scopes
			.lock()
			.unwrap_or_else(|poisoned| poisoned.into_inner());

		// A page can ask before the engine has built the module, and it may ask
		// for more than the engine allocated: replace rather than widen in place.
		if scopes
			.get(&id)
			.is_some_and(|scope| scope.channels() >= channels)
		{
			return scopes[&id].clone();
		}

		let scope = Arc::new(Scope::with_channels(channels));
		scopes.insert(id, scope.clone());
		scope
	}

	/// The script worker of a module, created the first time the engine or a
	/// page asks. The kind decides which product's worker it is.
	pub fn script(&self, id: u32, kind: ModuleKind) -> Arc<ScriptSlot> {
		let mut scripts = self
			.scripts
			.lock()
			.unwrap_or_else(|poisoned| poisoned.into_inner());

		scripts
			.entry(id)
			.or_insert_with(|| {
				Arc::new(match kind {
					ModuleKind::ScriptDistortion => ScriptSlot::distortion(),
					_ => ScriptSlot::mapper(),
				})
			})
			.clone()
	}

	/// The script worker of a module, when the engine has already built one.
	pub fn existing_script(&self, id: u32) -> Option<Arc<ScriptSlot>> {
		self.scripts
			.lock()
			.unwrap_or_else(|poisoned| poisoned.into_inner())
			.get(&id)
			.cloned()
	}

	/// The History pages of a module, created the first time the engine or a
	/// page asks.
	pub fn history(&self, id: u32) -> HistoryHandle {
		self.histories
			.lock()
			.unwrap_or_else(|poisoned| poisoned.into_inner())
			.entry(id)
			.or_default()
			.clone()
	}

	/// The History pages of a module, when the engine has already built one.
	pub fn existing_history(&self, id: u32) -> Option<HistoryHandle> {
		self.histories
			.lock()
			.unwrap_or_else(|poisoned| poisoned.into_inner())
			.get(&id)
			.cloned()
	}

	/// The impulse response slot of a module, created the first time the engine
	/// or a page asks.
	pub fn ir(&self, id: u32) -> Arc<IrSlot> {
		let mut slots = self
			.ir
			.lock()
			.unwrap_or_else(|poisoned| poisoned.into_inner());

		slots
			.entry(id)
			.or_insert_with(|| Arc::new(IrSlot::new()))
			.clone()
	}

	/// The impulse response slot of a module, when the engine has already built
	/// one.
	pub fn existing_ir(&self, id: u32) -> Option<Arc<IrSlot>> {
		self.ir
			.lock()
			.unwrap_or_else(|poisoned| poisoned.into_inner())
			.get(&id)
			.cloned()
	}

	/// The meter of a module, created the first time the engine or a page asks.
	pub fn meters(&self, id: u32) -> Arc<ModuleMeters> {
		let mut meters = self
			.meters
			.lock()
			.unwrap_or_else(|poisoned| poisoned.into_inner());

		meters
			.entry(id)
			.or_insert_with(|| Arc::new(ModuleMeters::default()))
			.clone()
	}

	/// The meter of a module, when the engine has already built one.
	pub fn existing_meters(&self, id: u32) -> Option<Arc<ModuleMeters>> {
		self.meters
			.lock()
			.unwrap_or_else(|poisoned| poisoned.into_inner())
			.get(&id)
			.cloned()
	}
	/// The scope of a module, when the engine has already built one. A page that
	/// asks about a module it has not created yet gets nothing, rather than
	/// allocating rings for it.
	pub fn existing_scope(&self, id: u32) -> Option<Arc<Scope>> {
		self.scopes
			.lock()
			.unwrap_or_else(|poisoned| poisoned.into_inner())
			.get(&id)
			.cloned()
	}

	/// Drops the scopes of modules that are gone: a long session of adding and
	/// removing modules would otherwise keep every ring it ever made.
	pub fn retain_scopes(&self, live: &std::collections::BTreeSet<u32>) {
		let mut scopes = self
			.scopes
			.lock()
			.unwrap_or_else(|poisoned| poisoned.into_inner());

		scopes.retain(|id, _| live.contains(id));

		let mut meters = self
			.meters
			.lock()
			.unwrap_or_else(|poisoned| poisoned.into_inner());
		meters.retain(|id, _| live.contains(id));

		let mut slots = self
			.ir
			.lock()
			.unwrap_or_else(|poisoned| poisoned.into_inner());
		slots.retain(|id, _| live.contains(id));

		let mut scripts = self
			.scripts
			.lock()
			.unwrap_or_else(|poisoned| poisoned.into_inner());
		scripts.retain(|id, _| live.contains(id));
	}

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
		for (name, value) in self
			.param_names
			.iter()
			.zip(state.macros.iter().chain(state.automation_values.iter()))
		{
			self.set_if_changed(name, *value);
		}
	}

	/// Hands the scripts a state carries to their workers.
	///
	/// Called when a state replaces the whole rack — a preset, an undo step, an
	/// A/B switch — because the workers outlive the module and would otherwise
	/// keep running whatever was typed last.
	pub fn apply_scripts(&self, state: &RackState) {
		let rate = self.rate.load(Ordering::Relaxed) as f32;

		for module in state
			.bands
			.iter()
			.chain(std::iter::once(&state.parallel))
			.flat_map(|band| band.slots.iter().flatten())
		{
			let Some(source) = &module.script else {
				continue;
			};

			if !crate::module::needs_script(module.kind) {
				continue;
			}

			let slot = self.script(module.id, module.kind);
			slot.set_source(source.clone());

			let value = |name: &str| module.params.get(name).copied().unwrap_or(0.0) as f32;
			let extra = match module.kind {
				ModuleKind::ScriptMapper => rate,
				_ => 10.0_f32.powf(value("in_gain") / 20.0),
			};

			slot.request([value("p1"), value("p2"), value("p3"), value("p4")], extra);
		}
	}

	/// Pulls host-written macro/automation values into the state.
	///
	/// Host parameter modulation changes the map without calling
	/// `set_parameter`, so the page path and the audio path both call this
	/// before applying their own edits.
	pub fn pull_params(&self, state: &mut RackState) -> bool {
		let mut changed = false;

		for (name, value) in self.param_names.iter().zip(
			state
				.macros
				.iter_mut()
				.chain(state.automation_values.iter_mut()),
		) {
			let Some(param) = self.params.get(name) else {
				continue;
			};

			if let SetValue::Float(host) = param.load(std::sync::atomic::Ordering::Relaxed)
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
