//! The audio-thread half: a three-band split, two slots per band.
//!
//! A module is a Paramed around a ModuleEffect. Paramed keeps the parameter map
//! the page writes and pushes it into the effect before every block, which is the
//! same arrangement EffectProcessor uses for a single product. The Box is the
//! only thing the rack needs to know about a module.
//!
//! The split itself is in crossover.rs. A band is muted/soloed after its
//! modules and before the sum; the band gain and pan are the last thing applied.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::atomic::Ordering;

use i_am_dsp::prelude::{Effect, ParamMap, Paramed, Parameter, Parameters, SetValue};
use i_am_dsp::{NoteEvent, ProcessContext};
use i_am_web_gui::{Control, Kind, Scope};

use crate::crossover::Crossover;
use crate::modulation::Modulation;
use crate::module::{self, ModuleKind};
use crate::shared::{MAX_METER_BANDS, ModuleMeters, RackShared};
use crate::state::{
	Assignment, AssignmentMode, MACRO_BASE, ModuleState, RackState, SourceKind, Target,
};

/// A boxed effect, adapted so Paramed can wrap it.
pub(crate) struct ModuleEffect(Box<dyn Effect<2>>);

impl Parameters for ModuleEffect {
	fn get_parameters(&self) -> Vec<Parameter> {
		self.0.get_parameters()
	}

	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		self.0.set_parameter(identifier, value)
	}
}

impl ModuleEffect {
	fn allowed_notes(&self) -> Option<u16> {
		self.0.allowed_notes()
	}

	fn gain_reduction(&self) -> Option<[f32; 3]> {
		self.0.gain_reduction()
	}

	fn band_meters(&self, input: &mut [f32], output: &mut [f32]) -> usize {
		self.0.band_meters(input, output)
	}
}

impl Effect<2> for ModuleEffect {
	fn process(
		&mut self,
		samples: &mut [f32; 2],
		other: &[&[f32; 2]],
		process_context: &mut Box<dyn ProcessContext>,
	) {
		self.0.process(samples, other, process_context);
	}

	fn delay(&self) -> usize {
		self.0.delay()
	}

	fn clear_tail(&mut self) {
		self.0.clear_tail();
	}
}

/// One slot: what it is, whether it is skipped, and its parameter map.
struct Module {
	id: u32,
	kind: ModuleKind,
	bypass: bool,
	paramed: Paramed<ModuleEffect>,
	map: ParamMap,
	/// The unmodulated target values, used as the centre of every modulation.
	base: BTreeMap<String, f64>,
	/// The values the effect currently holds, before modulation.
	current: BTreeMap<String, f64>,
	/// Whether a target is still moving toward the base.
	smoothing: bool,
	/// Ranges and labels of the controls this module accepts from a page.
	controls: BTreeMap<String, Control>,
	/// What entered and left this module, for its card.
	scope: Arc<Scope>,
	/// Whether the card shows the notes this module follows.
	notes: bool,
	/// Where this module's band meter goes, when its card draws one.
	meters: Option<Arc<ModuleMeters>>,
	/// The History pages this module's card edits.
	history: Option<i_am_effects::HistoryHandle>,
	/// The Filter Table frame this module's card edits.
	filter_table: Option<i_am_effects::FilterTableHandle>,
}

impl Module {
	#[allow(clippy::too_many_arguments)]
	fn new(
		state: &ModuleState,
		sample_rate: usize,
		scope: Arc<Scope>,
		meters: Option<Arc<ModuleMeters>>,
		ir: Option<Arc<i_am_effects::IrSlot>>,
		script: Option<Arc<crate::script::ScriptSlot>>,
		history: Option<i_am_effects::HistoryHandle>,
		filter_table: Option<i_am_effects::FilterTableHandle>,
	) -> Self {
		let effect = module::build(
			state.kind,
			sample_rate,
			ir.clone(),
			script,
			history.clone(),
			filter_table.clone(),
		);

		if let (Some(saved), Some(slot)) = (&state.ir, &ir) {
			let saved = saved.resampled(sample_rate);
			slot.set_host_rate(sample_rate);

			if slot.load().map(|current| current.fingerprint) != Some(saved.fingerprint) {
				slot.set(Arc::new(saved));
			}
		}
		if let (Some(saved), Some(slot)) = (&state.history, &history) {
			slot.feed().set_pages(saved.clone());
		}

		if let (Some(saved), Some(slot)) = (&state.filter_pages, &filter_table) {
			slot.frame().set_pages(saved.clone());
		}

		let mut paramed = Paramed::new(ModuleEffect(effect));
		let map = paramed.param_map();

		for (identifier, value) in &state.params {
			store(&map, identifier, *value);
		}

		// Make delay() see the state's lookahead before the first block.
		paramed.sync_params();

		let controls = i_am_web_gui::controls(&module::schema(state.kind))
			.into_iter()
			.map(|control| (control.id.clone(), control))
			.collect();

		Self {
			id: state.id,
			kind: state.kind,
			bypass: state.bypass,
			paramed,
			map,
			base: state.params.clone(),
			current: state.params.clone(),
			smoothing: false,
			controls,
			scope,
			notes: module::shows_notes(state.kind),
			meters,
			history: history.clone(),
			filter_table: filter_table.clone(),
		}
	}

	fn update(&mut self, state: &ModuleState) {
		if let (Some(saved), Some(history)) = (&state.history, &self.history) {
			history.feed().set_pages(saved.clone());
		}

		if let (Some(saved), Some(filter_table)) = (&state.filter_pages, &self.filter_table) {
			filter_table.frame().set_pages(saved.clone());
		}

		self.bypass = state.bypass;
		self.base = state.params.clone();
		// Rewrite the map at the next control tick, which also clears the
		// previous modulation when an assignment was removed.
		self.smoothing = true;
	}

	/// Moves every unmodulated value a little toward its target.
	fn smooth(&mut self, coefficient: f32) {
		if !self.smoothing {
			return;
		}

		let mut moving = false;

		for (identifier, target) in &self.base {
			let value = self.current.entry(identifier.clone()).or_insert(*target);
			let difference = *target - *value;

			if difference.abs() < 1e-5 {
				*value = *target;
			} else {
				*value += difference * coefficient as f64;
				moving = true;
			}

			store(&self.map, identifier, *value);
		}

		self.smoothing = moving;
	}

	fn current_value(&self, identifier: &str) -> Option<f64> {
		self.current.get(identifier).copied()
	}

	fn set_base_target(&mut self, identifier: &str, value: f64) {
		self.base.insert(identifier.to_string(), value);
		self.smoothing = true;
	}

	fn control(&self, identifier: &str) -> Option<&Control> {
		self.controls.get(identifier)
	}

	fn set_value(&self, identifier: &str, value: f64) {
		store(&self.map, identifier, value);
	}
}

/// A stereo delay line that aligns one band with the slowest band.
///
/// A module with lookahead or a delay makes its band later than the others; the
/// engine gives every shorter band the difference so the sum stays phase-aligned.
struct DelayLine {
	buffer: Vec<[f32; 2]>,
	index: usize,
}

impl DelayLine {
	fn new(length: usize) -> Self {
		Self {
			buffer: vec![[0.0; 2]; length],
			index: 0,
		}
	}

	fn set_length(&mut self, length: usize) {
		if self.buffer.len() == length {
			return;
		}

		self.buffer = vec![[0.0; 2]; length];
		self.index = 0;
	}

	#[cfg(test)]
	fn len(&self) -> usize {
		self.buffer.len()
	}

	fn process(&mut self, frame: [f32; 2]) -> [f32; 2] {
		if self.buffer.is_empty() {
			return frame;
		}

		let delayed = self.buffer[self.index];
		self.buffer[self.index] = frame;
		self.index = (self.index + 1) % self.buffer.len();
		delayed
	}
}

/// One band's controls and slots.
struct Lane {
	gain: f64,
	pan: f64,
	/// Gain before modulation, 0..2.
	base_gain: f64,
	/// Pan before modulation, -1..1.
	base_pan: f64,
	/// The smoothed gain and pan the modulation is applied to.
	smooth_gain: f64,
	smooth_pan: f64,
	mute: bool,
	solo: bool,
	slots: Vec<Option<Module>>,
	/// The latency of the active modules in this lane, in samples.
	delay: usize,
	/// The delay that aligns this lane with the slowest lane.
	compensation: DelayLine,
}

impl Lane {
	fn new() -> Self {
		Self {
			gain: 1.0,
			pan: 0.0,
			base_gain: 1.0,
			base_pan: 0.0,
			smooth_gain: 1.0,
			smooth_pan: 0.0,
			mute: false,
			solo: false,
			slots: vec![None, None],
			delay: 0,
			compensation: DelayLine::new(0),
		}
	}
}

/// Where a module's scope comes from: the plug-in's shared registry, or a
/// private one when the engine is built on its own in a test.
#[derive(Clone, Default)]
pub(crate) struct Scopes {
	shared: Option<Arc<RackShared>>,
}

impl Scopes {
	fn retain(&self, live: &BTreeSet<u32>) {
		if let Some(shared) = &self.shared {
			shared.retain_scopes(live);
		}
	}

	fn script(&self, id: u32, kind: ModuleKind) -> Option<Arc<crate::script::ScriptSlot>> {
		if !module::needs_script(kind) {
			return None;
		}

		self.shared.as_ref().map(|shared| shared.script(id, kind))
	}

	fn ir(&self, id: u32, kind: ModuleKind) -> Option<Arc<i_am_effects::IrSlot>> {
		if !module::needs_ir(kind) {
			return None;
		}

		Some(match &self.shared {
			Some(shared) => shared.ir(id),
			// An engine on its own (a test) still gets a slot, just not one a
			// page can reach.
			None => Arc::new(i_am_effects::IrSlot::new()),
		})
	}

	fn filter_table(&self, id: u32, kind: ModuleKind) -> Option<i_am_effects::FilterTableHandle> {
		if kind != ModuleKind::FilterTable {
			return None;
		}

		Some(match &self.shared {
			Some(shared) => shared.filter_table(id),
			None => i_am_effects::FilterTableHandle::new(),
		})
	}

	fn history(&self, id: u32, kind: ModuleKind) -> Option<i_am_effects::HistoryHandle> {
		if kind != ModuleKind::History {
			return None;
		}

		Some(match &self.shared {
			Some(shared) => shared.history(id),
			None => i_am_effects::HistoryHandle::new(),
		})
	}

	fn meters(&self, id: u32, kind: ModuleKind) -> Option<Arc<ModuleMeters>> {
		module::meter(kind)?;

		self.shared.as_ref().map(|shared| shared.meters(id))
	}

	fn get(&self, id: u32, kind: ModuleKind) -> Arc<Scope> {
		let channels = module::audio_scope(kind);

		match &self.shared {
			Some(shared) => shared.scope(id, channels),
			None => Arc::new(Scope::with_channels(channels)),
		}
	}
}

/// Copies one band's stage and slots into a running lane.
fn apply_lane(lane: &mut Lane, band: &crate::state::BandState, rate: usize, scopes: &Scopes) {
	lane.base_gain = band.gain;
	lane.base_pan = band.pan;
	lane.mute = band.mute;
	lane.solo = band.solo;
	lane.slots.resize_with(band.slots.len(), || None);

	for (slot, edited) in band.slots.iter().enumerate() {
		match edited {
			Some(edited) => {
				let rebuild = match &lane.slots[slot] {
					Some(module) => module.kind != edited.kind || module.id != edited.id,
					None => true,
				};

				if rebuild {
					lane.slots[slot] = Some(Module::new(
						edited,
						rate,
						scopes.get(edited.id, edited.kind),
						scopes.meters(edited.id, edited.kind),
						scopes.ir(edited.id, edited.kind),
						scopes.script(edited.id, edited.kind),
						scopes.history(edited.id, edited.kind),
						scopes.filter_table(edited.id, edited.kind),
					));
				} else if let Some(module) = &mut lane.slots[slot] {
					module.update(edited);
				}
			}
			None => lane.slots[slot] = None,
		}
	}
}

/// The latency of the active modules in a lane, in samples.
fn lane_delay(lane: &Lane) -> usize {
	lane.slots
		.iter()
		.flatten()
		.filter(|module| !module.bypass)
		.map(|module| module.paramed.delay())
		.sum()
}

/// The three-band engine the audio thread runs.
pub(crate) struct Engine {
	lanes: [Lane; 3],
	/// A fourth, full-range lane summed beside the crossover bands.
	parallel: Lane,
	crossover: Crossover,
	/// The four control-rate sources.
	modulation: Modulation,
	/// The eight assignments the page set up.
	assignments: Vec<Option<Assignment>>,
	/// The eight macro values, used as extra sources.
	macros: [f32; 8],
	/// The kind of each source id, for bipolar/unipolar placement.
	source_kinds: BTreeMap<u32, SourceKind>,
	/// The rate the modules were built for.
	pub rate: usize,
	/// Where each module's card scope comes from.
	scopes: Scopes,
	/// The note the host last played, or -1.
	note: i32,
	/// Every note the host is holding, one bit per MIDI note.
	held: u128,
	/// Counts blocks, so the note masks are republished at control rate.
	note_tick: usize,
}

mod render;

use render::store;
