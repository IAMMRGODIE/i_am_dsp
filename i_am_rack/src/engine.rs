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

use i_am_dsp::prelude::{
	Effect, ParamMap, Parameter, Parameters, Paramed, SetValue,
};
use i_am_dsp::{NoteEvent, ProcessContext};
use i_am_web_gui::{Control, Kind, Scope};

use crate::crossover::Crossover;
use crate::modulation::Modulation;
use crate::module::{self, ModuleKind};
use crate::shared::RackShared;
use crate::state::{Assignment, AssignmentMode, MACRO_BASE, ModuleState, RackState, SourceKind, Target};

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
}

impl Module {
	fn new(state: &ModuleState, sample_rate: usize, scope: Arc<Scope>) -> Self {
		let effect = module::build(state.kind, sample_rate);
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
		}
	}

	fn update(&mut self, state: &ModuleState) {
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

	fn get(&self, id: u32, kind: ModuleKind) -> Arc<Scope> {
		let channels = module::audio_scope(kind);

		match &self.shared {
			Some(shared) => shared.scope(id, channels),
			None => Arc::new(Scope::with_channels(channels)),
		}
	}
}

/// Copies one band's stage and slots into a running lane.
fn apply_lane(
	lane: &mut Lane,
	band: &crate::state::BandState,
	rate: usize,
	scopes: &Scopes,
) {
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
					));
				} else if let Some(module) = &mut lane.slots[slot] {
					module.update(edited);
				}
			},
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
	/// Counts blocks, so the note masks are republished at control rate.
	note_tick: usize,
}

impl Engine {
	/// Builds the bands described by a state, with scopes of its own. The
	/// plug-in builds through `with_shared`; this is for tests.
	#[cfg(test)]
	pub fn new(sample_rate: usize, state: &RackState) -> Self {
		Self::with_scopes(sample_rate, state, Scopes::default())
	}

	/// Builds the bands a plug-in runs, sharing the module scopes with its page.
	pub fn with_shared(sample_rate: usize, state: &RackState, shared: Arc<RackShared>) -> Self {
		Self::with_scopes(sample_rate, state, Scopes { shared: Some(shared) })
	}

	fn with_scopes(sample_rate: usize, state: &RackState, scopes: Scopes) -> Self {
		let mut engine = Self {
			lanes: [Lane::new(), Lane::new(), Lane::new()],
			parallel: Lane::new(),
			crossover: Crossover::new(sample_rate, state.split_low, state.split_high),
			modulation: Modulation::new(sample_rate),
			assignments: Vec::new(),
			macros: [0.0; 8],
			source_kinds: BTreeMap::new(),
			rate: sample_rate,
			scopes,
			note: -1,
			note_tick: 0,
		};

		engine.apply(state);
		engine
	}

	/// Copies the editable state into the running bands.
	pub fn apply(&mut self, state: &RackState) {
		self.crossover.set_splits(state.split_low, state.split_high);

		let mut live = BTreeSet::new();
		for band in &state.bands {
			for module in band.slots.iter().flatten() {
				live.insert(module.id);
			}
		}
		for module in state.parallel.slots.iter().flatten() {
			live.insert(module.id);
		}
		self.scopes.retain(&live);

		for (index, band) in state.bands.iter().enumerate() {
			apply_lane(&mut self.lanes[index], band, self.rate, &self.scopes);
		}

		apply_lane(&mut self.parallel, &state.parallel, self.rate, &self.scopes);

		// A lane is as late as the active modules it runs in series. Give every
		// shorter lane the difference, so a compressor with lookahead in one
		// band does not comb-filter the others.
		let mut delays = [0usize; 4];

		for (index, lane) in self.lanes.iter().enumerate() {
			delays[index] = lane_delay(lane);
		}

		delays[3] = lane_delay(&self.parallel);

		let longest = delays.iter().copied().max().unwrap_or(0);

		for (index, lane) in self.lanes.iter_mut().enumerate() {
			lane.delay = delays[index];
			lane.compensation.set_length(longest - delays[index]);
		}

		self.parallel.delay = delays[3];
		self.parallel
			.compensation
			.set_length(longest - delays[3]);

		self.modulation.set_slots(&state.source_slots);
		self.source_kinds.clear();

		for slot in &state.source_slots {
			self.source_kinds.insert(slot.id, slot.kind);
		}

		self.assignments = state.assignments.clone();
		self.macros = state.macros;
		self.apply_automation(state);
	}

	fn apply_automation(&mut self, state: &RackState) {
		for (slot, target) in state.automation.iter().enumerate() {
			let Some(target) = target else {
				continue;
			};
			let value = state.automation_values[slot].clamp(0.0, 1.0) as f64;

			match target {
				Target::Module { module, parameter } => {
					let Some(module) = self.module_mut(*module) else {
						continue;
					};
					let Some(control) = module.control(parameter).cloned() else {
						continue;
					};
					let mapped = denormalize(value, &control);
					module.set_base_target(parameter, mapped);
				},
				Target::BandGain(index) => {
					if let Some(lane) = self.lane_mut(*index) {
						lane.base_gain = value * 2.0;
					}
				},
				Target::BandPan(index) => {
					if let Some(lane) = self.lane_mut(*index) {
						lane.base_pan = value * 2.0 - 1.0;
					}
				},
			}
		}
	}

	/// Splits the frame, runs both slots of every band, and sums what is left.
	pub fn process(
		&mut self,
		samples: &mut [f32; 2],
		other: &[[f32; 2]],
		process_context: &mut Box<dyn ProcessContext>,
	) {
		let input = (samples[0] + samples[1]) * 0.5;
		// The products take the extra inputs as a slice of channel frames; the
		// compressor keys off `other[1]`, which is the rack's sidechain port.
		let sides: Vec<&[f32; 2]> = other.iter().collect();
		let sidechain = other
			.get(1)
			.map(|frame| (frame[0] + frame[1]) * 0.5)
			.unwrap_or(input);

		let events = process_context.events().to_vec();
		// The note the host is holding, so a card can show what its module hears.
		let mut note = self.note;
		for event in &events {
			match *event {
				NoteEvent::NoteOn { note: pitch, .. } => note = pitch as i32,
				NoteEvent::NoteOff { note: pitch, .. } if note == pitch as i32 => note = -1,
				NoteEvent::ImmediateStop { .. } => note = -1,
				_ => {},
			}
		}
		let note_changed = note != self.note;
		self.note = note;

		if let Some(values) = self
			.modulation
			.advance(input, process_context.infos(), &events)
		{
			let coefficient = 1.0 - (-(64.0 / (0.01 * self.rate as f32))).exp();
			self.smooth(coefficient);

			self.apply_modulation(&values);
		}

		let full = *samples;
		let bands = self.crossover.split(*samples, process_context);
		// An empty parallel lane is not a dry copy: it only joins the sum once
		// it actually holds an active module.
		let parallel_active = self
			.parallel
			.slots
			.iter()
			.flatten()
			.any(|module| !module.bypass);
		let any_solo =
			self.lanes.iter().any(|lane| lane.solo) || (parallel_active && self.parallel.solo);
		let mut output = [0.0_f32; 2];

		for (index, lane) in self.lanes.iter_mut().enumerate() {
			if lane.mute || (any_solo && !lane.solo) {
				// Keep the alignment line moving, so unmuting does not replay
				// audio from before the mute.
				lane.compensation.process([0.0, 0.0]);
				continue;
			}

			let mut band = bands[index];

			for slot in &mut lane.slots {
				let Some(module) = slot else {
					continue;
				};

				if module.bypass {
					// A skipped module still shows the signal it passes on.
					module.scope.push_stereo(band, band, sidechain);
				} else {
					let entered = band;
					// The product modules read the sidechain as `other[1]`; the
					// rack's second input port arrives in the same order.
					module.paramed.process(&mut band, &sides, process_context);
					module.scope.push_stereo(entered, band, sidechain);
				}
			}

			let gain = lane.gain as f32;
			let pan = lane.pan as f32;
			let left = gain * if pan > 0.0 { 1.0 - pan } else { 1.0 };
			let right = gain * if pan < 0.0 { 1.0 + pan } else { 1.0 };

			let frame = lane.compensation.process([band[0] * left, band[1] * right]);
			output[0] += frame[0];
			output[1] += frame[1];
		}

		if !parallel_active {
			// Nothing to add.
		} else if self.parallel.mute || (any_solo && !self.parallel.solo) {
			self.parallel.compensation.process([0.0, 0.0]);
		} else {
			let mut frame = full;

			for slot in &mut self.parallel.slots {
				let Some(module) = slot else {
					continue;
				};

				if module.bypass {
					module.scope.push_stereo(frame, frame, sidechain);
				} else {
					let entered = frame;
					module.paramed.process(&mut frame, &sides, process_context);
					module.scope.push_stereo(entered, frame, sidechain);
				}
			}

			let gain = self.parallel.gain as f32;
			let pan = self.parallel.pan as f32;
			let left = gain * if pan > 0.0 { 1.0 - pan } else { 1.0 };
			let right = gain * if pan < 0.0 { 1.0 + pan } else { 1.0 };

			let frame = self
				.parallel
				.compensation
				.process([frame[0] * left, frame[1] * right]);
			output[0] += frame[0];
			output[1] += frame[1];
		}

		*samples = output;

		self.note_tick += 1;

		if note_changed || self.note_tick % 64 == 0 {
			self.publish_notes();
		}
	}

	/// Publishes the held note, and the mask a note-following effect tunes to,
	/// to the scope its card reads. Only the modules whose card shows notes pay
	/// for it.
	fn publish_notes(&self) {
		for lane in self.lanes.iter().chain(std::iter::once(&self.parallel)) {
			for module in lane.slots.iter().flatten() {
				if !module.notes {
					continue;
				}

				module.scope.set_note(self.note);

				if let Some(mask) = module.paramed.value.allowed_notes() {
					module.scope.set_allowed_notes(mask as i32);
				}
			}
		}
	}

	/// Moves every base parameter and band control toward its target.
	fn smooth(&mut self, coefficient: f32) {
		let coefficient = coefficient as f64;
		let Self { lanes, parallel, .. } = self;

		for lane in lanes.iter_mut().chain(std::iter::once(parallel)) {
			let gain_difference = lane.base_gain - lane.smooth_gain;
			let pan_difference = lane.base_pan - lane.smooth_pan;

			lane.smooth_gain += gain_difference * coefficient;
			lane.smooth_pan += pan_difference * coefficient;

			if gain_difference.abs() < 1e-5 {
				lane.smooth_gain = lane.base_gain;
			}

			if pan_difference.abs() < 1e-5 {
				lane.smooth_pan = lane.base_pan;
			}

			lane.gain = lane.smooth_gain;
			lane.pan = lane.smooth_pan;

			for module in lane.slots.iter_mut().flatten() {
				module.smooth(coefficient as f32);
			}
		}
	}

	/// Applies the eight assignments once, at a control tick.
	fn apply_modulation(&mut self, values: &BTreeMap<u32, f32>) {
		let mut sums: BTreeMap<Target, f32> = BTreeMap::new();
		let macros = self.macros;

		for assignment in self.assignments.iter().flatten() {
			let value = if assignment.source >= MACRO_BASE {
				macros.get(assignment.source - MACRO_BASE)
			} else {
				values.get(&(assignment.source as u32))
			};
			let Some(value) = value else {
				continue;
			};
			let Some(target) = assignment.target.as_ref() else {
				continue;
			};

			let offset = match assignment.mode {
				AssignmentMode::Centered => *value * assignment.depth,
				AssignmentMode::FromBase => {
					let unipolar = assignment.source >= MACRO_BASE
						|| matches!(
							self.source_kinds.get(&(assignment.source as u32)),
							Some(SourceKind::Envelope | SourceKind::Pitch | SourceKind::Midi)
						);
					let unit = if unipolar { *value } else { (*value + 1.0) * 0.5 };
					unit * assignment.depth
				},
			};

			*sums.entry(target.clone()).or_insert(0.0) += offset;
		}

		for (target, offset) in sums {
			match target {
				Target::Module { module, parameter } => {
					let Some(module) = self.module_mut(module) else {
						continue;
					};
					let Some(control) = module.control(&parameter).cloned() else {
						continue;
					};

					if control.kind != Kind::Float {
						continue;
					}

					let base = module.current_value(&parameter).unwrap_or(control.default);
					let normalized = normalize(base, &control) + offset as f64;
					module.set_value(
						&parameter,
						denormalize(normalized.clamp(0.0, 1.0), &control),
					);
				},
				Target::BandGain(index) => {
					let Some(lane) = self.lane_mut(index) else {
						continue;
					};
					let normalized = lane.smooth_gain / 2.0 + offset as f64;
					lane.gain = normalized.clamp(0.0, 1.0) * 2.0;
				},
				Target::BandPan(index) => {
					let Some(lane) = self.lane_mut(index) else {
						continue;
					};
					let normalized = (lane.smooth_pan + 1.0) / 2.0 + offset as f64;
					lane.pan = normalized.clamp(0.0, 1.0) * 2.0 - 1.0;
				},
			}
		}
	}

	/// The lane at an index, with the parallel full-range lane after the bands.
	fn lane_mut(&mut self, index: usize) -> Option<&mut Lane> {
		match index {
			0..=2 => self.lanes.get_mut(index),
			3 => Some(&mut self.parallel),
			_ => None,
		}
	}

	/// The lane at an index, with the parallel full-range lane after the bands.
	#[cfg(test)]
	fn lane(&self, index: usize) -> Option<&Lane> {
		match index {
			0..=2 => self.lanes.get(index),
			3 => Some(&self.parallel),
			_ => None,
		}
	}

	/// The module with this id, whichever lane and slot holds it.
	fn module_mut(&mut self, id: u32) -> Option<&mut Module> {
		if let Some(module) = self
			.lanes
			.iter_mut()
			.flat_map(|lane| lane.slots.iter_mut())
			.flatten()
			.find(|module| module.id == id)
		{
			return Some(module);
		}

		self.parallel
			.slots
			.iter_mut()
			.flatten()
			.find(|module| module.id == id)
	}

	/// The live source values and LFO/table phases.
	pub fn modulation_snapshot(&self) -> (Vec<(u32, f32, f32)>, u64) {
		self.modulation.snapshot()
	}

	/// The latency of the chain, in samples.
	///
	/// This is the slowest band; the other bands are padded to match it.
	pub fn delay(&self) -> usize {
		self.lanes
			.iter()
			.map(|lane| lane.delay)
			.chain(std::iter::once(self.parallel.delay))
			.max()
			.unwrap_or(0)
	}

	/// The latency of each band, for a test.
	#[cfg(test)]
	pub fn lane_delays(&self) -> [usize; 3] {
		[self.lanes[0].delay, self.lanes[1].delay, self.lanes[2].delay]
	}

	/// The compensation delay of each band, for a test.
	#[cfg(test)]
	pub fn lane_compensations(&self) -> [usize; 3] {
		[
			self.lanes[0].compensation.len(),
			self.lanes[1].compensation.len(),
			self.lanes[2].compensation.len(),
		]
	}

	/// One module parameter as the audio thread currently holds it, for a test.
	#[cfg(test)]
	pub fn module_parameter(&self, id: u32, parameter: &str) -> Option<f32> {
		let module = self
			.lanes
			.iter()
			.chain(std::iter::once(&self.parallel))
			.flat_map(|lane| lane.slots.iter())
			.flatten()
			.find(|module| module.id == id)?;

		match module.map.get(parameter)?.load(Ordering::SeqCst) {
			SetValue::Float(value) => Some(value),
			SetValue::Int(value) => Some(value as f32),
			SetValue::Bool(value) => Some(if value { 1.0 } else { 0.0 }),
			_ => None,
		}
	}

	/// A band's modulated gain, for a test.
	#[cfg(test)]
	pub fn band_gain(&self, index: usize) -> Option<f64> {
		self.lane(index).map(|lane| lane.gain)
	}

	/// The populated slots, in band order, for a test.
	#[cfg(test)]
	pub fn kinds(&self) -> Vec<ModuleKind> {
		self.lanes
			.iter()
			.chain(std::iter::once(&self.parallel))
			.flat_map(|lane| &lane.slots)
			.flatten()
			.map(|module| module.kind)
			.collect()
	}
}

/// Maps a parameter value into 0..1, respecting a logarithmic range.
fn normalize(value: f64, control: &Control) -> f64 {
	let low = control.min;
	let high = control.max;

	if high <= low {
		return 0.0;
	}

	if control.log && low > 0.0 {
		((value.max(low).ln() - low.ln()) / (high.ln() - low.ln())).clamp(0.0, 1.0)
	} else {
		((value - low) / (high - low)).clamp(0.0, 1.0)
	}
}

/// Maps 0..1 back into a parameter's units.
fn denormalize(position: f64, control: &Control) -> f64 {
	let low = control.min;
	let high = control.max;
	let position = position.clamp(0.0, 1.0);

	if high <= low {
		return low;
	}

	if control.log && low > 0.0 {
		low * (high / low).powf(position)
	} else {
		low + (high - low) * position
	}
}

/// Writes a numeric value into the map, converting to the type the parameter holds.
fn store(map: &ParamMap, identifier: &str, value: f64) {
	let Some(atomic) = map.get(identifier) else {
		return;
	};

	if atomic.is_int() {
		atomic.store(value.round() as i32, Ordering::SeqCst);
	} else if atomic.is_bool() {
		atomic.store(value != 0.0, Ordering::SeqCst);
	} else {
		atomic.store(value as f32, Ordering::SeqCst);
	}
}
