//! The audio-thread half: a three-band split, two slots per band.
//!
//! A module is a Paramed around a ModuleEffect. Paramed keeps the parameter map
//! the page writes and pushes it into the effect before every block, which is the
//! same arrangement EffectProcessor uses for a single product. The Box is the
//! only thing the rack needs to know about a module.
//!
//! The split itself is in crossover.rs. A band is muted/soloed after its
//! modules and before the sum; the band gain and pan are the last thing applied.

use std::collections::BTreeMap;
use std::sync::atomic::Ordering;

use i_am_dsp::prelude::{
	Effect, ParamMap, Parameter, Parameters, Paramed, SetValue,
};
use i_am_dsp::ProcessContext;
use i_am_web_gui::{Control, Kind};

use crate::crossover::Crossover;
use crate::modulation::Modulation;
use crate::module::{self, ModuleKind};
use crate::state::{Assignment, ModuleState, RackState, Target};

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
	/// The unmodulated values, used as the centre of every modulation.
	base: BTreeMap<String, f64>,
	/// Ranges and labels of the controls this module accepts from a page.
	controls: BTreeMap<String, Control>,
}

impl Module {
	fn new(state: &ModuleState, sample_rate: usize) -> Self {
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
			controls,
		}
	}

	fn update(&mut self, state: &ModuleState) {
		self.bypass = state.bypass;
		self.base = state.params.clone();

		for (identifier, value) in &state.params {
			store(&self.map, identifier, *value);
		}

		// A changed lookahead changes this module's delay, and the engine's
		// lane compensation reads it.
		self.paramed.sync_params();
	}

	fn base_value(&self, identifier: &str) -> Option<f64> {
		self.base.get(identifier).copied()
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
	mute: bool,
	solo: bool,
	slots: [Option<Module>; 2],
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
			mute: false,
			solo: false,
			slots: [None, None],
			delay: 0,
			compensation: DelayLine::new(0),
		}
	}
}

/// The three-band engine the audio thread runs.
pub(crate) struct Engine {
	lanes: [Lane; 3],
	crossover: Crossover,
	/// The four control-rate sources.
	modulation: Modulation,
	/// The eight assignments the page set up.
	assignments: [Option<Assignment>; 8],
	/// The eight macro values, used as extra sources.
	macros: [f32; 8],
	/// The rate the modules were built for.
	pub rate: usize,
}

impl Engine {
	/// Builds the bands described by a state.
	pub fn new(sample_rate: usize, state: &RackState) -> Self {
		let mut engine = Self {
			lanes: [Lane::new(), Lane::new(), Lane::new()],
			crossover: Crossover::new(sample_rate, state.split_low, state.split_high),
			modulation: Modulation::new(sample_rate),
			assignments: std::array::from_fn(|_| None),
			macros: [0.0; 8],
			rate: sample_rate,
		};

		engine.apply(state);
		engine
	}

	/// Copies the editable state into the running bands.
	pub fn apply(&mut self, state: &RackState) {
		self.crossover.set_splits(state.split_low, state.split_high);

		for (index, band) in state.bands.iter().enumerate() {
			let lane = &mut self.lanes[index];

			lane.base_gain = band.gain;
			lane.gain = band.gain;
			lane.base_pan = band.pan;
			lane.pan = band.pan;
			lane.mute = band.mute;
			lane.solo = band.solo;

			for (slot, edited) in band.slots.iter().enumerate() {
				match edited {
					Some(edited) => {
						let rebuild = match &lane.slots[slot] {
							Some(module) => module.kind != edited.kind || module.id != edited.id,
							None => true,
						};

						if rebuild {
							lane.slots[slot] = Some(Module::new(edited, self.rate));
						} else if let Some(module) = &mut lane.slots[slot] {
							module.update(edited);
						}
					},
					None => lane.slots[slot] = None,
				}
			}
		}

		// A band is as late as the active modules it runs in series. Give every
		// shorter band the difference, so a compressor with lookahead in one
		// band does not comb-filter the others.
		let mut delays = [0usize; 3];

		for (index, lane) in self.lanes.iter().enumerate() {
			delays[index] = lane
				.slots
				.iter()
				.flatten()
				.filter(|module| !module.bypass)
				.map(|module| module.paramed.delay())
				.sum();
		}

		let longest = delays.iter().copied().max().unwrap_or(0);

		for (index, lane) in self.lanes.iter_mut().enumerate() {
			lane.delay = delays[index];
			lane.compensation.set_length(longest - delays[index]);
		}

		self.modulation.set_settings(state.sources.clone());
		self.assignments = state.assignments.clone();
		self.macros = state.macros;
	}

	/// Splits the frame, runs both slots of every band, and sums what is left.
	pub fn process(
		&mut self,
		samples: &mut [f32; 2],
		_other: &[[f32; 2]],
		process_context: &mut Box<dyn ProcessContext>,
	) {
		let input = (samples[0] + samples[1]) * 0.5;

		if let Some(values) = self.modulation.advance(input) {
			let mut sources = [0.0; 12];
			sources[..4].copy_from_slice(&values);
			sources[4..].copy_from_slice(&self.macros);
			self.apply_modulation(sources);
		}

		let bands = self.crossover.split(*samples, process_context);
		let any_solo = self.lanes.iter().any(|lane| lane.solo);
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

				if !module.bypass {
					module.paramed.process(&mut band, &[], process_context);
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

		*samples = output;
	}

	/// Applies the eight assignments once, at a control tick.
	fn apply_modulation(&mut self, values: [f32; 12]) {
		let mut sums: BTreeMap<Target, f32> = BTreeMap::new();

		for assignment in self.assignments.iter().flatten() {
			let Some(value) = values.get(assignment.source) else {
				continue;
			};
			let Some(target) = assignment.target.as_ref() else {
				continue;
			};

			*sums.entry(target.clone()).or_insert(0.0) += value * assignment.depth;
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

					let base = module.base_value(&parameter).unwrap_or(control.default);
					let normalized = normalize(base, &control) + offset as f64;
					module.set_value(
						&parameter,
						denormalize(normalized.clamp(0.0, 1.0), &control),
					);
				},
				Target::BandGain(index) => {
					let Some(lane) = self.lanes.get_mut(index) else {
						continue;
					};
					let normalized = lane.base_gain / 2.0 + offset as f64;
					lane.gain = normalized.clamp(0.0, 1.0) * 2.0;
				},
				Target::BandPan(index) => {
					let Some(lane) = self.lanes.get_mut(index) else {
						continue;
					};
					let normalized = (lane.base_pan + 1.0) / 2.0 + offset as f64;
					lane.pan = normalized.clamp(0.0, 1.0) * 2.0 - 1.0;
				},
			}
		}
	}

	/// The module with this id, whichever band and slot holds it.
	fn module_mut(&mut self, id: u32) -> Option<&mut Module> {
		self.lanes
			.iter_mut()
			.flat_map(|lane| lane.slots.iter_mut())
			.flatten()
			.find(|module| module.id == id)
	}

	/// The latency of the chain, in samples.
	///
	/// This is the slowest band; the other bands are padded to match it.
	pub fn delay(&self) -> usize {
		self.lanes
			.iter()
			.map(|lane| lane.delay)
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
		self.lanes.get(index).map(|lane| lane.gain)
	}

	/// The populated slots, in band order, for a test.
	#[cfg(test)]
	pub fn kinds(&self) -> Vec<ModuleKind> {
		self.lanes
			.iter()
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
