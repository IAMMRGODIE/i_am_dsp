//! The audio-thread half: a three-band split, two slots per band.
//!
//! A module is a Paramed around a ModuleEffect. Paramed keeps the parameter map
//! the page writes and pushes it into the effect before every block, which is the
//! same arrangement EffectProcessor uses for a single product. The Box is the
//! only thing the rack needs to know about a module.
//!
//! The split itself is in crossover.rs. A band is muted/soloed after its
//! modules and before the sum; the band gain and pan are the last thing applied.

use std::sync::atomic::Ordering;

use i_am_dsp::prelude::{
	Effect, ParamMap, Parameter, Parameters, Paramed, SetValue,
};
use i_am_dsp::ProcessContext;

use crate::crossover::Crossover;
use crate::module::{self, ModuleKind};
use crate::state::{ModuleState, RackState};

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
}

impl Module {
	fn new(state: &ModuleState, sample_rate: usize) -> Self {
		let effect = module::build(state.kind, sample_rate);
		let paramed = Paramed::new(ModuleEffect(effect));
		let map = paramed.param_map();

		for (identifier, value) in &state.params {
			store(&map, identifier, *value);
		}

		Self {
			id: state.id,
			kind: state.kind,
			bypass: state.bypass,
			paramed,
			map,
		}
	}

	fn update(&mut self, state: &ModuleState) {
		self.bypass = state.bypass;

		for (identifier, value) in &state.params {
			store(&self.map, identifier, *value);
		}
	}
}

/// One band's controls and slots.
struct Lane {
	gain: f64,
	pan: f64,
	mute: bool,
	solo: bool,
	slots: [Option<Module>; 2],
}

impl Lane {
	fn new() -> Self {
		Self {
			gain: 1.0,
			pan: 0.0,
			mute: false,
			solo: false,
			slots: [None, None],
		}
	}
}

/// The three-band engine the audio thread runs.
pub(crate) struct Engine {
	lanes: [Lane; 3],
	crossover: Crossover,
	/// The rate the modules were built for.
	pub rate: usize,
}

impl Engine {
	/// Builds the bands described by a state.
	pub fn new(sample_rate: usize, state: &RackState) -> Self {
		let mut engine = Self {
			lanes: [Lane::new(), Lane::new(), Lane::new()],
			crossover: Crossover::new(sample_rate, state.split_low, state.split_high),
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

			lane.gain = band.gain;
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
	}

	/// Splits the frame, runs both slots of every band, and sums what is left.
	pub fn process(
		&mut self,
		samples: &mut [f32; 2],
		_other: &[[f32; 2]],
		process_context: &mut Box<dyn ProcessContext>,
	) {
		let bands = self.crossover.split(*samples, process_context);
		let any_solo = self.lanes.iter().any(|lane| lane.solo);
		let mut output = [0.0_f32; 2];

		for (index, lane) in self.lanes.iter_mut().enumerate() {
			if lane.mute || (any_solo && !lane.solo) {
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

			output[0] += band[0] * left;
			output[1] += band[1] * right;
		}

		*samples = output;
	}

	/// The latency of the chain, in samples.
	pub fn delay(&self) -> usize {
		self.lanes
			.iter()
			.flat_map(|lane| &lane.slots)
			.flatten()
			.map(|module| module.paramed.delay())
			.max()
			.unwrap_or(0)
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
