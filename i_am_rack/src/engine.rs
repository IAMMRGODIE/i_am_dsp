//! The audio-thread half: a chain of erased effects.
//!
//! A module is a Paramed around a ModuleEffect. Paramed keeps the parameter map
//! the page writes and pushes it into the effect before every block, which is the
//! same arrangement EffectProcessor uses for a single product. The Box is the
//! only thing the rack needs to know about a module: everything else comes from
//! its Parameters implementation.

use std::sync::atomic::Ordering;

use i_am_dsp::prelude::{
	Effect, ParamMap, Parameter, Parameters, Paramed, SetValue,
};
use i_am_dsp::ProcessContext;

use crate::module::{self, ModuleKind};
use crate::state::RackState;

/// A boxed effect, adapted so Paramed can wrap it.
///
/// Paramed wants a concrete type with Parameters and Effect; a trait object is
/// neither nameable as a type parameter here nor extendable by an impl, so this
/// newtype delegates both traits to the box inside it.
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
	fn new(state: &crate::state::ModuleState, sample_rate: usize) -> Self {
		let effect = module::build(state.kind, sample_rate);

		// The map is made from the effect's own schema, then brought up to the
		// values the state file carries. Paramed pushes it back on every block.
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
}

/// Writes a numeric value into the map, converting to the type the parameter holds.
///
/// The state stores everything as f64; a mode is an integer and a bypass-like
/// flag is a boolean, and storing the wrong type would be ignored by the map.
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

/// The chain the audio thread runs.
pub(crate) struct Engine {
	modules: Vec<Module>,
	/// The rate the modules were built for.
	pub rate: usize,
}

impl Engine {
	/// Builds the chain described by a state.
	pub fn new(sample_rate: usize, state: &RackState) -> Self {
		Self {
			modules: state
				.modules
				.iter()
				.map(|module| Module::new(module, sample_rate))
				.collect(),
			rate: sample_rate,
		}
	}

	/// Copies the editable state into the running chain.
	///
	/// Called when the page changed something, not every block. Only values that
	/// differ are stored, so a block that changed nothing costs one compare.
	pub fn apply(&mut self, state: &RackState) {
		for module in &mut self.modules {
			let Some(edited) = state.modules.iter().find(|edited| edited.id == module.id)
			else {
				continue;
			};

			// A kind change would need a new effect, which M0 cannot do; the
			// state the page writes never changes one.
			if edited.kind != module.kind {
				continue;
			}

			module.bypass = edited.bypass;

			for (identifier, value) in &edited.params {
				store(&module.map, identifier, *value);
			}
		}
	}

	/// Runs every enabled module in order.
	pub fn process(
		&mut self,
		samples: &mut [f32; 2],
		other: &[[f32; 2]],
		process_context: &mut Box<dyn ProcessContext>,
	) {
		// Effect takes side-chains as slices of slices, which the processor is not
		// given; M0 has no side-chain module, so this list stays empty in practice.
		let sides: Vec<&[f32; 2]> = other.iter().collect();

		for module in &mut self.modules {
			if module.bypass {
				continue;
			}

			module
				.paramed
				.process(samples, &sides, process_context);
		}
	}

	/// The latency of the chain, in samples.
	pub fn delay(&self) -> usize {
		self.modules
			.iter()
			.map(|module| module.paramed.delay())
			.sum()
	}

	/// The kinds in the chain, in order, for a test.
	#[cfg(test)]
	pub fn kinds(&self) -> Vec<ModuleKind> {
		self.modules.iter().map(|module| module.kind).collect()
	}
}
