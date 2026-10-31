//! M2b of I Am Multipass: a three-band rack with modulation of its own.
//!
//! The audio path is a LR4 split into Low / Mid / High, each with two serial
//! module slots (Filter, Distortion, Compressor, Gain) and a band gain/pan/mute/
//! solo stage. Four control-rate sources (LFO, envelope, random, table) move
//! eight assignments across module parameters and band controls. The host sees exactly
//! one parameter, a serialized RackState; the page talks to the rack over /rack
//! and /rack-command instead of the host parameter table. That is the shape the
//! final product needs: module parameters are too dynamic to be automation
//! lanes, and a single state blob is what a project save carries.

mod crossover;
mod engine;
mod modulation;
mod module;
mod shared;
mod state;
mod web;

#[cfg(test)]
mod tests;

pub use module::ModuleKind;
pub use state::{ModuleState, RackState};

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use portable_atomic::{AtomicBool as PortableBool, AtomicF32};

use i_am_dsp::Processor;
use i_am_dsp::prelude::{
	AtomicValue, ParamMap, Parameter, Parameters, SetValue, Value, from_binary, to_binary,
};
use i_am_dsp::ProcessContext;
use i_am_plugin::processor::{DEFAULT_SAMPLE_RATE, EffectMessage};
use i_am_plugin::{
	Descriptor, Editor, GuiResizeHints, Plugin, WindowOptions,
};
use i_am_web_gui::{Faceplate, PageAssets, Scope};

use engine::Engine;
use shared::RackShared;

/// The logical size the page is laid out for.
const WIDTH: f32 = 1_000.0;
/// The logical height the page is laid out for.
const HEIGHT: f32 = 700.0;

const PAGE_HTML: &str = include_str!("page/rack.html");
const PAGE_CSS: &str = include_str!("page/rack.css");
const PAGE_JS: &str = include_str!("page/rack.js");

fn host_float(value: f32) -> AtomicValue {
	AtomicValue::Float {
		value: AtomicF32::new(value),
		range: 0.0..=1.0,
		logarithmic: false,
		changed: PortableBool::new(false),
	}
}

/// The host-facing automatable map: macro and automation controls.
fn host_param_map(state: &RackState) -> ParamMap {
	let mut entries: Vec<(String, Arc<AtomicValue>)> = Vec::new();

	for (index, value) in state.macros.iter().enumerate() {
		entries.push((
			format!("macro{}", index + 1),
			Arc::new(host_float(*value)),
		));
	}

	for (index, value) in state.automation_values.iter().enumerate() {
		entries.push((
			format!("automation{}", index + 1),
			Arc::new(host_float(*value)),
		));
	}

	ParamMap::new_with(entries)
}

/// The rack plug-in.
pub struct Rack {
	shared: Arc<RackShared>,
	engine: Engine,
	bypass: Arc<AtomicBool>,
	scope: Arc<Scope>,
	params: ParamMap,
	live_generation: u64,
}

impl Default for Rack {
	fn default() -> Self {
		let state = RackState::initial();

		let schemas = ModuleKind::ALL
			.into_iter()
			.map(|kind| (kind, i_am_web_gui::controls(&module::schema(kind))))
			.collect();

		let params = host_param_map(&state);

		let shared = Arc::new(RackShared {
			state: Mutex::new(state.clone()),
			schemas,
			dirty: AtomicBool::new(false),
			rate: AtomicUsize::new(DEFAULT_SAMPLE_RATE),
			live: Mutex::new(Vec::new()),
			params: params.clone(),
		});

		let engine = Engine::new(DEFAULT_SAMPLE_RATE, &state);

		Self {
			shared,
			engine,
			bypass: Arc::new(AtomicBool::new(false)),
			scope: Arc::new(Scope::new()),
			params,
			live_generation: 0,
		}
	}
}

impl Rack {
	/// The rate the modules are currently built for.
	pub fn sample_rate(&self) -> usize {
		self.engine.rate
	}

	/// Whether the bypass switch passes the input through.
	pub fn is_bypassed(&self) -> bool {
		self.bypass.load(Ordering::Relaxed)
	}

	/// Turns the bypass on or off from code, for a test.
	pub fn set_bypassed(&self, bypassed: bool) {
		self.bypass.store(bypassed, Ordering::Relaxed);
	}

	/// A copy of the editable state, for a test or a preview.
	pub fn state(&self) -> RackState {
		self.shared.lock().clone()
	}

	/// The JSON the page gets from /rack, for a preview or a test.
	pub fn rack_json(&self) -> String {
		web::rack_json(&self.shared)
	}
}

impl Parameters for Rack {
	fn get_parameters(&self) -> Vec<Parameter> {
		let state = self.shared.lock();
		let bytes = to_binary(&*state).unwrap_or_default();
		let mut parameters = Vec::with_capacity(self.params.len() + 1);
		parameters.push(Parameter::new("rack_state", Value::Serialized(bytes)));

		for index in 0..self.params.len() {
			let Some(identifier) = self.params.query_param_id(index) else {
				continue;
			};
			let meta = self.params.query_meta(index).unwrap_or_default();

			let Some(atomic) = self.params.get_by_index(index) else {
				continue;
			};
			let value = match atomic.load(Ordering::Relaxed) {
				SetValue::Float(value) => Value::Float {
					value,
					range: 0.0..=1.0,
					logarithmic: false,
				},
				SetValue::Int(value) => Value::Int {
					value,
					range: 0..=1,
					logarithmic: false,
				},
				SetValue::Bool(value) => Value::Bool(value),
				SetValue::Nothing => Value::Nothing,
				SetValue::Serialized(_) => Value::Nothing,
			};

			parameters.push(Parameter::new(identifier, value).with_meta(meta));
		}

		parameters
	}

	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		if identifier == "rack_state" {
			let SetValue::Serialized(bytes) = value else {
				return false;
			};

			let Ok(mut state) = from_binary::<RackState>(bytes) else {
				return false;
			};

			state.migrate();
			self.shared.sync_params(&state);
			*self.shared.lock() = state;
			self.shared.dirty.store(true, Ordering::Release);
			return true;
		}

		let SetValue::Float(value) = value else {
			return false;
		};
		let value = value.clamp(0.0, 1.0);
		let mut state = self.shared.lock();

		if let Some(index) = identifier
			.strip_prefix("macro")
			.and_then(|index| index.parse::<usize>().ok())
			.filter(|index| (1..=8).contains(index))
		{
			state.macros[index - 1] = value;
		} else if let Some(index) = identifier
			.strip_prefix("automation")
			.and_then(|index| index.parse::<usize>().ok())
			.filter(|index| (1..=crate::state::AUTOMATION_SLOTS).contains(index))
		{
			state.automation_values[index - 1] = value;
		} else {
			return false;
		}

		self.params
			.set(identifier, SetValue::Float(value), Ordering::SeqCst);
		drop(state);
		self.shared.dirty.store(true, Ordering::Release);
		true
	}
}

impl Processor for Rack {
	type Message = EffectMessage;

	fn delay(&self) -> usize {
		self.engine.delay()
	}

	fn process(
		&mut self,
		samples: &mut [f32; 2],
		other: &[[f32; 2]],
		process_context: &mut Box<dyn ProcessContext>,
	) {
		let input = *samples;
		let input_mono = (input[0] + input[1]) * 0.5;
		let sidechain = other
			.get(1)
			.map(|frame| (frame[0] + frame[1]) * 0.5)
			.unwrap_or(input_mono);

		if self.bypass.load(Ordering::Relaxed) {
			self.scope.push_stereo(input, input, sidechain);
			return;
		}

		// The page writes the state, not the module maps; a block after an edit
		// is the first one that needs the new values.
		if self.shared.dirty.swap(false, Ordering::AcqRel) {
			let state = self.shared.lock();
			self.engine.apply(&state);
		}

		self.engine.process(samples, other, process_context);

		let (live, generation) = self.engine.modulation_snapshot();

		if generation != self.live_generation {
			*self
				.shared
				.live
				.lock()
				.unwrap_or_else(|poisoned| poisoned.into_inner()) = live;
			self.live_generation = generation;
		}

		self.scope.push_stereo(input, *samples, sidechain);
	}

	fn on_message(&self, _: EffectMessage) {}
}

impl Plugin for Rack {
	const DESCRIPTOR: Descriptor = Descriptor::new("iamdsp.multipass", "I Am Multipass")
		.with_vendor("iamplugins")
		.with_version(env!("CARGO_PKG_VERSION"));

	const HAS_GUI: bool = true;

	fn new() -> Self {
		Self::default()
	}

	fn window_options() -> WindowOptions {
		// The rack is deliberately compact: the host opens it at the design
		// size instead of the larger frame the single-effect products use.
		WindowOptions::new()
			.with_size((WIDTH as f64, HEIGHT as f64))
			.with_title("I Am Multipass")
			.with_resize_hints(GuiResizeHints {
				can_resize_horizontally: true,
				can_resize_vertically: true,
				strategy: i_am_plugin::AspectRatioStrategy::Disregard,
			})
	}

	fn param_map(&self) -> ParamMap {
		self.params.clone()
	}

	fn on_activate(&mut self, sample_rate: usize) {
		if sample_rate == 0 || sample_rate == self.engine.rate {
			return;
		}

		let state = self.shared.lock().clone();
		self.engine = Engine::new(sample_rate, &state);
		self.shared.rate.store(sample_rate, Ordering::Relaxed);
	}

	fn editor(&self) -> Option<Box<dyn Editor<Self>>> {
		let shared = self.shared.clone();

		i_am_web_gui::editor_with_page(
			self,
			Faceplate::new("I Am Multipass", "#8a6aff", (WIDTH, HEIGHT)),
			PageAssets {
				html: PAGE_HTML,
				css: PAGE_CSS,
				js: PAGE_JS,
			},
			self.bypass.clone(),
			self.scope.clone(),
			self.shared.rate.load(Ordering::Relaxed),
			move |request| web::route(request, &shared),
			|_: &str| false,
		)
	}
}

i_am_plugin::export_clap!(Rack);
