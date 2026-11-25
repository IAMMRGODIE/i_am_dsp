//! The patch graph: nodes, typed ports and the edges between them.
//!
//! The model is pure data: no DSP, no page, no host. A node's kind is a stable
//! string id resolved against the catalog, so a saved patch never depends on the
//! order of an enum.

use std::collections::BTreeMap;

use i_am_dsp::prelude::{Parameter, Parameters, Value};

/// What flows through a port.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalKind {
	/// Note events.
	Midi,
	/// Sample frames.
	Audio(Channels),
	/// One value at the control rate.
	Control,
	/// A single-cycle table, read wherever a phase points.
	Wavetable,
	/// A running oscillator, played at a note's pitch.
	Osc,
}

/// How many channels an audio port carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channels {
	/// One channel.
	Mono,
	/// Left and right.
	Stereo,
}

impl Channels {
	/// The number of channels.
	pub fn count(self) -> usize {
		match self {
			Self::Mono => 1,
			Self::Stereo => 2,
		}
	}
}

/// One port on a node.
#[derive(Debug, Clone, Copy)]
pub struct PortDesc {
	/// The name the page shows.
	pub name: &'static str,
	/// What it carries.
	pub kind: SignalKind,
	/// Whether the node runs without a connection here.
	pub optional: bool,
}

/// One parameter a node exposes: its name, where it starts and how far it goes.
#[derive(Debug, Clone, Copy)]
pub struct ParamDesc {
	/// The stable name a patch stores.
	pub name: &'static str,
	/// The value a new node starts with.
	pub default: f32,
	/// The bottom of its travel.
	pub min: f32,
	/// The top of its travel.
	pub max: f32,
	/// Whether the travel is logarithmic.
	pub logarithmic: bool,
	/// The smallest step it takes; zero is continuous.
	pub step: f32,
	/// Named modes, when the value is a choice.
	pub choices: Option<&'static [&'static str]>,
	/// Whether the page draws it as a switch.
	pub boolean: bool,
}

const fn param(name: &'static str, default: f32, min: f32, max: f32) -> ParamDesc {
	ParamDesc {
		name,
		default,
		min,
		max,
		logarithmic: false,
		step: 0.0,
		choices: None,
		boolean: false,
	}
}

const fn log_param(name: &'static str, default: f32, min: f32, max: f32) -> ParamDesc {
	ParamDesc {
		name,
		default,
		min,
		max,
		logarithmic: true,
		step: 0.0,
		choices: None,
		boolean: false,
	}
}

const fn step_param(name: &'static str, default: f32, min: f32, max: f32, step: f32) -> ParamDesc {
	ParamDesc {
		name,
		default,
		min,
		max,
		logarithmic: false,
		step,
		choices: None,
		boolean: false,
	}
}

const fn bool_param(name: &'static str, default: f32) -> ParamDesc {
	ParamDesc {
		name,
		default,
		min: 0.0,
		max: 1.0,
		logarithmic: false,
		step: 1.0,
		choices: None,
		boolean: true,
	}
}

const fn choice_param(
	name: &'static str,
	default: f32,
	max: f32,
	choices: &'static [&'static str],
) -> ParamDesc {
	ParamDesc {
		name,
		default,
		min: 0.0,
		max,
		logarithmic: false,
		step: 1.0,
		choices: Some(choices),
		boolean: false,
	}
}

/// The LFO and table shapes, in the order the sources play them.
const SHAPES: &[&str] = &["Sine", "Triangle", "Saw", "Square"];
/// The shapes the oscillator and table nodes play.
const CONSTRUCTOR_SHAPES: &[&str] = &["Sine", "Triangle", "Saw", "Square", "Noise"];
/// What the MIDI source follows.
const MIDI_MODES: &[&str] = &["Note", "Velocity"];

/// Whether a kind builds a table or an oscillator instead of running audio.
///
/// A constructor has no side effect of its own, so the voice group counts it as
/// part of the voice it feeds rather than as a node between the boundaries.
pub fn is_constructor_kind(kind: &str) -> bool {
	matches!(
		kind,
		"table:shape"
			| "table:canvas"
			| "table:shapes"
			| "table:additive"
			| "table:spectrum"
			| "table:script"
			| "osc:shape"
			| "osc:am"
			| "osc:rm"
			| "osc:pm"
			| "osc:fm"
			| "warp:bend"
			| "warp:mirror"
			| "warp:sync"
			| "warp:quantize"
			| "warp:remap"
	)
}

/// One node kind.
#[derive(Debug, Clone)]
pub struct NodeDesc {
	/// The stable id a patch stores.
	pub id: &'static str,
	/// The name the page shows.
	pub name: &'static str,
	/// The input ports, in order.
	pub inputs: &'static [PortDesc],
	/// The output ports, in order.
	pub outputs: &'static [PortDesc],
	/// The latency the node adds, in frames.
	pub latency: usize,
	/// The parameters the node exposes.
	pub params: &'static [ParamDesc],
}

impl ParamDesc {
	/// Where a value sits on this parameter's own curve, 0..1.
	///
	/// A logarithmic parameter takes the log of the ratio, so the same step is
	/// the same musical interval rather than the same number of hertz.
	pub fn normalize(&self, value: f32) -> f32 {
		if self.max <= self.min {
			return 0.0;
		}

		if self.logarithmic && self.min > 0.0 {
			((value.max(self.min).ln() - self.min.ln()) / (self.max.ln() - self.min.ln()))
				.clamp(0.0, 1.0)
		} else {
			((value - self.min) / (self.max - self.min)).clamp(0.0, 1.0)
		}
	}

	/// Maps 0..1 back into the parameter's units.
	pub fn denormalize(&self, position: f32) -> f32 {
		let position = position.clamp(0.0, 1.0);

		if self.max <= self.min {
			return self.min;
		}

		if self.logarithmic && self.min > 0.0 {
			self.min * (self.max / self.min).powf(position)
		} else {
			self.min + (self.max - self.min) * position
		}
	}
}

impl NodeDesc {
	/// The index of a named input port.
	pub fn input(&self, name: &str) -> Option<usize> {
		self.inputs.iter().position(|port| port.name == name)
	}

	/// The index of a named output port.
	pub fn output(&self, name: &str) -> Option<usize> {
		self.outputs.iter().position(|port| port.name == name)
	}
}

const AUDIO: SignalKind = SignalKind::Audio(Channels::Stereo);

const fn port(name: &'static str, kind: SignalKind) -> PortDesc {
	PortDesc {
		name,
		kind,
		optional: false,
	}
}

const fn optional(name: &'static str, kind: SignalKind) -> PortDesc {
	PortDesc {
		name,
		kind,
		optional: true,
	}
}

/// The node kinds the patcher ships itself.
const BUILTINS: &[NodeDesc] = &[
	NodeDesc {
		id: "input:midi",
		name: "MIDI In",
		inputs: &[],
		outputs: &[port("midi", SignalKind::Midi)],
		latency: 0,
		params: &[],
	},
	NodeDesc {
		id: "input:audio",
		name: "Audio In",
		inputs: &[],
		outputs: &[port("audio", AUDIO)],
		latency: 0,
		params: &[],
	},
	NodeDesc {
		id: "input:sidechain",
		name: "Sidechain In",
		inputs: &[],
		outputs: &[optional("audio", AUDIO)],
		latency: 0,
		params: &[],
	},
	NodeDesc {
		id: "table:shape",
		name: "Table",
		inputs: &[],
		outputs: &[port("wave", SignalKind::Wavetable)],
		latency: 0,
		params: &[choice_param("shape", 0.0, 4.0, CONSTRUCTOR_SHAPES)],
	},
	NodeDesc {
		id: "osc:shape",
		name: "Oscillator",
		inputs: &[],
		outputs: &[port("osc", SignalKind::Osc)],
		latency: 0,
		params: &[choice_param("shape", 0.0, 4.0, CONSTRUCTOR_SHAPES)],
	},
	NodeDesc {
		id: "osc:am",
		name: "AM",
		inputs: &[
			port("carrier", SignalKind::Wavetable),
			port("modulator", SignalKind::Wavetable),
		],
		outputs: &[port("osc", SignalKind::Osc)],
		latency: 0,
		params: &[
			param("depth", 0.5, 0.0, 1.0),
			log_param("ratio", 1.0, 0.0625, 16.0),
		],
	},
	NodeDesc {
		id: "osc:rm",
		name: "RM",
		inputs: &[
			port("carrier", SignalKind::Wavetable),
			port("modulator", SignalKind::Wavetable),
		],
		outputs: &[port("osc", SignalKind::Osc)],
		latency: 0,
		params: &[
			param("depth", 0.5, 0.0, 1.0),
			log_param("ratio", 1.0, 0.0625, 16.0),
		],
	},
	NodeDesc {
		id: "osc:pm",
		name: "PM",
		inputs: &[
			port("carrier", SignalKind::Wavetable),
			port("modulator", SignalKind::Wavetable),
		],
		outputs: &[port("osc", SignalKind::Osc)],
		latency: 0,
		params: &[
			param("depth", 1.0, 0.0, 8.0),
			log_param("ratio", 1.0, 0.0625, 16.0),
		],
	},
	NodeDesc {
		id: "osc:fm",
		name: "FM",
		inputs: &[
			port("carrier", SignalKind::Wavetable),
			port("modulator", SignalKind::Wavetable),
		],
		outputs: &[port("osc", SignalKind::Osc)],
		latency: 0,
		params: &[
			param("depth", 2.0, 0.0, 16.0),
			log_param("ratio", 1.0, 0.0625, 16.0),
		],
	},
	NodeDesc {
		id: "warp:bend",
		name: "Bend",
		inputs: &[port("wave", SignalKind::Wavetable)],
		outputs: &[port("wave", SignalKind::Wavetable)],
		latency: 0,
		params: &[param("power", 1.0, 0.25, 4.0)],
	},
	NodeDesc {
		id: "warp:mirror",
		name: "Mirror",
		inputs: &[port("wave", SignalKind::Wavetable)],
		outputs: &[port("wave", SignalKind::Wavetable)],
		latency: 0,
		params: &[param("amount", 0.0, 0.0, 1.0)],
	},
	NodeDesc {
		id: "warp:sync",
		name: "Sync",
		inputs: &[port("wave", SignalKind::Wavetable)],
		outputs: &[port("wave", SignalKind::Wavetable)],
		latency: 0,
		params: &[log_param("ratio", 1.0, 1.0, 8.0)],
	},
	NodeDesc {
		id: "warp:quantize",
		name: "Quantize",
		inputs: &[port("wave", SignalKind::Wavetable)],
		outputs: &[port("wave", SignalKind::Wavetable)],
		latency: 0,
		params: &[step_param("steps", 1.0, 1.0, 64.0, 1.0)],
	},
	NodeDesc {
		id: "warp:remap",
		name: "Remap",
		inputs: &[
			port("wave", SignalKind::Wavetable),
			// Without a curve the remap reads the table straight through.
			optional("curve", SignalKind::Wavetable),
		],
		outputs: &[port("wave", SignalKind::Wavetable)],
		latency: 0,
		params: &[param("amount", 0.0, 0.0, 1.0)],
	},
	NodeDesc {
		id: "voice:in",
		name: "Voice In",
		inputs: &[port("midi", SignalKind::Midi)],
		outputs: &[port("midi", SignalKind::Midi)],
		latency: 0,
		params: &[step_param("voices", 8.0, 1.0, 8.0, 1.0)],
	},
	NodeDesc {
		id: "voice:out",
		name: "Voice Out",
		inputs: &[port("audio", AUDIO)],
		outputs: &[port("audio", AUDIO)],
		latency: 0,
		params: &[],
	},
	NodeDesc {
		id: "out:audio",
		name: "Audio Out",
		inputs: &[port("audio", AUDIO)],
		outputs: &[],
		latency: 0,
		params: &[param("gain", 1.0, 0.0, 2.0)],
	},
	NodeDesc {
		id: "split:ms",
		name: "M/S Split",
		inputs: &[port("audio", AUDIO)],
		outputs: &[
			port("mid", SignalKind::Audio(Channels::Mono)),
			port("side", SignalKind::Audio(Channels::Mono)),
		],
		latency: 0,
		params: &[],
	},
	NodeDesc {
		id: "split:lr",
		name: "L/R Split",
		inputs: &[port("audio", AUDIO)],
		outputs: &[port("left", AUDIO), port("right", AUDIO)],
		latency: 0,
		params: &[],
	},
	NodeDesc {
		id: "split:bands",
		name: "Band Split",
		inputs: &[port("audio", AUDIO)],
		outputs: &[port("low", AUDIO), port("mid", AUDIO), port("high", AUDIO)],
		latency: 0,
		params: &[
			log_param("low", 120.0, 20.0, 1_000.0),
			log_param("high", 2_000.0, 200.0, 20_000.0),
		],
	},
	NodeDesc {
		id: "mixer",
		name: "Mixer",
		inputs: &[port("a", AUDIO), optional("b", AUDIO)],
		outputs: &[port("audio", AUDIO)],
		latency: 0,
		params: &[param("a", 1.0, 0.0, 2.0), param("b", 1.0, 0.0, 2.0)],
	},
	NodeDesc {
		id: "mix:ms",
		name: "M/S Mix",
		inputs: &[port("mid", AUDIO), port("side", AUDIO)],
		outputs: &[port("audio", AUDIO)],
		latency: 0,
		params: &[],
	},
	NodeDesc {
		id: "mix:lr",
		name: "L/R Mix",
		inputs: &[port("left", AUDIO), port("right", AUDIO)],
		outputs: &[port("audio", AUDIO)],
		latency: 0,
		params: &[],
	},
	NodeDesc {
		id: "mix:bands",
		name: "Band Mix",
		inputs: &[port("low", AUDIO), port("mid", AUDIO), port("high", AUDIO)],
		outputs: &[port("audio", AUDIO)],
		latency: 0,
		params: &[
			param("low", 1.0, 0.0, 2.0),
			param("mid", 1.0, 0.0, 2.0),
			param("high", 1.0, 0.0, 2.0),
		],
	},
	NodeDesc {
		id: "lfo",
		name: "LFO",
		inputs: &[],
		outputs: &[port("control", SignalKind::Control)],
		latency: 0,
		params: &[
			log_param("rate", 1.0, 0.01, 20.0),
			choice_param("shape", 0.0, 3.0, SHAPES),
			bool_param("sync", 0.0),
			log_param("beats", 1.0, 0.03125, 32.0),
			bool_param("one_shot", 0.0),
		],
	},
	NodeDesc {
		id: "macro",
		name: "Macro",
		// No input: the value comes from the host's parameter table, and the port
		// is what a modulation link reads.
		inputs: &[],
		outputs: &[port("control", SignalKind::Control)],
		latency: 0,
		params: &[step_param("index", 1.0, 1.0, 64.0, 1.0)],
	},
	NodeDesc {
		id: "random",
		name: "Random",
		inputs: &[],
		outputs: &[port("control", SignalKind::Control)],
		latency: 0,
		params: &[
			log_param("rate", 4.0, 0.01, 20.0),
			log_param("smooth", 20.0, 0.1, 500.0),
			bool_param("sync", 0.0),
			log_param("beats", 1.0, 0.03125, 32.0),
		],
	},
	NodeDesc {
		id: "table",
		name: "Table",
		inputs: &[],
		outputs: &[port("control", SignalKind::Control)],
		latency: 0,
		params: &[
			log_param("rate", 1.0, 0.01, 20.0),
			choice_param("shape", 0.0, 3.0, SHAPES),
			param("morph", 0.0, 0.0, 1.0),
			bool_param("sync", 0.0),
			log_param("beats", 1.0, 0.03125, 32.0),
		],
	},
	NodeDesc {
		id: "pitch",
		name: "Pitch",
		inputs: &[port("audio", AUDIO)],
		outputs: &[port("control", SignalKind::Control)],
		latency: 0,
		params: &[
			param("clarity", 0.5, 0.0, 1.0),
			log_param("glide", 20.0, 0.1, 1000.0),
		],
	},
	NodeDesc {
		id: "midi",
		name: "MIDI",
		// The events come from the host through the graph; the port is what the
		// page draws, and says the node needs a MIDI source.
		inputs: &[port("midi", SignalKind::Midi)],
		outputs: &[port("control", SignalKind::Control)],
		latency: 0,
		params: &[
			choice_param("mode", 0.0, 1.0, MIDI_MODES),
			step_param("transpose", 0.0, -24.0, 24.0, 1.0),
			param("fine", 0.0, -100.0, 100.0),
		],
	},
	NodeDesc {
		id: "map",
		name: "Map",
		// The one node whose input is a control rather than audio, and whose
		// output is a curve of it: what makes it a mapping instead of a
		// modulation, which just adds to a parameter.
		inputs: &[port("control", SignalKind::Control)],
		outputs: &[port("control", SignalKind::Control)],
		latency: 0,
		params: &[],
	},
	NodeDesc {
		id: "envelope",
		name: "Envelope",
		inputs: &[port("audio", AUDIO)],
		outputs: &[port("control", SignalKind::Control)],
		latency: 0,
		params: &[
			log_param("attack", 10.0, 1.0, 500.0),
			log_param("release", 100.0, 1.0, 500.0),
		],
	},
	NodeDesc {
		id: "delay",
		name: "Lookahead",
		inputs: &[port("audio", AUDIO)],
		outputs: &[port("audio", AUDIO)],
		latency: 64,
		params: &[
			param("time", 0.25, 1.0, 1000.0),
			param("feedback", 0.3, 0.0, 0.95),
		],
	},
	// A note on the canvas: no ports, no parameters, nothing to play. It exists so
	// a patch can explain itself where it is, and its text rides in the patch text
	// beside the graph rather than in a slot, because it is a person's writing.
	NodeDesc {
		id: "text",
		name: "Text",
		inputs: &[],
		outputs: &[],
		latency: 0,
		params: &[],
	},
];

/// Every node kind: the built-ins and one `mod:<id>` per product module.
pub fn catalog() -> &'static [NodeDesc] {
	static CATALOG: std::sync::OnceLock<Vec<NodeDesc>> = std::sync::OnceLock::new();

	CATALOG.get_or_init(|| {
		let mut kinds = BUILTINS.to_vec();
		kinds.extend(module_catalog());
		kinds.extend(table_catalog());
		kinds.push(adsr_catalog());
		kinds.extend(synth_catalog());
		kinds
	})
}

/// The audio ports every product module gets.
const MODULE_INPUTS: &[PortDesc] = &[PortDesc {
	name: "audio",
	kind: AUDIO,
	optional: false,
}];
const MODULE_OUTPUTS: &[PortDesc] = &[PortDesc {
	name: "audio",
	kind: AUDIO,
	optional: false,
}];

/// One kind per module in the effect catalog, with the module's own schema.
fn module_catalog() -> Vec<NodeDesc> {
	let mut kinds = Vec::new();

	for module in i_am_effects::catalog::MODULES {
		let id: &'static str = Box::leak(format!("mod:{}", module.id).into_boxed_str());
		let latency = (module.build)(44_100).delay();
		let mut params: Vec<ParamDesc> = Vec::new();
		let mut sidechain = false;

		for parameter in module.parameters() {
			if parameter.identifier == "sidechain" {
				sidechain = true;
			}

			let name: &'static str = Box::leak(parameter.identifier.clone().into_boxed_str());

			if let Some(desc) = parameter_desc(name, &parameter) {
				params.push(desc);
			}
		}

		// The compressor is the one module that can read a second audio input;
		// it declares the port, the page draws it, and the engine routes it.
		let inputs: &'static [PortDesc] = if sidechain {
			Box::leak(vec![port("audio", AUDIO), optional("sidechain", AUDIO)].into_boxed_slice())
		} else {
			MODULE_INPUTS
		};

		kinds.push(NodeDesc {
			id,
			name: module.name,
			inputs,
			outputs: MODULE_OUTPUTS,
			latency,
			params: Box::leak(params.into_boxed_slice()),
		});
	}

	kinds
}

/// One kind per synth product, with a notes input and the product's schema.
/// The envelope node's controls: the products' own ADSR, envelope and voice.
pub fn adsr_parameters() -> Vec<Parameter> {
	use i_am_dsp::prelude::{Adsr, EqualTemperament, SineWave, TableOsc};

	Adsr::<TableOsc<SineWave>, EqualTemperament, 2>::new(
		TableOsc(SineWave),
		EqualTemperament,
		48_000,
	)
	.get_parameters()
}

/// The envelope node: the products' ADSR controls, with the graph's ports.
fn adsr_catalog() -> NodeDesc {
	let mut params: Vec<ParamDesc> = Vec::new();

	for parameter in adsr_parameters() {
		let name: &'static str = Box::leak(parameter.identifier.clone().into_boxed_str());

		if let Some(desc) = parameter_desc(name, &parameter) {
			params.push(desc);
		}
	}

	NodeDesc {
		id: "adsr",
		name: "ADSR",
		inputs: Box::leak(
			vec![port("midi", SignalKind::Midi), port("osc", SignalKind::Osc)].into_boxed_slice(),
		),
		outputs: Box::leak(vec![port("audio", AUDIO)].into_boxed_slice()),
		latency: 0,
		params: Box::leak(params.into_boxed_slice()),
	}
}

/// The product tables, and the parameters their product pages edit.
const TABLE_PRODUCTS: &[(&str, &str)] = &[
	("shapes", "Shapes Table"),
	("additive", "Additive Table"),
	("spectrum", "Spectrum Table"),
	("script", "Script Table"),
	("canvas", "Canvas Table"),
];

/// A table node: the product's own parameters, on the graph's ports.
///
/// The output stage (gain, pitch, pan) stays on the envelope node: a table is
/// read by it, not played by itself.
fn table_catalog() -> Vec<NodeDesc> {
	let mut kinds = Vec::new();

	for (product, name) in TABLE_PRODUCTS {
		let Some(descriptor) = crate::synths::synth(product) else {
			continue;
		};

		let id: &'static str = Box::leak(format!("table:{product}").into_boxed_str());
		let mut params: Vec<ParamDesc> = Vec::new();

		for parameter in (descriptor.parameters)() {
			if matches!(
				parameter.identifier.as_str(),
				"gain" | "pitch" | "pitch_factor" | "pan"
			) {
				continue;
			}

			let param_name: &'static str = Box::leak(parameter.identifier.clone().into_boxed_str());

			if let Some(desc) = parameter_desc(param_name, &parameter) {
				params.push(desc);
			}
		}

		kinds.push(NodeDesc {
			id,
			name,
			inputs: &[],
			outputs: Box::leak(vec![port("wave", SignalKind::Wavetable)].into_boxed_slice()),
			latency: 0,
			params: Box::leak(params.into_boxed_slice()),
		});
	}

	kinds
}

fn synth_catalog() -> Vec<NodeDesc> {
	let mut kinds = Vec::new();

	for descriptor in crate::synths::ALL {
		let id: &'static str = Box::leak(descriptor.kind().into_boxed_str());
		let mut params: Vec<ParamDesc> = Vec::new();

		for parameter in (descriptor.parameters)() {
			let name: &'static str = Box::leak(parameter.identifier.clone().into_boxed_str());

			if let Some(desc) = parameter_desc(name, &parameter) {
				params.push(desc);
			}
		}

		kinds.push(NodeDesc {
			id,
			name: descriptor.name,
			inputs: Box::leak(vec![port("notes", SignalKind::Midi)].into_boxed_slice()),
			outputs: Box::leak(vec![port("audio", AUDIO)].into_boxed_slice()),
			latency: 0,
			params: Box::leak(params.into_boxed_slice()),
		});
	}

	kinds
}

/// One catalog parameter, built from the product's own declaration.
fn parameter_desc(name: &'static str, parameter: &Parameter) -> Option<ParamDesc> {
	match &parameter.value {
		Value::Float {
			value,
			range,
			logarithmic,
		} => Some(ParamDesc {
			name,
			default: *value,
			min: *range.start(),
			max: *range.end(),
			logarithmic: *logarithmic,
			step: 0.0,
			choices: parameter.meta.choices,
			boolean: false,
		}),
		Value::Int {
			value,
			range,
			logarithmic,
		} => Some(ParamDesc {
			name,
			default: *value as f32,
			min: *range.start() as f32,
			max: *range.end() as f32,
			logarithmic: *logarithmic,
			step: 1.0,
			choices: parameter.meta.choices,
			boolean: false,
		}),
		Value::Bool(value) => Some(ParamDesc {
			name,
			default: if *value { 1.0 } else { 0.0 },
			min: 0.0,
			max: 1.0,
			logarithmic: false,
			step: 1.0,
			choices: None,
			boolean: true,
		}),
		_ => None,
	}
}

/// The descriptor for a kind id.
pub fn node(kind: &str) -> Option<&'static NodeDesc> {
	catalog().iter().find(|desc| desc.id == kind)
}

/// A node's identity inside one graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(pub u32);

/// One point of an editable LFO: time and value in 0..1, and the bend between
/// this point and the next.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Point {
	/// Where it sits in the cycle.
	pub time: f32,
	/// Its value.
	pub value: f32,
	/// How the line bends toward the next point.
	pub bend: f32,
}

impl Point {
	/// One curve's value at `x`, for the nodes that map a control through
	/// points.
	///
	/// A point's `bend` shapes the line to the point after it, which is what the
	/// page draws and drags; outside the first and last point the curve holds
	/// its end value, and a curve with fewer than two points is the identity.
	pub fn curve(points: &[Point], x: f32) -> f32 {
		let x = x.clamp(0.0, 1.0);
		let Some((first, rest)) = points.split_first() else {
			return x;
		};
		if rest.is_empty() || x <= first.time {
			return first.value.clamp(0.0, 1.0);
		}

		for pair in points.windows(2) {
			let (from, to) = (&pair[0], &pair[1]);
			if x > to.time {
				continue;
			}

			let span = to.time - from.time;
			if span <= 1e-6 {
				return to.value.clamp(0.0, 1.0);
			}

			let unit = (x - from.time) / span;
			let shaped = bend(unit, from.bend);
			return (from.value + (to.value - from.value) * shaped).clamp(0.0, 1.0);
		}

		points[points.len() - 1].value.clamp(0.0, 1.0)
	}
}

/// The exponential shape one segment bends by.
///
/// Zero leaves the segment straight; a positive bend ends it steeply and a
/// negative one gently, which is the same family the page's editor draws.
fn bend(t: f32, factor: f32) -> f32 {
	if factor.abs() < 1e-4 {
		return t;
	}

	((t * factor).exp() - 1.0) / (factor.exp() - 1.0)
}

/// One node placed in a graph.
#[derive(Debug, Clone)]
pub struct NodeInstance {
	/// Its id within the graph.
	pub id: NodeId,
	/// Its kind.
	pub kind: &'static NodeDesc,
	/// Where the page draws it.
	pub position: (f32, f32),
	/// The node's controls, keyed by the catalog's names.
	pub params: BTreeMap<&'static str, f32>,
	/// The points of an editable LFO or a MIDI note mapping.
	pub points: Vec<Point>,
	/// The points of a MIDI velocity mapping.
	pub velocity_points: Vec<Point>,
	/// The note a text node carries: page text, not part of the signal.
	///
	/// It lives in the graph so it saves with the patch and undoes with it, but
	/// the string never reaches the engine: a rebuild builds the node and plays
	/// it as silence.
	pub text: Option<String>,
}

impl NodeInstance {
	/// One parameter's value.
	pub fn param(&self, name: &str) -> f32 {
		self.params.get(name).copied().unwrap_or(0.0)
	}
}

/// One connection: an output port into an input port.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Edge {
	/// The source node and output port.
	pub from: (NodeId, usize),
	/// The destination node and input port.
	pub to: (NodeId, usize),
}

/// How a source moves the parameter it is assigned to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModulationMode {
	/// Around the base value, ± the depth.
	Centered,
	/// From the base value upward, 0..depth.
	#[default]
	FromBase,
}

/// One control-to-parameter assignment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModulationLink {
	/// The source node and its control output port.
	pub source: (NodeId, usize),
	/// The destination node and the parameter name.
	pub target: (NodeId, &'static str),
	/// How far the control moves the parameter, in parameter units.
	pub depth: f32,
	/// Where the movement starts.
	pub mode: ModulationMode,
}

/// A patch: nodes and the edges between them.
#[derive(Debug, Default, Clone)]
pub struct Graph {
	/// The nodes, in insertion order.
	pub nodes: Vec<NodeInstance>,
	/// The edges.
	pub edges: Vec<Edge>,
	/// The control-to-parameter assignments.
	pub modulations: Vec<ModulationLink>,
}

impl Graph {
	/// An empty patch.
	pub fn new() -> Self {
		Self::default()
	}

	/// Adds a node of the given kind, with the next free id.
	pub fn add_node(&mut self, kind: &str, position: (f32, f32)) -> Option<NodeId> {
		let desc = node(kind)?;
		let id = NodeId(
			self.nodes
				.iter()
				.map(|node| node.id.0)
				.max()
				.map_or(0, |id| id + 1),
		);

		let params = desc.params.iter().map(|p| (p.name, p.default)).collect();

		self.nodes.push(NodeInstance {
			id,
			kind: desc,
			position,
			params,
			points: Vec::new(),
			velocity_points: Vec::new(),
			text: None,
		});

		Some(id)
	}

	/// The node with the given id.
	pub fn get(&self, id: NodeId) -> Option<&NodeInstance> {
		self.nodes.iter().find(|node| node.id == id)
	}

	/// Adds a control-to-parameter assignment, or moves the depth of one that
	/// already drives the same parameter.
	pub fn modulate(
		&mut self,
		source: (NodeId, usize),
		target: (NodeId, &'static str),
		depth: f32,
	) {
		if let Some(link) = self
			.modulations
			.iter_mut()
			.find(|link| link.source == source && link.target == target)
		{
			link.depth = depth;
			return;
		}

		self.modulations.push(ModulationLink {
			source,
			target,
			depth,
			mode: ModulationMode::default(),
		});
	}

	/// Sets one modulation link's mode.
	pub fn set_modulation_mode(&mut self, index: usize, mode: ModulationMode) -> bool {
		let Some(link) = self.modulations.get_mut(index) else {
			return false;
		};

		link.mode = mode;
		true
	}

	/// Sets one modulation link's depth.
	pub fn set_modulation_depth(&mut self, index: usize, depth: f32) -> bool {
		let Some(link) = self.modulations.get_mut(index) else {
			return false;
		};

		link.depth = depth;
		true
	}

	/// Replaces one node's LFO points or MIDI note mapping.
	pub fn set_points(&mut self, id: NodeId, points: Vec<Point>) -> bool {
		let Some(node) = self.nodes.iter_mut().find(|node| node.id == id) else {
			return false;
		};

		node.points = points;
		true
	}

	/// Replaces one node's MIDI velocity mapping.
	pub fn set_velocity_points(&mut self, id: NodeId, points: Vec<Point>) -> bool {
		let Some(node) = self.nodes.iter_mut().find(|node| node.id == id) else {
			return false;
		};

		if node.kind.id != "midi" {
			return false;
		}

		node.velocity_points = points;

		true
	}

	/// Sets one node's parameter, if the kind has it.
	pub fn set_param(&mut self, id: NodeId, name: &str, value: f32) -> bool {
		let Some(node) = self.nodes.iter_mut().find(|node| node.id == id) else {
			return false;
		};

		if !node.params.contains_key(name) {
			return false;
		}

		node.params.insert(
			node.kind
				.params
				.iter()
				.find(|p| p.name == name)
				.map(|p| p.name)
				.expect("the key came from this node's params"),
			value,
		);

		true
	}

	/// Writes the note a text node carries; false when the node is gone.
	///
	/// The empty string is a note that was cleared, so it is kept as one: an empty
	/// field on a card is not the same as a node that never had a note.
	pub fn set_text(&mut self, id: NodeId, text: String) -> bool {
		let Some(node) = self.nodes.iter_mut().find(|node| node.id == id) else {
			return false;
		};

		node.text = Some(text);

		true
	}

	/// Connects one port to another without checking it.
	///
	/// The compiler is what decides whether the connection makes sense; the
	/// page can therefore draw a wire while the user is still choosing a port.
	pub fn connect(&mut self, from: (NodeId, usize), to: (NodeId, usize)) {
		self.edges.push(Edge { from, to });
	}

	/// Removes one modulation link by its index.
	pub fn remove_modulation(&mut self, index: usize) {
		if index < self.modulations.len() {
			self.modulations.remove(index);
		}
	}

	/// Removes one edge by its index in the edge list.
	pub fn remove_edge(&mut self, index: usize) {
		if index < self.edges.len() {
			self.edges.remove(index);
		}
	}

	/// Swaps one node's kind, keeping its id and where it sits.
	///
	/// What the new kind cannot use goes with the old one: a wire into a port it
	/// does not have, or that now carries something else, and a modulation onto a
	/// parameter it does not declare. The parameters the two kinds share keep
	/// their value, so a filter can become another filter without losing its
	/// cutoff.
	pub fn replace_kind(&mut self, id: NodeId, kind: &str) -> bool {
		let Some(desc) = node(kind) else {
			return false;
		};

		let Some(node) = self.nodes.iter_mut().find(|node| node.id == id) else {
			return false;
		};

		let old = node.kind;
		let mut params: BTreeMap<&'static str, f32> = desc
			.params
			.iter()
			.map(|param| (param.name, param.default))
			.collect();

		for (name, value) in &node.params {
			if let Some(param) = desc.params.iter().find(|param| param.name == *name) {
				params.insert(param.name, value.clamp(param.min, param.max));
			}
		}

		node.kind = desc;
		node.params = params;

		// Only the nodes whose curve is their content carry a pasted one.
		if desc.id != "lfo" && desc.id != "midi" && desc.id != "map" {
			node.points.clear();
		}

		if desc.id != "midi" {
			node.velocity_points.clear();
		}

		let survives = |old: &[PortDesc], new: &[PortDesc], index: usize| {
			matches!(
				(old.get(index), new.get(index)),
				(Some(before), Some(after)) if before.kind == after.kind
			)
		};

		self.edges.retain(|edge| {
			(edge.from.0 != id || survives(old.outputs, desc.outputs, edge.from.1))
				&& (edge.to.0 != id || survives(old.inputs, desc.inputs, edge.to.1))
		});

		self.modulations.retain(|link| {
			let source = link.source.0 != id || survives(old.outputs, desc.outputs, link.source.1);
			let target =
				link.target.0 != id || desc.params.iter().any(|param| param.name == link.target.1);

			source && target
		});

		true
	}

	/// Removes a node and every edge and modulation that touched it.
	pub fn remove_node(&mut self, id: NodeId) {
		self.nodes.retain(|node| node.id != id);
		self.edges
			.retain(|edge| edge.from.0 != id && edge.to.0 != id);
		self.modulations
			.retain(|link| link.source.0 != id && link.target.0 != id);
	}
}

#[cfg(test)]
mod tests {
	#[test]
	fn a_table_node_carries_the_product_parameters() {
		let catalog = super::catalog();
		let kind = catalog
			.iter()
			.find(|kind| kind.id == "table:shapes")
			.expect("the shapes table");

		assert!(
			kind.params
				.iter()
				.any(|param| param.name == "oscillator.smooth_factor")
		);
		assert!(!kind.params.iter().any(|param| param.name == "gain"));
	}

	use super::*;

	#[test]
	fn a_curve_reads_between_its_points() {
		let point = |time: f32, value: f32, bend: f32| Point { time, value, bend };
		let line = [point(0.0, 0.0, 0.0), point(1.0, 1.0, 0.0)];

		assert_eq!(Point::curve(&[], 0.3), 0.3, "no points is the identity");
		assert!((Point::curve(&line, 0.25) - 0.25).abs() < 1e-6);
		assert_eq!(
			Point::curve(&line, -1.0),
			0.0,
			"outside holds the first value"
		);
		assert_eq!(
			Point::curve(&line, 2.0),
			1.0,
			"outside holds the last value"
		);

		let falling = [point(0.0, 1.0, 0.0), point(1.0, 0.5, 0.0)];
		assert!((Point::curve(&falling, 0.5) - 0.75).abs() < 1e-6);

		let bent = [point(0.0, 0.0, 2.0), point(1.0, 1.0, 0.0)];
		let middle = Point::curve(&bent, 0.5);
		assert!(
			middle < 0.5,
			"a positive bend bows the segment down: {middle}"
		);
	}

	#[test]
	fn a_map_node_has_a_control_on_each_side() {
		let map = node("map").expect("a map node");
		assert_eq!(map.inputs.len(), 1);
		assert_eq!(map.outputs.len(), 1);
		assert_eq!(map.inputs[0].kind, SignalKind::Control);
		assert_eq!(map.outputs[0].kind, SignalKind::Control);
	}

	#[test]
	fn a_parameter_carries_the_range_its_dsp_declares() {
		let gain = node("mod:gain").expect("gain");
		let gain_param = gain.params.iter().find(|p| p.name == "gain").expect("gain");
		assert!(
			gain_param.min <= 0.01 && gain_param.max >= 4.0,
			"{gain_param:?}"
		);
		assert!(gain_param.logarithmic);

		let lfo = node("lfo").expect("lfo");
		let rate = lfo.params.iter().find(|p| p.name == "rate").expect("rate");
		assert_eq!((rate.min, rate.max), (0.01, 20.0));
		assert!(rate.logarithmic);

		let sync = lfo.params.iter().find(|p| p.name == "sync").expect("sync");
		assert!(sync.boolean);
		assert_eq!(sync.step, 1.0);
	}

	/// Every splitter has a mixer, and the two line up port for port.
	#[test]
	fn a_split_has_its_mix() {
		for (split_id, mix_id, ports) in [
			("split:ms", "mix:ms", 2),
			("split:lr", "mix:lr", 2),
			("split:bands", "mix:bands", 3),
		] {
			let split = node(split_id).unwrap_or_else(|| panic!("{split_id} is a kind"));
			let mix = node(mix_id).unwrap_or_else(|| panic!("{mix_id} is a kind"));

			assert_eq!(split.outputs.len(), ports, "{split_id}");
			assert_eq!(mix.inputs.len(), ports, "{mix_id}");

			for output in split.outputs {
				assert!(
					mix.inputs.iter().any(|input| input.name == output.name),
					"{mix_id} has no {} input",
					output.name
				);
			}
		}
	}

	#[test]
	fn a_logarithmic_parameter_normalizes_by_ratio() {
		let cutoff = node("mod:filter")
			.expect("filter")
			.params
			.iter()
			.find(|param| param.name == "cutoff")
			.copied()
			.expect("cutoff");

		assert!(cutoff.logarithmic);

		// The geometric middle of a log range is the middle of its curve.
		let middle = (cutoff.min * cutoff.max).sqrt();
		assert!((cutoff.normalize(middle) - 0.5).abs() < 1e-5);
		assert!((cutoff.denormalize(0.5) - middle).abs() < 1.0);

		// Doubling is the same step wherever it starts, which a linear
		// normaliser cannot do.
		let first = cutoff.normalize(cutoff.min * 4.0) - cutoff.normalize(cutoff.min * 2.0);
		let second = cutoff.normalize(cutoff.min * 8.0) - cutoff.normalize(cutoff.min * 4.0);
		assert!((first - second).abs() < 1e-5, "{first} vs {second}");
	}

	#[test]
	fn replacing_a_kind_keeps_the_wires_it_can_carry() {
		let mut graph = Graph::new();
		let input = graph.add_node("input:audio", (0.0, 0.0)).expect("input");
		let gain = graph.add_node("mod:gain", (100.0, 0.0)).expect("gain");
		let out = graph.add_node("out:audio", (200.0, 0.0)).expect("out");
		graph.connect((input, 0), (gain, 0));
		graph.connect((gain, 0), (out, 0));

		assert!(graph.replace_kind(gain, "mod:filter"));

		let node = graph.get(gain).expect("the node kept its id");
		assert_eq!(node.kind.id, "mod:filter");
		assert_eq!(node.position, (100.0, 0.0));
		assert_eq!(graph.edges.len(), 2, "an audio wire on port 0 was dropped");
		assert!(node.params.contains_key("cutoff"));
	}

	#[test]
	fn replacing_a_kind_keeps_a_parameter_it_shares() {
		let mut graph = Graph::new();
		let lfo = graph.add_node("lfo", (0.0, 0.0)).expect("lfo");
		graph.set_param(lfo, "rate", 3.0);
		graph.set_points(
			lfo,
			vec![
				Point {
					time: 0.0,
					value: 0.0,
					bend: 0.0,
				},
				Point {
					time: 1.0,
					value: 1.0,
					bend: 0.0,
				},
			],
		);

		assert!(graph.replace_kind(lfo, "table"));

		let node = graph.get(lfo).expect("the node kept its id");
		assert_eq!(node.params.get("rate"), Some(&3.0));
		assert!(node.points.is_empty(), "a table kept an lfo's curve");
	}

	#[test]
	fn replacing_a_kind_drops_what_the_new_one_cannot_use() {
		let mut graph = Graph::new();
		let lfo = graph.add_node("lfo", (0.0, 0.0)).expect("lfo");
		let filter = graph.add_node("mod:filter", (100.0, 0.0)).expect("filter");
		graph.modulate((lfo, 0), (filter, "cutoff"), 400.0);

		assert!(graph.replace_kind(filter, "mod:reverb"));

		assert!(
			graph.modulations.is_empty(),
			"a link onto a parameter the new kind does not have survived"
		);
	}

	#[test]
	fn the_catalog_includes_the_product_modules() {
		let compressor = node("mod:compressor").expect("the compressor is a kind");
		assert!(
			!compressor.params.is_empty(),
			"no parameters from the schema"
		);
		assert_eq!(compressor.inputs.len(), 2, "main and sidechain");
		assert!(node("mod:distortion").is_some());
		assert!(
			node("mod:filter").is_some(),
			"the effect modules are in the catalog"
		);
	}

	#[test]
	fn a_module_with_a_sidechain_declares_it() {
		let compressor = node("mod:compressor").expect("compressor");
		assert_eq!(compressor.inputs.len(), 2);
		assert_eq!(compressor.inputs[1].name, "sidechain");
		assert!(compressor.inputs[1].optional);

		let gain = node("mod:gain").expect("gain");
		assert_eq!(
			gain.inputs.len(),
			1,
			"only the compressor reads a sidechain"
		);
	}

	#[test]
	fn a_control_source_declares_the_input_it_needs() {
		let pitch = node("pitch").expect("pitch");
		assert_eq!(pitch.inputs.len(), 1, "a pitch tracker reads audio");
		assert_eq!(pitch.inputs[0].name, "audio");
		assert!(matches!(pitch.inputs[0].kind, SignalKind::Audio(_)));

		let midi = node("midi").expect("midi");
		assert_eq!(midi.inputs.len(), 1, "the MIDI source needs a MIDI input");
		assert_eq!(midi.inputs[0].name, "midi");
		assert!(matches!(midi.inputs[0].kind, SignalKind::Midi));

		let envelope = node("envelope").expect("envelope");
		assert!(matches!(envelope.inputs[0].kind, SignalKind::Audio(_)));
	}

	#[test]
	fn removing_a_node_drops_its_modulations() {
		let mut graph = Graph::new();
		let lfo = graph.add_node("lfo", (0.0, 0.0)).expect("lfo");
		let gain = graph.add_node("mod:gain", (0.0, 0.0)).expect("gain");
		graph.modulate((lfo, 0), (gain, "gain"), 0.5);
		assert_eq!(graph.modulations.len(), 1);

		graph.remove_node(lfo);

		assert!(
			graph.modulations.is_empty(),
			"a link from a deleted source would stop the graph compiling"
		);
	}

	#[test]
	fn a_second_link_to_the_same_parameter_moves_the_depth() {
		let mut graph = Graph::new();
		let lfo = graph.add_node("lfo", (0.0, 0.0)).expect("lfo");
		let gain = graph.add_node("mod:gain", (100.0, 0.0)).expect("gain");
		graph.modulate((lfo, 0), (gain, "gain"), 0.5);
		graph.modulate((lfo, 0), (gain, "gain"), 0.9);

		assert_eq!(graph.modulations.len(), 1);
		assert!((graph.modulations[0].depth - 0.9).abs() < 1e-6);
	}

	#[test]
	fn a_depth_can_be_set_by_index() {
		let mut graph = Graph::new();
		let lfo = graph.add_node("lfo", (0.0, 0.0)).expect("lfo");
		let gain = graph.add_node("mod:gain", (100.0, 0.0)).expect("gain");
		graph.modulate((lfo, 0), (gain, "gain"), 0.5);

		assert!(graph.set_modulation_depth(0, 0.25));
		assert!(!graph.set_modulation_depth(9, 0.25));
		assert!((graph.modulations[0].depth - 0.25).abs() < 1e-6);
	}
}
