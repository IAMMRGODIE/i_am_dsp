//! The patch graph: nodes, typed ports and the edges between them.
//!
//! The model is pure data: no DSP, no page, no host. A node's kind is a stable
//! string id resolved against the catalog, so a saved patch never depends on the
//! order of an enum.

use std::collections::BTreeMap;

use i_am_dsp::prelude::{Parameter, Value};

/// What flows through a port.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalKind {
	/// Note events.
	Midi,
	/// Sample frames.
	Audio(Channels),
	/// One value at the control rate.
	Control,
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
	/// The parameters the node exposes, with their defaults.
	pub params: &'static [(&'static str, f32)],
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
		id: "voice:in",
		name: "Voice In",
		inputs: &[port("midi", SignalKind::Midi)],
		outputs: &[port("midi", SignalKind::Midi)],
		latency: 0,
		params: &[],
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
		params: &[],
	},
	NodeDesc {
		id: "gen:additive",
		name: "Additive",
		inputs: &[port("midi", SignalKind::Midi)],
		outputs: &[port("audio", AUDIO)],
		latency: 0,
		params: &[("partials", 16.0), ("spread", 0.3), ("decay", 1.0)],
	},
	NodeDesc {
		id: "gain",
		name: "Gain",
		inputs: &[port("audio", AUDIO)],
		outputs: &[port("audio", AUDIO)],
		latency: 0,
		params: &[("gain", 1.0)],
	},
	NodeDesc {
		id: "filter",
		name: "Filter",
		inputs: &[port("audio", AUDIO)],
		outputs: &[port("audio", AUDIO)],
		latency: 0,
		params: &[
			("mode", 0.0),
			("cutoff", 1_000.0),
			("q", 0.707),
			("stages", 0.0),
			("gain_db", 0.0),
			("slope", 1.0),
		],
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
		id: "mixer",
		name: "Mixer",
		inputs: &[port("a", AUDIO), optional("b", AUDIO)],
		outputs: &[port("audio", AUDIO)],
		latency: 0,
		params: &[("a", 1.0), ("b", 1.0)],
	},
	NodeDesc {
		id: "lfo",
		name: "LFO",
		inputs: &[],
		outputs: &[port("control", SignalKind::Control)],
		latency: 0,
		params: &[("rate", 1.0), ("shape", 0.0), ("sync", 0.0), ("beats", 1.0)],
	},
	NodeDesc {
		id: "random",
		name: "Random",
		inputs: &[],
		outputs: &[port("control", SignalKind::Control)],
		latency: 0,
		params: &[
			("rate", 4.0),
			("smooth", 20.0),
			("sync", 0.0),
			("beats", 1.0),
		],
	},
	NodeDesc {
		id: "table",
		name: "Table",
		inputs: &[],
		outputs: &[port("control", SignalKind::Control)],
		latency: 0,
		params: &[
			("rate", 1.0),
			("shape", 0.0),
			("morph", 0.0),
			("sync", 0.0),
			("beats", 1.0),
		],
	},
	NodeDesc {
		id: "pitch",
		name: "Pitch",
		inputs: &[port("audio", AUDIO)],
		outputs: &[port("control", SignalKind::Control)],
		latency: 0,
		params: &[("clarity", 0.5), ("glide", 20.0)],
	},
	NodeDesc {
		id: "midi",
		name: "MIDI",
		inputs: &[],
		outputs: &[port("control", SignalKind::Control)],
		latency: 0,
		params: &[("mode", 0.0), ("transpose", 0.0), ("fine", 0.0)],
	},
	NodeDesc {
		id: "envelope",
		name: "Envelope",
		inputs: &[port("audio", AUDIO)],
		outputs: &[port("control", SignalKind::Control)],
		latency: 0,
		params: &[("attack", 10.0), ("release", 100.0)],
	},
	NodeDesc {
		id: "delay",
		name: "Delay",
		inputs: &[port("audio", AUDIO)],
		outputs: &[port("audio", AUDIO)],
		latency: 64,
		params: &[("time", 0.25), ("feedback", 0.3)],
	},
];

/// Every node kind: the built-ins and one `mod:<id>` per product module.
pub fn catalog() -> &'static [NodeDesc] {
	static CATALOG: std::sync::OnceLock<Vec<NodeDesc>> = std::sync::OnceLock::new();

	CATALOG.get_or_init(|| {
		let mut kinds = BUILTINS.to_vec();
		kinds.extend(module_catalog());
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
		let mut params: Vec<(&'static str, f32)> = Vec::new();

		for parameter in module.parameters() {
			let name: &'static str = Box::leak(parameter.identifier.clone().into_boxed_str());
			let value = match &parameter.value {
				Value::Float { value, .. } => *value,
				Value::Int { value, .. } => *value as f32,
				Value::Bool(value) => {
					if *value {
						1.0
					} else {
						0.0
					}
				}
				_ => continue,
			};

			params.push((name, value));
		}

		kinds.push(NodeDesc {
			id,
			name: module.name,
			inputs: MODULE_INPUTS,
			outputs: MODULE_OUTPUTS,
			latency,
			params: Box::leak(params.into_boxed_slice()),
		});
	}

	kinds
}

/// One kind per synth product, with a notes input and the product's schema.
fn synth_catalog() -> Vec<NodeDesc> {
	let mut kinds = Vec::new();

	for descriptor in crate::synths::ALL {
		let id: &'static str = Box::leak(descriptor.kind().into_boxed_str());
		let mut params: Vec<(&'static str, f32)> = Vec::new();

		for parameter in (descriptor.parameters)() {
			let name: &'static str = Box::leak(parameter.identifier.clone().into_boxed_str());

			if let Some(value) = parameter_number(&parameter) {
				params.push((name, value));
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

/// The default a parameter carries, when it is a number.
fn parameter_number(parameter: &Parameter) -> Option<f32> {
	match &parameter.value {
		Value::Float { value, .. } => Some(*value),
		Value::Int { value, .. } => Some(*value as f32),
		Value::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
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
	/// The points of an editable LFO, when the node is one.
	pub points: Vec<Point>,
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

		let params = desc.params.iter().copied().collect();

		self.nodes.push(NodeInstance {
			id,
			kind: desc,
			position,
			params,
			points: Vec::new(),
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

	/// Replaces one node's LFO points.
	pub fn set_points(&mut self, id: NodeId, points: Vec<Point>) -> bool {
		let Some(node) = self.nodes.iter_mut().find(|node| node.id == id) else {
			return false;
		};

		node.points = points;
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
				.find(|(key, _)| *key == name)
				.map(|(key, _)| *key)
				.expect("the key came from this node's params"),
			value,
		);

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
	use super::*;

	#[test]
	fn the_catalog_includes_the_product_modules() {
		let compressor = node("mod:compressor").expect("the compressor is a kind");
		assert!(
			!compressor.params.is_empty(),
			"no parameters from the schema"
		);
		assert_eq!(compressor.inputs.len(), 1);
		assert!(node("mod:distortion").is_some());
		assert!(node("filter").is_some(), "the built-ins are still there");
	}

	#[test]
	fn removing_a_node_drops_its_modulations() {
		let mut graph = Graph::new();
		let lfo = graph.add_node("lfo", (0.0, 0.0)).expect("lfo");
		let gain = graph.add_node("gain", (0.0, 0.0)).expect("gain");
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
		let gain = graph.add_node("gain", (100.0, 0.0)).expect("gain");
		graph.modulate((lfo, 0), (gain, "gain"), 0.5);
		graph.modulate((lfo, 0), (gain, "gain"), 0.9);

		assert_eq!(graph.modulations.len(), 1);
		assert!((graph.modulations[0].depth - 0.9).abs() < 1e-6);
	}

	#[test]
	fn a_depth_can_be_set_by_index() {
		let mut graph = Graph::new();
		let lfo = graph.add_node("lfo", (0.0, 0.0)).expect("lfo");
		let gain = graph.add_node("gain", (100.0, 0.0)).expect("gain");
		graph.modulate((lfo, 0), (gain, "gain"), 0.5);

		assert!(graph.set_modulation_depth(0, 0.25));
		assert!(!graph.set_modulation_depth(9, 0.25));
		assert!((graph.modulations[0].depth - 0.25).abs() < 1e-6);
	}
}
