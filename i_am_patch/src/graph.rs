//! The patch graph: nodes, typed ports and the edges between them.
//!
//! The model is pure data: no DSP, no page, no host. A node's kind is a stable
//! string id resolved against the catalog, so a saved patch never depends on the
//! order of an enum.

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
#[derive(Debug)]
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

/// Every node kind the patcher ships with.
pub const CATALOG: &[NodeDesc] = &[
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
		name: "Gain / Pan",
		inputs: &[port("audio", AUDIO)],
		outputs: &[port("audio", AUDIO)],
		latency: 0,
		params: &[("gain", 1.0), ("pan", 0.0)],
	},
	NodeDesc {
		id: "filter",
		name: "Filter",
		inputs: &[port("audio", AUDIO)],
		outputs: &[port("audio", AUDIO)],
		latency: 0,
		params: &[("cutoff", 2_000.0), ("resonance", 0.2)],
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
		params: &[("rate", 1.0), ("shape", 0.0)],
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

/// The descriptor for a kind id.
pub fn node(kind: &str) -> Option<&'static NodeDesc> {
	CATALOG.iter().find(|desc| desc.id == kind)
}

/// A node's identity inside one graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(pub u32);

/// One node placed in a graph.
#[derive(Debug, Clone, Copy)]
pub struct NodeInstance {
	/// Its id within the graph.
	pub id: NodeId,
	/// Its kind.
	pub kind: &'static NodeDesc,
	/// Where the page draws it.
	pub position: (f32, f32),
}

/// One connection: an output port into an input port.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Edge {
	/// The source node and output port.
	pub from: (NodeId, usize),
	/// The destination node and input port.
	pub to: (NodeId, usize),
}

/// A patch: nodes and the edges between them.
#[derive(Debug, Default, Clone)]
pub struct Graph {
	/// The nodes, in insertion order.
	pub nodes: Vec<NodeInstance>,
	/// The edges.
	pub edges: Vec<Edge>,
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

		self.nodes.push(NodeInstance {
			id,
			kind: desc,
			position,
		});

		Some(id)
	}

	/// The node with the given id.
	pub fn get(&self, id: NodeId) -> Option<&NodeInstance> {
		self.nodes.iter().find(|node| node.id == id)
	}

	/// Connects one port to another without checking it.
	///
	/// The compiler is what decides whether the connection makes sense; the
	/// page can therefore draw a wire while the user is still choosing a port.
	pub fn connect(&mut self, from: (NodeId, usize), to: (NodeId, usize)) {
		self.edges.push(Edge { from, to });
	}

	/// Removes a node and every edge that touched it.
	pub fn remove_node(&mut self, id: NodeId) {
		self.nodes.retain(|node| node.id != id);
		self.edges.retain(|edge| edge.from.0 != id && edge.to.0 != id);
	}
}
