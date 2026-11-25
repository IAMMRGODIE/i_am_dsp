//! Turning a graph into an execution order.
//!
//! This module only reads the model: it validates the ports, sorts the nodes
//! and works out each node's latency. Nothing here touches audio, so every rule
//! is testable without a sample rate.

use std::collections::{BTreeMap, BTreeSet};

use crate::graph::{Channels, Graph, NodeId, SignalKind};

/// Why a graph cannot run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompileError {
	/// Two nodes share an id.
	DuplicateId(NodeId),
	/// An edge names a node that is not in the graph.
	UnknownNode(NodeId),
	/// An edge names a port the node does not have.
	UnknownPort {
		/// The node.
		node: NodeId,
		/// The port index.
		port: usize,
		/// Whether it was an output port.
		output: bool,
	},
	/// The two ports do not carry the same kind.
	KindMismatch {
		/// The source port.
		from: (NodeId, usize),
		/// The destination port.
		to: (NodeId, usize),
	},
	/// An audio edge would have to fold stereo into mono.
	ChannelMismatch {
		/// The source port.
		from: (NodeId, usize),
		/// The destination port.
		to: (NodeId, usize),
	},
	/// The graph has a cycle.
	Cycle(Vec<NodeId>),
	/// A modulation source does not carry a control signal.
	NotControl {
		/// The source port.
		from: (NodeId, usize),
	},
	/// A modulation target is not a parameter of that node.
	UnknownParam {
		/// The target node.
		node: NodeId,
		/// The parameter name.
		param: &'static str,
	},
	/// A modulation crosses a voice boundary.
	CrossScope {
		/// The source port.
		from: (NodeId, usize),
		/// The target node.
		to: NodeId,
	},
}

impl std::fmt::Display for CompileError {
	fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			Self::DuplicateId(id) => write!(formatter, "node {} appears twice", id.0),
			Self::UnknownNode(id) => write!(formatter, "edge names missing node {}", id.0),
			Self::UnknownPort { node, port, output } => write!(
				formatter,
				"node {} has no {} port {}",
				node.0,
				if *output { "output" } else { "input" },
				port
			),
			Self::KindMismatch { from, to } => write!(
				formatter,
				"port {}.{} does not carry what port {}.{} wants",
				from.0.0, from.1, to.0.0, to.1
			),
			Self::ChannelMismatch { from, to } => write!(
				formatter,
				"port {}.{} cannot feed port {}.{}: a stereo signal cannot fold into mono",
				from.0.0, from.1, to.0.0, to.1
			),
			Self::Cycle(ids) => {
				let ids: Vec<String> = ids.iter().map(|id| id.0.to_string()).collect();
				write!(formatter, "the graph has a cycle: {}", ids.join(", "))
			}
			Self::CrossScope { from, to } => write!(
				formatter,
				"the control from {}.{} crosses a voice boundary to node {}",
				from.0.0, from.1, to.0
			),
			Self::NotControl { from } => write!(
				formatter,
				"port {}.{} is not a control output",
				from.0.0, from.1
			),
			Self::UnknownParam { node, param } => {
				write!(formatter, "node {} has no parameter {param}", node.0)
			}
		}
	}
}

impl std::error::Error for CompileError {}

/// The order the engine runs, and each node's latency.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Plan {
	/// The nodes, inputs before consumers.
	pub order: Vec<NodeId>,
	/// The latency of each node's output, in frames.
	pub latency: BTreeMap<NodeId, usize>,
	/// The validated modulation assignments.
	pub modulations: Vec<crate::graph::ModulationLink>,
	/// Frames of compensation for each graph edge, in the graph's edge order:
	/// the short paths into a node are delayed to meet its longest input.
	pub delays: Vec<usize>,
	/// The nodes inside a voice group, which restart on every note.
	pub retrigger: Vec<NodeId>,
}

/// Validates a graph and plans its execution.
pub fn compile(graph: &Graph) -> Result<Plan, CompileError> {
	let mut nodes: BTreeMap<NodeId, &crate::graph::NodeInstance> = BTreeMap::new();

	for node in &graph.nodes {
		if nodes.insert(node.id, node).is_some() {
			return Err(CompileError::DuplicateId(node.id));
		}
	}

	// Every edge must name real ports of the right kind.
	for edge in &graph.edges {
		let from = nodes
			.get(&edge.from.0)
			.ok_or(CompileError::UnknownNode(edge.from.0))?;
		let to = nodes
			.get(&edge.to.0)
			.ok_or(CompileError::UnknownNode(edge.to.0))?;
		let output = from
			.kind
			.outputs
			.get(edge.from.1)
			.ok_or(CompileError::UnknownPort {
				node: edge.from.0,
				port: edge.from.1,
				output: true,
			})?;
		let input = to
			.kind
			.inputs
			.get(edge.to.1)
			.ok_or(CompileError::UnknownPort {
				node: edge.to.0,
				port: edge.to.1,
				output: false,
			})?;

		match (output.kind, input.kind) {
			(SignalKind::Audio(from), SignalKind::Audio(to)) => {
				if from == Channels::Stereo && to == Channels::Mono {
					return Err(CompileError::ChannelMismatch {
						from: edge.from,
						to: edge.to,
					});
				}
			}
			// A table feeds an oscillator port through a table oscillator, and an
			// oscillator feeds a table port through the other adapter: a card can
			// be wired without a conversion node in between.
			(SignalKind::Wavetable, SignalKind::Osc) => {}
			(SignalKind::Osc, SignalKind::Wavetable) => {}
			(a, b) if a == b => {}
			_ => {
				return Err(CompileError::KindMismatch {
					from: edge.from,
					to: edge.to,
				});
			}
		}
	}

	let group = voice_group(graph);

	// A modulation source is a control output and its target is a parameter.
	for link in &graph.modulations {
		let from = nodes
			.get(&link.source.0)
			.ok_or(CompileError::UnknownNode(link.source.0))?;
		let port = from
			.kind
			.outputs
			.get(link.source.1)
			.ok_or(CompileError::UnknownPort {
				node: link.source.0,
				port: link.source.1,
				output: true,
			})?;

		if port.kind != SignalKind::Control {
			return Err(CompileError::NotControl { from: link.source });
		}

		let to = nodes
			.get(&link.target.0)
			.ok_or(CompileError::UnknownNode(link.target.0))?;

		if !to.kind.params.iter().any(|p| p.name == link.target.1) {
			return Err(CompileError::UnknownParam {
				node: link.target.0,
				param: link.target.1,
			});
		}

		// A control inside a voice may only steer that voice; a global control
		// may only steer the graph outside it.
		let source_inside = group.contains(&link.source.0);
		let target_inside = group.contains(&link.target.0);

		if source_inside != target_inside {
			return Err(CompileError::CrossScope {
				from: link.source,
				to: link.target.0,
			});
		}
	}

	// Kahn's algorithm, in id order so a plan is reproducible.
	let mut successors: BTreeMap<NodeId, Vec<NodeId>> = BTreeMap::new();
	let mut indegree: BTreeMap<NodeId, usize> = BTreeMap::new();

	for node in nodes.keys() {
		successors.insert(*node, Vec::new());
		indegree.insert(*node, 0);
	}

	for edge in &graph.edges {
		successors.entry(edge.from.0).or_default().push(edge.to.0);
		*indegree.entry(edge.to.0).or_default() += 1;
	}

	let mut ready: BTreeSet<NodeId> = indegree
		.iter()
		.filter(|(_, degree)| **degree == 0)
		.map(|(id, _)| *id)
		.collect();
	let mut order = Vec::with_capacity(nodes.len());

	while let Some(id) = ready.pop_first() {
		order.push(id);

		for next in successors.get(&id).into_iter().flatten() {
			let degree = indegree.get_mut(next).expect("a node in the graph");
			*degree -= 1;

			if *degree == 0 {
				ready.insert(*next);
			}
		}
	}

	if order.len() != nodes.len() {
		let mut left: Vec<NodeId> = nodes.keys().copied().collect();
		left.retain(|id| !order.contains(id));

		return Err(CompileError::Cycle(left));
	}

	// Latency: every input of a node must arrive at the same frame.
	let mut latency: BTreeMap<NodeId, usize> = BTreeMap::new();
	let mut delays = vec![0; graph.edges.len()];

	for id in &order {
		let node = nodes.get(id).expect("a node in the order");

		// The longest input wins; the shorter ones are delayed to meet it, so a
		// diamond does not have to be built out of equal-length paths.
		let target = graph
			.edges
			.iter()
			.filter(|edge| edge.to.0 == *id)
			.map(|edge| latency[&edge.from.0])
			.max()
			.unwrap_or(0);

		for (index, edge) in graph.edges.iter().enumerate() {
			if edge.to.0 == *id {
				let source = latency.get(&edge.from.0).copied().expect("an earlier node");
				delays[index] = target - source;
			}
		}

		latency.insert(*id, target + node.kind.latency);
	}

	Ok(Plan {
		order,
		latency,
		modulations: graph.modulations.clone(),
		delays,
		retrigger: group
			.into_iter()
			// A note starts the voice group over, but the constructor nodes and
			// the envelope of it must survive: rebuilding them would drop the
			// oscillator the envelope already owns.
			.filter(|id| {
				graph
					.get(*id)
					.map(|instance| {
						!matches!(
							instance.kind.id,
							"adsr"
								| "table:shape"
								| "osc:shape"
								| "osc:am"
								| "osc:rm"
								| "osc:pm"
								| "osc:fm"
								| "table:canvas"
								| "table:shapes"
								| "table:additive"
								| "table:spectrum"
								| "table:script"
								| "warp:bend"
								| "warp:mirror"
								| "warp:sync"
								| "warp:quantize"
								| "warp:remap"
						)
					})
					.unwrap_or(true)
			})
			.collect(),
	})
}

/// The nodes between `voice:in` and `voice:out`, which restart on every note.
///
/// First version: a single group, no nesting. Nodes reachable from the voice
/// input and reaching the voice output belong to it; the boundaries themselves
/// do not.
pub fn voice_group(graph: &Graph) -> BTreeSet<NodeId> {
	let Some(input) = graph.nodes.iter().find(|node| node.kind.id == "voice:in") else {
		return BTreeSet::new();
	};
	let Some(output) = graph.nodes.iter().find(|node| node.kind.id == "voice:out") else {
		return BTreeSet::new();
	};

	let reachable = |start: NodeId, forward: bool| {
		let mut visited = BTreeSet::new();
		let mut stack = vec![start];

		while let Some(id) = stack.pop() {
			for edge in &graph.edges {
				let (from, to) = if forward {
					(edge.from.0, edge.to.0)
				} else {
					(edge.to.0, edge.from.0)
				};

				if from == id && visited.insert(to) {
					stack.push(to);
				}
			}
		}

		visited
	};

	let forward = reachable(input.id, true);
	let backward = reachable(output.id, false);

	let mut group: BTreeSet<NodeId> = forward
		.intersection(&backward)
		.copied()
		.filter(|id| *id != input.id && *id != output.id)
		.collect();

	// A control inside the group belongs to it even though it has no audio
	// edge: the module that modulates the filter is part of the voice too.
	loop {
		let mut changed = false;

		for link in &graph.modulations {
			if group.contains(&link.target.0) && group.insert(link.source.0) {
				changed = true;
			}
		}

		// The tables and oscillators that feed a voice node are part of the
		// voice too: every voice builds its own copy, so their parameters are
		// per-note rather than shared.
		for edge in &graph.edges {
			if group.contains(&edge.to.0)
				&& !group.contains(&edge.from.0)
				&& graph
					.get(edge.from.0)
					.map(|node| crate::graph::is_constructor_kind(node.kind.id))
					.unwrap_or(false)
			{
				group.insert(edge.from.0);
				changed = true;
			}
		}

		if !changed {
			break;
		}
	}

	group
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::graph::{Graph, NodeId};

	fn chain() -> (Graph, NodeId, NodeId, NodeId, NodeId) {
		let mut graph = Graph::new();
		let midi = graph.add_node("input:midi", (0.0, 0.0)).expect("midi");
		let generator = graph.add_node("synth:additive", (100.0, 0.0)).expect("gen");
		let gain = graph.add_node("mod:gain", (200.0, 0.0)).expect("gain");
		let out = graph.add_node("out:audio", (300.0, 0.0)).expect("out");

		graph.connect((midi, 0), (generator, 0));
		graph.connect((generator, 0), (gain, 0));
		graph.connect((gain, 0), (out, 0));

		(graph, midi, generator, gain, out)
	}

	#[test]
	fn a_chain_compiles_in_order() {
		let (graph, midi, generator, gain, out) = chain();
		let plan = compile(&graph).expect("the chain compiles");

		assert_eq!(plan.order, vec![midi, generator, gain, out]);
		assert_eq!(plan.latency[&generator], 0);
	}

	#[test]
	fn a_wire_between_kinds_is_refused() {
		let mut graph = Graph::new();
		let midi = graph.add_node("input:midi", (0.0, 0.0)).expect("midi");
		let gain = graph.add_node("mod:gain", (100.0, 0.0)).expect("gain");
		graph.connect((midi, 0), (gain, 0));

		assert!(matches!(
			compile(&graph),
			Err(CompileError::KindMismatch { .. })
		));
	}

	#[test]
	fn a_table_that_feeds_a_voice_node_is_in_the_voice() {
		use crate::graph::Graph;

		let mut graph = Graph::new();
		let midi = graph.add_node("input:midi", (0.0, 0.0)).expect("midi");
		let voice_in = graph.add_node("voice:in", (50.0, 0.0)).expect("voice in");
		let table = graph.add_node("table:shape", (0.0, 100.0)).expect("table");
		let envelope = graph.add_node("adsr", (150.0, 0.0)).expect("adsr");
		let voice_out = graph
			.add_node("voice:out", (300.0, 0.0))
			.expect("voice out");
		let out = graph.add_node("out:audio", (400.0, 0.0)).expect("out");

		graph.connect((midi, 0), (voice_in, 0));
		graph.connect((voice_in, 0), (envelope, 0));
		graph.connect((table, 0), (envelope, 1));
		graph.connect((envelope, 0), (voice_out, 0));
		graph.connect((voice_out, 0), (out, 0));

		let group = super::voice_group(&graph);

		assert!(
			group.contains(&envelope),
			"the envelope is not in the voice"
		);
		assert!(group.contains(&table), "the table stayed outside the voice");
	}

	#[test]
	fn a_missing_node_is_refused() {
		let mut graph = Graph::new();
		let gain = graph.add_node("mod:gain", (0.0, 0.0)).expect("gain");
		graph.connect((NodeId(99), 0), (gain, 0));

		assert_eq!(compile(&graph), Err(CompileError::UnknownNode(NodeId(99))));
	}

	#[test]
	fn a_missing_port_is_refused() {
		let mut graph = Graph::new();
		let gain = graph.add_node("mod:gain", (0.0, 0.0)).expect("gain");
		graph.connect((gain, 4), (gain, 0));

		assert!(matches!(
			compile(&graph),
			Err(CompileError::UnknownPort { output: true, .. })
		));
	}

	#[test]
	fn a_cycle_is_refused() {
		let mut graph = Graph::new();
		let a = graph.add_node("mod:gain", (0.0, 0.0)).expect("a");
		let b = graph.add_node("mod:gain", (100.0, 0.0)).expect("b");
		graph.connect((a, 0), (b, 0));
		graph.connect((b, 0), (a, 0));

		assert!(matches!(compile(&graph), Err(CompileError::Cycle(_))));
	}

	#[test]
	fn an_output_can_feed_two_inputs() {
		let mut graph = Graph::new();
		let generator = graph.add_node("synth:additive", (0.0, 0.0)).expect("gen");
		let out = graph.add_node("out:audio", (100.0, 0.0)).expect("out");
		let mixer = graph.add_node("mixer", (100.0, 100.0)).expect("mixer");
		graph.connect((generator, 0), (out, 0));
		graph.connect((generator, 0), (mixer, 0));

		assert!(compile(&graph).is_ok());
	}

	#[test]
	fn a_mono_signal_may_feed_a_stereo_input() {
		let mut graph = Graph::new();
		let generator = graph.add_node("synth:additive", (0.0, 0.0)).expect("gen");
		let split = graph.add_node("split:ms", (100.0, 0.0)).expect("split");
		let mixer = graph.add_node("mixer", (200.0, 0.0)).expect("mixer");
		graph.connect((generator, 0), (split, 0));
		graph.connect((split, 0), (mixer, 0));

		assert!(compile(&graph).is_ok());
	}

	#[test]
	fn a_short_path_is_delayed_to_meet_the_long_one() {
		let mut graph = Graph::new();
		let generator = graph.add_node("synth:additive", (0.0, 0.0)).expect("gen");
		let delay = graph.add_node("delay", (100.0, 0.0)).expect("delay");
		let mixer = graph.add_node("mixer", (200.0, 0.0)).expect("mixer");
		graph.connect((generator, 0), (mixer, 0));
		graph.connect((generator, 0), (delay, 0));
		graph.connect((delay, 0), (mixer, 1));

		let plan = compile(&graph).expect("the short path is delayed");
		assert_eq!(plan.latency[&mixer], 64);
		// Edges in graph order: generator to mixer, generator to delay, delay to mixer.
		assert_eq!(plan.delays, vec![64, 0, 0]);
	}

	#[test]
	fn equal_latencies_compile() {
		let mut graph = Graph::new();
		let generator = graph.add_node("synth:additive", (0.0, 0.0)).expect("gen");
		let a = graph.add_node("delay", (100.0, 0.0)).expect("a");
		let b = graph.add_node("delay", (100.0, 100.0)).expect("b");
		let mixer = graph.add_node("mixer", (200.0, 0.0)).expect("mixer");
		graph.connect((generator, 0), (a, 0));
		graph.connect((generator, 0), (b, 0));
		graph.connect((a, 0), (mixer, 0));
		graph.connect((b, 0), (mixer, 1));

		let plan = compile(&graph).expect("both paths are 64 frames");
		assert_eq!(plan.latency[&mixer], 64);
	}

	#[test]
	fn a_modulation_may_not_cross_a_voice_boundary() {
		let mut graph = Graph::new();
		let midi = graph.add_node("input:midi", (0.0, 0.0)).expect("midi");
		let voice_in = graph.add_node("voice:in", (50.0, 0.0)).expect("voice in");
		let generator = graph.add_node("synth:additive", (100.0, 0.0)).expect("gen");
		let filter = graph.add_node("mod:filter", (200.0, 0.0)).expect("filter");
		let voice_out = graph
			.add_node("voice:out", (300.0, 0.0))
			.expect("voice out");
		let out = graph.add_node("out:audio", (400.0, 0.0)).expect("out");
		let gain = graph.add_node("mod:gain", (400.0, 100.0)).expect("gain");
		let lfo = graph.add_node("lfo", (0.0, 100.0)).expect("lfo");
		graph.connect((midi, 0), (voice_in, 0));
		graph.connect((voice_in, 0), (generator, 0));
		graph.connect((generator, 0), (filter, 0));
		graph.connect((filter, 0), (voice_out, 0));
		graph.connect((voice_out, 0), (out, 0));

		// The source drives the voice, so it belongs to the voice; driving the
		// global gain as well would make one control live in two scopes.
		graph.modulate((lfo, 0), (filter, "cutoff"), 400.0);
		graph.modulate((lfo, 0), (gain, "gain"), 0.3);

		assert!(matches!(
			compile(&graph),
			Err(CompileError::CrossScope { .. })
		));
	}

	#[test]
	fn a_modulation_must_come_from_a_control_port() {
		let mut graph = Graph::new();
		let generator = graph.add_node("synth:additive", (0.0, 0.0)).expect("gen");
		let gain = graph.add_node("mod:gain", (100.0, 0.0)).expect("gain");
		graph.modulate((generator, 0), (gain, "gain"), 0.5);

		assert!(matches!(
			compile(&graph),
			Err(CompileError::NotControl { .. })
		));
	}

	#[test]
	fn a_modulation_must_target_a_parameter() {
		let mut graph = Graph::new();
		let lfo = graph.add_node("lfo", (0.0, 0.0)).expect("lfo");
		let gain = graph.add_node("mod:gain", (100.0, 0.0)).expect("gain");
		graph.modulate((lfo, 0), (gain, "nope"), 0.5);

		assert_eq!(
			compile(&graph),
			Err(CompileError::UnknownParam {
				node: gain,
				param: "nope"
			})
		);
	}

	#[test]
	fn a_duplicate_id_is_refused() {
		let mut graph = Graph::new();
		let gain = graph.add_node("mod:gain", (0.0, 0.0)).expect("gain");
		graph.nodes.push(crate::graph::NodeInstance {
			id: gain,
			kind: crate::graph::node("mod:filter").expect("filter"),
			position: (0.0, 0.0),
			params: std::collections::BTreeMap::new(),
			points: Vec::new(),
			velocity_points: Vec::new(),
			text: None,
		});

		assert_eq!(compile(&graph), Err(CompileError::DuplicateId(gain)));
	}
}
