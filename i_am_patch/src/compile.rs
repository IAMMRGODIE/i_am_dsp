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
	/// Two paths into one node arrive with different latencies.
	UnequalLatency {
		/// The node whose inputs disagree.
		node: NodeId,
		/// The two latencies, in frames.
		latencies: [usize; 2],
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
			Self::UnequalLatency { node, latencies } => write!(
				formatter,
				"node {} receives {} and {} frames on different inputs",
				node.0, latencies[0], latencies[1]
			),
		}
	}
}

impl std::error::Error for CompileError {}

/// The order the engine runs, and each node's latency.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Plan {
	/// The nodes, inputs before consumers.
	pub order: Vec<NodeId>,
	/// The latency of each node's output, in frames.
	pub latency: BTreeMap<NodeId, usize>,
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
			(a, b) if a == b => {}
			_ => {
				return Err(CompileError::KindMismatch {
					from: edge.from,
					to: edge.to,
				});
			}
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

	for id in &order {
		let node = nodes.get(id).expect("a node in the order");
		let mut port_latencies: Vec<Option<usize>> = vec![None; node.kind.inputs.len()];

		for edge in &graph.edges {
			if edge.to.0 != *id {
				continue;
			}

			let from = *latency.get(&edge.from.0).expect("an earlier node");
			let slot = &mut port_latencies[edge.to.1];

			match slot {
				Some(existing) if *existing != from => {
					return Err(CompileError::UnequalLatency {
						node: *id,
						latencies: [*existing, from],
					});
				}
				Some(_) => {}
				None => *slot = Some(from),
			}
		}

		// Every input of the same node has to arrive at the same frame, not
		// only the edges that share one port.
		let mut inputs = port_latencies.iter().flatten().copied();
		let first = inputs.next().unwrap_or(0);
		let last = inputs.fold(first, |a, b| a.max(b));

		if first != last {
			return Err(CompileError::UnequalLatency {
				node: *id,
				latencies: [first.min(last), last],
			});
		}

		latency.insert(*id, last + node.kind.latency);
	}

	Ok(Plan { order, latency })
}


#[cfg(test)]
mod tests {
	use super::*;
	use crate::graph::{Graph, NodeId};

	fn chain() -> (Graph, NodeId, NodeId, NodeId, NodeId) {
		let mut graph = Graph::new();
		let midi = graph.add_node("input:midi", (0.0, 0.0)).expect("midi");
		let generator = graph.add_node("gen:additive", (100.0, 0.0)).expect("gen");
		let gain = graph.add_node("gain", (200.0, 0.0)).expect("gain");
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
		let gain = graph.add_node("gain", (100.0, 0.0)).expect("gain");
		graph.connect((midi, 0), (gain, 0));

		assert!(matches!(
			compile(&graph),
			Err(CompileError::KindMismatch { .. })
		));
	}

	#[test]
	fn a_missing_node_is_refused() {
		let mut graph = Graph::new();
		let gain = graph.add_node("gain", (0.0, 0.0)).expect("gain");
		graph.connect((NodeId(99), 0), (gain, 0));

		assert_eq!(compile(&graph), Err(CompileError::UnknownNode(NodeId(99))));
	}

	#[test]
	fn a_missing_port_is_refused() {
		let mut graph = Graph::new();
		let gain = graph.add_node("gain", (0.0, 0.0)).expect("gain");
		graph.connect((gain, 4), (gain, 0));

		assert!(matches!(
			compile(&graph),
			Err(CompileError::UnknownPort { output: true, .. })
		));
	}

	#[test]
	fn a_cycle_is_refused() {
		let mut graph = Graph::new();
		let a = graph.add_node("gain", (0.0, 0.0)).expect("a");
		let b = graph.add_node("gain", (100.0, 0.0)).expect("b");
		graph.connect((a, 0), (b, 0));
		graph.connect((b, 0), (a, 0));

		assert!(matches!(compile(&graph), Err(CompileError::Cycle(_))));
	}

	#[test]
	fn an_output_can_feed_two_inputs() {
		let mut graph = Graph::new();
		let generator = graph.add_node("gen:additive", (0.0, 0.0)).expect("gen");
		let out = graph.add_node("out:audio", (100.0, 0.0)).expect("out");
		let mixer = graph.add_node("mixer", (100.0, 100.0)).expect("mixer");
		graph.connect((generator, 0), (out, 0));
		graph.connect((generator, 0), (mixer, 0));

		assert!(compile(&graph).is_ok());
	}

	#[test]
	fn a_mono_signal_may_feed_a_stereo_input() {
		let mut graph = Graph::new();
		let generator = graph.add_node("gen:additive", (0.0, 0.0)).expect("gen");
		let split = graph.add_node("split:ms", (100.0, 0.0)).expect("split");
		let mixer = graph.add_node("mixer", (200.0, 0.0)).expect("mixer");
		graph.connect((generator, 0), (split, 0));
		graph.connect((split, 0), (mixer, 0));

		assert!(compile(&graph).is_ok());
	}

	#[test]
	fn inputs_must_arrive_at_the_same_frame() {
		let mut graph = Graph::new();
		let generator = graph.add_node("gen:additive", (0.0, 0.0)).expect("gen");
		let delay = graph.add_node("delay", (100.0, 0.0)).expect("delay");
		let mixer = graph.add_node("mixer", (200.0, 0.0)).expect("mixer");
		graph.connect((generator, 0), (mixer, 0));
		graph.connect((generator, 0), (delay, 0));
		graph.connect((delay, 0), (mixer, 1));

		assert!(matches!(
			compile(&graph),
			Err(CompileError::UnequalLatency { latencies: [0, 64], .. })
		));
	}

	#[test]
	fn equal_latencies_compile() {
		let mut graph = Graph::new();
		let generator = graph.add_node("gen:additive", (0.0, 0.0)).expect("gen");
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
	fn a_duplicate_id_is_refused() {
		let mut graph = Graph::new();
		let gain = graph.add_node("gain", (0.0, 0.0)).expect("gain");
		graph.nodes.push(crate::graph::NodeInstance {
			id: gain,
			kind: crate::graph::node("filter").expect("filter"),
			position: (0.0, 0.0),
			params: std::collections::BTreeMap::new(),
		});

		assert_eq!(compile(&graph), Err(CompileError::DuplicateId(gain)));
	}
}
