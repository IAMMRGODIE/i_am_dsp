//! The line patch format: the graph as text an agent can read and write.
//!
//! ```text
//! node 0 input:midi 0 0
//! node 1 gen:additive 100 0 decay=1.5
//! connect 0.midi -> 1.midi
//! set 1 decay=2
//! ```
//!
//! JSON is the truth; this is a projection of it, parsed into the same graph
//! and rejected by the same compiler. No expressions, no loops: a line is a
//! statement and a bad line is an error with a number.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use crate::graph::{Edge, Graph, NodeId, NodeInstance, node};

/// A line that could not be read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextError {
	/// The 1-based line.
	pub line: usize,
	/// What was wrong.
	pub message: String,
}

impl std::fmt::Display for TextError {
	fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(formatter, "line {}: {}", self.line, self.message)
	}
}

impl std::error::Error for TextError {}

/// Writes the graph as text, one statement per line.
pub fn to_text(graph: &Graph) -> String {
	let mut out = String::from("# i_am_patch\n");

	for node in &graph.nodes {
		let _ = writeln!(
			out,
			"node {} {} {} {}",
			node.id.0, node.kind.id, node.position.0, node.position.1
		);

		for (key, value) in &node.params {
			let default = node
				.kind
				.params
				.iter()
				.find(|(name, _)| name == key)
				.map(|(_, value)| *value);

			if default != Some(*value) {
				let _ = writeln!(out, "set {} {}={}", node.id.0, key, value);
			}
		}

		if !node.points.is_empty() {
			let spec: Vec<String> = node
				.points
				.iter()
				.map(|point| format!("{}:{}:{}", point.time, point.value, point.bend))
				.collect();
			let _ = writeln!(out, "points {} {}", node.id.0, spec.join(";"));
		}
	}

	for link in &graph.modulations {
		let from = graph.get(link.source.0).expect("a link names a node");
		let _ = writeln!(
			out,
			"mod {}.{} -> {}.{} depth={}{}",
			link.source.0 .0,
			from.kind.outputs[link.source.1].name,
			link.target.0 .0,
			link.target.1,
			link.depth,
			if link.mode == crate::graph::ModulationMode::Centered {
				" mode=center"
			} else {
				""
			}
		);
	}

	for edge in &graph.edges {
		let from = graph.get(edge.from.0).expect("an edge names a node");
		let to = graph.get(edge.to.0).expect("an edge names a node");
		let _ = writeln!(
			out,
			"connect {}.{} -> {}.{}",
			edge.from.0 .0,
			from.kind.outputs[edge.from.1].name,
			edge.to.0 .0,
			to.kind.inputs[edge.to.1].name
		);
	}

	out
}

/// Reads the text form into a graph.
pub fn from_text(text: &str) -> Result<Graph, TextError> {
	let mut graph = Graph::new();

	for (index, raw) in text.lines().enumerate() {
		let line = index + 1;
		let content = raw.split('#').next().unwrap_or("").trim();

		if content.is_empty() {
			continue;
		}

		let mut words = content.split_whitespace();
		let statement = words.next().unwrap_or("");

		match statement {
			"node" => {
				let id = parse_id(words.next(), line)?;
				let kind = words.next().ok_or_else(|| error(line, "node needs a kind"))?;
				let desc = node(kind).ok_or_else(|| error(line, format!("unknown kind {kind}")))?;
				let x = parse_f32(words.next(), line)?;
				let y = parse_f32(words.next(), line)?;
				let mut params: BTreeMap<&'static str, f32> = desc.params.iter().copied().collect();

				for pair in words {
					let (key, value) = pair
						.split_once('=')
						.ok_or_else(|| error(line, format!("expected key=value, found {pair}")))?;
					let key = desc
						.params
						.iter()
						.find(|(name, _)| *name == key)
						.map(|(name, _)| *name)
						.ok_or_else(|| error(line, format!("{kind} has no parameter {key}")))?;
					params.insert(key, parse_f32(Some(value), line)?);
				}

				if graph.get(NodeId(id)).is_some() {
					return Err(error(line, format!("node {id} appears twice")));
				}

				graph.nodes.push(NodeInstance {
					id: NodeId(id),
					kind: desc,
					position: (x, y),
					params,
					points: Vec::new(),
				});
			}
			"connect" => {
				let from = words.next().ok_or_else(|| error(line, "connect needs a source"))?;
				let arrow = words.next().ok_or_else(|| error(line, "connect needs ->"))?;

				if arrow != "->" {
					return Err(error(line, "connect needs -> between the ports"));
				}

				let to = words.next().ok_or_else(|| error(line, "connect needs a target"))?;
				let (from_node, from_port) = resolve(&graph, from, line, true)?;
				let (to_node, to_port) = resolve(&graph, to, line, false)?;

				graph.edges.push(Edge {
					from: (from_node, from_port),
					to: (to_node, to_port),
				});
			}
			"mod" => {
				let from = words.next().ok_or_else(|| error(line, "mod needs a source"))?;
				let arrow = words.next().ok_or_else(|| error(line, "mod needs ->"))?;

				if arrow != "->" {
					return Err(error(line, "mod needs -> between the ports"));
				}

				let to = words.next().ok_or_else(|| error(line, "mod needs a target"))?;
				let depth = words.next().ok_or_else(|| error(line, "mod needs depth="))?;
				let depth = depth
					.strip_prefix("depth=")
					.ok_or_else(|| error(line, "mod needs depth="))?;
				let mode = match words.next() {
					Some("mode=center") => crate::graph::ModulationMode::Centered,
					Some("mode=frombase") | None => crate::graph::ModulationMode::FromBase,
					Some(_) => return Err(error(line, "mode is frombase or center")),
				};
				let (from_node, from_port) = resolve(&graph, from, line, true)?;
				let (to_node, to_port) = resolve_param(&graph, to, line)?;

				graph.modulate(
					(from_node, from_port),
					(to_node, to_port),
					parse_f32(Some(depth), line)?,
				);

				if let Some(link) = graph.modulations.last_mut() {
					link.mode = mode;
				}
			}
			"points" => {
				let id = parse_id(words.next(), line)?;
				let spec = words
					.next()
					.ok_or_else(|| error(line, "points needs a spec"))?;
				let mut points = Vec::new();

				for part in spec.split(';').filter(|part| !part.is_empty()) {
					let mut fields = part.split(':');
					let time = parse_f32(fields.next(), line)?;
					let value = parse_f32(fields.next(), line)?;
					let bend = parse_f32(fields.next(), line).unwrap_or(0.0);
					points.push(crate::graph::Point { time, value, bend });
				}

				if !graph.set_points(NodeId(id), points) {
					return Err(error(line, format!("no node {id}")));
				}
			}
			"set" => {
				let id = parse_id(words.next(), line)?;
				let pair = words.next().ok_or_else(|| error(line, "set needs key=value"))?;
				let (key, value) = pair
					.split_once('=')
					.ok_or_else(|| error(line, format!("expected key=value, found {pair}")))?;

				if !graph.set_param(NodeId(id), key, parse_f32(Some(value), line)?) {
					return Err(error(line, format!("node {id} has no parameter {key}")));
				}
			}
			other => return Err(error(line, format!("unknown statement {other}"))),
		}
	}

	Ok(graph)
}

fn error(line: usize, message: impl Into<String>) -> TextError {
	TextError {
		line,
		message: message.into(),
	}
}

fn parse_id(word: Option<&str>, line: usize) -> Result<u32, TextError> {
	word.ok_or_else(|| error(line, "expected a node id"))?
		.parse()
		.map_err(|_| error(line, "the node id is not a number"))
}

fn parse_f32(word: Option<&str>, line: usize) -> Result<f32, TextError> {
	word.ok_or_else(|| error(line, "expected a number"))?
		.parse()
		.map_err(|_| error(line, "not a number"))
}

fn resolve_param(
	graph: &Graph,
	reference: &str,
	line: usize,
) -> Result<(NodeId, &'static str), TextError> {
	let (id, param) = reference
		.split_once('.')
		.ok_or_else(|| error(line, format!("expected node.param, found {reference}")))?;
	let id: u32 = id
		.parse()
		.map_err(|_| error(line, format!("{id} is not a node id")))?;
	let instance = graph
		.get(NodeId(id))
		.ok_or_else(|| error(line, format!("no node {id}")))?;
	let param = instance
		.kind
		.params
		.iter()
		.find(|(name, _)| *name == param)
		.map(|(name, _)| *name)
		.ok_or_else(|| error(line, format!("node {id} has no parameter {param}")))?;

	Ok((NodeId(id), param))
}

fn resolve(
	graph: &Graph,
	reference: &str,
	line: usize,
	output: bool,
) -> Result<(NodeId, usize), TextError> {
	let (id, port) = reference
		.split_once('.')
		.ok_or_else(|| error(line, format!("expected node.port, found {reference}")))?;
	let id: u32 = id
		.parse()
		.map_err(|_| error(line, format!("{id} is not a node id")))?;
	let instance = graph
		.get(NodeId(id))
		.ok_or_else(|| error(line, format!("no node {id}")))?;
	let ports = if output {
		instance.kind.outputs
	} else {
		instance.kind.inputs
	};
	let port = ports
		.iter()
		.position(|candidate| candidate.name == port)
		.ok_or_else(|| error(line, format!("node {id} has no port {port}")))?;

	Ok((NodeId(id), port))
}

#[cfg(test)]
mod tests {
	use super::*;

	fn patch() -> Graph {
		let mut graph = Graph::new();
		let midi = graph.add_node("input:midi", (0.0, 0.0)).expect("midi");
		let generator = graph.add_node("gen:additive", (100.0, 0.0)).expect("gen");
		let gain = graph.add_node("gain", (200.0, 0.0)).expect("gain");
		let out = graph.add_node("out:audio", (300.0, 0.0)).expect("out");
		graph.connect((midi, 0), (generator, 0));
		graph.connect((generator, 0), (gain, 0));
		graph.connect((gain, 0), (out, 0));
		graph.set_param(gain, "gain", 0.5);

		graph
	}

	#[test]
	fn a_graph_round_trips_through_text() {
		let graph = patch();
		let text = to_text(&graph);
		let back = from_text(&text).expect("the text parses");

		assert_eq!(back.nodes.len(), graph.nodes.len());
		assert_eq!(back.edges.len(), graph.edges.len());

		for (original, parsed) in graph.nodes.iter().zip(&back.nodes) {
			assert_eq!(original.id, parsed.id);
			assert_eq!(original.kind.id, parsed.kind.id);
			assert_eq!(original.params.len(), parsed.params.len());
		}

		for (original, parsed) in graph.edges.iter().zip(&back.edges) {
			assert_eq!(original, parsed);
		}

		assert!((back.get(graph.nodes[2].id).expect("gain").param("gain") - 0.5).abs() < 1e-6);
	}

	#[test]
	fn a_modulation_round_trips() {
		let mut graph = patch();
		let lfo = graph.add_node("lfo", (0.0, 100.0)).expect("lfo");
		let gain = graph.nodes[2].id;
		graph.modulate((lfo, 0), (gain, "gain"), 0.4);
		graph.set_modulation_mode(0, crate::graph::ModulationMode::Centered);

		let back = from_text(&to_text(&graph)).expect("the text parses");

		assert_eq!(back.modulations.len(), 1);
		assert_eq!(back.modulations[0].target.1, "gain");
		assert!((back.modulations[0].depth - 0.4).abs() < 1e-6);
		assert_eq!(back.modulations[0].mode, crate::graph::ModulationMode::Centered);
	}

	#[test]
	fn a_point_table_round_trips() {
		let mut graph = Graph::new();
		let lfo = graph.add_node("lfo", (0.0, 0.0)).expect("lfo");
		graph.set_points(
			lfo,
			vec![
				crate::graph::Point { time: 0.0, value: 0.5, bend: 0.0 },
				crate::graph::Point { time: 1.0, value: 1.0, bend: 0.2 },
			],
		);

		let back = from_text(&to_text(&graph)).expect("the text parses");
		let node = back.get(lfo).expect("the lfo");

		assert_eq!(node.points.len(), 2);
		assert!((node.points[1].value - 1.0).abs() < 1e-6);
		assert!((node.points[1].bend - 0.2).abs() < 1e-6);
	}

	#[test]
	fn a_bad_line_names_itself() {
		let error = from_text("node 0 warp 0 0").expect_err("no such kind");
		assert_eq!(error.line, 1);
		assert!(error.message.contains("unknown kind"));

		let error = from_text("node 0 gain 0 0\nconnect 0.audio -> 1.audio").expect_err("no node 1");
		assert_eq!(error.line, 2);
	}

	#[test]
	fn a_comment_and_a_blank_line_are_ignored() {
		let graph = from_text("# a patch\n\nnode 0 gain 0 0\n").expect("parses");
		assert_eq!(graph.nodes.len(), 1);
	}
}
