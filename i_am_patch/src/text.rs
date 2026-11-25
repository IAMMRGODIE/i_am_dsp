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
	to_text_with_slots(graph, &crate::Slots::default())
}

/// Writes the graph and the History pages the page owns as one text document.
pub fn to_text_with_slots(graph: &Graph, slots: &crate::Slots) -> String {
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
				.find(|p| p.name == *key)
				.map(|p| p.default);

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

		if !node.velocity_points.is_empty() {
			let spec: Vec<String> = node
				.velocity_points
				.iter()
				.map(|point| format!("{}:{}:{}", point.time, point.value, point.bend))
				.collect();
			let _ = writeln!(out, "velocity_points {} {}", node.id.0, spec.join(";"));
		}
	}

	for link in &graph.modulations {
		// A hand-written patch can name a node that is not there; the compiler
		// rejects it, so the projection drops it instead of panicking.
		let (Some(from), Some(_)) = (graph.get(link.source.0), graph.get(link.target.0)) else {
			continue;
		};

		if from.kind.outputs.get(link.source.1).is_none() {
			continue;
		}

		let _ = writeln!(
			out,
			"mod {}.{} -> {}.{} depth={}{}",
			link.source.0.0,
			from.kind.outputs[link.source.1].name,
			link.target.0.0,
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
		// An edge can name a node or a port the graph no longer has; the compiler
		// rejects it, so the projection drops it instead of panicking.
		let (Some(from), Some(to)) = (graph.get(edge.from.0), graph.get(edge.to.0)) else {
			continue;
		};

		if from.kind.outputs.get(edge.from.1).is_none() || to.kind.inputs.get(edge.to.1).is_none() {
			continue;
		}

		let _ = writeln!(
			out,
			"connect {}.{} -> {}.{}",
			edge.from.0.0,
			from.kind.outputs[edge.from.1].name,
			edge.to.0.0,
			to.kind.inputs[edge.to.1].name
		);
	}

	for node in &graph.nodes {
		// A note is writing, not a payload: it is short, it belongs beside the node
		// it explains, and a person reading the file should see it there. JSON
		// carries the quotes and the line breaks, so one note stays one line.
		if let Some(note) = &node.text {
			let json = serde_json::to_string(note).expect("the note serializes");
			let _ = writeln!(out, "text {} {}", node.id.0, json);
		}

		match node.kind.id {
			"mod:history" => {
				let pages = slots.history(node.id.0).feed().pages();
				let json = serde_json::to_string(&*pages).expect("the history pages serialize");
				let _ = writeln!(out, "history {} {}", node.id.0, json);
			}
			// Both curves live in a slot, not in a parameter: the text carries them
			// the way it carries the filter table's frame.
			"mod:spectral_dynamics" => {
				let state = slots.spectral(node.id.0).state_json();
				let _ = writeln!(out, "spectral {} {}", node.id.0, state);
			}
			"mod:filter_table" => {
				let pages = slots.filter_table(node.id.0).frame().pages();
				let json = serde_json::to_string(&pages).expect("the filter pages serialize");
				let _ = writeln!(out, "filter_table {} {}", node.id.0, json);
			}
			// A canvas is a drawing whether it is played (the generator) or read
			// (the wavetable): the same slot, the same line.
			"synth:canvas" | "table:canvas" => {
				let shape = slots.canvas(node.id.0).shape();
				let json = serde_json::to_string(&shape).expect("the canvas shape serializes");
				let _ = writeln!(out, "canvas {} {}", node.id.0, json);
			}
			"synth:spectrum" => {
				let state = slots.spectrum(node.id.0).state_json();
				let _ = writeln!(out, "spectrum {} {}", node.id.0, state);
			}
			"synth:script" | "table:script" => {
				let source = slots.script(node.id.0).source_text();
				let json = serde_json::to_string(&source).expect("the script serializes");
				let _ = writeln!(out, "script {} {}", node.id.0, json);
			}
			// The scripted effects keep their script in a worker of their own, and
			// the slot is the same one the node runs: what is written here is what
			// the patch plays.
			"mod:script_mapper" | "mod:script_distortion" => {
				let Some(script) = slots.script_slot(node.id.0, node.kind.id) else {
					continue;
				};
				let json = serde_json::to_string(&script.source()).expect("the script serializes");
				let _ = writeln!(out, "script {} {}", node.id.0, json);
			}
			// A table, a sample or a response is a payload, not a number: it travels in
			// the blob beside this text (see the state module), because a table's frames
			// would be forty kilobytes of digits in a line meant to be read.
			_ => {}
		}
	}

	out
}

/// Reads the text form into a graph.
pub fn from_text(text: &str) -> Result<Graph, TextError> {
	from_text_with_slots(text, &crate::Slots::default())
}

/// Reads the graph and applies the History pages to the slots the caller owns.
pub fn from_text_with_slots(text: &str, slots: &crate::Slots) -> Result<Graph, TextError> {
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
				let kind = words
					.next()
					.ok_or_else(|| error(line, "node needs a kind"))?;
				let desc = node(kind).ok_or_else(|| error(line, format!("unknown kind {kind}")))?;
				let x = parse_f32(words.next(), line)?;
				let y = parse_f32(words.next(), line)?;
				let mut params: BTreeMap<&'static str, f32> =
					desc.params.iter().map(|p| (p.name, p.default)).collect();

				for pair in words {
					let (key, value) = pair
						.split_once('=')
						.ok_or_else(|| error(line, format!("expected key=value, found {pair}")))?;
					let key = desc
						.params
						.iter()
						.find(|p| p.name == key)
						.map(|p| p.name)
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
					velocity_points: Vec::new(),
					text: None,
				});
			}
			"connect" => {
				let from = words
					.next()
					.ok_or_else(|| error(line, "connect needs a source"))?;
				let arrow = words
					.next()
					.ok_or_else(|| error(line, "connect needs ->"))?;

				if arrow != "->" {
					return Err(error(line, "connect needs -> between the ports"));
				}

				let to = words
					.next()
					.ok_or_else(|| error(line, "connect needs a target"))?;
				let (from_node, from_port) = resolve(&graph, from, line, true)?;
				let (to_node, to_port) = resolve(&graph, to, line, false)?;

				graph.edges.push(Edge {
					from: (from_node, from_port),
					to: (to_node, to_port),
				});
			}
			"mod" => {
				let from = words
					.next()
					.ok_or_else(|| error(line, "mod needs a source"))?;
				let arrow = words.next().ok_or_else(|| error(line, "mod needs ->"))?;

				if arrow != "->" {
					return Err(error(line, "mod needs -> between the ports"));
				}

				let to = words
					.next()
					.ok_or_else(|| error(line, "mod needs a target"))?;
				let depth = words
					.next()
					.ok_or_else(|| error(line, "mod needs depth="))?;
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
				let depth = parse_f32(Some(depth), line)?;
				// A depth is a share of the parameter's own range. A file written
				// before that was so stored it in the parameter's units, which is
				// always more than a whole range; read it as the same offset
				// measured against the range instead of losing the link.
				let depth = if depth.abs() > 1.0 {
					let span = graph
						.get(to_node)
						.and_then(|node| {
							node.kind.params.iter().find(|param| param.name == to_port)
						})
						.map(|param| param.max - param.min)
						.unwrap_or(1.0)
						.max(1e-9);

					depth / span
				} else {
					depth
				};

				graph.modulate((from_node, from_port), (to_node, to_port), depth);

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
			"velocity_points" => {
				let id = parse_id(words.next(), line)?;
				let spec = words
					.next()
					.ok_or_else(|| error(line, "velocity_points needs a spec"))?;
				let mut points = Vec::new();

				for part in spec.split(';').filter(|part| !part.is_empty()) {
					let mut fields = part.split(':');
					let time = parse_f32(fields.next(), line)?;
					let value = parse_f32(fields.next(), line)?;
					let bend = parse_f32(fields.next(), line).unwrap_or(0.0);
					points.push(crate::graph::Point { time, value, bend });
				}

				if !graph.set_velocity_points(NodeId(id), points) {
					return Err(error(line, format!("no midi node {id}")));
				}
			}
			"set" => {
				let id = parse_id(words.next(), line)?;
				let pair = words
					.next()
					.ok_or_else(|| error(line, "set needs key=value"))?;
				let (key, value) = pair
					.split_once('=')
					.ok_or_else(|| error(line, format!("expected key=value, found {pair}")))?;

				if !graph.set_param(NodeId(id), key, parse_f32(Some(value), line)?) {
					return Err(error(line, format!("node {id} has no parameter {key}")));
				}
			}
			"history" => {
				let id = parse_id(words.next(), line)?;
				let rest = words.collect::<Vec<_>>().join(" ");
				let pages: Vec<i_am_dsp::prelude::HistoryPage> = serde_json::from_str(&rest)
					.map_err(|parse_error| error(line, parse_error.to_string()))?;
				slots.history(id).feed().set_pages(pages);
			}
			"filter_table" => {
				let id = parse_id(words.next(), line)?;
				let rest = words.collect::<Vec<_>>().join(" ");
				let pages: Vec<Vec<i_am_dsp::prelude::FilterPoint>> =
					serde_json::from_str(&rest)
						.map_err(|parse_error| error(line, parse_error.to_string()))?;
				slots.filter_table(id).frame().set_pages(pages);
			}
			"spectral" => {
				let id = parse_id(words.next(), line)?;
				let rest = words.collect::<Vec<_>>().join(" ");
				let answer = slots.spectral(id).set_json(&rest);

				if answer.contains("error") {
					return Err(error(line, answer));
				}
			}
			"canvas" => {
				let id = parse_id(words.next(), line)?;
				let rest = words.collect::<Vec<_>>().join(" ");
				let shape: String = serde_json::from_str(&rest)
					.map_err(|parse_error| error(line, parse_error.to_string()))?;
				slots.canvas(id).set_shape(shape);
			}
			"spectrum" => {
				let id = parse_id(words.next(), line)?;
				let rest = words.collect::<Vec<_>>().join(" ");
				let value: serde_json::Value = serde_json::from_str(&rest)
					.map_err(|parse_error| error(line, parse_error.to_string()))?;
				let frames = value.get("frames").cloned().unwrap_or(value);
				slots.spectrum(id).set_json(&frames.to_string());
			}
			"script" => {
				let id = parse_id(words.next(), line)?;
				let rest = words.collect::<Vec<_>>().join(" ");
				let source: String = serde_json::from_str(&rest)
					.map_err(|parse_error| error(line, parse_error.to_string()))?;
				let Some(kind) = graph.get(NodeId(id)).map(|node| node.kind.id) else {
					return Err(error(line, format!("no node {id} to write a script on")));
				};

				// A scripted effect runs a worker of its own; the synth products run
				// the handle they were built with. The file says which node, and the
				// node's kind is what says where its script belongs.
				match kind {
					"mod:script_mapper" | "mod:script_distortion" => {
						let Some(slot) = slots.script_slot(id, kind) else {
							return Err(error(line, format!("node {id} has no script")));
						};

						slot.set_source(source);
					}
					_ => {
						slots.script(id).apply(source);
					}
				}
			}
			"text" => {
				let id = parse_id(words.next(), line)?;
				let rest = words.collect::<Vec<_>>().join(" ");
				let note: String = serde_json::from_str(&rest)
					.map_err(|parse_error| error(line, parse_error.to_string()))?;

				if !graph.set_text(NodeId(id), note) {
					return Err(error(line, format!("no node {id} to write a note on")));
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
		.find(|p| p.name == param)
		.map(|p| p.name)
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
	use crate::graph::{ModulationLink, ModulationMode};

	use super::*;

	fn patch() -> Graph {
		let mut graph = Graph::new();
		let midi = graph.add_node("input:midi", (0.0, 0.0)).expect("midi");
		let generator = graph.add_node("synth:additive", (100.0, 0.0)).expect("gen");
		let gain = graph.add_node("mod:gain", (200.0, 0.0)).expect("gain");
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

	/// A note is a person's writing, so it survives the trip through the one-line
	/// form with its quotes and its line breaks — and it costs the file exactly one
	/// line, however long it is.
	#[test]
	fn a_note_round_trips_through_the_text() {
		let graph = patch();
		let lines = to_text(&graph).lines().count();
		let mut graph = graph;
		let note = graph.add_node("text", (500.0, 0.0)).expect("a text node");
		let written = "where this patch starts:
\"cutoff\" is 52 Hz on purpose";

		assert!(graph.set_text(note, written.to_string()));

		let text = to_text(&graph);

		assert_eq!(
			text.lines().count(),
			lines + 2,
			"a node and its note are two lines: {text}"
		);
		assert_eq!(
			text.lines().filter(|line| line.starts_with("text ")).count(),
			1,
			"the note broke into more than one line: {text}"
		);

		let back = from_text(&text).expect("the text parses");

		assert_eq!(
			back.get(note).and_then(|node| node.text.as_deref()),
			Some(written)
		);
	}

	/// A note names a node: one that is not there is a mistake in the file, not a
	/// line to drop.
	#[test]
	fn a_note_on_a_node_that_is_not_there_is_an_error() {
		let text = "node 1 out:audio 0 0
text 9 \"ghost\"\n";

		assert!(from_text(text).is_err());
	}

	/// The spectral dynamics curves live in a slot, not in a parameter: the text
	/// carries them and reading it back puts them in the same handle the node runs.
	#[test]
	fn the_spectral_dynamics_curves_round_trip_through_the_text() {
		let mut graph = Graph::new();
		let node = graph
			.add_node("mod:spectral_dynamics", (0.0, 0.0))
			.expect("a spectral dynamics node");
		let slots = crate::Slots::default();
		let handle = slots.spectral(node.0);
		let flat = handle.state_json();
		let edited = flat.replace("0.5", "0.75");

		assert_ne!(flat, edited, "the curve changed");

		let answer = handle.set_json(&edited);

		assert!(!answer.contains("error"), "{answer}");

		let text = to_text_with_slots(&graph, &slots);

		assert!(text.contains("spectral 0 "), "{text}");

		let back = crate::Slots::default();
		from_text_with_slots(&text, &back).expect("the text parses");

		assert_eq!(back.spectral(node.0).state_json(), handle.state_json());
	}

	/// A canvas is a drawing whether it is played or read: the text carries the
	/// wavetable's drawing as well as the generator's.
	#[test]
	fn a_wavetable_canvas_round_trips_through_the_text() {
		let mut graph = Graph::new();
		let node = graph
			.add_node("table:canvas", (0.0, 0.0))
			.expect("a canvas table");
		let slots = crate::Slots::default();
		// The drawing a fresh canvas starts with, so the shape is one the parser
		// accepts: what matters here is that the line is written and read back.
		let drawing = slots.canvas(node.0).shape();

		assert!(!drawing.is_empty(), "a canvas starts with a drawing");

		let text = to_text_with_slots(&graph, &slots);

		assert!(text.contains("canvas 0"), "{text}");

		let back = crate::Slots::default();
		from_text_with_slots(&text, &back).expect("the text parses");

		assert_eq!(back.canvas(node.0).shape(), drawing);
	}

	/// A scripted effect's script is part of the patch: the text carries the source
	/// the node runs, and reading the text back puts it in the worker again.
	#[test]
	fn a_scripted_effect_keeps_its_script_in_the_text() {
		let mut graph = Graph::new();
		let node = graph
			.add_node("mod:script_mapper", (0.0, 0.0))
			.expect("a mapper");
		let slots = crate::Slots::default();
		let Some(crate::ScriptSlot::Mapper(worker)) =
			slots.script_slot(node.0, "mod:script_mapper")
		else {
			panic!("the mapper's worker");
		};

		worker.set_source("[x * 4.0, a]".to_string());

		let text = to_text_with_slots(&graph, &slots);

		assert!(text.contains("script 0 \"[x * 4.0, a]\""), "{text}");

		let back = crate::Slots::default();
		from_text_with_slots(&text, &back).expect("the text parses");

		let Some(crate::ScriptSlot::Mapper(worker)) =
			back.script_slot(node.0, "mod:script_mapper")
		else {
			panic!("the mapper's worker");
		};

		assert_eq!(worker.source(), "[x * 4.0, a]");
	}

	/// Every payload a node plays travels in the blob beside the text: a table, a fitted
	/// table, a sample with its slices, a response with its fades, and a drum kit.
	#[test]
	fn the_slot_state_carries_every_payload() {
		use crate::state::SlotState;
		use i_am_dsp::prelude::{ImportedTable, SampleData, WaveBank};
		use i_am_effects::{IrData, IrFades};

		let slots = crate::Slots::default();
		let (wave, fitted, sample, ir, click) = (11u32, 12, 13, 14, 15);

		slots.waves(wave).lock().expect("the slot").bank = Some(WaveBank::new(
			Some("note.wav".to_string()),
			vec![vec![0.0, 0.5, 0.0, -0.5]],
		));
		slots.fitted(fitted).set(ImportedTable {
			name: "peel".to_string(),
			bytes: vec![1, 2, 3],
		});
		{
			let slot = slots.sample(sample);
			let mut slot = slot.lock().expect("the slot");

			slot.sample = Some(std::sync::Arc::new(SampleData {
				name: "kick.wav".to_string(),
				rate: 48_000,
				pcm: vec![0.0, 0.5, -0.5],
				right: None,
			}));
			slot.slices = vec![0.0, 0.5];
		}
		slots.ir(ir).set(std::sync::Arc::new(IrData::from_samples(
			"room.wav".to_string(),
			48_000,
			vec![0.0, 0.2, 0.0],
			vec![0.0, 0.2, 0.0],
		)));
		slots.ir(ir).set_fades(IrFades {
			fade_in: 0.1,
			fade_out: 0.2,
		});
		slots.drummer(click).restore_kit(
			Some(SampleData {
				name: "click.wav".to_string(),
				rate: 48_000,
				pcm: vec![0.1],
				right: None,
			}),
			std::array::from_fn(|index| {
				(index == 0).then(|| SampleData {
					name: "kick.wav".to_string(),
					rate: 48_000,
					pcm: vec![0.9, -0.9],
					right: None,
				})
			}),
		);

		let state = slots.capture();
		let binary = i_am_dsp::prelude::to_binary(&state).expect("the state serializes");
		let back: SlotState = i_am_dsp::prelude::from_binary(binary).expect("the state decodes");
		let loaded = crate::Slots::default();
		loaded.restore(&back);

		let wave = loaded.waves(wave).lock().expect("the slot").bank.clone();
		let wave = wave.expect("the table came back");
		assert_eq!(wave.name(), Some("note.wav"));

		let fitted = loaded.fitted(fitted).current().expect("the fitted table");
		assert_eq!(fitted.bytes, vec![1, 2, 3]);

		{
			let slot = loaded.sample(sample);
			let slot = slot.lock().expect("the slot");
			let sample = slot.sample.as_ref().expect("the sample came back");
			assert_eq!(sample.name, "kick.wav");
			assert_eq!(sample.pcm, vec![0.0, 0.5, -0.5]);
			assert_eq!(slot.slices, vec![0.0, 0.5]);
		}

		let ir = loaded.ir(ir);
		assert_eq!(ir.load().expect("the response came back").name, "room.wav");
		assert_eq!(
			ir.fades(),
			IrFades {
				fade_in: 0.1,
				fade_out: 0.2
			}
		);

		let kit = loaded.drummer(click);
		assert_eq!(kit.click_sample().expect("the click").name, "click.wav");
		let shots = kit.one_shots();
		assert_eq!(shots[0].as_ref().expect("the kick").name, "kick.wav");
		assert!(shots[1].is_none(), "an empty slot came back full");
	}

	#[test]
	fn a_velocity_curve_round_trips() {
		let mut graph = Graph::new();
		let midi = graph.add_node("midi", (0.0, 0.0)).expect("midi");

		graph.set_velocity_points(
			midi,
			vec![
				crate::graph::Point {
					time: 0.0,
					value: 0.0,
					bend: 0.0,
				},
				crate::graph::Point {
					time: 1.0,
					value: 1.0,
					bend: 0.2,
				},
			],
		);

		let back = from_text(&to_text(&graph)).expect("the text parses");
		let node = back.get(midi).expect("the midi node");

		assert_eq!(node.velocity_points.len(), 2);
		assert!((node.velocity_points[1].bend - 0.2).abs() < 1e-6);
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
		assert_eq!(
			back.modulations[0].mode,
			crate::graph::ModulationMode::Centered
		);
	}

	#[test]
	fn a_point_table_round_trips() {
		let mut graph = Graph::new();
		let lfo = graph.add_node("lfo", (0.0, 0.0)).expect("lfo");
		graph.set_points(
			lfo,
			vec![
				crate::graph::Point {
					time: 0.0,
					value: 0.5,
					bend: 0.0,
				},
				crate::graph::Point {
					time: 1.0,
					value: 1.0,
					bend: 0.2,
				},
			],
		);

		let back = from_text(&to_text(&graph)).expect("the text parses");
		let node = back.get(lfo).expect("the lfo");

		assert_eq!(node.points.len(), 2);
		assert!((node.points[1].value - 1.0).abs() < 1e-6);
		assert!((node.points[1].bend - 0.2).abs() < 1e-6);
	}

	#[test]
	fn a_link_to_a_missing_node_is_skipped() {
		let mut graph = Graph::new();
		graph.add_node("mod:gain", (0.0, 0.0)).expect("gain");
		graph.modulations.push(ModulationLink {
			source: (NodeId(9), 0),
			target: (NodeId(0), "gain"),
			depth: 0.5,
			mode: ModulationMode::FromBase,
		});

		let text = to_text(&graph);

		assert!(
			!text.contains("mod "),
			"the dangling link was written: {text}"
		);
	}

	#[test]
	fn a_bad_line_names_itself() {
		let error = from_text("node 0 warp 0 0").expect_err("no such kind");
		assert_eq!(error.line, 1);
		assert!(error.message.contains("unknown kind"));

		let error =
			from_text("node 0 mod:gain 0 0\nconnect 0.audio -> 1.audio").expect_err("no node 1");
		assert_eq!(error.line, 2);
	}

	#[test]
	fn a_comment_and_a_blank_line_are_ignored() {
		let graph = from_text("# a patch\n\nnode 0 mod:gain 0 0\n").expect("parses");
		assert_eq!(graph.nodes.len(), 1);
	}

	#[test]
	fn history_pages_travel_with_the_text() {
		use i_am_dsp::prelude::{HistoryPage, HistoryPoint};

		let mut graph = Graph::new();
		let id = graph.add_node("mod:history", (0.0, 0.0)).expect("history");
		let slots = crate::Slots::default();
		slots.history(id.0).feed().set_pages(vec![HistoryPage {
			points: vec![
				HistoryPoint {
					time: 0.0,
					value: 0.0,
					bend: 0.0,
				},
				HistoryPoint {
					time: 1.0,
					value: 1.0,
					bend: 0.0,
				},
			],
		}]);

		let text = to_text_with_slots(&graph, &slots);
		assert!(text.contains("history "), "{text}");

		let restored = crate::Slots::default();
		let back = from_text_with_slots(&text, &restored).expect("the text parses");

		assert_eq!(back.nodes.len(), 1);
		assert_eq!(restored.history(id.0).feed().pages().len(), 1);
		assert_eq!(
			restored.history(id.0).feed().pages()[0].points[0].value,
			0.0
		);
	}
}
