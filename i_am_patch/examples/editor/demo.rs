//! The graph the editor opens on.

use super::*;

pub(super) fn demo() -> Graph {
	let mut graph = Graph::new();
	let midi = graph.add_node("input:midi", (0.0, 120.0)).expect("midi");
	let voice_in = graph
		.add_node("voice:in", (300.0, 120.0))
		.expect("voice in");
	let generator = graph
		.add_node("synth:additive", (560.0, 120.0))
		.expect("gen");
	let voice_out = graph
		.add_node("voice:out", (860.0, 120.0))
		.expect("voice out");
	let filter = graph
		.add_node("mod:filter", (880.0, 120.0))
		.expect("filter");
	let lfo = graph.add_node("lfo", (880.0, 580.0)).expect("lfo");
	let out = graph.add_node("out:audio", (1500.0, 120.0)).expect("out");
	graph.connect((midi, 0), (voice_in, 0));
	graph.connect((voice_in, 0), (generator, 0));
	graph.connect((generator, 0), (voice_out, 0));
	graph.connect((voice_out, 0), (filter, 0));
	graph.connect((filter, 0), (out, 0));
	// A depth is a share of the cutoff's own range, so a third of it is a sweep
	// of about two octaves either way.
	graph.modulate((lfo, 0), (filter, "cutoff"), 0.33);
	graph.set_param(filter, "cutoff", 1_200.0);

	graph
}

pub(super) fn patch_json(graph: &Graph) -> Value {
	json!({
		"nodes": graph.nodes.iter().map(|node| json!({
			"id": node.id.0,
			"kind": node.kind.id,
			"x": node.position.0,
			"y": node.position.1,
			"params": node.params.iter().map(|(key, value)| (key.to_string(), json!(value))).collect::<serde_json::Map<_, _>>(),
			"points": node.points.iter().map(|point| json!([point.time, point.value, point.bend])).collect::<Vec<_>>(),
			"velocity_points": node.velocity_points.iter().map(|point| json!([point.time, point.value, point.bend])).collect::<Vec<_>>(),
		})).collect::<Vec<_>>(),
		"voice": i_am_patch::compile::voice_group(graph)
			.iter()
			.map(|id| id.0)
			.collect::<Vec<_>>(),
		"edges": graph.edges.iter().map(|edge| json!({
			"from": [edge.from.0 .0, edge.from.1],
			"to": [edge.to.0 .0, edge.to.1],
		})).collect::<Vec<_>>(),
		"mods": graph.modulations.iter().map(|link| json!({
			"from": [link.source.0 .0, link.source.1],
			"to": [link.target.0 .0, link.target.1],
			"depth": link.depth,
			"mode": if link.mode == i_am_patch::graph::ModulationMode::FromBase {
				"base"
			} else {
				"center"
			},
		})).collect::<Vec<_>>(),
	})
}
