//! What the route table has to answer: the page's calls, the ops and the slots.

use super::*;

/// Every route and op the patcher page asks for.
///
/// The page is the definition of what a person can do to a patch, so it is
/// also the definition of what an agent must be able to do: a control the
/// page offers and the API cannot answer is a hole nothing else would
/// notice. This reads the page's own source and asks the route table for
/// each thing it calls.
#[test]
fn the_api_answers_everything_the_page_asks_for() {
	let page = i_am_patch::PAGE_JS;
	let mut routes: Vec<&str> = Vec::new();
	let mut ops: Vec<&str> = Vec::new();

	for marker in ["fetch(\"", "fetch(`"] {
		for (index, _) in page.match_indices(marker) {
			let rest = &page[index + marker.len()..];
			let end = rest.find(['"', '`', '?', '$']).unwrap_or(0);
			let path = &rest[..end];

			// A path stitched together at run time is not a route this can
			// check; the ones the page really calls are literals.
			if path.starts_with('/') && path.len() > 1 {
				routes.push(path);
			}
		}
	}

	// `snapshot` names an undo step and never sends one: it reads /text
	// into the page's own history stack. Those names are not ops.
	let mut marks: Vec<&str> = Vec::new();
	for (index, _) in page.match_indices("snapshot({ op: \"") {
		let rest = &page[index + 16..];
		let end = rest.find('"').unwrap_or(0);

		if end > 0 {
			marks.push(&rest[..end]);
		}
	}

	for (index, _) in page.match_indices("op: \"") {
		let rest = &page[index + 5..];
		let end = rest.find('"').unwrap_or(0);

		if end > 0 && !marks.contains(&&rest[..end]) {
			ops.push(&rest[..end]);
		}
	}

	routes.sort_unstable();
	routes.dedup();
	ops.sort_unstable();
	ops.dedup();

	println!(
		"the page asks for {} routes and {} ops",
		routes.len(),
		ops.len()
	);

	let shared = PatcherShared::default();
	let mut missing = Vec::new();

	for path in &routes {
		let get = ProtocolRequest {
			method: "GET",
			path,
			query: None,
			body: &[],
		};
		let post = ProtocolRequest {
			method: "POST",
			path,
			query: None,
			body: &[],
		};

		if route(&get, &shared).is_none() && route(&post, &shared).is_none() {
			missing.push(format!("route {path}"));
		}
	}

	for handler in &ops {
		let reply = op(&shared, &format!("{{\"op\":\"{handler}\"}}")).to_string();

		if reply.contains("unknown op") {
			missing.push(format!("op {handler}"));
		}
	}

	assert!(
		missing.is_empty(),
		"the page offers operations the agent's API cannot do: {missing:?}"
	);
}
/// One Save button, two files: the dialog has to offer the extension of the
/// format that was picked, or it writes the bundle out as `.txt`.
#[test]
fn the_save_dialog_offers_the_format_that_was_picked() {
	assert_eq!(save_filter(true), ("Patch bundle", &["iamp"][..]));
	assert_eq!(save_filter(false), ("Patch text", &["txt"][..]));
}

/// The cards that draw an envelope ask the poll for their level, and the answer
/// is one dB per node: the frames stay with the one card the page is watching.
#[test]
fn the_poll_answers_the_level_of_each_card_that_asked() {
	let shared = PatcherShared::default();
	let request = ProtocolRequest {
		method: "GET",
		path: "/scopes",
		query: Some("since=&meters=1,2"),
		body: &[],
	};
	let answer = route(&request, &shared).expect("the scopes route");
	let state: serde_json::Value = serde_json::from_slice(&answer.body).expect("json");
	let meters = state["meters"].as_array().expect("meters");

	assert_eq!(meters.len(), 2);
	assert_eq!(meters[0]["node"], 1);
	assert!(meters[0]["input"].as_f64().is_some(), "{:?}", meters[0]);
	assert!(state["live"].is_null(), "no card is being watched");
}

/// Each scripted effect's card asks its own node: the product's own default
/// source comes back, not one shared with the other scripted products.
#[test]
fn a_scripted_effect_answers_its_own_default() {
	let shared = PatcherShared::default();

	assert!(shared.set_text("node 1 mod:script_mapper 0 0\nnode 2 mod:script_distortion 0 0\n"));

	let source = |node: u32| -> String {
		let query = format!("node={node}");
		let request = ProtocolRequest {
			method: "GET",
			path: "/script",
			query: Some(&query),
			body: &[],
		};
		let answer = route(&request, &shared).expect("the script route");
		let state: serde_json::Value = serde_json::from_slice(&answer.body).expect("json");

		state["source"].as_str().unwrap_or_default().to_string()
	};

	let mapper = source(1);
	let distortion = source(2);

	assert!(!mapper.is_empty(), "the mapper's card got no script");
	assert!(
		!distortion.is_empty(),
		"the distortion's card got no script"
	);
	assert_ne!(mapper, distortion, "both products answered with one script");
}

/// A note is written into the graph, so it saves and undoes with the patch, and
/// it is not a rebuild: the revision must not move under a caret.
#[test]
fn a_note_saves_with_the_patch_and_rebuilds_nothing() {
	use std::sync::atomic::Ordering;

	let shared = PatcherShared::default();
	let before = shared.generation.load(Ordering::Relaxed);
	let node = shared.graph().nodes[0].id.0;
	let body = format!("{{\"op\":\"text\",\"id\":{node},\"text\":\"the low end is folded here\"}}");
	let request = ProtocolRequest {
		method: "POST",
		path: "/op",
		query: None,
		body: body.as_bytes(),
	};
	let answer = route(&request, &shared).expect("the op route owns /op");
	let answer = String::from_utf8(answer.body).expect("utf-8");

	assert!(answer.contains("\"ok\":true"), "{answer}");
	assert_eq!(
		shared.generation.load(Ordering::Relaxed),
		before,
		"a note rebuilt the engine"
	);
	assert!(
		shared.text().contains("the low end is folded here"),
		"the note is not in the patch text"
	);
	assert!(
		patch_json(&shared.graph()).contains("the low end is folded here"),
		"a rebuilt card would not get its note back"
	);
}

use crate::shared::Body;
use i_am_dsp::prelude::NoteEvent;
use i_am_patch::engine::Engine;
use i_am_patch::graph::Graph;
use std::sync::Arc;

/// A 16 bit mono WAV holding one decaying note at 220 Hz.
fn wav(rate: u32) -> Vec<u8> {
	let samples: Vec<f32> = (0..rate)
		.map(|index| {
			let time = index as f32 / rate as f32;
			let decay = (-3.0 * time).exp();
			decay
				* (0.5 * (std::f32::consts::TAU * 220.0 * time).sin()
					+ 0.25 * (std::f32::consts::TAU * 330.0 * time).sin())
		})
		.collect();
	let data = (samples.len() * 2) as u32;
	let mut out = Vec::new();

	out.extend_from_slice(b"RIFF");
	out.extend_from_slice(&(36 + data).to_le_bytes());
	out.extend_from_slice(b"WAVEfmt ");
	out.extend_from_slice(&16u32.to_le_bytes());
	out.extend_from_slice(&1u16.to_le_bytes());
	out.extend_from_slice(&1u16.to_le_bytes());
	out.extend_from_slice(&rate.to_le_bytes());
	out.extend_from_slice(&(rate * 2).to_le_bytes());
	out.extend_from_slice(&2u16.to_le_bytes());
	out.extend_from_slice(&16u16.to_le_bytes());
	out.extend_from_slice(b"data");
	out.extend_from_slice(&data.to_le_bytes());

	for sample in &samples {
		out.extend_from_slice(&((sample.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
	}

	out
}

/// A graph with one Modal node feeding the output.
fn modal_graph() -> (Graph, u32) {
	let mut graph = Graph::new();
	let modal = graph
		.add_node("synth:fitted", (100.0, 0.0))
		.expect("a modal node");
	let out = graph.add_node("out:audio", (200.0, 0.0)).expect("out");

	graph.connect((modal, 0), (out, 0));
	(graph, modal.0)
}

/// Posts one recorded note to a node, the way the card does.
fn post(shared: &PatcherShared, node: u32) -> String {
	let bytes = wav(44_100);
	let query = format!("node={node}&name=note_A4_127.wav");
	let request = ProtocolRequest {
		method: "POST",
		path: "/sample",
		query: Some(&query),
		body: &bytes,
	};

	String::from_utf8(route(&request, shared).expect("the sample route").body).expect("utf-8")
}

/// The whole way a knob goes: the page posts an op, the graph takes it, and
/// the body that is already playing follows without a rebuild.
#[test]
fn a_knob_op_moves_the_running_body() {
	let shared = Arc::new(PatcherShared::default());
	let mut body = Body::new(shared.clone());
	let graph = shared.graph();
	let filter = graph
		.nodes
		.iter()
		.find(|node| node.kind.id == "mod:filter")
		.expect("the demo has a filter");

	body.push_event(&NoteEvent::NoteOn {
		channel: 0,
		note: 60,
		velocity: 1.0,
	});

	let peak = |body: &mut Body, blocks: usize| {
		let mut best = 0.0_f32;

		for _ in 0..blocks {
			for _ in 0..i_am_patch::block::BLOCK {
				let frame = body.process([0.0; 2], [0.0; 2]);
				best = best.max(frame[0].abs());
			}
		}

		best
	};

	peak(&mut body, 40);
	let before = peak(&mut body, 4);

	assert!(before > 0.01, "the demo did not sound: {before}");

	// A cutoff well above the note's partials, the way the knob's drag ends.
	let request_body = format!(
		r#"{{"op":"param","id":{},"key":"cutoff","value":12000}}"#,
		filter.id.0
	);
	let request = ProtocolRequest {
		method: "POST",
		path: "/op",
		query: None,
		body: request_body.as_bytes(),
	};

	route(&request, &shared).expect("the op route owns /op");

	peak(&mut body, 4);
	let after = peak(&mut body, 4);

	assert!(
		after > before * 1.2,
		"{after} did not follow {before}: the op never reached the body"
	);
}

/// A bundle is the one file a patch moves in: the graph text and the payloads the
/// text cannot carry. A plain text file is refused rather than read as a bundle.
#[test]
fn a_bundle_carries_the_graph_and_its_payloads() {
	use std::sync::Arc;

	use i_am_dsp::prelude::SampleData;

	let shared = PatcherShared::default();
	assert!(shared.set_text("node 1 table:shapes 0 0\n"));

	{
		let slot = shared.slots.sample(1);
		let mut slot = slot.lock().expect("the sample slot");

		slot.sample = Some(Arc::new(SampleData {
			name: "kick.wav".to_string(),
			rate: 48_000,
			pcm: vec![0.1, -0.1],
			right: None,
		}));
	}

	// The text goes through the parser and comes back in its own form.
	let text = shared.text();
	let bytes = bundle(&shared);
	assert_eq!(&bytes[..4], b"IAMP");

	// A fresh instance takes it back, text and payload alike.
	let fresh = PatcherShared::default();
	assert!(
		unbundle(&fresh, &bytes).is_ok(),
		"the bundle did not decode"
	);
	assert_eq!(fresh.text(), text);
	assert!(
		fresh
			.slots
			.sample(1)
			.lock()
			.expect("the slot")
			.sample
			.is_some()
	);

	// What a person would save with the text button is not a bundle.
	assert!(unbundle(&fresh, b"node 0 out:audio 0 0\n").is_err());
}

/// A parameter edit has to reach both halves: the graph, which is the state
/// the host saves and the page reads back, and the engine that is playing.
#[test]
fn a_parameter_op_reaches_the_graph_and_the_engine() {
	let shared = PatcherShared::default();
	let graph = shared.graph();
	let filter = graph
		.nodes
		.iter()
		.find(|node| node.kind.id == "mod:filter")
		.expect("the demo has a filter");
	let body = format!(
		r#"{{"op":"param","id":{},"key":"cutoff","value":2500}}"#,
		filter.id.0
	);
	let request = ProtocolRequest {
		method: "POST",
		path: "/op",
		query: None,
		body: body.as_bytes(),
	};

	let response = route(&request, &shared).expect("the op route owns /op");
	let text = String::from_utf8(response.body).expect("utf-8 json");

	assert!(text.contains("\"ok\":true"), "{text}");

	// The graph is what the page reads back and the host saves.
	let stored = shared
		.graph()
		.get(filter.id)
		.expect("the filter is still there")
		.params
		.get("cutoff")
		.copied();

	assert_eq!(stored, Some(2500.0), "the graph kept the old cutoff");

	// And the engine that is playing is told without a rebuild.
	let queued = shared.edits.lock().expect("the edit queue").len();

	assert_eq!(queued, 1, "the engine was never told");
}

/// A control that moves is not a graph change: it goes to the engine that is
/// already playing, because a rebuild re-creates the nodes and cuts the
/// notes in flight.
#[test]
fn a_control_change_does_not_ask_for_a_rebuild() {
	assert_eq!(
		live_edit(r#"{"op":"param","id":3,"key":"oscillator.gain","value":0.4}"#),
		Some(i_am_patch::edit::Edit::Param {
			node: i_am_patch::graph::NodeId(3),
			key: "oscillator.gain".to_string(),
			value: 0.4,
		})
	);

	assert_eq!(
		live_edit(r#"{"op":"mod_depth","index":1,"depth":-0.25}"#),
		Some(i_am_patch::edit::Edit::Depth {
			index: 1,
			depth: -0.25,
		})
	);

	// Anything that changes the shape of the graph still rebuilds.
	assert_eq!(
		live_edit(r#"{"op":"add","kind":"mixer","x":0,"y":0}"#),
		None
	);
	assert_eq!(
		live_edit(r#"{"op":"connect","from":[1,0],"to":[2,0]}"#),
		None
	);
	assert_eq!(live_edit(r#"{"op":"set_points","id":4,"points":[]}"#), None);
}

#[test]
fn a_sample_posted_to_a_node_is_fitted_into_it() {
	let (graph, modal) = modal_graph();
	let shared = PatcherShared::default();

	*shared.graph.lock().expect("the graph lock") = graph;

	let body = post(&shared, modal);
	assert!(body.contains("\"ok\":true"), "{body}");
	assert!(body.contains("\"note\":57"), "{body}");
	assert!(
		shared.slots.fitted(modal).current().is_some(),
		"the fitted table never reached the node"
	);

	let graph = shared.graph();
	let node = graph
		.get(i_am_patch::graph::NodeId(modal))
		.expect("the node");
	assert_eq!(node.params.get("instrument"), Some(&5.0));
	assert_eq!(node.params.get("derive"), Some(&1.0));
}

#[test]
fn a_node_with_a_fitted_sample_plays() {
	let (graph, modal) = modal_graph();
	let shared = PatcherShared::default();

	*shared.graph.lock().expect("the graph lock") = graph;
	post(&shared, modal);

	let graph = shared.graph();
	let note = NoteEvent::NoteOn {
		channel: 0,
		note: 57,
		velocity: 1.0,
	};
	let rendered = Engine::render_graph_with(&graph, 48_000, 0.5, &[(0, note)], &shared.slots)
		.expect("the engine");
	let peak = rendered
		.iter()
		.fold(0.0f32, |peak, frame| peak.max(frame[0].abs()));

	assert!(peak > 1e-4, "the fitted node was silent: {peak}");
}
