//! The routes the patcher page owns: the catalog, the graph and the ops.

use i_am_patch::graph::{Graph, NodeId};
use i_am_plugin::wry::{ProtocolRequest, ProtocolResponse};
use serde_json::{Value, json};

use crate::shared::PatcherShared;

/// Answers a page request, or `None` for the kit's own routes.
pub fn route(request: &ProtocolRequest<'_>, shared: &PatcherShared) -> Option<ProtocolResponse> {
	match (request.method, request.path) {
		("GET", "/catalog") => Some(ProtocolResponse::json(catalog_json())),
		("GET", "/patch") => Some(ProtocolResponse::json(patch_json(&shared.graph()))),
		("GET", "/status") => Some(ProtocolResponse::json(
			json!({ "error": shared.error() }).to_string(),
		)),
		("POST", "/clear") => {
			let node = request
				.query
				.unwrap_or("")
				.split('&')
				.find_map(|pair| pair.strip_prefix("node="))
				.and_then(|value| value.parse::<u32>().ok())
				.unwrap_or(0);

			if let Ok(mut clears) = shared.clear_requests.lock() {
				clears.push(node);
			}

			Some(ProtocolResponse::json(json!({ "ok": true }).to_string()))
		}
		("GET", "/script") | ("POST", "/script") => {
			let node = request
				.query
				.unwrap_or("")
				.split('&')
				.find_map(|pair| pair.strip_prefix("node="))
				.and_then(|value| value.parse::<u32>().ok())
				.unwrap_or(0);
			let scripts = shared.scripts.lock().expect("the script lock");

			if request.method == "POST" {
				let body = std::str::from_utf8(request.body).unwrap_or("");

				if let Some(script) = scripts.get(&node) {
					script.set_script_source(body.to_string());
				}

				Some(ProtocolResponse::json(json!({ "ok": true }).to_string()))
			} else {
				let source = scripts
					.get(&node)
					.map(|script| script.script_source())
					.unwrap_or_default();

				Some(ProtocolResponse::json(
					json!({ "source": source }).to_string(),
				))
			}
		}
		("GET", "/sampler") | ("POST", "/sampler") => Some(ProtocolResponse::json(
			sampler_route(request, shared).to_string(),
		)),
		("GET", "/waves") | ("POST", "/waves") => Some(ProtocolResponse::json(
			waves_route(request, shared).to_string(),
		)),
		("GET", "/ir") | ("POST", "/ir") => Some(ProtocolResponse::json(ir_route(request, shared))),
		("POST", "/sample") => {
			// One recorded note, fitted where it is loaded: the table goes into
			// the node's slot, because a graph parameter holds a number and a
			// serialized table cannot travel through it.
			let node = request
				.query
				.unwrap_or("")
				.split('&')
				.find_map(|pair| pair.strip_prefix("node="))
				.and_then(|value| value.parse::<u32>().ok())
				.unwrap_or(0);
			let name = request
				.query
				.unwrap_or("")
				.split('&')
				.find_map(|pair| pair.strip_prefix("name="))
				.map(i_am_web_gui::percent_decode)
				.filter(|name| !name.is_empty())
				.unwrap_or_else(|| "sample".to_string());
			let note = request
				.query
				.unwrap_or("")
				.split('&')
				.find_map(|pair| pair.strip_prefix("note="))
				.and_then(|value| value.parse::<u8>().ok())
				.or_else(|| i_am_dsp::prelude::note_of_name(&name));

			match i_am_dsp::prelude::fit_sample(request.body, note) {
				Ok(fitted) => {
					shared
						.slots
						.fitted(node)
						.set(i_am_dsp::prelude::ImportedTable {
							name,
							bytes: fitted.bytes,
						});

					if let Ok(mut graph) = shared.graph.lock() {
						let id = NodeId(node);

						graph.set_param(id, "instrument", 5.0);
						graph.set_param(id, "derive", 1.0);
						graph.set_param(id, "transpose", 0.0);
					}

					shared.bump();

					Some(ProtocolResponse::json(
						json!({
							"ok": true,
							"note": fitted.note,
							"frequency": fitted.frequency,
							"modes": fitted.modes,
						})
						.to_string(),
					))
				}
				Err(error) => Some(ProtocolResponse::json(
					json!({ "ok": false, "error": error }).to_string(),
				)),
			}
		}
		("GET", "/text") => Some(ProtocolResponse::json(json!(shared.text()).to_string())),
		("POST", "/text") => {
			let body = std::str::from_utf8(request.body).unwrap_or("");
			let ok = shared.set_text(body);

			Some(ProtocolResponse::json(json!({ "ok": ok }).to_string()))
		}
		("GET", "/scope") => {
			use std::sync::atomic::Ordering;

			let node = request
				.query
				.unwrap_or("")
				.split('&')
				.find_map(|pair| pair.strip_prefix("node="))
				.and_then(|value| value.parse::<usize>().ok())
				.unwrap_or(0);
			let which = request
				.query
				.unwrap_or("")
				.split('&')
				.find_map(|pair| pair.strip_prefix("which="))
				.unwrap_or("output");
			let (output, input) = shared.taps_for(node as u32);
			let scope = if which == "input" { input } else { output };

			Some(ProtocolResponse::json(
				json!({
					"rate": shared.rate.load(Ordering::Relaxed),
					"samples": scope.read(2048),
				})
				.to_string(),
			))
		}
		("GET", "/scopes") => {
			use std::sync::atomic::Ordering;

			// The page only needs the latest value of each source, so the graph is
			// read for its ids and released; cloning it here ran a graph copy on
			// every poll.
			let sources: Vec<u32> = {
				let graph = shared.graph.lock().expect("the graph lock");

				graph
					.nodes
					.iter()
					.filter(|node| {
						matches!(
							node.kind.id,
							"lfo" | "envelope" | "random" | "table" | "pitch" | "midi"
						)
					})
					.map(|node| node.id.0)
					.collect()
			};

			let mut scopes = Vec::new();

			for node in sources {
				let (scope, _) = shared.taps_for(node);
				scopes.push(json!({
					"node": node,
					"samples": scope.read(64),
					"phase": shared.phase_for(node),
				}));
			}

			Some(ProtocolResponse::json(
				json!({
					"rate": shared.rate.load(Ordering::Relaxed),
					"scopes": scopes,
				})
				.to_string(),
			))
		}
		("POST", "/op") => {
			let body = std::str::from_utf8(request.body).unwrap_or("");
			let result = op(shared, body);

			if result.get("ok").and_then(Value::as_bool) == Some(true) {
				// A control that moved is not a graph change: it goes to the
				// engine that is already playing, so the notes in flight keep
				// sounding under the hand.
				match live_edit(body) {
					Some(edit) => shared.edit(edit),
					None => shared.bump(),
				}
			}

			Some(ProtocolResponse::json(result.to_string()))
		}
		_ => None,
	}
}

/// The edit an op makes without changing the graph's shape, if it is one.
///
/// Everything else — a node, a wire, a curve, a new modulation — needs the
/// engine rebuilt, and a rebuild is what cuts the notes.
fn live_edit(body: &str) -> Option<i_am_patch::edit::Edit> {
	let value: Value = serde_json::from_str(body).ok()?;

	match value["op"].as_str()? {
		"param" => Some(i_am_patch::edit::Edit::Param {
			node: i_am_patch::graph::NodeId(value["id"].as_u64()? as u32),
			key: value["key"].as_str()?.to_string(),
			value: value["value"].as_f64()? as f32,
		}),
		"mod_depth" => Some(i_am_patch::edit::Edit::Depth {
			index: value["index"].as_u64()? as usize,
			depth: value["depth"].as_f64()? as f32,
		}),
		"mod_mode" => Some(i_am_patch::edit::Edit::Mode {
			index: value["index"].as_u64()? as usize,
			mode: if value["mode"].as_str() == Some("base") {
				i_am_patch::graph::ModulationMode::FromBase
			} else {
				i_am_patch::graph::ModulationMode::Centered
			},
		}),
		_ => None,
	}
}

/// The sample a sampler node plays: a GET describes it, a POST loads a WAV.
///
/// The slot is the page's, so the import reaches the voice the way the
/// product's own page does, without a rebuild.
fn sampler_route(request: &ProtocolRequest<'_>, shared: &PatcherShared) -> String {
	let slot = shared.slots.sample(query_node(request));

	if request.method == "GET" {
		return i_am_patch::sample_preview(&slot);
	}

	// The sampler hero's Slice button works on what the slot already holds, and
	// its gestures add, move and take marks off.
	if let Some(action) = query_value(request, "action") {
		let mode = query_value(request, "mode").unwrap_or("grid");
		let at = query_value(request, "at")
			.and_then(|value| value.parse::<f32>().ok())
			.unwrap_or(0.0);
		let to = query_value(request, "to")
			.and_then(|value| value.parse::<f32>().ok())
			.unwrap_or(at);

		return match action {
			"slices" => match i_am_patch::generate_slices(&slot, mode) {
				Ok(count) => json!({ "ok": true, "slices": count }).to_string(),
				Err(error) => json!({ "ok": false, "error": error }).to_string(),
			},
			_ => match i_am_patch::edit_slices(&slot, action, at, to) {
				Ok(count) => json!({ "ok": true, "slices": count }).to_string(),
				Err(error) => json!({ "ok": false, "error": error }).to_string(),
			},
		};
	}

	let name = query_name(request, "sample");

	match i_am_patch::import_sample(&slot, request.body, &name) {
		Ok((name, samples, seconds)) => json!({
			"ok": true,
			"name": name,
			"samples": samples,
			"seconds": seconds,
		})
		.to_string(),
		Err(error) => json!({ "ok": false, "error": error }).to_string(),
	}
}

/// The wavetable a shapes node plays: a GET describes it, a POST loads a WAV.
fn waves_route(request: &ProtocolRequest<'_>, shared: &PatcherShared) -> String {
	let slot = shared.slots.waves(query_node(request));

	if request.method == "GET" {
		return i_am_patch::waves_preview(&slot);
	}

	// The shape hero's buttons edit the table in place; an upload replaces it.
	if let Some(action) = query_value(request, "action") {
		let frame = query_value(request, "frame").and_then(|frame| frame.parse::<usize>().ok());

		return match i_am_patch::edit_waves(&slot, action, frame) {
			Ok(count) => json!({ "ok": true, "count": count }).to_string(),
			Err(error) => json!({ "ok": false, "error": error }).to_string(),
		};
	}

	let name = query_name(request, "table");

	match i_am_patch::import_waves(&slot, request.body, &name, query_value(request, "mode")) {
		Ok((name, frames)) => json!({ "ok": true, "name": name, "frames": frames }).to_string(),
		Err(error) => json!({ "ok": false, "error": error }).to_string(),
	}
}

/// The response a convolver node plays: a GET describes it, a POST loads one.
fn ir_route(request: &ProtocolRequest<'_>, shared: &PatcherShared) -> String {
	let slot = shared.slots.ir(query_node(request));

	if request.method == "GET" {
		return slot.json();
	}

	// Dragging the dots on the picture moves the fades; a file replaces the lot.
	if let (Some(fade_in), Some(fade_out)) = (
		query_value(request, "fade_in").and_then(|value| value.parse::<f32>().ok()),
		query_value(request, "fade_out").and_then(|value| value.parse::<f32>().ok()),
	) {
		slot.set_fades(i_am_effects::IrFades {
			fade_in: fade_in.clamp(0.0, 1.0),
			fade_out: fade_out.clamp(0.0, 1.0),
		});

		return json!({ "ok": true }).to_string();
	}

	let name = query_name(request, "impulse");

	match i_am_effects::decode_ir(request.body, slot.host_rate(), name) {
		Ok(ir) => {
			slot.set(std::sync::Arc::new(ir));
			json!({ "ok": true }).to_string()
		}
		Err(error) => {
			slot.fail(error.clone());
			json!({ "ok": false, "error": error }).to_string()
		}
	}
}

/// The node a route was asked about.
fn query_node(request: &ProtocolRequest<'_>) -> u32 {
	request
		.query
		.unwrap_or("")
		.split('&')
		.find_map(|pair| pair.strip_prefix("node="))
		.and_then(|value| value.parse::<u32>().ok())
		.unwrap_or(0)
}

/// One value out of a query string like name=a%20b&action=slices.
fn query_value<'a>(request: &'a ProtocolRequest<'_>, key: &str) -> Option<&'a str> {
	request.query?.split('&').find_map(|pair| {
		let (name, value) = pair.split_once('=')?;

		(name == key).then_some(value)
	})
}

/// The name a page sent with a file, decoded.
fn query_name(request: &ProtocolRequest<'_>, fallback: &str) -> String {
	query_value(request, "name")
		.map(i_am_web_gui::percent_decode)
		.filter(|name| !name.is_empty())
		.unwrap_or_else(|| fallback.to_string())
}

/// Every node kind, with its ports and parameters.
pub fn catalog_json() -> String {
	serde_json::json!(i_am_patch::catalog::kinds_json()).to_string()
}

/// The graph as the page reads it.
pub fn patch_json(graph: &Graph) -> String {
	json!({
		"nodes": graph.nodes.iter().map(|node| json!({
			"id": node.id.0,
			"kind": node.kind.id,
			"x": node.position.0,
			"y": node.position.1,
			"params": node.params.iter().map(|(key, value)| (key.to_string(), json!(value))).collect::<serde_json::Map<_, _>>(),
			"points": node.points.iter().map(|point| json!([point.time, point.value, point.bend])).collect::<Vec<_>>(),
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
	.to_string()
}

fn op(shared: &PatcherShared, body: &str) -> Value {
	let Ok(value) = serde_json::from_str::<Value>(body) else {
		return json!({ "error": "bad json" });
	};
	let mut graph = shared.graph.lock().expect("the graph lock");
	let id = |value: &Value| NodeId(value.as_u64().unwrap_or(u64::MAX) as u32);

	match value["op"].as_str().unwrap_or("") {
		"add" => {
			let kind = value["kind"].as_str().unwrap_or("");
			let x = value["x"].as_f64().unwrap_or(0.0) as f32;
			let y = value["y"].as_f64().unwrap_or(0.0) as f32;

			match graph.add_node(kind, (x, y)) {
				Some(id) => json!({ "ok": true, "id": id.0 }),
				None => json!({ "error": "unknown kind" }),
			}
		}
		"move" => {
			let id = id(&value["id"]);

			if let Some(node) = graph.nodes.iter_mut().find(|node| node.id == id) {
				node.position.0 = value["x"].as_f64().unwrap_or(0.0) as f32;
				node.position.1 = value["y"].as_f64().unwrap_or(0.0) as f32;
			}

			json!({ "ok": true })
		}
		"connect" => {
			let from = (
				id(&value["from"][0]),
				value["from"][1].as_u64().unwrap_or(0) as usize,
			);
			let to = (
				id(&value["to"][0]),
				value["to"][1].as_u64().unwrap_or(0) as usize,
			);
			let mut replaced = 0;

			// A port takes one wire from the page: an older one into the same
			// input gives way, so the drag does not stack duplicates.
			let mut index = 0;

			while index < graph.edges.len() {
				if graph.edges[index].to == to {
					graph.remove_edge(index);
					replaced += 1;
				} else {
					index += 1;
				}
			}

			graph.connect(from, to);
			json!({ "ok": true, "replaced": replaced })
		}
		"disconnect" => {
			graph.remove_edge(value["index"].as_u64().unwrap_or(u64::MAX) as usize);
			json!({ "ok": true })
		}
		"set_points" => {
			let id = id(&value["id"]);
			let mut points = Vec::new();

			if let Some(items) = value["points"].as_array() {
				for item in items {
					points.push(i_am_patch::graph::Point {
						time: item[0].as_f64().unwrap_or(0.0) as f32,
						value: item[1].as_f64().unwrap_or(0.0) as f32,
						bend: item[2].as_f64().unwrap_or(0.0) as f32,
					});
				}
			}

			json!({ "ok": graph.set_points(id, points) })
		}
		"mod_mode" => {
			let done = graph.set_modulation_mode(
				value["index"].as_u64().unwrap_or(u64::MAX) as usize,
				if value["mode"].as_str() == Some("base") {
					i_am_patch::graph::ModulationMode::FromBase
				} else {
					i_am_patch::graph::ModulationMode::Centered
				},
			);
			json!({ "ok": done })
		}
		"mod_depth" => {
			let done = graph.set_modulation_depth(
				value["index"].as_u64().unwrap_or(u64::MAX) as usize,
				value["depth"].as_f64().unwrap_or(0.0) as f32,
			);
			json!({ "ok": done })
		}
		"unmodulate" => {
			graph.remove_modulation(value["index"].as_u64().unwrap_or(u64::MAX) as usize);
			json!({ "ok": true })
		}
		"duplicate" => {
			let source = id(&value["id"]);
			let Some(node) = graph.get(source).cloned() else {
				return json!({ "error": "no node" });
			};
			let new_id = graph.add_node(
				node.kind.id,
				(node.position.0 + 40.0, node.position.1 + 40.0),
			);

			if let Some(new_id) = new_id {
				for (key, value) in &node.params {
					graph.set_param(new_id, key, *value);
				}
			}

			json!({ "ok": true, "id": new_id.map(|id| id.0) })
		}
		"replace" => {
			let kind = value["kind"].as_str().unwrap_or("");
			json!({ "ok": graph.replace_kind(id(&value["id"]), kind) })
		}
		"remove" => {
			graph.remove_node(id(&value["id"]));
			json!({ "ok": true })
		}
		"param" => {
			let id = id(&value["id"]);
			let key = value["key"].as_str().unwrap_or("");
			let done = graph.set_param(id, key, value["value"].as_f64().unwrap_or(0.0) as f32);
			json!({ "ok": done })
		}
		"modulate" => {
			let from = (
				id(&value["from"][0]),
				value["from"][1].as_u64().unwrap_or(0) as usize,
			);
			let target = id(&value["to"][0]);
			let key = value["to"][1].as_str().unwrap_or("");
			let depth = value["depth"].as_f64().unwrap_or(0.0) as f32;
			let Some(param) = graph.get(target).and_then(|node| {
				node.kind
					.params
					.iter()
					.find(|p| p.name == key)
					.map(|p| p.name)
			}) else {
				return json!({ "error": "unknown parameter" });
			};

			graph.modulate(from, (target, param), depth);
			json!({ "ok": true })
		}
		other => json!({ "error": format!("unknown op {other}") }),
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use i_am_dsp::prelude::NoteEvent;
	use i_am_patch::engine::Engine;
	use i_am_patch::graph::Graph;

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
}
