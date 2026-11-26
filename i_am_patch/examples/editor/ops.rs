//! The ops the page posts, and the edits that reach the running engine.

use super::*;

pub(super) fn op(shared: &Shared, body: &str) -> Value {
	let Ok(value) = serde_json::from_str::<Value>(body) else {
		return json!({ "error": "bad json" });
	};
	let result = op_graph(shared, &value);

	if result.get("ok").and_then(Value::as_bool) == Some(true) {
		// A control that moved is not a graph change: it goes to the engine
		// that is already playing.
		match live_edit(&value) {
			Some(edit) => shared.edits.lock().expect("edits").push(edit),
			None => {
				shared.generation.fetch_add(1, Ordering::Relaxed);
			}
		}
	}

	result
}

/// The edit an op makes without changing the graph's shape, if it is one.
pub(super) fn live_edit(value: &Value) -> Option<i_am_patch::edit::Edit> {
	match value["op"].as_str()? {
		"param" => Some(i_am_patch::edit::Edit::Param {
			node: NodeId(value["id"].as_u64()? as u32),
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

pub(super) fn op_graph(shared: &Shared, value: &Value) -> Value {
	let mut graph = shared.graph.lock().expect("graph");
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

			// The page can name a port a node does not have (a stale card, a
			// hand-written op). Refusing here keeps an unplayable edge out of the
			// graph; the text projection used to panic on one.
			if graph
				.get(from.0)
				.and_then(|node| node.kind.outputs.get(from.1))
				.is_none()
			{
				return json!({ "error": "the source port does not exist" });
			}

			if graph
				.get(to.0)
				.and_then(|node| node.kind.inputs.get(to.1))
				.is_none()
			{
				return json!({ "error": "the target port does not exist" });
			}

			graph.connect(from, to);
			json!({ "ok": true, "replaced": replaced })
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

			let ok = if value["curve"].as_str() == Some("velocity") {
				graph.set_velocity_points(id, points)
			} else {
				graph.set_points(id, points)
			};

			json!({ "ok": ok })
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
