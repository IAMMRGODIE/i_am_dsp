//! The routes the patcher page owns: the catalog, the graph and the ops.

use i_am_patch::graph::{CATALOG, Graph, NodeId, SignalKind};
use i_am_plugin::wry::{ProtocolRequest, ProtocolResponse};
use serde_json::{Value, json};

use crate::shared::PatcherShared;

/// Answers a page request, or `None` for the kit's own routes.
pub fn route(request: &ProtocolRequest<'_>, shared: &PatcherShared) -> Option<ProtocolResponse> {
	match (request.method, request.path) {
		("GET", "/catalog") => Some(ProtocolResponse::json(catalog_json())),
		("GET", "/patch") => Some(ProtocolResponse::json(patch_json(&shared.graph()))),
		("POST", "/op") => {
			let body = std::str::from_utf8(request.body).unwrap_or("");
			let result = op(shared, body);

			if result.get("ok").and_then(Value::as_bool) == Some(true) {
				shared.bump();
			}

			Some(ProtocolResponse::json(result.to_string()))
		}
		_ => None,
	}
}

/// Every node kind, with its ports and parameters.
pub fn catalog_json() -> String {
	let port = |port: &i_am_patch::graph::PortDesc| {
		json!({
			"name": port.name,
			"kind": match port.kind {
				SignalKind::Midi => "midi",
				SignalKind::Control => "control",
				SignalKind::Audio(_) => "audio",
			},
			"optional": port.optional,
		})
	};

	let kinds: Vec<Value> = CATALOG
		.iter()
		.map(|kind| {
			// A kind that is a product takes its controls from the product's
			// parameter schema, so the card and the DSP cannot drift apart.
			let mut controls: Vec<Value> = Vec::new();
			let mut hero: Option<String> = None;

			if let Some(module) = product_module(kind.id)
				&& let Some(descriptor) = i_am_effects::catalog::get(module)
			{
				let parameters = descriptor.parameters();
				let sections = i_am_web_gui::sections(&parameters);
				hero = sections.iter().find_map(|section| section.hero.clone());

				for section in &sections {
					for control in &section.controls {
						controls.push(serde_json::to_value(control).unwrap_or(Value::Null));
					}
				}
			}

			json!({
				"id": kind.id,
				"name": kind.name,
				"inputs": kind.inputs.iter().map(port).collect::<Vec<_>>(),
				"outputs": kind.outputs.iter().map(port).collect::<Vec<_>>(),
				"params": kind.params.iter().map(|(key, value)| json!([key, value])).collect::<Vec<_>>(),
				"controls": controls,
				"hero": hero,
			})
		})
		.collect();

	Value::Array(kinds).to_string()
}

/// The effect catalog id of a node kind, when the kind is a product module.
fn product_module(kind: &str) -> Option<&'static str> {
	match kind {
		"filter" => Some("filter"),
		_ => None,
	}
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
		})).collect::<Vec<_>>(),
		"edges": graph.edges.iter().map(|edge| json!({
			"from": [edge.from.0 .0, edge.from.1],
			"to": [edge.to.0 .0, edge.to.1],
		})).collect::<Vec<_>>(),
		"mods": graph.modulations.iter().map(|link| json!({
			"from": [link.source.0 .0, link.source.1],
			"to": [link.target.0 .0, link.target.1],
			"depth": link.depth,
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
			let from = (id(&value["from"][0]), value["from"][1].as_u64().unwrap_or(0) as usize);
			let to = (id(&value["to"][0]), value["to"][1].as_u64().unwrap_or(0) as usize);
			graph.connect(from, to);
			json!({ "ok": true })
		}
		"disconnect" => {
			graph.remove_edge(value["index"].as_u64().unwrap_or(u64::MAX) as usize);
			json!({ "ok": true })
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
			let from = (id(&value["from"][0]), value["from"][1].as_u64().unwrap_or(0) as usize);
			let target = id(&value["to"][0]);
			let key = value["to"][1].as_str().unwrap_or("");
			let depth = value["depth"].as_f64().unwrap_or(0.0) as f32;
			let Some(param) = graph.get(target).and_then(|node| {
				node.kind
					.params
					.iter()
					.find(|(name, _)| *name == key)
					.map(|(name, _)| *name)
			}) else {
				return json!({ "error": "unknown parameter" });
			};

			graph.modulate(from, (target, param), depth);
			json!({ "ok": true })
		}
		other => json!({ "error": format!("unknown op {other}") }),
	}
}
