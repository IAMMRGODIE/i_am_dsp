//! A tiny editor server: the graph page and the ops behind it.
//!
//! cargo run -p i_am_patch --example editor
//! then open the URL it prints.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

use i_am_patch::graph::{CATALOG, Graph, NodeId, SignalKind};
use serde_json::{Value, json};

fn demo() -> Graph {
	let mut graph = Graph::new();
	let midi = graph.add_node("input:midi", (0.0, 120.0)).expect("midi");
	let voice_in = graph.add_node("voice:in", (180.0, 120.0)).expect("voice in");
	let generator = graph.add_node("gen:additive", (360.0, 120.0)).expect("gen");
	let filter = graph.add_node("filter", (540.0, 120.0)).expect("filter");
	let lfo = graph.add_node("lfo", (540.0, 260.0)).expect("lfo");
	let voice_out = graph.add_node("voice:out", (720.0, 120.0)).expect("voice out");
	let out = graph.add_node("out:audio", (900.0, 120.0)).expect("out");
	graph.connect((midi, 0), (voice_in, 0));
	graph.connect((voice_in, 0), (generator, 0));
	graph.connect((generator, 0), (filter, 0));
	graph.connect((filter, 0), (voice_out, 0));
	graph.connect((voice_out, 0), (out, 0));
	graph.modulate((lfo, 0), (filter, "cutoff"), 800.0);
	graph.set_param(filter, "cutoff", 1_200.0);

	graph
}

fn kind_json(kind: &i_am_patch::graph::NodeDesc) -> Value {
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

	json!({
		"id": kind.id,
		"name": kind.name,
		"inputs": kind.inputs.iter().map(port).collect::<Vec<_>>(),
		"outputs": kind.outputs.iter().map(port).collect::<Vec<_>>(),
		"params": kind.params.iter().map(|(key, value)| json!([key, value])).collect::<Vec<_>>(),
	})
}

fn patch_json(graph: &Graph) -> Value {
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
}

fn main() {
	let graph = Arc::new(Mutex::new(demo()));
	let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
	println!("http://{}", listener.local_addr().expect("address"));

	for stream in listener.incoming().flatten() {
		handle(stream, &graph);
	}
}

fn handle(mut stream: TcpStream, graph: &Arc<Mutex<Graph>>) {
	let mut request = Vec::new();
	let mut buffer = [0u8; 4096];

	// Read the head, then exactly the body the head announces.
	loop {
		let read = match stream.read(&mut buffer) {
			Ok(0) | Err(_) => break,
			Ok(read) => read,
		};
		request.extend_from_slice(&buffer[..read]);

		let text = String::from_utf8_lossy(&request);
		if let Some(head_end) = text.find("\r\n\r\n") {
			let head = &text[..head_end];
			let length = head
				.lines()
				.find_map(|line| line.to_ascii_lowercase().strip_prefix("content-length: ").map(|value| value.trim().parse::<usize>().ok()))
				.flatten()
				.unwrap_or(0);

			if request.len() >= head_end + 4 + length {
				break;
			}
		}
	}

	let text = String::from_utf8_lossy(&request).to_string();
	let head_end = text.find("\r\n\r\n").unwrap_or(text.len());
	let head = &text[..head_end];
	let body = &text[(head_end + 4).min(text.len())..];
	let mut lines = head.lines();
	let request_line = lines.next().unwrap_or("");
	let mut parts = request_line.split_whitespace();
	let method = parts.next().unwrap_or("");
	let path = parts.next().unwrap_or("/");

	let (status, content_type, payload) = match (method, path) {
		("GET", "/") => ("200 OK", "text/html; charset=utf-8", include_str!("../src/page/index.html").to_string()),
		("GET", "/patcher.css") | ("GET", "/kit.css") => ("200 OK", "text/css", include_str!("../src/page/patcher.css").to_string()),
		("GET", "/patcher.js") | ("GET", "/kit.js") => ("200 OK", "text/javascript", include_str!("../src/page/patcher.js").to_string()),
		("GET", "/catalog") => (
			"200 OK",
			"application/json",
			json!(CATALOG.iter().map(kind_json).collect::<Vec<_>>()).to_string(),
		),
		("GET", "/patch") => (
			"200 OK",
			"application/json",
			patch_json(&graph.lock().expect("graph")).to_string(),
		),
		("POST", "/op") => ("200 OK", "application/json", op(graph, body).to_string()),
		_ => ("404 Not Found", "text/plain", "not found".to_string()),
	};

	let response = format!(
		"HTTP/1.1 {status}\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\ncache-control: no-store\r\nconnection: close\r\n\r\n{payload}",
		payload.len()
	);
	let _ = stream.write_all(response.as_bytes());
}

fn op(graph: &Arc<Mutex<Graph>>, body: &str) -> Value {
	let Ok(value) = serde_json::from_str::<Value>(body) else {
		return json!({ "error": "bad json" });
	};
	let mut graph = graph.lock().expect("graph");
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
		"remove" => {
			graph.remove_node(id(&value["id"]));
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
			let Some(param) = graph
				.get(target)
				.and_then(|node| node.kind.params.iter().find(|(name, _)| *name == key).map(|(name, _)| *name))
			else {
				return json!({ "error": "unknown parameter" });
			};

			graph.modulate(from, (target, param), depth);
			json!({ "ok": true })
		}
		other => json!({ "error": format!("unknown op {other}") }),
	}
}
