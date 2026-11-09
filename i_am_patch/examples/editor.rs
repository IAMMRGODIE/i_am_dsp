//! A tiny editor server: the graph page and the ops behind it.
//!
//! cargo run -p i_am_patch --example editor
//! then open the URL it prints.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use i_am_dsp::prelude::NoteEvent;
use i_am_patch::poly::Poly;
use i_am_patch::scope::Scope;

use i_am_patch::graph::{Graph, NodeId, SignalKind, catalog};
use serde_json::{Value, json};

const PAGE_CSS: &str = concat!(
	include_str!("../../i_am_web_gui/src/page/card.css"),
	"\n",
	include_str!("../src/page/patcher.css"),
);

fn demo() -> Graph {
	let mut graph = Graph::new();
	let midi = graph.add_node("input:midi", (0.0, 120.0)).expect("midi");
	let voice_in = graph.add_node("voice:in", (300.0, 120.0)).expect("voice in");
	let generator = graph.add_node("gen:additive", (560.0, 120.0)).expect("gen");
	let voice_out = graph.add_node("voice:out", (860.0, 120.0)).expect("voice out");
	let filter = graph.add_node("filter", (880.0, 120.0)).expect("filter");
	let lfo = graph.add_node("lfo", (880.0, 580.0)).expect("lfo");
	let out = graph.add_node("out:audio", (1500.0, 120.0)).expect("out");
	graph.connect((midi, 0), (voice_in, 0));
	graph.connect((voice_in, 0), (generator, 0));
	graph.connect((generator, 0), (voice_out, 0));
	graph.connect((voice_out, 0), (filter, 0));
	graph.connect((filter, 0), (out, 0));
	graph.modulate((lfo, 0), (filter, "cutoff"), 800.0);
	graph.set_param(filter, "cutoff", 1_200.0);

	graph
}

fn product_module(kind: &str) -> Option<&str> {
	match kind {
		"filter" => Some("filter"),
		"gain" => Some("gain"),
		_ => kind.strip_prefix("mod:"),
	}
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

	let mut controls: Vec<Value> = Vec::new();
	let mut hero: Option<String> = None;
	let mut mode: Option<&'static str> = None;
	let mut header: &'static [&'static str] = &[];
	let mut visible: Option<Vec<(i32, Vec<&'static str>)>> = None;
	let mut switches: Vec<Value> = Vec::new();
	let mut accent = "#e07a2f";
	let mut face: Option<(f32, f32)> = None;

	if let Some(module) = product_module(kind.id)
		&& let Some(descriptor) = i_am_effects::catalog::get(module)
	{
		let sections = i_am_web_gui::sections(&descriptor.parameters());
		hero = sections.iter().find_map(|section| section.hero.clone());
		mode = descriptor.mode;
		header = descriptor.header;
		visible = descriptor.visible.map(|rows| {
			rows.iter()
				.map(|(mode, names)| (*mode, names.to_vec()))
				.collect()
		});
		accent = descriptor.accent;
		face = i_am_effects::catalog::face(descriptor.id);
		switches = descriptor
			.switches
			.iter()
			.map(|row| json!({ "control": row.control, "on": row.on, "off": row.off }))
			.collect::<Vec<Value>>();

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
		"mode": mode,
		"header": header,
		"visible": visible,
		"switches": switches,
		"accent": accent,
		"face": face.map(|(width, height)| [width, height]),
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
}

/// What the server and the demo audio thread share.
#[derive(Clone)]
struct Shared {
	graph: Arc<Mutex<Graph>>,
	generation: Arc<AtomicU64>,
	rate: usize,
	scope: Arc<Scope>,
	tap: Arc<AtomicUsize>,
}

fn main() {
	let shared = Shared {
		graph: Arc::new(Mutex::new(demo())),
		generation: Arc::new(AtomicU64::new(1)),
		rate: 48_000,
		scope: Arc::new(Scope::new()),
		tap: Arc::new(AtomicUsize::new(0)),
	};
	spawn_audio(shared.clone());

	let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
	println!("http://{}", listener.local_addr().expect("address"));

	for stream in listener.incoming().flatten() {
		handle(stream, &shared);
	}
}

/// Renders the demo graph on a stand-in clock, so the page's scope has audio.
fn spawn_audio(shared: Shared) {
	std::thread::spawn(move || {
		let mut poly: Option<Poly> = None;
		let mut built = 0u64;
		let mut blocks = 0u32;

		loop {
			let generation = shared.generation.load(Ordering::Relaxed);

			if generation != built {
				let graph = shared.graph.lock().expect("graph").clone();

				poly = match Poly::new(&graph, shared.rate) {
					Ok(poly) => Some(poly),
					Err(error) => {
						eprintln!("the demo graph does not play: {error:?}");
						None
					}
				};
				built = generation;
				blocks = 0;
			}

			let mut events = Vec::new();

			// A note a second, so the voice group has something to play.
			if blocks.is_multiple_of(750) {
				events.push(NoteEvent::NoteOn {
					channel: 0,
					note: 60,
					velocity: 0.8,
				});
			}

			if blocks % 750 == 500 {
				events.push(NoteEvent::NoteOff {
					channel: 0,
					note: 60,
					velocity: 0.0,
				});
			}

			let input = [[0.0f32; 2]; i_am_patch::block::BLOCK];
			let mut output = [[0.0f32; 2]; i_am_patch::block::BLOCK];

			if let Some(poly) = &mut poly {
				poly.process(&input, &events, &mut output);

				let tap = shared.tap.load(Ordering::Relaxed);

				if tap != 0
					&& let Some(block) = poly
						.step_of(NodeId(tap as u32))
						.and_then(|step| poly.tap_block(step))
						.or_else(|| poly.tap_voice_block(NodeId(tap as u32)))
				{
					shared.scope.write(&block);
				}
			}

			blocks = blocks.wrapping_add(1);
			std::thread::sleep(std::time::Duration::from_micros(1_300));
		}
	});
}

fn handle(mut stream: TcpStream, shared: &Shared) {
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
		("GET", "/patcher.css") | ("GET", "/kit.css") => ("200 OK", "text/css", PAGE_CSS.to_string()),
		("GET", "/card.css") => ("200 OK", "text/css", i_am_web_gui::CARD_STYLE.to_string()),
		("GET", "/card.js") => ("200 OK", "text/javascript", i_am_web_gui::CARD_SCRIPT.to_string()),
		("GET", "/patcher.js") | ("GET", "/kit.js") => ("200 OK", "text/javascript", include_str!("../src/page/patcher.js").to_string()),
		("GET", "/catalog") => (
			"200 OK",
			"application/json",
			json!(catalog().iter().map(kind_json).collect::<Vec<_>>()).to_string(),
		),
		("GET", "/patch") => (
			"200 OK",
			"application/json",
			patch_json(&shared.graph.lock().expect("graph")).to_string(),
		),
		("GET", "/text") => (
			"200 OK",
			"application/json",
			json!(i_am_patch::text::to_text(&shared.graph.lock().expect("graph"))).to_string(),
		),
		("POST", "/text") => match i_am_patch::text::from_text(body) {
			Ok(graph) => {
				*shared.graph.lock().expect("graph") = graph;
				shared.generation.fetch_add(1, Ordering::Relaxed);
				("200 OK", "application/json", json!({ "ok": true }).to_string())
			}
			Err(error) => (
				"200 OK",
				"application/json",
				json!({ "error": error.to_string() }).to_string(),
			),
		},
		("GET", path) if path.starts_with("/scope") => {
			let node = path
				.split_once('?')
				.and_then(|(_, query)| query.split('&').find_map(|pair| pair.strip_prefix("node=")))
				.and_then(|value| value.parse::<usize>().ok())
				.unwrap_or(0);
			shared.tap.store(node, Ordering::Relaxed);

			(
				"200 OK",
				"application/json",
				json!({
					"rate": shared.rate,
					"samples": shared.scope.read(2048),
				})
				.to_string(),
			)
		}
		("POST", "/op") => ("200 OK", "application/json", op(shared, body).to_string()),
		_ => ("404 Not Found", "text/plain", "not found".to_string()),
	};

	let response = format!(
		"HTTP/1.1 {status}\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\ncache-control: no-store\r\nconnection: close\r\n\r\n{payload}",
		payload.len()
	);
	let _ = stream.write_all(response.as_bytes());
}

fn op(shared: &Shared, body: &str) -> Value {
	let Ok(value) = serde_json::from_str::<Value>(body) else {
		return json!({ "error": "bad json" });
	};
	let result = op_graph(shared, &value);

	if result.get("ok").and_then(Value::as_bool) == Some(true) {
		shared.generation.fetch_add(1, Ordering::Relaxed);
	}

	result
}

fn op_graph(shared: &Shared, value: &Value) -> Value {
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
			let from = (id(&value["from"][0]), value["from"][1].as_u64().unwrap_or(0) as usize);
			let to = (id(&value["to"][0]), value["to"][1].as_u64().unwrap_or(0) as usize);
			graph.connect(from, to);
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
