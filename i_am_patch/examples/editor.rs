//! A tiny editor server: the graph page and the ops behind it.
//!
//! cargo run -p i_am_patch --example editor
//! then open the URL it prints.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use i_am_dsp::prelude::NoteEvent;
use i_am_patch::block::AudioBlock;
use i_am_patch::compile::voice_group;
use i_am_patch::engine::Engine;
use i_am_patch::poly::Poly;
use i_am_patch::scope::Scope;

use i_am_patch::graph::{Graph, NodeId};
use serde_json::{Value, json};

const PAGE_CSS: &str = concat!(
	include_str!("../../i_am_web_gui/src/page/card.css"),
	"\n",
	include_str!("../src/page/patcher.css"),
);

fn demo() -> Graph {
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
	graph.modulate((lfo, 0), (filter, "cutoff"), 800.0);
	graph.set_param(filter, "cutoff", 1_200.0);

	graph
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

/// The watched nodes and their rings.
type Taps = Arc<Mutex<Vec<(u32, Arc<Scope>)>>>;

/// What the server and the demo audio thread share.
#[derive(Clone)]
struct Shared {
	graph: Arc<Mutex<Graph>>,
	generation: Arc<AtomicU64>,
	rate: usize,
	taps: Taps,
	phases: Arc<Mutex<Vec<(u32, f32)>>>,
	error: Arc<Mutex<Option<String>>>,
	/// The fitted tables a Modal node plays; the demo graph has none.
	tables: i_am_patch::Tables,
}

impl Shared {
	fn tap_for(&self, node: u32) -> Arc<Scope> {
		let mut taps = self.taps.lock().expect("taps");

		if let Some((_, scope)) = taps.iter().find(|(id, _)| *id == node) {
			return scope.clone();
		}

		let scope = Arc::new(Scope::new());
		taps.push((node, scope.clone()));
		scope
	}

	fn tap_snapshot_into(&self, out: &mut Vec<(u32, Arc<Scope>)>) {
		out.clear();

		if let Ok(taps) = self.taps.try_lock() {
			out.extend(taps.iter().cloned());
		}
	}

	fn set_phase(&self, node: u32, phase: f32) {
		let Ok(mut phases) = self.phases.try_lock() else {
			return;
		};

		if let Some(entry) = phases.iter_mut().find(|(id, _)| *id == node) {
			entry.1 = phase;
		} else {
			phases.push((node, phase));
		}
	}

	fn phase_for(&self, node: u32) -> Option<f32> {
		self.phases
			.lock()
			.expect("phases")
			.iter()
			.find(|(id, _)| *id == node)
			.map(|(_, phase)| *phase)
	}
}

fn main() {
	let shared = Shared {
		graph: Arc::new(Mutex::new(demo())),
		generation: Arc::new(AtomicU64::new(1)),
		rate: 48_000,
		taps: Arc::new(Mutex::new(Vec::new())),
		phases: Arc::new(Mutex::new(Vec::new())),
		error: Arc::new(Mutex::new(None)),
		tables: Default::default(),
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
		let mut engine: Option<Engine> = None;
		let mut built = 0u64;
		let mut blocks = 0u32;
		let mut taps: Vec<(u32, std::sync::Arc<Scope>)> = Vec::new();

		loop {
			let generation = shared.generation.load(Ordering::Relaxed);

			if generation != built {
				let graph = shared.graph.lock().expect("graph").clone();

				// The product plays a graph with a voice group through Poly and
				// one without through the plain engine; the preview does the
				// same, so a patch that never used a voice group still has audio.
				if voice_group(&graph).is_empty() {
					poly = None;
					engine = match Engine::new_with(&graph, shared.rate, &shared.tables) {
						Ok(engine) => {
							*shared.error.lock().expect("error") = None;
							Some(engine)
						}
						Err(error) => {
							eprintln!("the demo graph does not play: {error:?}");
							*shared.error.lock().expect("error") = Some(format!("{error:?}"));
							None
						}
					};
				} else {
					engine = None;
					poly = match Poly::new_with(&graph, shared.rate, &shared.tables) {
						Ok(poly) => {
							*shared.error.lock().expect("error") = None;
							Some(poly)
						}
						Err(error) => {
							eprintln!("the demo graph does not play: {error:?}");
							*shared.error.lock().expect("error") = Some(format!("{error:?}"));
							None
						}
					};
				}
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
				poly.set_transport(
					Some(120.0),
					Some(
						blocks as f32 * i_am_patch::block::BLOCK as f32 / shared.rate as f32 * 2.0,
					),
				);
				poly.process(&input, &events, &mut output);

				shared.tap_snapshot_into(&mut taps);

				for (node, scope) in &taps {
					if let Some(block) = poly
						.step_of(NodeId(*node))
						.and_then(|step| poly.tap_block(step))
						.or_else(|| poly.tap_voice_block(NodeId(*node)))
					{
						scope.write(&block);
					}

					if let Some(phase) = poly.control_phase(NodeId(*node)) {
						shared.set_phase(*node, phase);
					}
				}
			} else if let Some(engine) = &mut engine {
				engine.set_transport(
					Some(120.0),
					Some(
						blocks as f32 * i_am_patch::block::BLOCK as f32 / shared.rate as f32 * 2.0,
					),
				);
				engine.process(&input, &events, &mut output);

				shared.tap_snapshot_into(&mut taps);

				for (node, scope) in &taps {
					let id = NodeId(*node);
					let block: Option<AudioBlock> =
						engine.step_of(id).and_then(|step| engine.tap_block(step));

					if let Some(block) = block {
						scope.write(&block);
					}

					if let Some(phase) = engine
						.step_of(id)
						.and_then(|step| engine.control_phase(step))
					{
						shared.set_phase(*node, phase);
					}
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
				.find_map(|line| {
					line.to_ascii_lowercase()
						.strip_prefix("content-length: ")
						.map(|value| value.trim().parse::<usize>().ok())
				})
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
	// The header is ASCII, so its length is the same in bytes; the body is not
	// text at all when it is a WAV, so it is sliced from the raw request.
	let body_start = (head_end + 4).min(request.len());
	let body_bytes: &[u8] = &request[body_start..];
	let body = &text[body_start.min(text.len())..];
	let mut lines = head.lines();
	let request_line = lines.next().unwrap_or("");
	let mut parts = request_line.split_whitespace();
	let method = parts.next().unwrap_or("");
	let path = parts.next().unwrap_or("/");

	let (status, content_type, payload) = match (method, path) {
		("GET", "/") => (
			"200 OK",
			"text/html; charset=utf-8",
			include_str!("../src/page/index.html").to_string(),
		),
		("GET", "/patcher.css") | ("GET", "/kit.css") => {
			("200 OK", "text/css", PAGE_CSS.to_string())
		}
		("GET", "/card.css") => ("200 OK", "text/css", i_am_web_gui::CARD_STYLE.to_string()),
		("GET", "/card.js") => (
			"200 OK",
			"text/javascript",
			i_am_web_gui::CARD_SCRIPT.to_string(),
		),
		("GET", "/patcher.js") | ("GET", "/kit.js") => (
			"200 OK",
			"text/javascript",
			include_str!("../src/page/patcher.js").to_string(),
		),
		("GET", "/catalog") => (
			"200 OK",
			"application/json",
			json!(i_am_patch::catalog::kinds_json()).to_string(),
		),
		("GET", "/patch") => (
			"200 OK",
			"application/json",
			patch_json(&shared.graph.lock().expect("graph")).to_string(),
		),
		("GET", "/status") => (
			"200 OK",
			"application/json",
			json!({ "error": shared.error.lock().expect("error").clone() }).to_string(),
		),
		("GET", "/text") => (
			"200 OK",
			"application/json",
			json!(i_am_patch::text::to_text(
				&shared.graph.lock().expect("graph")
			))
			.to_string(),
		),
		("POST", "/text") => match i_am_patch::text::from_text(body) {
			Ok(graph) => {
				*shared.graph.lock().expect("graph") = graph;
				shared.generation.fetch_add(1, Ordering::Relaxed);
				(
					"200 OK",
					"application/json",
					json!({ "ok": true }).to_string(),
				)
			}
			Err(error) => (
				"200 OK",
				"application/json",
				json!({ "error": error.to_string() }).to_string(),
			),
		},
		("GET", path) if path == "/scope" || path.starts_with("/scope?") => {
			let node = path
				.split_once('?')
				.and_then(|(_, query)| query.split('&').find_map(|pair| pair.strip_prefix("node=")))
				.and_then(|value| value.parse::<usize>().ok())
				.unwrap_or(0);
			let scope = shared.tap_for(node as u32);

			(
				"200 OK",
				"application/json",
				json!({
					"rate": shared.rate,
					"samples": scope.read(2048),
				})
				.to_string(),
			)
		}
		("GET", "/scopes") => {
			let graph = shared.graph.lock().expect("graph");
			let mut scopes = Vec::new();

			for node in &graph.nodes {
				if !matches!(
					node.kind.id,
					"lfo" | "envelope" | "random" | "table" | "pitch" | "midi"
				) {
					continue;
				}

				let scope = shared.tap_for(node.id.0);
				scopes.push(json!({
					"node": node.id.0,
					"samples": scope.read(64),
					"phase": shared.phase_for(node.id.0),
				}));
			}

			(
				"200 OK",
				"application/json",
				json!({ "rate": shared.rate, "scopes": scopes }).to_string(),
			)
		}
		("POST", "/op") => ("200 OK", "application/json", op(shared, body).to_string()),
		("POST", path) if path == "/sample" || path.starts_with("/sample?") => (
			"200 OK",
			"application/json",
			sample_route(shared, path, body_bytes).to_string(),
		),
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

/// Fits one recorded note into a Modal node, the way the product's route does.
fn sample_route(shared: &Shared, path: &str, body: &[u8]) -> Value {
	let query = path.split_once('?').map(|(_, query)| query).unwrap_or("");
	let node = query
		.split('&')
		.find_map(|pair| pair.strip_prefix("node="))
		.and_then(|value| value.parse::<u32>().ok())
		.unwrap_or(0);
	let name = query
		.split('&')
		.find_map(|pair| pair.strip_prefix("name="))
		.unwrap_or("sample");

	match i_am_dsp::prelude::fit_sample(body, i_am_dsp::prelude::note_of_name(name)) {
		Ok(fitted) => {
			i_am_patch::nodes::table_for(&shared.tables, node).set(
				i_am_dsp::prelude::ImportedTable {
					name: name.to_string(),
					bytes: fitted.bytes,
				},
			);

			if let Ok(mut graph) = shared.graph.lock() {
				let id = i_am_patch::graph::NodeId(node);

				graph.set_param(id, "instrument", 5.0);
				graph.set_param(id, "derive", 1.0);
				graph.set_param(id, "transpose", 0.0);
			}

			shared.generation.fetch_add(1, Ordering::Relaxed);

			json!({
				"ok": true,
				"note": fitted.note,
				"frequency": fitted.frequency,
				"modes": fitted.modes,
			})
		}
		Err(error) => json!({ "ok": false, "error": error }),
	}
}
