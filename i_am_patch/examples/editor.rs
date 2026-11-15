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
	// A depth is a share of the cutoff's own range, so a third of it is a sweep
	// of about two octaves either way.
	graph.modulate((lfo, 0), (filter, "cutoff"), 0.33);
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

/// The watched nodes and their rings.
type Taps = Arc<Mutex<Vec<(u32, Arc<Scope>, Arc<Scope>)>>>;

/// What the server and the demo audio thread share.
#[derive(Clone)]
struct Shared {
	graph: Arc<Mutex<Graph>>,
	generation: Arc<AtomicU64>,
	rate: usize,
	taps: Taps,
	phases: Arc<Mutex<Vec<(u32, f32)>>>,
	midi_inputs: Arc<Mutex<Vec<(u32, f32)>>>,
	error: Arc<Mutex<Option<String>>>,
	/// What the page handed the nodes; the demo graph has none.
	slots: i_am_patch::Slots,
	/// The control edits that reach the running engine without a rebuild.
	edits: Arc<Mutex<Vec<i_am_patch::edit::Edit>>>,
}

impl Shared {
	/// The rings for a watched node: output and input.
	fn tap_for(&self, node: u32) -> (Arc<Scope>, Arc<Scope>) {
		let mut taps = self.taps.lock().expect("taps");

		if let Some((_, scope, input)) = taps.iter().find(|(id, _, _)| *id == node) {
			return (scope.clone(), input.clone());
		}

		let scope = Arc::new(Scope::new());
		let input = Arc::new(Scope::new());
		taps.push((node, scope.clone(), input.clone()));
		(scope, input)
	}

	fn tap_snapshot_into(&self, out: &mut Vec<(u32, Arc<Scope>, Arc<Scope>)>) {
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

	fn set_midi_input(&self, node: u32, position: f32) {
		let Ok(mut inputs) = self.midi_inputs.try_lock() else {
			return;
		};

		if let Some(entry) = inputs.iter_mut().find(|(id, _)| *id == node) {
			entry.1 = position;
		} else {
			inputs.push((node, position));
		}
	}

	fn midi_input_for(&self, node: u32) -> Option<f32> {
		self.midi_inputs
			.lock()
			.expect("midi inputs")
			.iter()
			.find(|(id, _)| *id == node)
			.map(|(_, position)| *position)
	}
}

fn main() {
	let shared = Shared {
		graph: Arc::new(Mutex::new(demo())),
		generation: Arc::new(AtomicU64::new(1)),
		rate: 48_000,
		taps: Arc::new(Mutex::new(Vec::new())),
		phases: Arc::new(Mutex::new(Vec::new())),
		midi_inputs: Arc::new(Mutex::new(Vec::new())),
		error: Arc::new(Mutex::new(None)),
		slots: Default::default(),
		edits: Arc::new(Mutex::new(Vec::new())),
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
		let mut taps: Vec<(u32, std::sync::Arc<Scope>, std::sync::Arc<Scope>)> = Vec::new();

		loop {
			let generation = shared.generation.load(Ordering::Relaxed);

			if generation != built {
				// The new engine is built from the graph, which already holds the
				// queued edits, so they are dropped rather than replayed.
				if let Ok(mut edits) = shared.edits.try_lock() {
					edits.clear();
				}

				let graph = shared.graph.lock().expect("graph").clone();

				// The product plays a graph with a voice group through Poly and
				// one without through the plain engine; the preview does the
				// same, so a patch that never used a voice group still has audio.
				if voice_group(&graph).is_empty() {
					poly = None;
					engine = match Engine::new_with(&graph, shared.rate, &shared.slots) {
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
					poly = match Poly::new_with(&graph, shared.rate, &shared.slots) {
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

			// The edits a hand made since the last block reach the engine that
			// is already playing, so a knob does not cut the notes.
			if let Ok(mut edits) = shared.edits.try_lock() {
				for edit in edits.drain(..) {
					if let Some(poly) = &mut poly {
						edit.apply_poly(poly);
					} else if let Some(engine) = &mut engine {
						edit.apply(engine);
					}
				}
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

				for (node, scope, input) in &taps {
					if let Some(block) = poly
						.step_of(NodeId(*node))
						.and_then(|step| poly.tap_block(step))
						.or_else(|| poly.tap_voice_block(NodeId(*node)))
					{
						scope.write(&block);
					}

					if let Some(block) = poly
						.step_of(NodeId(*node))
						.and_then(|step| poly.input_block(step))
					{
						input.write(&block);
					}

					if let Some(phase) = poly.control_phase(NodeId(*node)) {
						shared.set_phase(*node, phase);
					}

					if let Some(position) = poly.midi_position(NodeId(*node)) {
						shared.set_midi_input(*node, position);
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

				for (node, scope, input) in &taps {
					let id = NodeId(*node);
					let block: Option<AudioBlock> =
						engine.step_of(id).and_then(|step| engine.tap_block(step));

					if let Some(block) = block {
						scope.write(&block);
					}

					if let Some(block) =
						engine.step_of(id).and_then(|step| engine.input_block(step))
					{
						input.write(&block);
					}

					if let Some(phase) = engine
						.step_of(id)
						.and_then(|step| engine.control_phase(step))
					{
						shared.set_phase(*node, phase);
					}

					if let Some(position) = engine
						.step_of(id)
						.and_then(|step| engine.midi_position(step))
					{
						shared.set_midi_input(*node, position);
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
		("GET", "/state") => (
			"200 OK",
			"application/json",
			json!({
				"revision": shared.generation.load(Ordering::Relaxed),
				"error": shared.error.lock().expect("error").clone(),
			})
			.to_string(),
		),
		("GET", "/text") => (
			"200 OK",
			"application/json",
			json!(i_am_patch::text::to_text(
				&shared.graph.lock().expect("graph")
			))
			.to_string(),
		),
		("POST", "/text") => match i_am_patch::text::from_text_with_slots(body, &shared.slots) {
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
		("GET", path) if path == "/canvas" || path.starts_with("/canvas?") => {
			let node = path
				.split_once('?')
				.and_then(|(_, query)| query.split('&').find_map(|pair| pair.strip_prefix("node=")))
				.and_then(|value| value.parse::<u32>().ok())
				.unwrap_or(0);

			(
				"200 OK",
				"application/json",
				shared.slots.canvas(node).shape_json(),
			)
		}
		("POST", path) if path == "/canvas" || path.starts_with("/canvas?") => {
			let node = path
				.split_once('?')
				.and_then(|(_, query)| query.split('&').find_map(|pair| pair.strip_prefix("node=")))
				.and_then(|value| value.parse::<u32>().ok())
				.unwrap_or(0);

			(
				"200 OK",
				"application/json",
				shared.slots.canvas(node).set_shape(body.to_string()),
			)
		}
		("GET", path) if path == "/spectrum" || path.starts_with("/spectrum?") => {
			let node = path
				.split_once('?')
				.and_then(|(_, query)| query.split('&').find_map(|pair| pair.strip_prefix("node=")))
				.and_then(|value| value.parse::<u32>().ok())
				.unwrap_or(0);

			(
				"200 OK",
				"application/json",
				shared.slots.spectrum(node).state_json(),
			)
		}
		("POST", path) if path == "/spectrum" || path.starts_with("/spectrum?") => {
			let node = path
				.split_once('?')
				.and_then(|(_, query)| query.split('&').find_map(|pair| pair.strip_prefix("node=")))
				.and_then(|value| value.parse::<u32>().ok())
				.unwrap_or(0);

			(
				"200 OK",
				"application/json",
				shared.slots.spectrum(node).set_json(body),
			)
		}
		("GET", path) if path == "/click" || path.starts_with("/click?") => {
			let node = path
				.split_once('?')
				.and_then(|(_, query)| query.split('&').find_map(|pair| pair.strip_prefix("node=")))
				.and_then(|value| value.parse::<u32>().ok())
				.unwrap_or(0);

			(
				"200 OK",
				"application/json",
				shared.slots.drummer(node).click_json(),
			)
		}
		("POST", path) if path == "/click" || path.starts_with("/click?") => {
			let node = path
				.split_once('?')
				.and_then(|(_, query)| query.split('&').find_map(|pair| pair.strip_prefix("node=")))
				.and_then(|value| value.parse::<u32>().ok())
				.unwrap_or(0);

			(
				"200 OK",
				"application/json",
				shared.slots.drummer(node).set_click(body_bytes),
			)
		}
		("GET", path) if path == "/samples" || path.starts_with("/samples?") => {
			let node = path
				.split_once('?')
				.and_then(|(_, query)| query.split('&').find_map(|pair| pair.strip_prefix("node=")))
				.and_then(|value| value.parse::<u32>().ok())
				.unwrap_or(0);

			(
				"200 OK",
				"application/json",
				shared.slots.drummer(node).samples_json(),
			)
		}
		("POST", path) if path == "/samples" || path.starts_with("/samples?") => {
			let query = path.split_once('?').map(|(_, query)| query).unwrap_or("");
			let node = query
				.split('&')
				.find_map(|pair| pair.strip_prefix("node="))
				.and_then(|value| value.parse::<u32>().ok())
				.unwrap_or(0);
			let slot = query
				.split('&')
				.find_map(|pair| pair.strip_prefix("slot="))
				.unwrap_or("");
			let clear = query.split('&').any(|pair| pair == "clear=1");

			(
				"200 OK",
				"application/json",
				shared
					.slots
					.drummer(node)
					.set_sample(body_bytes, slot, clear),
			)
		}
		("GET", path) if path == "/pitch" || path.starts_with("/pitch?") => {
			let node = path
				.split_once('?')
				.and_then(|(_, query)| query.split('&').find_map(|pair| pair.strip_prefix("node=")))
				.and_then(|value| value.parse::<u32>().ok())
				.unwrap_or(0);

			(
				"200 OK",
				"application/json",
				shared.slots.drummer(node).pitch_json(),
			)
		}
		("POST", path) if path == "/pitch" || path.starts_with("/pitch?") => {
			let node = path
				.split_once('?')
				.and_then(|(_, query)| query.split('&').find_map(|pair| pair.strip_prefix("node=")))
				.and_then(|value| value.parse::<u32>().ok())
				.unwrap_or(0);

			(
				"200 OK",
				"application/json",
				shared.slots.drummer(node).set_pitch(body_bytes),
			)
		}
		("GET", path) if path == "/amp" || path.starts_with("/amp?") => {
			let node = path
				.split_once('?')
				.and_then(|(_, query)| query.split('&').find_map(|pair| pair.strip_prefix("node=")))
				.and_then(|value| value.parse::<u32>().ok())
				.unwrap_or(0);

			(
				"200 OK",
				"application/json",
				shared.slots.drummer(node).amp_json(),
			)
		}
		("POST", path) if path == "/amp" || path.starts_with("/amp?") => {
			let query = path.split_once('?').map(|(_, query)| query).unwrap_or("");
			let node = query
				.split('&')
				.find_map(|pair| pair.strip_prefix("node="))
				.and_then(|value| value.parse::<u32>().ok())
				.unwrap_or(0);
			let drum = query
				.split('&')
				.find_map(|pair| pair.strip_prefix("drum="))
				.and_then(|value| value.parse::<usize>().ok())
				.unwrap_or(0);

			(
				"200 OK",
				"application/json",
				shared.slots.drummer(node).set_amp(body_bytes, drum),
			)
		}
		("POST", path)
			if path == "/filter-table/import" || path.starts_with("/filter-table/import?") =>
		{
			let query = path.split_once('?').map(|(_, query)| query).unwrap_or("");
			let value = |key: &str| {
				query
					.split('&')
					.find_map(|pair| pair.strip_prefix(&format!("{key}=")))
					.and_then(|value| value.parse::<usize>().ok())
			};
			let node = value("node").unwrap_or(0) as u32;
			let window = value("window").unwrap_or(2048);
			let page = value("page").unwrap_or(0);
			let mode = query
				.split('&')
				.find_map(|pair| pair.strip_prefix("mode="))
				.unwrap_or("time");
			let slice = query
				.split('&')
				.find_map(|pair| pair.strip_prefix("slice="))
				.unwrap_or("auto");

			(
				"200 OK",
				"application/json",
				shared
					.slots
					.filter_table(node)
					.import_wav(body_bytes, window, page, mode, slice),
			)
		}
		("POST", path)
			if path == "/filter-table/edit" || path.starts_with("/filter-table/edit?") =>
		{
			let query = path.split_once('?').map(|(_, query)| query).unwrap_or("");
			let value = |key: &str| {
				query
					.split('&')
					.find_map(|pair| pair.strip_prefix(&format!("{key}=")))
					.and_then(|value| value.parse::<usize>().ok())
			};
			let node = value("node").unwrap_or(0) as u32;
			let page = value("page").unwrap_or(0);
			let action = query
				.split('&')
				.find_map(|pair| pair.strip_prefix("action="))
				.unwrap_or("");

			(
				"200 OK",
				"application/json",
				shared.slots.filter_table(node).edit_page(action, page),
			)
		}
		("GET", path) if path == "/filter-table" || path.starts_with("/filter-table?") => {
			let node = path
				.split_once('?')
				.and_then(|(_, query)| query.split('&').find_map(|pair| pair.strip_prefix("node=")))
				.and_then(|value| value.parse::<u32>().ok())
				.unwrap_or(0);

			(
				"200 OK",
				"application/json",
				shared.slots.filter_table(node).state_json(),
			)
		}
		("POST", path) if path == "/filter-table" || path.starts_with("/filter-table?") => {
			let node = path
				.split_once('?')
				.and_then(|(_, query)| query.split('&').find_map(|pair| pair.strip_prefix("node=")))
				.and_then(|value| value.parse::<u32>().ok())
				.unwrap_or(0);

			(
				"200 OK",
				"application/json",
				shared.slots.filter_table(node).set_json(body),
			)
		}
		("GET", path) if path == "/history-phase" || path.starts_with("/history-phase?") => {
			let node = path
				.split_once('?')
				.and_then(|(_, query)| query.split('&').find_map(|pair| pair.strip_prefix("node=")))
				.and_then(|value| value.parse::<u32>().ok())
				.unwrap_or(0);

			(
				"200 OK",
				"application/json",
				shared.slots.history(node).phase_json(),
			)
		}
		("GET", path) if path == "/history" || path.starts_with("/history?") => {
			let node = path
				.split_once('?')
				.and_then(|(_, query)| query.split('&').find_map(|pair| pair.strip_prefix("node=")))
				.and_then(|value| value.parse::<u32>().ok())
				.unwrap_or(0);

			(
				"200 OK",
				"application/json",
				shared.slots.history(node).state_json(),
			)
		}
		("POST", path) if path == "/history" || path.starts_with("/history?") => {
			let node = path
				.split_once('?')
				.and_then(|(_, query)| query.split('&').find_map(|pair| pair.strip_prefix("node=")))
				.and_then(|value| value.parse::<u32>().ok())
				.unwrap_or(0);

			(
				"200 OK",
				"application/json",
				shared.slots.history(node).set_json(body),
			)
		}
		("GET", path) if path == "/script" || path.starts_with("/script?") => {
			let node = path
				.split_once('?')
				.and_then(|(_, query)| query.split('&').find_map(|pair| pair.strip_prefix("node=")))
				.and_then(|value| value.parse::<u32>().ok())
				.unwrap_or(0);

			(
				"200 OK",
				"application/json",
				shared.slots.script(node).state_json(),
			)
		}
		("POST", path) if path == "/script" || path.starts_with("/script?") => {
			let node = path
				.split_once('?')
				.and_then(|(_, query)| query.split('&').find_map(|pair| pair.strip_prefix("node=")))
				.and_then(|value| value.parse::<u32>().ok())
				.unwrap_or(0);

			(
				"200 OK",
				"application/json",
				shared.slots.script(node).apply(body.to_string()),
			)
		}
		("GET", path) if path == "/scope" || path.starts_with("/scope?") => {
			let node = path
				.split_once('?')
				.and_then(|(_, query)| query.split('&').find_map(|pair| pair.strip_prefix("node=")))
				.and_then(|value| value.parse::<usize>().ok())
				.unwrap_or(0);
			let which = path
				.split_once('?')
				.and_then(|(_, query)| {
					query
						.split('&')
						.find_map(|pair| pair.strip_prefix("which="))
				})
				.unwrap_or("output");
			let (output, input) = shared.tap_for(node as u32);
			let scope = if which == "input" { input } else { output };

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
		("GET", path) if path == "/scopes" || path.starts_with("/scopes?") => {
			let query = path.split_once('?').map(|(_, query)| query).unwrap_or("");
			let selected = query
				.split('&')
				.find_map(|pair| pair.strip_prefix("node="))
				.and_then(|node| node.parse::<u32>().ok());
			let cursors: std::collections::BTreeMap<u32, u64> = query
				.split('&')
				.find_map(|pair| pair.strip_prefix("since="))
				.map(|value| {
					value
						.split(',')
						.filter_map(|item| {
							let (node, block) = item.split_once(':')?;

							Some((node.parse().ok()?, block.parse().ok()?))
						})
						.collect()
				})
				.unwrap_or_default();
			let mut scopes = Vec::new();

			{
				let graph = shared.graph.lock().expect("graph");

				for node in &graph.nodes {
					if !matches!(
						node.kind.id,
						"lfo" | "envelope" | "random" | "table" | "pitch" | "midi"
					) {
						continue;
					}

					let (scope, _) = shared.tap_for(node.id.0);
					let since = cursors.get(&node.id.0).copied().unwrap_or(0);
					let (block, blocks) = scope.read_blocks_since(since, Scope::BLOCKS);

					scopes.push(json!({
						"node": node.id.0,
						"blocks": blocks,
						"block": block,
						"phase": shared.phase_for(node.id.0),
						"input": shared.midi_input_for(node.id.0),
					}));
				}
			}

			let live = selected.map(|node| {
				let (output, input) = shared.tap_for(node);

				json!({
					"node": node,
					"output": output.read(2048),
					"input": input.read(2048),
				})
			});

			(
				"200 OK",
				"application/json",
				json!({
					"rate": shared.rate,
					"block_frames": i_am_patch::block::BLOCK,
					"scopes": scopes,
					"live": live,
				})
				.to_string(),
			)
		}
		("POST", "/op") => ("200 OK", "application/json", op(shared, body).to_string()),
		("POST", path) if path == "/sample" || path.starts_with("/sample?") => (
			"200 OK",
			"application/json",
			sample_route(shared, path, body_bytes).to_string(),
		),
		(method, path) if path == "/sampler" || path.starts_with("/sampler?") => (
			"200 OK",
			"application/json",
			import_route(shared, method, path, body_bytes, Import::Sample),
		),
		(method, path) if path == "/waves" || path.starts_with("/waves?") => (
			"200 OK",
			"application/json",
			import_route(shared, method, path, body_bytes, Import::Waves),
		),
		(method, path) if path == "/ir" || path.starts_with("/ir?") => (
			"200 OK",
			"application/json",
			import_route(shared, method, path, body_bytes, Import::Ir),
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
fn live_edit(value: &Value) -> Option<i_am_patch::edit::Edit> {
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

/// Fits one recorded note into a Modal node, the way the product's route does.
/// The three files a node can take besides a fitted note.
#[derive(Clone, Copy)]
enum Import {
	/// The sampler's sample.
	Sample,
	/// The shapes' wavetable.
	Waves,
	/// The convolver's impulse response.
	Ir,
}

/// Answers the sampler, the shapes and the convolver: a GET describes what the
/// slot holds, a POST loads a file into it.
fn import_route(shared: &Shared, method: &str, path: &str, body: &[u8], which: Import) -> String {
	let query = path.split_once('?').map(|(_, query)| query).unwrap_or("");
	let node = query
		.split('&')
		.find_map(|pair| pair.strip_prefix("node="))
		.and_then(|value| value.parse::<u32>().ok())
		.unwrap_or(0);
	let name = query
		.split('&')
		.find_map(|pair| pair.strip_prefix("name="))
		.map(i_am_web_gui::percent_decode)
		.filter(|name| !name.is_empty())
		.unwrap_or_else(|| "file".to_string());

	let action = query
		.split('&')
		.find_map(|pair| pair.strip_prefix("action="))
		.filter(|action| !action.is_empty());

	match which {
		Import::Sample => {
			let slot = shared.slots.sample(node);

			if method == "GET" {
				return i_am_patch::sample_preview(&slot);
			}

			if let Some(action) = action {
				let number = |key: &str| {
					query
						.split('&')
						.find_map(|pair| pair.strip_prefix(key)?.strip_prefix('='))
						.and_then(|value| value.parse::<f32>().ok())
				};
				let at = number("at").unwrap_or(0.0);
				let to = number("to").unwrap_or(at);

				return match action {
					"slices" => match i_am_patch::generate_slices(&slot, "grid") {
						Ok(count) => json!({ "ok": true, "slices": count }).to_string(),
						Err(error) => json!({ "ok": false, "error": error }).to_string(),
					},
					_ => match i_am_patch::edit_slices(&slot, action, at, to) {
						Ok(count) => json!({ "ok": true, "slices": count }).to_string(),
						Err(error) => json!({ "ok": false, "error": error }).to_string(),
					},
				};
			}

			match i_am_patch::import_sample(&slot, body, &name) {
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
		Import::Waves => {
			let slot = shared.slots.waves(node);

			if method == "GET" {
				return i_am_patch::waves_preview(&slot);
			}

			if let Some(action) = action {
				let frame = query
					.split('&')
					.find_map(|pair| pair.strip_prefix("frame="))
					.and_then(|frame| frame.parse::<usize>().ok());

				return match i_am_patch::edit_waves(&slot, action, frame) {
					Ok(count) => json!({ "ok": true, "count": count }).to_string(),
					Err(error) => json!({ "ok": false, "error": error }).to_string(),
				};
			}

			let mode = query
				.split('&')
				.find_map(|pair| pair.strip_prefix("mode="))
				.filter(|mode| !mode.is_empty());

			match i_am_patch::import_waves(&slot, body, &name, mode) {
				Ok((name, frames)) => {
					json!({ "ok": true, "name": name, "frames": frames }).to_string()
				}
				Err(error) => json!({ "ok": false, "error": error }).to_string(),
			}
		}
		Import::Ir => {
			let slot = shared.slots.ir(node);

			if method == "GET" {
				return slot.json();
			}

			if let (Some(fade_in), Some(fade_out)) = (
				query
					.split('&')
					.find_map(|pair| pair.strip_prefix("fade_in="))
					.and_then(|value| value.parse::<f32>().ok()),
				query
					.split('&')
					.find_map(|pair| pair.strip_prefix("fade_out="))
					.and_then(|value| value.parse::<f32>().ok()),
			) {
				slot.set_fades(i_am_effects::IrFades {
					fade_in: fade_in.clamp(0.0, 1.0),
					fade_out: fade_out.clamp(0.0, 1.0),
				});

				return json!({ "ok": true }).to_string();
			}

			match i_am_effects::decode_ir(body, slot.host_rate(), name) {
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
	}
}

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
			shared
				.slots
				.fitted(node)
				.set(i_am_dsp::prelude::ImportedTable {
					name: name.to_string(),
					bytes: fitted.bytes,
				});

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
