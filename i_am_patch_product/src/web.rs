//! The routes the patcher page owns: the catalog, the graph and the ops.

use std::collections::BTreeMap;

use std::sync::atomic::Ordering;

use i_am_dsp::prelude::SetValue;
use i_am_patch::graph::{Graph, NodeId};
use i_am_patch::scope::Scope;
use i_am_plugin::wry::{ProtocolRequest, ProtocolResponse};
use serde_json::{Value, json};

use crate::shared::PatcherShared;

/// What every saved bundle starts with, so a text file and a bundle can be told apart.
const BUNDLE_MAGIC: &[u8; 4] = b"IAMP";
/// The layout a bundle is written in.
const BUNDLE_VERSION: u32 = 1;

/// The whole patch as one file: the graph text and the payloads the slots hold.
///
/// The host saves the two in separate parameters, and the text alone is what a person
/// reads and an agent writes. A file meant to move a patch to another machine needs
/// both, so it is one container rather than two files somebody has to keep together.
fn bundle(shared: &PatcherShared) -> Vec<u8> {
	let text = shared.text().into_bytes();
	let payload = shared.slot_state_bytes();
	let mut bytes = Vec::with_capacity(12 + text.len() + payload.len());

	bytes.extend_from_slice(BUNDLE_MAGIC);
	bytes.extend_from_slice(&BUNDLE_VERSION.to_le_bytes());
	bytes.extend_from_slice(&(text.len() as u32).to_le_bytes());
	bytes.extend_from_slice(&text);
	bytes.extend_from_slice(&payload);

	bytes
}

/// Puts a bundle back, text first and then its payloads.
fn unbundle(shared: &PatcherShared, bytes: &[u8]) -> Result<(), String> {
	if bytes.len() < 12 || &bytes[..4] != BUNDLE_MAGIC {
		return Err("not a patch bundle".to_string());
	}

	let version = u32::from_le_bytes(bytes[4..8].try_into().expect("four bytes"));

	if version != BUNDLE_VERSION {
		return Err(format!(
			"a bundle version {version}, this build reads {BUNDLE_VERSION}"
		));
	}

	let length = u32::from_le_bytes(bytes[8..12].try_into().expect("four bytes")) as usize;
	let end = 12 + length;

	if end > bytes.len() {
		return Err("the bundle is shorter than its graph".to_string());
	}

	let Ok(text) = std::str::from_utf8(&bytes[12..end]) else {
		return Err("the graph in the bundle is not text".to_string());
	};

	if !shared.set_text(text) {
		return Err("the graph in the bundle did not parse".to_string());
	}

	let payload = &bytes[end..];

	if !payload.is_empty() && !shared.set_slot_state(payload) {
		return Err("the payloads in the bundle did not decode".to_string());
	}

	Ok(())
}

/// The file the save dialog offers, matching the format the page picked.
///
/// One Save button offers both formats, so the filter has to follow the one that
/// was chosen: a bundle put through the text filter would be written with a
/// `.txt` extension by the dialog's own default.
fn save_filter(bundled: bool) -> i_am_effects::dialog::Filter {
	if bundled {
		("Patch bundle", &["iamp"])
	} else {
		("Patch text", &["txt"])
	}
}

/// Answers a page request, or `None` for the kit's own routes.
pub fn route(request: &ProtocolRequest<'_>, shared: &PatcherShared) -> Option<ProtocolResponse> {
	match (request.method, request.path) {
		("GET", "/catalog") => Some(ProtocolResponse::json(catalog_json())),
		("GET", "/patch") => Some(ProtocolResponse::json(patch_json(&shared.graph()))),
		("GET", "/status") => Some(ProtocolResponse::json(
			json!({ "error": shared.error() }).to_string(),
		)),
		// The page's Save button: a native save-as beside the editor. The dialog is a
		// modal window on a worker thread, so asking starts it and a poll says how it
		// ended; a page with no host beside it gets told there is no dialog and falls
		// back to the browser's download.
		("POST", "/save") => {
			let bundled = query_value(request, "bundle").is_some();
			let bytes = if bundled {
				bundle(shared)
			} else {
				shared.text().into_bytes()
			};
			let asked = shared.save.ask(
				bytes,
				query_value(request, "name").unwrap_or("patch.txt"),
				save_filter(bundled),
			);

			Some(ProtocolResponse::json(
				match asked {
					crate::save::Asked::Started => json!({ "ok": true, "dialog": true }),
					crate::save::Asked::Busy => json!({ "ok": true, "dialog": true, "pending": true }),
					crate::save::Asked::Unavailable => {
						json!({ "ok": false, "reason": "no native dialog" })
					}
				}
				.to_string(),
			))
		}
		("GET", "/save") => Some(ProtocolResponse::json(shared.save.json())),
		// The whole patch as one file, for saving it somewhere the host's own state
		// does not reach: the graph text and the payloads the slots hold.
		("GET", "/bundle") => Some(ProtocolResponse {
			status: 200,
			content_type: "application/octet-stream",
			body: bundle(shared),
		}),
		("POST", "/bundle") => {
			let answer = match unbundle(shared, request.body) {
				Ok(()) => json!({ "ok": true }),
				Err(error) => json!({ "ok": false, "error": error }),
			};

			Some(ProtocolResponse::json(answer.to_string()))
		}
		// The host's macro controls, for the page: it reads them here and writes
		// them through the same atomics a host does, so a drag is automation.
		("GET", "/macros") => {
			let values: Vec<f32> = (0..i_am_patch::MACRO_COUNT)
				.map(|index| {
					shared
						.params
						.get_by_index(index)
						.and_then(|param| match param.load(Ordering::Relaxed) {
							SetValue::Float(value) => Some(value),
							_ => None,
						})
						.unwrap_or(0.0)
				})
				.collect();

			Some(ProtocolResponse::json(
				json!({ "values": values }).to_string(),
			))
		}
		("POST", "/macro") => {
			let index = query_value(request, "index")
				.and_then(|value| value.parse::<usize>().ok())
				.filter(|index| (1..=i_am_patch::MACRO_COUNT).contains(index));
			let value = query_value(request, "value").and_then(|value| value.parse::<f32>().ok());

			let Some((index, value)) = index.zip(value) else {
				return Some(ProtocolResponse::json(
					json!({ "ok": false, "error": "index and value" }).to_string(),
				));
			};

			let ok = shared.params.set(
				&format!("macro{index}"),
				SetValue::Float(value.clamp(0.0, 1.0)),
				Ordering::SeqCst,
			);

			Some(ProtocolResponse::json(json!({ "ok": ok }).to_string()))
		}
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
		("GET", "/canvas") | ("POST", "/canvas") => {
			let node = query_value(request, "node")
				.and_then(|value| value.parse::<u32>().ok())
				.unwrap_or(0);
			let canvas = shared.slots.canvas(node);
			let answer = if request.method == "POST" {
				canvas.set_shape(String::from_utf8_lossy(request.body).to_string())
			} else {
				canvas.shape_json()
			};

			Some(ProtocolResponse::json(answer))
		}
		("GET", "/spectrum") | ("POST", "/spectrum") => {
			let node = query_value(request, "node")
				.and_then(|value| value.parse::<u32>().ok())
				.unwrap_or(0);
			let spectrum = shared.slots.spectrum(node);
			let answer = if request.method == "POST" {
				spectrum.set_json(&String::from_utf8_lossy(request.body))
			} else {
				spectrum.state_json()
			};

			Some(ProtocolResponse::json(answer))
		}
		("GET", "/click") | ("POST", "/click") => {
			let node = query_value(request, "node")
				.and_then(|value| value.parse::<u32>().ok())
				.unwrap_or(0);
			let drummer = shared.slots.drummer(node);
			let answer = if request.method == "POST" {
				drummer.set_click(request.body)
			} else {
				drummer.click_json()
			};

			Some(ProtocolResponse::json(answer))
		}
		("GET", "/samples") | ("POST", "/samples") => {
			let node = query_value(request, "node")
				.and_then(|value| value.parse::<u32>().ok())
				.unwrap_or(0);
			let drummer = shared.slots.drummer(node);
			let answer = if request.method == "POST" {
				let query = request.query.unwrap_or("");
				let slot = query
					.split('&')
					.find_map(|pair| pair.strip_prefix("slot="))
					.unwrap_or("");
				let clear = query.split('&').any(|pair| pair == "clear=1");

				drummer.set_sample(request.body, slot, clear)
			} else {
				drummer.samples_json()
			};

			Some(ProtocolResponse::json(answer))
		}
		("GET", "/pitch") | ("POST", "/pitch") => {
			let node = query_value(request, "node")
				.and_then(|value| value.parse::<u32>().ok())
				.unwrap_or(0);
			let drummer = shared.slots.drummer(node);
			let answer = if request.method == "POST" {
				drummer.set_pitch(request.body)
			} else {
				drummer.pitch_json()
			};

			Some(ProtocolResponse::json(answer))
		}
		("GET", "/amp") | ("POST", "/amp") => {
			let node = query_value(request, "node")
				.and_then(|value| value.parse::<u32>().ok())
				.unwrap_or(0);
			let drummer = shared.slots.drummer(node);
			let answer = if request.method == "POST" {
				let drum = query_value(request, "drum")
					.and_then(|value| value.parse::<usize>().ok())
					.unwrap_or(0);

				drummer.set_amp(request.body, drum)
			} else {
				drummer.amp_json()
			};

			Some(ProtocolResponse::json(answer))
		}
		("GET", "/script") | ("POST", "/script") => {
			let node = query_value(request, "node")
				.and_then(|value| value.parse::<u32>().ok())
				.unwrap_or(0);
			let script = shared.slots.script(node);
			let answer = if request.method == "POST" {
				script.apply(String::from_utf8_lossy(request.body).to_string())
			} else {
				script.state_json()
			};

			Some(ProtocolResponse::json(answer))
		}
		("GET", "/filter-table/import/progress") => {
			let node = query_value(request, "node")
				.and_then(|value| value.parse::<u32>().ok())
				.unwrap_or(0);

			Some(ProtocolResponse::json(
				shared.slots.filter_table(node).progress(),
			))
		}
		("POST", "/filter-table/import") => {
			let node = query_value(request, "node")
				.and_then(|value| value.parse::<u32>().ok())
				.unwrap_or(0);
			let window = query_value(request, "window")
				.and_then(|value| value.parse::<usize>().ok())
				.unwrap_or(2048);
			let page = query_value(request, "page")
				.and_then(|value| value.parse::<usize>().ok())
				.unwrap_or(0);
			let mode = query_value(request, "mode").unwrap_or("time");
			let slice = query_value(request, "slice").unwrap_or("auto");

			Some(ProtocolResponse::json(
				shared.slots.filter_table(node).start_import(
					request.body.to_vec(),
					window,
					page,
					mode.to_string(),
					slice.to_string(),
				),
			))
		}
		("POST", "/filter-table/edit") => {
			let node = query_value(request, "node")
				.and_then(|value| value.parse::<u32>().ok())
				.unwrap_or(0);
			let page = query_value(request, "page")
				.and_then(|value| value.parse::<usize>().ok())
				.unwrap_or(0);
			let action = query_value(request, "action").unwrap_or("");

			Some(ProtocolResponse::json(
				shared.slots.filter_table(node).edit_page(action, page),
			))
		}
		("GET", "/filter-table") | ("POST", "/filter-table") => {
			let node = query_value(request, "node")
				.and_then(|value| value.parse::<u32>().ok())
				.unwrap_or(0);
			let table = shared.slots.filter_table(node);
			let answer = if request.method == "POST" {
				table.set_json(&String::from_utf8_lossy(request.body))
			} else {
				table.state_json()
			};

			Some(ProtocolResponse::json(answer))
		}
		("GET", "/history-phase") => {
			let node = query_value(request, "node")
				.and_then(|value| value.parse::<u32>().ok())
				.unwrap_or(0);

			Some(ProtocolResponse::json(
				shared.slots.history(node).phase_json(),
			))
		}
		("GET", "/history") | ("POST", "/history") => {
			let node = query_value(request, "node")
				.and_then(|value| value.parse::<u32>().ok())
				.unwrap_or(0);
			let history = shared.slots.history(node);
			let answer = if request.method == "POST" {
				history.set_json(&String::from_utf8_lossy(request.body))
			} else {
				history.state_json()
			};

			Some(ProtocolResponse::json(answer))
		}
		("GET", "/sampler") | ("POST", "/sampler") => Some(ProtocolResponse::json(
			sampler_route(request, shared).to_string(),
		)),
		("GET", "/waves") | ("POST", "/waves") => Some(ProtocolResponse::json(
			waves_route(request, shared).to_string(),
		)),
		// A constructor's curve, sampled for the card: the page draws it, it
		// does not appear in the patch, so it is not worth a route of its own
		// on anything but this product.
		("GET", "/wave") => {
			let node = query_value(request, "node")
				.and_then(|value| value.parse::<u32>().ok())
				.unwrap_or(0);
			let points = query_value(request, "points")
				.and_then(|value| value.parse::<usize>().ok())
				.unwrap_or(192);
			let values = i_am_patch::sample_constructor(
				&shared.graph(),
				&shared.slots,
				shared.rate.load(Ordering::Relaxed).max(1),
				NodeId(node),
				points,
			);

			Some(ProtocolResponse::json(
				json!({ "values": values }).to_string(),
			))
		}
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
		// One small answer to "did anything change": the page polls this and only
		// reads the whole graph when the revision moved.
		("GET", "/state") => {
			use std::sync::atomic::Ordering;

			Some(ProtocolResponse::json(
				json!({
					"revision": shared.generation.load(Ordering::Relaxed),
					"error": shared.error(),
				})
				.to_string(),
			))
		}
		("GET", "/text") => Some(ProtocolResponse::json(json!(shared.text()).to_string())),
		("POST", "/text") => {
			let body = std::str::from_utf8(request.body).unwrap_or("");
			let ok = shared.set_text(body);

			Some(ProtocolResponse::json(json!({ "ok": ok }).to_string()))
		}
		// The OTT's three bands and the vocoder's bank are measured inside the
		// node's effect, so a card asks the node for them: the product's own page
		// has one product to ask, and a patch has as many as it has cards.
		("GET", "/bands") => {
			let node = query_value(request, "node").and_then(|node| node.parse::<u32>().ok());
			// A node that meters nothing answers with an empty meter rather than a
			// made-up one: the card only asks the nodes that have a meter, and a
			// route the page owns has to own the question too.
			let body = node
				.and_then(|node| shared.slots.bands(node))
				.map_or_else(|| "{\"bands\":[]}".to_string(), |bands| bands.json());

			Some(ProtocolResponse::json(body))
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

			// The card the page is watching rides along with the sources, so one
			// request per tick feeds every live picture on the page.
			let selected = query_value(request, "node").and_then(|node| node.parse::<u32>().ok());
			let cursors = since_cursors(request);

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
				let since = cursors.get(&node).copied().unwrap_or(0);
				let (block, blocks) = scope.read_blocks_since(since, Scope::BLOCKS);

				scopes.push(json!({
					"node": node,
					"blocks": blocks,
					"block": block,
					"phase": shared.phase_for(node),
					"input": shared.midi_input_for(node),
				}));
			}

			let live = selected.map(|node| {
				let (output, input) = shared.taps_for(node);

				json!({
					"node": node,
					"output": output.read(2048),
					"input": input.read(2048),
				})
			});

			Some(ProtocolResponse::json(
				json!({
					"rate": shared.rate.load(Ordering::Relaxed),
					"block_frames": i_am_patch::block::BLOCK,
					"scopes": scopes,
					"live": live,
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

/// The block each source's history had reached at the page's last poll.
fn since_cursors(request: &ProtocolRequest<'_>) -> BTreeMap<u32, u64> {
	let mut cursors = BTreeMap::new();
	let Some(value) = query_value(request, "since").map(i_am_web_gui::percent_decode) else {
		return cursors;
	};

	for item in value.split(',') {
		if let Some((node, block)) = item.split_once(':')
			&& let (Ok(node), Ok(block)) = (node.parse::<u32>(), block.parse::<u64>())
		{
			cursors.insert(node, block);
		}
	}

	cursors
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

	/// Every route and op the patcher page asks for.
	///
	/// The page is the definition of what a person can do to a patch, so it is
	/// also the definition of what an agent must be able to do: a control the
	/// page offers and the API cannot answer is a hole nothing else would
	/// notice. This reads the page's own source and asks the route table for
	/// each thing it calls.
	#[test]
	fn the_api_answers_everything_the_page_asks_for() {
		let page = include_str!("../../i_am_patch/src/page/patcher.js");
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
}
