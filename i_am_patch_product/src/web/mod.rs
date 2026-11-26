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
					crate::save::Asked::Busy => {
						json!({ "ok": true, "dialog": true, "pending": true })
					}
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
			// The node says which script the page is asking about: the two
			// scripted effects keep theirs in a worker of their own, the synth
			// products in the handle their node was built with.
			let Some(script) = node_script(shared, node) else {
				return Some(ProtocolResponse::json("{}".to_string()));
			};

			if request.method == "POST" {
				script.set_source(String::from_utf8_lossy(request.body).to_string());
			}

			Some(ProtocolResponse::json(script.state_json()))
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
		("GET", "/spectral-dynamics") | ("POST", "/spectral-dynamics") => {
			let node = query_value(request, "node")
				.and_then(|value| value.parse::<u32>().ok())
				.unwrap_or(0);
			let curves = shared.slots.spectral(node);
			let answer = if request.method == "POST" {
				curves.set_json(&String::from_utf8_lossy(request.body))
			} else {
				curves.state_json()
			};

			Some(ProtocolResponse::json(answer))
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

			// The level of the cards that draw an envelope, in dB. One number per
			// node per tick is all a dynamics hero needs to keep its window
			// moving, and it is cheap enough for every such card on the page —
			// the full frames stay with the one card the page is watching.
			let meters: Vec<serde_json::Value> = query_value(request, "meters")
				.unwrap_or("")
				.split(',')
				.filter_map(|node| node.parse::<u32>().ok())
				.map(|node| {
					let (output, input) = shared.taps_for(node);
					let level = |scope: &i_am_patch::scope::Scope| -> f32 {
						let frames = scope.read(2048);
						let count = frames.len().max(1) as f32;
						let energy: f32 = frames
							.iter()
							.map(|[left, right]| {
								let sample = (left + right) * 0.5;

								sample * sample
							})
							.sum();

						(energy / count).sqrt()
					};
					let db = |level: f32| 20.0 * level.max(1e-6).log10();

					json!({
						"node": node,
						"input": db(level(&input)),
						"output": db(level(&output)),
					})
				})
				.collect();

			Some(ProtocolResponse::json(
				json!({
					"rate": shared.rate.load(Ordering::Relaxed),
					"block_frames": i_am_patch::block::BLOCK,
					"scopes": scopes,
					"live": live,
					"meters": meters,
					// The cards a modulation moves draw the value the engine is
					// running: one list per such node, in the node's own units.
					"values": shared
						.values()
						.into_iter()
						.map(|(node, list)| {
							let pairs: Vec<serde_json::Value> = list
								.iter()
								.map(|(name, value)| json!([name, value]))
								.collect();

							json!([node, pairs])
						})
						.collect::<Vec<_>>(),
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
					// A note is text on the canvas, not part of the signal: nothing
					// is rebuilt for it, because a rebuild would close the field it
					// is being typed in.
					None if writes_text(body) => {}
					None => shared.bump(),
				}
			}

			Some(ProtocolResponse::json(result.to_string()))
		}
		_ => None,
	}
}

/// The script one node runs, whichever product's it is.
fn node_script(shared: &PatcherShared, node: u32) -> Option<i_am_patch::ScriptSlot> {
	let kind = shared
		.graph()
		.get(i_am_patch::graph::NodeId(node))
		.map(|node| node.kind.id)?;

	match kind {
		"mod:script_mapper" | "mod:script_distortion" => shared.slots.script_slot(node, kind),
		"synth:script" | "table:script" => {
			Some(i_am_patch::ScriptSlot::Synth(shared.slots.script(node)))
		}
		_ => None,
	}
}

/// Whether an op only writes a note. Nothing plays a note, so nothing is rebuilt
/// for it: the string reaches the patch text through the graph it was written into.
fn writes_text(body: &str) -> bool {
	serde_json::from_str::<Value>(body)
		.ok()
		.and_then(|value| value["op"].as_str().map(|op| op == "text"))
		.unwrap_or(false)
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
		"nodes": graph.nodes.iter().map(|node| {
			let mut entry = json!({
				"id": node.id.0,
				"kind": node.kind.id,
				"x": node.position.0,
				"y": node.position.1,
				"params": node.params.iter().map(|(key, value)| (key.to_string(), json!(value))).collect::<serde_json::Map<_, _>>(),
				"points": node.points.iter().map(|point| json!([point.time, point.value, point.bend])).collect::<Vec<_>>(),
				"velocity_points": node.velocity_points.iter().map(|point| json!([point.time, point.value, point.bend])).collect::<Vec<_>>(),
			});

			// Only a note carries text; every other node leaves the field out
			// rather than sending a null for each of them.
			if let Some(note) = &node.text {
				entry["text"] = json!(note);
			}

			entry
		}).collect::<Vec<_>>(),
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

				if let Some(note) = &node.text {
					graph.set_text(new_id, note.clone());
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
		// A note on the canvas. It is written into the graph so it saves and
		// undoes with the patch, but it is not a graph *change*: see the route
		// below, which does not rebuild for it.
		"text" => {
			let note = value["text"].as_str().unwrap_or("").to_string();
			let done = graph.set_text(id(&value["id"]), note);
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
mod tests;
