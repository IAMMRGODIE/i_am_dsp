//! One request: the page's files, or the route table behind them.

use super::*;

pub(super) fn handle(mut stream: TcpStream, shared: &Shared) {
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
			include_str!("../../src/page/index.html").to_string(),
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
		("GET", "/patcher.js") | ("GET", "/kit.js") => {
			("200 OK", "text/javascript", i_am_patch::PAGE_JS.to_string())
		}
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
		// The page's macro controls: the same routes the product answers, so the
		// standalone editor shows the same bar a host-driven instance does. The pad
		// the nodes read is the one every rebuild keeps.
		("GET", "/macros") => {
			let values: Vec<f32> = (0..i_am_patch::MACRO_COUNT)
				.map(|index| shared.slots.macros().get(index))
				.collect();

			(
				"200 OK",
				"application/json",
				json!({ "values": values }).to_string(),
			)
		}
		("POST", path) if path == "/macro" || path.starts_with("/macro?") => {
			let query = path.split_once('?').map(|(_, query)| query).unwrap_or("");
			let value = |name: &str| {
				query
					.split('&')
					.find_map(|pair| pair.strip_prefix(name))
					.and_then(|value| value.strip_prefix('='))
			};

			let index = value("index").and_then(|value| value.parse::<usize>().ok());
			let level = value("value").and_then(|value| value.parse::<f32>().ok());

			match (index, level) {
				(Some(index), Some(level)) if (1..=i_am_patch::MACRO_COUNT).contains(&index) => {
					shared.slots.macros().set(index - 1, level);

					(
						"200 OK",
						"application/json",
						json!({ "ok": true }).to_string(),
					)
				}
				_ => (
					"200 OK",
					"application/json",
					json!({ "ok": false, "error": "index and value" }).to_string(),
				),
			}
		}
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
		("GET", path)
			if path == "/filter-table/import/progress"
				|| path.starts_with("/filter-table/import/progress?") =>
		{
			let node = path
				.split_once('?')
				.and_then(|(_, query)| query.split('&').find_map(|pair| pair.strip_prefix("node=")))
				.and_then(|value| value.parse::<u32>().ok())
				.unwrap_or(0);

			(
				"200 OK",
				"application/json",
				shared.slots.filter_table(node).progress(),
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
				shared.slots.filter_table(node).start_import(
					body_bytes.to_vec(),
					window,
					page,
					mode.to_string(),
					slice.to_string(),
				),
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
		("GET", path) if path == "/wave" || path.starts_with("/wave?") => {
			let query = path.split_once('?').map(|(_, query)| query).unwrap_or("");
			let value = |key: &str| {
				query
					.split('&')
					.find_map(|pair| pair.strip_prefix(&format!("{key}=")))
					.and_then(|value| value.parse::<usize>().ok())
			};
			let node = value("node").unwrap_or(0) as u32;
			let points = value("points").unwrap_or(192);
			let graph = shared.graph.lock().expect("graph").clone();
			let values = i_am_patch::sample_constructor(
				&graph,
				&shared.slots,
				shared.rate,
				i_am_patch::graph::NodeId(node),
				points,
			);

			(
				"200 OK",
				"application/json",
				json!({ "values": values }).to_string(),
			)
		}
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
