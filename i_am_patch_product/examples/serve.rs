//! The patcher's own web API without a host.
//!
//! This is the server the product runs inside the plug-in, with the plug-in
//! taken out of the way: the same page, the same routes, the same ops. An
//! agent drives it exactly as a person drives the window.
//!
//! cargo run -p i_am_patch_product --example serve -- [port]

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::thread;
use std::time::{Duration, Instant};

use i_am_patch_product::shared::{Body, PatcherShared};
use i_am_plugin::wry::ProtocolRequest;

/// The page, as the plug-in's webview gets it.
const PAGE_HTML: &str = include_str!("../../i_am_patch/src/page/index.html");
const PAGE_JS: &str = include_str!("../../i_am_patch/src/page/patcher.js");
/// The style: the card kit's, then the page's own.
const PAGE_CSS: &str = concat!(
	include_str!("../../i_am_web_gui/src/page/card.css"),
	"\n",
	include_str!("../../i_am_patch/src/page/patcher.css"),
);
/// What the page opens on: the demo patch, so an empty canvas is not the
/// first thing an agent has to explain to itself.
const DEMO: &str = include_str!("../../i_am_patch/patches/demo.patch");

fn main() {
	let port: u16 = std::env::args()
		.nth(1)
		.and_then(|value| value.parse().ok())
		.unwrap_or(0);

	let shared = Arc::new(PatcherShared::default());
	shared.set_text(DEMO);

	// The graph runs, in real time, whether or not anyone is listening: the
	// scopes, the meters and the control taps are what an agent reads to check
	// its work, and an empty ring says the same thing as a silent patch.
	let audio = shared.clone();
	thread::spawn(move || play(audio));

	let listener = TcpListener::bind(("127.0.0.1", port)).expect("bind the port");
	let local = listener.local_addr().expect("the bound address");

	println!("http://127.0.0.1:{}", local.port());

	for stream in listener.incoming() {
		let Ok(stream) = stream else { continue };
		let shared = shared.clone();

		thread::spawn(move || answer(stream, &shared));
	}
}

/// Renders the graph in real time, as the plug-in's audio thread does.
fn play(shared: Arc<PatcherShared>) {
	let mut body = Body::new(shared.clone());
	let rate = shared.rate.load(Ordering::Relaxed).max(1) as u64;
	// A block of work, then a block of sleep: the rings stay current without
	// a busy loop.
	// Sixty-four samples of audio per sixty-four samples of wall clock.
	let block = Duration::from_secs_f64(64.0 / rate as f64);
	let mut next = Instant::now();

	loop {
		shared.pull_macros();

		if let Ok(mut clears) = shared.clear_requests.try_lock() {
			for node in clears.drain(..) {
				body.clear_tail(node);
			}
		}

		for _ in 0..64 {
			body.process([0.0; 2], [0.0; 2]);
		}

		next += block;
		let now = Instant::now();
		if next > now {
			thread::sleep(next - now);
		} else {
			next = now;
		}
	}
}

/// One request: the page's files, or the routes the product answers.
fn answer(mut stream: TcpStream, shared: &PatcherShared) {
	let mut request = Vec::new();
	let mut buffer = [0u8; 8 * 1024];

	// Read the head, then exactly as much body as it announced.
	let head_end = loop {
		if let Some(found) = window(&request) {
			break found;
		}

		match stream.read(&mut buffer) {
			Ok(0) | Err(_) => return,
			Ok(count) => request.extend_from_slice(&buffer[..count]),
		}

		if request.len() > 64 * 1024 * 1024 {
			return;
		}
	};

	let head = String::from_utf8_lossy(&request[..head_end]).to_string();
	let mut lines = head.lines();
	let mut parts = lines.next().unwrap_or("").split_whitespace();
	let method = parts.next().unwrap_or("GET").to_string();
	let target = parts.next().unwrap_or("/").to_string();
	let (path, query) = match target.split_once('?') {
		Some((path, query)) => (path.to_string(), Some(query.to_string())),
		None => (target.clone(), None),
	};
	let length = head
		.lines()
		.find_map(|line| {
			let (name, value) = line.split_once(':')?;
			(name.eq_ignore_ascii_case("content-length"))
				.then(|| value.trim().parse::<usize>().ok())
				.flatten()
		})
		.unwrap_or(0);
	let body_start = head_end + 4;

	while request.len() < body_start + length {
		match stream.read(&mut buffer) {
			Ok(0) | Err(_) => break,
			Ok(count) => request.extend_from_slice(&buffer[..count]),
		}
	}

	let body = &request[body_start.min(request.len())..];

	// The page's files are the page's, not the API's.
	let asset = match path.as_str() {
		"/" => Some(("text/html; charset=utf-8", PAGE_HTML)),
		"/patcher.js" | "/kit.js" => Some(("text/javascript", PAGE_JS)),
		"/patcher.css" | "/kit.css" => Some(("text/css", PAGE_CSS)),
		"/card.css" => Some(("text/css", i_am_web_gui::CARD_STYLE)),
		"/card.js" => Some(("text/javascript", i_am_web_gui::CARD_SCRIPT)),
		_ => None,
	};

	let (status, content_type, payload) = match asset {
		Some((content_type, text)) => (200, content_type, text.as_bytes().to_vec()),
		None => {
			let request = ProtocolRequest {
				method: &method,
				path: &path,
				query: query.as_deref(),
				body,
			};

			match i_am_patch_product::web::route(&request, shared) {
				Some(response) => (response.status, response.content_type, response.body),
				None => (404, "text/plain", b"not found".to_vec()),
			}
		}
	};

	let reason = match status {
		200 => "OK",
		404 => "Not Found",
		_ => "OK",
	};
	let head = format!(
		"HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
		payload.len()
	);

	let _ = stream.write_all(head.as_bytes());
	let _ = stream.write_all(&payload);
	let _ = stream.flush();
}

/// Where the head ends, if it has ended: the first `\r\n\r\n`.
fn window(bytes: &[u8]) -> Option<usize> {
	bytes.windows(4).position(|four| four == b"\r\n\r\n")
}
