//! A WebView2 editor for `i_am_plugin`, as a test of the Editor seam.
//!
//! The plug-in wrapper knows nothing about a toolkit; this crate is everything a
//! web-view GUI costs it. `WryEditor` builds a WebView2 child inside the window the
//! host hands over, loads a page into it, and forwards what the page sends through
//! `window.ipc.postMessage` to the processor over the same message channel the iced
//! backend uses.
//!
//! A page can also serve itself from a custom protocol, which is how a GUI keeps a
//! `/state` route open for the values the host changes: see
//! `WryEditor::with_custom_protocol` and `WryEditor::with_url`.
//!
//! Each platform embeds the view in the window the host hands over, and each takes a
//! different route there: Windows builds its own child window for a WebView2
//! controller, Linux puts a WebKitGTK view in the host's X11 window and macOS adds a
//! WKWebView as a subview of the host's `NSView`. Linux also needs the caller to own
//! the GTK side (see `unix`): a plug-in does not run the main loop, so the wrapper's
//! GUI timer advances it. Every other platform keeps a stub that reports the
//! limitation instead of failing to build, so a workspace that contains it still
//! checks.
//!
//! WebView2 needs a writable profile folder. Unless the plug-in picks one with
//! `WryEditor::with_data_directory`, the editor uses
//! `%LOCALAPPDATA%\i_am_plugin_wry\WebView2-v2` and falls back to `%TEMP%`; the
//! default next to a host's executable is often under Program Files and cannot be
//! written, which leaves a blank editor in a DAW. The folder is shared by all
//! instances of this build; the version suffix changes when the browser options do.
//! Old per-process `WebView2-<pid>` folders are deleted on the first open, and the
//! disk caches are bounded. `I_AM_PLUGIN_WRY_PER_PROCESS_PROFILE=1` restores the
//! per-process layout if a host disagrees with sharing. An open failure is written
//! to `%TEMP%\i_am_plugin_wry\last-error.log`, and a timestamped `open.log` records
//! how far it got. The logs rotate at 256 KiB (`I_AM_PLUGIN_WRY_LOG_BYTES`), the
//! WebView2 caches are capped, and profile folders from older builds are removed
//! after a week.
//!
//! ```ignore
//! use i_am_plugin::{EditorContext, Plugin};
//!
//! impl i_am_plugin::Plugin for MyProcessor {
//!     const HAS_GUI: bool = true;
//!
//!     fn editor(&self) -> Option<Box<dyn i_am_plugin::Editor<Self>>> {
//!         Some(Box::new(i_am_plugin::wry::WryEditor::new(
//!             "<html>...</html>",
//!             |body| Some(MyMessage::from_js(body)),
//!         )))
//!     }
//!
//!     // ...
//! }
//! ```

#[cfg(any(windows, target_os = "linux", target_os = "macos"))]
use crate::EditorError;

#[cfg(windows)]
mod windows;
#[cfg(windows)]
pub use windows::WryEditor;

#[cfg(any(target_os = "linux", target_os = "macos"))]
mod unix;
#[cfg(any(target_os = "linux", target_os = "macos"))]
pub use unix::WryEditor;

#[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
mod stub;
#[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
pub use stub::WryEditor;

/// What a custom-protocol route answers with.
///
/// A web GUI can serve its own files and state this way: the page loads from a URL
/// such as `crusher://localhost/`, and the handler answers `/` with the HTML and
/// `/state` with a JSON snapshot of the parameters. It runs off the UI thread, so the
/// handler must not touch anything thread-bound.
/// One request the page made to a custom-protocol route.
///
/// A route that only serves state can look at [path](Self::path) alone; a route
/// that takes an upload needs the method and the body as well.
pub struct ProtocolRequest<'a> {
	/// The HTTP method, such as `GET` or `POST`.
	pub method: &'a str,
	/// The path, without the query string.
	pub path: &'a str,
	/// The query string, without the leading `?`.
	pub query: Option<&'a str>,
	/// The request body, empty for a request that has none.
	pub body: &'a [u8],
}

#[derive(Debug, Clone)]
pub struct ProtocolResponse {
	/// The HTTP status code.
	pub status: u16,
	/// The HTTP content type, such as `text/html`.
	pub content_type: &'static str,
	/// The body bytes.
	pub body: Vec<u8>,
}

impl ProtocolResponse {
	/// An HTML page.
	pub fn html(body: impl Into<Vec<u8>>) -> Self {
		Self {
			status: 200,
			content_type: "text/html; charset=utf-8",
			body: body.into(),
		}
	}

	/// A JSON document.
	pub fn json(body: impl Into<Vec<u8>>) -> Self {
		Self {
			status: 200,
			content_type: "application/json",
			body: body.into(),
		}
	}

	/// Plain text.
	pub fn text(body: impl Into<Vec<u8>>) -> Self {
		Self {
			status: 200,
			content_type: "text/plain; charset=utf-8",
			body: body.into(),
		}
	}

	/// Nothing at this path.
	pub fn not_found() -> Self {
		Self {
			status: 404,
			content_type: "text/plain; charset=utf-8",
			body: b"not found".to_vec(),
		}
	}
}

/// Turn one of our protocol answers into what wry's protocol callback returns.
///
/// The same on every platform: a bad status or header would panic inside the
/// protocol callback, and a panic crossing that FFI boundary takes the host down,
/// so an unbuildable response falls back to an empty 500 instead.
#[cfg(any(windows, target_os = "linux", target_os = "macos"))]
pub(crate) fn to_wry(
	response: ProtocolResponse,
) -> wry::http::Response<std::borrow::Cow<'static, [u8]>> {
	let built = wry::http::Response::builder()
		.status(response.status)
		.header(wry::http::header::CONTENT_TYPE, response.content_type)
		// A plug-in's page is embedded in the binary, and a host that keeps the
		// browser cache from an older version would run yesterday's script against
		// today's routes. Nothing here is worth caching.
		.header(wry::http::header::CACHE_CONTROL, "no-store")
		.body(response.body.into());

	match built {
		Ok(response) => response,
		Err(_) => wry::http::Response::builder()
			.status(500)
			.body(std::borrow::Cow::Borrowed(&[][..]))
			.expect("a fixed empty response is well formed"),
	}
}

/// Record why an editor could not open, where a host's log cannot be read.
///
/// A DAW swallows the error a plug-in's GUI open returns, so the blank window
/// gives no clue. The error log lives in the temp directory and is capped by the
/// log module.
#[cfg(any(windows, target_os = "linux", target_os = "macos"))]
pub(crate) fn report_error(error: &EditorError) {
	eprintln!("i_am_plugin_wry: the editor failed: {error}");
	crate::log::error(&format!("error: {error}"));
}

/// Record panic messages (including ones that happen on a web-view thread) in the
/// same log the open path writes.
///
/// A host cannot show a Rust panic, and a panic crossing the host's FFI boundary
/// is a crash, so the trail is the only thing left after one.
#[cfg(any(windows, target_os = "linux", target_os = "macos"))]
pub(crate) fn install_panic_hook() {
	static ONCE: std::sync::OnceLock<()> = std::sync::OnceLock::new();

	ONCE.get_or_init(|| {
		let previous = std::panic::take_hook();
		std::panic::set_hook(Box::new(move |info| {
			let location = info
				.location()
				.map(|location| format!("{}:{}", location.file(), location.line()))
				.unwrap_or_default();
			crate::log::note(&format!("panic: {info} at {location}"));
			previous(info);
		}));
	});
}

/// The message of a caught panic, when it is a string.
#[cfg(any(windows, target_os = "linux", target_os = "macos"))]
pub(crate) fn panic_payload(payload: &(dyn std::any::Any + Send)) -> String {
	if let Some(message) = payload.downcast_ref::<&str>() {
		(*message).to_string()
	} else if let Some(message) = payload.downcast_ref::<String>() {
		message.clone()
	} else {
		"non-string panic".to_string()
	}
}
