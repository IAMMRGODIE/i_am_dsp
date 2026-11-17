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
//! It is Windows-only for now: the wrapper creates the WebView2 control as a child of
//! the CLAP parent window, which is the one platform this was tried on. The rest of
//! the crate is a stub that reports the limitation instead of failing to build, so a
//! workspace that contains it still checks on other platforms.
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

#[cfg(windows)]
mod windows;
#[cfg(windows)]
pub use windows::WryEditor;

#[cfg(not(windows))]
mod stub;
#[cfg(not(windows))]
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
