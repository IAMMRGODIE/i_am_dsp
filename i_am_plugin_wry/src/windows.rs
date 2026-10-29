//! The WebView2 implementation.

use std::marker::PhantomData;
use std::path::PathBuf;

use crossbeam_channel::Sender;
use i_am_plugin::{Editor, EditorContext, EditorError, ParentWindow, Plugin, WindowSize};
use wry::dpi::{LogicalPosition, LogicalSize};
use wry::raw_window_handle::RawWindowHandle;
use wry::WebViewExtWindows;

use crate::ProtocolResponse;

/// What turns an IPC string from the page into a processor message.
type IpcMapper<P> = Box<dyn Fn(&str) -> Option<<P as i_am_plugin::Processor>::Message>>;

/// One route of a page served over a custom protocol.
type ProtocolHandler = Box<dyn Fn(&str) -> ProtocolResponse + Send + Sync>;

/// Where WebView2 keeps its profile when the plug-in does not choose.
///
/// It must be writable. WebView2's default is next to the executable, and a DAW's
/// executable is usually under Program Files, where the browser cannot create its
/// profile; a host that loads the plug-in from there gets a blank editor. A folder
/// under the user's local application data always is writable, so that is the fallback.
fn default_data_directory() -> Option<PathBuf> {
	let mut candidates = Vec::with_capacity(2);

	if let Some(base) = std::env::var_os("LOCALAPPDATA").map(PathBuf::from) {
		candidates.push(base.join("i_am_plugin_wry").join("WebView2"));
	}

	// The temp directory is the fallback: it is writable even when a profile folder
	// is locked down (or when the plug-in runs under a restrictive sandbox).
	candidates.push(std::env::temp_dir().join("i_am_plugin_wry").join("WebView2"));

	candidates
		.into_iter()
		.find(|directory| std::fs::create_dir_all(directory).is_ok())
}

/// Where [report_error] writes its note. The temp directory is always writable.
fn error_log_path() -> PathBuf {
	std::env::temp_dir()
		.join("i_am_plugin_wry")
		.join("last-error.log")
}

/// Record why an editor could not open, where a host's log cannot be read.
///
/// A DAW swallows the error a plug-in's GUI open returns, so the blank window gives no
/// clue. The file is `%LOCALAPPDATA%\i_am_plugin_wry\last-error.log`, and a successful
/// open removes it.
fn report_error(error: &EditorError) {
	eprintln!("i_am_plugin_wry: WebView2 editor failed: {error}");

	let path = error_log_path();
	if let Some(directory) = path.parent() {
		let _ = std::fs::create_dir_all(directory);
	}
	let _ = std::fs::write(path, format!("{error}\n"));
	note(&format!("error: {error}"));
}

/// Append a line to the open log next to the error log.
///
/// A host swallows what a plug-in's GUI open returns, so a blank editor in a DAW
/// leaves nothing to read. This records how far an open got: whether it was called,
/// which profile folder it chose, and whether WebView2 came up.
fn note(message: &str) {
	use std::io::Write;

	let path = error_log_path().with_file_name("open.log");
	if let Some(directory) = path.parent() {
		let _ = std::fs::create_dir_all(directory);
	}

	if let Ok(mut file) = std::fs::OpenOptions::new()
		.create(true)
		.append(true)
		.open(&path)
	{
		let _ = writeln!(file, "{message}");
	}
}

fn clear_error_log() {
	let _ = std::fs::remove_file(error_log_path());
}

/// A GUI that draws a plug-in's editor as an HTML page in a WebView2 control.
///
/// The page sends strings to the processor with:
///
/// ```js
/// window.ipc.postMessage("hello");
/// ```
///
/// and the closure given to [new](Self::new) decides what message (if any) that
/// becomes. Everything else -- a request/response bridge for parameters, say -- is
/// ordinary Rust and JavaScript on top of this one seam.
pub struct WryEditor<P: Plugin> {
	html: String,
	/// Taken by `open`: the IPC handler is called from the webview's thread and has
	/// to own the mapper, not borrow it from an editor the host also owns.
	on_ipc: Option<IpcMapper<P>>,
	webview: Option<wry::WebView>,
	/// The logical size the page is laid out for.
	size: WindowSize,
	/// Where WebView2 keeps its profile and caches, if the plug-in picks a place.
	data_directory: Option<std::path::PathBuf>,
	/// A URL to load instead of [html](Self::html), for pages served over a
	/// custom protocol.
	url: Option<String>,
	/// Custom protocols the page is served from, by name.
	protocols: Vec<(String, ProtocolHandler)>,
	_plugin: PhantomData<fn() -> P>,
}

impl<P: Plugin> WryEditor<P> {
	/// Build an editor that shows `html` and turns each IPC string into a message.
	pub fn new(
		html: impl Into<String>,
		on_ipc: impl Fn(&str) -> Option<P::Message> + 'static,
	) -> Self {
		Self {
			html: html.into(),
			on_ipc: Some(Box::new(on_ipc)),
			webview: None,
			size: WindowSize::default(),
			data_directory: None,
			url: None,
			protocols: Vec::new(),
			_plugin: PhantomData,
		}
	}

	/// Keep the WebView2 profile in `path` instead of the process default.
	///
	/// A host whose process directory is read-only, or one that wants each plug-in's
	/// browser state under its own name, sets this. It also makes the smoke example
	/// self-contained: the default lives under `%LOCALAPPDATA%`.
	pub fn with_data_directory(mut self, path: impl Into<std::path::PathBuf>) -> Self {
		self.data_directory = Some(path.into());
		self
	}

	/// Load `url` instead of the HTML given to [new](Self::new).
	///
	/// Use this with [with_custom_protocol](Self::with_custom_protocol): the page is
	/// then served from its own origin, which is what lets it `fetch` its state from
	/// the same protocol instead of being blocked as cross-origin.
	pub fn with_url(mut self, url: impl Into<String>) -> Self {
		self.url = Some(url.into());
		self
	}

	/// Answer requests for `name` from `handler`.
	///
	/// The handler is called off the UI thread with the request path, and must be
	/// self-contained: a parameter map (atomics) or a prebuilt page is fine, a
	/// thread-bound widget is not.
	pub fn with_custom_protocol(
		mut self,
		name: impl Into<String>,
		handler: impl Fn(&str) -> ProtocolResponse + Send + Sync + 'static,
	) -> Self {
		self.protocols.push((name.into(), Box::new(handler)));
		self
	}

	/// The bounds a WebView2 control is laid out with.
	///
	/// They are logical: wry asks the window for its DPI and converts to physical
	/// pixels itself, exactly as it does for the iced backend's window.
	fn bounds(size: WindowSize) -> wry::Rect {
		wry::Rect {
			position: LogicalPosition::new(0.0, 0.0).into(),
			size: LogicalSize::new(size.width as f64, size.height as f64).into(),
		}
	}

	fn webview(&self) -> Result<&wry::WebView, EditorError> {
		self.webview
			.as_ref()
			.ok_or_else(|| "the wry editor has no webview".into())
	}
}

/// Turn one of our protocol answers into what wry's protocol callback returns.
fn to_wry(response: ProtocolResponse) -> wry::http::Response<std::borrow::Cow<'static, [u8]>> {
	wry::http::Response::builder()
		.status(response.status)
		.header(wry::http::header::CONTENT_TYPE, response.content_type)
		// A plug-in's page is embedded in the binary, and a host that keeps the
		// browser cache from an older version would run yesterday's script against
		// today's routes. Nothing here is worth caching.
		.header(wry::http::header::CACHE_CONTROL, "no-store")
		.body(response.body.into())
		.expect("a response we built ourselves is well formed")
}

impl<P: Plugin> Editor<P> for WryEditor<P> {
	fn open(&mut self, context: EditorContext<P>) -> Result<(), EditorError> {
		let EditorContext {
			parent,
			size,
			scale_factor,
			title,
			messages,
			..
		} = context;

		self.size = size;

		note(&format!(
			"open: {}x{} scale={:?}",
			size.width, size.height, scale_factor
		));

		let Some(on_ipc) = self.on_ipc.take() else {
			return Err("the wry editor was opened more than once".into());
		};

		let html = std::mem::take(&mut self.html);
		let sender: Sender<P::Message> = messages;

		// A WebContext may borrow the builder, so it has to outlive the build call.
		// The data folder has to be writable: a plug-in loaded from Program Files
		// cannot use WebView2's default, which sits next to the host's executable.
		let data_directory = self
			.data_directory
			.clone()
			.or_else(default_data_directory);
		note(&format!("profile: {data_directory:?}"));

		let mut web_context = data_directory
			.as_ref()
			.map(|path| wry::WebContext::new(Some(path.clone())));
		let builder = match web_context.as_mut() {
			Some(context) => wry::WebViewBuilder::new_with_web_context(context),
			None => wry::WebViewBuilder::new(),
		};

		let mut builder = builder;
		for (name, handler) in std::mem::take(&mut self.protocols) {
			builder = builder.with_custom_protocol(name, move |_id, request| {
				to_wry(handler(request.uri().path()))
			});
		}

		// A page served from a custom protocol has to be loaded by URL: that is its
		// origin, and a fetch of the same protocol is then same-origin.
		let builder = match self.url.take() {
			Some(url) => builder.with_url(url),
			None => builder.with_html(html),
		};

		let webview = builder
			// The page is the whole of the control, and the host owns the sizing:
			// the title is only a suggestion, and WebView2 has no chrome of its own.
			.with_bounds(Self::bounds(size))
			.with_ipc_handler(move |request: wry::http::Request<String>| {
				if let Some(message) = on_ipc(request.body()) {
					// The receiver is drained on the host's main thread by the
					// wrapper, just like a message an iced view produces.
					let _ = sender.send(message);
				}
			})
			.build_as_child(&parent)
			.map_err(|error| {
				let error = EditorError::new(error);
				report_error(&error);
				error
			})?;

		webview.set_visible(true).map_err(|error| {
			let error = EditorError::new(error);
			report_error(&error);
			error
		})?;

		clear_error_log();
		note("open: ok");

		// Keep the title around only to make the intent explicit: WebView2 is a
		// child control, so its chrome (and title) belongs to the host's window.
		let _ = title;

		self.webview = Some(webview);
		Ok(())
	}

	fn show(&mut self) -> Result<(), EditorError> {
		self.webview()?.set_visible(true).map_err(EditorError::new)
	}

	fn hide(&mut self) -> Result<(), EditorError> {
		self.webview()?.set_visible(false).map_err(EditorError::new)
	}

	fn close(&mut self) {
		// Dropping the web view closes its controller and destroys the child
		// window; there is nothing else to unwind.
		let _ = self.webview.take();
	}

	fn set_parent(&mut self, parent: ParentWindow) -> Result<(), EditorError> {
		if let RawWindowHandle::Win32(handle) = parent.as_raw() {
			self.webview()?
				.reparent(handle.hwnd.get())
				.map_err(EditorError::new)?;
		}

		Ok(())
	}

	fn set_scale_factor(&mut self, _scale_factor: f64) -> Result<(), EditorError> {
		// The page is a child control: it takes the DPI of the window it lives in,
		// and wry converts the logical bounds with that DPI. The host's separate
		// scale factor is already reflected in the size the wrapper hands us.
		Ok(())
	}

	fn set_size(&mut self, size: WindowSize) -> Result<(), EditorError> {
		self.size = size;

		if self.webview.is_some() {
			self.webview()?
				.set_bounds(Self::bounds(size))
				.map_err(EditorError::new)?;
		}

		Ok(())
	}
}
