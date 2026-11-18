//! The WebView implementation for Linux and macOS.
//!
//! Both platforms hand a plug-in a window of the host's own toolkit: an X11 window
//! on Linux, an `NSView` on macOS. wry turns either one into a child view without a
//! window of ours in between, which is the whole difference from the Windows backend
//! (WebView2 wants a container window that outlives its controller).
//!
//! Linux is WebKitGTK, so this module also owns the GTK side of the deal: `gtk::init`
//! before the view is built, and `gtk::main_iteration_do` from the host's GUI timer,
//! because a plug-in never owns the host's main loop. A host that offers no
//! `clap_plugin_timer` leaves the page without its own event loop, which shows up as
//! a page that paints once and then stops reacting. macOS runs its loop for the
//! whole process, so it needs neither.
//!
//! Only X11 embedding is supported on Linux: Wayland has no cross-process
//! child-window protocol, so there is nothing to embed into and this backend refuses
//! the window with an error instead of letting wry panic.

use std::marker::PhantomData;
use std::path::PathBuf;

use crossbeam_channel::Sender;
use wry::dpi::{PhysicalPosition, PhysicalSize};

use super::{
	ProtocolRequest, ProtocolResponse, install_panic_hook, panic_payload, report_error, to_wry,
};
use crate::{Editor, EditorContext, EditorError, ParentWindow, Plugin, WindowSize};

/// What turns an IPC string from the page into a processor message.
type IpcMapper<P> = Box<dyn Fn(&str) -> Option<<P as crate::Processor>::Message>>;

/// One route of a page served over a custom protocol.
type ProtocolHandler = Box<dyn Fn(&ProtocolRequest<'_>) -> ProtocolResponse + Send + Sync>;

/// A GUI that draws a plug-in's editor as an HTML page in the host's window.
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
	/// Taken by `open`: the IPC handler is called from the web view's thread and has
	/// to own the mapper, not borrow it from an editor the host also owns.
	on_ipc: Option<IpcMapper<P>>,
	/// The view, declared before the context it came from. Fields are dropped in
	/// declaration order, and a WebKit view outliving its context loses its custom
	/// protocol handlers.
	webview: Option<wry::WebView>,
	context: Option<wry::WebContext>,
	/// The logical size the page is laid out for.
	size: WindowSize,
	/// The host's scale factor, used to turn logical sizes into device pixels.
	scale: f64,
	/// A URL to load instead of [html](Self::html), for pages served over a
	/// custom protocol.
	url: Option<String>,
	/// Custom protocols the page is served from, by name.
	protocols: Vec<(String, ProtocolHandler)>,
	/// Where the browser keeps its data, if the plug-in picks a place.
	data_directory: Option<PathBuf>,
	_plugin: PhantomData<fn() -> P>,
}

impl<P: Plugin> WryEditor<P> {
	/// Build an editor that shows `html` and turns each IPC string into a message.
	pub fn new(
		html: impl Into<String>,
		on_ipc: impl Fn(&str) -> Option<P::Message> + 'static,
	) -> Self {
		install_panic_hook();

		Self {
			html: html.into(),
			on_ipc: Some(Box::new(on_ipc)),
			webview: None,
			context: None,
			size: WindowSize::default(),
			scale: 1.0,
			url: None,
			protocols: Vec::new(),
			data_directory: None,
			_plugin: PhantomData,
		}
	}

	/// Keep the browser's data in `path` instead of its own default.
	///
	/// On Linux this is WebKitGTK's website data directory; macOS keeps its data per
	/// bundle identifier and has no per-view equivalent, so the path is only logged
	/// there.
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
		self,
		name: impl Into<String>,
		handler: impl Fn(&str) -> ProtocolResponse + Send + Sync + 'static,
	) -> Self {
		self.with_custom_protocol_request(name, move |request| handler(request.path))
	}

	/// Answer requests for `name` with the method, path and body the page sent.
	///
	/// This is what a route that takes an upload needs: the page can POST bytes to
	/// its own origin, and the handler decodes them without going through IPC.
	pub fn with_custom_protocol_request(
		mut self,
		name: impl Into<String>,
		handler: impl Fn(&ProtocolRequest<'_>) -> ProtocolResponse + Send + Sync + 'static,
	) -> Self {
		self.protocols.push((name.into(), Box::new(handler)));
		self
	}

	/// The bounds the view is laid out with, in device pixels.
	///
	/// The host's window is sized in device pixels and its scale does not always
	/// match the monitor's, so a logical size would be scaled twice on some hosts.
	fn bounds(size: WindowSize, scale: f64) -> wry::Rect {
		let width = (size.width as f64 * scale).round().clamp(1.0, 32768.0) as u32;
		let height = (size.height as f64 * scale).round().clamp(1.0, 32768.0) as u32;
		wry::Rect {
			position: PhysicalPosition::new(0, 0).into(),
			size: PhysicalSize::new(width, height).into(),
		}
	}

	fn webview(&self) -> Result<&wry::WebView, EditorError> {
		self.webview
			.as_ref()
			.ok_or_else(|| "the wry editor has no webview".into())
	}
}

impl<P: Plugin> Editor<P> for WryEditor<P> {
	fn open(&mut self, context: EditorContext<P>) -> Result<(), EditorError> {
		let EditorContext {
			parent,
			size,
			scale_factor,
			messages,
			..
		} = context;

		self.size = size;
		self.scale = scale_factor.unwrap_or(1.0).clamp(0.25, 4.0);

		crate::log::note(&format!(
			"open: {}x{} scale={:?}",
			size.width, size.height, scale_factor
		));

		// GTK has to be up before the view exists, and the host's window has to be
		// one wry can embed into.
		if let Err(error) = platform::prepare(parent) {
			report_error(&error);
			return Err(error);
		}

		// The host's window is the truth for the visible area: a DAW may size the
		// plug-in window differently from the design size the wrapper reports.
		if let Some((width, height)) = crate::editor::client_size(parent) {
			self.size = WindowSize::new((width / self.scale) as f32, (height / self.scale) as f32);
		}

		let size = self.size;
		let scale = self.scale;

		let Some(on_ipc) = self.on_ipc.take() else {
			return Err("the wry editor was opened more than once".into());
		};

		let html = std::mem::take(&mut self.html);
		let protocols = std::mem::take(&mut self.protocols);
		let url = self.url.take();
		let data_directory = self.data_directory.clone();
		let sender: Sender<P::Message> = messages;

		let opened = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
			move || -> Result<(wry::WebView, wry::WebContext), EditorError> {
				// One context per editor. WebKitGTK keeps custom protocol schemes on
				// the context, so a shared one would refuse the second editor's
				// routes as duplicates of the first one's.
				let mut web_context = wry::WebContext::new(data_directory);
				let mut builder = wry::WebViewBuilder::new_with_web_context(&mut web_context);

				for (name, handler) in protocols {
					builder = builder.with_custom_protocol(name, move |_id, request| {
						let request = ProtocolRequest {
							method: request.method().as_str(),
							path: request.uri().path(),
							query: request.uri().query(),
							body: request.body(),
						};

						to_wry(handler(&request))
					});
				}

				// A page served from a custom protocol has to be loaded by URL: that
				// is its origin, and a fetch of the same protocol is then same-origin.
				let builder = match url {
					Some(url) => builder.with_url(url),
					None => builder.with_html(html),
				};

				let webview = builder
					// The page is the whole area of the host's window: it has no
					// chrome of its own to leave room for.
					.with_bounds(Self::bounds(size, scale))
					.with_ipc_handler(move |request: wry::http::Request<String>| {
						if let Some(message) = on_ipc(request.body()) {
							// The receiver is drained on the host's main thread by the
							// wrapper, which then hands the message to the processor.
							let _ = sender.send(message);
						}
					})
					.build_as_child(&parent)
					.map_err(EditorError::new)?;

				Ok((webview, web_context))
			},
		));

		match opened {
			Ok(Ok((webview, context))) => {
				if let Err(error) = webview.set_visible(true) {
					let error = EditorError::new(error);
					report_error(&error);
					return Err(error);
				}

				self.webview = Some(webview);
				self.context = Some(context);
				crate::log::clear_error();
				crate::log::note("open: ok");
				Ok(())
			}
			Ok(Err(error)) => {
				report_error(&error);
				Err(error)
			}
			Err(payload) => {
				let message = panic_payload(payload.as_ref());
				crate::log::note(&format!("panic in open: {message}"));
				let error =
					EditorError::from(format!("the web view panicked while opening: {message}"));
				report_error(&error);
				Err(error)
			}
		}
	}

	fn show(&mut self) -> Result<(), EditorError> {
		crate::log::note("show");
		let webview = self.webview()?;
		webview.set_visible(true).map_err(EditorError::new)?;
		// The host focuses its own window, not the child view; without this the page
		// does not see the keyboard until the user clicks it.
		let _ = webview.focus();
		Ok(())
	}

	fn hide(&mut self) -> Result<(), EditorError> {
		crate::log::note("hide");
		self.webview()?.set_visible(false).map_err(EditorError::new)
	}

	fn close(&mut self) {
		crate::log::note("close: start");
		// The view goes first: dropping the context under a live view is what makes
		// a WebKit view lose its protocol handlers.
		let _ = self.webview.take();
		let _ = self.context.take();
		crate::log::note("close: done");
	}

	fn set_parent(&mut self, parent: ParentWindow) -> Result<(), EditorError> {
		// The view is a child of the window it was built in and wry has no way to
		// move it. A host that reparents has to close and reopen the editor, which is
		// what the wrapper does when the GUI is re-created.
		crate::log::note(&format!("set_parent: {:?} (not moveable)", parent.as_raw()));
		Ok(())
	}

	fn set_scale_factor(&mut self, scale_factor: f64) -> Result<(), EditorError> {
		self.scale = if scale_factor.is_finite() {
			scale_factor.clamp(0.25, 4.0)
		} else {
			1.0
		};
		platform::log(&format!("set_scale: {}", self.scale));

		if self.webview.is_some() {
			self.webview()?
				.set_bounds(Self::bounds(self.size, self.scale))
				.map_err(EditorError::new)?;
		}

		Ok(())
	}

	fn set_size(&mut self, size: WindowSize) -> Result<(), EditorError> {
		self.size = size;

		if self.webview.is_some() {
			self.webview()?
				.set_bounds(Self::bounds(size, self.scale))
				.map_err(EditorError::new)?;
			// A resized GTK view only repaints inside an iteration.
			platform::pump();
		}

		Ok(())
	}

	fn pump(&mut self) {
		platform::pump();
	}
}

/// The part of the backend that differs between X11 and AppKit.
#[cfg(target_os = "linux")]
mod platform {
	use raw_window_handle::RawWindowHandle;

	use crate::{EditorError, ParentWindow};

	/// Bring up GTK on this thread and check that the window can be embedded into.
	pub(super) fn prepare(parent: ParentWindow) -> Result<(), EditorError> {
		if !matches!(parent.as_raw(), RawWindowHandle::Xlib(_)) {
			return Err(
				"the wry backend needs an X11 window on Linux; Wayland has no child-window 				 protocol to embed into"
					.into(),
			);
		}

		// A plug-in's GUI runs on the host's main thread, and GTK may already be up
		// there (a GTK host, or a second editor). Only a failure of the first init is
		// fatal.
		if !gtk::is_initialized() && gtk::init().is_err() {
			return Err("GTK could not be initialised on the GUI thread".into());
		}

		Ok(())
	}

	/// Run whatever the toolkit has queued.
	///
	/// The loop is bounded: a page that keeps producing work must not starve the
	/// host's own message pump, and the timer comes back in 33 ms anyway.
	pub(super) fn pump() {
		for _ in 0..64 {
			if !gtk::events_pending() {
				break;
			}
			gtk::main_iteration_do(false);
		}
	}

	pub(super) fn log(message: &str) {
		crate::log::note(message);
	}
}

/// The part of the backend that differs between X11 and AppKit.
#[cfg(target_os = "macos")]
mod platform {
	use raw_window_handle::RawWindowHandle;

	use crate::{EditorError, ParentWindow};

	/// Check that the host gave us a view to embed into.
	///
	/// macOS runs one application event loop for the whole process and the host's
	/// NSApplication is already pumping it, so there is nothing to initialise.
	pub(super) fn prepare(parent: ParentWindow) -> Result<(), EditorError> {
		if !matches!(parent.as_raw(), RawWindowHandle::AppKit(_)) {
			return Err("the wry backend needs an NSView on macOS".into());
		}

		Ok(())
	}

	/// macOS has no separate loop to advance.
	pub(super) fn pump() {}

	pub(super) fn log(message: &str) {
		crate::log::note(message);
	}
}
