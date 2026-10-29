//! The WebView2 implementation.

use std::cell::RefCell;
use std::marker::PhantomData;
use std::path::PathBuf;

use crossbeam_channel::Sender;
use i_am_plugin::{Editor, EditorContext, EditorError, ParentWindow, Plugin, WindowSize};
use windows_sys::Win32::Foundation::{GetLastError, HWND};
use windows_sys::Win32::Graphics::Gdi::COLOR_WINDOW;
use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
use windows_sys::Win32::UI::WindowsAndMessaging::{
	CreateWindowExW, DefWindowProcW, DestroyWindow, GetClientRect, RegisterClassW, SetParent,
	SetWindowPos, ShowWindow, SWP_NOACTIVATE, SWP_NOZORDER, SW_HIDE, SW_SHOW, WNDCLASSW,
	WS_CHILD, WS_CLIPCHILDREN, WS_VISIBLE,
};
use wry::dpi::{LogicalPosition, LogicalSize};

use crate::ProtocolResponse;

/// What turns an IPC string from the page into a processor message.
type IpcMapper<P> = Box<dyn Fn(&str) -> Option<<P as i_am_plugin::Processor>::Message>>;

/// One route of a page served over a custom protocol.
type ProtocolHandler = Box<dyn Fn(&str) -> ProtocolResponse + Send + Sync>;

/// The window class the WebView2 child lives in. The host's editor window only
/// has to host one normal child; WebView2 creates its controller inside it.
fn host_class() -> &'static Vec<u16> {
	static CLASS: std::sync::OnceLock<Vec<u16>> = std::sync::OnceLock::new();

	CLASS.get_or_init(|| "i_am_plugin_wry_host\0".encode_utf16().collect())
}

fn register_host_class() {
	static ONCE: std::sync::OnceLock<()> = std::sync::OnceLock::new();

	ONCE.get_or_init(|| {
		let class = host_class();
		let window_class = WNDCLASSW {
			style: 0,
			lpfnWndProc: Some(DefWindowProcW),
			cbClsExtra: 0,
			cbWndExtra: 0,
			hInstance: unsafe { GetModuleHandleW(std::ptr::null()) },
			hIcon: std::ptr::null_mut(),
			hCursor: std::ptr::null_mut(),
			hbrBackground: COLOR_WINDOW as usize as _,
			lpszMenuName: std::ptr::null(),
			lpszClassName: class.as_ptr(),
		};

		// A failure here is reported by the CreateWindowExW that follows; the
		// class may also already exist from an earlier instance, which is fine.
		unsafe {
			RegisterClassW(&window_class);
		}
	});
}

fn create_host_window(
	parent: *mut std::ffi::c_void,
	size: WindowSize,
	scale: f64,
) -> Result<HWND, EditorError> {
	register_host_class();

	let width = (size.width as f64 * scale).round() as i32;
	let height = (size.height as f64 * scale).round() as i32;
	let instance = unsafe { GetModuleHandleW(std::ptr::null()) };
	let window = unsafe {
		CreateWindowExW(
			0,
			host_class().as_ptr(),
			std::ptr::null(),
			WS_CHILD | WS_VISIBLE | WS_CLIPCHILDREN,
			0,
			0,
			width,
			height,
			parent,
			std::ptr::null_mut(),
			instance,
			std::ptr::null_mut(),
		)
	};

	if window.is_null() {
		return Err(format!("CreateWindowExW failed: {}", unsafe { GetLastError() }).into());
	}

	Ok(window)
}

thread_local! {
	/// One WebView2 environment per UI thread. Creating a new environment for
	/// every editor against the same user-data folder is what made the first
	/// open of every plug-in risky: WebView2 does not like two environments on
	/// one profile, and a host that swallows GUI errors can take the whole DAW
	/// down with it. WebView2 supports many WebViews in one environment, so all
	/// editors on this thread share it.
	static WEB_CONTEXT: RefCell<Option<wry::WebContext>> = const { RefCell::new(None) };
}

/// Where WebView2 keeps its profile when the plug-in does not choose.
///
/// It must be writable. WebView2's default is next to the executable, and a DAW's
/// executable is usually under Program Files, where the browser cannot create its
/// profile; a host that loads the plug-in from there gets a blank editor. A folder
/// under the user's local application data always is writable, so that is the fallback.
///
/// A profile per process. A single shared user-data folder crashes the WebView2
/// controller when more than one process (or more than one environment) opens
/// it at once; that failure is native, so no Rust panic hook can record it.
/// The directory is small and can be cleaned when the DAW is closed.
fn default_data_directory() -> Option<PathBuf> {
	let mut candidates = Vec::with_capacity(2);
	let process = std::process::id();

	if let Some(base) = std::env::var_os("LOCALAPPDATA").map(PathBuf::from) {
		candidates.push(
			base.join("i_am_plugin_wry")
				.join(format!("WebView2-{process}")),
		);
	}

	// The temp directory is the fallback: it is writable even when a profile folder
	// is locked down (or when the plug-in runs under a restrictive sandbox).
	candidates.push(
		std::env::temp_dir()
			.join("i_am_plugin_wry")
			.join(format!("WebView2-{process}")),
	);

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

/// Record panic messages (including ones that happen on WebView2 threads) in
/// the same log the open path writes. A host cannot show a Rust panic, and a
/// panic crossing the host's FFI boundary is a crash.
fn install_panic_hook() {
	static ONCE: std::sync::OnceLock<()> = std::sync::OnceLock::new();

	ONCE.get_or_init(|| {
		let previous = std::panic::take_hook();
		std::panic::set_hook(Box::new(move |info| {
			let location = info
				.location()
				.map(|location| format!("{}:{}", location.file(), location.line()))
				.unwrap_or_default();
			note(&format!("panic: {info} at {location}"));
			previous(info);
		}));
	});
}

fn panic_payload(payload: &(dyn std::any::Any + Send)) -> String {
	if let Some(message) = payload.downcast_ref::<&str>() {
		(*message).to_string()
	} else if let Some(message) = payload.downcast_ref::<String>() {
		message.clone()
	} else {
		"non-string panic".to_string()
	}
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
	/// The Win32 child window the WebView2 lives in. The host's editor window is
	/// its parent; the WebView2 controller's parent is this window, which is a
	/// plain window class of our own.
	window: Option<HWND>,
	/// The logical size the page is laid out for.
	size: WindowSize,
	/// The host's scale factor, used to turn logical sizes into Win32 pixels.
	scale: f64,
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
		install_panic_hook();

		Self {
			html: html.into(),
			on_ipc: Some(Box::new(on_ipc)),
			webview: None,
			window: None,
			size: WindowSize::default(),
			scale: 1.0,
			data_directory: None,
			url: None,
			protocols: Vec::new(),
			_plugin: PhantomData,
		}
	}

	/// Keep the WebView2 profile in `path` instead of the process default.
	///
	/// The first editor on a UI thread creates the shared WebView2 context with
	/// this path; later editors reuse that context, so only the first path wins.
	/// It also makes the smoke example self-contained: the default lives under
	/// `%LOCALAPPDATA%`.
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
	let built = wry::http::Response::builder()
		.status(response.status)
		.header(wry::http::header::CONTENT_TYPE, response.content_type)
		// A plug-in's page is embedded in the binary, and a host that keeps the
		// browser cache from an older version would run yesterday's script against
		// today's routes. Nothing here is worth caching.
		.header(wry::http::header::CACHE_CONTROL, "no-store")
		.body(response.body.into());

	// An invalid response used to panic inside the WebView2 protocol callback;
	// a panic crossing that FFI boundary takes the host down. Fall back instead.
	match built {
		Ok(response) => response,
		Err(_) => wry::http::Response::builder()
			.status(500)
			.body(std::borrow::Cow::Borrowed(&[][..]))
			.expect("a fixed empty response is well formed"),
	}
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
		self.scale = scale_factor.unwrap_or(1.0).max(0.1);

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

		// WebView2 wants a COM apartment on the thread it is created from. The
		// host usually owns this thread's COM mode; logging it tells a crash in
		// CreateCoreWebView2Environment apart from one in the WebView itself.
		#[cfg(windows)]
		{
			use windows_sys::Win32::System::Com::{CoInitializeEx, COINIT_APARTMENTTHREADED};
			// SAFETY: a null reserved pointer is what CoInitializeEx expects; the
			// host's existing apartment, if any, is left in place.
			let result = unsafe {
				CoInitializeEx(std::ptr::null_mut(), COINIT_APARTMENTTHREADED as u32)
			};
			note(&format!("com: {result:#x}"));
		}

		let parent_hwnd = parent
			.as_win32_hwnd()
			.ok_or("the wry editor needs a Win32 parent window")?;

		// The host's window is the truth for the visible area: a DAW may size the
		// plug-in window differently from the design size the wrapper reports.
		// The child and the WebView2 fill that client rectangle.
		let mut client = unsafe { std::mem::zeroed() };
		let size = if unsafe { GetClientRect(parent_hwnd, &mut client) } != 0
			&& client.right > client.left
			&& client.bottom > client.top
		{
			WindowSize::new(
				(client.right - client.left) as f32 / self.scale as f32,
				(client.bottom - client.top) as f32 / self.scale as f32,
			)
		} else {
			size
		};
		self.size = size;

		let window = create_host_window(parent_hwnd, size, self.scale)?;
		note(&format!("window: created {}x{} logical", size.width, size.height));
		let outer_parent = ParentWindow::from_win32_hwnd(window)
			.ok_or("the host window handle was invalid")?;

		let protocols = std::mem::take(&mut self.protocols);
		let url = self.url.take();

		// The WebView2 runs inside our own child window; the host's editor window
		// only has to host that child. Creating the controller directly in the
		// host's window is what crashed some DAWs at build time.
		let opened = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
			WEB_CONTEXT.with(move |shared| {
				let mut shared = shared.borrow_mut();

				if shared.is_none() {
					note("context: creating");
					*shared = Some(wry::WebContext::new(data_directory.clone()));
					note("context: created");
				}

				let context = shared
					.as_mut()
					.expect("the WebView2 context was just created");
				note("build: creating");
				let mut builder = wry::WebViewBuilder::new_with_web_context(context);

				for (name, handler) in protocols {
					builder = builder.with_custom_protocol(name, move |_id, request| {
						to_wry(handler(request.uri().path()))
					});
				}

				// A page served from a custom protocol has to be loaded by URL: that
				// is its origin, and a fetch of the same protocol is then same-origin.
				let builder = match url {
					Some(url) => builder.with_url(url),
					None => builder.with_html(html),
				};

				builder
					// The page is the whole of the control, and the host owns the
					// sizing: the title is only a suggestion, and WebView2 has no
					// chrome of its own.
					.with_bounds(Self::bounds(size))
					.with_ipc_handler(move |request: wry::http::Request<String>| {
						if let Some(message) = on_ipc(request.body()) {
							// The receiver is drained on the host's main thread by the
							// wrapper, just like a message an iced view produces.
							let _ = sender.send(message);
						}
					})
					.build_as_child(&outer_parent)
			})
		}));

		let webview = match opened {
			Ok(Ok(webview)) => webview,
			Ok(Err(error)) => {
				unsafe {
					DestroyWindow(window);
				}
				let error = EditorError::new(error);
				report_error(&error);
				return Err(error);
			}
			Err(payload) => {
				unsafe {
					DestroyWindow(window);
				}
				let message = panic_payload(payload.as_ref());
				note(&format!("panic in open: {message}"));
				let error = EditorError::from(format!("WebView2 panicked while opening: {message}"));
				report_error(&error);
				return Err(error);
			}
		};

		note("build: created");

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
		self.window = Some(window);
		Ok(())
	}

	fn show(&mut self) -> Result<(), EditorError> {
		if let Some(window) = self.window {
			unsafe {
				ShowWindow(window, SW_SHOW);
			}
		}
		self.webview()?.set_visible(true).map_err(EditorError::new)
	}

	fn hide(&mut self) -> Result<(), EditorError> {
		if let Some(window) = self.window {
			unsafe {
				ShowWindow(window, SW_HIDE);
			}
		}
		self.webview()?.set_visible(false).map_err(EditorError::new)
	}

	fn close(&mut self) {
		// Drop the WebView2 controller before destroying the window it lives in.
		let _ = self.webview.take();

		if let Some(window) = self.window.take() {
			unsafe {
				DestroyWindow(window);
			}
		}
	}

	fn set_parent(&mut self, parent: ParentWindow) -> Result<(), EditorError> {
		if let (Some(window), Some(parent)) = (self.window, parent.as_win32_hwnd()) {
			unsafe {
				SetParent(window, parent);
			}
		}

		Ok(())
	}

	fn set_scale_factor(&mut self, scale_factor: f64) -> Result<(), EditorError> {
		self.scale = scale_factor.max(0.1);

		// The WebView2 parent is a Win32 window, so its pixel size has to follow
		// the DPI; the WebView2 itself keeps taking logical bounds.
		if let Some(window) = self.window {
			let width = (self.size.width as f64 * self.scale).round() as i32;
			let height = (self.size.height as f64 * self.scale).round() as i32;
			unsafe {
				SetWindowPos(
					window,
					std::ptr::null_mut(),
					0,
					0,
					width,
					height,
					SWP_NOZORDER | SWP_NOACTIVATE,
				);
			}
		}

		if self.webview.is_some() {
			self.webview()?
				.set_bounds(Self::bounds(self.size))
				.map_err(EditorError::new)?;
		}

		Ok(())
	}

	fn set_size(&mut self, size: WindowSize) -> Result<(), EditorError> {
		self.size = size;

		if let Some(window) = self.window {
			let width = (size.width as f64 * self.scale).round() as i32;
			let height = (size.height as f64 * self.scale).round() as i32;
			unsafe {
				SetWindowPos(
					window,
					std::ptr::null_mut(),
					0,
					0,
					width,
					height,
					SWP_NOZORDER | SWP_NOACTIVATE,
				);
			}
		}

		if self.webview.is_some() {
			self.webview()?
				.set_bounds(Self::bounds(size))
				.map_err(EditorError::new)?;
		}

		Ok(())
	}
}
