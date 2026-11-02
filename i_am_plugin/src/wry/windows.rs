//! The WebView2 implementation.

use std::cell::RefCell;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use crate::log::{clear_error, note};
use crate::{Editor, EditorContext, EditorError, ParentWindow, Plugin, WindowSize};
use crossbeam_channel::Sender;
use windows_sys::Win32::Foundation::{
	ERROR_ACCESS_DENIED, GetLastError, HWND, LPARAM, LRESULT, SetLastError, WPARAM,
};
use windows_sys::Win32::Graphics::Gdi::COLOR_WINDOW;
use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
use windows_sys::Win32::UI::HiDpi::GetDpiForWindow;
use windows_sys::Win32::UI::WindowsAndMessaging::{
	CreateWindowExW, DefWindowProcW, DestroyWindow, GWLP_WNDPROC, GetClientRect, GetParent,
	IsWindow, KillTimer, RegisterClassW, SW_HIDE, SW_SHOW, SWP_NOACTIVATE, SWP_NOZORDER, SetParent,
	SetTimer, SetWindowLongPtrW, SetWindowPos, ShowWindow, WM_TIMER, WNDCLASSW, WS_CHILD,
	WS_CLIPCHILDREN, WS_VISIBLE,
};
use wry::WebViewBuilderExtWindows;
use wry::dpi::{PhysicalPosition, PhysicalSize};

/// COM already initialized on this thread with another apartment model.
const RPC_E_CHANGED_MODE: i32 = 0x8001_0106u32 as i32;

use super::{ProtocolRequest, ProtocolResponse};

/// What turns an IPC string from the page into a processor message.
type IpcMapper<P> = Box<dyn Fn(&str) -> Option<<P as crate::Processor>::Message>>;

/// One route of a page served over a custom protocol.
type ProtocolHandler = Box<dyn Fn(&ProtocolRequest<'_>) -> ProtocolResponse + Send + Sync>;

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

/// What the window's timer procedure needs to follow the host's window.
///
/// A host may resize its window without telling the plug-in. A Win32 timer on
/// our own child window is delivered by whichever message loop the host runs,
/// so it works even when the host offers no CLAP timer at all.
#[derive(Copy, Clone)]
struct EditorHook {
	editor: *mut (),
	follow: unsafe fn(*mut ()),
}

thread_local! {
	static EDITOR_HOOK: RefCell<Option<EditorHook>> = const { RefCell::new(None) };
}

const FOLLOW_TIMER: usize = 0x1A33;

unsafe extern "system" fn follow_timer(
	hwnd: HWND,
	message: u32,
	wparam: WPARAM,
	lparam: LPARAM,
) -> LRESULT {
	if message == WM_TIMER && wparam == FOLLOW_TIMER {
		let hook = EDITOR_HOOK.with(|slot| *slot.borrow());

		if let Some(hook) = hook {
			// SAFETY: the pointer was stored by an open editor and cleared
			// before that editor is dropped; both happen on this thread.
			unsafe { (hook.follow)(hook.editor) };
		}

		return 0;
	}

	// SAFETY: the remaining window procedure is the class's own.
	unsafe { DefWindowProcW(hwnd, message, wparam, lparam) }
}

unsafe fn follow_editor<P: Plugin>(pointer: *mut ()) {
	// SAFETY: the pointer is the editor that stored this function.
	let editor = unsafe { &mut *(pointer as *mut WryEditor<P>) };

	let _ = editor.follow_parent();
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

/// The profile layout version. Bump it when the browser options below change,
/// so an older plug-in build with different options cannot share the folder.
const PROFILE_VERSION: u32 = 2;

/// Browser switches for the embedded WebView2.
///
/// The disk caches are bounded: a DAW session used to leave a fresh profile of
/// ten to seventy megabytes under a pid folder, and those folders accumulated for
/// over a gigabyte. The code and shader caches are turned off, and the HTTP and
/// media caches are capped. This string is also part of the profile identity:
/// change it and bump [PROFILE_VERSION].
const BROWSER_ARGS: &str = "--disable-features=msWebOOUI,msPdfOOUI,msSmartScreenProtection --autoplay-policy=no-user-gesture-required --disk-cache-size=67108864 --media-cache-size=16777216 --disable-gpu-shader-disk-cache --disable-gpu-program-cache --disable-background-networking --disable-component-update --no-first-run --disable-sync --disable-breakpad";

/// Where WebView2 keeps its profile when the plug-in does not choose.
///
/// It must be writable. WebView2's default is next to the executable, and a DAW's
/// executable is usually under Program Files, where the browser cannot create its
/// profile; a host that loads the plug-in from there gets a blank editor. A folder
/// under the user's local application data always is writable, so that is the fallback.
///
/// The folder is shared by every instance of this build, with the profile version in
/// the name. WebView2 allows a user-data folder to be shared by environments that
/// use the same options; it rejects one with different options, natively rather than
/// through a Rust error. The pid suffix made a new profile every process and session,
/// which is what grew to a gigabyte of dead folders. PROFILE_VERSION is the guard
/// for builds whose options differ.
///
/// I_AM_PLUGIN_WRY_PER_PROCESS_PROFILE=1 restores the old per-process layout if a
/// host turns out to dislike sharing.
fn profile_name() -> String {
	if std::env::var_os("I_AM_PLUGIN_WRY_PER_PROCESS_PROFILE").is_some() {
		format!("WebView2-{}", std::process::id())
	} else {
		format!("WebView2-v{PROFILE_VERSION}")
	}
}

/// A profile above this size is worth reporting, even though the browser args
/// already cap the HTTP and media caches. The rest is code/GPU cache that
/// WebView2 maintains itself.
const MAX_PROFILE_BYTES: u64 = 256 * 1024 * 1024;

fn default_data_directory() -> Option<PathBuf> {
	clean_stale_profiles_once();

	let mut candidates = Vec::with_capacity(2);

	if let Some(base) = std::env::var_os("LOCALAPPDATA").map(PathBuf::from) {
		candidates.push(base.join("i_am_plugin_wry").join(profile_name()));
	}

	// The temp directory is the fallback: it is writable even when a profile folder
	// is locked down (or when the plug-in runs under a restrictive sandbox).
	candidates.push(
		std::env::temp_dir()
			.join("i_am_plugin_wry")
			.join(profile_name()),
	);

	let directory = candidates
		.into_iter()
		.find(|directory| std::fs::create_dir_all(directory).is_ok())?;

	clean_old_profile_versions(&directory);
	log_profile_size(&directory);
	Some(directory)
}

/// Remove old profile-version folders once they have been idle for a week.
///
/// A stale profile can be hundreds of megabytes after a browser update. The
/// current version is never touched, and a folder with a live WebView2 process
/// keeps files open, so the removal simply fails and is retried next time.
fn clean_old_profile_versions(current: &Path) {
	use std::sync::Once;
	use std::time::{Duration, SystemTime};

	static ONCE: Once = Once::new();
	ONCE.call_once(|| {
		let Some(base) = current.parent() else {
			return;
		};
		let Some(current_name) = current.file_name().and_then(|name| name.to_str()) else {
			return;
		};
		let Ok(entries) = std::fs::read_dir(base) else {
			return;
		};
		let now = SystemTime::now();

		for entry in entries.flatten() {
			let name = entry.file_name();
			let Some(name) = name.to_str() else {
				continue;
			};
			if name == current_name || !name.starts_with("WebView2-v") {
				continue;
			}
			let Ok(metadata) = entry.metadata() else {
				continue;
			};
			let Ok(modified) = metadata.modified() else {
				continue;
			};
			let Ok(age) = now.duration_since(modified) else {
				continue;
			};
			if age < Duration::from_secs(7 * 24 * 60 * 60) {
				continue;
			}

			if std::fs::remove_dir_all(entry.path()).is_ok() {
				note(&format!("profile: removed old version {name}"));
			}
		}
	});
}

/// Record the profile size once per process; it is a diagnostic, not a hot path.
fn log_profile_size(directory: &Path) {
	use std::sync::Once;

	static ONCE: Once = Once::new();
	ONCE.call_once(|| {
		let mut bytes = 0u64;
		let mut files = 0u32;
		size_of_tree(directory, &mut bytes, &mut files);
		note(&format!(
			"profile: {} ({:.1} MiB, {files} files)",
			directory.display(),
			bytes as f64 / 1_048_576.0
		));
		if bytes > MAX_PROFILE_BYTES {
			note("profile: above the 256 MiB soft cap; old cache should be cleaned");
		}
	});
}

fn size_of_tree(directory: &Path, bytes: &mut u64, files: &mut u32) {
	let Ok(entries) = std::fs::read_dir(directory) else {
		return;
	};
	for entry in entries.flatten() {
		if *files > 20_000 {
			return;
		}
		let Ok(metadata) = entry.metadata() else {
			continue;
		};
		if metadata.is_dir() {
			size_of_tree(&entry.path(), bytes, files);
		} else {
			*bytes = bytes.saturating_add(metadata.len());
			*files = files.saturating_add(1);
		}
	}
}

/// Runs the old-profile cleanup once per process.
fn clean_stale_profiles_once() {
	static CLEANED: std::sync::Once = std::sync::Once::new();

	CLEANED.call_once(|| {
		if let Some(base) = std::env::var_os("LOCALAPPDATA").map(PathBuf::from) {
			clean_stale_profiles(&base.join("i_am_plugin_wry"));
		}
		clean_stale_profiles(&std::env::temp_dir().join("i_am_plugin_wry"));
	});
}

/// Deletes WebView2-<pid> folders whose process no longer exists.
///
/// The current folder is fixed, so every other pid folder is a leftover from an
/// older build or an older per-process profile. A live process's folder is left
/// alone, and a folder that cannot be removed (an orphaned child still holds a
/// file) is retried on the next open.
fn clean_stale_profiles(directory: &Path) {
	let Ok(entries) = std::fs::read_dir(directory) else {
		return;
	};

	for entry in entries.flatten() {
		let name = entry.file_name();
		let Some(name) = name.to_str() else {
			continue;
		};
		let Some(pid) = name
			.strip_prefix("WebView2-")
			.and_then(|pid| pid.parse::<u32>().ok())
		else {
			continue;
		};

		if pid == std::process::id() || process_is_alive(pid) {
			continue;
		}

		let path = entry.path();
		if path.is_dir() {
			let _ = std::fs::remove_dir_all(path);
		}
	}
}

/// Whether a process with this pid exists.
fn process_is_alive(pid: u32) -> bool {
	use windows_sys::Win32::Foundation::CloseHandle;
	use windows_sys::Win32::System::Threading::{OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION};

	// SAFETY: OpenProcess does not read process memory; the handle is closed.
	let handle = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
	if handle.is_null() {
		// Access denied means the process exists but is protected; do not delete
		// the profile of something that may still be using it.
		return unsafe { GetLastError() } == ERROR_ACCESS_DENIED;
	}
	unsafe { CloseHandle(handle) };
	true
}

/// Record why an editor could not open, where a host's log cannot be read.
///
/// A DAW swallows the error a plug-in's GUI open returns, so the blank window gives no
/// clue. The error log lives in the temp directory and is capped by the log module.
fn report_error(error: &EditorError) {
	eprintln!("i_am_plugin_wry: WebView2 editor failed: {error}");
	crate::log::error(&format!("error: {error}"));
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
	/// Make the page the size of the room the host's window has.
	///
	/// Called from the window's timer: a host can resize without calling
	/// set_size (some only do that on release, others never), and the client
	/// rectangle is the truth in every case.
	fn follow_parent(&mut self) -> Result<(), EditorError> {
		let Some(window) = self.window else {
			return Ok(());
		};
		let parent = unsafe { GetParent(window) };

		if parent.is_null() {
			return Ok(());
		}

		let mut client = unsafe { std::mem::zeroed() };

		if unsafe { GetClientRect(parent, &mut client) } == 0 {
			return Ok(());
		}

		let width = (client.right - client.left) as f64;
		let height = (client.bottom - client.top) as f64;

		if width < 1.0 || height < 1.0 {
			return Ok(());
		}

		let logical = WindowSize::new((width / self.scale) as f32, (height / self.scale) as f32);

		if (logical.width - self.size.width).abs() < 1.0
			&& (logical.height - self.size.height).abs() < 1.0
		{
			return Ok(());
		}

		note(&format!(
			"follow: parent is {}x{} -> {}x{}",
			width, height, logical.width, logical.height
		));

		self.set_size(logical)
	}
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

	/// The bounds a WebView2 control is laid out with.
	///
	/// They are logical: wry asks the window for its DPI and converts to physical
	/// pixels itself, exactly as it does for the iced backend's window.
	fn bounds(size: WindowSize, scale: f64) -> wry::Rect {
		// Use physical pixels here: the host window is created in physical pixels
		// too, and FL Studio's GUI scale does not always match the monitor DPI.
		// Letting wry convert a logical size would then scale the page twice.
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

	/// Tear down the WebView2 controller before the window it lives in.
	///
	/// This is used by both `close` and `Drop`: a host that drops the editor
	/// without an explicit destroy must not leave a visible child window behind.
	fn shutdown(&mut self) {
		if let Some(window) = self.window {
			unsafe {
				KillTimer(window, FOLLOW_TIMER);
				SetWindowLongPtrW(window, GWLP_WNDPROC, DefWindowProcW as *const () as isize);
			}
		}

		EDITOR_HOOK.with(|slot| *slot.borrow_mut() = None);

		if self.webview.is_none() && self.window.is_none() {
			return;
		}

		note("close: start");
		let _ = self.webview.take();

		if let Some(window) = self.window.take() {
			let destroyed = unsafe { DestroyWindow(window) };
			let last_error = if destroyed == 0 {
				unsafe { GetLastError() }
			} else {
				0
			};
			note(&format!(
				"close: DestroyWindow={destroyed} last_error={last_error}"
			));
		}

		note("close: done");
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
		self.scale = scale_factor.unwrap_or(1.0).clamp(0.25, 4.0);

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
		let data_directory = self.data_directory.clone().or_else(default_data_directory);
		note(&format!("profile: {data_directory:?}"));
		if let Some(directory) = &data_directory {
			log_profile_size(directory);
		}

		// WebView2 wants a COM apartment on the thread it is created from. The
		// host usually owns this thread's COM mode; logging it tells a crash in
		// CreateCoreWebView2Environment apart from one in the WebView itself.
		#[cfg(windows)]
		{
			use windows_sys::Win32::System::Com::{COINIT_APARTMENTTHREADED, CoInitializeEx};
			// SAFETY: a null reserved pointer is what CoInitializeEx expects; the
			// host's existing apartment, if any, is left in place.
			let result =
				unsafe { CoInitializeEx(std::ptr::null_mut(), COINIT_APARTMENTTHREADED as u32) };
			note(&format!("com: {result:#x}"));

			if result == RPC_E_CHANGED_MODE {
				let error = EditorError::from(
					"the GUI thread is already in a multi-threaded COM apartment; WebView2 needs STA",
				);
				report_error(&error);
				return Err(error);
			} else if result < 0 {
				let error = EditorError::from(format!("CoInitializeEx failed: {result:#x}"));
				report_error(&error);
				return Err(error);
			}
		}

		let parent_hwnd = parent
			.as_win32_hwnd()
			.ok_or("the wry editor needs a Win32 parent window")?;

		if unsafe { IsWindow(parent_hwnd) } == 0 {
			let error = EditorError::from("the host parent window is gone");
			report_error(&error);
			return Err(error);
		}

		// The host's window is the truth for the visible area: a DAW may size the
		// plug-in window differently from the design size the wrapper reports.
		// The child and the WebView2 fill that client rectangle.
		let mut client = unsafe { std::mem::zeroed() };
		let parent_size = if unsafe { GetClientRect(parent_hwnd, &mut client) } != 0
			&& client.right > client.left
			&& client.bottom > client.top
		{
			Some((
				(client.right - client.left) as f64,
				(client.bottom - client.top) as f64,
			))
		} else {
			None
		};
		let size = match parent_size {
			Some((width, height)) => {
				WindowSize::new((width / self.scale) as f32, (height / self.scale) as f32)
			}
			None => size,
		};
		self.size = size;
		let scale = self.scale;

		let dpi = unsafe { GetDpiForWindow(parent_hwnd) };
		note(&format!(
			"parent: hwnd={parent_hwnd:?} client={:?} dpi={dpi} scale={scale}",
			parent_size
		));

		let window = create_host_window(parent_hwnd, size, scale)?;
		note(&format!(
			"window: created {}x{} logical",
			size.width, size.height
		));
		let outer_parent =
			ParentWindow::from_win32_hwnd(window).ok_or("the host window handle was invalid")?;

		let protocols = std::mem::take(&mut self.protocols);
		let url = self.url.take();

		// The WebView2 runs inside our own child window; the host's editor window
		// only has to host that child. Creating the controller directly in the
		// host's window is what crashed some DAWs at build time.
		let opened = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
			move || -> Result<wry::WebView, EditorError> {
				WEB_CONTEXT.with(move |shared| {
					let mut shared = shared.try_borrow_mut().map_err(|_| {
						EditorError::from("WebView2 context is already in use; re-entrant GUI open")
					})?;

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
						// The same arguments are part of the profile identity: a build
						// with different arguments must use another profile version.
						.with_additional_browser_args(BROWSER_ARGS)
						// A hidden editor should stop painting and running timers.
						.with_background_throttling(wry::BackgroundThrottlingPolicy::Suspend)
						// The page is the whole of the control, and the host owns the
						// sizing: the title is only a suggestion, and WebView2 has no
						// chrome of its own.
						.with_bounds(Self::bounds(size, scale))
						.with_ipc_handler(move |request: wry::http::Request<String>| {
							if let Some(message) = on_ipc(request.body()) {
								// The receiver is drained on the host's main thread by the
								// wrapper, just like a message an iced view produces.
								let _ = sender.send(message);
							}
						})
						.build_as_child(&outer_parent)
						.map_err(EditorError::new)?;

					Ok(webview)
				})
			},
		));

		let webview = match opened {
			Ok(Ok(webview)) => webview,
			Ok(Err(error)) => {
				unsafe {
					DestroyWindow(window);
				}
				report_error(&error);
				return Err(error);
			}
			Err(payload) => {
				unsafe {
					DestroyWindow(window);
				}
				let message = panic_payload(payload.as_ref());
				note(&format!("panic in open: {message}"));
				let error =
					EditorError::from(format!("WebView2 panicked while opening: {message}"));
				report_error(&error);
				return Err(error);
			}
		};

		note("build: created");

		if unsafe { IsWindow(parent_hwnd) } == 0 {
			let error = EditorError::from("the host parent window disappeared while opening");
			note("open: parent gone after build");
			drop(webview);
			unsafe {
				DestroyWindow(window);
			}
			report_error(&error);
			return Err(error);
		}

		if let Err(error) = webview.set_visible(true) {
			let error = EditorError::new(error);
			report_error(&error);
			drop(webview);
			unsafe {
				DestroyWindow(window);
			}
			return Err(error);
		}

		clear_error();
		note("open: ok");

		// Keep the title around only to make the intent explicit: WebView2 is a
		// child control, so its chrome (and title) belongs to the host's window.
		let _ = title;

		self.webview = Some(webview);
		self.window = Some(window);

		// The timer is ours, so no host support is needed for it: whichever
		// message loop the host runs delivers WM_TIMER to this window.
		EDITOR_HOOK.with(|slot| {
			*slot.borrow_mut() = Some(EditorHook {
				editor: self as *mut WryEditor<P> as *mut (),
				follow: follow_editor::<P>,
			});
		});
		unsafe {
			SetWindowLongPtrW(window, GWLP_WNDPROC, follow_timer as *const () as isize);
			SetTimer(window, FOLLOW_TIMER, 33, None);
		}
		Ok(())
	}

	fn show(&mut self) -> Result<(), EditorError> {
		if let Some(window) = self.window {
			unsafe {
				ShowWindow(window, SW_SHOW);
			}
		}
		note("show");
		let webview = self.webview()?;
		webview.set_visible(true).map_err(EditorError::new)?;
		// FL Studio focuses the wrapper window, not the WebView2 child. Without
		// this the page does not see the keyboard until the user clicks it.
		let _ = webview.focus();
		Ok(())
	}

	fn hide(&mut self) -> Result<(), EditorError> {
		if let Some(window) = self.window {
			unsafe {
				ShowWindow(window, SW_HIDE);
			}
		}
		note("hide");
		self.webview()?.set_visible(false).map_err(EditorError::new)
	}

	fn close(&mut self) {
		self.shutdown();
	}

	fn set_parent(&mut self, parent: ParentWindow) -> Result<(), EditorError> {
		let Some(parent_hwnd) = parent.as_win32_hwnd() else {
			return Ok(());
		};
		let Some(window) = self.window else {
			return Ok(());
		};

		note(&format!(
			"set_parent: from={:?} to={parent_hwnd:?}",
			self.window
		));
		unsafe {
			SetLastError(0);
		}
		let previous = unsafe { SetParent(window, parent_hwnd) };
		if previous.is_null() {
			let last_error = unsafe { GetLastError() };
			if last_error != 0 {
				let error = EditorError::from(format!("SetParent failed: {last_error}"));
				report_error(&error);
				return Err(error);
			}
		}

		// Moving the host window does not resize it. FL Studio can detach and
		// reattach a plug-in, and the new parent may have a different client area.
		let mut client = unsafe { std::mem::zeroed() };
		if unsafe { GetClientRect(parent_hwnd, &mut client) } != 0
			&& client.right > client.left
			&& client.bottom > client.top
		{
			let size = WindowSize::new(
				((client.right - client.left) as f64 / self.scale) as f32,
				((client.bottom - client.top) as f64 / self.scale) as f32,
			);
			self.size = size;
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
			if self.webview.is_some() {
				self.webview()?
					.set_bounds(Self::bounds(size, self.scale))
					.map_err(EditorError::new)?;
			}
			note(&format!(
				"set_parent: resized to {}x{}",
				size.width, size.height
			));
		}

		Ok(())
	}

	fn set_scale_factor(&mut self, scale_factor: f64) -> Result<(), EditorError> {
		self.scale = if scale_factor.is_finite() {
			scale_factor.clamp(0.25, 4.0)
		} else {
			1.0
		};
		note(&format!("set_scale: {}", self.scale));

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
				.set_bounds(Self::bounds(self.size, self.scale))
				.map_err(EditorError::new)?;
		}

		Ok(())
	}

	fn set_size(&mut self, size: WindowSize) -> Result<(), EditorError> {
		note(&format!("set_size: {}x{}", size.width, size.height));
		self.size = size;

		if let Some(window) = self.window {
			let width = (size.width as f64 * self.scale).round() as i32;
			let height = (size.height as f64 * self.scale).round() as i32;
			let placed = unsafe {
				SetWindowPos(
					window,
					std::ptr::null_mut(),
					0,
					0,
					width,
					height,
					SWP_NOZORDER | SWP_NOACTIVATE,
				)
			};

			note(&format!(
				"set_size: window -> {}x{} ok={placed} parent={:?}",
				width,
				height,
				unsafe { GetParent(window) }
			));
		}

		if self.webview.is_some() {
			match self.webview()?.set_bounds(Self::bounds(size, self.scale)) {
				Ok(()) => note("set_size: bounds applied"),
				Err(error) => {
					note(&format!("set_size: bounds failed: {error}"));
					return Err(EditorError::new(error));
				}
			}
		}

		Ok(())
	}
}

impl<P: Plugin> Drop for WryEditor<P> {
	fn drop(&mut self) {
		note("drop: editor");
		self.shutdown();
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn stale_pid_profiles_are_removed_and_live_ones_kept() {
		let directory = std::env::current_dir()
			.unwrap()
			.join("target")
			.join(format!(
				"i_am_plugin_wry_test_{}_{}",
				std::process::id(),
				std::time::SystemTime::now()
					.duration_since(std::time::UNIX_EPOCH)
					.unwrap()
					.as_nanos()
			));
		std::fs::create_dir_all(directory.join("WebView2-4294967295")).unwrap();
		std::fs::create_dir_all(directory.join(format!("WebView2-{}", std::process::id())))
			.unwrap();
		std::fs::create_dir_all(directory.join("WebView2-v2")).unwrap();

		clean_stale_profiles(&directory);

		assert!(!directory.join("WebView2-4294967295").exists());
		assert!(
			directory
				.join(format!("WebView2-{}", std::process::id()))
				.exists()
		);
		assert!(directory.join("WebView2-v2").exists());

		let _ = std::fs::remove_dir_all(&directory);
	}
}
