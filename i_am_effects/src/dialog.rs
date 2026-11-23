//! The one place that puts a native file dialog on screen.
//!
//! A page's own file input has no owner inside a host, so its picker can open behind
//! the DAW or appear not to open at all; a dialog whose owner is the foreground window
//! always comes up in front of it. The dialog also runs on a worker thread, because a
//! modal window opened from inside the click WebView2 is still dispatching only showed
//! up some of the time.
//!
//! Only Windows has this path. The other backends keep the page's own file input for
//! opening and the browser's download for saving, so CAN_SAVE is what a page asks
//! before it offers a dialog that would do nothing.

use std::path::PathBuf;

/// Whether this build can put a native dialog on screen.
pub const CAN_SAVE: bool = cfg!(windows);

/// The name the dialog shows and the extensions it offers.
pub type Filter = (&'static str, &'static [&'static str]);

/// Opens the picker for the filter and hands the chosen path to the callback.
#[cfg(windows)]
pub fn pick(filter: Filter, done: impl FnOnce(Option<PathBuf>) + Send + 'static) {
	let owner = owner();

	std::thread::spawn(move || {
		// Let the click that asked for the picker finish before a modal dialog takes
		// over: it is dispatched by WebView2 inside the same input event.
		std::thread::sleep(std::time::Duration::from_millis(50));

		let mut dialog = rfd::FileDialog::new().add_filter(filter.0, filter.1);

		if let Some(owner) = owner {
			dialog = dialog.set_parent(&owner);
		}

		done(dialog.pick_file());
	});
}

/// Opens the native save-as dialog and hands the chosen path to the callback.
#[cfg(windows)]
pub fn save(suggested: &str, filter: Filter, done: impl FnOnce(Option<PathBuf>) + Send + 'static) {
	let owner = owner();
	let suggested = suggested.to_string();

	std::thread::spawn(move || {
		std::thread::sleep(std::time::Duration::from_millis(50));

		let mut dialog = rfd::FileDialog::new()
			.add_filter(filter.0, filter.1)
			.set_file_name(&suggested);

		if let Some(owner) = owner {
			dialog = dialog.set_parent(&owner);
		}

		done(dialog.save_file());
	});
}

/// No native dialog here: the page keeps its own input, and a save falls back to the
/// browser's download.
#[cfg(not(windows))]
pub fn pick(_filter: Filter, done: impl FnOnce(Option<PathBuf>) + Send + 'static) {
	done(None);
}

/// No native dialog here: a save falls back to the browser's download.
#[cfg(not(windows))]
pub fn save(
	_suggested: &str,
	_filter: Filter,
	done: impl FnOnce(Option<PathBuf>) + Send + 'static,
) {
	done(None);
}

/// A window handle for the window that should own the dialog.
#[cfg(windows)]
struct Owner(isize);

#[cfg(windows)]
impl raw_window_handle::HasWindowHandle for Owner {
	fn window_handle(
		&self,
	) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
		let hwnd = std::num::NonZeroIsize::new(self.0)
			.ok_or(raw_window_handle::HandleError::Unavailable)?;
		let handle = raw_window_handle::RawWindowHandle::Win32(
			raw_window_handle::Win32WindowHandle::new(hwnd),
		);

		// SAFETY: the window outlives the dialog, and the handle is only borrowed for
		// the duration of the call.
		Ok(unsafe { raw_window_handle::WindowHandle::borrow_raw(handle) })
	}
}

#[cfg(windows)]
impl raw_window_handle::HasDisplayHandle for Owner {
	fn display_handle(
		&self,
	) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
		// SAFETY: the Windows display handle carries no state.
		Ok(unsafe {
			raw_window_handle::DisplayHandle::borrow_raw(
				raw_window_handle::RawDisplayHandle::Windows(
					raw_window_handle::WindowsDisplayHandle::new(),
				),
			)
		})
	}
}

/// The root of the foreground window, when it is a window of this process.
///
/// In a host that is the DAW's window, and the dialog then behaves as modal inside it.
/// A page served to a browser has the browser in front instead: a dialog owned by
/// another process does not come up at all (rfd returns as if it had been cancelled),
/// so the owner is dropped there and the dialog stands on its own.
#[cfg(windows)]
fn owner() -> Option<Owner> {
	use windows_sys::Win32::UI::WindowsAndMessaging::{
		GA_ROOT, GetAncestor, GetForegroundWindow, GetWindowThreadProcessId,
	};

	let mut window = unsafe { GetForegroundWindow() };

	if window.is_null() {
		return None;
	}

	let root = unsafe { GetAncestor(window, GA_ROOT) };

	if !root.is_null() {
		window = root;
	}

	let mut process = 0_u32;
	unsafe { GetWindowThreadProcessId(window, &mut process) };

	if process != std::process::id() {
		return None;
	}

	Some(Owner(window as isize))
}
