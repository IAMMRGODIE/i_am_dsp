//! The GUI side of a plug-in, kept free of any particular toolkit.
//!
//! A host hands a plug-in's GUI two things: a window to live in, and a stream of
//! messages it makes. Neither is a toolkit type. [ParentWindow] is a raw handle from
//! `raw-window-handle`, which every windowing library in the ecosystem understands,
//! and messages travel over a channel of the plug-in's own message type. An editor
//! is what joins those to a GUI: this module defines the seam, and a crate such as
//! `i_am_plugin_iced` provides the implementation.

use std::ffi::c_void;

use crossbeam_channel::Sender;
use raw_window_handle::{HandleError, HasWindowHandle, RawWindowHandle, WindowHandle};

use crate::plugin::{Plugin, WindowSize};

/// The host's parent window, copied out of the handle it lent us.
///
/// The host owns the window and keeps it open for as long as the editor may use it,
/// which is what makes copying the raw handle sound. It is stored rather than borrowed
/// because the host may hand it to us before the GUI exists, and the GUI is opened
/// later.
#[derive(Debug, Clone, Copy)]
pub struct ParentWindow {
	raw: RawWindowHandle,
}

impl ParentWindow {
	/// Wrap a raw handle taken from the host.
	///
	/// # Safety
	///
	/// The handle has to stay valid while the editor that receives it is open.
	pub const fn new(raw: RawWindowHandle) -> Self {
		Self { raw }
	}

	/// The raw handle, for a backend that wants to convert it itself.
	pub const fn as_raw(&self) -> RawWindowHandle {
		self.raw
	}

	/// The Win32 `HWND` this window stands for, if it is a Win32 window.
	pub fn as_win32_hwnd(&self) -> Option<*mut c_void> {
		match self.raw {
			RawWindowHandle::Win32(handle) => Some(handle.hwnd.get() as *mut c_void),
			_ => None,
		}
	}

	/// A parent window from a Win32 `HWND`.
	///
	/// Backends that create a window of their own (a standalone runner, a test that
	/// needs a host window) can hand it over this way instead of going through the
	/// raw-window-handle types themselves.
	pub fn from_win32_hwnd(hwnd: *mut c_void) -> Option<Self> {
		let hwnd = std::num::NonZeroIsize::new(hwnd as isize)?;

		Some(Self::new(RawWindowHandle::Win32(
			raw_window_handle::Win32WindowHandle::new(hwnd),
		)))
	}
}

impl HasWindowHandle for ParentWindow {
	fn window_handle(&self) -> Result<WindowHandle<'_>, HandleError> {
		// SAFETY: the handle was copied from the host while the host's window was
		// valid, and the host keeps that window open for as long as the editor may
		// use it.
		Ok(unsafe { WindowHandle::borrow_raw(self.raw) })
	}
}

/// What a GUI backend reports when it cannot open, show, resize or hide its window.
///
/// The concrete error is boxed because a backend's own error type is its business;
/// the plug-in wrapper only needs to display it and pass it on.
#[derive(Debug)]
pub struct EditorError(Box<dyn std::error::Error + 'static>);

impl EditorError {
	/// Wrap any error a backend ran into.
	///
	/// The error does not have to be `Send`: a GUI lives on the main thread, and so do
	/// the errors it reports.
	pub fn new(error: impl std::error::Error + 'static) -> Self {
		Self(Box::new(error))
	}
}

impl std::fmt::Display for EditorError {
	fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		std::fmt::Display::fmt(&self.0, formatter)
	}
}

impl std::error::Error for EditorError {
	fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
		Some(self.0.as_ref())
	}
}

impl From<&str> for EditorError {
	fn from(message: &str) -> Self {
		Self(message.into())
	}
}

impl From<String> for EditorError {
	fn from(message: String) -> Self {
		Self(message.into())
	}
}

/// Everything a GUI needs to open itself.
pub struct EditorContext<P: Plugin> {
	/// The window the host wants the GUI to live in.
	pub parent: ParentWindow,
	/// The logical size the plug-in's [window options](crate::WindowOptions) ask for.
	pub size: WindowSize,
	/// The host's scale factor, if it reported one.
	///
	/// A host that never called `set_scale` leaves the window to the platform's own
	/// DPI handling, which is what `None` means.
	pub scale_factor: Option<f64>,
	/// The title the host suggests for the window.
	pub title: String,
	/// Whether the host allows the window to be resized.
	pub resizable: bool,
	/// Where the editor sends the messages the user makes.
	///
	/// The other end is drained on the host's main thread and handed to
	/// [Processor::on_message](i_am_dsp::Processor::on_message).
	pub messages: Sender<P::Message>,
}

/// A GUI a plug-in can open in the host's window.
///
/// Implement this to connect a toolkit (iced is in `i_am_plugin_iced`; egui, slint or
/// a bespoke renderer would be a crate of their own) and return it from
/// [Plugin::editor](crate::Plugin::editor).
///
/// The lifecycle is the host's, not the editor's: `open` creates the window, the
/// host calls `show`/`hide` as the person opens and closes it, and `close` destroys
/// it for good. The defaults for reparenting, scale and size let a backend ignore the
/// host callbacks it has nothing to do for.
pub trait Editor<P: Plugin>: 'static {
	/// Create the GUI window, as a child of `context.parent`.
	///
	/// The window is expected to be shown by the time this returns; `show` afterwards
	/// is the host re-opening a hidden one.
	fn open(&mut self, context: EditorContext<P>) -> Result<(), EditorError>;

	/// Show the window.
	fn show(&mut self) -> Result<(), EditorError>;

	/// Hide the window, keeping it ready to be shown again.
	fn hide(&mut self) -> Result<(), EditorError>;

	/// Destroy the window and everything it holds.
	fn close(&mut self);

	/// Move the window under a different parent.
	fn set_parent(&mut self, parent: ParentWindow) -> Result<(), EditorError> {
		let _ = parent;
		Ok(())
	}

	/// The host's scale factor changed.
	fn set_scale_factor(&mut self, scale_factor: f64) -> Result<(), EditorError> {
		let _ = scale_factor;
		Ok(())
	}

	/// The host asks for a different logical size.
	fn set_size(&mut self, size: WindowSize) -> Result<(), EditorError> {
		let _ = size;
		Ok(())
	}
}
