//! What the crate is on a platform this backend has not been tried on.

use std::marker::PhantomData;

use i_am_plugin::{Editor, EditorContext, EditorError, Plugin};

use crate::ProtocolResponse;

/// A placeholder that reports the limitation from `open` instead of failing to build.
pub struct WryEditor<P: Plugin> {
	_plugin: PhantomData<fn() -> P>,
}

impl<P: Plugin> WryEditor<P> {
	/// Build an editor. It can be constructed everywhere; only opening it is refused.
	pub fn new(
		_html: impl Into<String>,
		_on_ipc: impl Fn(&str) -> Option<P::Message> + 'static,
	) -> Self {
		Self {
			_plugin: PhantomData,
		}
	}

	/// Ignored on this platform; see the Windows implementation.
	pub fn with_data_directory(self, _path: impl Into<std::path::PathBuf>) -> Self {
		self
	}

	/// Ignored on this platform; see the Windows implementation.
	pub fn with_url(self, _url: impl Into<String>) -> Self {
		self
	}

	/// Ignored on this platform; see the Windows implementation.
	pub fn with_custom_protocol(
		self,
		_name: impl Into<String>,
		_handler: impl Fn(&str) -> ProtocolResponse + Send + Sync + 'static,
	) -> Self {
		self
	}
}

impl<P: Plugin> Editor<P> for WryEditor<P> {
	fn open(&mut self, _context: EditorContext<P>) -> Result<(), EditorError> {
		Err("the wry backend is only implemented on Windows".into())
	}

	fn show(&mut self) -> Result<(), EditorError> {
		Err("the wry backend is only implemented on Windows".into())
	}

	fn hide(&mut self) -> Result<(), EditorError> {
		Err("the wry backend is only implemented on Windows".into())
	}

	fn close(&mut self) {}
}
