//! The main thread: the processor, the host's window, and the GUI that runs in it.
//!
//! Nothing here names a GUI toolkit. The processor builds an [Editor] when the host
//! opens the GUI, the host's window is passed to it as a raw handle, and the messages
//! it makes come back over a channel. Which toolkit draws the window is the plug-in's
//! choice.

use crate::editor::{Editor, EditorContext, ParentWindow};
use crate::latency::LatencyWatch;
use crate::plugin::{Plugin, WindowSize};
use crate::shared::*;
use clack_extensions::timer::{HostTimer, PluginTimerImpl, TimerId};

/// The main-thread half of a plug-in.
///
/// Public mainly for `export_plugin` macro. At most case, you don't need to use it directly.
pub struct PluginMain<P: Plugin> {
	/// The logical size the GUI is laid out for.
	pub(crate) window_size: WindowSize,
	/// The host's scale factor, and whether it ever reported one.
	pub(crate) scale_factor: f64,
	pub(crate) scale_set: bool,
	/// The title of the plugin GUI.
	pub(crate) title: String,
	/// The resize hints of the plugin GUI.
	///
	/// None for can't resize.
	pub(crate) resize_hints: Option<GuiResizeHints>,

	pub(crate) param_map: ParamMap,
	pub(crate) processor: Pin<Box<P>>,
	/// The handover the state extension uses to reach the processor while the
	/// audio thread is rendering.
	pub(crate) state_guard: std::sync::Arc<crate::state_guard::StateGuard>,
	/// The host, kept so the plugin can tell it when the latency moves.
	///
	/// The handle is valid for as long as the plugin lives: `new_main_thread` receives it from
	/// the host and the host owns the plugin, so its lifetime outlasts this struct.
	pub(crate) host: Option<HostMainThreadHandle<'static>>,
	/// What the audio thread is running at, and what the host was last told.
	pub(crate) latency: LatencyWatch,
	/// The window the host handed us, kept until the GUI is opened.
	pub(crate) parent: Option<ParentWindow>,
	/// The open GUI, if it is open.
	pub(crate) editor: Option<Box<dyn Editor<P>>>,
	/// Where the editors send the messages the user makes: one channel, so the audio
	/// thread applies them all and a product never sees two threads at once.
	pub(crate) message_sender: crossbeam_channel::Sender<P::Message>,
	pub(crate) message_receiver: std::sync::Arc<crossbeam_channel::Receiver<P::Message>>,
	/// True while `open` is inside the editor's own open call.
	///
	/// WebView2 pumps the host's message loop while it creates its controller, so
	/// a host can re-enter `show`, `hide`, `set_parent` or `destroy` before the
	/// editor has been stored. These flags turn those re-entrant calls into a
	/// small pending-state queue instead of a second editor.
	pub(crate) opening: bool,
	pub(crate) close_requested: bool,
	/// The host timer the GUI uses to follow the window's size, when the host
	/// offers one.
	pub(crate) editor_timer: Option<TimerId>,
	pub(crate) parent_dirty: bool,
	pub(crate) geometry_dirty: bool,
	pub(crate) hide_requested: bool,
}

impl<P: Plugin> PluginMain<P> {
	fn scale_value(&self) -> f64 {
		if self.scale_factor.is_finite() && self.scale_factor > 0.0 {
			self.scale_factor.clamp(0.25, 4.0)
		} else {
			1.0
		}
	}

	fn reset_pending(&mut self) {
		self.close_requested = false;
		self.parent_dirty = false;
		self.geometry_dirty = false;
		self.hide_requested = false;
	}

	/// Open the plugin GUI in the window the host provided.
	pub fn open(&mut self) -> Result<(), PluginError> {
		if self.opening {
			crate::log::note("gui.open: ignored, already opening");
			return Ok(());
		}
		if self.editor.is_some() {
			return Ok(());
		}

		let Some(parent) = self.parent else {
			return Err(PluginError::Message("No parent window provided"));
		};

		// Building the editor reads the product's parameters; the audio thread holds
		// the same processor, so the read happens in a handover window. The window is
		// only this call: the window itself is created below, after the handover ends.
		let editor = {
			let _handover = self.state_guard.take();
			self.processor.editor()
		};
		let Some(mut editor) = editor else {
			return Err(PluginError::Message("This plug-in has no GUI"));
		};

		let context = EditorContext {
			parent,
			size: self.window_size,
			// A host that never reported a scale leaves the window at the platform's own
			// DPI, which is the same choice the processor's view makes.
			scale_factor: self.scale_set.then(|| self.scale_value()),
			title: self.title.clone(),
			resizable: self.resize_hints.is_some(),
			messages: self.message_sender.clone(),
		};

		self.opening = true;
		self.reset_pending();
		crate::log::note(&format!(
			"gui.open: start parent={parent:?} size={}x{} scale={:?}",
			self.window_size.width, self.window_size.height, context.scale_factor
		));

		let opened = editor.open(context);
		self.opening = false;

		if let Err(error) = opened {
			self.reset_pending();
			crate::log::note(&format!("gui.open: failed: {error}"));
			return Err(PluginError::Error(Box::new(error)));
		}

		crate::log::note("gui.open: editor opened");
		self.finish_open(editor)
	}

	fn finish_open(
		&mut self,
		mut editor: Box<dyn Editor<P>>,
	) -> Result<(), PluginError> {
		if self.close_requested {
			self.reset_pending();
			crate::log::note("gui.open: host closed while opening");
			editor.close();
			return Ok(());
		}

		if self.parent_dirty {
			if let Some(parent) = self.parent {
				crate::log::note("gui.open: applying a parent change");
				if let Err(error) = editor.set_parent(parent) {
					return Err(self.fail_open(editor, error));
				}
			}
			self.parent_dirty = false;
		}

		if self.geometry_dirty {
			crate::log::note("gui.open: applying a size or scale change");
			if self.scale_set
				&& let Err(error) = editor.set_scale_factor(self.scale_factor)
			{
				return Err(self.fail_open(editor, error));
			}
			if let Err(error) = editor.set_size(self.window_size) {
				return Err(self.fail_open(editor, error));
			}
			self.geometry_dirty = false;
		}

		if self.hide_requested {
			let _ = editor.hide();
			self.hide_requested = false;
		}

		self.editor = Some(editor);
		self.start_editor_timer();
		crate::log::note("gui.open: done");
		Ok(())
	}

	fn fail_open(
		&mut self,
		mut editor: Box<dyn Editor<P>>,
		error: crate::EditorError,
	) -> PluginError {
		self.reset_pending();
		crate::log::note(&format!("gui.open: setup failed: {error}"));
		editor.close();
		PluginError::Error(Box::new(error))
	}

	/// Destroy the plugin GUI and its state.
	///
	/// Hiding only leaves the window off screen; this is what the host calls
	/// destroy for, and it is the only place the GUI is thrown away.
	pub fn close(&mut self) {
		self.stop_editor_timer();

		if self.opening {
			self.close_requested = true;
			crate::log::note("gui.close: deferred until open returns");
			return;
		}

		if let Some(mut editor) = self.editor.take() {
			crate::log::note("gui.close: closing editor");
			editor.close();
			crate::log::note("gui.close: editor closed");
		}
		// The channel outlives the editor: a message already in flight is still
		// applied by the audio thread on its next block.
	}

	/// Hide the plugin GUI, keeping it ready to be shown again.
	pub fn hide(&mut self) -> Result<(), PluginError> {
		if self.opening {
			self.hide_requested = true;
			crate::log::note("gui.hide: deferred until open returns");
			return Ok(());
		}
		if let Some(editor) = &mut self.editor {
			crate::log::note("gui.hide");
			editor
				.hide()
				.map_err(|error| PluginError::Error(Box::new(error)))?;
		}

		Ok(())
	}

	/// Set the parent window handle.
	///
	/// The handle belongs to the host and is copied out of it, so it may outlive the
	/// window it was taken from exactly as long as the host keeps that window open.
	pub fn set_parent(&mut self, parent: GuiWindow<'_>) -> Result<(), PluginError> {
		// SAFETY: the handle is borrowed only long enough to copy the platform window out
		// of it; ParentWindow owns what it needs from then on.
		let handle = unsafe { parent.borrow_handle_unchecked() }
			.map_err(|_| PluginError::Message("Unsupported window API"))?;
		let parent = ParentWindow::new(handle.as_raw());
		let raw = parent.as_win32_hwnd().map(|hwnd| hwnd as usize);

		self.parent = Some(parent);
		crate::log::note(&format!(
			"gui.set_parent: hwnd={raw:?} opening={}",
			self.opening
		));

		if self.opening {
			self.parent_dirty = true;
			return Ok(());
		}

		if let Some(editor) = &mut self.editor {
			editor
				.set_parent(parent)
				.map_err(|error| PluginError::Error(Box::new(error)))?;
		}

		Ok(())
	}
}

impl<P: Plugin> PluginGuiImpl for PluginMain<P> {
	fn is_api_supported(&mut self, configuration: GuiConfiguration) -> bool {
		configuration.api_type
			== GuiApiType::default_for_current_platform().expect("Unsupported platform")
			&& !configuration.is_floating
	}

	fn get_preferred_api(&mut self) -> Option<GuiConfiguration<'_>> {
		Some(GuiConfiguration {
			api_type: GuiApiType::default_for_current_platform().expect("Unsupported platform"),
			is_floating: false,
		})
	}

	fn create(&mut self, _: GuiConfiguration) -> Result<(), clack_plugin::prelude::PluginError> {
		Ok(())
	}

	fn destroy(&mut self) {
		self.close();
	}

	fn set_scale(&mut self, scale: f64) -> Result<(), clack_plugin::prelude::PluginError> {
		let scale = if scale.is_finite() {
			scale.clamp(0.25, 4.0)
		} else {
			1.0
		};
		self.scale_factor = scale;
		self.scale_set = true;
		crate::log::note(&format!(
			"gui.set_scale: {scale} (opening={})",
			self.opening
		));

		// A backend decides how its own window has to change: on one platform the
		// window keeps its pixel size and the view zooms, on another the window grows
		// and the view stays put.
		if self.opening {
			self.geometry_dirty = true;
			return Ok(());
		}
		if let Some(editor) = &mut self.editor {
			editor
				.set_scale_factor(scale)
				.map_err(|error| PluginError::Error(Box::new(error)))?;
		}

		Ok(())
	}

	fn can_resize(&mut self) -> bool {
		self.resize_hints.is_some()
	}

	fn get_resize_hints(&mut self) -> Option<GuiResizeHints> {
		self.resize_hints
	}

	fn get_size(&mut self) -> Option<clack_extensions::gui::GuiSize> {
		let scale = self.scale_value();
		Some(clack_extensions::gui::GuiSize {
			width: (self.window_size.width as f64 * scale)
				.round()
				.clamp(1.0, 32768.0) as u32,
			height: (self.window_size.height as f64 * scale)
				.round()
				.clamp(1.0, 32768.0) as u32,
		})
	}

	fn set_size(
		&mut self,
		size: clack_extensions::gui::GuiSize,
	) -> Result<(), clack_plugin::prelude::PluginError> {
		crate::log::note(&format!("gui.set_size: {}x{}", size.width, size.height));
		let scale = self.scale_value();
		let logical = WindowSize::new(
			(size.width as f64 / scale) as f32,
			(size.height as f64 / scale) as f32,
		);
		self.window_size = logical;

		if self.opening {
			self.geometry_dirty = true;
			return Ok(());
		}
		if let Some(editor) = &mut self.editor {
			editor
				.set_size(logical)
				.map_err(|error| PluginError::Error(Box::new(error)))?;
		}

		Ok(())
	}

	fn set_parent(
		&mut self,
		window: clack_extensions::gui::Window,
	) -> Result<(), clack_plugin::prelude::PluginError> {
		PluginMain::set_parent(self, window)?;
		Ok(())
	}

	fn set_transient(
		&mut self,
		_window: clack_extensions::gui::Window,
	) -> Result<(), clack_plugin::prelude::PluginError> {
		// CLAP only defines a transient window for floating GUIs, and this backend
		// is embedded-only. Treating it as set_parent used to move the editor onto
		// FL Studio's main window when a wrapper passed it through anyway.
		crate::log::note("gui.set_transient: ignored (embedded only)");
		Ok(())
	}

	fn show(&mut self) -> Result<(), clack_plugin::prelude::PluginError> {
		if self.opening {
			crate::log::note("gui.show: ignored while opening");
			return Ok(());
		}
		if let Some(editor) = &mut self.editor {
			crate::log::note("gui.show");
			return editor
				.show()
				.map_err(|error| PluginError::Error(Box::new(error)));
		}

		self.open()
	}

	fn hide(&mut self) -> Result<(), clack_plugin::prelude::PluginError> {
		PluginMain::hide(self)
	}

	fn adjust_size(
		&mut self,
		size: clack_extensions::gui::GuiSize,
	) -> Option<clack_extensions::gui::GuiSize> {
		let scale = self.scale_value();
		self.window_size = WindowSize::new(
			(size.width as f64 / scale) as f32,
			(size.height as f64 / scale) as f32,
		);
		if self.opening {
			self.geometry_dirty = true;
		}
		Some(size)
	}

	fn suggest_title(&mut self, title: &str) {
		self.title = title.to_string();
	}
}

impl<P: Plugin> PluginMain<P> {
	/// Ask the host for a timer the GUI can use to notice its window changing
	/// size. Hosts that do not offer the timer extension simply never call it.
	fn start_editor_timer(&mut self) {
		if self.editor_timer.is_some() {
			return;
		}

		let Some(host) = self.host.as_mut() else {
			return;
		};
		let Some(timer) = host.get_extension::<HostTimer>() else {
			return;
		};

		match timer.register_timer(host, 33) {
			Ok(id) => {
				crate::log::note(&format!("editor timer: {id}"));
				self.editor_timer = Some(id);
			}
			Err(error) => crate::log::note(&format!("editor timer: refused: {error}")),
		}
	}

	fn stop_editor_timer(&mut self) {
		let Some(id) = self.editor_timer.take() else {
			return;
		};
		let Some(host) = self.host.as_mut() else {
			return;
		};
		if let Some(timer) = host.get_extension::<HostTimer>() {
			let _ = timer.unregister_timer(host, id);
		}
	}
}

impl<P: Plugin> PluginTimerImpl for PluginMain<P> {
	fn on_timer(&mut self, timer_id: TimerId) {
		if Some(timer_id) != self.editor_timer {
			return;
		}
		if self.opening {
			return;
		}

		let Some(parent) = self.parent else {
			return;
		};
		// The host owns the window: it may resize it without calling set_size
		// (some hosts only do that on release, others never). Following the
		// client rectangle is what makes the page react while the edge is
		// still being dragged.
		let Some((width, height)) = crate::editor::client_size(parent) else {
			return;
		};

		let scale = self.scale_value();
		let logical = WindowSize::new((width / scale) as f32, (height / scale) as f32);

		if (logical.width - self.window_size.width).abs() < 1.0
			&& (logical.height - self.window_size.height).abs() < 1.0
		{
			return;
		}

		crate::log::note(&format!(
			"editor timer: following the window to {}x{}",
			logical.width, logical.height
		));

		self.window_size = logical;

		if let Some(editor) = self.editor.as_mut() {
			let _ = editor.set_size(logical);
		}
	}
}

impl<P: Plugin> Drop for PluginMain<P> {
	fn drop(&mut self) {
		self.close();
	}
}
