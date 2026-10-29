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

/// The main-thread half of a plug-in.
/// 
/// Public mainly for `export_clap` macro. At most case, you don't need to use it directly.
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
	pub(crate) message_receivers: Vec<Receiver<P::Message>>,
}

impl<P: Plugin> PluginMain<P> {
	/// Open the plugin GUI in the window the host provided.
	pub fn open(&mut self) -> Result<(), PluginError> {
		let Some(parent) = self.parent else {
			return Err(PluginError::Message("No parent window provided"));
		};

		let Some(mut editor) = self.processor.editor() else {
			return Err(PluginError::Message("This plug-in has no GUI"));
		};

		let (message_sender, message_receiver) = crossbeam_channel::unbounded();
		let context = EditorContext {
			parent,
			size: self.window_size,
			// A host that never reported a scale leaves the window at the platform's own
			// DPI, which is the same choice the processor's view makes.
			scale_factor: self.scale_set.then_some(self.scale_factor),
			title: self.title.clone(),
			resizable: self.resize_hints.is_some(),
			messages: message_sender,
		};

		editor.open(context).map_err(|error| PluginError::Error(Box::new(error)))?;

		// Only a GUI that is actually running has messages worth collecting; a failed
		// open leaves no receiver behind for it.
		self.message_receivers.push(message_receiver);
		self.editor = Some(editor);
		Ok(())
	}

	/// Destroy the plugin GUI and its state.
	///
	/// Hiding only leaves the window off screen; this is what the host calls
	/// `destroy` for, and it is the only place the GUI is thrown away.
	pub fn close(&mut self) {
		if let Some(mut editor) = self.editor.take() {
			editor.close();
		}
	}

	/// Hide the plugin GUI, keeping it ready to be shown again.
	pub fn hide(&mut self) -> Result<(), PluginError> {
		if let Some(editor) = &mut self.editor {
			editor.hide().map_err(|error| PluginError::Error(Box::new(error)))?;
		}

		Ok(())
	}

	/// Set the parent window handle.
	///
	/// The handle belongs to the host and is copied out of it, so it may outlive the
	/// window it was taken from exactly as long as the host keeps that window open.
	pub fn set_parent(&mut self, parent: GuiWindow<'_>) -> Result<(), PluginError> {
		// SAFETY: the handle is borrowed only long enough to copy the platform window out
		// of it; `ParentWindow` owns what it needs from then on.
		let handle = unsafe { parent.borrow_handle_unchecked() }
			.map_err(|_| PluginError::Message("Unsupported window API"))?;
		let parent = ParentWindow::new(handle.as_raw());

		self.parent = Some(parent);

		if let Some(editor) = &mut self.editor {
			editor.set_parent(parent).map_err(|error| PluginError::Error(Box::new(error)))?;
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
		self.scale_factor = scale;
		self.scale_set = true;

		// A backend decides how its own window has to change: on one platform the
		// window keeps its pixel size and the view zooms, on another the window grows
		// and the view stays put.
		if let Some(editor) = &mut self.editor {
			editor.set_scale_factor(scale).map_err(|error| PluginError::Error(Box::new(error)))?;
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
		Some(clack_extensions::gui::GuiSize {
			width: (self.window_size.width as f64 * self.scale_factor) as u32,
			height: (self.window_size.height as f64 * self.scale_factor) as u32,
		})
	}

	fn set_size(&mut self, size: clack_extensions::gui::GuiSize) -> Result<(), clack_plugin::prelude::PluginError> {
		let scale_factor = if self.scale_factor > 0.0 { self.scale_factor } else { 1.0 };
		let logical = WindowSize::new(
			(size.width as f64 / scale_factor) as f32,
			(size.height as f64 / scale_factor) as f32,
		);
		self.window_size = logical;

		if let Some(editor) = &mut self.editor {
			editor.set_size(logical).map_err(|error| PluginError::Error(Box::new(error)))?;
		}

		Ok(())
	}

	fn set_parent(&mut self, window: clack_extensions::gui::Window) -> Result<(), clack_plugin::prelude::PluginError> {
		PluginMain::set_parent(self, window)?;
		Ok(())
	}

	fn set_transient(&mut self, window: clack_extensions::gui::Window) -> Result<(), clack_plugin::prelude::PluginError> {
		PluginMain::set_parent(self, window)?;
		Ok(())
	}

	fn show(&mut self) -> Result<(), clack_plugin::prelude::PluginError> {
		if let Some(editor) = &mut self.editor {
			return editor.show().map_err(|error| PluginError::Error(Box::new(error)));
		}

		self.open()
	}

	fn hide(&mut self) -> Result<(), clack_plugin::prelude::PluginError> {
		PluginMain::hide(self)
	}

	fn adjust_size(&mut self, size: clack_extensions::gui::GuiSize) -> Option<clack_extensions::gui::GuiSize> {
		let scale_factor = if self.scale_factor > 0.0 { self.scale_factor } else { 1.0 };
		self.window_size = WindowSize::new(
			(size.width as f64 / scale_factor) as f32,
			(size.height as f64 / scale_factor) as f32,
		);
		Some(size)
	}

	fn suggest_title(&mut self, title: &str) {
		self.title = title.to_string();
	}
}
