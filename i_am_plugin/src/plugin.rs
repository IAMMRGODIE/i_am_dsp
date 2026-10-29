//! plugin

use clack_extensions::gui::GuiResizeHints;
use i_am_dsp::Processor;
use i_am_dsp::prelude::ParamMap;

use crate::descriptor::{AudioPort, Descriptor, MidiPort};
use crate::editor::Editor;

/// A trait for plug-ins exported by this crate.
///
/// It is a processor from `i_am_dsp` plus everything a host asks about: an id, the
/// audio and note ports it has, the parameters it declares, and -- if it has one --
/// the GUI it opens. Nothing here names a GUI toolkit: [editor](Self::editor) hands
/// back an [Editor], and the toolkit that implements it is chosen by the plug-in's
/// crate, not by this one.
///
/// Will automatically generate the id for audio ports.
/// 
/// Currently dynamical audio ports are not supported.
pub trait Plugin: Processor {
	/// The plugin's descriptor.
	const DESCRIPTOR: Descriptor;
	/// The input ports of the plugin.
	const INPUT_PORTS: &'static [AudioPort] = &[AudioPort {
		name: "Main",
	}];
	/// The output ports of the plugin.
	const OUTPUT_PORTS: &'static [AudioPort] = &[AudioPort {
		name: "Main",
	}];
	/// The MIDI input ports of the plugin.
	const INPUT_MIDI_PORTS: &'static [MidiPort] = &[];
	/// The MIDI output ports of the plugin.
	const OUTPUT_MIDI_PORTS: &'static [MidiPort] = &[];

	/// Whether this plug-in has a GUI.
	///
	/// This is a constant rather than something worked out from
	/// [editor](Self::editor) because a host has to be told whether the GUI extension
	/// is there before any editor is built. Set it to `true` in the same impl that
	/// returns an editor; leave it alone for a plug-in that has no interface.
	const HAS_GUI: bool = false;

	/// Create a new instance of the plugin.
	fn new() -> Self;

	/// Get window options for the plugin GUI.
	///
	/// Defaults to a 640x480 window called "Plugin GUI"; a plug-in with no GUI never
	/// gets asked.
	fn window_options() -> WindowOptions {
		WindowOptions::default()
	}

	/// Get the plugin's parameter map.
	fn param_map(&self) -> ParamMap;

	/// Push interface-only parameter changes into the effect.
	///
	/// A GUI writes the shared map and the effect reads it on the next process
	/// block, but a host can save between the two. The wrapper calls this before
	/// collecting the state so the saved values are the ones the user sees.
	fn sync_params(&mut self) {}

	/// Build the editor for this plug-in, if it has one.
	///
	/// Called once each time the host opens the GUI, so a closed and reopened editor
	/// starts from a fresh view. The editor is generic over the GUI toolkit: an
	/// `i_am_plugin_iced` editor, one for egui, or one written for a bespoke renderer
	/// are all handed back the same way.
	fn editor(&self) -> Option<Box<dyn Editor<Self>>> {
		None
	}

	/// Called when the host activates the plug-in, with the sample rate it will run
	/// at.
	///
	/// This is the first moment a plug-in can know it. One is built before the host
	/// has said anything about the session -- [new](Self::new) takes no arguments,
	/// because a host asks for a plug-in before it has a session to ask about -- so an
	/// effect whose coefficients depend on the rate is built with the wrong ones until
	/// this arrives, and rebuilding here is what fixes it.
	///
	/// A plug-in that rebuilds has to keep the parameter map it already handed out:
	/// a host reads parameters through the map it was given, and a new one would leave
	/// its automation writing into a map nothing is reading.
	fn on_activate(&mut self, sample_rate: usize) {
		let _ = sample_rate;
	}

	/// The function to be called when host sends `reset` to plugin.
	fn on_reset(&mut self) {}

	/// The function to be called when host requests to start processing.
	fn on_start_processing(&mut self) {}

	/// The function to be called when host requests to stop processing.
	fn on_stop_processing(&mut self) {}

	/// Whether the host should be told when the plugin's latency changes.
	///
	/// It does when a synced delay takes its length from the tempo: a host that implements
	/// plugin delay compensation is told, and compensates again. A host that only reads latency
	/// when the plugin is activated cannot, however, and a plugin may prefer to leave it with
	/// the number it saw then rather than to ask for something it will not do. Overriding this
	/// to `false` keeps the report to that first number.
	fn reports_latency_changes(&self) -> bool {
		true
	}
}

/// A trait for plugin to export as Auv2 plugin.
pub trait PluginAuExt: Plugin {
	/// The AU type of the plugin.
	const AU_TYPE: [u8; 4];
	/// The AU subtype of the plugin.
	const AU_SUBTYPE: [u8; 4];
}

/// A logical size, in points, as a GUI lays itself out.
///
/// It is deliberately not a toolkit's size type: the host asks in logical points and
/// the scale factor turns them into pixels, which every backend does for itself.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindowSize {
	/// The width, in logical points.
	pub width: f32,
	/// The height, in logical points.
	pub height: f32,
}

impl WindowSize {
	/// Create a new size.
	pub const fn new(width: f32, height: f32) -> Self {
		Self { width, height }
	}
}

impl Default for WindowSize {
	fn default() -> Self {
		Self {
			width: 640.0,
			height: 480.0,
		}
	}
}

/// The window options for the plugin GUI.
#[derive(Debug, Clone)]
pub struct WindowOptions {
	/// The logical size the GUI is laid out for.
	pub window_size: WindowSize,
	/// The title of the plugin GUI.
	pub title: String,
	/// The resize hints of the plugin GUI.
	/// 
	/// None for can't resize.
	pub resize_hints: Option<GuiResizeHints>,
}

impl WindowOptions {
	/// Create a new `WindowOptions` with default values.
	pub fn new() -> Self {
		Self::default()
	}

	/// Set the window size of the plugin GUI.
	pub fn with_size(self, size: (f64, f64)) -> Self {
		Self {
			window_size: WindowSize::new(size.0 as f32, size.1 as f32),
			..self
		}
	}

	/// Set the title of the plugin GUI.
	pub fn with_title(self, title: &str) -> Self {
		Self {
			title: title.to_string(),
			..self
		}
	}

	/// Let the host resize the GUI.
	pub fn with_resize_hints(self, hints: GuiResizeHints) -> Self {
		Self {
			resize_hints: Some(hints),
			..self
		}
	}
}

impl Default for WindowOptions {
	fn default() -> Self {
		Self {
			window_size: WindowSize::default(),
			title: "Plugin GUI".to_string(),
			resize_hints: None,
		}
	}
}
