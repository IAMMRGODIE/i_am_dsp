//! plugin

use crate::shared::*;
use crate::descriptor::{AudioPort, Descriptor, MidiPort};

/// A trait for plugins that can be used with the `i_am_dsp_iced` crate.
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

	/// Create a new instance of the plugin.
	fn new() -> Self;

	/// Get window options for the plugin GUI.
	fn window_options() -> WindowOptions;

	/// Get the plugin's parameter map.
	fn param_map(&self) -> ParamMap;

	/// Called when the host activates the plug-in, with the sample rate it will run
	/// at.
	///
	/// This is the first moment a plug-in can know it. One is built before the host
	/// has said anything about the session -- [`new`](Self::new) takes no arguments,
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

	/// Optional embedded font bytes (e.g. `include_bytes!(...)`). When set,
	/// the GUI loads it as it opens so `Family::Name(...)` text renders with it.
	const EMBEDDED_FONT: Option<&'static [u8]> = None;
}

/// A trait for plugin to export as Auv2 plugin.
pub trait PluginAuExt: Plugin {
	/// The AU type of the plugin.
	const AU_TYPE: [u8; 4];
	/// The AU subtype of the plugin.
	const AU_SUBTYPE: [u8; 4];
}

/// The window options for the plugin GUI.
#[derive(Debug, Clone)]
pub struct WindowOptions {
	/// The window Options used to create the plugin GUI.
	pub window_size: Size,
	// /// The window scale policy used to create the plugin GUI.
	// pub scale_factor: WindowScalePolicy,
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
			window_size: Size::new(size.0, size.1),
			..self
		}
	}

	/* /// Set the window size of the plugin GUI to the system scale factor.
	pub fn with_system_scale_factor(self) -> Self {
		Self {
			scale_factor: WindowScalePolicy::SystemScaleFactor,
			..self
		}
	}

	/// Set the window scale factor of the plugin GUI.
	pub fn with_scale_factor(self, scale_factor: f64) -> Self {
		Self {
			scale_factor: WindowScalePolicy::ScaleFactor(scale_factor),
			..self
		}
	} */

	/// Set the title of the plugin GUI.
	pub fn with_title(self, title: &str) -> Self {
		Self {
			title: title.to_string(),
			..self
		}
	}
}

impl Default for WindowOptions {
	fn default() -> Self {
		Self {
			window_size: Size::new(640.0, 480.0),
			// scale_factor: WindowScalePolicy::SystemScaleFactor,
			title: "Plugin GUI".to_string(),
			resize_hints: None,
		}
	}
}
