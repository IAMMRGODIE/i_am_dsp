//! The first products with a web faceplate.
//!
//! Each one is the DSP, the metadata its parameters already carry, and one line
//! that names the window. Gain, Drive and Filter are the simple cases on purpose:
//! one or two controls and one picture, so the faceplate itself is what is being
//! tried out, not the effect.
//!
//! The ids end in .web, so a host can load these next to the released iced
//! plug-ins without confusing the two.
//!
//! ```text
//! cargo build -p i_am_effects_web
//! cargo bundle -p i_am_effects_web --no-vst3
//! cargo run -p i_am_effects_web --example dump_face -- target/web-preview gain
//! ```

mod filter;

use filter::Multimode;
use i_am_dsp::prelude::{Clipper, Gain as GainDsp};
use i_am_plugin::export_clap;
use i_am_web_gui::web_plugin;

web_plugin!(
	GainWeb,
	GainDsp,
	|_| GainDsp::new(1.0),
	"iamdsp.effects.gain.web",
	"I Am Gain",
	(360.0, 230.0),
	"#35c77a"
);

web_plugin!(
	DriveWeb,
	Clipper,
	|_| Clipper::new(0, 0.8, 1.0),
	"iamdsp.effects.drive.web",
	"I Am Drive",
	(420.0, 310.0),
	"#ff9a3d",
	mode = "mode"
);

web_plugin!(
	FilterWeb,
	Multimode,
	|rate| Multimode::new(rate, 1_000.0, 0.707),
	"iamdsp.effects.filter.web",
	"I Am Filter",
	(520.0, 400.0),
	"#4c8dff",
	mode = "mode",
	scope = true
);

export_clap!(GainWeb, DriveWeb, FilterWeb);

#[cfg(test)]
mod tests {
	use super::*;
	use std::sync::atomic::Ordering;

	use i_am_dsp::prelude::SetValue;
	use i_am_plugin::Plugin;
	use i_am_web_gui::Faceplate;

	fn gain_face() -> String {
		i_am_web_gui::face_json(
			&GainWeb::new(),
			&Faceplate::new("I Am Gain", "#35c77a", (360.0, 230.0)),
		)
	}

	/// The page builds its controls from this, so every control has to be here with
	/// the layout the DSP declared.
	#[test]
	fn the_gain_faceplate_is_generated_from_metadata() {
		let face = gain_face();

		assert!(face.contains("\"id\":\"gain\""), "{face}");
		assert!(face.contains("\"label\":\"Gain\""), "{face}");
		assert!(face.contains("\"unit\":\"percent\""), "{face}");
		assert!(face.contains("\"log\":true"), "{face}");
		assert!(face.contains("\"size\":\"large\""), "{face}");
		assert!(face.contains("\"hero\":\"waveform\""), "{face}");
		assert!(face.contains("\"size\":[360,230]"), "{face}");
	}

	/// Drive's picture is its transfer curve, and Filter's is its response.
	#[test]
	fn each_effect_names_its_own_picture() {
		let drive = i_am_web_gui::face_json(
			&DriveWeb::new(),
			&Faceplate::new("I Am Drive", "#ff9a3d", (420.0, 310.0)).with_mode("mode"),
		);
		let filter = i_am_web_gui::face_json(
			&FilterWeb::new(),
			&Faceplate::new("I Am Filter", "#4c8dff", (520.0, 400.0)).with_scope(),
		);

		assert!(drive.contains("\"hero\":\"transfer\""), "{drive}");
		assert!(drive.contains("\"id\":\"threshold\""), "{drive}");
		assert!(filter.contains("\"hero\":\"spectrum\""), "{filter}");
		assert!(filter.contains("\"unit\":\"hz\""), "{filter}");
	}

	/// A drag posts a number and it has to reach the map the audio thread reads.
	#[test]
	fn a_web_command_writes_the_parameter() {
		let plugin = GainWeb::new();

		assert!(i_am_web_gui::apply_command(&plugin, "gain=0.5"));

		let map = plugin.param_map();
		assert_eq!(
			map.get("gain").unwrap().load(Ordering::Relaxed),
			SetValue::Float(0.5)
		);
	}

	/// Nothing a page posts may panic or write past a bad parse.
	#[test]
	fn a_malformed_command_is_ignored() {
		let plugin = GainWeb::new();

		assert!(!i_am_web_gui::apply_command(&plugin, "nonsense"));
		assert!(!i_am_web_gui::apply_command(&plugin, "gain=not-a-number"));
		assert!(!i_am_web_gui::apply_command(&plugin, "missing.parameter=1.0"));

		// And the good value is still the one it started with.
		assert_eq!(
			plugin.param_map().get("gain").unwrap().load(Ordering::Relaxed),
			SetValue::Float(1.0)
		);
	}

	/// The page polls this; a control missing from it freezes at its default.
	#[test]
	fn the_state_has_every_control() {
		let plugin = FilterWeb::new();
		let state = i_am_web_gui::state_json(&plugin);

		assert!(state.contains("\"cutoff\""), "{state}");
		assert!(state.contains("\"q\""), "{state}");
	}

	/// A product with controls has an editor, and it is the web one.
	#[test]
	fn each_product_has_a_web_editor() {
		assert!(GainWeb::new().editor().is_some());
		assert!(DriveWeb::new().editor().is_some());
		assert!(FilterWeb::new().editor().is_some());
	}

	/// The filter has three algorithms behind one set of controls, and the page
	/// is told to draw the switch in its header.
	#[test]
	fn the_filter_has_a_mode_switch() {
		let plugin = FilterWeb::new();
		let face = i_am_web_gui::face_json(
			&plugin,
			&Faceplate::new("I Am Filter", "#4c8dff", (520.0, 400.0))
				.with_mode("mode")
				.with_scope(),
		);

		assert!(face.contains("\"mode\":\"mode\""), "{face}");
		assert!(face.contains("\"scope\":true"), "{face}");
		assert!(face.contains("\"id\":\"mode\""), "{face}");
		assert!(
			face.contains("\"choices\":[\"Lowpass\",\"Highpass\",\"Bandpass\"]"),
			"{face}"
		);
	}

	/// Drive's two algorithms share one parameter set, so a mode switch keeps
	/// the knobs where they were.
	#[test]
	fn the_drive_has_a_mode_switch() {
		let face = i_am_web_gui::face_json(
			&DriveWeb::new(),
			&Faceplate::new("I Am Drive", "#ff9a3d", (420.0, 310.0)).with_mode("mode"),
		);

		assert!(face.contains("\"mode\":\"mode\""), "{face}");
		assert!(face.contains("\"choices\":[\"Hard\",\"Soft\"]"), "{face}");
		assert!(face.contains("\"id\":\"threshold\""), "{face}");
		assert!(face.contains("\"label\":\"Amount\""), "{face}");
	}

	/// Crusher Web asked for resizing with an aspect ratio; these lay themselves
	/// out, so they ask for resizing without one.
	#[test]
	fn the_windows_are_resizable() {
		for options in [
			GainWeb::window_options(),
			DriveWeb::window_options(),
			FilterWeb::window_options(),
		] {
			let hints = options.resize_hints.expect("the web windows are resizable");

			assert!(hints.can_resize_horizontally);
			assert!(hints.can_resize_vertically);
			assert!(matches!(
				hints.strategy,
				i_am_plugin::AspectRatioStrategy::Disregard
			));
		}
	}

	/// Bypass is not a parameter: the processor has to skip its effect and leave
	/// the frames it was handed alone.
	#[test]
	fn bypass_passes_the_input_untouched() {
		use i_am_dsp::Processor;

		let mut plugin = GainWeb::new();
		// A host, and the editor, write the map; set_parameter on the product is
		// for a caller that owns it and would be overwritten by the next sync.
		plugin
			.param_map()
			.set("gain", SetValue::Float(0.5), Ordering::SeqCst);

		let mut context: Box<dyn i_am_dsp::ProcessContext> = Box::new(());
		let mut bypassed = [0.25_f32, -0.5];
		plugin.set_bypassed(true);
		plugin.process(&mut bypassed, &[], &mut context);
		assert_eq!(bypassed, [0.25, -0.5]);

		let mut active = [0.25_f32, -0.5];
		plugin.set_bypassed(false);
		plugin.process(&mut active, &[], &mut context);
		assert_eq!(active, [0.125, -0.25]);
	}
}
