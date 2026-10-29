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

mod disperser;
mod filter;

use disperser::Dispersion;
use filter::Multimode;
use i_am_dsp::prelude::{
	Crusher, Distortion, Gain as GainDsp, IIRFreqShifter, LrControl, PitchShifter,
	StereoController,
};
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

/// The controls each distortion mode reads; the page hides the rest.
pub const DISTORTION_VISIBLE: &[(i32, &[&str])] = &[
	(0, &["threshold", "gain"]),
	(1, &["alpha", "gain"]),
	(2, &["k", "gain"]),
	(3, &["a", "p", "gain"]),
	(4, &["bits", "gain"]),
	(5, &["target_sample_rate", "gain"]),
];

web_plugin!(
	DistortionWeb,
	Distortion<2>,
	Distortion::<2>::new,
	"iamdsp.effects.distortion.web",
	"I Am Distortion",
	(420.0, 310.0),
	"#ff9a3d",
	mode = "mode",
	visible = DISTORTION_VISIBLE
);

web_plugin!(
	FilterWeb,
	Multimode,
	|rate| Multimode::new(rate, 1_000.0, 0.707),
	"iamdsp.effects.filter.web",
	"I Am Filter",
	(520.0, 330.0),
	"#4c8dff",
	mode = "mode"
);

web_plugin!(
	WidthWeb,
	StereoController,
	|_| StereoController::new(),
	"iamdsp.effects.width.web",
	"I Am Width",
	(360.0, 230.0),
	"#ff6fae"
);

web_plugin!(
	PitchWeb,
	PitchShifter<2>,
	|_| PitchShifter::<2>::new(1_024),
	"iamdsp.effects.pitch.web",
	"I Am Pitch",
	(360.0, 230.0),
	"#c8e04a"
);

/// A shift in hertz crosses zero, so the knob edits it through atan: fine near
/// zero, quick at the ends.
pub const SHIFTER_CURVES: &[(&str, &str)] = &[("shift_freq", "atan")];

web_plugin!(
	ShifterWeb,
	IIRFreqShifter<8, 2>,
	|rate| IIRFreqShifter::<8, 2>::new(rate, 0.0),
	"iamdsp.effects.shifter.web",
	"I Am Shifter",
	(360.0, 230.0),
	"#7aa2ff",
	curves = SHIFTER_CURVES
);

web_plugin!(
	DisperserWeb,
	Dispersion,
	Dispersion::new,
	"iamdsp.effects.disperser.web",
	"I Am Disperser",
	(380.0, 250.0),
	"#2fc6c0"
);

web_plugin!(
	BalanceWeb,
	LrControl,
	|_| LrControl::default(),
	"iamdsp.effects.balance.web",
	"I Am Balance",
	(360.0, 230.0),
	"#d96aff"
);

web_plugin!(
	CrusherWeb,
	Crusher<2>,
	Crusher::<2>::new,
	"iamdsp.effects.crusher.web",
	"I Am Crusher Web",
	(400.0, 640.0),
	"#8a6aff"
);

export_clap!(
	GainWeb,
	DistortionWeb,
	FilterWeb,
	WidthWeb,
	PitchWeb,
	ShifterWeb,
	DisperserWeb,
	BalanceWeb,
	CrusherWeb
);

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

	/// Distortion's picture is its transfer curve, and Filter's is its response.
	#[test]
	fn each_effect_names_its_own_picture() {
		let distortion = i_am_web_gui::face_json(
			&DistortionWeb::new(),
			&Faceplate::new("I Am Distortion", "#ff9a3d", (420.0, 310.0))
				.with_mode("mode")
				.with_visible(DISTORTION_VISIBLE),
		);
		let filter = i_am_web_gui::face_json(
			&FilterWeb::new(),
			&Faceplate::new("I Am Filter", "#4c8dff", (520.0, 330.0)),
		);

		assert!(distortion.contains("\"hero\":\"transfer\""), "{distortion}");
		assert!(distortion.contains("\"id\":\"threshold\""), "{distortion}");
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
		assert!(DistortionWeb::new().editor().is_some());
		assert!(FilterWeb::new().editor().is_some());
		assert!(WidthWeb::new().editor().is_some());
		assert!(PitchWeb::new().editor().is_some());
		assert!(ShifterWeb::new().editor().is_some());
		assert!(DisperserWeb::new().editor().is_some());
		assert!(BalanceWeb::new().editor().is_some());
		assert!(CrusherWeb::new().editor().is_some());
	}

	/// The crusher has three named sections, and the Drive one shows the scope.
	#[test]
	fn the_crusher_has_its_sections_and_a_scope() {
		let face = i_am_web_gui::face_json(
			&CrusherWeb::new(),
			&Faceplate::new("I Am Crusher Web", "#8a6aff", (400.0, 640.0)),
		);

		assert!(face.contains("\"hero\":\"waveform\""), "{face}");
		assert!(face.contains("\"id\":\"amount\""), "{face}");
		assert!(face.contains("\"name\":\"Drive\""), "{face}");
		assert!(face.contains("\"name\":\"Amplitude\""), "{face}");
		assert!(face.contains("\"name\":\"Gain\""), "{face}");
	}

	/// Two more modules, each with the picture its section asked for.
	#[test]
	fn width_and_pitch_have_their_own_pictures() {
		let width = i_am_web_gui::face_json(
			&WidthWeb::new(),
			&Faceplate::new("I Am Width", "#ff6fae", (360.0, 230.0)),
		);
		let pitch = i_am_web_gui::face_json(
			&PitchWeb::new(),
			&Faceplate::new("I Am Pitch", "#c8e04a", (360.0, 230.0)),
		);

		assert!(width.contains("\"hero\":\"stereo\""), "{width}");
		assert!(width.contains("\"id\":\"mid_gain\""), "{width}");
		assert!(width.contains("\"label\":\"Mid\""), "{width}");
		assert!(pitch.contains("\"hero\":\"pitch\""), "{pitch}");
		assert!(pitch.contains("\"id\":\"pitch_shift_factor\""), "{pitch}");
		assert!(pitch.contains("\"label\":\"Shift\""), "{pitch}");
	}

	/// The filter has three algorithms behind one set of controls, and the page
	/// is told to draw the switch in its header.
	#[test]
	fn the_filter_has_a_mode_switch() {
		let plugin = FilterWeb::new();
		let face = i_am_web_gui::face_json(
			&plugin,
			&Faceplate::new("I Am Filter", "#4c8dff", (520.0, 330.0)).with_mode("mode"),
		);

		assert!(face.contains("\"mode\":\"mode\""), "{face}");
		assert!(face.contains("\"id\":\"mode\""), "{face}");
		assert!(
			face.contains("\"choices\":[\"Lowpass\",\"Highpass\",\"Bandpass\"]"),
			"{face}"
		);
	}

	/// Distortion has six algorithms, and each mode lists the controls it reads.
	#[test]
	fn the_distortion_has_modes_and_visible_controls() {
		let face = i_am_web_gui::face_json(
			&DistortionWeb::new(),
			&Faceplate::new("I Am Distortion", "#ff9a3d", (420.0, 310.0))
				.with_mode("mode")
				.with_visible(DISTORTION_VISIBLE),
		);

		assert!(face.contains("\"mode\":\"mode\""), "{face}");
		assert!(
			face.contains("\"choices\":[\"Hard\",\"Soft\",\"Overdrive\",\"Saturator\",\"Bitcrush\",\"Downsample\"]"),
			"{face}"
		);
		assert!(face.contains("\"id\":\"threshold\""), "{face}");
		assert!(face.contains("\"id\":\"bits\""), "{face}");
		assert!(face.contains("\"visible\":[[0,[\"threshold\",\"gain\"]]"), "{face}");
		assert!(face.contains("[4,[\"bits\",\"gain\"]]"), "{face}");
	}

	/// The A group: shifter, disperser and balance, each with its picture.
	#[test]
	fn the_a_group_has_its_pictures() {
		let shifter = i_am_web_gui::face_json(
			&ShifterWeb::new(),
			&Faceplate::new("I Am Shifter", "#7aa2ff", (360.0, 230.0))
				.with_curves(SHIFTER_CURVES),
		);
		let disperser = i_am_web_gui::face_json(
			&DisperserWeb::new(),
			&Faceplate::new("I Am Disperser", "#2fc6c0", (380.0, 250.0)),
		);
		let balance = i_am_web_gui::face_json(
			&BalanceWeb::new(),
			&Faceplate::new("I Am Balance", "#d96aff", (360.0, 230.0)),
		);

		assert!(shifter.contains("\"hero\":\"spectra\""), "{shifter}");
		assert!(shifter.contains("\"id\":\"shift_freq\""), "{shifter}");
		assert!(shifter.contains("\"curves\":{\"shift_freq\":\"atan\"}"), "{shifter}");
		assert!(disperser.contains("\"hero\":\"phase\""), "{disperser}");
		assert!(disperser.contains("\"id\":\"amount\""), "{disperser}");
		assert!(disperser.contains("\"id\":\"cutoff\""), "{disperser}");
		assert!(balance.contains("\"hero\":\"balance\""), "{balance}");
		assert!(balance.contains("\"label\":\"Left\""), "{balance}");
	}

	/// Crusher Web asked for resizing with an aspect ratio; these lay themselves
	/// out, so they ask for resizing without one.
	#[test]
	fn the_windows_are_resizable() {
		for options in [
			GainWeb::window_options(),
			DistortionWeb::window_options(),
			FilterWeb::window_options(),
			WidthWeb::window_options(),
			PitchWeb::window_options(),
			ShifterWeb::window_options(),
			DisperserWeb::window_options(),
			BalanceWeb::window_options(),
			CrusherWeb::window_options(),
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
