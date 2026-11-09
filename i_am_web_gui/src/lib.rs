//! A web faceplate for i_am_plugin.
//!
//! A plug-in's parameters already say which section they belong to, whether they
//! want a knob or a slider, how big they are and what picture their section shows.
//! This crate turns that into a page: [Faceplate] adds the name, the identity
//! colour, an optional mode switch and a bypass flag; [editor] serves the page over
//! a custom protocol, and the page draws and drives the controls. Nothing here
//! links a toolkit; WebView2 comes from i_am_plugin::wry.
//!
//! The page gets these routes:
//!
//! | Route | What it is |
//! | --- | --- |
//! | /face | the controls, their layout, the mode switch and the identity, as JSON |
//! | /state | the values the host holds, plus whether the plug-in is bypassed |
//! | /kit.css, /kit.js | the page's own assets |
//!
//! A control writes back with window.ipc.postMessage("id=value"), which is parsed
//! into the parameter's own type and stored in the map the audio thread reads. The
//! bypass switch writes __bypass=1 or __bypass=0, which is not a parameter: it is
//! state of the plug-in, and the product's processor reads the same flag.
//!
//! ```ignore
//! impl i_am_plugin::Plugin for MyPlugin {
//!     const HAS_GUI: bool = true;
//!
//!     fn editor(&self) -> Option<Box<dyn i_am_plugin::Editor<Self>>> {
//!         i_am_web_gui::editor(
//!             self,
//!             i_am_web_gui::Faceplate::new("My Plugin", "#8a6aff", (360.0, 230.0)),
//!             std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
//!             std::sync::Arc::new(i_am_web_gui::Scope::new()),
//!             48_000,
//!         )
//!     }
//! }
//! ```

mod assets;
mod editor;
mod face;
mod json;
mod macros;
mod scope;
mod text;

pub use assets::{CARD_SCRIPT, CARD_STYLE, PageAssets};
#[cfg(test)]
pub(crate) use assets::{SCRIPT, STYLE};
pub use editor::{editor, editor_with, editor_with_page};
pub use face::{Control, Faceplate, Kind, Page, Section, Switch, sections};
#[cfg(test)]
use i_am_plugin::Plugin;
pub use json::{apply_command, controls, face_json, state_json};
pub use scope::{Channels, Scope};
pub use text::{json_escape, percent_decode, window_options};

pub use i_am_plugin::wry::{ProtocolRequest, ProtocolResponse};

/// A test binary does not export a plug-in, but the clap-wrapper static library
/// that the vst3 feature links still references the clap_entry data symbol.
#[cfg(test)]
#[unsafe(no_mangle)]
static clap_entry: [u8; 64] = [0; 64];

#[cfg(test)]
mod ui_contract_tests {
	use super::*;

	/// Both pages must load the same card assets, and the card first: the
	/// faceplate script delegates to the card, so the card has to exist before
	/// the page's own script runs.
	#[test]
	fn both_pages_load_the_shared_card() {
		let face = PageAssets::FACEPLATE.html;

		assert!(face.contains("/card.css"));
		assert!(face.contains("/card.js"));
		assert!(
			face.find("/card.js").unwrap() < face.find("/kit.js").unwrap(),
			"card.js must load before kit.js"
		);
	}

	/// The faceplate no longer owns a control implementation; every widget is
	/// a card handle. Catching an accidental re-implementation here is cheaper
	/// than noticing the two pages drift apart in a screenshot.
	#[test]
	fn the_faceplate_only_delegates_controls() {
		for handle in [
			"IAmCard.controlHandle",
			"IAmCard.switchHandle",
			"IAmCard.choiceHandle",
			"IAmCard.sliderHandle",
		] {
			assert!(SCRIPT.contains(handle), "missing {handle}");
		}

		for legacy in [
			"function knobGeometry",
			"function arcPath",
			"function open_knob_editor",
			"class=\"mode-choice\"",
		] {
			assert!(!SCRIPT.contains(legacy), "legacy faceplate code: {legacy}");
		}

		assert!(
			!SCRIPT.contains("IAmCard.registerHero"),
			"hero renderers live in card.js"
		);
	}

	/// The card is the public surface both pages draw through.
	#[test]
	fn the_card_exposes_every_shared_control() {
		for export in [
			"controlHandle:",
			"switchHandle:",
			"choiceHandle:",
			"sliderHandle:",
			"hero:",
			"registerHero:",
			"bandsOf:",
			"smoothBands:",
			"bandsPath:",
		] {
			assert!(CARD_SCRIPT.contains(export), "missing card export {export}");
		}
	}

	/// A logarithmic range that reaches zero or crosses it uses the same symlog
	/// curve in both the card and the faceplate, so the freq shifter no longer
	/// needs its own atan curve.
	#[test]
	fn bipolar_log_parameters_use_symlog_in_both_pages() {
		for script in [CARD_SCRIPT, SCRIPT] {
			assert!(script.contains("Math.asinh"), "missing symlog forward map");
			assert!(script.contains("Math.sinh"), "missing symlog inverse map");
			assert!(
				!script.contains("atanKnee"),
				"the old atan curve is still present"
			);
		}
	}

	/// A control can replace its formatted readout without changing its unit.
	#[test]
	fn a_control_can_override_its_readout() {
		assert!(CARD_SCRIPT.contains("function display(spec, value)"));
		assert!(CARD_SCRIPT.contains("typeof spec.format === \"function\""));
	}

	/// The utility products' pictures are part of the shared registry too.
	#[test]
	fn the_new_product_heroes_are_registered() {
		for kind in [
			"resonator",
			"formant",
			"autotune",
			"partials",
			"string",
			"vibrato",
			"shape",
			"envelope",
			"additive",
			"canvas",
		] {
			assert!(
				CARD_SCRIPT.contains(&format!("registerHero(\"{kind}\"")),
				"missing {kind} hero"
			);
		}
		assert!(
			CARD_SCRIPT.contains("notePickerHandle"),
			"the note picker handle is missing"
		);
		assert!(
			CARD_STYLE.contains(".mcard-note.on"),
			"the note picker needs a selected state"
		);
		assert!(
			SCRIPT.contains("buildNotePicker"),
			"the faceplate does not build the note picker"
		);
		assert!(
			include_str!("page/kit.css").contains(".face-notes"),
			"the faceplate does not size the note picker"
		);
	}

	/// Hover feedback belongs to the shared controls, not to one page.
	#[test]
	fn shared_controls_keep_their_hover_feedback() {
		for selector in [
			".knob:hover",
			".mcard-switch:not(.on):hover",
			".mcard-select:hover",
			".mcard-slider-track:hover",
		] {
			assert!(
				CARD_STYLE.contains(selector),
				"missing shared hover {selector}"
			);
		}

		assert!(STYLE.contains(".face-knob .knob:hover{"));
		assert!(STYLE.contains(".choice .face-choice:hover{"));
	}

	/// Both script editors — the faceplate drawer and the rack card — scroll like
	/// the rack's panes, not with the platform's own scrollbar.
	#[test]
	fn the_script_editors_have_the_rack_scrollbar() {
		for (css, editor) in [(STYLE, ".script-area"), (CARD_STYLE, ".mcard-script-text")] {
			for part in [
				"::-webkit-scrollbar",
				"::-webkit-scrollbar-track",
				"::-webkit-scrollbar-thumb",
			] {
				let rule = format!("{editor}{part}");

				assert!(css.contains(&rule), "missing {rule}");
			}

			assert!(
				css.contains("background:#3a3a3a; border-radius:4px"),
				"{editor} does not use the rack's thumb"
			);
		}
	}

	/// The voice page draws the unison the controls describe: one dot per copy in
	/// the pitch/stereo field, and the wave the copies add up to beside it.
	#[test]
	fn the_voice_page_draws_the_unison() {
		assert!(
			CARD_SCRIPT.contains("registerHero(\"unison\", unisonHero)"),
			"the shared card never registers the unison"
		);
		assert!(
			SCRIPT.contains("hero.kind === \"unison\""),
			"the voice page never asks the host for its output"
		);
		assert!(STYLE.contains(".unison-hero"));
		assert!(CARD_STYLE.contains(".mcard-hero .unison-hero"));
		assert!(
			SCRIPT.contains("active: activeControl"),
			"the held control never reaches the hero"
		);
		assert!(
			SCRIPT.contains("setActiveControl"),
			"the page never tracks the held control"
		);
	}

	/// The spectrum page is an editor: the shared card draws the levels and the
	/// phase strip, and the page fetches and posts the pages it draws.
	#[test]
	fn the_spectrum_page_draws_and_edits_the_pages() {
		assert!(
			CARD_SCRIPT.contains("registerHero(\"spectrum\", spectrumHero)"),
			"the shared card never registers the spectrum editor"
		);
		assert!(CARD_SCRIPT.contains("wireSpectrum"));
		assert!(
			SCRIPT.contains("spectrum: spectrumState"),
			"the hero never sees the pages"
		);
		assert!(
			SCRIPT.contains("function pollSpectrum"),
			"the page never asks for the pages"
		);

		// Both editors' page buttons are HTML: text in a stretched hero SVG
		// changes shape with the window.
		assert!(
			CARD_SCRIPT.contains("data-action=\"spectrum-add\">+ PAGE"),
			"the spectrum buttons are not html"
		);
		assert!(
			CARD_SCRIPT.contains("data-action=\"page-add\">+ PAGE"),
			"the canvas buttons are not html"
		);

		// The canvas is drawn in a centred square, so a circle stays a circle, and
		// every bezier handle is joined to the point it bends.
		assert!(CARD_SCRIPT.contains("class=\"canvas-box\""));
		assert!(STYLE.contains(".canvas-hero"));
		assert!(CARD_STYLE.contains(".mcard-hero .canvas-box"));
		assert!(CARD_SCRIPT.contains("guide(previous, control1)"));
	}

	/// A faceplate knob keeps the footprint it had before the shared-card
	/// migration: a 56 px dial in a 64 px slot, not the rack's 112 px slot. The
	/// larger slot changes the row count and squeezes the hero out.
	#[test]
	fn a_faceplate_knob_keeps_its_old_footprint() {
		let css = PageAssets::FACEPLATE.css;

		assert!(css.contains(".face-knob{width:auto; min-width:calc(64px * var(--scale))}"));
		assert!(css.contains("width:calc(56px * var(--scale)); height:calc(56px * var(--scale))"));
	}

	/// The shared renderers need the page's current parameter values; passing
	/// null silently falls back to every default and detaches a hero from its knobs.
	#[test]
	fn heroes_receive_the_current_parameter_values() {
		assert!(SCRIPT.contains("function heroValues"));
		assert!(SCRIPT.contains("IAmCard.hero(hero.kind, values"));
		assert!(
			!SCRIPT.contains("IAmCard.hero(hero.kind, null"),
			"the faceplate must not discard its control values"
		);
		assert!(
			CARD_SCRIPT.contains("typeof value === \"boolean\""),
			"hero values include switches, which are booleans"
		);
	}

	/// A live hero draws the audio, so the card has to hand its context to the
	/// renderer and let the page ask for a redraw between knob moves.
	#[test]
	fn a_card_hands_its_context_to_the_hero() {
		assert!(
			CARD_SCRIPT.contains("hero(options.hero, values, options.context)"),
			"the hero never sees the scope"
		);
		assert!(
			CARD_SCRIPT.contains("card.repaintHero = paintHero"),
			"a live hero cannot be redrawn"
		);
	}

	/// The string instrument's metadata is what its faceplate will be: three
	/// sections, and a preset that is a menu rather than a number.
	#[test]
	fn the_string_instrument_describes_a_faceplate() {
		use i_am_dsp::prelude::{ParamWidget, Parameters, StringInstrument};

		let parameters = StringInstrument::new(48_000).get_parameters();
		let sections = crate::face::sections(&parameters);
		let names: Vec<&str> = sections
			.iter()
			.map(|section| section.name.as_deref().unwrap_or(""))
			.collect();

		assert_eq!(names, ["String", "Body", "Performance"]);
		assert_eq!(sections[0].hero.as_deref(), Some("string"));
		assert_eq!(sections[1].hero.as_deref(), Some("partials"));
		assert_eq!(sections[2].hero.as_deref(), Some("vibrato"));

		let preset = parameters
			.iter()
			.find(|parameter| parameter.identifier == "preset")
			.expect("the preset parameter");
		let choices = preset.meta.choices.expect("preset choices");

		assert_eq!(choices, ["Violin", "Cello"].as_slice());
		assert!(matches!(preset.meta.layout.widget, ParamWidget::Choice));
	}

	/// A mode is a choice: two options and seven options use the same control.
	#[test]
	fn a_mode_is_the_shared_select() {
		let start = SCRIPT.find("function buildMode").expect("buildMode");
		let end = SCRIPT[start..]
			.find("function buildFace")
			.expect("buildFace")
			+ start;
		let mode = &SCRIPT[start..end];

		assert!(mode.contains("IAmCard.choiceHandle"));
		assert!(!mode.contains("mode-choice"));
	}
}

/// A synthesizer is a plug-in with no audio input and a note port, and the
/// declaration has to be able to say so. This is the smallest one that compiles.
#[cfg(test)]
mod synth_contract_tests {
	// The declared methods are part of every product's interface; this one is
	// only ever compiled, not opened.
	#![allow(dead_code)]

	use super::*;
	use i_am_dsp::prelude::{Generator, Parameter, Parameters, SetValue};
	use i_am_plugin::Tag;

	struct Tone;

	impl Parameters for Tone {
		fn get_parameters(&self) -> Vec<Parameter> {
			Vec::new()
		}

		fn set_parameter(&mut self, _: &str, _: SetValue) -> bool {
			false
		}
	}

	impl Generator<2> for Tone {
		fn generate(&mut self, _: &mut Box<dyn i_am_dsp::ProcessContext>) -> [f32; 2] {
			[0.0; 2]
		}
	}

	web_plugin!(
		SynthContract,
		i_am_dsp::GeneratorEffect<Tone>,
		|_rate| i_am_dsp::GeneratorEffect::new(Tone),
		"com.iamplugins.test.synth",
		"Synth Contract",
		(400.0, 300.0),
		"#88cc44",
		ports = &[],
		midi = &[i_am_plugin::MidiPort { name: "Notes" }],
		tags = &[Tag::Instrument, Tag::Synthesizer]
	);

	#[test]
	fn a_synthesizer_declares_a_note_port_and_no_audio_input() {
		assert!(<SynthContract as Plugin>::INPUT_PORTS.is_empty());
		assert_eq!(<SynthContract as Plugin>::INPUT_MIDI_PORTS.len(), 1);
	}

	#[test]
	fn a_synthesizer_lands_in_the_instrument_menu() {
		let tags = <SynthContract as Plugin>::DESCRIPTOR.tags;

		assert_eq!(tags.len(), 2);
		assert!(matches!(tags[0], Tag::Instrument));
		assert!(matches!(tags[1], Tag::Synthesizer));
	}
}
