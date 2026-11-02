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

mod face;
mod scope;

pub use face::{Control, Faceplate, Kind, Section, Switch};
pub use scope::{Channels, Scope};

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use i_am_plugin::wry::WryEditor;
pub use i_am_plugin::wry::{ProtocolRequest, ProtocolResponse};
use i_am_plugin::{Editor, Plugin};

/// The page itself.
const PAGE: &str = include_str!("page/index.html");
/// The page's styles.
const STYLE: &str = include_str!("page/kit.css");
/// The page's script.
const SCRIPT: &str = include_str!("page/kit.js");

/// The card a faceplate and a rack module share, served at `/card.css`.
pub const CARD_STYLE: &str = include_str!("page/card.css");
/// The card's drawing code, served at `/card.js`.
pub const CARD_SCRIPT: &str = include_str!("page/card.js");

/// The three assets a page is made of.
///
/// A product that draws something the metadata cannot describe -- a rack, a
/// curve editor, a browser of its own -- hands its own assets here. The routes
/// stay the same (`/`, `/kit.css`, `/kit.js`), so the page can be embedded
/// without adding another protocol or another editor backend.
#[derive(Debug, Clone, Copy)]
pub struct PageAssets {
	/// The HTML the editor loads.
	pub html: &'static str,
	/// The stylesheet served at `/kit.css`.
	pub css: &'static str,
	/// The script served at `/kit.js`.
	pub js: &'static str,
}

impl PageAssets {
	/// The faceplate page: controls generated from parameter metadata.
	pub const FACEPLATE: Self = Self {
		html: PAGE,
		css: STYLE,
		js: SCRIPT,
	};
}

/// The editor for a plug-in whose controls the metadata describes.
///
/// Returns nothing when the plug-in has no controls a person can move: a page with
/// no knobs is a blank window, which is worse than telling the host there is none.
///
/// The bypass flag is shared with the product's processor, so turning the switch
/// off passes the input through untouched.
pub fn editor<P: Plugin>(
	processor: &P,
	face: Faceplate,
	bypass: Arc<AtomicBool>,
	scope: Arc<Scope>,
	rate: usize,
) -> Option<Box<dyn Editor<P>>> {
	editor_with(
		processor,
		face,
		bypass,
		scope,
		rate,
		|_: &ProtocolRequest<'_>| None,
		|_: &str| false,
	)
}

/// The editor for a plug-in that also serves routes and commands of its own.
///
/// `extra` sees every request before the kit's routes do, and answers the ones
/// the product owns: an upload route, or the state of a resource the page draws.
/// Returning `None` passes the request on to the kit.
///
/// `command` sees every IPC string before the parameter parser does. A command
/// that needs the host -- a native file dialog, say -- returns true and is not
/// mistaken for `id=value`.
pub fn editor_with<P: Plugin>(
	processor: &P,
	face: Faceplate,
	bypass: Arc<AtomicBool>,
	scope: Arc<Scope>,
	rate: usize,
	extra: impl Fn(&ProtocolRequest<'_>) -> Option<ProtocolResponse> + Send + Sync + 'static,
	command: impl Fn(&str) -> bool + Send + Sync + 'static,
) -> Option<Box<dyn Editor<P>>> {
	if face::sections(&processor.get_parameters()).is_empty() {
		return None;
	}

	editor_with_page(
		processor,
		face,
		PageAssets::FACEPLATE,
		bypass,
		scope,
		rate,
		extra,
		command,
	)
}

/// The editor for a plug-in with a page of its own.
///
/// This is [editor_with] with the three embedded assets replaced. The routes the
/// kit serves -- `/face`, `/state`, `/scope`, `/stereo`, `/note`, `/meter` --
/// are still there, and `extra` still answers first, so a page that draws its own
/// state can add `/rack` and keep the scope if it wants one.
///
/// Unlike [editor_with], a page with no metadata controls is not refused: a rack
/// may take all of its controls from its own route.
// The argument list matches editor_with, plus the page: splitting it into a
// builder would make the one call site that matters longer, not shorter.
#[allow(clippy::too_many_arguments)]
pub fn editor_with_page<P: Plugin>(
	processor: &P,
	face: Faceplate,
	page: PageAssets,
	bypass: Arc<AtomicBool>,
	scope: Arc<Scope>,
	rate: usize,
	extra: impl Fn(&ProtocolRequest<'_>) -> Option<ProtocolResponse> + Send + Sync + 'static,
	command: impl Fn(&str) -> bool + Send + Sync + 'static,
) -> Option<Box<dyn Editor<P>>> {
	let sections = face::sections(&processor.get_parameters());
	let page_face = face::face_json_with_rate(&face, &sections, Some(rate), true);
	let map = processor.param_map();

	let state_sections = sections.clone();
	let state_map = map.clone();
	let state_bypass = bypass.clone();
	let command_sections = sections;
	let command_map = map;
	let command_bypass = bypass;

	Some(Box::new(
		WryEditor::<P>::new(page.html, move |body| {
			if command(body) {
				return None;
			}

			if !face::apply_bypass(&command_bypass, body) {
				face::apply_command(&command_map, &command_sections, body);
			}

			// The page writes the value into the shared map itself, which is what
			// the audio thread reads; there is no message to carry it.
			None
		})
		.with_custom_protocol_request("webgui", move |request| {
			if let Some(response) = extra(request) {
				return response;
			}

			match request.path {
				"/" | "/index.html" => ProtocolResponse::html(page.html),
				"/kit.css" => ProtocolResponse {
					status: 200,
					content_type: "text/css; charset=utf-8",
					body: page.css.as_bytes().to_vec(),
				},
				"/kit.js" => ProtocolResponse {
					status: 200,
					content_type: "text/javascript; charset=utf-8",
					body: page.js.as_bytes().to_vec(),
				},
				"/card.css" => ProtocolResponse {
					status: 200,
					content_type: "text/css; charset=utf-8",
					body: CARD_STYLE.as_bytes().to_vec(),
				},
				"/card.js" => ProtocolResponse {
					status: 200,
					content_type: "text/javascript; charset=utf-8",
					body: CARD_SCRIPT.as_bytes().to_vec(),
				},
				"/face" => ProtocolResponse::json(page_face.clone()),
				"/state" => ProtocolResponse::json(face::state_json_with_bypass(
					&state_map,
					&state_sections,
					state_bypass.load(Ordering::Relaxed),
				)),
				"/scope" => ProtocolResponse {
					status: 200,
					content_type: "application/octet-stream",
					body: scope.snapshot(),
				},
				"/stereo" => ProtocolResponse {
					status: 200,
					content_type: "application/octet-stream",
					body: scope.stereo_snapshot(),
				},
				"/note" => ProtocolResponse::json(format!("{{\"note\":{}}}", scope.note())),
				"/allowed" => {
					ProtocolResponse::json(format!("{{\"notes\":{}}}", scope.allowed_notes()))
				}
				"/meter" => {
					let (input, sidechain) = scope.meter();
					ProtocolResponse::json(format!(
						"{{\"input\":{input:.2},\"sidechain\":{sidechain:.2}}}"
					))
				}
				_ => ProtocolResponse::not_found(),
			}
		})
		.with_url("webgui://localhost/"),
	))
}

/// The controls a parameter list describes, flattened across sections.
///
/// A rack shows one card per module and gives each card the controls its effect
/// already declares; the conversion is the same one the faceplate page uses.
pub fn controls(parameters: &[i_am_dsp::prelude::Parameter]) -> Vec<Control> {
	face::sections(parameters)
		.into_iter()
		.flat_map(|section| section.controls)
		.collect()
}

/// The faceplate JSON for a plug-in, for a test or a preview.
pub fn face_json<P: Plugin>(processor: &P, face: &Faceplate) -> String {
	face::face_json(face, &face::sections(&processor.get_parameters()))
}

/// The state JSON for a plug-in, for a test or a preview.
pub fn state_json<P: Plugin>(processor: &P) -> String {
	let sections = face::sections(&processor.get_parameters());

	face::state_json(&processor.param_map(), &sections)
}

/// Applies one id=value command, for a test.
pub fn apply_command<P: Plugin>(processor: &P, body: &str) -> bool {
	let sections = face::sections(&processor.get_parameters());

	face::apply_command(&processor.param_map(), &sections, body)
}

/// Declares a plug-in with a web faceplate and nothing else.
///
/// The DSP, the parameter metadata and this one line are the whole product. The
/// window is resizable; the page reflows rather than scaling a picture of itself.
///
/// A parameter named by `mode = "id"` is drawn as a switch in the header instead of
/// a control in the body, which is how an effect with several algorithms changes
/// one.
///
/// An instrument is declared with the ports and tags it has rather than the effect
/// defaults: `ports = &[]` says there is no audio to process, `midi = &[...]` says
/// where the notes come in, and `tags = &[...]` puts it in the host's instrument
/// menu. The generator itself is played through
/// [`GeneratorEffect`](i_am_dsp::GeneratorEffect).
///
/// ```ignore
/// web_plugin!(
///     FilterWeb,
///     Multimode,
///     |rate| Multimode::new(rate, 1_000.0, 0.707),
///     "iamdsp.effects.filter.web",
///     "I Am Filter",
///     (520.0, 330.0),
///     "#4c8dff",
///     mode = "mode"
/// );
/// ```
#[macro_export]
macro_rules! web_plugin {
	($name:ident, $effect:ty, $build:expr, $id:literal, $title:literal, $size:expr, $accent:literal $(, mode = $mode:literal)? $(, visible = $visible:expr)? $(, header = $header:expr)? $(, switches = $switches:expr)? $(, ports = $ports:expr)? $(, midi = $midi:expr)? $(, tags = $tags:expr)? $(, telemetry = $telemetry:expr)? $(, clear = $clear:literal)?) => {
		#[doc = concat!("The ", $title, " plug-in, with a web faceplate.")]
		pub struct $name {
			processor: i_am_plugin::processor::EffectProcessor<$effect>,
			bypass: std::sync::Arc<std::sync::atomic::AtomicBool>,
			scope: std::sync::Arc<$crate::Scope>,
			clear: std::sync::Arc<std::sync::atomic::AtomicBool>,
		}

		impl Default for $name {
			fn default() -> Self {
				Self {
					processor: i_am_plugin::processor::EffectProcessor::new($build),
					bypass: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
					scope: std::sync::Arc::new($crate::Scope::new()),
					clear: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
				}
			}
		}

		impl $name {
			/// The rate the effect inside is currently built for.
			pub fn sample_rate(&self) -> usize {
				self.processor.sample_rate()
			}

			/// Whether the editor's bypass switch is on.
			pub fn is_bypassed(&self) -> bool {
				self.bypass.load(std::sync::atomic::Ordering::Relaxed)
			}

			/// Turn the bypass on or off from code, for a test or a host that
			/// grows its own bypass later.
			pub fn set_bypassed(&self, bypassed: bool) {
				self.bypass
					.store(bypassed, std::sync::atomic::Ordering::Relaxed);
			}
		}

		impl i_am_dsp::prelude::Parameters for $name {
			fn get_parameters(&self) -> Vec<i_am_dsp::prelude::Parameter> {
				self.processor.parameters()
			}

			fn set_parameter(
				&mut self,
				identifier: &str,
				value: i_am_dsp::prelude::SetValue,
			) -> bool {
				self.processor.set_parameter(identifier, value)
			}
		}

		impl i_am_dsp::Processor for $name {
			type Message = i_am_plugin::processor::EffectMessage;

			fn delay(&self) -> usize {
				i_am_dsp::Processor::delay(&self.processor)
			}

			fn process(
				&mut self,
				samples: &mut [f32; 2],
				other: &[[f32; 2]],
				process_context: &mut Box<dyn i_am_dsp::ProcessContext>,
			) {
				// The scope keeps the last 4096 frames at the host's rate: about
				// 85 ms at 48 kHz, long enough to read a waveform and full-rate
				// enough for the page to run an FFT over it.
				for event in process_context.events() {
					match event {
						i_am_dsp::NoteEvent::NoteOn { note, .. } => {
							self.scope.set_note(*note as i32)
						}
						i_am_dsp::NoteEvent::NoteOff { note, .. }
						| i_am_dsp::NoteEvent::Stop { note, .. } => {
							if self.scope.note() == *note as i32 {
								self.scope.set_note(-1);
							}
						}
						i_am_dsp::NoteEvent::ImmediateStop => self.scope.set_note(-1),
						_ => {}
					}
				}

				if self.clear.swap(false, std::sync::atomic::Ordering::Relaxed) {
					self.processor.clear_tail();
				}

				let input = *samples;
				let input_mono = (input[0] + input[1]) * 0.5;
				// A compressor with a sidechain port reads other[1]; without one the
				// input stands in for it, which is also what the DSP does.
				let sidechain = other
					.get(1)
					.map(|frame| (frame[0] + frame[1]) * 0.5)
					.unwrap_or(input_mono);

				// Bypassed is the input, untouched: an effect has nothing to add.
				if self.bypass.load(std::sync::atomic::Ordering::Relaxed) {
					self.scope.push_stereo(input, input, sidechain);
					return;
				}

				i_am_dsp::Processor::process(&mut self.processor, samples, other, process_context);

				$(
					($telemetry)(self.processor.effect(), &self.scope);
				)?

				let output = *samples;
				self.scope.push_stereo(input, output, sidechain);
			}

			fn on_message(&self, message: Self::Message) {
				i_am_dsp::Processor::on_message(&self.processor, message);
			}
		}

		impl i_am_plugin::Plugin for $name {
			const DESCRIPTOR: i_am_plugin::Descriptor = i_am_plugin::Descriptor::new($id, $title)
				.with_vendor("iamplugins")
				.with_version(env!("CARGO_PKG_VERSION"))
				$(.with_tags($tags))?;

			const HAS_GUI: bool = true;

			$(const INPUT_PORTS: &'static [i_am_plugin::AudioPort] = $ports;)?
			$(const INPUT_MIDI_PORTS: &'static [i_am_plugin::MidiPort] = $midi;)?

			fn new() -> Self {
				Self::default()
			}

			fn window_options() -> i_am_plugin::WindowOptions {
				let (width, height) = $size;

				// The page draws the faceplate at 85% of the window, so the window
				// asks for the design size over that: the first open has the same
				// faceplate as before, with a frame around it.
				const FACE_FRACTION: f64 = 0.85;

				i_am_plugin::WindowOptions::new()
					.with_size((
						width as f64 / FACE_FRACTION,
						height as f64 / FACE_FRACTION,
					))
					.with_title($title)
					// The page is a flex layout, not a picture of one: any size
					// it is given is a size it can lay itself out in.
					.with_resize_hints(i_am_plugin::GuiResizeHints {
						can_resize_horizontally: true,
						can_resize_vertically: true,
						strategy: i_am_plugin::AspectRatioStrategy::Disregard,
					})
			}

			fn param_map(&self) -> i_am_dsp::prelude::ParamMap {
				self.processor.param_map()
			}

			fn sync_params(&mut self) {
				self.processor.sync_params();
			}

			fn on_activate(&mut self, sample_rate: usize) {
				self.processor.activate(sample_rate);
			}

			fn editor(&self) -> Option<Box<dyn i_am_plugin::Editor<Self>>> {
				let face = $crate::Faceplate::new($title, $accent, $size) $(.with_mode($mode))? $(.with_visible($visible))? $(.with_header($header))? $(.with_switches($switches))?;
				$(let face = if $clear { face.with_clear() } else { face };)?

				let clear = self.clear.clone();

				$crate::editor_with(
					self,
					face,
					self.bypass.clone(),
					self.scope.clone(),
					self.processor.sample_rate(),
					move |request: &$crate::ProtocolRequest<'_>| {
						if request.method == "POST" && request.path == "/clear" {
							clear.store(true, std::sync::atomic::Ordering::Relaxed);
							return Some($crate::ProtocolResponse::json("{\"ok\":true}"));
						}

						None
					},
					|_: &str| false,
				)
			}
		}
	};
}

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
		for script in [CARD_SCRIPT, include_str!("page/kit.js")] {
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
		for kind in ["resonator", "formant", "autotune"] {
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
			include_str!("page/kit.js").contains("buildNotePicker"),
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

	/// The string instrument's metadata is what its faceplate will be: five
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

		assert_eq!(
			names,
			["Instrument", "Contact", "Body", "Performance", "Vibrato"]
		);
		assert_eq!(sections[0].hero.as_deref(), Some("spectrum"));

		let preset = parameters
			.iter()
			.find(|parameter| parameter.identifier == "preset")
			.expect("the preset parameter");
		assert_eq!(preset.meta.choices.expect("preset choices").len(), 4);
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
