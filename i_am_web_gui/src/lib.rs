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
pub use scope::Scope;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use i_am_plugin::{Editor, Plugin};
pub use i_am_plugin::wry::{ProtocolRequest, ProtocolResponse};
use i_am_plugin::wry::WryEditor;

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
			"/meter" => {
				let (input, sidechain) = scope.meter();
				ProtocolResponse::json(format!(
					"{{\"input\":{input:.2},\"sidechain\":{sidechain:.2}}}"
				))
			},
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
	($name:ident, $effect:ty, $build:expr, $id:literal, $title:literal, $size:expr, $accent:literal $(, mode = $mode:literal)? $(, visible = $visible:expr)? $(, curves = $curves:expr)? $(, header = $header:expr)? $(, switches = $switches:expr)? $(, ports = $ports:expr)? $(, midi = $midi:expr)? $(, clear = $clear:literal)?) => {
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
				.with_version(env!("CARGO_PKG_VERSION"));

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
				let face = $crate::Faceplate::new($title, $accent, $size) $(.with_mode($mode))? $(.with_visible($visible))? $(.with_curves($curves))? $(.with_header($header))? $(.with_switches($switches))?;
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

	/// A mode is a choice: two options and seven options use the same control.
	#[test]
	fn a_mode_is_the_shared_select() {
		let start = SCRIPT.find("function buildMode").expect("buildMode");
		let end = SCRIPT[start..].find("function buildFace").expect("buildFace") + start;
		let mode = &SCRIPT[start..end];

		assert!(mode.contains("IAmCard.choiceHandle"));
		assert!(!mode.contains("mode-choice"));
	}
}
