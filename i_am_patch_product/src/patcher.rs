//! The two products: one effect, one generator, the same graph engine.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use i_am_dsp::prelude::{Parameter, ParamMap, Processor, SetValue};
use i_am_dsp::ProcessContext;
use i_am_plugin::processor::EffectMessage;
use i_am_plugin::{
	AudioPort, Descriptor, Editor, MidiPort, Plugin, Tag, WindowOptions,
};
use i_am_web_gui::{Faceplate, PageAssets};

use crate::shared::{Body, PatcherShared, UiState};
use crate::web;

/// The faceplate size, in points.
pub const SIZE: (f32, f32) = (960.0, 640.0);
const TITLE: &str = "I Am Patch";

const PAGE_HTML: &str = include_str!("../../i_am_patch/src/page/index.html");
const PAGE_CSS: &str = include_str!("../../i_am_patch/src/page/patcher.css");
const PAGE_JS: &str = include_str!("../../i_am_patch/src/page/patcher.js");

/// What the two products share: the graph body, the shared state and the UI.
pub struct Patcher {
	body: Body,
	shared: Arc<PatcherShared>,
	ui: UiState,
}

impl Patcher {
	fn new() -> Self {
		let shared = Arc::new(PatcherShared::default());

		Self {
			body: Body::new(shared.clone()),
			shared,
			ui: UiState::default(),
		}
	}
}

/// Implements the four traits a product needs, with the boundary differences
/// passed in: the effect takes audio, the generator does not.
macro_rules! product {
	($name:ident, $id:literal, $description:literal, $tags:expr, $ports:expr, $generator:expr) => {
		/// See the module docs.
		pub struct $name(Patcher);

		impl Default for $name {
			fn default() -> Self {
				Self(Patcher::new())
			}
		}

		impl Parameters for $name {
			fn get_parameters(&self) -> Vec<Parameter> {
				Vec::new()
			}

			fn set_parameter(&mut self, _: &str, _: SetValue) -> bool {
				false
			}
		}

		impl Processor for $name {
			type Message = EffectMessage;

			fn process(
				&mut self,
				samples: &mut [f32; 2],
				_other: &[[f32; 2]],
				context: &mut Box<dyn ProcessContext>,
			) {
				if self.0.ui.bypass.load(Ordering::Relaxed) {
					*samples = [0.0; 2];
					return;
				}

				for event in context.events() {
					self.0.body.push_event(event);
				}

				let input = if $generator { [0.0; 2] } else { *samples };
				*samples = self.0.body.process(input);
			}

			fn delay(&self) -> usize {
				i_am_patch::block::BLOCK
			}

			fn on_message(&self, _: Self::Message) {}
		}

		impl Plugin for $name {
			const DESCRIPTOR: Descriptor = Descriptor::new($id, "I Am Patch")
				.with_vendor("iamplugins")
				.with_version(env!("CARGO_PKG_VERSION"))
				.with_description($description)
				.with_tags($tags);

			const INPUT_PORTS: &'static [AudioPort] = $ports;
			const INPUT_MIDI_PORTS: &'static [MidiPort] = &[MidiPort { name: "Notes" }];
			const HAS_GUI: bool = true;

			fn new() -> Self {
				Self::default()
			}

			fn param_map(&self) -> ParamMap {
				ParamMap::default()
			}

			fn window_options() -> WindowOptions {
				WindowOptions::new()
					.with_size((SIZE.0 as f64 / 0.92, SIZE.1 as f64 / 0.92))
					.with_title(TITLE)
			}

			fn on_activate(&mut self, sample_rate: usize) {
				self.0.shared.rate.store(sample_rate, Ordering::Relaxed);
				self.0.shared.bump();
			}

			fn editor(&self) -> Option<Box<dyn Editor<Self>>> {
				let shared = self.0.shared.clone();

				i_am_web_gui::editor_with_page(
					self,
					Faceplate::new(TITLE, "#e07a2f", SIZE),
					PageAssets {
						html: PAGE_HTML,
						css: PAGE_CSS,
						js: PAGE_JS,
					},
					self.0.ui.bypass.clone(),
					self.0.ui.scope.clone(),
					48_000,
					move |request| web::route(request, &shared),
					|_: &str| false,
				)
			}
		}
	};
}

use i_am_dsp::prelude::Parameters;

product!(
	PatcherEffect,
	"iamdsp.patch.effect",
	"A patcher: an effect whose graph the page edits.",
	&[Tag::AudioEffect],
	&[AudioPort { name: "Input" }],
	false
);

product!(
	PatcherGenerator,
	"iamdsp.patch.generator",
	"A patcher: a generator whose graph the page edits.",
	&[Tag::Instrument, Tag::Synthesizer],
	&[],
	true
);
