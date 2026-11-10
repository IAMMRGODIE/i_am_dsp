//! The two products: one effect, one generator, the same graph engine.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use i_am_dsp::ProcessContext;
use i_am_dsp::prelude::{ParamMap, Parameter, Processor, SetValue};
use i_am_plugin::processor::EffectMessage;
use i_am_plugin::{
	AspectRatioStrategy, AudioPort, Descriptor, Editor, GuiResizeHints, MidiPort, Plugin, Tag,
	WindowOptions,
};
use i_am_web_gui::{Faceplate, PageAssets};

use crate::shared::{Body, PatcherShared, UiState};
use crate::web;

/// The faceplate size, in points.
pub const SIZE: (f32, f32) = (960.0, 640.0);
const TITLE: &str = "I Am Patch";

const PAGE_HTML: &str = include_str!("../../i_am_patch/src/page/index.html");
const PAGE_CSS: &str = concat!(
	include_str!("../../i_am_web_gui/src/page/card.css"),
	"\n",
	include_str!("../../i_am_patch/src/page/patcher.css"),
);
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
				// The graph is the state; it rides in a serialized parameter so the
				// host's save and load carry it without a new extension.
				vec![Parameter::new(
					"patch",
					Value::Serialized(self.0.shared.text().into_bytes()),
				)]
			}

			fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
				match (identifier, value) {
					("patch", SetValue::Serialized(bytes)) => {
						let Ok(text) = String::from_utf8(bytes) else {
							return false;
						};
						self.0.shared.set_text(&text)
					}
					_ => false,
				}
			}
		}

		impl Processor for $name {
			type Message = EffectMessage;

			fn process(
				&mut self,
				samples: &mut [f32; 2],
				other: &[[f32; 2]],
				context: &mut Box<dyn ProcessContext>,
			) {
				for event in context.events() {
					self.0.body.push_event(event);
				}

				let infos = context.infos();
				self.0
					.body
					.set_transport(infos.tempo, infos.position_beats.map(|value| value as f32));

				let bypassed = self.0.ui.bypass.load(Ordering::Relaxed);
				let input = if $generator || bypassed {
					[0.0; 2]
				} else {
					*samples
				};
				let sidechain = other.get(1).copied().unwrap_or([0.0; 2]);
				let out = self.0.body.process(input, sidechain);

				// A bypassed graph keeps running on silence, so the notes it was
				// holding are released instead of sounding again on un-bypass.
				*samples = if bypassed { [0.0; 2] } else { out };
			}

			fn delay(&self) -> usize {
				self.0.shared.latency()
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
					.with_resize_hints(GuiResizeHints {
						can_resize_horizontally: true,
						can_resize_vertically: true,
						strategy: AspectRatioStrategy::Disregard,
					})
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

use i_am_dsp::prelude::{Parameters, Value};

product!(
	PatcherEffect,
	"iamdsp.patch.effect",
	"A patcher: an effect whose graph the page edits.",
	&[Tag::AudioEffect],
	&[AudioPort { name: "Main" }, AudioPort { name: "Sidechain" }],
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

#[cfg(test)]
mod tests {
	#[test]
	fn the_effect_declares_its_sidechain() {
		assert_eq!(PatcherEffect::INPUT_PORTS.len(), 2);
		assert_eq!(PatcherEffect::INPUT_PORTS[0].name, "Main");
		assert_eq!(PatcherEffect::INPUT_PORTS[1].name, "Sidechain");
		assert!(PatcherGenerator::INPUT_PORTS.is_empty());
	}

	#[test]
	fn the_host_sidechain_reaches_the_graph() {
		let shared = std::sync::Arc::new(PatcherShared::default());
		assert!(shared.set_text(
			"node 0 input:sidechain 0 0\nnode 1 out:audio 100 0\nconnect 0.audio -> 1.audio\n"
		));

		let mut body = Body::new(shared);
		let block = i_am_patch::block::BLOCK;

		for _ in 0..block {
			body.process([0.0; 2], [0.5, -0.25]);
		}

		let mut out = [0.0f32; 2];

		for _ in 0..block {
			out = body.process([0.0; 2], [0.5, -0.25]);
		}

		assert!((out[0] - 0.5).abs() < 1e-6, "{}", out[0]);
		assert!((out[1] + 0.25).abs() < 1e-6, "{}", out[1]);
	}

	use super::*;

	fn patch_text(effect: &PatcherEffect) -> String {
		let params = effect.get_parameters();
		let Value::Serialized(bytes) = &params[0].value else {
			panic!("the patch parameter is not serialized");
		};

		String::from_utf8(bytes.clone()).expect("utf-8")
	}

	#[test]
	fn the_host_is_told_the_latency() {
		let effect = PatcherEffect::default();

		assert_eq!(effect.delay(), i_am_patch::block::BLOCK);
	}

	#[test]
	fn the_host_state_carries_the_graph() {
		let mut effect = PatcherEffect::default();
		let text = patch_text(&effect);

		assert!(text.contains("synth:additive"), "{text}");
		assert!(text.contains("voice:out"), "{text}");

		let changed = text.replace("set 3 cutoff=1200", "set 3 cutoff=500");
		assert!(effect.set_parameter("patch", SetValue::Serialized(changed.into_bytes())));
		assert!(patch_text(&effect).contains("cutoff=500"));
	}

	#[test]
	fn a_broken_state_is_refused() {
		let mut effect = PatcherEffect::default();

		assert!(!effect.set_parameter("patch", SetValue::Serialized(b"node nope".to_vec())));
		assert!(patch_text(&effect).contains("synth:additive"));
	}
}
