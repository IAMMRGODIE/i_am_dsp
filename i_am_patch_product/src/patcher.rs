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
	fn new(generator: bool) -> Self {
		let graph = if generator {
			crate::shared::demo()
		} else {
			crate::shared::demo_effect()
		};
		let shared = Arc::new(PatcherShared::with_graph(graph));

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
				Self(Patcher::new($generator))
			}
		}

		impl Parameters for $name {
			fn get_parameters(&self) -> Vec<Parameter> {
				// The macros first: a host that restores a project writes them back
				// through `set_parameter`, and a host that only automates them keeps
				// its own lanes. Then the graph, which rides in a serialized parameter
				// so the host's save and load carry it without a new extension.
				let mut parameters = Vec::with_capacity(i_am_patch::MACRO_COUNT + 1);

				for index in 0..i_am_patch::MACRO_COUNT {
					let (Some(identifier), Some(atomic)) = (
						self.0.shared.params.query_param_id(index),
						self.0.shared.params.get_by_index(index),
					) else {
						continue;
					};

					let value = match atomic.load(Ordering::Relaxed) {
						SetValue::Float(value) => value,
						_ => 0.0,
					};

					parameters.push(Parameter::new(
						identifier,
						Value::Float {
							value,
							range: 0.0..=1.0,
							logarithmic: false,
						},
					));
				}

				parameters.push(Parameter::new(
					"patch",
					Value::Serialized(self.0.shared.text().into_bytes()),
				));

				// The payloads the nodes play live beside the text, in their own
				// serialized parameter: a table or a sample is not a number the text
				// format can carry.
				parameters.push(Parameter::new(
					"patch_slots",
					Value::Serialized(self.0.shared.slot_state_bytes()),
				));

				parameters
			}

			fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
				match (identifier, value) {
					(name, value) if name.starts_with("macro") => {
						self.0.shared.set_macro(name, value)
					}
					("patch", SetValue::Serialized(bytes)) => {
						let Ok(text) = String::from_utf8(bytes) else {
							return false;
						};
						self.0.shared.set_text(&text)
					}
					("patch_slots", SetValue::Serialized(bytes)) => {
						self.0.shared.set_slot_state(&bytes)
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
				// The host's macro values reach the nodes here, once per block: the
				// sixty-four reads are the block's business, and doing them per sample
				// was most of what an idle patcher cost.
				if context.sample_offset() == 0 {
					self.0.shared.pull_macros();
				}

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
				// A clear request from the page runs before the block it belongs to.
				if let Ok(mut clears) = self.0.shared.clear_requests.try_lock() {
					for node in clears.drain(..) {
						self.0.body.clear_tail(node);
					}
				}

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
				self.0.shared.params.clone()
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

	/// A save that changes nothing copies the last blob; a payload that moves makes a
	/// new one. The stamp is what tells them apart, and it must see a new sample.
	#[test]
	fn the_slot_blob_is_reused_until_a_slot_moves() {
		use std::sync::Arc;

		use i_am_dsp::prelude::SampleData;

		let shared = PatcherShared::default();
		let first = shared.slot_state_bytes();
		let second = shared.slot_state_bytes();

		assert_eq!(
			first, second,
			"the blob was encoded again with nothing changed"
		);

		{
			let slot = shared.slots.sample(4);
			let mut slot = slot.lock().expect("the sample slot");

			slot.sample = Some(Arc::new(SampleData {
				name: "kick.wav".to_string(),
				rate: 48_000,
				pcm: vec![0.0, 0.5, -0.5],
				right: None,
			}));
		}

		let third = shared.slot_state_bytes();

		assert_ne!(first, third, "a new sample did not reach the blob");
	}

	/// A table the page dropped on a Shape node rides in the blob beside the text, so a
	/// project keeps the sound the node was saved with.
	#[test]
	fn the_slot_blob_survives_a_save_and_load() {
		use i_am_dsp::prelude::{Parameters, SetValue, Value, from_binary, to_binary};
		use i_am_patch::state::SlotState;

		let text = "node 1 table:shapes 100 0\n";
		let mut state = SlotState::default();
		state.waves.insert(
			1,
			i_am_dsp::prelude::ImportedWave::new("note.wav", vec![vec![0.0, 0.5, 0.0, -0.5]]),
		);

		let mut product = PatcherEffect::default();
		assert!(product.set_parameter("patch", SetValue::Serialized(text.as_bytes().to_vec())));
		assert!(product.set_parameter(
			"patch_slots",
			SetValue::Serialized(to_binary(&state).expect("the state serializes")),
		));

		let saved = product
			.get_parameters()
			.into_iter()
			.find(|parameter| parameter.identifier == "patch_slots")
			.expect("the slots parameter");
		let Value::Serialized(bytes) = saved.value else {
			panic!("the slots are not serialized");
		};
		let back: SlotState = from_binary(bytes).expect("the state decodes");

		assert_eq!(
			back.waves.get(&1).map(|wave| wave.name.as_str()),
			Some("note.wav")
		);
	}

	#[test]
	fn the_product_path_matches_the_engine() {
		use i_am_dsp::Processor;
		use i_am_dsp::prelude::*;

		struct Context {
			info: ProcessInfos,
			events: Vec<NoteEvent>,
		}

		impl ProcessContext for Context {
			fn infos(&self) -> &ProcessInfos {
				&self.info
			}

			fn next_event(&mut self) -> Option<NoteEvent> {
				self.events.pop()
			}

			fn send_event(&mut self, event: NoteEvent) {
				self.events.push(event);
			}

			fn clear_events(&mut self) {
				self.events.clear();
			}

			fn events(&self) -> &[NoteEvent] {
				&self.events
			}
		}

		let text = "node 0 input:midi 0 0\nnode 1 synth:additive 100 0 random_phase=0 random_pan=0\nnode 2 out:audio 200 0\nconnect 0.midi -> 1.notes\nconnect 1.audio -> 2.audio\n";
		let note = NoteEvent::NoteOn {
			channel: 0,
			note: 60,
			velocity: 0.8,
		};

		let mut product = PatcherEffect::default();
		assert!(product.set_parameter("patch", SetValue::Serialized(text.as_bytes().to_vec())));

		let mut info = ProcessInfos::default();
		info.sample_rate = 48_000;
		info.playing = true;
		let mut context: Box<dyn ProcessContext> = Box::new(Context {
			info,
			events: vec![note.clone()],
		});
		let mut product_peak = 0.0f32;
		let mut samples = [0.0f32; 2];

		for sample in 0..4096 {
			product.process(&mut samples, &[], &mut context);
			product_peak = product_peak.max(samples[0].abs());

			if sample == 0 {
				context.clear_events();
			}
		}

		let graph = i_am_patch::text::from_text(text).expect("parses");
		let frames = i_am_patch::engine::Engine::render_graph(
			&graph,
			48_000,
			4096.0 / 48_000.0,
			&[(0, note)],
		)
		.expect("renders");
		let engine_peak = frames
			.iter()
			.fold(0.0f32, |peak, frame| peak.max(frame[0].abs()));

		assert!(
			(product_peak - engine_peak).abs() < 0.2 * engine_peak,
			"product {product_peak} vs engine {engine_peak}"
		);
	}

	/// The two products are not the same instrument: an effect is inserted on a track and
	/// has to pass its audio, an instrument plays the notes the host sends.
	#[test]
	fn each_product_opens_on_a_graph_that_shows_what_it_is_for() {
		let effect = crate::shared::demo_effect();
		let effect_kinds: Vec<&str> = effect.nodes.iter().map(|node| node.kind.id).collect();

		assert!(
			effect_kinds.contains(&"input:audio"),
			"an effect's default has no audio input: {effect_kinds:?}"
		);
		assert!(
			!effect_kinds.contains(&"input:midi"),
			"an effect's default listens for notes: {effect_kinds:?}"
		);
		assert!(effect_kinds.contains(&"out:audio"));

		let instrument = crate::shared::demo();
		let instrument_kinds: Vec<&str> =
			instrument.nodes.iter().map(|node| node.kind.id).collect();

		assert!(
			instrument_kinds.contains(&"input:midi"),
			"an instrument's default takes no notes: {instrument_kinds:?}"
		);
		assert!(
			instrument_kinds.contains(&"voice:in"),
			"an instrument's default has no voice: {instrument_kinds:?}"
		);
	}

	/// A patcher dropped on a track must not be silent before anything is edited.
	#[test]
	fn the_effect_default_passes_audio_through() {
		let shared = std::sync::Arc::new(PatcherShared::with_graph(crate::shared::demo_effect()));
		let mut body = Body::new(shared);
		let block = i_am_patch::block::BLOCK;
		let mut peak = 0.0f32;

		// The body holds one block of latency, so the first block comes back empty.
		for _ in 0..4 * block {
			let out = body.process([0.25, 0.25], [0.0; 2]);
			peak = peak.max(out[0].abs());
		}

		assert!(peak > 0.05, "the default effect patch was silent: {peak}");
	}

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

	fn patch_text(product: &impl Parameters) -> String {
		// The macros come first in the table, so the graph is looked up by name:
		// the parameter order is not part of what a test should pin.
		let params = product.get_parameters();
		let patch = params
			.iter()
			.find(|param| param.identifier == "patch")
			.expect("the patch parameter");

		let Value::Serialized(bytes) = &patch.value else {
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
		let mut generator = PatcherGenerator::default();
		let text = patch_text(&generator);

		assert!(text.contains("synth:additive"), "{text}");
		assert!(text.contains("voice:out"), "{text}");

		let changed = text.replace("set 3 cutoff=1200", "set 3 cutoff=500");
		assert!(generator.set_parameter("patch", SetValue::Serialized(changed.into_bytes())));
		assert!(patch_text(&generator).contains("cutoff=500"));
	}

	#[test]
	fn a_broken_state_is_refused() {
		let mut generator = PatcherGenerator::default();

		assert!(!generator.set_parameter("patch", SetValue::Serialized(b"node nope".to_vec())));
		assert!(patch_text(&generator).contains("synth:additive"));
	}
}
