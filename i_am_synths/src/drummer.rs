//! The first drum voice: a kick, on note 36.
//!
//! The DSP lives in `i_am_dsp::prelude::DrumInstrument`; this module is the
//! window, the note port and the shared instrument body, exactly like the other
//! synth products.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use i_am_dsp::prelude::{
	DrumInstrument, GeneratorEffect, Parameter, Parameters, SetValue,
};
use i_am_dsp::{ProcessContext, Processor};
use i_am_plugin::processor::{EffectMessage, EffectProcessor};
use i_am_plugin::{
	AspectRatioStrategy, Descriptor, GuiResizeHints, MidiPort, Plugin, Tag, WindowOptions,
};
use i_am_web_gui::{Faceplate, Page, ProtocolRequest, Scope, editor_with};

use super::play_synth;

const SIZE: (f32, f32) = (620.0, 320.0);
const TITLE: &str = "I Am Drummer";
const ACCENT: &str = "#c8793d";

const PAGES: &[Page] = &[Page::new("Kick", &["Kick"])];

/// The faceplate of the first drum.
pub fn faceplate() -> Faceplate {
	Faceplate::new(TITLE, ACCENT, SIZE).with_pages(PAGES)
}

/// A kick drum, with a web faceplate.
pub struct DrummerWeb {
	processor: EffectProcessor<GeneratorEffect<DrumInstrument>>,
	bypass: Arc<AtomicBool>,
	scope: Arc<Scope>,
}

impl Default for DrummerWeb {
	fn default() -> Self {
		Self {
			processor: EffectProcessor::new(|rate: usize| {
				GeneratorEffect::new(DrumInstrument::new(rate))
			}),
			bypass: Arc::new(AtomicBool::new(false)),
			scope: Arc::new(Scope::new()),
		}
	}
}

impl Parameters for DrummerWeb {
	fn get_parameters(&self) -> Vec<Parameter> {
		self.processor.parameters()
	}

	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		self.processor.set_parameter(identifier, value)
	}
}

impl Processor for DrummerWeb {
	type Message = EffectMessage;

	fn delay(&self) -> usize {
		Processor::delay(&self.processor)
	}

	fn process(
		&mut self,
		samples: &mut [f32; 2],
		other: &[[f32; 2]],
		process_context: &mut Box<dyn ProcessContext>,
	) {
		play_synth(
			&mut self.processor,
			&self.scope,
			&self.bypass,
			samples,
			other,
			process_context,
		);
	}

	fn on_message(&self, message: Self::Message) {
		Processor::on_message(&self.processor, message);
	}
}

impl Plugin for DrummerWeb {
	const DESCRIPTOR: Descriptor = Descriptor::new("iamdsp.synths.drummer.web", TITLE)
		.with_vendor("iamplugins")
		.with_version(env!("CARGO_PKG_VERSION"))
		.with_description(
			"A drum kit synthesised from partials, clicks and noise. M0: the kick.",
		)
		.with_tags(&[
			Tag::Instrument,
			Tag::Synthesizer,
			Tag::Drum,
			Tag::DrumMachine,
		]);

	const INPUT_PORTS: &'static [i_am_plugin::AudioPort] = &[];
	const INPUT_MIDI_PORTS: &'static [MidiPort] = &[MidiPort { name: "Notes" }];
	const HAS_GUI: bool = true;

	fn new() -> Self {
		Self::default()
	}

	fn window_options() -> WindowOptions {
		const FACE_FRACTION: f64 = 0.85;

		WindowOptions::new()
			.with_size((SIZE.0 as f64 / FACE_FRACTION, SIZE.1 as f64 / FACE_FRACTION))
			.with_title(TITLE)
			.with_resize_hints(GuiResizeHints {
				can_resize_horizontally: true,
				can_resize_vertically: true,
				strategy: AspectRatioStrategy::Disregard,
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
		let rate = self.processor.sample_rate();

		editor_with(
			self,
			faceplate(),
			self.bypass.clone(),
			self.scope.clone(),
			rate,
			|_: &ProtocolRequest<'_>| None,
			|_: &str| false,
		)
	}
}
