//! The web product for the library's STFT vocoder.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use i_am_dsp::prelude::{ParamMap, Parameter, ProcessContext, SetValue, StftVocoderEffect};
use i_am_effects_core::processor::{EffectMessage, EffectProcessor};
use i_am_plugin::{
	AspectRatioStrategy, Descriptor, Editor, GuiResizeHints, Plugin, WindowOptions,
};
use i_am_web_gui::Scope;

/// The product.
pub struct StftVocoderWeb {
	processor: EffectProcessor<StftVocoderEffect>,
	bypass: Arc<AtomicBool>,
	scope: Arc<Scope>,
}

impl Default for StftVocoderWeb {
	fn default() -> Self {
		Self {
			processor: EffectProcessor::new(StftVocoderEffect::new),
			bypass: Arc::new(AtomicBool::new(false)),
			scope: Arc::new(Scope::new()),
		}
	}
}

impl i_am_dsp::prelude::Parameters for StftVocoderWeb {
	fn get_parameters(&self) -> Vec<Parameter> {
		self.processor.parameters()
	}

	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		self.processor.set_parameter(identifier, value)
	}
}

impl i_am_dsp::Processor for StftVocoderWeb {
	type Message = EffectMessage;

	fn delay(&self) -> usize {
		i_am_dsp::Processor::delay(&self.processor)
	}

	fn process(
		&mut self,
		samples: &mut [f32; 2],
		other: &[[f32; 2]],
		process_context: &mut Box<dyn ProcessContext>,
	) {
		let input = *samples;
		let input_mono = (input[0] + input[1]) * 0.5;
		let sidechain = other
			.get(1)
			.map(|frame| (frame[0] + frame[1]) * 0.5)
			.unwrap_or(input_mono);

		if self.bypass.load(Ordering::Relaxed) {
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

impl Plugin for StftVocoderWeb {
	const DESCRIPTOR: Descriptor =
		Descriptor::new("iamdsp.effects.stft_vocoder.web", "I Am STFT Vocoder")
			.with_vendor("iamplugins")
			.with_version(env!("CARGO_PKG_VERSION"));

	const HAS_GUI: bool = true;

	const INPUT_PORTS: &'static [i_am_plugin::AudioPort] = &[
		i_am_plugin::AudioPort { name: "Main" },
		i_am_plugin::AudioPort { name: "Sidechain" },
	];

	fn new() -> Self {
		Self::default()
	}

	fn window_options() -> WindowOptions {
		const FACE_FRACTION: f64 = 0.85;

		WindowOptions::new()
			.with_size((700.0 / FACE_FRACTION, 340.0 / FACE_FRACTION))
			.with_title("I Am STFT Vocoder")
			.with_resize_hints(GuiResizeHints {
				can_resize_horizontally: true,
				can_resize_vertically: true,
				strategy: AspectRatioStrategy::Disregard,
			})
	}

	fn param_map(&self) -> ParamMap {
		self.processor.param_map()
	}

	fn sync_params(&mut self) {
		self.processor.sync_params();
	}

	fn on_activate(&mut self, sample_rate: usize) {
		self.processor.activate(sample_rate);
	}

	fn editor(&self) -> Option<Box<dyn Editor<Self>>> {
		let face = i_am_web_gui::Faceplate::new("I Am STFT Vocoder", "#7bb8ff", (700.0, 340.0));

		i_am_web_gui::editor(
			self,
			face,
			self.bypass.clone(),
			self.scope.clone(),
			self.processor.sample_rate(),
		)
	}
}
