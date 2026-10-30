//! The web product for the library's classic band vocoder.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use i_am_dsp::prelude::{ParamMap, Parameter, ProcessContext, SetValue, VocoderEffect, VocoderMeters};
use i_am_effects_core::processor::{EffectMessage, EffectProcessor};
use i_am_plugin::{
	AspectRatioStrategy, Descriptor, Editor, GuiResizeHints, Plugin, WindowOptions,
};
use i_am_web_gui::{ProtocolRequest, ProtocolResponse, Scope};

fn route(request: &ProtocolRequest<'_>, meters: &VocoderMeters) -> Option<ProtocolResponse> {
	if request.method != "GET" || request.path != "/bands" {
		return None;
	}

	let (bands, input, output) = meters.snapshot();
	let mut json = format!("{{\"bands\":{bands},\"input\":[");

	for (index, value) in input.iter().enumerate() {
		if index > 0 {
			json.push(',');
		}
		let _ = std::fmt::Write::write_fmt(&mut json, format_args!("{value:.4}"));
	}

	json.push_str("],\"output\":[");

	for (index, value) in output.iter().enumerate() {
		if index > 0 {
			json.push(',');
		}
		let _ = std::fmt::Write::write_fmt(&mut json, format_args!("{value:.4}"));
	}

	json.push_str("]}");
	Some(ProtocolResponse::json(json))
}

/// The product.
pub struct VocoderWeb {
	processor: EffectProcessor<VocoderEffect>,
	bypass: Arc<AtomicBool>,
	scope: Arc<Scope>,
	meters: Arc<VocoderMeters>,
}

impl Default for VocoderWeb {
	fn default() -> Self {
		let meters = Arc::new(VocoderMeters::default());
		let build_meters = meters.clone();

		Self {
			processor: EffectProcessor::with_build(move |rate| {
				VocoderEffect::new(rate, build_meters.clone())
			}),
			bypass: Arc::new(AtomicBool::new(false)),
			scope: Arc::new(Scope::new()),
			meters,
		}
	}
}

impl i_am_dsp::prelude::Parameters for VocoderWeb {
	fn get_parameters(&self) -> Vec<Parameter> {
		self.processor.parameters()
	}

	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		self.processor.set_parameter(identifier, value)
	}
}

impl i_am_dsp::Processor for VocoderWeb {
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

impl Plugin for VocoderWeb {
	const DESCRIPTOR: Descriptor =
		Descriptor::new("iamdsp.effects.vocoder.web", "I Am Vocoder")
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
			.with_size((700.0 / FACE_FRACTION, 380.0 / FACE_FRACTION))
			.with_title("I Am Vocoder")
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
		let face = i_am_web_gui::Faceplate::new("I Am Vocoder", "#e0a45a", (700.0, 380.0));
		let meters = self.meters.clone();

		i_am_web_gui::editor_with(
			self,
			face,
			self.bypass.clone(),
			self.scope.clone(),
			self.processor.sample_rate(),
			move |request: &ProtocolRequest<'_>| route(request, &meters),
			|_: &str| false,
		)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn the_band_route_reports_a_level_per_band() {
		let meters = Arc::new(VocoderMeters::default());
		let effect: VocoderEffect<2> = VocoderEffect::new(48_000, meters.clone());
		drop(effect);

		let request = ProtocolRequest {
			method: "GET",
			path: "/bands",
			query: None,
			body: &[],
		};
		let response = route(&request, &meters).expect("the bands route");
		let body = String::from_utf8(response.body).expect("utf-8");

		assert!(body.contains("\"bands\":24"), "{body}");
		assert!(body.contains("\"input\""), "{body}");
	}
}
