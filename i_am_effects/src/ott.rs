//! The OTT web product: the shared OTT effect with its meter route and window.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use i_am_dsp::prelude::{
	OttEffect, OttMeters, ParamMap, Parameter, ProcessContext, SetValue,
};
use i_am_plugin::processor::{EffectMessage, EffectProcessor};
use i_am_plugin::{
	AspectRatioStrategy, Descriptor, Editor, GuiResizeHints, Plugin, WindowOptions,
};
use i_am_web_gui::{ProtocolRequest, ProtocolResponse, Scope};

fn route(request: &ProtocolRequest<'_>, meters: &OttMeters) -> Option<ProtocolResponse> {
	if request.method != "GET" || request.path != "/bands" {
		return None;
	}

	let bands = meters.get();
	Some(ProtocolResponse::json(format!(
		"{{\"bands\":[{:.2},{:.2},{:.2}]}}",
		bands[0], bands[1], bands[2]
	)))
}

/// The product.
pub struct OttWeb {
	processor: EffectProcessor<OttEffect>,
	bypass: Arc<AtomicBool>,
	scope: Arc<Scope>,
	meters: Arc<OttMeters>,
}

impl Default for OttWeb {
	fn default() -> Self {
		let meters = Arc::new(OttMeters::default());
		let build_meters = meters.clone();

		Self {
			processor: EffectProcessor::with_build(move |rate| {
				OttEffect::new(rate, build_meters.clone())
			}),
			bypass: Arc::new(AtomicBool::new(false)),
			scope: Arc::new(Scope::new()),
			meters,
		}
	}
}

impl i_am_dsp::prelude::Parameters for OttWeb {
	fn get_parameters(&self) -> Vec<Parameter> {
		self.processor.parameters()
	}

	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		self.processor.set_parameter(identifier, value)
	}
}

impl i_am_dsp::Processor for OttWeb {
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
			self.meters.set([0.0; 3]);
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

impl Plugin for OttWeb {
	const DESCRIPTOR: Descriptor =
		Descriptor::new("iamdsp.effects.ott.web", "I Am OTT")
			.with_vendor("iamplugins")
			.with_version(env!("CARGO_PKG_VERSION"));

	const HAS_GUI: bool = true;

	fn new() -> Self {
		Self::default()
	}

	fn window_options() -> WindowOptions {
		const FACE_FRACTION: f64 = 0.85;

		WindowOptions::new()
			.with_size((700.0 / FACE_FRACTION, 580.0 / FACE_FRACTION))
			.with_title("I Am OTT")
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
		let face = i_am_web_gui::Faceplate::new("I Am OTT", "#ffb454", (700.0, 580.0));
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
	use i_am_dsp::prelude::{OttCore, OttSettings};

	fn rms(samples: &[f32]) -> f32 {
		(samples.iter().map(|sample| sample * sample).sum::<f32>()
			/ samples.len() as f32)
			.sqrt()
	}

	#[test]
	fn depth_zero_is_dry() {
		let mut core = OttCore::new(44_100.0);
		core.set_settings(OttSettings {
			depth: 0.0,
			..Default::default()
		});

		for i in 0..1024 {
			let input = 0.4 * (2.0 * std::f32::consts::PI * 500.0 * i as f32 / 44_100.0).sin();
			let (left, right) = core.process(input, input);
			assert!((left - input).abs() < 1e-6, "{left} vs {input}");
			assert!((right - input).abs() < 1e-6);
		}
	}

	#[test]
	fn an_empty_threshold_and_zero_gains_pass_the_signal_flat() {
		let mut core = OttCore::new(44_100.0);
		core.set_settings(OttSettings {
			input_gain: 0.0,
			output_gain: 0.0,
			upward: 0.0,
			downward: 0.0,
			band_gain: [0.0; 3],
			..Default::default()
		});
		core.reset();

		let mut input = Vec::new();
		let mut output = Vec::new();
		let mut state = 0xdead_beef_u32;

		for _ in 0..65_536 {
			state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
			let x = (state >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0;
			input.push(x);
			let (left, _) = core.process(x, x);
			output.push(left);
		}

		let ratio = rms(&output) / rms(&input);
		assert!(ratio > 0.99 && ratio < 1.01, "{ratio}");
	}

	#[test]
	fn upward_compression_boosts_a_quiet_signal() {
		let mut core = OttCore::new(44_100.0);
		core.set_settings(OttSettings {
			input_gain: 0.0,
			output_gain: 0.0,
			downward: 0.0,
			band_gain: [0.0; 3],
			..Default::default()
		});
		core.reset();

		let amplitude = 10f32.powf(-40.0 / 20.0);
		let mut input = Vec::new();
		let mut output = Vec::new();

		for i in 0..16_384 {
			let x = amplitude * (2.0 * std::f32::consts::PI * 220.0 * i as f32 / 44_100.0).sin();
			input.push(x);
			let (left, _) = core.process(x, x);
			output.push(left);
		}

		let start = input.len() / 4;
		let gain = rms(&output[start..]) / rms(&input[start..]);
		assert!(gain > 1.5, "{gain}");
	}

	#[test]
	fn downward_compression_reduces_a_loud_signal() {
		let mut core = OttCore::new(44_100.0);
		core.set_settings(OttSettings {
			input_gain: 0.0,
			output_gain: 0.0,
			upward: 0.0,
			band_gain: [0.0; 3],
			..Default::default()
		});
		core.reset();

		let mut input = Vec::new();
		let mut output = Vec::new();

		for i in 0..16_384 {
			let x = 0.9 * (2.0 * std::f32::consts::PI * 220.0 * i as f32 / 44_100.0).sin();
			input.push(x);
			let (left, _) = core.process(x, x);
			output.push(left);
		}

		let start = input.len() / 4;
		let gain = rms(&output[start..]) / rms(&input[start..]);
		assert!(gain < 0.95, "{gain}");
	}

	#[test]
	fn the_stereo_detector_is_linked() {
		let mut core = OttCore::new(44_100.0);
		core.reset();

		let mut in_left = Vec::new();
		let mut in_right = Vec::new();
		let mut out_left = Vec::new();
		let mut out_right = Vec::new();

		for i in 0..8192 {
			let t = i as f32 / 44_100.0;
			let left = 0.6 * (2.0 * std::f32::consts::PI * 440.0 * t).sin();
			let right = 0.2 * (2.0 * std::f32::consts::PI * 440.0 * t).sin();
			in_left.push(left);
			in_right.push(right);
			let (ol, or) = core.process(left, right);
			out_left.push(ol);
			out_right.push(or);
		}

		let start = in_left.len() / 4;
		let gain_left = rms(&out_left[start..]) / rms(&in_left[start..]);
		let gain_right = rms(&out_right[start..]) / rms(&in_right[start..]);
		let relative = (gain_left - gain_right).abs() / (0.5 * (gain_left + gain_right));
		assert!(relative < 1e-3, "{relative}");
	}

	#[test]
	fn the_band_meter_route_reports_three_bands() {
		let meters = OttMeters::default();
		meters.set([-3.0, 1.5, 10.0]);

		let request = ProtocolRequest {
			method: "GET",
			path: "/bands",
			query: None,
			body: &[],
		};
		let response = route(&request, &meters).expect("the bands route");
		let body = String::from_utf8(response.body).expect("utf-8");

		assert!(body.contains("\"bands\""), "{body}");
		assert!(body.contains("-3.00"), "{body}");
		assert!(body.contains("1.50"), "{body}");
	}

	#[test]
	fn the_meters_stay_finite() {
		let mut core = OttCore::new(44_100.0);
		core.reset();

		for i in 0..4096 {
			let x = (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 44_100.0).sin() * 0.5;
			core.process(x, x);
		}

		for meter in core.meters() {
			assert!(meter.is_finite(), "{meter}");
		}
	}
}
