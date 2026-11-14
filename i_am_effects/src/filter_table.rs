//! The Filter Table product: a wavetable frame used as a convolution kernel.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use i_am_dsp::prelude::{
	FilterFrame, FilterPoint, FilterTableEffect, ParamMap, Parameter, Parameters, SetValue, Value,
	from_binary, to_binary,
};
use i_am_dsp::{ProcessContext, Processor};
use i_am_plugin::processor::{EffectMessage, EffectProcessor};
use i_am_plugin::{Descriptor, Plugin, WindowOptions};
use i_am_web_gui::{Faceplate, ProtocolRequest, ProtocolResponse, Scope};

const SIZE: (f32, f32) = (760.0, 420.0);
const TITLE: &str = "I Am Filter Table";
const ACCENT: &str = "#c8e04a";

/// A handle to one Filter Table frame, shared by the page and the effect.
#[derive(Clone)]
pub struct FilterTableHandle(Arc<FilterFrame>);

impl FilterTableHandle {
	pub fn new() -> Self {
		Self(FilterFrame::new())
	}

	/// The route's GET answer.
	pub fn state_json(&self) -> String {
		format!(
			"{{\"generation\":{},\"points\":{}}}",
			self.0.generation(),
			serde_json::to_string(&*self.0.points()).expect("the points serialize")
		)
	}

	/// Applies the route's POST body.
	pub fn set_json(&self, body: &str) -> String {
		match serde_json::from_str::<Vec<FilterPoint>>(body) {
			Ok(points) => {
				self.0.set_points(points);
				"{\"ok\":true}".to_string()
			}
			Err(error) => format!(
				"{{\"ok\":false,\"error\":\"{}\"}}",
				error.to_string().replace('"', "")
			),
		}
	}

	pub fn frame(&self) -> Arc<FilterFrame> {
		self.0.clone()
	}
}

impl Default for FilterTableHandle {
	fn default() -> Self {
		Self::new()
	}
}

fn filter_route(
	request: &ProtocolRequest<'_>,
	handle: &FilterTableHandle,
) -> Option<ProtocolResponse> {
	if request.path != "/filter-table" {
		return None;
	}

	match request.method {
		"GET" => Some(ProtocolResponse::json(handle.state_json())),
		"POST" => Some(ProtocolResponse::json(
			handle.set_json(&String::from_utf8_lossy(request.body)),
		)),
		_ => Some(ProtocolResponse::json("{\"error\":\"method\"}")),
	}
}

/// The product.
pub struct FilterTableWeb {
	processor: EffectProcessor<FilterTableEffect>,
	bypass: Arc<AtomicBool>,
	scope: Arc<Scope>,
	handle: FilterTableHandle,
}

impl Default for FilterTableWeb {
	fn default() -> Self {
		Self::with_handle(FilterTableHandle::new())
	}
}

impl FilterTableWeb {
	pub fn with_handle(handle: FilterTableHandle) -> Self {
		let frame = handle.frame();
		Self {
			processor: EffectProcessor::with_build(move |rate| {
				FilterTableEffect::with_frame(rate, frame.clone())
			}),
			bypass: Arc::new(AtomicBool::new(false)),
			scope: Arc::new(Scope::new()),
			handle,
		}
	}

	pub fn handle(&self) -> FilterTableHandle {
		self.handle.clone()
	}

	/// The faceplate size this product declares to a host.
	pub const fn size() -> (f32, f32) {
		SIZE
	}

	pub fn sample_rate(&self) -> usize {
		self.processor.sample_rate()
	}
}

impl Parameters for FilterTableWeb {
	fn get_parameters(&self) -> Vec<Parameter> {
		let mut parameters = self.processor.parameters();
		let points = to_binary(&*self.handle.frame().points()).expect("the points serialize");
		parameters.push(Parameter::new("filter_points", Value::Serialized(points)));
		parameters
	}

	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		if identifier == "filter_points"
			&& let SetValue::Serialized(bytes) = value
		{
			if let Ok(points) = from_binary::<Vec<FilterPoint>>(bytes) {
				self.handle.frame().set_points(points);
				return true;
			}

			return false;
		}

		self.processor.set_parameter(identifier, value)
	}
}

impl Processor for FilterTableWeb {
	type Message = EffectMessage;

	fn delay(&self) -> usize {
		i_am_dsp::Processor::delay(&self.processor)
	}

	fn process(
		&mut self,
		samples: &mut [f32; 2],
		other: &[[f32; 2]],
		context: &mut Box<dyn ProcessContext>,
	) {
		let input = *samples;
		if self.bypass.load(Ordering::Relaxed) {
			self.scope.push_stereo(input, input, 0.0);
			return;
		}

		i_am_dsp::Processor::process(&mut self.processor, samples, other, context);
		self.scope.push_stereo(input, *samples, 0.0);
	}

	fn on_message(&self, message: Self::Message) {
		i_am_dsp::Processor::on_message(&self.processor, message);
	}
}

/// The faceplate: the cutoff, the resonance and the mix; the frame hero is on Cutoff.
pub fn faceplate() -> Faceplate {
	Faceplate::new(TITLE, ACCENT, SIZE)
}

impl Plugin for FilterTableWeb {
	const DESCRIPTOR: Descriptor = Descriptor::new("iamdsp.effects.filter_table.web", TITLE)
		.with_vendor("iamplugins")
		.with_version(env!("CARGO_PKG_VERSION"));

	const HAS_GUI: bool = true;

	fn new() -> Self {
		Self::default()
	}

	fn window_options() -> WindowOptions {
		i_am_web_gui::window_options(SIZE, TITLE)
	}

	fn param_map(&self) -> ParamMap {
		self.processor.param_map()
	}

	fn sync_params(&mut self) {
		self.processor.sync_params()
	}

	fn on_activate(&mut self, sample_rate: usize) {
		self.processor.activate(sample_rate)
	}

	fn editor(&self) -> Option<Box<dyn i_am_plugin::Editor<Self>>> {
		let handle = self.handle.clone();
		i_am_web_gui::editor_with(
			self,
			faceplate(),
			self.bypass.clone(),
			self.scope.clone(),
			self.processor.sample_rate(),
			move |request: &ProtocolRequest<'_>| filter_route(request, &handle),
			|_: &str| false,
		)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn points_survive_the_state() {
		let handle = FilterTableHandle::new();
		handle.frame().set_points(vec![
			FilterPoint {
				time: 0.0,
				value: 0.25,
				bend: 0.0,
			},
			FilterPoint {
				time: 1.0,
				value: 0.75,
				bend: 0.0,
			},
		]);

		let source = FilterTableWeb::with_handle(handle);
		let value = source
			.get_parameters()
			.into_iter()
			.find(|parameter| parameter.identifier == "filter_points")
			.expect("the archived points")
			.value
			.clone();
		let mut restored = FilterTableWeb::new();
		let Value::Serialized(bytes) = value else {
			panic!("the points are not serialized");
		};

		assert!(restored.set_parameter("filter_points", SetValue::Serialized(bytes)));
		assert_eq!(restored.handle().frame().points().len(), 2);
	}
}
