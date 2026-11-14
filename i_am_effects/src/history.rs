//! The history time-remap product: the page draws the pages, the effect replays
//! the recent past through them.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use i_am_dsp::prelude::{ParamMap, Parameter, Parameters, SetValue, Value, from_binary, to_binary};
use i_am_dsp::{ProcessContext, Processor};
use i_am_plugin::processor::{EffectMessage, EffectProcessor};
use i_am_plugin::{Descriptor, Plugin, WindowOptions};
use i_am_web_gui::{Faceplate, ProtocolRequest, ProtocolResponse, Scope, Switch};

use i_am_dsp::prelude::{HistoryEffect, HistoryFeed, HistoryPage};

const SIZE: (f32, f32) = (760.0, 420.0);
const TITLE: &str = "I Am History";
const ACCENT: &str = "#6ee7ff";

/// A handle to one History node's curve pages, shared by the page and the effect.
#[derive(Clone)]
pub struct HistoryHandle(Arc<HistoryFeed>);

impl HistoryHandle {
	pub fn new() -> Self {
		Self(HistoryFeed::new())
	}

	/// The route's GET answer.
	pub fn state_json(&self) -> String {
		format!(
			"{{\"generation\":{},\"phase\":{},\"pages\":{}}}",
			self.0.generation(),
			self.0.phase(),
			serde_json::to_string(&*self.0.pages()).expect("the pages serialize")
		)
	}

	/// Applies the route's POST body.
	pub fn set_json(&self, body: &str) -> String {
		match serde_json::from_str::<Vec<HistoryPage>>(body) {
			Ok(pages) => {
				self.0.set_pages(pages);
				"{\"ok\":true}".to_string()
			}
			Err(error) => format!(
				"{{\"ok\":false,\"error\":\"{}\"}}",
				error.to_string().replace('"', "")
			),
		}
	}

	pub fn feed(&self) -> Arc<HistoryFeed> {
		self.0.clone()
	}
}

impl Default for HistoryHandle {
	fn default() -> Self {
		Self::new()
	}
}

fn history_route(
	request: &ProtocolRequest<'_>,
	handle: &HistoryHandle,
) -> Option<ProtocolResponse> {
	if request.path != "/history" {
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

/// The faceplate: one history length, a morph and a mix; the curve hero lives on Morph.
const SWITCHES: &[Switch] = &[Switch::new("time.sync", &["time.beats"], &["history_time"])];

/// The faceplate: one history length, a morph and a mix; the curve hero lives on Morph.
pub fn faceplate() -> Faceplate {
	Faceplate::new(TITLE, ACCENT, SIZE)
		.with_header(&["time.sync"])
		.with_switches(SWITCHES)
}

/// The web product.
pub struct HistoryWeb {
	processor: EffectProcessor<HistoryEffect>,
	bypass: Arc<AtomicBool>,
	scope: Arc<Scope>,
	handle: HistoryHandle,
}

impl Default for HistoryWeb {
	fn default() -> Self {
		Self::with_handle(HistoryHandle::new())
	}
}

impl HistoryWeb {
	pub fn with_handle(handle: HistoryHandle) -> Self {
		let build_feed = handle.feed();
		Self {
			processor: EffectProcessor::with_build(move |rate| {
				HistoryEffect::with_feed(rate, build_feed.clone())
			}),
			bypass: Arc::new(AtomicBool::new(false)),
			scope: Arc::new(Scope::new()),
			handle,
		}
	}

	pub fn handle(&self) -> HistoryHandle {
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

impl Parameters for HistoryWeb {
	fn get_parameters(&self) -> Vec<Parameter> {
		let mut parameters = self.processor.parameters();
		// The pages are not a parameter the page can write through ParamMap, but
		// the host saves them with the rest of the state: this is the archive copy.
		let pages = to_binary(&*self.handle.feed().pages()).expect("the pages serialize");
		parameters.push(Parameter::new("history_pages", Value::Serialized(pages)));

		parameters
	}

	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		if identifier == "history_pages"
			&& let SetValue::Serialized(bytes) = value
		{
			if let Ok(pages) = from_binary::<Vec<HistoryPage>>(bytes) {
				self.handle.feed().set_pages(pages);
				return true;
			}

			return false;
		}

		self.processor.set_parameter(identifier, value)
	}
}

impl Processor for HistoryWeb {
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

impl Plugin for HistoryWeb {
	const DESCRIPTOR: Descriptor = Descriptor::new("iamdsp.effects.history.web", TITLE)
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
		self.processor.sync_params();
	}

	fn on_activate(&mut self, sample_rate: usize) {
		self.processor.activate(sample_rate);
	}

	fn editor(&self) -> Option<Box<dyn i_am_plugin::Editor<Self>>> {
		let handle = self.handle.clone();
		i_am_web_gui::editor_with(
			self,
			faceplate(),
			self.bypass.clone(),
			self.scope.clone(),
			self.processor.sample_rate(),
			move |request: &ProtocolRequest<'_>| history_route(request, &handle),
			|_: &str| false,
		)
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use i_am_dsp::prelude::HistoryPoint;

	#[test]
	fn pages_round_trip_the_route() {
		let handle = HistoryHandle::new();
		let state = handle.state_json();
		assert!(state.contains("\"pages\""), "{state}");

		let pages: Vec<HistoryPage> = serde_json::from_str(
			state
				.split("\"pages\":")
				.nth(1)
				.expect("pages")
				.trim_end_matches('}'),
		)
		.expect("pages parse");
		assert!(!pages.is_empty());

		let result = handle.set_json(&serde_json::to_string(&pages).expect("pages"));
		assert!(result.contains("\"ok\":true"), "{result}");
	}

	#[test]
	fn pages_survive_the_state() {
		let handle = HistoryHandle::new();
		handle.feed().set_pages(vec![
			HistoryPage::flat(),
			HistoryPage {
				points: vec![
					HistoryPoint {
						time: 0.0,
						value: 0.0,
						bend: 0.0,
					},
					HistoryPoint {
						time: 1.0,
						value: 0.0,
						bend: 0.0,
					},
				],
			},
		]);

		let source = HistoryWeb::with_handle(handle);
		let value = source
			.get_parameters()
			.into_iter()
			.find(|parameter| parameter.identifier == "history_pages")
			.expect("the archived pages")
			.value
			.clone();
		let mut restored = HistoryWeb::new();

		assert!(restored.set_parameter(
			"history_pages",
			SetValue::Serialized(match value {
				Value::Serialized(bytes) => bytes,
				_ => panic!("the pages are not serialized"),
			})
		));
		assert_eq!(restored.handle().feed().pages().len(), 2);
	}
}
