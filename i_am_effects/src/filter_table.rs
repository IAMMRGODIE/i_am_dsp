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
			"{{\"generation\":{},\"pages\":{}}}",
			self.0.generation(),
			serde_json::to_string(&self.0.pages()).expect("the pages serialize")
		)
	}

	/// Applies the route's POST body.
	pub fn set_json(&self, body: &str) -> String {
		match serde_json::from_str::<Vec<Vec<FilterPoint>>>(body) {
			Ok(pages) => {
				self.0.set_pages(pages);
				"{\"ok\":true}".to_string()
			}
			Err(_) => match serde_json::from_str::<Vec<FilterPoint>>(body) {
				Ok(points) => {
					self.0.set_points(points);
					"{\"ok\":true}".to_string()
				}
				Err(error) => format!(
					"{{\"ok\":false,\"error\":\"{}\"}}",
					error.to_string().replace('"', "")
				),
			},
		}
	}

	pub fn frame(&self) -> Arc<FilterFrame> {
		self.0.clone()
	}

	/// Decodes a WAV into the selected page. `window` is in source samples.
	///
	/// `mode` is `time` for the waveform itself or `freq` to take only its
	/// magnitude and turn that into a zero-phase kernel.
	pub fn import_wav(&self, bytes: &[u8], window: usize, page: usize, mode: &str) -> String {
		let Ok(sample) = i_am_dsp::prelude::decode_click(bytes) else {
			return "{\"error\":\"decode\"}".to_string();
		};

		let window = window.max(1).min(sample.pcm.len());
		if window == 0 {
			return "{\"error\":\"empty\"}".to_string();
		}

		let signal = if mode == "freq" {
			i_am_dsp::prelude::magnitude_kernel(&sample.pcm[..window], 1024)
		} else {
			sample.pcm[..window].to_vec()
		};

		self.0.set_page(page, points_from_signal(&signal, 256));
		self.state_json()
	}

	/// Applies a Shapes-style page edit: reverse, invert, dc or normalize.
	pub fn edit_page(&self, action: &str, page: usize) -> String {
		let mut pages = self.0.pages();
		if pages.is_empty() {
			return self.state_json();
		}

		let index = page.min(pages.len() - 1);
		let points = &mut pages[index];

		sort_points(points);

		match action {
			"reverse" => {
				for point in points.iter_mut() {
					point.time = 1.0 - point.time;
				}
				points.sort_by(|a, b| {
					a.time
						.partial_cmp(&b.time)
						.unwrap_or(core::cmp::Ordering::Equal)
				});
			}
			"invert" => {
				for point in points.iter_mut() {
					point.value = 1.0 - point.value;
				}
			}
			"dc" => {
				let mean =
					points.iter().map(|point| point.value).sum::<f32>() / points.len() as f32;
				for point in points.iter_mut() {
					point.value = (point.value - mean).clamp(0.0, 1.0);
				}
			}
			"normalize" => {
				let peak = points
					.iter()
					.map(|point| (point.value - 0.5).abs())
					.fold(0.0f32, f32::max);
				if peak > 1e-6 {
					for point in points.iter_mut() {
						point.value = (0.5f32 + (point.value - 0.5) / peak).clamp(0.0, 1.0);
					}
				}
			}
			_ => {}
		}

		let pages = pages;
		self.0.set_pages(pages);
		self.state_json()
	}
}

impl Default for FilterTableHandle {
	fn default() -> Self {
		Self::new()
	}
}

fn sort_points(points: &mut [FilterPoint]) {
	points.sort_by(|a, b| {
		a.time
			.partial_cmp(&b.time)
			.unwrap_or(core::cmp::Ordering::Equal)
	});
}

fn points_from_signal(signal: &[f32], count: usize) -> Vec<FilterPoint> {
	let mut points = Vec::with_capacity(count);
	let last = signal.len().saturating_sub(1).max(1);

	for index in 0..count {
		let at = index as f32 / count.saturating_sub(1).max(1) as f32 * last as f32;
		let left = at.floor() as usize;
		let right = (left + 1).min(signal.len().saturating_sub(1));
		let value = signal.get(left).copied().unwrap_or(0.0)
			+ (signal.get(right).copied().unwrap_or(0.0)
				- signal.get(left).copied().unwrap_or(0.0))
				* (at - left as f32);

		points.push(FilterPoint {
			time: index as f32 / count.saturating_sub(1).max(1) as f32,
			value: (value * 0.5 + 0.5).clamp(0.0, 1.0),
			bend: 0.0,
		});
	}

	points
}

fn filter_route(
	request: &ProtocolRequest<'_>,
	handle: &FilterTableHandle,
) -> Option<ProtocolResponse> {
	if request.path == "/filter-table/import" {
		if request.method != "POST" {
			return Some(ProtocolResponse::json("{\"error\":\"method\"}"));
		}

		let value = |key: &str| {
			request
				.query
				.unwrap_or("")
				.split('&')
				.find_map(|pair| pair.strip_prefix(&format!("{key}=")))
				.and_then(|value| value.parse::<usize>().ok())
		};
		let text = |key: &str| {
			request
				.query
				.unwrap_or("")
				.split('&')
				.find_map(|pair| pair.strip_prefix(&format!("{key}=")))
		};
		let window = value("window").unwrap_or(2048);
		let page = value("page").unwrap_or(0);
		let mode = text("mode").unwrap_or("time");

		return Some(ProtocolResponse::json(handle.import_wav(
			request.body,
			window,
			page,
			mode,
		)));
	}

	if request.path == "/filter-table/edit" {
		if request.method != "POST" {
			return Some(ProtocolResponse::json("{\"error\":\"method\"}"));
		}

		let value = |key: &str| {
			request
				.query
				.unwrap_or("")
				.split('&')
				.find_map(|pair| pair.strip_prefix(&format!("{key}=")))
		};
		let action = value("action").unwrap_or("");
		let page = value("page")
			.and_then(|value| value.parse::<usize>().ok())
			.unwrap_or(0);

		return Some(ProtocolResponse::json(handle.edit_page(action, page)));
	}

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
		let pages = to_binary(&self.handle.frame().pages()).expect("the pages serialize");
		parameters.push(Parameter::new("filter_pages", Value::Serialized(pages)));
		parameters
	}

	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		if identifier == "filter_pages"
			&& let SetValue::Serialized(bytes) = value
		{
			if let Ok(pages) = from_binary::<Vec<Vec<FilterPoint>>>(bytes) {
				self.handle.frame().set_pages(pages);
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
		handle.frame().set_pages(vec![
			vec![
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
			],
			vec![
				FilterPoint {
					time: 0.0,
					value: 0.75,
					bend: 0.0,
				},
				FilterPoint {
					time: 1.0,
					value: 0.25,
					bend: 0.0,
				},
			],
		]);

		let source = FilterTableWeb::with_handle(handle);
		let value = source
			.get_parameters()
			.into_iter()
			.find(|parameter| parameter.identifier == "filter_pages")
			.expect("the archived points")
			.value
			.clone();
		let mut restored = FilterTableWeb::new();
		let Value::Serialized(bytes) = value else {
			panic!("the points are not serialized");
		};

		assert!(restored.set_parameter("filter_pages", SetValue::Serialized(bytes)));
		assert_eq!(restored.handle().frame().pages().len(), 2);
	}

	#[test]
	fn an_imported_wav_fills_the_page() {
		fn wav(samples: &[i16]) -> Vec<u8> {
			let data_len = (samples.len() * 2) as u32;
			let mut bytes = Vec::new();
			bytes.extend_from_slice(b"RIFF");
			bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
			bytes.extend_from_slice(b"WAVEfmt ");
			bytes.extend_from_slice(&16u32.to_le_bytes());
			bytes.extend_from_slice(&1u16.to_le_bytes());
			bytes.extend_from_slice(&1u16.to_le_bytes());
			bytes.extend_from_slice(&48_000u32.to_le_bytes());
			bytes.extend_from_slice(&96_000u32.to_le_bytes());
			bytes.extend_from_slice(&2u16.to_le_bytes());
			bytes.extend_from_slice(&16u16.to_le_bytes());
			bytes.extend_from_slice(b"data");
			bytes.extend_from_slice(&data_len.to_le_bytes());
			for sample in samples {
				bytes.extend_from_slice(&sample.to_le_bytes());
			}
			bytes
		}

		let handle = FilterTableHandle::new();
		let answer = handle.import_wav(&wav(&[0, 16_000, -16_000, 8_000]), 4, 0, "time");

		assert!(answer.contains("\"pages\""), "{answer}");
		assert_eq!(handle.frame().pages()[0].len(), 256);
	}

	#[test]
	fn a_page_edit_changes_the_page() {
		let handle = FilterTableHandle::new();
		let before = handle.frame().pages()[0][0].value;
		handle.edit_page("invert", 0);
		let after = handle.frame().pages()[0][0].value;

		assert!((before + after - 1.0).abs() < 1e-6);
	}
}
