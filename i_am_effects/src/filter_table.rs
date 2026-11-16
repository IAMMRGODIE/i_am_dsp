//! The Filter Table product: a wavetable frame used as a convolution kernel.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

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

#[derive(Default)]
struct ImportProgress {
	active: bool,
	done: usize,
	total: usize,
	stage: String,
	error: Option<String>,
}

/// A handle to one Filter Table frame, shared by the page and the effect.
#[derive(Clone)]
pub struct FilterTableHandle(Arc<FilterFrame>, Arc<Mutex<ImportProgress>>);

impl FilterTableHandle {
	pub fn new() -> Self {
		Self(
			FilterFrame::new(),
			Arc::new(Mutex::new(ImportProgress::default())),
		)
	}

	fn progress_json(&self) -> String {
		let progress = self.1.lock().unwrap_or_else(|error| error.into_inner());
		format!(
			"{{\"active\":{},\"done\":{},\"total\":{},\"stage\":\"{}\",\"error\":{}}}",
			progress.active,
			progress.done,
			progress.total,
			progress.stage.replace('"', ""),
			progress
				.error
				.as_ref()
				.map(|error| format!("\"{}\"", error.replace('"', "")))
				.unwrap_or_else(|| "null".to_string())
		)
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

	/// Starts a WAV import on a worker thread and returns its progress.
	pub fn start_import(
		&self,
		bytes: Vec<u8>,
		window: usize,
		page: usize,
		mode: String,
		slice: String,
	) -> String {
		{
			let mut progress = self.1.lock().unwrap_or_else(|error| error.into_inner());
			if progress.active {
				return self.progress_json();
			}
			progress.active = true;
			progress.done = 0;
			progress.total = 0;
			progress.stage = "decode".to_string();
			progress.error = None;
		}

		let frame = self.0.clone();
		let progress = self.1.clone();

		std::thread::spawn(move || {
			let result = import_pages(&frame, &bytes, window, page, &mode, &slice, Some(&progress));

			let mut progress = progress.lock().unwrap_or_else(|error| error.into_inner());
			progress.active = false;

			match result {
				Ok(()) => {
					progress.stage = "done".to_string();
					progress.error = None;
				}
				Err(error) => {
					progress.stage = "error".to_string();
					progress.error = Some(error);
				}
			}
		});

		self.progress_json()
	}

	pub fn progress(&self) -> String {
		self.progress_json()
	}

	/// Decodes a WAV into the table synchronously.
	///
	/// window is in source samples and also the per-frame length; the file is
	/// cut into as many frames as it fills. mode is time or freq, and slice is
	/// auto, fixed, pitch or single.
	pub fn import_wav(
		&self,
		bytes: &[u8],
		window: usize,
		page: usize,
		mode: &str,
		slice: &str,
	) -> String {
		match import_pages(&self.0, bytes, window, page, mode, slice, None) {
			Ok(()) => self.state_json(),
			Err(error) => format!("{{\"error\":\"{}\"}}", error.replace('"', "")),
		}
	}
	/// Applies a Shapes-style page edit: reverse, invert or dc.
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

fn import_pages(
	frame: &Arc<FilterFrame>,
	bytes: &[u8],
	window: usize,
	page: usize,
	mode: &str,
	slice: &str,
	progress: Option<&Arc<Mutex<ImportProgress>>>,
) -> Result<(), String> {
	let sample = i_am_dsp::prelude::decode_click(bytes).map_err(|error| error.to_string())?;
	let window = window.max(1);

	if sample.pcm.is_empty() {
		return Err("the file holds no samples".to_string());
	}

	let ranges = slice_ranges(&sample.pcm, window, slice);
	if ranges.is_empty() {
		return Err("the file holds no frames".to_string());
	}

	const MAX_FRAMES: usize = 256;
	let step = if ranges.len() > MAX_FRAMES {
		ranges.len().div_ceil(MAX_FRAMES)
	} else {
		1
	};
	let total = ranges.iter().step_by(step).count();

	if let Some(progress) = progress {
		let mut progress = progress.lock().unwrap_or_else(|error| error.into_inner());
		progress.stage = "render".to_string();
		progress.total = total;
	}

	let mut pages = Vec::with_capacity(total);

	for (index, (start, end)) in ranges.iter().step_by(step).enumerate() {
		let frame = &sample.pcm[*start..*end];
		let points = if mode == "freq" {
			points_from_signal(&i_am_dsp::prelude::magnitude_kernel(frame, 1024), 256)
		} else {
			points_from_signal(frame, 256)
		};

		pages.push(points);

		if let Some(progress) = progress {
			let mut progress = progress.lock().unwrap_or_else(|error| error.into_inner());
			progress.done = index + 1;
		}
	}

	if pages.is_empty() {
		return Err("the file holds no frames".to_string());
	}

	if pages.len() == 1 {
		frame.set_page(page, pages.remove(0));
	} else {
		frame.set_pages(pages);
	}

	Ok(())
}

/// The byte ranges of the frames a sample cuts into. Auto follows the Shapes
/// rules: a short file is one cycle, a whole number of windows is fixed, and
/// anything else that has a stable pitch is cut at its cycle.
fn slice_ranges(samples: &[f32], window: usize, slice: &str) -> Vec<(usize, usize)> {
	let window = window.max(1);
	let cycle = if slice == "auto" || slice == "pitch" {
		detect_cycle(samples)
	} else {
		None
	};

	let step = match slice {
		"single" => samples.len(),
		"pitch" => cycle.unwrap_or(window),
		"fixed" => window,
		_ => {
			if samples.len() < window + window / 2 {
				samples.len()
			} else if samples.len().is_multiple_of(window) {
				window
			} else {
				cycle.unwrap_or(window)
			}
		}
	};

	if step == 0 {
		return Vec::new();
	}

	(0..samples.len())
		.step_by(step)
		.map(|start| (start, (start + step).min(samples.len())))
		.collect()
}

/// The first strong period in a sample, in samples.
fn detect_cycle(samples: &[f32]) -> Option<usize> {
	let max_lag = (samples.len() / 2).min(1024);

	if max_lag < 16 {
		return None;
	}

	let span = samples.len().min(2048);
	let mut best = 0.0f32;
	let mut best_lag = 0;

	for lag in 16..=max_lag {
		let mut dot = 0.0f32;
		let mut left = 0.0f32;
		let mut right = 0.0f32;

		for index in 0..span.saturating_sub(lag) {
			dot += samples[index] * samples[index + lag];
			left += samples[index] * samples[index];
			right += samples[index + lag] * samples[index + lag];
		}

		let score = dot / (left.sqrt() * right.sqrt() + 1e-9);

		if score > best {
			best = score;
			best_lag = lag;
		}
	}

	(best > 0.5 && best_lag > 0).then_some(best_lag)
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
	scope: &Scope,
) -> Option<ProtocolResponse> {
	if request.path == "/filter-table/sidechain" {
		// The page shows the sidechain the effect heard, and nothing else: the
		// scope's third trace is exactly that.
		let snapshot = scope.snapshot();
		let third = snapshot.len() / 3;

		return Some(ProtocolResponse {
			status: 200,
			content_type: "application/octet-stream",
			body: snapshot[2 * third..].to_vec(),
		});
	}

	if request.path == "/filter-table/import/progress" {
		return Some(ProtocolResponse::json(handle.progress()));
	}

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
		let slice = text("slice").unwrap_or("auto");

		return Some(ProtocolResponse::json(handle.start_import(
			request.body.to_vec(),
			window,
			page,
			mode.to_string(),
			slice.to_string(),
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
		let sidechain = other
			.get(1)
			.map(|frame| (frame[0] + frame[1]) * 0.5)
			.unwrap_or(0.0);

		if self.bypass.load(Ordering::Relaxed) {
			self.scope.push_stereo(input, input, sidechain);
			return;
		}

		i_am_dsp::Processor::process(&mut self.processor, samples, other, context);
		self.scope.push_stereo(input, *samples, sidechain);
	}

	fn on_message(&self, message: Self::Message) {
		i_am_dsp::Processor::on_message(&self.processor, message);
	}
}

/// The sidechain mode hides the frame controls: a kernel from the sidechain has
/// no page to edit.
const FILTER_TABLE_VISIBLE: &[(i32, &[&str])] = &[
	(0, &["cutoff", "resonance", "mix", "phase", "morph"]),
	// The sidechain supplies the shape, so the frame's cutoff, resonance and
	// phase would be knobs that do nothing.
	(1, &["mix"]),
];

/// The faceplate: the cutoff, the resonance and the mix; the frame hero is on Cutoff.
pub fn faceplate() -> Faceplate {
	Faceplate::new(TITLE, ACCENT, SIZE)
		.with_mode("mode")
		.with_visible(FILTER_TABLE_VISIBLE)
}

impl Plugin for FilterTableWeb {
	const DESCRIPTOR: Descriptor = Descriptor::new("iamdsp.effects.filter_table.web", TITLE)
		.with_vendor("iamplugins")
		.with_version(env!("CARGO_PKG_VERSION"));

	const HAS_GUI: bool = true;

	/// The main input, and the sidechain the kernel can come from.
	const INPUT_PORTS: &'static [i_am_plugin::AudioPort] = crate::SIDECHAIN_PORTS;

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
		let scope = self.scope.clone();
		i_am_web_gui::editor_with(
			self,
			faceplate(),
			self.bypass.clone(),
			self.scope.clone(),
			self.processor.sample_rate(),
			move |request: &ProtocolRequest<'_>| filter_route(request, &handle, &scope),
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
		let answer = handle.import_wav(&wav(&[0, 16_000, -16_000, 8_000]), 4, 0, "time", "fixed");

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

	#[test]
	fn an_import_can_make_several_frames() {
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
		let samples = [0i16, 8_000, 16_000, 8_000, -8_000, -16_000, -8_000, 0];
		let answer = handle.import_wav(&wav(&samples), 4, 0, "time", "fixed");

		assert!(answer.contains("\"pages\""), "{answer}");
		assert_eq!(handle.frame().pages().len(), 2);
		assert_eq!(handle.frame().pages()[0].len(), 256);
	}

	#[test]
	fn the_import_reports_progress() {
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
		let samples: Vec<i16> = (0..4096).map(|index| (index % 64) as i16 * 256).collect();
		let first = handle.start_import(
			wav(&samples),
			256,
			0,
			"time".to_string(),
			"fixed".to_string(),
		);
		assert!(
			serde_json::from_str::<serde_json::Value>(&first).is_ok(),
			"{first}"
		);

		for _ in 0..100 {
			let progress = handle.progress();
			let value: serde_json::Value = serde_json::from_str(&progress).expect("progress json");

			if value["active"] == false {
				assert!(value["error"].is_null(), "{progress}");
				assert_eq!(handle.frame().pages().len(), 16);
				return;
			}

			std::thread::sleep(std::time::Duration::from_millis(10));
		}

		panic!("the import did not finish");
	}

	/// The product asks the host for a sidechain input alongside the main one.
	#[test]
	fn the_product_declares_its_sidechain() {
		use i_am_plugin::Plugin;

		assert_eq!(FilterTableWeb::INPUT_PORTS.len(), 2);
		assert_eq!(FilterTableWeb::INPUT_PORTS[1].name, "Sidechain");
	}

	/// The panel's sidechain route reports the signal the effect heard.
	#[test]
	fn the_sidechain_route_reports_the_input() {
		let mut product = FilterTableWeb::default();
		let mut context: Box<dyn ProcessContext> = Box::new(());
		// Port 0 is the main input; the sidechain is the second frame.
		let sidechain = [[0.0f32; 2], [0.25f32; 2]];
		let mut samples = [0.0f32; 2];

		for _ in 0..64 {
			i_am_dsp::Processor::process(&mut product, &mut samples, &sidechain, &mut context);
		}

		let request = ProtocolRequest {
			method: "GET",
			path: "/filter-table/sidechain",
			query: None,
			body: &[],
		};
		let response =
			filter_route(&request, &product.handle(), &product.scope).expect("the sidechain route");
		let values = response
			.body
			.as_chunks::<4>()
			.0
			.iter()
			.map(|bytes| f32::from_le_bytes(*bytes))
			.collect::<Vec<_>>();

		assert!(
			values.iter().any(|value| (*value - 0.25).abs() < 1e-3),
			"the sidechain never reached the scope"
		);
	}
}
