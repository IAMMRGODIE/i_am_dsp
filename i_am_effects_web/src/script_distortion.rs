//! The scripted distortion: a user-written transfer function, rendered into a
//! table so the audio thread never runs Rhai.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use i_am_dsp::prelude::{
	Effect, ParamMap, Parameter, ProcessContext, SetValue,
};
use i_am_dsp_derive::Parameters;
use i_am_effects_core::processor::{EffectMessage, EffectProcessor};
use i_am_plugin::{
	AspectRatioStrategy, Descriptor, Editor, GuiResizeHints, Plugin, WindowOptions,
};
use i_am_web_gui::{ProtocolRequest, ProtocolResponse, Scope};

use crate::script::{CompiledScript, ScriptWorker, json_escape};

/// The number of points in the transfer table.
const TABLE_SIZE: usize = 4096;

/// The script a new instance starts with.
pub const DEFAULT_SCRIPT: &str = "tanh(x * (1.0 + p1 * 8.0) + p2)";

/// A transfer curve sampled on `x ∈ [-1, 1]`.
pub struct DistortionTable {
	values: Vec<f32>,
}

impl DistortionTable {
	/// The shaped output for an input sample.
	pub fn sample(&self, x: f32) -> f32 {
		let position = (x.clamp(-1.0, 1.0) + 1.0) * 0.5 * (TABLE_SIZE - 1) as f32;
		let index = position as usize;
		let fraction = position - index as f32;
		let low = self.values[index];
		let high = self.values[(index + 1).min(TABLE_SIZE - 1)];

		low + (high - low) * fraction
	}

	/// Renders the script over the whole input range.
	fn render(
		script: &CompiledScript,
		params: [f32; 4],
		in_gain: f32,
	) -> Result<Self, String> {
		let mut scope = CompiledScript::scope(params);
		let mut values = Vec::with_capacity(TABLE_SIZE);

		for index in 0..TABLE_SIZE {
			let x = -1.0 + 2.0 * index as f32 / (TABLE_SIZE - 1) as f32;
			scope.set_value("x", x * in_gain);

			let y = script.number(&mut scope)?;

			if !y.is_finite() {
				return Err("the script returned a non-finite value".to_string());
			}

			values.push(y.clamp(-8.0, 8.0));
		}

		Ok(Self { values })
	}

	/// The points the page draws.
	fn curve_points(&self) -> Vec<[f32; 3]> {
		const POINTS: usize = 256;

		(0..POINTS)
			.map(|index| {
				let x = -1.0 + 2.0 * index as f32 / (POINTS - 1) as f32;
				[x, self.sample(x), 0.0]
			})
			.collect()
	}
}

/// The product's DSP.
#[derive(Parameters)]
pub struct ScriptDistortionEffect {
	/// The gain before the shaping curve.
	#[range(min = -24.0, max = 24.0)]
	#[unit(db)]
	#[default(0.0)]
	#[id(display = "In")]
	#[layout(group = "Script Distortion", order = 0, widget = knob)]
	pub in_gain: f32,

	/// The gain after the shaping curve.
	#[range(min = -24.0, max = 24.0)]
	#[unit(db)]
	#[default(0.0)]
	#[id(display = "Out")]
	#[layout(group = "Script Distortion", order = 1, widget = knob)]
	pub out_gain: f32,

	/// How much of the shaped signal is heard.
	#[range(min = 0.0, max = 1.0)]
	#[unit(percent)]
	#[default(1.0)]
	#[id(display = "Mix")]
	#[layout(group = "Script Distortion", order = 2, widget = knob)]
	pub mix: f32,

	/// The first script control.
	#[range(min = -1.0, max = 1.0)]
	#[default(0.0)]
	#[id(display = "P1")]
	#[layout(group = "Script Distortion", order = 3, widget = knob, hero = "curve")]
	pub p1: f32,

	/// The second script control.
	#[range(min = -1.0, max = 1.0)]
	#[default(0.0)]
	#[id(display = "P2")]
	#[layout(group = "Script Distortion", order = 4, widget = knob)]
	pub p2: f32,

	/// The third script control.
	#[range(min = -1.0, max = 1.0)]
	#[default(0.0)]
	#[id(display = "P3")]
	#[layout(group = "Script Distortion", order = 5, widget = knob)]
	pub p3: f32,

	/// The fourth script control.
	#[range(min = -1.0, max = 1.0)]
	#[default(0.0)]
	#[id(display = "P4")]
	#[layout(group = "Script Distortion", order = 6, widget = knob)]
	pub p4: f32,

	/// Run the script on the audio thread instead of the table.
	///
	/// The table is the safe default; this is for sound design, where the exact
	/// curve matters more than the cost. A heavy script can starve the audio
	/// thread, which is why it is a switch and not the default.
	#[default(false)]
	#[id(display = "Realtime")]
	#[layout(group = "Script Distortion", order = 7, widget = toggle)]
	pub realtime: bool,

	/// The Rhai body, saved with the plug-in state.
	#[serde]
	#[id(display = "Script")]
	pub script: String,

	#[skip]
	worker: Arc<ScriptWorker<DistortionTable>>,
	#[skip]
	table: Option<Arc<DistortionTable>>,
	#[skip]
	seen_table: u64,
	#[skip]
	synced_source: u64,
	#[skip]
	last_script: String,
	#[skip]
	last_request: Option<(u64, [f32; 4], f32)>,
	#[skip]
	realtime_script: Option<Arc<CompiledScript>>,
	#[skip]
	seen_compiled: u64,
	#[skip]
	realtime_scope: rhai::Scope<'static>,
}

impl ScriptDistortionEffect {
	/// A distortion with the default script, sharing `worker` with its editor.
	pub fn new(worker: Arc<ScriptWorker<DistortionTable>>) -> Self {
		Self {
			in_gain: 0.0,
			out_gain: 0.0,
			mix: 1.0,
			p1: 0.0,
			p2: 0.0,
			p3: 0.0,
			p4: 0.0,
			realtime: false,
			script: DEFAULT_SCRIPT.to_string(),
			worker,
			table: None,
			seen_table: 0,
			synced_source: 0,
			last_script: DEFAULT_SCRIPT.to_string(),
			last_request: None,
			realtime_script: None,
			seen_compiled: 0,
			realtime_scope: CompiledScript::scope([0.0; 4]),
		}
	}

	/// Keeps the saved script and the worker's text in step.
	fn sync_script(&mut self) {
		let generation = self.worker.source_generation();

		if generation != self.synced_source {
			self.synced_source = generation;
			let source = self.worker.source();
			self.script = source.clone();
			self.last_script = source;
			self.realtime_script = None;
		} else if self.script != self.last_script {
			// A state load replaced the field; push it to the worker and render.
			let generation = self.worker.set_source(self.script.clone());
			self.synced_source = generation;
			self.last_script = self.script.clone();
			self.realtime_script = None;
		}
	}

	/// Asks for a table when the script or an input to it changed.
	fn request_render(&mut self) {
		let generation = self.worker.source_generation();
		let params = [self.p1, self.p2, self.p3, self.p4];
		let in_gain = 10.0_f32.powf(self.in_gain / 20.0);
		let request = (generation, params, in_gain);

		if self.last_request != Some(request) && self.worker.request(params, in_gain) {
			self.last_request = Some(request);
		}
	}

	/// Takes a freshly rendered table.
	fn sync_table(&mut self) {
		let generation = self.worker.table_generation();

		if generation != self.seen_table {
			self.seen_table = generation;
			self.table = self.worker.table();
		}
	}

	/// Takes a freshly compiled script for the realtime path.
	fn sync_compiled(&mut self) {
		let generation = self.worker.compiled_generation();

		if generation != self.seen_compiled {
			self.seen_compiled = generation;
			self.realtime_script = self.worker.compiled();
		}
	}
}

impl Effect<2> for ScriptDistortionEffect {
	fn delay(&self) -> usize {
		0
	}

	fn process(
		&mut self,
		samples: &mut [f32; 2],
		_other: &[&[f32; 2]],
		_context: &mut Box<dyn ProcessContext>,
	) {
		self.sync_script();
		self.request_render();
		self.sync_table();
		self.sync_compiled();

		let dry = *samples;
		let out_gain = 10.0_f32.powf(self.out_gain / 20.0);
		let mix = self.mix.clamp(0.0, 1.0);
		let in_gain = 10.0_f32.powf(self.in_gain / 20.0);

		if self.realtime && let Some(script) = self.realtime_script.clone() {
				let _ = self.realtime_scope.set_value("p1", self.p1);
				let _ = self.realtime_scope.set_value("p2", self.p2);
				let _ = self.realtime_scope.set_value("p3", self.p3);
				let _ = self.realtime_scope.set_value("p4", self.p4);

				for (channel, sample) in samples.iter_mut().enumerate() {
					let input = dry[channel];
					let _ = self.realtime_scope.set_value("x", input * in_gain);

					let shaped = script
						.number(&mut self.realtime_scope)
						.ok()
						.filter(|value| value.is_finite())
						.map(|value| value * out_gain)
						.or_else(|| {
							self.table
								.as_ref()
								.map(|table| table.sample(input) * out_gain)
						})
						.unwrap_or(input);

					*sample = input * (1.0 - mix) + shaped * mix;
				}

				return;
		}

		let Some(table) = self.table.as_ref() else {
			// Nothing rendered yet: pass the dry signal rather than silence.
			return;
		};

		for (channel, sample) in samples.iter_mut().enumerate() {
			let shaped = table.sample(dry[channel]) * out_gain;
			*sample = dry[channel] * (1.0 - mix) + shaped * mix;
		}
	}
}

/// One control's current value.
fn value(map: &ParamMap, id: &str, fallback: f32) -> f32 {
	match map.get(id).map(|value| value.load(Ordering::Relaxed)) {
		Some(SetValue::Float(value)) => value,
		_ => fallback,
	}
}

/// Answers the page's script and curve routes.
fn route(
	request: &ProtocolRequest<'_>,
	worker: &ScriptWorker<DistortionTable>,
	map: &ParamMap,
) -> Option<ProtocolResponse> {
	if request.method == "GET" && request.path == "/script" {
		let status = worker.status();
		let error = match status.error {
			Some(error) => format!("\"{}\"", json_escape(&error)),
			None => "null".to_string(),
		};

		return Some(ProtocolResponse::json(format!(
			"{{\"source\":\"{}\",\"error\":{},\"applied\":{}}}",
			json_escape(&worker.source()),
			error,
			status.applied,
		)));
	}

	if request.method == "POST" && request.path == "/script" {
		let source = String::from_utf8_lossy(request.body).to_string();
		worker.set_source(source);

		let params = [
			value(map, "p1", 0.0),
			value(map, "p2", 0.0),
			value(map, "p3", 0.0),
			value(map, "p4", 0.0),
		];
		let in_gain = 10.0_f32.powf(value(map, "in_gain", 0.0) / 20.0);
		worker.request(params, in_gain);

		return Some(ProtocolResponse::json("{\"ok\":true}"));
	}

	if request.method == "GET" && request.path == "/curve" {
		let points = worker.curve();
		let mut json = String::from("{\"points\":[");

		for (index, [x, y, _]) in points.iter().enumerate() {
			if index > 0 {
				json.push(',');
			}
			let _ = std::fmt::Write::write_fmt(&mut json, format_args!("[{x:.4},{y:.4}]"));
		}

		json.push_str("]}");
		return Some(ProtocolResponse::json(json));
	}

	None
}

fn distortion_worker() -> Arc<ScriptWorker<DistortionTable>> {
	let worker = ScriptWorker::spawn(
		DistortionTable::render,
		DistortionTable::curve_points,
	);
	worker.set_source(DEFAULT_SCRIPT.to_string());
	worker
}

/// The product.
pub struct ScriptDistortionWeb {
	processor: EffectProcessor<ScriptDistortionEffect>,
	bypass: Arc<AtomicBool>,
	scope: Arc<Scope>,
	worker: Arc<ScriptWorker<DistortionTable>>,
}

impl Default for ScriptDistortionWeb {
	fn default() -> Self {
		let worker = distortion_worker();
		let build_worker = worker.clone();

		Self {
			processor: EffectProcessor::with_build(move |_| {
				ScriptDistortionEffect::new(build_worker.clone())
			}),
			bypass: Arc::new(AtomicBool::new(false)),
			scope: Arc::new(Scope::new()),
			worker,
		}
	}
}

impl i_am_dsp::prelude::Parameters for ScriptDistortionWeb {
	fn get_parameters(&self) -> Vec<Parameter> {
		self.processor.parameters()
	}

	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		self.processor.set_parameter(identifier, value)
	}
}

impl i_am_dsp::Processor for ScriptDistortionWeb {
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

impl Plugin for ScriptDistortionWeb {
	const DESCRIPTOR: Descriptor =
		Descriptor::new("iamdsp.effects.script_distortion.web", "I Am Script Distortion")
			.with_vendor("iamplugins")
			.with_version(env!("CARGO_PKG_VERSION"));

	const HAS_GUI: bool = true;

	fn new() -> Self {
		Self::default()
	}

	fn window_options() -> WindowOptions {
		const FACE_FRACTION: f64 = 0.85;

		WindowOptions::new()
			.with_size((700.0 / FACE_FRACTION, 560.0 / FACE_FRACTION))
			.with_title("I Am Script Distortion")
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
		let face = i_am_web_gui::Faceplate::new(
			"I Am Script Distortion",
			"#ff7a6b",
			(700.0, 560.0),
		)
		.with_script()
		.with_header(&["realtime"]);
		let worker = self.worker.clone();
		let map = self.processor.param_map();

		i_am_web_gui::editor_with(
			self,
			face,
			self.bypass.clone(),
			self.scope.clone(),
			self.processor.sample_rate(),
			move |request: &ProtocolRequest<'_>| route(request, &worker, &map),
			|_: &str| false,
		)
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use std::time::{Duration, Instant};

	fn table_of(source: &str, params: [f32; 4], in_gain: f32) -> DistortionTable {
		let script = CompiledScript::compile(source).expect("a compiled script");
		DistortionTable::render(&script, params, in_gain).expect("a rendered table")
	}

	#[test]
	fn the_table_samples_the_script() {
		let table = table_of("x * 2.0", [0.0; 4], 1.0);

		assert!((table.sample(0.25) - 0.5).abs() < 1e-4);
		assert!((table.sample(-0.5) + 1.0).abs() < 1e-4);
	}

	#[test]
	fn the_input_gain_is_baked_into_the_table() {
		let table = table_of("x * 2.0", [0.0; 4], 2.0);

		assert!((table.sample(0.25) - 1.0).abs() < 1e-4);
	}

	#[test]
	fn the_controls_are_in_scope() {
		let table = table_of("x * (1.0 + p1)", [1.0, 0.0, 0.0, 0.0], 1.0);

		assert!((table.sample(0.5) - 1.0).abs() < 1e-4);
	}

	#[test]
	fn a_script_with_a_syntax_error_does_not_compile() {
		assert!(CompiledScript::compile("x +").is_err());
	}

	#[test]
	fn a_script_that_fails_at_runtime_is_an_error() {
		let script = CompiledScript::compile("nope").expect("the lookup compiles");
		assert!(DistortionTable::render(&script, [0.0; 4], 1.0).is_err());
	}

	#[test]
	fn the_worker_renders_a_requested_table() {
		let worker = distortion_worker();
		let started = Instant::now();
		assert!(worker.request([0.0; 4], 1.0));

		while worker.table_generation() == 0 {
			assert!(
				started.elapsed() < Duration::from_secs(2),
				"the worker did not render"
			);
			std::thread::sleep(Duration::from_millis(5));
		}

		let table = worker.table().expect("a table");
		assert!(table.sample(0.5).is_finite());
		assert!(worker.status().error.is_none());
	}

	#[test]
	fn a_broken_script_keeps_the_previous_table() {
		let worker = distortion_worker();
		worker.request([0.0; 4], 1.0);
		let started = Instant::now();

		while worker.table_generation() == 0 {
			assert!(started.elapsed() < Duration::from_secs(2), "no first table");
			std::thread::sleep(Duration::from_millis(5));
		}

		worker.set_source("x +".to_string());
		worker.request([0.0; 4], 1.0);

		while worker.status().error.is_none() {
			assert!(started.elapsed() < Duration::from_secs(2), "no error");
			std::thread::sleep(Duration::from_millis(5));
		}

		assert!(worker.table().is_some(), "the old table is kept");
	}

	#[test]
	fn the_realtime_switch_runs_the_script_on_the_audio_thread() {
		let worker = distortion_worker();
		worker.set_source("x * 2.0".to_string());
		worker.request([0.0; 4], 1.0);

		let started = Instant::now();

		while worker.compiled_generation() == 0 {
			assert!(started.elapsed() < Duration::from_secs(2), "no compiled script");
			std::thread::sleep(Duration::from_millis(5));
		}

		let mut effect = ScriptDistortionEffect::new(worker);
		effect.realtime = true;
		effect.mix = 1.0;

		let mut context: Box<dyn ProcessContext> = Box::new(());
		let mut frame = [0.25, -0.5];
		effect.process(&mut frame, &[], &mut context);

		assert!((frame[0] - 0.5).abs() < 1e-3, "{}", frame[0]);
		assert!((frame[1] + 1.0).abs() < 1e-3, "{}", frame[1]);
	}

	#[test]
	fn the_script_routes_round_trip() {
		let worker = distortion_worker();
		let plugin = ScriptDistortionWeb::new();
		let map = plugin.param_map();
		let get = ProtocolRequest {
			method: "GET",
			path: "/script",
			query: None,
			body: &[],
		};

		let body = String::from_utf8(
			route(&get, &worker, &map).expect("the script route").body,
		)
		.expect("utf-8");
		assert!(body.contains("\"source\""), "{body}");
		assert!(body.contains("tanh"), "{body}");

		let post = ProtocolRequest {
			method: "POST",
			path: "/script",
			query: None,
			body: b"x * 3.0",
		};
		assert_eq!(route(&post, &worker, &map).expect("the route").status, 200);

		let started = Instant::now();

		while worker.table_generation() == 0 {
			assert!(started.elapsed() < Duration::from_secs(2), "no table");
			std::thread::sleep(Duration::from_millis(5));
		}

		let curve = ProtocolRequest {
			method: "GET",
			path: "/curve",
			query: None,
			body: &[],
		};
		let body = String::from_utf8(
			route(&curve, &worker, &map).expect("the curve route").body,
		)
		.expect("utf-8");
		assert!(body.contains("\"points\""), "{body}");
		assert!(body.contains("["), "{body}");

		let body = String::from_utf8(
			route(&get, &worker, &map).expect("the script route").body,
		)
		.expect("utf-8");
		assert!(body.contains("x * 3.0"), "{body}");
	}

	#[test]
	fn the_effect_mixes_dry_and_shaped() {
		let worker = distortion_worker();
		let mut effect = ScriptDistortionEffect::new(worker);
		effect.out_gain = 0.0;
		effect.mix = 0.5;

		let script = CompiledScript::compile("x * 2.0").expect("a script");
		effect.table = Some(Arc::new(
			DistortionTable::render(&script, [0.0; 4], 1.0).expect("a table"),
		));

		let mut context: Box<dyn ProcessContext> = Box::new(());
		let mut frame = [0.5, -0.5];
		effect.process(&mut frame, &[], &mut context);

		assert!((frame[0] - 0.75).abs() < 1e-4, "{}", frame[0]);
		assert!((frame[1] + 0.75).abs() < 1e-4, "{}", frame[1]);
	}
}
