//! The scripted frequency mapper: the script maps every bin to a new frequency and
//! amplitude, rendered into a frequency/amplitude table for the audio thread.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use i_am_dsp::prelude::{
	Effect, FrequencyMapper, ParamMap, Parameter, PhaseVocoder, ProcessContext, SetValue,
	overlap_add_gain,
};
use i_am_dsp_derive::Parameters;
use i_am_plugin::processor::{EffectMessage, EffectProcessor};
use i_am_plugin::{Descriptor, Editor, Plugin, WindowOptions};
use i_am_web_gui::{ProtocolRequest, ProtocolResponse, Scope};

use crate::script::{CompiledScript, ScriptWorker, json_escape};

/// Frequency points in the table, from 20 Hz to 20 kHz.
const FREQ_POINTS: usize = 1024;
/// Amplitude points in the table, from -80 dB to +80 dB.
///
/// The phase vocoder hands the mapper the raw FFT magnitude, which for a loud
/// bin is in the hundreds, not in 0..1. A 0 dB ceiling used to clamp every loud
/// bin to the same amplitude, flattening the spectrum into noise.
const AMP_POINTS: usize = 32;
/// The bottom of the frequency grid.
const MIN_HZ: f32 = 20.0;
/// The top of the frequency grid.
const MAX_HZ: f32 = 20_000.0;
/// The bottom of the amplitude grid, in dB.
const MIN_DB: f32 = -80.0;
/// The top of the amplitude grid, in dB.
const MAX_DB: f32 = 80.0;

/// The script a new instance starts with: the identity mapping.
pub const DEFAULT_SCRIPT: &str = "[x * (1.0 + p1), a]";

/// A mapping sampled on a frequency/amplitude grid.
///
/// The script receives both the bin frequency and its amplitude, so the stored
/// result depends on both: the same frequency can map differently for a quiet and
/// a loud bin.
pub struct MapperTable {
	freqs: Vec<f32>,
	amps: Vec<f32>,
}

impl MapperTable {
	/// Renders the script over the whole grid.
	fn render(script: &CompiledScript, params: [f32; 4], sample_rate: f32) -> Result<Self, String> {
		let nyquist = (sample_rate * 0.5).max(1_000.0);
		let mut scope = CompiledScript::scope(params);
		scope.push("a", 0.0_f32);
		scope.push("nyquist", nyquist);

		let points = FREQ_POINTS * AMP_POINTS;
		let mut freqs = Vec::with_capacity(points);
		let mut amps = Vec::with_capacity(points);

		for amp_index in 0..AMP_POINTS {
			let t = amp_index as f32 / (AMP_POINTS - 1) as f32;
			let amplitude = 10.0_f32.powf((MIN_DB + (MAX_DB - MIN_DB) * t) / 20.0);

			for freq_index in 0..FREQ_POINTS {
				let t = freq_index as f32 / (FREQ_POINTS - 1) as f32;
				let frequency = MIN_HZ * (MAX_HZ / MIN_HZ).powf(t);

				let _ = scope.set_value("x", frequency);
				let _ = scope.set_value("a", amplitude);

				let (mapped_frequency, mapped_amplitude) = script.pair(&mut scope)?;

				if !mapped_frequency.is_finite() || !mapped_amplitude.is_finite() {
					return Err("the script returned a non-finite value".to_string());
				}

				freqs.push(mapped_frequency.clamp(1.0, nyquist));
				amps.push(mapped_amplitude.clamp(0.0, 10_000.0));
			}
		}

		Ok(Self { freqs, amps })
	}

	/// The mapped frequency and amplitude for one bin.
	pub fn sample(&self, frequency: f32, amplitude: f32) -> (f32, f32) {
		let x = log_position(frequency);
		let y = amp_position(amplitude);

		let x0 = x.floor() as usize;
		let x1 = (x0 + 1).min(FREQ_POINTS - 1);
		let y0 = y.floor() as usize;
		let y1 = (y0 + 1).min(AMP_POINTS - 1);
		let tx = x - x0 as f32;
		// The grid is spaced in dB, but a script such as a * 0.5 is linear in the
		// amplitude; interpolating by the dB position would overshoot it at the
		// top of the range.
		let amp0 = row_amplitude(y0);
		let amp1 = row_amplitude(y1);
		let ty = if amp1 > amp0 {
			((amplitude - amp0) / (amp1 - amp0)).clamp(0.0, 1.0)
		} else {
			0.0
		};

		let at = |y: usize, x: usize| {
			let index = y * FREQ_POINTS + x;
			(self.freqs[index], self.amps[index])
		};

		let (f00, a00) = at(y0, x0);
		let (f10, a10) = at(y0, x1);
		let (f01, a01) = at(y1, x0);
		let (f11, a11) = at(y1, x1);

		let f0 = f00 + (f10 - f00) * tx;
		let f1 = f01 + (f11 - f01) * tx;
		let a0 = a00 + (a10 - a00) * tx;
		let a1 = a01 + (a11 - a01) * tx;

		(f0 + (f1 - f0) * ty, a0 + (a1 - a0) * ty)
	}

	/// The mapping the page draws, at the 0 dB amplitude.
	fn curve_points(&self) -> Vec<[f32; 3]> {
		const POINTS: usize = 256;

		(0..POINTS)
			.map(|index| {
				let t = index as f32 / (POINTS - 1) as f32;
				let frequency = MIN_HZ * (MAX_HZ / MIN_HZ).powf(t);
				let grid = (t * (FREQ_POINTS - 1) as f32).round() as usize;
				let row =
					((0.0 - MIN_DB) / (MAX_DB - MIN_DB) * (AMP_POINTS - 1) as f32).round() as usize;
				let index = row * FREQ_POINTS + grid;

				[frequency, self.freqs[index], self.amps[index]]
			})
			.collect()
	}
}

/// Where a frequency sits on the log grid.
fn log_position(frequency: f32) -> f32 {
	let frequency = frequency.clamp(MIN_HZ, MAX_HZ);
	let t = (frequency / MIN_HZ).ln() / (MAX_HZ / MIN_HZ).ln();
	t * (FREQ_POINTS - 1) as f32
}

/// Where an amplitude sits on the dB grid.
fn amp_position(amplitude: f32) -> f32 {
	let db = 20.0 * amplitude.max(1e-5).log10();
	let t = ((db - MIN_DB) / (MAX_DB - MIN_DB)).clamp(0.0, 1.0);
	t * (AMP_POINTS - 1) as f32
}

/// The amplitude at one row of the dB grid.
fn row_amplitude(row: usize) -> f32 {
	let t = row as f32 / (AMP_POINTS - 1) as f32;
	10.0_f32.powf((MIN_DB + (MAX_DB - MIN_DB) * t) / 20.0)
}

/// The mapper the phase vocoder calls per bin.
pub struct ScriptMapper {
	table: Option<Arc<MapperTable>>,
	realtime_script: Option<Arc<CompiledScript>>,
	realtime: bool,
	params: [f32; 4],
	scope: rhai::Scope<'static>,
	mix: f32,
	nyquist: f32,
	moved: bool,
}

impl Default for ScriptMapper {
	fn default() -> Self {
		let mut scope = CompiledScript::scope([0.0; 4]);
		scope.push("a", 0.0_f32);
		scope.push("nyquist", 24_000.0_f32);

		Self {
			table: None,
			realtime_script: None,
			realtime: false,
			params: [0.0; 4],
			scope,
			mix: 1.0,
			nyquist: 24_000.0,
			moved: false,
		}
	}
}

impl i_am_dsp::prelude::Parameters for ScriptMapper {
	fn get_parameters(&self) -> Vec<Parameter> {
		Vec::new()
	}

	fn set_parameter(&mut self, _: &str, _: SetValue) -> bool {
		false
	}
}

impl FrequencyMapper for ScriptMapper {
	fn map_frequency(&mut self, frequency: f32, amplitude: f32) -> (f32, f32) {
		let input = frequency;
		let mix = self.mix.clamp(0.0, 1.0);

		// Realtime: evaluate the script for this bin here, on the audio thread.
		// With a 2048 frame and fourfold overlap that is about 96 thousand
		// evaluations a second, rather than the table's sixteen thousand renders
		// per change.
		if self.realtime
			&& let Some(script) = self.realtime_script.clone()
		{
			let _ = self.scope.set_value("x", input);
			let _ = self.scope.set_value("a", amplitude);
			let _ = self.scope.set_value("p1", self.params[0]);
			let _ = self.scope.set_value("p2", self.params[1]);
			let _ = self.scope.set_value("p3", self.params[2]);
			let _ = self.scope.set_value("p4", self.params[3]);
			let _ = self.scope.set_value("nyquist", self.nyquist);

			if let Ok((mapped_frequency, mapped_amplitude)) = script.pair(&mut self.scope)
				&& mapped_frequency.is_finite()
				&& mapped_amplitude.is_finite()
			{
				let mapped_frequency = (input
					+ (mapped_frequency.clamp(1.0, self.nyquist) - input) * mix)
					.clamp(1.0, self.nyquist);
				let mapped_amplitude =
					amplitude + (mapped_amplitude.clamp(0.0, 10_000.0) - amplitude) * mix;

				self.moved = (mapped_frequency - input).abs() > 0.5;
				return (mapped_frequency, mapped_amplitude.max(0.0));
			}
		}

		let Some(table) = &self.table else {
			self.moved = false;
			return (frequency, amplitude);
		};

		let (mapped_frequency, mapped_amplitude) = table.sample(input, amplitude);
		let mapped_frequency = (input + (mapped_frequency - input) * mix).clamp(1.0, self.nyquist);
		let mapped_amplitude = amplitude + (mapped_amplitude - amplitude) * mix;

		self.moved = (mapped_frequency - input).abs() > 0.5;
		(mapped_frequency, mapped_amplitude.max(0.0))
	}

	fn synthesis_phase(
		&mut self,
		_frequency: f32,
		analysis_phase: f32,
		propagated_phase: f32,
	) -> f32 {
		if self.moved {
			propagated_phase
		} else {
			analysis_phase
		}
	}
}

/// The product's DSP.
#[derive(Parameters)]
pub struct ScriptMapperEffect {
	/// How much of the mapped spectrum is heard.
	#[range(min = 0.0, max = 1.0)]
	#[unit(percent)]
	#[default(1.0)]
	#[id(display = "Mix")]
	#[layout(group = "Script Mapper", order = 0, widget = knob, hero = "mapping")]
	pub mix: f32,

	/// The first script control.
	#[range(min = -1.0, max = 1.0)]
	#[default(0.0)]
	#[id(display = "P1")]
	#[layout(group = "Script Mapper", order = 1, widget = knob)]
	pub p1: f32,

	/// The second script control.
	#[range(min = -1.0, max = 1.0)]
	#[default(0.0)]
	#[id(display = "P2")]
	#[layout(group = "Script Mapper", order = 2, widget = knob)]
	pub p2: f32,

	/// The third script control.
	#[range(min = -1.0, max = 1.0)]
	#[default(0.0)]
	#[id(display = "P3")]
	#[layout(group = "Script Mapper", order = 3, widget = knob)]
	pub p3: f32,

	/// The fourth script control.
	#[range(min = -1.0, max = 1.0)]
	#[default(0.0)]
	#[id(display = "P4")]
	#[layout(group = "Script Mapper", order = 4, widget = knob)]
	pub p4: f32,

	/// The FFT window size, from the header.
	#[choices("512", "1024", "2048", "4096")]
	#[default(2)]
	#[id(display = "Window Size")]
	#[layout(group = "Script Mapper", order = 5, widget = choice)]
	pub window_size: i32,

	/// Evaluate the script on the audio thread instead of the rendered table.
	#[default(false)]
	#[id(display = "Realtime")]
	#[layout(group = "Script Mapper", order = 6, widget = toggle)]
	pub realtime: bool,

	/// The Rhai body, saved with the plug-in state.
	#[serde]
	#[id(display = "Script")]
	pub script: String,

	#[skip]
	vocoder: PhaseVocoder<ScriptMapper, 2>,
	#[skip]
	worker: Arc<ScriptWorker<MapperTable>>,
	#[skip]
	table: Option<Arc<MapperTable>>,
	#[skip]
	sample_rate: usize,
	#[skip]
	seen_table: u64,
	#[skip]
	synced_source: u64,
	#[skip]
	last_script: String,
	#[skip]
	last_request: Option<(u64, [f32; 4], f32)>,
	#[skip]
	compiled: Option<Arc<CompiledScript>>,
	#[skip]
	seen_compiled: u64,
	#[skip]
	last_window_size: i32,
}

impl ScriptMapperEffect {
	/// A mapper with the default script, sharing the worker with its editor.
	pub fn new(sample_rate: usize, worker: Arc<ScriptWorker<MapperTable>>) -> Self {
		let mut vocoder: PhaseVocoder<ScriptMapper, 2> =
			PhaseVocoder::new(ScriptMapper::default(), 2048, sample_rate);
		vocoder.window_factor = 0.6;
		vocoder.gain = 1.0 / overlap_add_gain(vocoder.window_factor);

		Self {
			mix: 1.0,
			p1: 0.0,
			p2: 0.0,
			p3: 0.0,
			p4: 0.0,
			window_size: 2,
			realtime: false,
			script: DEFAULT_SCRIPT.to_string(),
			vocoder,
			worker,
			table: None,
			sample_rate,
			seen_table: 0,
			synced_source: 0,
			last_script: DEFAULT_SCRIPT.to_string(),
			last_request: None,
			compiled: None,
			seen_compiled: 0,
			last_window_size: 2,
		}
	}

	fn sync_script(&mut self) {
		let generation = self.worker.source_generation();

		if generation != self.synced_source {
			self.synced_source = generation;
			let source = self.worker.source();
			self.script = source.clone();
			self.last_script = source;
		} else if self.script != self.last_script {
			let generation = self.worker.set_source(self.script.clone());
			self.synced_source = generation;
			self.last_script = self.script.clone();
		}
	}

	fn request_render(&mut self) {
		let generation = self.worker.source_generation();
		let params = [self.p1, self.p2, self.p3, self.p4];
		let request = (generation, params, self.sample_rate as f32);

		if self.last_request != Some(request)
			&& self.worker.request(params, self.sample_rate as f32)
		{
			self.last_request = Some(request);
		}
	}

	fn sync_table(&mut self) {
		let generation = self.worker.table_generation();

		if generation != self.seen_table {
			self.seen_table = generation;
			self.table = self.worker.table();
			self.vocoder.mapper.table = self.table.clone();
		}

		self.vocoder.mapper.mix = self.mix;
		self.vocoder.mapper.nyquist = self.sample_rate as f32 * 0.5;
	}

	/// Takes a freshly compiled script for the realtime path.
	fn sync_compiled(&mut self) {
		let generation = self.worker.compiled_generation();

		if generation != self.seen_compiled {
			self.seen_compiled = generation;
			self.compiled = self.worker.compiled();
			self.vocoder.mapper.realtime_script = self.compiled.clone();
		}

		self.vocoder.mapper.realtime = self.realtime;
		self.vocoder.mapper.params = [self.p1, self.p2, self.p3, self.p4];
	}

	/// Rebuilds the vocoder when the window size changes.
	fn sync_window(&mut self) {
		if self.window_size == self.last_window_size {
			return;
		}

		let size = [512, 1024, 2048, 4096][self.window_size.clamp(0, 3) as usize];

		if self.vocoder.renew_window_size(size).is_some() {
			self.vocoder.gain = 1.0 / overlap_add_gain(self.vocoder.window_factor);
		}

		self.last_window_size = self.window_size;
	}
}

impl Effect<2> for ScriptMapperEffect {
	fn delay(&self) -> usize {
		self.vocoder.delay()
	}

	fn process(
		&mut self,
		samples: &mut [f32; 2],
		other: &[&[f32; 2]],
		context: &mut Box<dyn ProcessContext>,
	) {
		self.sync_script();
		self.request_render();
		self.sync_table();
		self.sync_compiled();
		self.sync_window();
		self.vocoder.process(samples, other, context);
	}
}

fn value(map: &ParamMap, id: &str, fallback: f32) -> f32 {
	match map.get(id).map(|value| value.load(Ordering::Relaxed)) {
		Some(SetValue::Float(value)) => value,
		_ => fallback,
	}
}

fn route(
	request: &ProtocolRequest<'_>,
	worker: &ScriptWorker<MapperTable>,
	map: &ParamMap,
	sample_rate: f32,
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
		worker.request(params, sample_rate);

		return Some(ProtocolResponse::json("{\"ok\":true}"));
	}

	if request.method == "GET" && request.path == "/curve" {
		let points = worker.curve();
		let mut json = String::from("{\"points\":[");

		for (index, [frequency, mapped, amplitude]) in points.iter().enumerate() {
			if index > 0 {
				json.push(',');
			}
			let _ = std::fmt::Write::write_fmt(
				&mut json,
				format_args!("[{frequency:.4},{mapped:.4},{amplitude:.3}]"),
			);
		}

		json.push_str("]}");
		return Some(ProtocolResponse::json(json));
	}

	None
}

/// The worker a new script mapper starts with.
pub fn mapper_worker() -> Arc<ScriptWorker<MapperTable>> {
	let worker = ScriptWorker::spawn(MapperTable::render, MapperTable::curve_points);
	worker.set_source(DEFAULT_SCRIPT.to_string());
	worker
}

/// The product.
pub struct ScriptMapperWeb {
	processor: EffectProcessor<ScriptMapperEffect>,
	bypass: Arc<AtomicBool>,
	scope: Arc<Scope>,
	worker: Arc<ScriptWorker<MapperTable>>,
}

impl ScriptMapperWeb {
	/// The worker the page edits.
	pub fn worker(&self) -> Arc<dyn crate::ScriptSource> {
		self.worker.clone()
	}

	/// The faceplate size this product declares to a host.
	pub const fn size() -> (f32, f32) {
		(700.0, 560.0)
	}
}

impl Default for ScriptMapperWeb {
	fn default() -> Self {
		let worker = mapper_worker();
		let build_worker = worker.clone();

		Self {
			processor: EffectProcessor::with_build(move |rate| {
				ScriptMapperEffect::new(rate, build_worker.clone())
			}),
			bypass: Arc::new(AtomicBool::new(false)),
			scope: Arc::new(Scope::new()),
			worker,
		}
	}
}

impl Drop for ScriptMapperWeb {
	fn drop(&mut self) {
		// The worker thread must be gone before the host unloads this DLL.
		self.worker.shutdown();
	}
}

impl i_am_dsp::prelude::Parameters for ScriptMapperWeb {
	fn get_parameters(&self) -> Vec<Parameter> {
		self.processor.parameters()
	}

	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		self.processor.set_parameter(identifier, value)
	}
}

impl i_am_dsp::Processor for ScriptMapperWeb {
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

impl Plugin for ScriptMapperWeb {
	const DESCRIPTOR: Descriptor =
		Descriptor::new("iamdsp.effects.script_mapper.web", "I Am Script Mapper")
			.with_vendor("iamplugins")
			.with_version(env!("CARGO_PKG_VERSION"));

	const HAS_GUI: bool = true;

	fn new() -> Self {
		Self::default()
	}

	fn window_options() -> WindowOptions {
		i_am_web_gui::window_options((700.0, 560.0), "I Am Script Mapper")
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
		let face = i_am_web_gui::Faceplate::new("I Am Script Mapper", "#8ad3ff", (700.0, 560.0))
			.with_script()
			.with_header(&["window_size", "realtime"]);
		let worker = self.worker.clone();
		let map = self.processor.param_map();
		let sample_rate = self.processor.sample_rate() as f32;

		i_am_web_gui::editor_with(
			self,
			face,
			self.bypass.clone(),
			self.scope.clone(),
			self.processor.sample_rate(),
			move |request: &ProtocolRequest<'_>| route(request, &worker, &map, sample_rate),
			|_: &str| false,
		)
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use std::time::{Duration, Instant};

	fn table_of(source: &str, params: [f32; 4], rate: f32) -> MapperTable {
		let script = CompiledScript::compile(source).expect("a compiled script");
		MapperTable::render(&script, params, rate).expect("a rendered table")
	}

	#[test]
	fn the_identity_mapping_passes_the_grid_through() {
		let table = table_of("[x, a]", [0.0; 4], 48_000.0);
		let t = 512.0 / (FREQ_POINTS - 1) as f32;
		let frequency = MIN_HZ * (MAX_HZ / MIN_HZ).powf(t);
		let (mapped, amplitude) = table.sample(frequency, 1.0);

		assert!((mapped - frequency).abs() < 1.0, "{mapped} vs {frequency}");
		assert!((amplitude - 1.0).abs() < 0.02, "{amplitude}");
	}

	#[test]
	fn the_amplitude_axis_matters() {
		let table = table_of("[x, a * 0.5]", [0.0; 4], 48_000.0);
		let (_, quiet) = table.sample(1000.0, 0.1);
		let (_, loud) = table.sample(1000.0, 1.0);

		assert!(loud > quiet, "{quiet} vs {loud}");
		assert!((loud - 0.5).abs() < 0.02, "{loud}");
	}

	#[test]
	fn the_mix_blends_towards_the_mapping() {
		let table = Arc::new(table_of("[x * 2.0, a]", [0.0; 4], 48_000.0));
		let mut mapper = ScriptMapper {
			table: Some(table),
			mix: 0.0,
			nyquist: 24_000.0,
			..ScriptMapper::default()
		};

		assert!((mapper.map_frequency(1000.0, 0.5).0 - 1000.0).abs() < 1.0);

		mapper.mix = 1.0;
		let (frequency, amplitude) = mapper.map_frequency(1000.0, 0.5);

		assert!((frequency - 2000.0).abs() < 2.0, "{frequency}");
		assert!((amplitude - 0.5).abs() < 0.01, "{amplitude}");
	}

	fn shifted_sine_tone_ratio(realtime: bool) -> f32 {
		let source = "[x * 2.0, a]";
		let mut mapper = ScriptMapper {
			realtime,
			realtime_script: if realtime {
				Some(Arc::new(
					CompiledScript::compile(source).expect("a compiled script"),
				))
			} else {
				None
			},
			table: if realtime {
				None
			} else {
				Some(Arc::new(table_of(source, [0.0; 4], 48_000.0)))
			},
			..ScriptMapper::default()
		};
		mapper.nyquist = 24_000.0;

		let mut vocoder: PhaseVocoder<ScriptMapper, 2> = PhaseVocoder::new(mapper, 2048, 48_000);
		vocoder.window_factor = 0.6;
		vocoder.gain = 1.0 / overlap_add_gain(vocoder.window_factor);

		let mut context: Box<dyn ProcessContext> = Box::new(());
		let sample_rate = 48_000.0;
		let total = 4096 + 12_288;
		let mut output = Vec::with_capacity(total);

		for i in 0..total {
			let x = 0.5 * (2.0 * std::f32::consts::PI * 500.0 * i as f32 / sample_rate).sin();
			let mut frame = [x, x];
			vocoder.process(&mut frame, &[], &mut context);
			output.push(frame[0]);
		}

		let tail = &output[4096..];
		let length = tail.len() as f32;
		let total_power: f32 = tail.iter().map(|sample| sample * sample).sum();
		let mut real = 0.0;
		let mut imaginary = 0.0;

		for (index, sample) in tail.iter().enumerate() {
			let phase = 2.0 * std::f32::consts::PI * 1000.0 * index as f32 / sample_rate;
			real += sample * phase.cos();
			imaginary += sample * phase.sin();
		}

		let tone = 2.0 * (real * real + imaginary * imaginary) / (length * length);
		2.0 * tone / (total_power / length)
	}

	#[test]
	fn a_shifted_sine_stays_tonal() {
		let table = shifted_sine_tone_ratio(false);
		let realtime = shifted_sine_tone_ratio(true);

		assert!(table > 0.8, "table: {table}");
		assert!(realtime > 0.8, "realtime: {realtime}");
	}

	#[test]
	fn the_realtime_mapping_runs_the_script_per_bin() {
		let script =
			Arc::new(CompiledScript::compile("[x * 2.0, a * 0.5]").expect("a compiled script"));
		let mut mapper = ScriptMapper {
			realtime: true,
			realtime_script: Some(script),
			..ScriptMapper::default()
		};

		let (frequency, amplitude) = mapper.map_frequency(1000.0, 0.5);

		assert!((frequency - 2000.0).abs() < 1.0, "{frequency}");
		assert!((amplitude - 0.25).abs() < 0.01, "{amplitude}");
		assert!(mapper.moved);
	}

	#[test]
	fn a_broken_mapping_script_does_not_compile() {
		assert!(CompiledScript::compile("[x").is_err());
	}

	#[test]
	fn the_mapper_routes_round_trip() {
		let worker = mapper_worker();
		let plugin = ScriptMapperWeb::new();
		let map = plugin.param_map();
		let get = ProtocolRequest {
			method: "GET",
			path: "/script",
			query: None,
			body: &[],
		};

		let body = String::from_utf8(
			route(&get, &worker, &map, 48_000.0)
				.expect("the script route")
				.body,
		)
		.expect("utf-8");
		assert!(body.contains("\"source\""), "{body}");

		let post = ProtocolRequest {
			method: "POST",
			path: "/script",
			query: None,
			body: b"[x * 2.0, a]",
		};
		assert_eq!(
			route(&post, &worker, &map, 48_000.0)
				.expect("the route")
				.status,
			200
		);

		let started = Instant::now();

		while worker.table_generation() == 0 {
			assert!(started.elapsed() < Duration::from_secs(3), "no table");
			std::thread::sleep(Duration::from_millis(10));
		}

		let curve = ProtocolRequest {
			method: "GET",
			path: "/curve",
			query: None,
			body: &[],
		};
		let body = String::from_utf8(
			route(&curve, &worker, &map, 48_000.0)
				.expect("the curve route")
				.body,
		)
		.expect("utf-8");
		assert!(body.contains("\"points\""), "{body}");
		assert!(body.contains("["), "{body}");
	}
}
