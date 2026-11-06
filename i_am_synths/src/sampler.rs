//! The sampler product: one sample, played as a sampler, a slicer or a grain
//! cloud.

use std::io::Cursor;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use hound::{SampleFormat, WavReader};
use i_am_dsp::prelude::{
	Adsr, EqualTemperament, GeneratorEffect, ParamMap, Parameter, Parameters, SampleData,
	SampleSlot, SamplerVoice, SetValue,
};
use i_am_dsp::{ProcessContext, Processor};
use i_am_plugin::processor::{EffectMessage, EffectProcessor};
use i_am_plugin::{
	AspectRatioStrategy, Descriptor, GuiResizeHints, MidiPort, Plugin, Tag, WindowOptions,
};
use i_am_web_gui::{
	Faceplate, Page, ProtocolRequest, ProtocolResponse, Scope, Switch, editor_with,
};

use super::{json_escape, play_synth};

/// The sampler face's design size, in points.
const SAMPLER_SIZE: (f32, f32) = (820.0, 460.0);
/// The sampler window's title and the face's title.
const SAMPLER_TITLE: &str = "I Am Sampler";
/// The sampler identity colour, as #rrggbb.
const SAMPLER_ACCENT: &str = "#f0a24c";
/// How many points of the sample the page draws.
const PEAKS: usize = 512;

/// The sampler face's tabs: the sample, the envelope, and the voice's unisons.
const SAMPLER_PAGES: &[Page] = &[
	Page::new("Sample", &["Sample", "Loop"]),
	Page::new("Modes", &["Slice", "Grain"]),
	Page::new("Envelope", &["Envelope"]),
	Page::new("Voice", &["Voice"]),
];

/// The per-stage sync switches stay the host's: the face has one for the envelope.
const SAMPLER_HIDDEN: &[&str] = &[
	"delay.sync",
	"attack.sync",
	"hold.sync",
	"decay.sync",
	"release.sync",
	// The sample's own gain and pan sit on the Sound page; the envelope's copies
	// and its "shape" hero would be a second, unrelated picture.
	"gain",
	"pitch_factor",
	"pan",
	// The loop and its ends are drawn and dragged on the picture, and the two
	// flips are the hero's switches: the knobs would be a second way to do what
	// the hand already does.
	"oscillator.loop",
	"oscillator.loop_start",
	"oscillator.loop_end",
	"oscillator.reverse",
	"oscillator.polarity",
	"oscillator.pingpong",
];

/// The envelope's one Sync switch trades the free lengths for musical ones.
const SAMPLER_SWITCHES: &[Switch] = &[Switch::new(
	"sync",
	&[
		"delay.beats",
		"attack.beats",
		"hold.beats",
		"decay.beats",
		"release.beats",
	],
	&[
		"delay_time",
		"attack_time",
		"hold_time",
		"decay_time",
		"release_time",
	],
)];

/// The sampler voice: an envelope around the sample.
type SamplerSynth = Adsr<SamplerVoice, EqualTemperament, 2>;

fn sampler_voice(rate: usize, slot: Arc<Mutex<SampleSlot>>) -> SamplerSynth {
	Adsr::new(SamplerVoice::with_slot(slot), EqualTemperament, rate)
}

/// What the page draws for the sample.
#[derive(Clone, Default)]
struct SamplePreview {
	version: u64,
	name: Option<String>,
	rate: u32,
	samples: usize,
	peaks: Vec<f32>,
	/// Where each slice starts, as a fraction of the sample.
	slices: Vec<f32>,
	error: Option<String>,
}

/// The faceplate of the sampler.
pub fn sampler_faceplate() -> Faceplate {
	Faceplate::new(SAMPLER_TITLE, SAMPLER_ACCENT, SAMPLER_SIZE)
		.with_pages(SAMPLER_PAGES)
		.with_header(&["sync"])
		.with_switches(SAMPLER_SWITCHES)
		.with_hidden(SAMPLER_HIDDEN)
		.with_no_hero(&["Shape"])
}

/// Answers the editor's sample: a GET describes what is loaded, a POST imports a
/// WAV.
fn sample_route(
	request: &ProtocolRequest<'_>,
	slot: &Arc<Mutex<SampleSlot>>,
	preview: &Arc<Mutex<SamplePreview>>,
) -> Option<ProtocolResponse> {
	if request.path != "/sample" {
		return None;
	}

	match request.method {
		"GET" => {
			let (sample, slices) = {
				let guard = slot.lock().unwrap_or_else(|error| error.into_inner());

				(guard.sample.clone(), guard.slices.clone())
			};
			let mut cache = preview.lock().unwrap_or_else(|error| error.into_inner());

			let stale = match &sample {
				Some(sample) => {
					cache.version != Arc::as_ptr(sample) as u64 || cache.slices != slices
				}
				None => cache.version != 0,
			};

			if stale {
				*cache = match &sample {
					Some(sample) => build_preview(sample, &slices),
					None => SamplePreview::default(),
				};
			}

			Some(ProtocolResponse::json(preview_json(&cache)))
		}
		"POST" => {
			if let Some(action) = query_value(request.query, "action") {
				return Some(edit_sample(action, request.query, slot, preview));
			}

			let name = query_value(request.query, "name")
				.map(percent_decode)
				.filter(|name| !name.is_empty())
				.unwrap_or_else(|| "sample".to_string());

			match decode_sample(request.body, &name) {
				Ok(sample) => {
					let sample = Arc::new(sample);
					let mut guard = slot.lock().unwrap_or_else(|error| error.into_inner());

					guard.sample = Some(sample.clone());
					guard.slices.clear();
					drop(guard);

					let cache = build_preview(&sample, &[]);

					*preview.lock().unwrap_or_else(|error| error.into_inner()) = cache.clone();

					Some(ProtocolResponse::json(preview_json(&cache)))
				}
				Err(error) => {
					*preview.lock().unwrap_or_else(|error| error.into_inner()) = SamplePreview {
						error: Some(error.clone()),
						..SamplePreview::default()
					};

					Some(ProtocolResponse::json(format!(
						"{{\"ok\":false,\"error\":\"{}\"}}",
						json_escape(&error)
					)))
				}
			}
		}
		_ => Some(ProtocolResponse::not_found()),
	}
}

/// One value out of a query string like name=a%20b&action=slices.
fn query_value<'a>(query: Option<&'a str>, key: &str) -> Option<&'a str> {
	query?.split('&').find_map(|part| {
		let (name, value) = part.split_once('=')?;

		(name == key).then_some(value)
	})
}

fn build_preview(sample: &Arc<SampleData>, slices: &[f32]) -> SamplePreview {
	SamplePreview {
		version: Arc::as_ptr(sample) as u64,
		name: Some(sample.name.clone()),
		rate: sample.rate,
		samples: sample.pcm.len(),
		peaks: peaks(&sample.pcm),
		slices: slices.to_vec(),
		error: None,
	}
}

/// Edits the sample's slice marks: make a grid, follow the transients, or add
/// and remove single marks from the picture.
fn edit_sample(
	action: &str,
	query: Option<&str>,
	slot: &Arc<Mutex<SampleSlot>>,
	preview: &Arc<Mutex<SamplePreview>>,
) -> ProtocolResponse {
	let (sample, mut slices) = {
		let guard = slot.lock().unwrap_or_else(|error| error.into_inner());

		(guard.sample.clone(), guard.slices.clone())
	};
	let Some(sample) = sample else {
		return ProtocolResponse::json("{\"ok\":false,\"error\":\"no sample\"}");
	};

	match action {
		"slices" => {
			let mode = query_value(query, "mode").unwrap_or("grid");

			slices = if mode == "transient" {
				onsets(&sample.pcm, sample.rate)
			} else {
				let start = query_value(query, "start")
					.and_then(|value| value.parse::<f32>().ok())
					.unwrap_or(0.0);
				let step = query_value(query, "step")
					.and_then(|value| value.parse::<f32>().ok())
					.unwrap_or(0.125);

				grid(start, step)
			};
		}
		"slice-add" => {
			if let Some(position) =
				query_value(query, "at").and_then(|value| value.parse::<f32>().ok())
			{
				slices.push(position.clamp(0.0, 1.0));
				slices.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
				slices.dedup_by(|a, b| (*a - *b).abs() < 1e-4);
			}
		}
		"slice-move" => {
			let from = query_value(query, "from").and_then(|value| value.parse::<f32>().ok());
			let to = query_value(query, "to").and_then(|value| value.parse::<f32>().ok());

			if let (Some(from), Some(to)) = (from, to)
				&& let Some(index) = slices
					.iter()
					.enumerate()
					.min_by(|a, b| {
						(a.1 - from)
							.abs()
							.partial_cmp(&(b.1 - from).abs())
							.unwrap_or(std::cmp::Ordering::Equal)
					})
					.map(|(index, _)| index)
			{
				slices[index] = to.clamp(0.0, 1.0);
				slices.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
				slices.dedup_by(|a, b| (*a - *b).abs() < 1e-4);
			}
		}
		"slice-remove" => {
			if let Some(position) =
				query_value(query, "at").and_then(|value| value.parse::<f32>().ok())
				&& let Some(index) = slices
					.iter()
					.enumerate()
					.min_by(|a, b| {
						(a.1 - position)
							.abs()
							.partial_cmp(&(b.1 - position).abs())
							.unwrap_or(std::cmp::Ordering::Equal)
					})
					.map(|(index, _)| index)
			{
				slices.remove(index);
			}
		}
		_ => return ProtocolResponse::json("{\"ok\":false,\"error\":\"unknown action\"}"),
	}

	slices.truncate(64);
	slot.lock()
		.unwrap_or_else(|error| error.into_inner())
		.slices = slices.clone();

	let cache = build_preview(&sample, &slices);

	*preview.lock().unwrap_or_else(|error| error.into_inner()) = cache.clone();

	ProtocolResponse::json(preview_json(&cache))
}

/// Evenly spaced marks, from the start, at most 64 of them.
fn grid(start: f32, step: f32) -> Vec<f32> {
	let step = step.clamp(0.002, 1.0);
	let mut marks = Vec::new();
	let mut at = start.clamp(0.0, 1.0);

	while at < 1.0 && marks.len() < 64 {
		marks.push(at);
		at += step;
	}

	if marks.is_empty() {
		marks.push(0.0);
	}

	marks
}

/// The marks the transients suggest: where the energy jumps above the recent
/// average, which is where a drum or a pluck starts.
fn onsets(pcm: &[f32], rate: u32) -> Vec<f32> {
	let hop = (rate as usize / 200).max(64);
	let energy: Vec<f32> = pcm
		.chunks(hop)
		.map(|chunk| {
			chunk
				.iter()
				.map(|sample| sample * sample)
				.sum::<f32>()
				.sqrt()
		})
		.collect();

	if energy.is_empty() {
		return vec![0.0];
	}

	let loudest = energy.iter().copied().fold(0.0_f32, f32::max);
	let floor = loudest * 0.02;
	let mut marks = Vec::new();

	for (index, value) in energy.iter().enumerate() {
		let from = index.saturating_sub(8);
		let average = energy[from..index].iter().sum::<f32>() / (index - from).max(1) as f32;
		let position = (index * hop) as f32 / pcm.len() as f32;

		if *value > floor
			&& *value > average * 1.8
			&& position > 0.01
			&& marks
				.last()
				.map(|last| position - last > 0.005)
				.unwrap_or(true)
		{
			marks.push(position);
		}
	}

	marks.truncate(64);
	marks
}

fn preview_json(preview: &SamplePreview) -> String {
	let name = match &preview.name {
		Some(name) => format!("\"{}\"", json_escape(name)),
		None => "null".to_string(),
	};
	let error = match &preview.error {
		Some(error) => format!("\"{}\"", json_escape(error)),
		None => "null".to_string(),
	};
	let peaks: Vec<String> = preview
		.peaks
		.iter()
		.map(|peak| format!("{peak:.4}"))
		.collect();
	let slices: Vec<String> = preview
		.slices
		.iter()
		.map(|slice| format!("{slice:.4}"))
		.collect();

	format!(
		"{{\"version\":{},\"name\":{name},\"rate\":{},\"samples\":{},\"error\":{error},\"peaks\":[{}],\"slices\":[{}]}}",
		preview.version,
		preview.rate,
		preview.samples,
		peaks.join(","),
		slices.join(",")
	)
}

/// The sample's picture: the loudest value in each bucket, with its sign.
fn peaks(pcm: &[f32]) -> Vec<f32> {
	if pcm.is_empty() {
		return Vec::new();
	}

	let bucket = (pcm.len() / PEAKS).max(1);
	let mut peaks = Vec::with_capacity(PEAKS);

	for chunk in pcm.chunks(bucket).take(PEAKS) {
		let peak = chunk
			.iter()
			.copied()
			.max_by(|a, b| {
				a.abs()
					.partial_cmp(&b.abs())
					.unwrap_or(std::cmp::Ordering::Equal)
			})
			.unwrap_or(0.0);

		peaks.push(peak);
	}

	peaks
}

/// Reads a WAV into one mono sample, averaged across channels.
fn decode_sample(bytes: &[u8], name: &str) -> Result<SampleData, String> {
	let mut reader =
		WavReader::new(Cursor::new(bytes)).map_err(|error| format!("read wav: {error}"))?;
	let spec = reader.spec();
	let channels = spec.channels.max(1) as usize;
	let mut interleaved: Vec<f32> = Vec::new();

	match spec.sample_format {
		SampleFormat::Float => {
			for sample in reader.samples::<f32>() {
				interleaved.push(sample.map_err(|error| format!("read wav: {error}"))?);
			}
		}
		SampleFormat::Int => {
			let scale = (1u32 << (spec.bits_per_sample.saturating_sub(1)).min(31)) as f32;

			for sample in reader.samples::<i32>() {
				interleaved
					.push(sample.map_err(|error| format!("read wav: {error}"))? as f32 / scale);
			}
		}
	}

	if interleaved.is_empty() {
		return Err("the file holds no samples".to_string());
	}

	let pcm: Vec<f32> = if channels == 1 {
		interleaved
	} else {
		interleaved
			.chunks(channels)
			.map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
			.collect()
	};

	Ok(SampleData {
		name: name.to_string(),
		rate: spec.sample_rate,
		pcm,
	})
}

/// Decodes %20 style escapes; the page sends names with encodeURIComponent.
fn percent_decode(value: &str) -> String {
	let bytes = value.as_bytes();
	let mut out = Vec::with_capacity(bytes.len());
	let mut index = 0;

	while index < bytes.len() {
		if bytes[index] == b'%' && index + 2 < bytes.len() {
			let high = (bytes[index + 1] as char).to_digit(16);
			let low = (bytes[index + 2] as char).to_digit(16);

			if let (Some(high), Some(low)) = (high, low) {
				out.push((high * 16 + low) as u8);
				index += 3;
				continue;
			}
		}

		out.push(bytes[index]);
		index += 1;
	}

	String::from_utf8_lossy(&out).to_string()
}

/// The sampler instrument, with a web faceplate.
pub struct SamplerWeb {
	processor: EffectProcessor<GeneratorEffect<SamplerSynth>>,
	bypass: Arc<AtomicBool>,
	scope: Arc<Scope>,
	slot: Arc<Mutex<SampleSlot>>,
	preview: Arc<Mutex<SamplePreview>>,
}

impl Default for SamplerWeb {
	fn default() -> Self {
		let slot = Arc::new(Mutex::new(SampleSlot::default()));
		let build_slot = slot.clone();

		Self {
			processor: EffectProcessor::with_build(move |rate: usize| {
				GeneratorEffect::new(sampler_voice(rate, build_slot.clone()))
			}),
			bypass: Arc::new(AtomicBool::new(false)),
			scope: Arc::new(Scope::new()),
			slot,
			preview: Arc::new(Mutex::new(SamplePreview::default())),
		}
	}
}

impl SamplerWeb {
	/// The rate the instrument inside is currently built for.
	pub fn sample_rate(&self) -> usize {
		self.processor.sample_rate()
	}

	/// Whether the editor's bypass switch is on.
	pub fn is_bypassed(&self) -> bool {
		self.bypass.load(Ordering::Relaxed)
	}

	/// Turn the bypass on or off from code, for a test.
	pub fn set_bypassed(&self, bypassed: bool) {
		self.bypass.store(bypassed, Ordering::Relaxed);
	}

	/// The sample the page imported, if any.
	pub fn imported(&self) -> Option<SampleData> {
		self.slot
			.lock()
			.unwrap_or_else(|error| error.into_inner())
			.sample
			.as_ref()
			.map(|sample| (**sample).clone())
	}
}

impl Parameters for SamplerWeb {
	fn get_parameters(&self) -> Vec<Parameter> {
		self.processor.parameters()
	}

	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		self.processor.set_parameter(identifier, value)
	}
}

impl Processor for SamplerWeb {
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

impl Plugin for SamplerWeb {
	const DESCRIPTOR: Descriptor = Descriptor::new("iamdsp.synths.sampler.web", SAMPLER_TITLE)
		.with_vendor("iamplugins")
		.with_version(env!("CARGO_PKG_VERSION"))
		.with_description("A sampler with a slicer and a grain cloud around one sample.")
		.with_tags(&[Tag::Instrument, Tag::Synthesizer]);

	const INPUT_PORTS: &'static [i_am_plugin::AudioPort] = &[];
	const INPUT_MIDI_PORTS: &'static [MidiPort] = &[MidiPort { name: "Notes" }];
	const HAS_GUI: bool = true;

	fn new() -> Self {
		Self::default()
	}

	fn window_options() -> WindowOptions {
		const FACE_FRACTION: f64 = 0.85;

		WindowOptions::new()
			.with_size((
				SAMPLER_SIZE.0 as f64 / FACE_FRACTION,
				SAMPLER_SIZE.1 as f64 / FACE_FRACTION,
			))
			.with_title(SAMPLER_TITLE)
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

	fn editor(&self) -> Option<Box<dyn i_am_plugin::Editor<Self>>> {
		let slot = self.slot.clone();
		let preview = self.preview.clone();

		editor_with(
			self,
			sampler_faceplate(),
			self.bypass.clone(),
			self.scope.clone(),
			self.processor.sample_rate(),
			move |request: &ProtocolRequest<'_>| sample_route(request, &slot, &preview),
			|_: &str| false,
		)
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use hound::{WavSpec, WavWriter};

	fn wav(samples: &[f32], rate: u32) -> Vec<u8> {
		let spec = WavSpec {
			channels: 1,
			sample_rate: rate,
			bits_per_sample: 16,
			sample_format: SampleFormat::Int,
		};
		let mut cursor = Cursor::new(Vec::new());

		{
			let mut writer = WavWriter::new(&mut cursor, spec).expect("a writer");

			for sample in samples {
				writer
					.write_sample((sample.clamp(-1.0, 1.0) * 32767.0) as i16)
					.expect("a sample");
			}

			writer.finalize().expect("a finished file");
		}

		cursor.into_inner()
	}

	/// The route imports a WAV and describes it back to the page.
	#[test]
	fn the_route_imports_and_describes_a_sample() {
		let slot = Arc::new(Mutex::new(SampleSlot::default()));
		let preview = Arc::new(Mutex::new(SamplePreview::default()));
		let samples: Vec<f32> = (0..4_800)
			.map(|index| (index as f32 / 48_000.0 * 440.0 * std::f32::consts::TAU).sin())
			.collect();
		let body = wav(&samples, 48_000);
		let post = ProtocolRequest {
			method: "POST",
			path: "/sample",
			query: Some("name=test"),
			body: &body,
		};
		let response = sample_route(&post, &slot, &preview).expect("the route");
		let text = String::from_utf8(response.body).expect("utf-8");

		assert!(text.contains("\"name\":\"test\""), "{text}");
		assert!(text.contains("\"rate\":48000"), "{text}");
		assert!(text.contains("\"peaks\":["), "{text}");
		assert_eq!(
			slot.lock()
				.unwrap()
				.sample
				.as_ref()
				.map(|sample| sample.pcm.len()),
			Some(4_800)
		);

		let get = ProtocolRequest {
			method: "GET",
			path: "/sample",
			query: None,
			body: &[],
		};
		let response = sample_route(&get, &slot, &preview).expect("the route");

		assert!(
			String::from_utf8(response.body)
				.unwrap()
				.contains("\"name\":\"test\"")
		);
	}

	/// The route makes a grid of marks, and single marks can be added or removed.
	#[test]
	fn the_route_makes_and_edits_slice_marks() {
		let slot = Arc::new(Mutex::new(SampleSlot::default()));
		let preview = Arc::new(Mutex::new(SamplePreview::default()));
		let samples: Vec<f32> = (0..4_800)
			.map(|index| (index as f32 / 48_000.0 * 440.0 * std::f32::consts::TAU).sin())
			.collect();
		let body = wav(&samples, 48_000);
		let import = ProtocolRequest {
			method: "POST",
			path: "/sample",
			query: Some("name=marks"),
			body: &body,
		};
		sample_route(&import, &slot, &preview).expect("the import");

		let grid = ProtocolRequest {
			method: "POST",
			path: "/sample",
			query: Some("action=slices&mode=grid&start=0&step=0.25"),
			body: &[],
		};
		let response = sample_route(&grid, &slot, &preview).expect("the grid");
		let text = String::from_utf8(response.body).expect("utf-8");

		assert!(
			text.contains("\"slices\":[0.0000,0.2500,0.5000,0.7500]"),
			"{text}"
		);

		let add = ProtocolRequest {
			method: "POST",
			path: "/sample",
			query: Some("action=slice-add&at=0.3"),
			body: &[],
		};
		sample_route(&add, &slot, &preview).expect("the add");
		assert_eq!(slot.lock().unwrap().slices, vec![0.0, 0.25, 0.3, 0.5, 0.75]);

		let remove = ProtocolRequest {
			method: "POST",
			path: "/sample",
			query: Some("action=slice-remove&at=0.3"),
			body: &[],
		};
		let response = sample_route(&remove, &slot, &preview).expect("the remove");

		assert!(
			String::from_utf8(response.body)
				.unwrap()
				.contains("\"slices\":[0.0000,0.2500,0.5000,0.7500]")
		);
	}

	/// The sample is the voice's serialized parameter, so it travels with the
	/// project.
	#[test]
	fn the_sample_is_saved_with_the_state() {
		use i_am_dsp::prelude::{Value, to_binary};

		let mut plugin = SamplerWeb::new();
		let sample = SampleData {
			name: "saved".into(),
			rate: 48_000,
			pcm: vec![0.25; 128],
		};
		let binary = to_binary(&Some(sample)).expect("a sample serializes");

		assert!(plugin.set_parameter("oscillator.sample", SetValue::Serialized(binary.clone())));

		let parameter = plugin
			.get_parameters()
			.into_iter()
			.find(|parameter| parameter.identifier == "oscillator.sample")
			.expect("the serialized sample");
		let Value::Serialized(data) = parameter.value else {
			panic!("the sample is not serialized")
		};

		assert_eq!(data, binary);
		assert_eq!(
			plugin.imported().map(|sample| sample.name),
			Some("saved".into())
		);
	}
}
