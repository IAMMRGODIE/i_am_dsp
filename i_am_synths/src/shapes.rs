//! The morphing wavetable instrument: four classic shapes, or a table the user
//! brought, plus the import that turns a WAV into band limited frames.

use std::io::Cursor;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use hound::{SampleFormat, WavReader};
use i_am_dsp::prelude::{
	Adsr, EqualTemperament, GeneratorEffect, ImportedWave, MAX_WAVE_FRAMES, ParamMap, Parameter,
	Parameters, PitchTracker, SetValue, WAVE_FRAME_LEN, WaveBank, WaveFramesOsc, WaveSlot,
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

/// The shapes face's design size, in points.
const SHAPES_SIZE: (f32, f32) = (820.0, 420.0);
/// The shapes window's title and the face's title.
const SHAPES_TITLE: &str = "I Am Shapes";
/// The shapes identity colour, as #rrggbb.
const SHAPES_ACCENT: &str = "#4cc9f0";
/// How many points of a frame the page draws.
const PREVIEW_POINTS: usize = 96;

/// The shapes face's tabs: the sound, the envelope, and the voice's unisons.
const SHAPES_PAGES: &[Page] = &[
	Page::new("Shape", &["Shape"]),
	Page::new("Envelope", &["Envelope"]),
	Page::new("Voice", &["Voice"]),
];

/// The per-stage sync switches stay the host's: the face has one for the envelope.
const SHAPES_HIDDEN: &[&str] = &[
	"delay.sync",
	"attack.sync",
	"hold.sync",
	"decay.sync",
	"release.sync",
];

/// The envelope's one Sync switch trades the free lengths for musical ones.
const SHAPES_SWITCHES: &[Switch] = &[Switch::new(
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

/// The shapes voice: a band limited morphing wavetable under the envelope.
type ShapeVoice = Adsr<WaveFramesOsc, EqualTemperament, 2>;

fn shape_voice(rate: usize, slot: Arc<Mutex<WaveSlot>>) -> ShapeVoice {
	Adsr::new(WaveFramesOsc::with_slot(slot), EqualTemperament, rate)
}

/// What the page draws for the imported table: one line per frame, so the hero
/// can draw whichever pair the morph sits between.
#[derive(Clone, Default)]
struct WavePreview {
	version: u64,
	name: Option<String>,
	count: usize,
	previews: Vec<Vec<f32>>,
	error: Option<String>,
}

/// The faceplate of the shapes instrument.
pub fn shapes_faceplate() -> Faceplate {
	Faceplate::new(SHAPES_TITLE, SHAPES_ACCENT, SHAPES_SIZE)
		.with_pages(SHAPES_PAGES)
		.with_header(&["sync"])
		.with_switches(SHAPES_SWITCHES)
		.with_hidden(SHAPES_HIDDEN)
}

/// Answers the editor's table: a GET describes what is loaded, a POST turns a WAV
/// into band limited frames.
///
/// The frames go straight into the slot the voice reads, because a serialized
/// parameter cannot travel through the parameter map and the upload happens on
/// the UI thread. A save reads the same slot, so nothing is lost.
fn waves_route(
	request: &ProtocolRequest<'_>,
	slot: &Arc<Mutex<WaveSlot>>,
	preview: &Arc<Mutex<WavePreview>>,
) -> Option<ProtocolResponse> {
	if request.path != "/waves" {
		return None;
	}

	match request.method {
		"GET" => {
			let bank = slot
				.lock()
				.unwrap_or_else(|error| error.into_inner())
				.bank
				.clone();
			let version = bank.as_ref().map(|bank| bank.id()).unwrap_or(0);
			let since =
				query_value(request.query, "since").and_then(|since| since.parse::<u64>().ok());

			// The page keeps the frames it already has: only a new import, or a
			// load, makes it fetch them again.
			if since == Some(version) {
				return Some(ProtocolResponse::json(format!(
					"{{\"version\":{version},\"unchanged\":true}}"
				)));
			}

			let mut cache = preview.lock().unwrap_or_else(|error| error.into_inner());

			if cache.version != version {
				*cache = match &bank {
					Some(bank) => WavePreview {
						version,
						name: bank.name().map(str::to_string),
						count: bank.len(),
						previews: bank
							.frames()
							.iter()
							.map(|frame| preview_points(frame, PREVIEW_POINTS))
							.collect(),
						error: None,
					},
					None => WavePreview {
						version: 0,
						..WavePreview::default()
					},
				};
			}

			Some(ProtocolResponse::json(preview_json(&cache)))
		}
		"POST" => {
			if let Some(action) = query_value(request.query, "action") {
				return Some(edit_table(request.query, action, slot, preview));
			}

			let name = query_value(request.query, "name")
				.map(percent_decode)
				.filter(|name| !name.is_empty())
				.unwrap_or_else(|| "table".to_string());
			let mode = SliceMode::parse(query_value(request.query, "mode"));

			match decode_table(request.body, &name, mode) {
				Ok(wave) => Some(store_bank(
					slot,
					preview,
					WaveBank::new(Some(name), wave.frames),
				)),
				Err(error) => {
					*preview.lock().unwrap_or_else(|error| error.into_inner()) = WavePreview {
						error: Some(error.clone()),
						..WavePreview::default()
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

/// Puts a rebuilt table in the slot and describes it back to the page.
fn store_bank(
	slot: &Arc<Mutex<WaveSlot>>,
	preview: &Arc<Mutex<WavePreview>>,
	bank: Arc<WaveBank>,
) -> ProtocolResponse {
	let version = bank.id();
	let mut cache = preview.lock().unwrap_or_else(|error| error.into_inner());

	*cache = WavePreview {
		version,
		name: bank.name().map(str::to_string),
		count: bank.len(),
		previews: bank
			.frames()
			.iter()
			.map(|frame| preview_points(frame, PREVIEW_POINTS))
			.collect(),
		error: None,
	};
	slot.lock().unwrap_or_else(|error| error.into_inner()).bank = Some(bank);

	ProtocolResponse::json(preview_json(&cache))
}

/// Subtracts each frame's mean, which is the DC offset a cut from a recording
/// usually carries.
fn remove_dc(frames: &mut [Vec<f32>]) {
	for frame in frames.iter_mut() {
		if frame.is_empty() {
			continue;
		}

		let mean = frame.iter().sum::<f32>() / frame.len() as f32;

		for value in frame.iter_mut() {
			*value -= mean;
		}
	}
}

/// Scales every frame to the same peak, so a table whose frames were cut from
/// different parts of a recording does not fade in and out under Morph.
fn normalize_frames(frames: &mut [Vec<f32>]) {
	for frame in frames.iter_mut() {
		let peak = frame
			.iter()
			.fold(0.0_f32, |peak, value| peak.max(value.abs()));

		if peak > 1e-6 {
			let gain = 0.99 / peak;

			for value in frame.iter_mut() {
				*value *= gain;
			}
		}
	}
}

/// Edits the loaded table: reverse plays every frame backwards, invert flips the
/// polarity, delete removes one frame, dc and normalize even the frames out.
fn edit_table(
	query: Option<&str>,
	action: &str,
	slot: &Arc<Mutex<WaveSlot>>,
	preview: &Arc<Mutex<WavePreview>>,
) -> ProtocolResponse {
	let bank = slot
		.lock()
		.unwrap_or_else(|error| error.into_inner())
		.bank
		.clone();
	let Some(bank) = bank else {
		return ProtocolResponse::json("{\"ok\":false,\"error\":\"no table\"}");
	};

	let mut frames = bank.frames().to_vec();

	match action {
		"reverse" => {
			for frame in frames.iter_mut() {
				frame.reverse();
			}
		}
		"invert" => {
			for frame in frames.iter_mut() {
				for value in frame.iter_mut() {
					*value = -*value;
				}
			}
		}
		"dc" => remove_dc(&mut frames),
		"normalize" => normalize_frames(&mut frames),
		"delete" => {
			if frames.len() <= 1 {
				return ProtocolResponse::json("{\"ok\":false,\"error\":\"the last frame stays\"}");
			}

			let index = query_value(query, "frame")
				.and_then(|frame| frame.parse::<usize>().ok())
				.unwrap_or(0)
				.min(frames.len() - 1);

			frames.remove(index);
		}
		_ => return ProtocolResponse::json("{\"ok\":false,\"error\":\"unknown action\"}"),
	}

	store_bank(
		slot,
		preview,
		WaveBank::new(bank.name().map(str::to_string), frames),
	)
}

fn preview_json(preview: &WavePreview) -> String {
	let name = match &preview.name {
		Some(name) => format!("\"{}\"", json_escape(name)),
		None => "null".to_string(),
	};
	let error = match &preview.error {
		Some(error) => format!("\"{}\"", json_escape(error)),
		None => "null".to_string(),
	};
	let previews: Vec<String> = preview
		.previews
		.iter()
		.map(|points| {
			let points: Vec<String> = points.iter().map(|value| format!("{value:.4}")).collect();

			format!("[{}]", points.join(","))
		})
		.collect();

	format!(
		"{{\"version\":{},\"name\":{name},\"count\":{},\"error\":{error},\"previews\":[{}]}}",
		preview.version,
		preview.count,
		previews.join(",")
	)
}

/// One period of a frame, downsampled.
fn preview_points(frame: &[f32], count: usize) -> Vec<f32> {
	if frame.is_empty() || count == 0 {
		return Vec::new();
	}

	(0..count)
		.map(|index| {
			let at = (index as f32 / count as f32 * frame.len() as f32) as usize % frame.len();

			frame[at]
		})
		.collect()
}

/// How an imported file is cut into frames.
#[derive(Clone, Copy, PartialEq)]
enum SliceMode {
	/// Decide from the file: metadata, whole frames, one cycle, or pitch.
	Auto,
	/// Cut at a fixed frame size: the metadata's, or 2048.
	Fixed,
	/// Find the recording's pitch and cut at its cycle.
	Pitch,
	/// The whole file is one cycle.
	Single,
}

impl SliceMode {
	fn parse(value: Option<&str>) -> Self {
		match value.unwrap_or("") {
			"fixed" => Self::Fixed,
			"pitch" => Self::Pitch,
			"single" => Self::Single,
			_ => Self::Auto,
		}
	}
}

/// Reads a WAV into frames: the frame size comes from the file's metadata when
/// it has any, and otherwise from the audio itself.
fn decode_table(bytes: &[u8], name: &str, mode: SliceMode) -> Result<ImportedWave, String> {
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

	let mono: Vec<f32> = if channels == 1 {
		interleaved
	} else {
		interleaved
			.chunks(channels)
			.map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
			.collect()
	};

	let frames = slice_table(
		&mono,
		spec.sample_rate as usize,
		serum_frame_len(bytes),
		mode,
	);

	if frames.is_empty() {
		return Err("the file holds no cycles".to_string());
	}

	// Peak normalize the whole table, so the frames keep their relative levels.
	let peak = frames
		.iter()
		.flat_map(|frame| frame.iter())
		.fold(0.0_f32, |peak, sample| peak.max(sample.abs()));
	let gain = if peak > 1e-6 { 0.99 / peak } else { 1.0 };

	Ok(ImportedWave::new(
		name,
		frames
			.into_iter()
			.map(|frame| frame.into_iter().map(|sample| sample * gain).collect())
			.collect(),
	))
}

/// Cuts the samples into frames and resamples each one to [WAVE_FRAME_LEN].
fn slice_table(
	samples: &[f32],
	sample_rate: usize,
	metadata: Option<usize>,
	mode: SliceMode,
) -> Vec<Vec<f32>> {
	if samples.is_empty() {
		return Vec::new();
	}

	let mut raw: Vec<&[f32]> = Vec::new();

	match mode {
		SliceMode::Single => raw.push(samples),
		SliceMode::Fixed => {
			let frame_len = metadata.unwrap_or(WAVE_FRAME_LEN);

			raw = samples.chunks(frame_len).collect();
		}
		SliceMode::Pitch => {
			raw = match detect_cycle(samples, sample_rate) {
				Some(cycle) => samples.chunks(cycle).collect(),
				None => samples.chunks(WAVE_FRAME_LEN).collect(),
			}
		}
		SliceMode::Auto => {
			if let Some(frame_len) = metadata {
				raw = samples.chunks(frame_len).collect();
			}

			if raw.is_empty() {
				let single_cycle = samples.len() < WAVE_FRAME_LEN + WAVE_FRAME_LEN / 2;

				if single_cycle {
					// A file shorter than a frame and a half is one cycle: this is
					// the Adventure Kid case, which a fixed 2048 cut would ruin.
					raw.push(samples);
				} else if samples.len().is_multiple_of(WAVE_FRAME_LEN) {
					raw = samples.chunks(WAVE_FRAME_LEN).collect();
				} else if let Some(cycle) = detect_cycle(samples, sample_rate) {
					// The file is longer than a table and not a whole number of
					// frames: find its pitch and cut at the cycle, which is what
					// Serum and Vital call a pitch splice.
					raw = samples.chunks(cycle).collect();
				} else {
					raw = samples.chunks(WAVE_FRAME_LEN).collect();
				}
			}
		}
	}

	// A trailing chunk shorter than half a frame is a cut cycle, not a frame.
	let frame_hint = raw.first().map(|frame| frame.len()).unwrap_or(0);

	if raw.len() > 1
		&& raw
			.last()
			.map(|frame| frame.len() * 2 < frame_hint)
			.unwrap_or(false)
	{
		raw.pop();
	}

	let mut frames: Vec<Vec<f32>> = raw
		.into_iter()
		.map(|frame| resample(frame, WAVE_FRAME_LEN))
		.collect();

	if frames.len() > MAX_WAVE_FRAMES {
		let step = frames.len() as f32 / MAX_WAVE_FRAMES as f32;

		frames = (0..MAX_WAVE_FRAMES)
			.map(|index| frames[(index as f32 * step) as usize].clone())
			.collect();
	}

	frames
}

/// The cycle length a recording's pitch implies, or nothing when the recording
/// has no stable pitch.
///
/// A complex or wandering source (a chord, noise, a drum loop) has no single
/// cycle length; cutting it at one would make hundreds of incommensurate
/// fragments, so the fixed cut is the honest answer there. The estimate is the
/// median over a long analysis window, and the estimates have to agree: one
/// stray reading must not decide the cut.
fn detect_cycle(samples: &[f32], sample_rate: usize) -> Option<usize> {
	// Only the head of a long file: a table source is short, and the analysis is
	// the expensive part of an import.
	let analysis = &samples[..samples.len().min(sample_rate.saturating_mul(4))];
	let mut tracker = PitchTracker::new(sample_rate, 4_096);
	let mut estimates = Vec::new();

	for sample in analysis {
		if let Some(estimate) = tracker.process_sample(*sample)
			&& estimate.clarity >= 0.5
			&& estimate.frequency.is_finite()
			&& estimate.frequency > 20.0
		{
			estimates.push(estimate.frequency);
		}
	}

	if estimates.is_empty() {
		return None;
	}

	let pitch = median(&estimates);

	// A source whose pitch moves has no one cycle length.
	if estimates.len() >= 3 {
		let deviations: Vec<f32> = estimates
			.iter()
			.map(|frequency| (frequency - pitch).abs())
			.collect();

		if median(&deviations) / pitch > 0.05 {
			return None;
		}
	}

	let cycle = (sample_rate as f32 / pitch).round() as usize;

	// A cycle of a few dozen samples resamples into a smear; the fixed cut does
	// better on anything that high.
	if cycle < 64 || cycle > samples.len() / 2 {
		return None;
	}

	Some(cycle)
}

fn median(values: &[f32]) -> f32 {
	let mut sorted = values.to_vec();

	sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
	sorted[sorted.len() / 2]
}

/// Reads a frame at a new length with linear interpolation.
fn resample(frame: &[f32], len: usize) -> Vec<f32> {
	let mut output = vec![0.0; len];

	if frame.is_empty() || len == 0 {
		return output;
	}

	if frame.len() == 1 {
		output.fill(frame[0]);
		return output;
	}

	for (index, value) in output.iter_mut().enumerate() {
		let position = index as f32 / len as f32 * frame.len() as f32;
		let start = position.floor() as usize;
		let blend = position - start as f32;
		let before = frame[start.min(frame.len() - 1)];
		let after = frame[(start + 1).min(frame.len() - 1)];

		*value = before + (after - before) * blend;
	}

	output
}

/// The frame size in a Serum style file: its clm chunk records the number of
/// samples in a frame. A plain WAV has none, and the audio decides instead.
fn serum_frame_len(bytes: &[u8]) -> Option<usize> {
	if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
		return None;
	}

	let mut at = 12;

	while at + 8 <= bytes.len() {
		let id = &bytes[at..at + 4];
		let size = u32::from_le_bytes([bytes[at + 4], bytes[at + 5], bytes[at + 6], bytes[at + 7]])
			as usize;
		let data = at + 8;

		if id == b"clm " && size >= 4 && data + 4 <= bytes.len() {
			let frame_len = u32::from_le_bytes([
				bytes[data],
				bytes[data + 1],
				bytes[data + 2],
				bytes[data + 3],
			]) as usize;

			if (64..=8192).contains(&frame_len) {
				return Some(frame_len);
			}
		}

		at = data.saturating_add(size).saturating_add(size & 1);

		if at == data {
			break;
		}
	}

	None
}

/// One value out of a query string like name=a%20b&mode=pitch.
fn query_value<'a>(query: Option<&'a str>, key: &str) -> Option<&'a str> {
	query?.split('&').find_map(|part| {
		let (name, value) = part.split_once('=')?;

		(name == key).then_some(value)
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

/// The wavetable shapes instrument, with a web faceplate.
pub struct ShapesWeb {
	processor: EffectProcessor<GeneratorEffect<ShapeVoice>>,
	bypass: Arc<AtomicBool>,
	scope: Arc<Scope>,
	slot: Arc<Mutex<WaveSlot>>,
	preview: Arc<Mutex<WavePreview>>,
}

impl Default for ShapesWeb {
	fn default() -> Self {
		let slot = Arc::new(Mutex::new(WaveSlot::default()));
		let build_slot = slot.clone();

		Self {
			processor: EffectProcessor::with_build(move |rate: usize| {
				GeneratorEffect::new(shape_voice(rate, build_slot.clone()))
			}),
			bypass: Arc::new(AtomicBool::new(false)),
			scope: Arc::new(Scope::new()),
			slot,
			preview: Arc::new(Mutex::new(WavePreview::default())),
		}
	}
}

impl ShapesWeb {
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

	/// The table the page imported, if any.
	pub fn imported(&self) -> Option<ImportedWave> {
		self.slot
			.lock()
			.unwrap_or_else(|error| error.into_inner())
			.bank
			.as_ref()
			.map(|bank| bank.imported())
	}
}

impl Parameters for ShapesWeb {
	fn get_parameters(&self) -> Vec<Parameter> {
		self.processor.parameters()
	}

	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		self.processor.set_parameter(identifier, value)
	}
}

impl Processor for ShapesWeb {
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

impl Plugin for ShapesWeb {
	const DESCRIPTOR: Descriptor = Descriptor::new("iamdsp.synths.shapes.web", SHAPES_TITLE)
		.with_vendor("iamplugins")
		.with_version(env!("CARGO_PKG_VERSION"))
		.with_description(
			"An ADSR around a morphing wavetable: four classic shapes, or a table the user brings.",
		)
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
				SHAPES_SIZE.0 as f64 / FACE_FRACTION,
				SHAPES_SIZE.1 as f64 / FACE_FRACTION,
			))
			.with_title(SHAPES_TITLE)
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
			shapes_faceplate(),
			self.bypass.clone(),
			self.scope.clone(),
			self.processor.sample_rate(),
			move |request: &ProtocolRequest<'_>| waves_route(request, &slot, &preview),
			|_: &str| false,
		)
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use hound::{WavSpec, WavWriter};
	use i_am_dsp::prelude::{Value, to_binary};

	/// A 16 bit mono WAV of the given samples, for the import tests.
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

	/// A file of whole frames keeps them: 4096 samples are two frames of 2048.
	#[test]
	fn a_table_of_frames_keeps_them() {
		let mut samples = Vec::new();

		for index in 0..4096 {
			samples.push((index as f32 / 2048.0 * std::f32::consts::TAU).sin());
		}

		let table = decode_table(&wav(&samples, 44_100), "two", SliceMode::Auto).expect("a table");

		assert_eq!(table.frames(), 2);
		assert!(
			table
				.frames
				.iter()
				.all(|frame| frame.len() == WAVE_FRAME_LEN)
		);
	}

	/// A short file is one cycle, not a broken frame.
	#[test]
	fn a_short_file_is_one_cycle() {
		let samples: Vec<f32> = (0..1000)
			.map(|index| (index as f32 / 1000.0 * std::f32::consts::TAU).sin())
			.collect();
		let table =
			decode_table(&wav(&samples, 44_100), "cycle", SliceMode::Auto).expect("a table");

		assert_eq!(table.frames(), 1);
		assert_eq!(table.frames[0].len(), WAVE_FRAME_LEN);
	}

	/// A recording that is not a whole number of frames is cut at its pitch, the
	/// pitch splice Serum and Vital offer.
	#[test]
	fn a_recording_is_cut_at_its_pitch() {
		let samples: Vec<f32> = (0..12_000)
			.map(|index| (index as f32 / 100.0 * std::f32::consts::TAU).sin())
			.collect();
		let frames = slice_table(&samples, 44_100, None, SliceMode::Auto);

		assert!(frames.len() > 20, "{} frames", frames.len());
		assert!(frames.iter().all(|frame| frame.len() == WAVE_FRAME_LEN));
	}

	/// A source whose pitch moves has no one cycle: the import falls back to the
	/// fixed cut instead of shattering it into stray fragments.
	#[test]
	fn an_unsteady_recording_falls_back_to_fixed_frames() {
		let rate = 44_100;
		let samples: Vec<f32> = (0..rate / 2)
			.map(|index| {
				let time = index as f32 / rate as f32;
				let phase = 2.0 * std::f32::consts::PI * (200.0 * time + 600.0 * time * time);

				phase.sin()
			})
			.collect();

		assert_eq!(detect_cycle(&samples, rate), None);

		let frames = slice_table(&samples, rate, None, SliceMode::Pitch);

		assert!(!frames.is_empty());
		assert!(frames.iter().all(|frame| frame.len() == WAVE_FRAME_LEN));
	}

	/// The import can be told how to cut instead of guessing.
	#[test]
	fn the_import_modes_cut_differently() {
		let samples: Vec<f32> = (0..5000)
			.map(|index| ((index as f32 * 12.9898).sin() * 43_758.55).fract())
			.collect();

		let fixed = slice_table(&samples, 44_100, None, SliceMode::Fixed);
		assert_eq!(fixed.len(), 2);

		let single = slice_table(&samples, 44_100, None, SliceMode::Single);
		assert_eq!(single.len(), 1);
	}

	/// The clm chunk Serum writes says how long a frame is.
	#[test]
	fn the_clm_chunk_sets_the_frame_size() {
		let mut bytes = Vec::new();
		bytes.extend_from_slice(b"RIFF");
		bytes.extend_from_slice(&0u32.to_le_bytes());
		bytes.extend_from_slice(b"WAVE");
		bytes.extend_from_slice(b"clm ");
		bytes.extend_from_slice(&4u32.to_le_bytes());
		bytes.extend_from_slice(&1024u32.to_le_bytes());

		assert_eq!(serum_frame_len(&bytes), Some(1024));
	}

	/// The table can be flipped: the route rebuilds it and describes the new one.
	#[test]
	fn the_route_flips_a_table() {
		let slot = Arc::new(Mutex::new(WaveSlot::default()));
		let preview = Arc::new(Mutex::new(WavePreview::default()));
		let samples: Vec<f32> = (0..WAVE_FRAME_LEN)
			.map(|index| index as f32 / WAVE_FRAME_LEN as f32 - 0.5)
			.collect();
		let body = wav(&samples, 44_100);

		let post = ProtocolRequest {
			method: "POST",
			path: "/waves",
			query: Some("name=ramp"),
			body: &body,
		};
		waves_route(&post, &slot, &preview).expect("the import");

		let first = preview.lock().unwrap().previews[0][0];
		assert!(first < -0.4, "{first}");

		let reverse = ProtocolRequest {
			method: "POST",
			path: "/waves",
			query: Some("action=reverse"),
			body: &[],
		};
		let response = waves_route(&reverse, &slot, &preview).expect("the reverse");

		assert!(
			String::from_utf8(response.body)
				.unwrap()
				.contains("\"version\""),
			"the route did not describe the flipped table"
		);

		let reversed = preview.lock().unwrap().previews[0][0];
		assert!(reversed > 0.4, "{reversed}");

		let invert = ProtocolRequest {
			method: "POST",
			path: "/waves",
			query: Some("action=invert"),
			body: &[],
		};
		waves_route(&invert, &slot, &preview).expect("the invert");

		let inverted = preview.lock().unwrap().previews[0][0];
		assert!(inverted < -0.4, "{inverted}");
	}

	/// DC and normalize even the frames out: one click zeroes each mean, the other
	/// brings every frame to the same peak.
	#[test]
	fn the_route_removes_dc_and_normalizes_frames() {
		let slot = Arc::new(Mutex::new(WaveSlot::default()));
		let preview = Arc::new(Mutex::new(WavePreview::default()));
		let mut samples = Vec::new();

		for frame in 0..2 {
			let (offset, level) = if frame == 0 { (0.5, 0.4) } else { (0.2, 0.1) };

			for index in 0..WAVE_FRAME_LEN {
				samples.push(
					offset
						+ level
							* (index as f32 / WAVE_FRAME_LEN as f32 * std::f32::consts::TAU).sin(),
				);
			}
		}

		let body = wav(&samples, 44_100);
		let post = ProtocolRequest {
			method: "POST",
			path: "/waves",
			query: Some("name=dc"),
			body: &body,
		};
		waves_route(&post, &slot, &preview).expect("the import");

		let mean = |frame: &[f32]| frame.iter().sum::<f32>() / frame.len() as f32;
		let frames = slot
			.lock()
			.unwrap()
			.bank
			.as_ref()
			.unwrap()
			.frames()
			.to_vec();

		assert!(mean(&frames[0]).abs() > 0.2);

		let dc = ProtocolRequest {
			method: "POST",
			path: "/waves",
			query: Some("action=dc"),
			body: &[],
		};
		waves_route(&dc, &slot, &preview).expect("the dc removal");

		let frames = slot
			.lock()
			.unwrap()
			.bank
			.as_ref()
			.unwrap()
			.frames()
			.to_vec();

		assert!(frames.iter().all(|frame| mean(frame).abs() < 1e-4));

		let normalize = ProtocolRequest {
			method: "POST",
			path: "/waves",
			query: Some("action=normalize"),
			body: &[],
		};
		waves_route(&normalize, &slot, &preview).expect("the normalize");

		let frames = slot
			.lock()
			.unwrap()
			.bank
			.as_ref()
			.unwrap()
			.frames()
			.to_vec();

		for frame in &frames {
			let peak = frame
				.iter()
				.fold(0.0_f32, |peak, value| peak.max(value.abs()));

			assert!((peak - 0.99).abs() < 2e-3, "{peak}");
		}
	}

	/// The route takes the slicing mode from the query string.
	#[test]
	fn the_route_takes_the_slice_mode() {
		let slot = Arc::new(Mutex::new(WaveSlot::default()));
		let preview = Arc::new(Mutex::new(WavePreview::default()));
		let samples: Vec<f32> = (0..5000)
			.map(|index| ((index as f32 * 12.9898).sin() * 43_758.55).fract())
			.collect();
		let body = wav(&samples, 44_100);
		let post = ProtocolRequest {
			method: "POST",
			path: "/waves",
			query: Some("name=odd&mode=fixed"),
			body: &body,
		};
		waves_route(&post, &slot, &preview).expect("the import");

		assert_eq!(preview.lock().unwrap().count, 2);
		assert_eq!(preview.lock().unwrap().name.as_deref(), Some("odd"));
	}

	/// Deleting a frame shortens the table; the last frame cannot be deleted.
	#[test]
	fn the_route_deletes_a_frame() {
		let slot = Arc::new(Mutex::new(WaveSlot::default()));
		let preview = Arc::new(Mutex::new(WavePreview::default()));
		let samples: Vec<f32> = (0..4096)
			.map(|index| (index as f32 / 2048.0 * std::f32::consts::TAU).sin())
			.collect();
		let body = wav(&samples, 44_100);
		let post = ProtocolRequest {
			method: "POST",
			path: "/waves",
			query: Some("name=two"),
			body: &body,
		};
		waves_route(&post, &slot, &preview).expect("the import");
		assert_eq!(preview.lock().unwrap().count, 2);

		let delete = ProtocolRequest {
			method: "POST",
			path: "/waves",
			query: Some("action=delete&frame=0"),
			body: &[],
		};
		waves_route(&delete, &slot, &preview).expect("the delete");
		assert_eq!(preview.lock().unwrap().count, 1);

		// The last frame stays: there is nothing to morph across without it.
		let response = waves_route(&delete, &slot, &preview).expect("the second delete");
		let text = String::from_utf8(response.body).expect("utf-8");

		assert!(text.contains("the last frame stays"), "{text}");
		assert_eq!(preview.lock().unwrap().count, 1);
	}

	/// The imported table travels as the voice's serialized parameter, so it
	/// survives a save and a load.
	#[test]
	fn an_imported_table_is_saved_with_the_state() {
		let mut plugin = ShapesWeb::new();
		let wave = ImportedWave::new("saved", vec![vec![0.2; WAVE_FRAME_LEN]]);
		let binary = to_binary(&Some(wave)).expect("a table serializes");

		assert!(plugin.set_parameter("oscillator.imported", SetValue::Serialized(binary.clone())));

		let parameter = plugin
			.get_parameters()
			.into_iter()
			.find(|parameter| parameter.identifier == "oscillator.imported")
			.expect("the serialized table");
		let Value::Serialized(data) = parameter.value else {
			panic!("the table is not serialized")
		};

		assert_eq!(data, binary);
		assert_eq!(
			plugin.imported().map(|wave| wave.name),
			Some("saved".into())
		);
	}

	/// The route stores what it decodes and describes it back, which is what the
	/// page draws and what a save picks up.
	#[test]
	fn the_route_imports_and_describes_a_table() {
		let slot = Arc::new(Mutex::new(WaveSlot::default()));
		let preview = Arc::new(Mutex::new(WavePreview::default()));
		let samples: Vec<f32> = (0..4096)
			.map(|index| (index as f32 / 2048.0 * std::f32::consts::TAU).sin())
			.collect();
		let body = wav(&samples, 44_100);
		let post = ProtocolRequest {
			method: "POST",
			path: "/waves",
			query: Some("name=two"),
			body: &body,
		};

		let response = waves_route(&post, &slot, &preview).expect("the route");
		let text = String::from_utf8(response.body).expect("utf-8");

		assert!(text.contains("\"name\":\"two\""), "{text}");
		assert!(text.contains("\"count\":2"), "{text}");
		assert!(text.contains("\"previews\":[["), "{text}");
		assert!(text.contains("\"version\":"), "{text}");
		assert_eq!(
			slot.lock().unwrap().bank.as_ref().map(|bank| bank.len()),
			Some(2)
		);

		// The page asks again with the version it holds, and gets a short answer.
		let version = slot
			.lock()
			.unwrap()
			.bank
			.as_ref()
			.map(|bank| bank.id())
			.unwrap();
		let get = ProtocolRequest {
			method: "GET",
			path: "/waves",
			query: Some("since="),
			body: &[],
		};

		assert!(waves_route(&get, &slot, &preview).is_some());

		let query = format!("since={version}");
		let get = ProtocolRequest {
			method: "GET",
			path: "/waves",
			query: Some(&query),
			body: &[],
		};
		let response = waves_route(&get, &slot, &preview).expect("the route");
		let text = String::from_utf8(response.body).expect("utf-8");

		assert!(text.contains("\"unchanged\":true"), "{text}");
	}
}
