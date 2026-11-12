//! The morphing wavetable instrument: four classic shapes, or a table the user
//! brought, plus the import that turns a WAV into band limited frames.

use std::io::Cursor;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use hound::{SampleFormat, WavReader};
use i_am_dsp::prelude::{
	Adsr, EqualTemperament, GeneratorEffect, ImportedWave, MAX_WAVE_FRAMES, ParamMap, PitchTracker,
	WAVE_FRAME_LEN, WaveBank, WaveFramesOsc, WaveSlot,
};
use i_am_plugin::processor::EffectProcessor;
use i_am_plugin::{Descriptor, MidiPort, Plugin, Tag, WindowOptions};
use i_am_web_gui::{
	Faceplate, Page, ProtocolRequest, ProtocolResponse, Scope, Switch, editor_with,
};

use super::json_escape;

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

/// Turns a WAV into a table and puts it in a slot, the way a drop on the
/// product's own page does.
pub fn import_waves(
	slot: &Arc<Mutex<WaveSlot>>,
	bytes: &[u8],
	name: &str,
) -> Result<(String, usize), String> {
	let wave = decode_table(bytes, name, SliceMode::parse(None))?;
	let bank = WaveBank::new(Some(name.to_string()), wave.frames);
	let count = bank.len();

	slot.lock().unwrap_or_else(|error| error.into_inner()).bank = Some(bank);

	Ok((name.to_string(), count))
}

/// What the page draws for a slot: one preview per frame, and the name.
pub fn waves_preview(slot: &Arc<Mutex<WaveSlot>>) -> String {
	let bank = slot
		.lock()
		.unwrap_or_else(|error| error.into_inner())
		.bank
		.clone();

	let cache = match &bank {
		Some(bank) => WavePreview {
			version: bank.id(),
			name: bank.name().map(str::to_string),
			count: bank.len(),
			previews: bank
				.frames()
				.iter()
				.map(|frame| preview_points(frame, PREVIEW_POINTS))
				.collect(),
			error: None,
		},
		None => WavePreview::default(),
	};

	preview_json(&cache)
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

mod import;
#[cfg(test)]
mod tests;

use import::*;

/// Decodes %20 style escapes; the page sends names with encodeURIComponent.
use i_am_web_gui::percent_decode;

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
		Self::with_slot(Arc::new(Mutex::new(WaveSlot::default())))
	}
}

impl ShapesWeb {
	/// The instrument over a slot the caller already holds.
	///
	/// The patcher keeps the slot itself, so a table dropped on a patch node
	/// reaches the voice the same way the product's own page does.
	pub fn with_slot(slot: Arc<Mutex<WaveSlot>>) -> Self {
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

	/// The slot the page writes a table into.
	pub fn slot(&self) -> Arc<Mutex<WaveSlot>> {
		self.slot.clone()
	}

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

synth_product!(ShapesWeb);

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
		i_am_web_gui::window_options((SHAPES_SIZE.0, SHAPES_SIZE.1), SHAPES_TITLE)
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
