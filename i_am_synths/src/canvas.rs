//! The instrument whose waveform is drawn on a stereo canvas.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use i_am_dsp::prelude::{
	Adsr, EqualTemperament, Generator, GeneratorEffect, MultiStereoGenerators, Node, ParamMap,
	Parameter, Parameters, SetValue, StereoGenerator,
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

/// The canvas face's design size, in points.
const CANVAS_SIZE: (f32, f32) = (820.0, 420.0);
/// The canvas window's title and the face's title.
const CANVAS_TITLE: &str = "I Am Canvas";
/// The canvas identity colour, as #rrggbb.
const CANVAS_ACCENT: &str = "#06d6a0";

/// The canvas face's tabs.
const CANVAS_PAGES: &[Page] = &[
	Page::new("Canvas", &["Shape"]),
	Page::new("Envelope", &["Envelope"]),
	Page::new("Voice", &["Voice"]),
];

/// The per-stage sync switches stay the host's: the face has one for the envelope.
const CANVAS_HIDDEN: &[&str] = &[
	"delay.sync",
	"attack.sync",
	"hold.sync",
	"decay.sync",
	"release.sync",
];

/// The envelope's one Sync switch trades the free lengths for musical ones.
const CANVAS_SWITCHES: &[Switch] = &[Switch::new(
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

/// How many points one drawn page is sampled into.
const CANVAS_CACHE: usize = 1024;

/// One circle in the page's token form: start, node count, then six numbers per
/// node (two controls and the end point).
fn circle_tokens(radius: f32) -> String {
	let k = radius * 0.552_284_8;
	let nodes = [
		[radius, k, k, radius, 0.0, radius],
		[-k, radius, -radius, k, -radius, 0.0],
		[-radius, -k, -k, -radius, 0.0, -radius],
		[k, -radius, radius, -k, radius, 0.0],
	];
	let mut text = format!("{radius} 0 4");

	for node in nodes {
		for value in node {
			text.push_str(&format!(" {value}"));
		}
	}

	text
}

/// The two curves the face starts with: a wide stereo circle and a narrow one,
/// so the Morph control has something to move between.
pub(crate) fn default_canvas_text() -> String {
	format!("2 {} {}", circle_tokens(0.8), circle_tokens(0.35))
}

/// Parses the page's token form into generators with their caches built.
pub(crate) fn parse_canvas(text: &str) -> Result<Vec<StereoGenerator>, String> {
	let mut tokens = text.split_whitespace();

	let mut next = |name: &str| -> Result<f32, String> {
		tokens
			.next()
			.ok_or_else(|| format!("missing {name}"))
			.and_then(|token| {
				token
					.parse::<f32>()
					.map_err(|_| format!("{token} is not a number"))
			})
	};

	let page_count = next("page count")? as usize;

	if page_count == 0 {
		return Err("a canvas needs at least one page".to_string());
	}

	// No cap on either count: the text itself bounds them, because a count
	// larger than the numbers after it fails on the first missing token.
	// Vec::new rather than with_capacity keeps a bogus count from asking for
	// its memory.
	let mut pages = Vec::new();

	for _ in 0..page_count {
		let start = (next("start x")?, next("start y")?);
		let count = next("node count")? as i32;

		if count < 1 {
			return Err("a page needs at least one node".to_string());
		}

		let mut graph = Vec::new();

		for _ in 0..count {
			graph.push(Node {
				ctrl1: (next("control 1 x")?, next("control 1 y")?),
				ctrl2: (next("control 2 x")?, next("control 2 y")?),
				to: (next("node x")?, next("node y")?),
			});
		}

		let mut generator = StereoGenerator::new();
		generator.start = start;
		generator.graph = graph;
		generator.update_cache(CANVAS_CACHE);
		pages.push(generator);
	}

	Ok(pages)
}

/// The state the page and the audio thread share.
#[derive(Default)]
pub(crate) struct CanvasState {
	text: String,
	error: Option<String>,
	pending: Option<Vec<StereoGenerator>>,
}

/// The page's drawing and the audio thread's copy of it.
pub(crate) struct CanvasFeed {
	generation: AtomicU64,
	state: Mutex<CanvasState>,
}

impl CanvasFeed {
	pub(crate) fn new() -> Arc<Self> {
		Arc::new(Self {
			generation: AtomicU64::new(0),
			state: Mutex::new(CanvasState {
				text: default_canvas_text(),
				..CanvasState::default()
			}),
		})
	}

	pub(crate) fn take(&self, seen: &mut u64) -> Option<Vec<StereoGenerator>> {
		let generation = self.generation.load(Ordering::Acquire);

		if generation == *seen {
			return None;
		}

		*seen = generation;
		self.state
			.lock()
			.unwrap_or_else(|error| error.into_inner())
			.pending
			.take()
	}

	fn text(&self) -> String {
		self.state
			.lock()
			.unwrap_or_else(|error| error.into_inner())
			.text
			.clone()
	}

	fn apply(&self, text: String) -> String {
		match parse_canvas(&text) {
			Ok(pages) => {
				{
					let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
					state.text = text;
					state.error = None;
					state.pending = Some(pages);
				}

				self.generation.fetch_add(1, Ordering::Release);
				"{\"ok\":true}".to_string()
			}
			Err(error) => {
				let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
				state.error = Some(error.clone());

				format!("{{\"ok\":false,\"error\":\"{}\"}}", json_escape(&error))
			}
		}
	}
}

pub(crate) fn canvas_route(
	request: &ProtocolRequest<'_>,
	feed: &Arc<CanvasFeed>,
) -> Option<ProtocolResponse> {
	if request.method == "GET" && request.path == "/canvas" {
		return Some(ProtocolResponse::json(format!(
			"{{\"shape\":\"{}\"}}",
			json_escape(&feed.text())
		)));
	}

	if request.method == "POST" && request.path == "/canvas" {
		let text = String::from_utf8_lossy(request.body).to_string();
		return Some(ProtocolResponse::json(feed.apply(text)));
	}

	None
}

/// The canvas voice: the drawn curves and the morph between them.
type CanvasVoiceInner = Adsr<MultiStereoGenerators, EqualTemperament, 2>;

struct CanvasVoice {
	voice: CanvasVoiceInner,
	feed: Arc<CanvasFeed>,
	generation: u64,
}

impl CanvasVoice {
	fn new(rate: usize, feed: Arc<CanvasFeed>) -> Self {
		let mut multi = MultiStereoGenerators::new();
		multi.generators = parse_canvas(&default_canvas_text()).unwrap_or_default();
		multi.smooth_factor = 0.0;

		Self {
			voice: Adsr::new(multi, EqualTemperament, rate),
			feed,
			generation: 0,
		}
	}
}

impl Parameters for CanvasVoice {
	fn get_parameters(&self) -> Vec<Parameter> {
		self.voice.get_parameters()
	}

	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		self.voice.set_parameter(identifier, value)
	}
}

impl Generator<2> for CanvasVoice {
	fn generate(&mut self, process_context: &mut Box<dyn ProcessContext>) -> [f32; 2] {
		if let Some(pages) = self.feed.take(&mut self.generation) {
			self.voice.oscillator.generators = pages;
		}

		self.voice.generate(process_context)
	}
}

/// The default drawing, for a preview that has no host to ask.
pub fn canvas_shape() -> String {
	default_canvas_text()
}

/// The faceplate of the canvas instrument.
pub fn canvas_faceplate() -> Faceplate {
	Faceplate::new(CANVAS_TITLE, CANVAS_ACCENT, CANVAS_SIZE)
		.with_pages(CANVAS_PAGES)
		.with_header(&["sync"])
		.with_switches(CANVAS_SWITCHES)
		.with_hidden(CANVAS_HIDDEN)
}

/// The drawn stereo instrument, with a web faceplate.
pub struct CanvasWeb {
	processor: EffectProcessor<GeneratorEffect<CanvasVoice>>,
	bypass: Arc<AtomicBool>,
	scope: Arc<Scope>,
	feed: Arc<CanvasFeed>,
}

impl Default for CanvasWeb {
	fn default() -> Self {
		let feed = CanvasFeed::new();
		let build_feed = feed.clone();

		Self {
			processor: EffectProcessor::with_build(move |rate: usize| {
				GeneratorEffect::new(CanvasVoice::new(rate, build_feed.clone()))
			}),
			bypass: Arc::new(AtomicBool::new(false)),
			scope: Arc::new(Scope::new()),
			feed,
		}
	}
}

impl CanvasWeb {
	pub fn sample_rate(&self) -> usize {
		self.processor.sample_rate()
	}

	pub fn is_bypassed(&self) -> bool {
		self.bypass.load(Ordering::Relaxed)
	}

	pub fn set_bypassed(&self, bypassed: bool) {
		self.bypass.store(bypassed, Ordering::Relaxed);
	}
}

impl Parameters for CanvasWeb {
	fn get_parameters(&self) -> Vec<Parameter> {
		self.processor.parameters()
	}

	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		self.processor.set_parameter(identifier, value)
	}
}

impl Processor for CanvasWeb {
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

impl Plugin for CanvasWeb {
	const DESCRIPTOR: Descriptor = Descriptor::new("iamdsp.synths.canvas.web", CANVAS_TITLE)
		.with_vendor("iamplugins")
		.with_version(env!("CARGO_PKG_VERSION"))
		.with_description("Two drawn stereo curves, morphed into each other.")
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
				CANVAS_SIZE.0 as f64 / FACE_FRACTION,
				CANVAS_SIZE.1 as f64 / FACE_FRACTION,
			))
			.with_title(CANVAS_TITLE)
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
		let feed = self.feed.clone();

		editor_with(
			self,
			canvas_faceplate(),
			self.bypass.clone(),
			self.scope.clone(),
			self.processor.sample_rate(),
			move |request: &ProtocolRequest<'_>| canvas_route(request, &feed),
			|_: &str| false,
		)
	}
}
