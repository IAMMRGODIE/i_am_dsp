//! The instrument whose waveform is drawn on a stereo canvas.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use i_am_dsp::prelude::{
	Adsr, EqualTemperament, Generator, GeneratorEffect, MultiStereoGenerators, Node, Oscillator,
	ParamMap, Parameter, Parameters, SetValue, StereoGenerator,
};
use i_am_dsp::{ProcessContext, Processor};
use i_am_plugin::processor::{EffectMessage, EffectProcessor};
use i_am_plugin::{Descriptor, MidiPort, Plugin, Tag, WindowOptions};
use i_am_web_gui::{
	Faceplate, Page, ProtocolRequest, ProtocolResponse, Scope, Switch, editor_with,
};

use super::{VoiceFeed, json_escape, play_synth};

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
/// node (the two control point offsets and the end point).
fn circle_tokens(radius: f32) -> String {
	let k = radius * 0.552_284_8;
	// The handles pull from the end they belong to, so every quarter of the
	// circle stores the same pair of offsets.
	let nodes = [
		[0.0, k, k, 0.0, 0.0, radius],
		[-k, 0.0, 0.0, k, -radius, 0.0],
		[0.0, -k, -k, 0.0, 0.0, -radius],
		[k, 0.0, 0.0, -k, radius, 0.0],
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
			// The two controls are offsets from the ends they pull: the first
			// from the previous point, the second from this node's end.
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

	fn text(&self) -> String {
		self.state
			.lock()
			.unwrap_or_else(|error| error.into_inner())
			.text
			.clone()
	}

	fn apply(&self, text: String) -> String {
		match parse_canvas(&text) {
			Ok(_pages) => {
				{
					let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
					state.text = text;
					state.error = None;
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

impl VoiceFeed for CanvasFeed {
	fn generation(&self) -> u64 {
		self.generation.load(Ordering::Acquire)
	}
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
		// A voice born after a drag starts from the drawing the page last sent,
		// not from the factory shape: every voice snapshots the same current
		// state, so a new note cannot jump back to the first picture.
		let mut multi = MultiStereoGenerators::new();
		multi.generators = parse_canvas(&feed.text()).unwrap_or_default();
		multi.smooth_factor = 0.0;

		let generation = feed.generation();

		Self {
			voice: Adsr::new(multi, EqualTemperament, rate),
			feed,
			generation,
		}
	}

	/// Applies the page's current drawing when the feed has moved since this
	/// voice last looked. Every held voice polls the same generation, so an edit
	/// does not belong to whichever voice happened to run first.
	fn sync(&mut self) {
		if self.feed.changed(&mut self.generation)
			&& let Ok(pages) = parse_canvas(&self.feed.text())
		{
			self.voice.oscillator.generators = pages;
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
		self.sync();
		self.voice.generate(process_context)
	}
}

/// The canvas oscillator as a patch constructor: the drawing, refreshed when
/// the page sends a new one.
pub struct CanvasOscillator {
	multi: MultiStereoGenerators,
	feed: Arc<CanvasFeed>,
	generation: u64,
}

impl CanvasOscillator {
	/// The drawing the handle currently holds.
	pub fn new(handle: CanvasHandle) -> Self {
		let feed = handle.0.clone();
		let mut multi = MultiStereoGenerators::new();
		multi.generators = parse_canvas(&feed.text()).unwrap_or_default();
		multi.smooth_factor = 0.0;
		let generation = feed.generation();

		Self {
			multi,
			feed,
			generation,
		}
	}
}

impl Oscillator<2> for CanvasOscillator {
	fn play_at(&self, frequency: f32, time: f32, phase: [f32; 2]) -> [f32; 2] {
		self.multi.play_at(frequency, time, phase)
	}

	fn sync(&mut self) {
		if self.feed.changed(&mut self.generation)
			&& let Ok(pages) = parse_canvas(&self.feed.text())
		{
			self.multi.generators = pages;
		}
	}
}

impl Parameters for CanvasOscillator {
	fn get_parameters(&self) -> Vec<Parameter> {
		self.multi.get_parameters()
	}

	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		self.multi.set_parameter(identifier, value)
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
		Self::with_handle(CanvasHandle::new())
	}
}

/// A handle to one Canvas node's drawing, shared by the page and the voice.
///
/// The drawing is not a parameter: it is a list of stereo curves the page edits
/// and the voice reads, so the patch's card and the server need the same handle
/// the instrument was built with.
#[derive(Clone)]
pub struct CanvasHandle(Arc<CanvasFeed>);

impl CanvasHandle {
	pub fn new() -> Self {
		Self(CanvasFeed::new())
	}

	/// The drawing as the page stores it.
	pub fn shape(&self) -> String {
		self.0.text()
	}

	/// The route's GET answer.
	pub fn shape_json(&self) -> String {
		format!("{{\"shape\":\"{}\"}}", json_escape(&self.0.text()))
	}

	/// Applies the route's POST body, and answers it.
	pub fn set_shape(&self, text: String) -> String {
		self.0.apply(text)
	}
}

impl Default for CanvasHandle {
	fn default() -> Self {
		Self::new()
	}
}

impl CanvasWeb {
	/// Builds the canvas over a drawing the page already holds.
	pub fn with_handle(handle: CanvasHandle) -> Self {
		let feed = handle.0.clone();
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

	/// The drawing handle this instrument uses.
	pub fn handle(&self) -> CanvasHandle {
		CanvasHandle(self.feed.clone())
	}

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
		i_am_web_gui::window_options((CANVAS_SIZE.0, CANVAS_SIZE.1), CANVAS_TITLE)
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

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn a_new_voice_starts_from_the_current_drawing() {
		let feed = CanvasFeed::new();
		let one_page = format!("1 {}", circle_tokens(0.5));

		assert!(feed.apply(one_page).contains("\"ok\":true"));

		let voice = CanvasVoice::new(48_000, feed);

		assert_eq!(voice.voice.oscillator.generators.len(), 1);
		assert_eq!(voice.voice.oscillator.generators[0].graph.len(), 4);
	}

	#[test]
	fn a_held_voice_follows_the_page() {
		let feed = CanvasFeed::new();
		let mut voice = CanvasVoice::new(48_000, feed.clone());
		let one_page = format!("1 {}", circle_tokens(0.5));

		assert!(feed.apply(one_page).contains("\"ok\":true"));

		voice.sync();

		assert_eq!(voice.voice.oscillator.generators.len(), 1);
		assert_eq!(voice.voice.oscillator.generators[0].graph.len(), 4);
	}

	#[test]
	fn a_canvas_handle_keeps_the_drawing_it_is_given() {
		let handle = CanvasHandle::new();
		let shape = handle.shape();

		assert!(handle.shape_json().contains("shape"));

		let result = handle.set_shape(shape.clone());

		assert!(result.contains("\"ok\":true"), "{result}");
		assert_eq!(handle.shape(), shape);
	}
}
