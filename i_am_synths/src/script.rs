//! The additive instrument whose partials a Rhai script decides.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use i_am_dsp::prelude::{
	AdditiveOsc, Adsr, EqualTemperament, Generator, GeneratorEffect, ParamMap, Parameter,
	Parameters, RhaiFreqGen, RhaiScript, SetValue,
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

/// The scripted face's design size, in points.
const SCRIPT_SIZE: (f32, f32) = (820.0, 420.0);
/// The scripted window's title and the face's title.
const SCRIPT_TITLE: &str = "I Am Script";
/// The scripted identity colour, as #rrggbb.
const SCRIPT_ACCENT: &str = "#c77dff";

/// The scripted face's tabs.
const SCRIPT_PAGES: &[Page] = &[
	Page::new("Partials", &["Shape"]),
	Page::new("Envelope", &["Envelope"]),
	Page::new("Voice", &["Voice"]),
];

/// The per-stage sync switches stay the host's: the face has one for the envelope.
///
/// The oscillator keeps a level of its own, and the host owns the sample rate;
/// the envelope's gain is the product's level.
const SCRIPT_HIDDEN: &[&str] = &[
	"delay.sync",
	"attack.sync",
	"hold.sync",
	"decay.sync",
	"release.sync",
	"oscillator.gain",
	"oscillator.freq_gen.sample_rate",
];

/// The envelope's one Sync switch trades the free lengths for musical ones.
const SCRIPT_SWITCHES: &[Switch] = &[Switch::new(
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

/// The body the page shows before anyone edits it.
///
/// Every output is declared with `let` on purpose: Rhai treats an assignment to
/// an undeclared name as a runtime error, and the oscillator reads the four
/// values back from the script's scope. The conversion matters too — an integer
/// is not read back as an `f32`.
pub(crate) const SCRIPT_DEFAULT: &str = "\
// a script decides the spectrum: one ratio, amplitude, damping and phase per
// partial. index runs from 0 to total_amount - 1; a, b, c and d are the knobs.
let n = to_float(index + 1);
let ratio = n;
let amplitude = 1.0 / n;
let damping = 0.0;
let phase = 0.0;
";

/// The state the page and the audio thread share.
#[derive(Default)]
pub(crate) struct ScriptState {
	source: String,
	error: Option<String>,
	applied: u64,
	pending: Option<RhaiScript>,
}

/// The page's script and the audio thread's copy of it.
pub(crate) struct ScriptFeed {
	generation: AtomicU64,
	state: Mutex<ScriptState>,
}

impl ScriptFeed {
	pub(crate) fn new() -> Arc<Self> {
		Arc::new(Self {
			generation: AtomicU64::new(0),
			state: Mutex::new(ScriptState {
				source: SCRIPT_DEFAULT.to_string(),
				..ScriptState::default()
			}),
		})
	}

	/// The script, if one was applied since `seen` was last updated.
	pub(crate) fn take(&self, seen: &mut u64) -> Option<RhaiScript> {
		let generation = self.generation.load(Ordering::Acquire);

		if generation == *seen {
			return None;
		}

		*seen = generation;
		let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
		let script = state.pending.take();

		if script.is_some() {
			state.applied += 1;
		}

		script
	}

	/// The page's view of the script: its text, the compile error and how many
	/// edits the audio thread has actually taken.
	fn state_json(&self) -> String {
		let state = self.state.lock().unwrap_or_else(|error| error.into_inner());
		let error = match &state.error {
			Some(error) => format!("\"{}\"", json_escape(error)),
			None => "null".to_string(),
		};

		format!(
			"{{\"source\":\"{}\",\"error\":{},\"applied\":{}}}",
			json_escape(&state.source),
			error,
			state.applied
		)
	}

	/// Compiles a posted body and queues it for the audio thread.
	fn apply(&self, source: String) -> String {
		match RhaiScript::compile(&source) {
			Ok(script) => {
				{
					let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
					state.source = source;
					state.error = None;
					state.pending = Some(script);
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

/// Answers the page's script routes.
pub(crate) fn script_route(
	request: &ProtocolRequest<'_>,
	feed: &Arc<ScriptFeed>,
) -> Option<ProtocolResponse> {
	if request.method == "GET" && request.path == "/script" {
		return Some(ProtocolResponse::json(feed.state_json()));
	}

	if request.method == "POST" && request.path == "/script" {
		let source = String::from_utf8_lossy(request.body).to_string();
		return Some(ProtocolResponse::json(feed.apply(source)));
	}

	None
}

/// The scripted voice: the additive oscillator and the script it plays.
type ScriptVoice = Adsr<AdditiveOsc<RhaiFreqGen, 64, 2>, EqualTemperament, 2>;

/// The voice and the feed it takes new scripts from.
struct ScriptedVoice {
	voice: ScriptVoice,
	feed: Arc<ScriptFeed>,
	generation: u64,
}

impl ScriptedVoice {
	fn new(rate: usize, feed: Arc<ScriptFeed>) -> Self {
		Self {
			voice: Adsr::new(
				AdditiveOsc::new(RhaiFreqGen::new(rate), 32),
				EqualTemperament,
				rate,
			),
			feed,
			generation: 0,
		}
	}
}

impl Parameters for ScriptedVoice {
	fn get_parameters(&self) -> Vec<Parameter> {
		self.voice.get_parameters()
	}

	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		self.voice.set_parameter(identifier, value)
	}
}

impl Generator<2> for ScriptedVoice {
	fn generate(&mut self, process_context: &mut Box<dyn ProcessContext>) -> [f32; 2] {
		if let Some(script) = self.feed.take(&mut self.generation) {
			let oscillator = &mut self.voice.oscillator;
			oscillator.freq_gen_mut().load_compiled(script);
			oscillator.recaculate_ratio();
		}

		self.voice.generate(process_context)
	}
}

/// The faceplate of the scripted instrument.
pub fn script_faceplate() -> Faceplate {
	Faceplate::new(SCRIPT_TITLE, SCRIPT_ACCENT, SCRIPT_SIZE)
		.with_pages(SCRIPT_PAGES)
		.with_header(&["sync"])
		.with_switches(SCRIPT_SWITCHES)
		.with_hidden(SCRIPT_HIDDEN)
		.with_script()
}

/// The scripted additive instrument, with a web faceplate.
pub struct ScriptWeb {
	processor: EffectProcessor<GeneratorEffect<ScriptedVoice>>,
	bypass: Arc<AtomicBool>,
	scope: Arc<Scope>,
	feed: Arc<ScriptFeed>,
}

impl Default for ScriptWeb {
	fn default() -> Self {
		let feed = ScriptFeed::new();
		let build_feed = feed.clone();

		Self {
			processor: EffectProcessor::with_build(move |rate: usize| {
				GeneratorEffect::new(ScriptedVoice::new(rate, build_feed.clone()))
			}),
			bypass: Arc::new(AtomicBool::new(false)),
			scope: Arc::new(Scope::new()),
			feed,
		}
	}
}

impl ScriptWeb {
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
}

impl Parameters for ScriptWeb {
	fn get_parameters(&self) -> Vec<Parameter> {
		self.processor.parameters()
	}

	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		self.processor.set_parameter(identifier, value)
	}
}

impl Processor for ScriptWeb {
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

impl Plugin for ScriptWeb {
	const DESCRIPTOR: Descriptor = Descriptor::new("iamdsp.synths.script.web", SCRIPT_TITLE)
		.with_vendor("iamplugins")
		.with_version(env!("CARGO_PKG_VERSION"))
		.with_description("An additive voice whose spectrum is written in Rhai.")
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
				SCRIPT_SIZE.0 as f64 / FACE_FRACTION,
				SCRIPT_SIZE.1 as f64 / FACE_FRACTION,
			))
			.with_title(SCRIPT_TITLE)
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
			script_faceplate(),
			self.bypass.clone(),
			self.scope.clone(),
			self.processor.sample_rate(),
			move |request: &ProtocolRequest<'_>| script_route(request, &feed),
			|_: &str| false,
		)
	}
}
