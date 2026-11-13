//! The additive instrument whose partials a Rhai script decides.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use i_am_dsp::prelude::{
	AdditiveOsc, Adsr, EqualTemperament, Generator, GeneratorEffect, ParamMap, Parameter,
	Parameters, RhaiFreqGen, RhaiScript, SetValue,
};
use i_am_dsp::{ProcessContext, Processor};
use i_am_plugin::processor::{EffectMessage, EffectProcessor};
use i_am_plugin::{Descriptor, MidiPort, Plugin, Tag, WindowOptions};
use i_am_web_gui::{
	Faceplate, Page, ProtocolRequest, ProtocolResponse, Scope, Switch, editor_with,
};

use super::{VoiceFeed, json_escape, play_synth};

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

/// The four knobs and the partial count the picture is drawn from.
#[derive(Clone, Copy)]
pub(crate) struct ScriptPreview {
	a: f32,
	b: f32,
	c: f32,
	d: f32,
	count: usize,
	rate: usize,
}

impl Default for ScriptPreview {
	fn default() -> Self {
		Self {
			a: 0.0,
			b: 0.0,
			c: 0.0,
			d: 0.0,
			count: 32,
			rate: 44_100,
		}
	}
}

/// The state the page and the audio thread share.
#[derive(Default)]
pub(crate) struct ScriptState {
	source: String,
	error: Option<String>,
	/// The last script that compiled, kept so the page can draw its spectrum.
	script: Option<Arc<RhaiScript>>,
	applied: u64,
}

/// The page's script and the audio thread's copy of it.
pub struct ScriptFeed {
	generation: AtomicU64,
	state: Mutex<ScriptState>,
	preview: Mutex<ScriptPreview>,
	/// An error the script raised while it ran, which is not a compile error.
	eval_error: Mutex<Option<String>>,
	/// Whether there is a runtime error, so the audio thread skips the lock when
	/// the last block ran cleanly.
	has_eval_error: AtomicBool,
}

impl ScriptFeed {
	pub(crate) fn new() -> Arc<Self> {
		// The default body is a real script, not just text: compiling it here
		// gives the page a spectrum and the first voice something to play before
		// anyone presses Apply.
		let script = RhaiScript::compile(SCRIPT_DEFAULT).ok().map(Arc::new);
		let generation = u64::from(script.is_some());

		Arc::new(Self {
			generation: AtomicU64::new(generation),
			preview: Mutex::new(ScriptPreview::default()),
			eval_error: Mutex::new(None),
			has_eval_error: AtomicBool::new(false),
			state: Mutex::new(ScriptState {
				source: SCRIPT_DEFAULT.to_string(),
				script,
				..ScriptState::default()
			}),
		})
	}

	/// Records the four knobs and the partial count the picture uses.
	fn set_preview(&self, update: impl FnOnce(&mut ScriptPreview)) {
		if let Ok(mut preview) = self.preview.lock() {
			update(&mut preview);
		}
	}

	/// Records an error the script raised while it ran.
	fn set_eval_error(&self, error: String) {
		self.has_eval_error.store(true, Ordering::Relaxed);

		if let Ok(mut current) = self.eval_error.lock()
			&& current.as_deref() != Some(&error)
		{
			*current = Some(error);
		}
	}

	/// Clears the runtime error after a block that ran cleanly, without locking
	/// once the error is already gone.
	fn clear_eval_error(&self) {
		if !self.has_eval_error.swap(false, Ordering::Relaxed) {
			return;
		}

		if let Ok(mut current) = self.eval_error.lock() {
			*current = None;
		}
	}

	/// The page's view of the script: its text, the error, how many edits the
	/// audio thread has taken and the spectrum the script draws right now.
	fn state_json(&self) -> String {
		let eval_error = self.eval_error.lock().ok().and_then(|error| error.clone());
		let (state_source, state_error, applied, script) = {
			let state = self.state.lock().unwrap_or_else(|error| error.into_inner());

			(
				state.source.clone(),
				state.error.clone(),
				state.applied,
				state.script.clone(),
			)
		};
		let error = match eval_error.as_ref().or(state_error.as_ref()) {
			Some(error) => format!("\"{}\"", json_escape(error)),
			None => "null".to_string(),
		};

		let spectrum = match &script {
			Some(script) if eval_error.is_none() && state_error.is_none() => {
				let preview = self
					.preview
					.lock()
					.map(|preview| *preview)
					.unwrap_or_default();
				let partials = script.partials(
					preview.a,
					preview.b,
					preview.c,
					preview.d,
					preview.rate,
					preview.count.clamp(1, 64),
				);
				let mut out = String::from("[");

				for (index, [ratio, amplitude]) in partials.iter().enumerate() {
					if index > 0 {
						out.push(',');
					}

					let _ = std::fmt::Write::write_fmt(
						&mut out,
						format_args!("[{ratio:.4},{amplitude:.4}]"),
					);
				}

				out.push(']');
				out
			}
			_ => "[]".to_string(),
		};

		format!(
			"{{\"source\":\"{}\",\"error\":{},\"applied\":{},\"spectrum\":{}}}",
			json_escape(&state_source),
			error,
			applied,
			spectrum
		)
	}

	/// The last script that compiled, for a voice born after an apply.
	pub(crate) fn current_script(&self) -> Option<RhaiScript> {
		self.state
			.lock()
			.unwrap_or_else(|error| error.into_inner())
			.script
			.as_ref()
			.map(|script| (**script).clone())
	}

	/// The source the feed is running.
	pub fn source(&self) -> String {
		self.state
			.lock()
			.unwrap_or_else(|error| error.into_inner())
			.source
			.clone()
	}

	/// Compiles a posted body and queues it for the audio thread.
	fn apply(&self, source: String) -> String {
		// A new script starts from a clean slate, whatever the old one raised.
		self.clear_eval_error();

		match RhaiScript::compile(&source) {
			Ok(script) => {
				{
					let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
					state.source = source;
					state.error = None;
					state.script = Some(Arc::new(script));
					state.applied += 1;
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

/// A handle to one Script node's feed, shared by the page and the voice.
#[derive(Clone)]
pub struct ScriptHandle(Arc<ScriptFeed>);

impl ScriptHandle {
	pub fn new() -> Self {
		Self(ScriptFeed::new())
	}

	/// The page's view: source, error, applied edits and the spectrum.
	pub fn state_json(&self) -> String {
		self.0.state_json()
	}

	/// Compiles a posted body and queues it for the audio thread.
	pub fn apply(&self, source: String) -> String {
		self.0.apply(source)
	}

	/// The feed as the node's script source.
	pub fn source(&self) -> Arc<dyn i_am_effects::ScriptSource> {
		self.0.clone()
	}

	/// The source the feed is running.
	pub fn source_text(&self) -> String {
		self.0.source()
	}
}

impl Default for ScriptHandle {
	fn default() -> Self {
		Self::new()
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

impl VoiceFeed for ScriptFeed {
	fn generation(&self) -> u64 {
		self.generation.load(Ordering::Acquire)
	}
}

/// The voice and the feed it takes new scripts from.
struct ScriptedVoice {
	voice: ScriptVoice,
	feed: Arc<ScriptFeed>,
	generation: u64,
}

impl ScriptedVoice {
	fn new(rate: usize, feed: Arc<ScriptFeed>) -> Self {
		feed.set_preview(|preview| preview.rate = rate);

		// A voice born after an apply starts from that script: every voice reads
		// the current compiled one, so a new note cannot fall back to the
		// built-in body.
		let generation = feed.generation();
		let mut oscillator = AdditiveOsc::new(RhaiFreqGen::new(rate), 32);

		if let Some(script) = feed.current_script() {
			oscillator.freq_gen_mut().load_compiled(script);
			oscillator.recaculate_ratio();
		}

		let mut voice = Adsr::new(oscillator, EqualTemperament, rate);

		// The four knobs rebuild the wavetable through the script, so a move
		// crossfades from the table it replaces instead of clicking.
		voice.oscillator.set_rebuild_fade(0.02);

		Self {
			voice,
			feed,
			generation,
		}
	}
}

impl Parameters for ScriptedVoice {
	fn get_parameters(&self) -> Vec<Parameter> {
		self.voice.get_parameters()
	}

	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		// The picture follows the knobs as they move, so the preview reads them
		// from the same values the audio thread plays.
		match identifier {
			"oscillator.freq_gen.a" => {
				if let SetValue::Float(value) = &value {
					self.feed.set_preview(|preview| preview.a = *value);
				}
			}
			"oscillator.freq_gen.b" => {
				if let SetValue::Float(value) = &value {
					self.feed.set_preview(|preview| preview.b = *value);
				}
			}
			"oscillator.freq_gen.c" => {
				if let SetValue::Float(value) = &value {
					self.feed.set_preview(|preview| preview.c = *value);
				}
			}
			"oscillator.freq_gen.d" => {
				if let SetValue::Float(value) = &value {
					self.feed.set_preview(|preview| preview.d = *value);
				}
			}
			"oscillator.num_sines" => {
				if let SetValue::Int(value) = &value {
					self.feed
						.set_preview(|preview| preview.count = (*value).max(1) as usize);
				}
			}
			_ => {}
		}

		self.voice.set_parameter(identifier, value)
	}
}

impl Generator<2> for ScriptedVoice {
	fn generate(&mut self, process_context: &mut Box<dyn ProcessContext>) -> [f32; 2] {
		if self.feed.changed(&mut self.generation)
			&& let Some(script) = self.feed.current_script()
		{
			let oscillator = &mut self.voice.oscillator;
			oscillator.freq_gen_mut().load_compiled(script);
			oscillator.recaculate_ratio();
		}

		let output = self.voice.generate(process_context);

		// A script that fails while it runs is reported like one that fails to
		// compile: the page has one error line either way.
		match self.voice.oscillator.freq_gen_mut().get_eval_error() {
			Ok(()) => self.feed.clear_eval_error(),
			Err(error) => self.feed.set_eval_error(error.to_string()),
		}

		output
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

impl ScriptWeb {
	/// Builds the script over a feed the page already holds.
	pub fn with_handle(handle: ScriptHandle) -> Self {
		let feed = handle.0.clone();
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

	/// The feed the page edits.
	pub fn feed(&self) -> Arc<dyn i_am_effects::ScriptSource> {
		self.feed.clone()
	}

	/// The feed handle this instrument uses.
	pub fn handle(&self) -> ScriptHandle {
		ScriptHandle(self.feed.clone())
	}
}

impl Default for ScriptWeb {
	fn default() -> Self {
		Self::with_handle(ScriptHandle::new())
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
		i_am_web_gui::window_options((SCRIPT_SIZE.0, SCRIPT_SIZE.1), SCRIPT_TITLE)
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

impl i_am_effects::ScriptSource for ScriptFeed {
	fn script_source(&self) -> String {
		self.source()
	}

	fn set_script_source(&self, source: String) {
		let _ = self.apply(source);
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn a_new_voice_starts_from_the_current_script() {
		let feed = ScriptFeed::new();
		let applied = feed.apply(
			"let ratio = 2.0; let amplitude = 1.0; let damping = 0.0; let phase = 0.0;".to_string(),
		);

		assert!(applied.contains("\"ok\":true"), "{applied}");

		let voice = ScriptedVoice::new(48_000, feed.clone());

		assert_eq!(voice.generation, feed.generation.load(Ordering::Acquire));
		assert!(feed.current_script().is_some());
	}
}
