//! The synthesizer products.
//!
//! An instrument is a plug-in with no audio input and a note input, and its face
//! is the same web faceplate the effects use: the parameter metadata is the whole
//! layout, and [`i_am_web_gui`] draws it. The one thing an instrument needs that
//! an effect does not is a preset: choosing an instrument has to move the other
//! controls as well, which a single parameter cannot do, so the editor answers the
//! preset choice itself before the generic command parser sees it.
//!
//! The CLAP and VST3 entry points live in `i_am_synths_bundle`, because
//! `clap_entry` is a global symbol.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use i_am_dsp::prelude::{
	AdditiveOsc, Adsr, BendedSawGen, EqualTemperament, GeneratorEffect, ParamMap, Parameter,
	Parameters, SawWave, SetValue, SineWave, SpectralGen, SquareWave, StringInstrument, TableOsc,
	TriangleWave, WaveTableSmoother,
};
use i_am_dsp::{NoteEvent, ProcessContext, Processor};
use i_am_plugin::processor::EffectMessage;
use i_am_plugin::processor::EffectProcessor;
use i_am_plugin::{
	AspectRatioStrategy, Descriptor, GuiResizeHints, MidiPort, Plugin, Tag, WindowOptions,
};
use i_am_web_gui::{Faceplate, Page, ProtocolRequest, Scope, Switch, editor_with};

/// The face's design size, in points.
const SIZE: (f32, f32) = (820.0, 420.0);
/// The window's title and the face's title.
const TITLE: &str = "I Am Strings";
/// The identity colour, as #rrggbb.
const ACCENT: &str = "#8ac926";

/// The face's tabs: the string and its contact on one page, the body on the
/// next, and the controls a player moves while playing on the last.
const PAGES: &[Page] = &[
	Page::new("Voice", &["String"]),
	Page::new("Body", &["Body"]),
	Page::new("Performance", &["Performance"]),
];

/// The faceplate of the instrument.
pub fn faceplate() -> Faceplate {
	Faceplate::new(TITLE, ACCENT, SIZE).with_pages(PAGES)
}

/// The string instrument, with a web faceplate.
pub struct StringWeb {
	processor: EffectProcessor<GeneratorEffect<StringInstrument>>,
	bypass: Arc<AtomicBool>,
	scope: Arc<Scope>,
}

impl Default for StringWeb {
	fn default() -> Self {
		Self {
			processor: EffectProcessor::new(|rate: usize| {
				GeneratorEffect::new(StringInstrument::new(rate))
			}),
			bypass: Arc::new(AtomicBool::new(false)),
			scope: Arc::new(Scope::new()),
		}
	}
}

impl StringWeb {
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

impl Parameters for StringWeb {
	fn get_parameters(&self) -> Vec<Parameter> {
		self.processor.parameters()
	}

	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		self.processor.set_parameter(identifier, value)
	}
}

impl Processor for StringWeb {
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

impl Plugin for StringWeb {
	const DESCRIPTOR: Descriptor = Descriptor::new("iamdsp.synths.strings.web", TITLE)
		.with_vendor("iamplugins")
		.with_version(env!("CARGO_PKG_VERSION"))
		.with_description(
			"A physically modelled string instrument: piano, guitar, violin and cello.",
		)
		.with_tags(&[Tag::Instrument, Tag::Synthesizer]);

	/// A synthesizer makes its sound instead of processing one, so it has no audio
	/// input at all.
	const INPUT_PORTS: &'static [i_am_plugin::AudioPort] = &[];
	const INPUT_MIDI_PORTS: &'static [MidiPort] = &[MidiPort { name: "Notes" }];
	const HAS_GUI: bool = true;

	fn new() -> Self {
		Self::default()
	}

	fn window_options() -> WindowOptions {
		// The page draws the faceplate at 85% of the window, so the window asks for
		// the design size over that: the first open has the same faceplate, with a
		// frame around it.
		const FACE_FRACTION: f64 = 0.85;

		WindowOptions::new()
			.with_size((SIZE.0 as f64 / FACE_FRACTION, SIZE.1 as f64 / FACE_FRACTION))
			.with_title(TITLE)
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
		let map = self.processor.param_map();
		let rate = self.processor.sample_rate();

		editor_with(
			self,
			faceplate(),
			self.bypass.clone(),
			self.scope.clone(),
			rate,
			|_: &ProtocolRequest<'_>| None,
			move |body: &str| apply_preset_command(&map, rate, body),
		)
	}
}

/// The body both instruments share: the note badge, the bypass and the audio.
///
/// An instrument has nothing to pass through, so bypass is silence, and the scope
/// always sees silence as its input: the pictures draw the sound the voice made.
fn play_synth<P: Processor>(
	processor: &mut P,
	scope: &Scope,
	bypass: &AtomicBool,
	samples: &mut [f32; 2],
	other: &[[f32; 2]],
	process_context: &mut Box<dyn ProcessContext>,
) {
	// The note badge: the page shows the note the host is playing.
	for event in process_context.events() {
		match event {
			NoteEvent::NoteOn { note, .. } => scope.set_note(*note as i32),
			NoteEvent::NoteOff { note, .. } | NoteEvent::Stop { note, .. } => {
				if scope.note() == *note as i32 {
					scope.set_note(-1);
				}
			}
			NoteEvent::ImmediateStop => scope.set_note(-1),
			_ => {}
		}
	}

	if bypass.load(Ordering::Relaxed) {
		*samples = [0.0; 2];
		scope.push_stereo([0.0; 2], [0.0; 2], 0.0);
		return;
	}

	Processor::process(processor, samples, other, process_context);

	scope.push_stereo([0.0; 2], *samples, 0.0);
}

/// Answers the preset choice before the generic parameter parser sees it.
///
/// The DSP follows the `preset` parameter every block, so it switches the string
/// and the body on its own, but the other controls are independent parameters:
/// loading a patch means moving all of them. The map is the shared state the page
/// and the audio thread both read, so writing them here is the whole operation.
fn apply_preset_command(map: &ParamMap, rate: usize, body: &str) -> bool {
	let Some(value) = body.strip_prefix("preset=") else {
		return false;
	};

	let Ok(index) = value.trim().parse::<i32>() else {
		return false;
	};

	apply_preset(map, rate, index);

	true
}

/// Moves every control to the preset's values, except the performance controls.
fn apply_preset(map: &ParamMap, rate: usize, index: i32) {
	let mut instrument = StringInstrument::new(rate);
	instrument.select_preset(index);

	for parameter in instrument.get_parameters() {
		// The transpose is a performance control, not part of an instrument; the
		// DSP's own preset loading leaves it alone and so does this.
		if parameter.identifier == "transpose" {
			continue;
		}

		let identifier = parameter.identifier;
		let value = parameter.value.to_set_value();
		map.set(&identifier, value, Ordering::Relaxed);
	}
}

/// The shapes face's design size, in points.
const SHAPES_SIZE: (f32, f32) = (820.0, 420.0);
/// The shapes window's title and the face's title.
const SHAPES_TITLE: &str = "I Am Shapes";
/// The shapes identity colour, as #rrggbb.
const SHAPES_ACCENT: &str = "#4cc9f0";

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

/// The shapes voice: four classic tables, morphed by the Shape control.
type ShapeVoice = Adsr<TableOsc<WaveTableSmoother>, EqualTemperament, 2>;

fn shape_voice(rate: usize) -> ShapeVoice {
	Adsr::new(
		TableOsc(WaveTableSmoother::new(
			vec![
				Box::new(SineWave),
				Box::new(TriangleWave),
				Box::new(SawWave),
				Box::new(SquareWave),
			],
			0.0,
		)),
		EqualTemperament,
		rate,
	)
}

/// The faceplate of the shapes instrument.
pub fn shapes_faceplate() -> Faceplate {
	Faceplate::new(SHAPES_TITLE, SHAPES_ACCENT, SHAPES_SIZE)
		.with_pages(SHAPES_PAGES)
		.with_header(&["sync"])
		.with_switches(SHAPES_SWITCHES)
		.with_hidden(SHAPES_HIDDEN)
}

/// The wavetable shapes instrument, with a web faceplate.
pub struct ShapesWeb {
	processor: EffectProcessor<GeneratorEffect<ShapeVoice>>,
	bypass: Arc<AtomicBool>,
	scope: Arc<Scope>,
}

impl Default for ShapesWeb {
	fn default() -> Self {
		Self {
			processor: EffectProcessor::new(|rate: usize| GeneratorEffect::new(shape_voice(rate))),
			bypass: Arc::new(AtomicBool::new(false)),
			scope: Arc::new(Scope::new()),
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
		.with_description("An ADSR around a morphing wavetable: sine, triangle, saw and square.")
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
		editor_with(
			self,
			shapes_faceplate(),
			self.bypass.clone(),
			self.scope.clone(),
			self.processor.sample_rate(),
			|_: &ProtocolRequest<'_>| None,
			|_: &str| false,
		)
	}
}

/// The additive face's design size, in points.
const ADDITIVE_SIZE: (f32, f32) = (820.0, 420.0);
/// The additive window's title and the face's title.
const ADDITIVE_TITLE: &str = "I Am Additive";
/// The additive identity colour, as #rrggbb.
const ADDITIVE_ACCENT: &str = "#f72585";

/// The additive face's tabs: the partials, the envelope and the voice.
const ADDITIVE_PAGES: &[Page] = &[
	Page::new("Partials", &["Shape"]),
	Page::new("Envelope", &["Envelope"]),
	Page::new("Voice", &["Voice"]),
];

/// The oscillator keeps a level of its own; the envelope's gain is the product's.
const ADDITIVE_HIDDEN: &[&str] = &["oscillator.gain"];

/// The additive voice: a saw drawn as sines, bent into other spectra.
type AdditiveVoice = Adsr<AdditiveOsc<BendedSawGen, 64, 2>, EqualTemperament, 2>;

fn additive_voice(rate: usize) -> AdditiveVoice {
	Adsr::new(
		AdditiveOsc::new(BendedSawGen::default(), 32),
		EqualTemperament,
		rate,
	)
}

/// The faceplate of the additive instrument.
pub fn additive_faceplate() -> Faceplate {
	Faceplate::new(ADDITIVE_TITLE, ADDITIVE_ACCENT, ADDITIVE_SIZE)
		.with_pages(ADDITIVE_PAGES)
		.with_hidden(ADDITIVE_HIDDEN)
}

/// The additive instrument, with a web faceplate.
pub struct AdditiveWeb {
	processor: EffectProcessor<GeneratorEffect<AdditiveVoice>>,
	bypass: Arc<AtomicBool>,
	scope: Arc<Scope>,
}

impl Default for AdditiveWeb {
	fn default() -> Self {
		Self {
			processor: EffectProcessor::new(|rate: usize| {
				GeneratorEffect::new(additive_voice(rate))
			}),
			bypass: Arc::new(AtomicBool::new(false)),
			scope: Arc::new(Scope::new()),
		}
	}
}

impl AdditiveWeb {
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

impl Parameters for AdditiveWeb {
	fn get_parameters(&self) -> Vec<Parameter> {
		self.processor.parameters()
	}

	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		self.processor.set_parameter(identifier, value)
	}
}

impl Processor for AdditiveWeb {
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

impl Plugin for AdditiveWeb {
	const DESCRIPTOR: Descriptor = Descriptor::new("iamdsp.synths.additive.web", ADDITIVE_TITLE)
		.with_vendor("iamplugins")
		.with_version(env!("CARGO_PKG_VERSION"))
		.with_description(
			"An additive voice: a saw built from sine partials, bent into new spectra.",
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
				ADDITIVE_SIZE.0 as f64 / FACE_FRACTION,
				ADDITIVE_SIZE.1 as f64 / FACE_FRACTION,
			))
			.with_title(ADDITIVE_TITLE)
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
		editor_with(
			self,
			additive_faceplate(),
			self.bypass.clone(),
			self.scope.clone(),
			self.processor.sample_rate(),
			|_: &ProtocolRequest<'_>| None,
			|_: &str| false,
		)
	}
}

/// The spectrum face's design size, in points.
const SPECTRUM_SIZE: (f32, f32) = (820.0, 420.0);
/// The spectrum window's title and the face's title.
const SPECTRUM_TITLE: &str = "I Am Spectrum";
/// The spectrum identity colour, as #rrggbb.
const SPECTRUM_ACCENT: &str = "#ffd166";

/// The spectrum face's tabs: the partials, the envelope and the voice.
const SPECTRUM_PAGES: &[Page] = &[
	Page::new("Spectrum", &["Shape"]),
	Page::new("Envelope", &["Envelope"]),
	Page::new("Voice", &["Voice"]),
];

/// The oscillator keeps a level of its own; the envelope's gain is the product's.
const SPECTRUM_HIDDEN: &[&str] = &["oscillator.gain"];

/// The spectrum voice: every partial is an integer ratio, so the whole sum is
/// an inverse-FFT wavetable and playback is one lookup per sample.
type SpectrumVoice = Adsr<AdditiveOsc<SpectralGen, 128, 2>, EqualTemperament, 2>;

fn spectrum_voice(rate: usize) -> SpectrumVoice {
	Adsr::new(
		AdditiveOsc::new(SpectralGen::default(), 64),
		EqualTemperament,
		rate,
	)
}

/// The faceplate of the spectrum instrument.
pub fn spectrum_faceplate() -> Faceplate {
	Faceplate::new(SPECTRUM_TITLE, SPECTRUM_ACCENT, SPECTRUM_SIZE)
		.with_pages(SPECTRUM_PAGES)
		.with_hidden(SPECTRUM_HIDDEN)
}

/// The additive spectrum instrument, with a web faceplate.
pub struct SpectrumWeb {
	processor: EffectProcessor<GeneratorEffect<SpectrumVoice>>,
	bypass: Arc<AtomicBool>,
	scope: Arc<Scope>,
}

impl Default for SpectrumWeb {
	fn default() -> Self {
		Self {
			processor: EffectProcessor::new(|rate: usize| {
				GeneratorEffect::new(spectrum_voice(rate))
			}),
			bypass: Arc::new(AtomicBool::new(false)),
			scope: Arc::new(Scope::new()),
		}
	}
}

impl SpectrumWeb {
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

impl Parameters for SpectrumWeb {
	fn get_parameters(&self) -> Vec<Parameter> {
		self.processor.parameters()
	}

	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		self.processor.set_parameter(identifier, value)
	}
}

impl Processor for SpectrumWeb {
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

impl Plugin for SpectrumWeb {
	const DESCRIPTOR: Descriptor = Descriptor::new("iamdsp.synths.spectrum.web", SPECTRUM_TITLE)
		.with_vendor("iamplugins")
		.with_version(env!("CARGO_PKG_VERSION"))
		.with_description(
			"An additive spectrum: a tilt, an odd/even balance and a formant, played from an inverse-FFT wavetable.",
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
				SPECTRUM_SIZE.0 as f64 / FACE_FRACTION,
				SPECTRUM_SIZE.1 as f64 / FACE_FRACTION,
			))
			.with_title(SPECTRUM_TITLE)
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
		editor_with(
			self,
			spectrum_faceplate(),
			self.bypass.clone(),
			self.scope.clone(),
			self.processor.sample_rate(),
			|_: &ProtocolRequest<'_>| None,
			|_: &str| false,
		)
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use i_am_dsp::ProcessInfos;

	struct TestContext {
		events: Vec<NoteEvent>,
		infos: ProcessInfos,
	}

	impl TestContext {
		fn new() -> Self {
			let mut infos = ProcessInfos::default();
			infos.sample_rate = 48_000;
			infos.playing = true;
			Self {
				events: Vec::new(),
				infos,
			}
		}
	}

	impl ProcessContext for TestContext {
		fn infos(&self) -> &ProcessInfos {
			&self.infos
		}

		fn next_event(&mut self) -> Option<NoteEvent> {
			self.events.pop()
		}

		fn send_event(&mut self, event: NoteEvent) {
			self.events.push(event);
		}

		fn events(&self) -> &[NoteEvent] {
			&self.events
		}

		fn clear_events(&mut self) {
			self.events.clear();
		}
	}

	fn value(map: &ParamMap, identifier: &str) -> SetValue {
		map.get(identifier)
			.expect(identifier)
			.load(Ordering::Relaxed)
	}

	#[test]
	fn a_synthesizer_has_a_note_port_and_no_audio_input() {
		assert!(<StringWeb as Plugin>::INPUT_PORTS.is_empty());
		assert_eq!(<StringWeb as Plugin>::INPUT_MIDI_PORTS.len(), 1);
		assert_eq!(<StringWeb as Plugin>::INPUT_MIDI_PORTS[0].name, "Notes");

		let tags = <StringWeb as Plugin>::DESCRIPTOR.tags;
		assert!(tags.iter().any(|tag| matches!(tag, Tag::Instrument)));
		assert!(tags.iter().any(|tag| matches!(tag, Tag::Synthesizer)));
	}

	#[test]
	fn the_face_offers_the_four_instruments_and_five_sections() {
		let face = i_am_web_gui::face_json(&StringWeb::new(), &faceplate());

		assert!(face.contains("\"hero\":\"string\""), "{face}");
		assert!(face.contains("\"hero\":\"partials\""), "{face}");
		assert!(face.contains("\"hero\":\"vibrato\""), "{face}");
		assert!(
			face.contains("\"choices\":[\"Piano\",\"Guitar\",\"Violin\",\"Cello\"]"),
			"{face}"
		);
		assert!(
			face.contains(
				"\"pages\":[{\"name\":\"Voice\",\"sections\":[\"String\"]},\
				{\"name\":\"Body\",\"sections\":[\"Body\"]},\
				{\"name\":\"Performance\",\"sections\":[\"Performance\"]}]"
			),
			"{face}"
		);

		for name in ["String", "Body", "Performance"] {
			assert!(face.contains(&format!("\"name\":\"{name}\"")), "{face}");
		}
	}

	#[test]
	fn choosing_an_instrument_moves_the_controls() {
		let plugin = StringWeb::new();
		let map = plugin.param_map();

		map.set("transpose", 5.0f32, Ordering::Relaxed);
		apply_preset_command(&map, 48_000, "preset=1");

		assert!(matches!(value(&map, "preset"), SetValue::Int(1)));
		assert!(
			matches!(value(&map, "gain"), SetValue::Float(gain) if (gain - 0.9).abs() < 1e-6),
			"the guitar's gain did not arrive: {:?}",
			value(&map, "gain")
		);
		assert!(
			matches!(value(&map, "transpose"), SetValue::Float(shift) if (shift - 5.0).abs() < 1e-6),
			"the performance transpose was overwritten: {:?}",
			value(&map, "transpose")
		);
	}

	#[test]
	fn a_note_plays_through_the_plugin() {
		let mut plugin = StringWeb::new();
		Plugin::on_activate(&mut plugin, 48_000);

		let mut context: Box<dyn ProcessContext> = Box::new(TestContext {
			events: vec![NoteEvent::NoteOn {
				channel: 0,
				note: 57,
				velocity: 1.0,
			}],
			infos: TestContext::new().infos,
		});
		let mut peak = 0.0f32;

		for _ in 0..24_000 {
			let mut frame = [0.0f32; 2];
			Processor::process(&mut plugin, &mut frame, &[], &mut context);
			peak = peak.max(frame[0].abs());
		}

		assert!(peak > 1e-3, "the instrument was silent: {peak}");
		assert!(peak < 4.0, "the instrument ran away: {peak}");
	}

	#[test]
	fn a_shapes_synth_has_a_note_port_and_no_audio_input() {
		assert!(<ShapesWeb as Plugin>::INPUT_PORTS.is_empty());
		assert_eq!(<ShapesWeb as Plugin>::INPUT_MIDI_PORTS.len(), 1);

		let tags = <ShapesWeb as Plugin>::DESCRIPTOR.tags;
		assert!(tags.iter().any(|tag| matches!(tag, Tag::Instrument)));
		assert!(tags.iter().any(|tag| matches!(tag, Tag::Synthesizer)));
	}

	#[test]
	fn the_shapes_face_is_three_pages_with_one_sync_switch() {
		let face = i_am_web_gui::face_json(&ShapesWeb::new(), &shapes_faceplate());

		assert!(face.contains("\"hero\":\"shape\""), "{face}");
		assert!(face.contains("\"hero\":\"envelope\""), "{face}");
		assert!(face.contains("\"hero\":\"waveform\""), "{face}");
		assert!(
			face.contains("\"id\":\"oscillator.smooth_factor\""),
			"{face}"
		);
		assert!(
			face.contains(
				"\"pages\":[{\"name\":\"Shape\",\"sections\":[\"Shape\"]},\
				{\"name\":\"Envelope\",\"sections\":[\"Envelope\"]},\
				{\"name\":\"Voice\",\"sections\":[\"Voice\"]}]"
			),
			"{face}"
		);
		// One Sync switch in the header, trading the free lengths for beats.
		assert!(face.contains("\"header\":[\"sync\"]"), "{face}");
		assert!(
			face.contains("\"switches\":[{\"control\":\"sync\",\"on\":[\"delay.beats\""),
			"{face}"
		);
		// The per-stage switches stay the host's; the beats lengths are on the face.
		assert!(
			face.contains(
				"\"hidden\":[\"delay.sync\",\"attack.sync\",\"hold.sync\",\"decay.sync\",\"release.sync\"]"
			),
			"{face}"
		);

		for name in ["Shape", "Envelope", "Voice"] {
			assert!(face.contains(&format!("\"name\":\"{name}\"")), "{face}");
		}
	}

	#[test]
	fn a_note_plays_through_the_shapes_plugin() {
		let mut plugin = ShapesWeb::new();
		Plugin::on_activate(&mut plugin, 48_000);

		let mut context: Box<dyn ProcessContext> = Box::new(TestContext {
			events: vec![NoteEvent::NoteOn {
				channel: 0,
				note: 57,
				velocity: 1.0,
			}],
			infos: TestContext::new().infos,
		});
		// A host shows a note only at the sample it arrived at, and the Adsr reads
		// the events without consuming them: clear them after the first frame, as
		// the host would.
		let mut frame = [0.0f32; 2];
		Processor::process(&mut plugin, &mut frame, &[], &mut context);
		context.clear_events();

		let mut peak = frame[0].abs();

		for _ in 1..24_000 {
			let mut frame = [0.0f32; 2];
			Processor::process(&mut plugin, &mut frame, &[], &mut context);
			peak = peak.max(frame[0].abs());
		}

		assert!(peak > 1e-3, "the instrument was silent: {peak}");
		assert!(peak < 4.0, "the instrument ran away: {peak}");
	}
	#[test]
	fn an_additive_synth_has_a_note_port_and_no_audio_input() {
		assert!(<AdditiveWeb as Plugin>::INPUT_PORTS.is_empty());
		assert_eq!(<AdditiveWeb as Plugin>::INPUT_MIDI_PORTS.len(), 1);

		let tags = <AdditiveWeb as Plugin>::DESCRIPTOR.tags;
		assert!(tags.iter().any(|tag| matches!(tag, Tag::Instrument)));
		assert!(tags.iter().any(|tag| matches!(tag, Tag::Synthesizer)));
	}

	#[test]
	fn the_additive_face_has_three_pages_and_a_partial_hero() {
		let face = i_am_web_gui::face_json(&AdditiveWeb::new(), &additive_faceplate());

		assert!(face.contains("\"hero\":\"additive\""), "{face}");
		for id in [
			"oscillator.num_sines",
			"oscillator.freq_gen.bend_amount",
			"oscillator.freq_gen.center_space",
			"oscillator.freq_gen.total_bend_amount",
			"oscillator.freq_gen.scale_amount",
			"oscillator.gain",
		] {
			assert!(face.contains(&format!("\"id\":\"{id}\"")), "{face}");
		}
		for label in ["Bend", "Center", "Overall", "Scale", "Partials"] {
			assert!(face.contains(&format!("\"label\":\"{label}\"")), "{face}");
		}
		assert!(
			face.contains(
				"\"pages\":[{\"name\":\"Partials\",\"sections\":[\"Shape\"]},\
				{\"name\":\"Envelope\",\"sections\":[\"Envelope\"]},\
				{\"name\":\"Voice\",\"sections\":[\"Voice\"]}]"
			),
			"{face}"
		);
		assert!(
			face.contains("\"hidden\":[\"oscillator.gain\"]"),
			"the oscillator's own level stays off the face: {face}"
		);
	}

	#[test]
	fn a_note_plays_through_the_additive_plugin() {
		let mut plugin = AdditiveWeb::new();
		Plugin::on_activate(&mut plugin, 48_000);

		let mut context: Box<dyn ProcessContext> = Box::new(TestContext {
			events: vec![NoteEvent::NoteOn {
				channel: 0,
				note: 57,
				velocity: 1.0,
			}],
			infos: TestContext::new().infos,
		});
		let mut frame = [0.0f32; 2];
		Processor::process(&mut plugin, &mut frame, &[], &mut context);
		context.clear_events();

		let mut peak = frame[0].abs();

		for _ in 1..24_000 {
			let mut frame = [0.0f32; 2];
			Processor::process(&mut plugin, &mut frame, &[], &mut context);
			peak = peak.max(frame[0].abs());
		}

		assert!(peak > 1e-3, "the instrument was silent: {peak}");
		assert!(peak < 4.0, "the instrument ran away: {peak}");
	}

	#[test]
	fn a_spectrum_synth_has_a_note_port_and_no_audio_input() {
		assert!(<SpectrumWeb as Plugin>::INPUT_PORTS.is_empty());
		assert_eq!(<SpectrumWeb as Plugin>::INPUT_MIDI_PORTS.len(), 1);

		let tags = <SpectrumWeb as Plugin>::DESCRIPTOR.tags;
		assert!(tags.iter().any(|tag| matches!(tag, Tag::Instrument)));
		assert!(tags.iter().any(|tag| matches!(tag, Tag::Synthesizer)));
	}

	#[test]
	fn the_spectrum_face_is_three_pages_with_the_partial_hero() {
		let face = i_am_web_gui::face_json(&SpectrumWeb::new(), &spectrum_faceplate());

		assert!(face.contains("\"hero\":\"additive\""), "{face}");
		for id in [
			"oscillator.num_sines",
			"oscillator.freq_gen.tilt",
			"oscillator.freq_gen.odd",
			"oscillator.freq_gen.formant",
			"oscillator.freq_gen.formant_center",
			"oscillator.gain",
		] {
			assert!(face.contains(&format!("\"id\":\"{id}\"")), "{face}");
		}
		for label in ["Tilt", "Odd", "Formant", "Centre", "Partials"] {
			assert!(face.contains(&format!("\"label\":\"{label}\"")), "{face}");
		}
		assert!(
			face.contains(
				"\"pages\":[{\"name\":\"Spectrum\",\"sections\":[\"Shape\"]},\
				{\"name\":\"Envelope\",\"sections\":[\"Envelope\"]},\
				{\"name\":\"Voice\",\"sections\":[\"Voice\"]}]"
			),
			"{face}"
		);
		assert!(
			face.contains("\"hidden\":[\"oscillator.gain\"]"),
			"the oscillator's own level stays off the face: {face}"
		);
	}

	#[test]
	fn a_note_plays_through_the_spectrum_plugin() {
		let mut plugin = SpectrumWeb::new();
		Plugin::on_activate(&mut plugin, 48_000);

		let mut context: Box<dyn ProcessContext> = Box::new(TestContext {
			events: vec![NoteEvent::NoteOn {
				channel: 0,
				note: 57,
				velocity: 1.0,
			}],
			infos: TestContext::new().infos,
		});
		let mut frame = [0.0f32; 2];
		Processor::process(&mut plugin, &mut frame, &[], &mut context);
		context.clear_events();

		let mut peak = frame[0].abs();

		for _ in 1..24_000 {
			let mut frame = [0.0f32; 2];
			Processor::process(&mut plugin, &mut frame, &[], &mut context);
			peak = peak.max(frame[0].abs());
		}

		assert!(peak > 1e-3, "the instrument was silent: {peak}");
		assert!(peak < 4.0, "the instrument ran away: {peak}");
	}
}
