//! The additive instrument whose spectrum is a set of drawn bands.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use i_am_dsp::prelude::{
	AdditiveOsc, Adsr, EqualTemperament, GeneratorEffect, ParamMap, Parameter, Parameters,
	SetValue, SpectralGen,
};
use i_am_dsp::{ProcessContext, Processor};
use i_am_plugin::processor::{EffectMessage, EffectProcessor};
use i_am_plugin::{
	AspectRatioStrategy, Descriptor, GuiResizeHints, MidiPort, Plugin, Tag, WindowOptions,
};
use i_am_web_gui::{Faceplate, Page, ProtocolRequest, Scope, Switch, editor_with};

use super::play_synth;

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

/// The per-stage sync switches stay the host's: the face has one for the envelope.
///
/// The oscillator keeps a level of its own; the envelope's gain is the product's.
const SPECTRUM_HIDDEN: &[&str] = &[
	"delay.sync",
	"attack.sync",
	"hold.sync",
	"decay.sync",
	"release.sync",
	"oscillator.gain",
];

/// The envelope's one Sync switch trades the free lengths for musical ones.
const SPECTRUM_SWITCHES: &[Switch] = &[Switch::new(
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
		.with_header(&["sync"])
		.with_switches(SPECTRUM_SWITCHES)
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
