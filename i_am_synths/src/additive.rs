//! The additive instrument driven by a bendable saw spectrum.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use i_am_dsp::prelude::{
	AdditiveOsc, Adsr, BendedSawGen, EqualTemperament, GeneratorEffect, ParamMap,
};
use i_am_plugin::processor::EffectProcessor;
use i_am_plugin::{Descriptor, MidiPort, Plugin, Tag, WindowOptions};
use i_am_web_gui::{Faceplate, Page, ProtocolRequest, Scope, Switch, editor_with};

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

/// The per-stage sync switches stay the host's: the face has one for the envelope.
///
/// The oscillator keeps a level of its own; the envelope's gain is the product's.
const ADDITIVE_HIDDEN: &[&str] = &[
	"delay.sync",
	"attack.sync",
	"hold.sync",
	"decay.sync",
	"release.sync",
	"oscillator.gain",
];

/// The envelope's one Sync switch trades the free lengths for musical ones.
const ADDITIVE_SWITCHES: &[Switch] = &[Switch::new(
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
		.with_header(&["sync"])
		.with_switches(ADDITIVE_SWITCHES)
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

synth_product!(AdditiveWeb);

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
		i_am_web_gui::window_options((ADDITIVE_SIZE.0, ADDITIVE_SIZE.1), ADDITIVE_TITLE)
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
