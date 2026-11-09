//! The string instrument: a bowed or plucked body with its own faceplate.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use i_am_dsp::prelude::{GeneratorEffect, ParamMap, StringInstrument};
use i_am_plugin::processor::EffectProcessor;
use i_am_plugin::{Descriptor, MidiPort, Plugin, Tag, WindowOptions};
use i_am_web_gui::{Faceplate, Page, ProtocolRequest, Scope, editor_with};

use super::apply_preset_command;

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
	Faceplate::new(TITLE, ACCENT, SIZE)
		.with_pages(PAGES)
		// The unison is fixed at one string and the detune only does something to
		// more than one, so both controls do nothing a player can hear.
		.with_hidden(&["strings", "detune_cents"])
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

synth_product!(StringWeb);

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
		i_am_web_gui::window_options((SIZE.0, SIZE.1), TITLE)
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
