//! The morphing wavetable instrument.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use i_am_dsp::prelude::{
	Adsr, EqualTemperament, GeneratorEffect, ParamMap, Parameter, Parameters, SawWave, SetValue,
	SineWave, SquareWave, TableOsc, TriangleWave, WaveTableSmoother,
};
use i_am_dsp::{ProcessContext, Processor};
use i_am_plugin::processor::{EffectMessage, EffectProcessor};
use i_am_plugin::{
	AspectRatioStrategy, Descriptor, GuiResizeHints, MidiPort, Plugin, Tag, WindowOptions,
};
use i_am_web_gui::{Faceplate, Page, ProtocolRequest, Scope, Switch, editor_with};

use super::play_synth;

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
