//! The first drum voice: a kick, on note 36.
//!
//! The DSP lives in `i_am_dsp::prelude::DrumInstrument`; this module is the
//! window, the note port, the shared instrument body, and the curve editor's
//! route.

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use i_am_dsp::prelude::{
	Curve, DrumInstrument, GeneratorEffect, Parameter, Parameters, SetValue, from_binary, to_binary,
};
use i_am_dsp::{ProcessContext, Processor};
use i_am_plugin::processor::{EffectMessage, EffectProcessor};
use i_am_plugin::{
	AspectRatioStrategy, Descriptor, GuiResizeHints, MidiPort, Plugin, Tag, WindowOptions,
};
use i_am_web_gui::{Faceplate, Page, ProtocolRequest, ProtocolResponse, Scope, editor_with};

use super::play_synth;

const SIZE: (f32, f32) = (620.0, 320.0);
const TITLE: &str = "I Am Drummer";
const ACCENT: &str = "#c8793d";

const PAGES: &[Page] = &[
	Page::new("Kick", &["Kick"]),
	Page::new("Snare", &["Snare"]),
];

/// The faceplate of the first drum.
pub fn faceplate() -> Faceplate {
	Faceplate::new(TITLE, ACCENT, SIZE).with_pages(PAGES)
}

/// A kick drum, with a web faceplate and an editable pitch fall.
pub struct DrummerWeb {
	processor: EffectProcessor<GeneratorEffect<DrumInstrument>>,
	curve: Arc<Mutex<Curve>>,
	bypass: Arc<AtomicBool>,
	scope: Arc<Scope>,
}

impl Default for DrummerWeb {
	fn default() -> Self {
		let curve = Arc::new(Mutex::new(DrumInstrument::default().pitch_curve.clone()));
		let factory_curve = curve.clone();

		Self {
			processor: EffectProcessor::with_build(move |rate: usize| {
				GeneratorEffect::new(DrumInstrument::new(rate).with_shared(factory_curve.clone()))
			}),
			curve,
			bypass: Arc::new(AtomicBool::new(false)),
			scope: Arc::new(Scope::new()),
		}
	}
}

impl Parameters for DrummerWeb {
	fn get_parameters(&self) -> Vec<Parameter> {
		self.processor.parameters()
	}

	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		// A host restoring the state writes the saved curve; the voice has to
		// hear it too, not only the parameter the next save will read.
		if identifier == "pitch_curve"
			&& let SetValue::Serialized(bytes) = &value
			&& let Ok(curve) = from_binary::<Curve>(bytes.clone())
			&& let Ok(mut shared) = self.curve.lock()
		{
			*shared = curve;
		}

		self.processor.set_parameter(identifier, value)
	}
}

impl Processor for DrummerWeb {
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

impl Plugin for DrummerWeb {
	const DESCRIPTOR: Descriptor = Descriptor::new("iamdsp.synths.drummer.web", TITLE)
		.with_vendor("iamplugins")
		.with_version(env!("CARGO_PKG_VERSION"))
		.with_description("A drum kit synthesised from partials, clicks and noise. M1: the kick.")
		.with_tags(&[
			Tag::Instrument,
			Tag::Synthesizer,
			Tag::Drum,
			Tag::DrumMachine,
		]);

	const INPUT_PORTS: &'static [i_am_plugin::AudioPort] = &[];
	const INPUT_MIDI_PORTS: &'static [MidiPort] = &[MidiPort { name: "Notes" }];
	const HAS_GUI: bool = true;

	fn new() -> Self {
		Self::default()
	}

	fn window_options() -> WindowOptions {
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

	fn param_map(&self) -> i_am_dsp::prelude::ParamMap {
		self.processor.param_map()
	}

	fn sync_params(&mut self) {
		// The host asks for the parameters before it saves them; this is where
		// the curve the editor changed becomes the one the state carries.
		if let Ok(curve) = self.curve.lock()
			&& let Ok(bytes) = to_binary(&*curve)
		{
			self.processor
				.set_parameter("pitch_curve", SetValue::Serialized(bytes));
		}

		self.processor.sync_params();
	}

	fn on_activate(&mut self, sample_rate: usize) {
		self.processor.activate(sample_rate);
	}

	fn editor(&self) -> Option<Box<dyn i_am_plugin::Editor<Self>>> {
		let rate = self.processor.sample_rate();
		let curve = self.curve.clone();

		editor_with(
			self,
			faceplate(),
			self.bypass.clone(),
			self.scope.clone(),
			rate,
			move |request: &ProtocolRequest<'_>| {
				if request.path != "/pitch" {
					return None;
				}

				match request.method {
					"GET" => Some(ProtocolResponse::json(pitch_json(&curve))),
					"POST" => {
						let Some(text) = std::str::from_utf8(request.body).ok() else {
							return Some(ProtocolResponse::json("{\"error\":\"utf8\"}"));
						};
						let Some(points) = parse_pitch(text) else {
							return Some(ProtocolResponse::json("{\"error\":\"points\"}"));
						};
						if let Ok(mut shared) = curve.lock() {
							*shared = points;
						}
						Some(ProtocolResponse::json("{\"ok\":true}"))
					}
					_ => Some(ProtocolResponse::json("{\"ok\":false}")),
				}
			},
			|_: &str| false,
		)
	}
}

/// The curve as the page draws it: `{"points":[[time, ratio, bend], ...]}`.
fn pitch_json(curve: &Arc<Mutex<Curve>>) -> String {
	let Ok(guard) = curve.lock() else {
		return "{\"points\":[]}".to_string();
	};
	let mut out = String::from("{\"points\":[");

	for (index, point) in guard.points().iter().enumerate() {
		if index > 0 {
			out.push(',');
		}
		out.push_str(&format!("[{},{},{}]", point.time, point.value, point.bend));
	}

	out.push_str("]}");
	out
}

/// `time:ratio:bend;...`, the same wire format the rack's curve editor uses.
fn parse_pitch(text: &str) -> Option<Curve> {
	let mut triples = Vec::new();

	for pair in text.trim().split(';') {
		if pair.trim().is_empty() {
			continue;
		}

		let mut parts = pair.split(':');
		let time: f32 = parts.next()?.trim().parse().ok()?;
		let value: f32 = parts.next()?.trim().parse().ok()?;
		let bend: f32 = parts.next().unwrap_or("0").trim().parse().unwrap_or(0.0);

		if !time.is_finite() || !value.is_finite() {
			return None;
		}

		triples.push((
			time.clamp(0.0, 1.0),
			value.clamp(0.5, 16.0),
			bend.clamp(-10.0, 10.0),
		));
	}

	if triples.len() < 2 {
		return None;
	}

	Some(Curve::from_triples(triples))
}
