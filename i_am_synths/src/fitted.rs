//! A fitted instrument: a mode table from a recording, with its own faceplate.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use i_am_dsp::prelude::{
	FittedInstrument, GeneratorEffect, ImportedTable, ParamMap, Parameter, Parameters, SetValue,
};
use i_am_dsp::{ProcessContext, Processor};
use i_am_plugin::processor::{EffectMessage, EffectProcessor};
use i_am_plugin::{
	AspectRatioStrategy, Descriptor, GuiResizeHints, MidiPort, Plugin, Tag, WindowOptions,
};
use i_am_web_gui::{Faceplate, Page, ProtocolRequest, ProtocolResponse, Scope, editor_with};

use super::play_synth;

const SIZE: (f32, f32) = (620.0, 320.0);
const TITLE: &str = "I Am Modal";
const ACCENT: &str = "#5a7cd4";

const PAGES: &[Page] = &[Page::new("Voice", &["Voice"])];

/// The instrument picker sits in the header, like the filter's mode.
const HEADER: &[&str] = &["instrument"];

/// The faceplate of the instrument.
pub fn faceplate() -> Faceplate {
	Faceplate::new(TITLE, ACCENT, SIZE)
		.with_pages(PAGES)
		.with_header(HEADER)
}

/// The fitted instrument, with a web faceplate.
pub struct FittedWeb {
	processor: EffectProcessor<GeneratorEffect<FittedInstrument>>,
	bypass: Arc<AtomicBool>,
	scope: Arc<Scope>,
}

impl Default for FittedWeb {
	fn default() -> Self {
		Self {
			processor: EffectProcessor::new(|rate: usize| {
				GeneratorEffect::new(FittedInstrument::new(rate))
			}),
			bypass: Arc::new(AtomicBool::new(false)),
			scope: Arc::new(Scope::new()),
		}
	}
}

impl Parameters for FittedWeb {
	fn get_parameters(&self) -> Vec<Parameter> {
		self.processor.parameters()
	}

	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		self.processor.set_parameter(identifier, value)
	}
}

impl Processor for FittedWeb {
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

impl Plugin for FittedWeb {
	const DESCRIPTOR: Descriptor = Descriptor::new("iamdsp.synths.modal.web", TITLE)
		.with_vendor("iamplugins")
		.with_version(env!("CARGO_PKG_VERSION"))
		.with_description("An instrument fitted from recordings: a measured mode table, played back.")
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
		let rate = self.processor.sample_rate();
		let map = self.processor.param_map();
		let route_map = map.clone();
		editor_with(
			self,
			faceplate(),
			self.bypass.clone(),
			self.scope.clone(),
			rate,
			move |request: &ProtocolRequest<'_>| {
				if request.path != "/table" {
					return None;
				}
				if request.method != "POST" {
					return Some(ProtocolResponse::json("{\"ok\":false}"));
				}
				let table = ImportedTable {
					name: "imported".to_string(),
					bytes: request.body.to_vec(),
				};
				let Ok(binary) = i_am_dsp::prelude::to_binary(&Some(table)) else {
					return Some(ProtocolResponse::json("{\"error\":\"serialize\"}"));
				};
				route_map.set(
					"imported",
					SetValue::Serialized(binary),
					std::sync::atomic::Ordering::Relaxed,
				);
				route_map.set("instrument", 5i32, std::sync::atomic::Ordering::Relaxed);
				Some(ProtocolResponse::json("{\"ok\":true}"))
			},
			move |_: &str| false,
		)
	}
}