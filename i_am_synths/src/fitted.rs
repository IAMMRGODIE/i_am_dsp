//! A fitted instrument: a mode table from a recording, with its own faceplate.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use i_am_dsp::prelude::{FittedInstrument, GeneratorEffect, ImportedTable, ParamMap, SetValue};
use i_am_plugin::processor::EffectProcessor;
use i_am_plugin::{Descriptor, MidiPort, Plugin, Tag, WindowOptions};
use i_am_web_gui::{
	Faceplate, Page, ProtocolRequest, ProtocolResponse, Scope, editor_with, json_escape,
	percent_decode,
};

const SIZE: (f32, f32) = (620.0, 320.0);
const TITLE: &str = "I Am Modal";
const ACCENT: &str = "#5a7cd4";

const PAGES: &[Page] = &[Page::new("Voice", &["Voice"])];

/// The instrument picker sits in the header, like the filter's mode.
const HEADER: &[&str] = &["instrument"];

/// Answers `/table`: the page posts a table the fitter wrote elsewhere.
pub(crate) fn table_route(
	request: &ProtocolRequest<'_>,
	route_map: &ParamMap,
) -> Option<ProtocolResponse> {
	if request.method != "POST" {
		return Some(ProtocolResponse::json("{\"ok\":false}"));
	}
	let table = ImportedTable {
		name: "imported".to_string(),
		bytes: request.body.to_vec(),
	};
	if !import(route_map, table) {
		return Some(ProtocolResponse::json("{\"error\":\"serialize\"}"));
	}
	Some(ProtocolResponse::json("{\"ok\":true}"))
}

/// Answers `/sample`: the page posts one recorded note, and the product fits
/// it into a table and plays it.
///
/// The fit is a few hundred milliseconds of arithmetic on the request thread,
/// which is where the page can afford to wait for it; the audio thread only
/// ever sees the table the result installs.
pub(crate) fn sample_route(
	request: &ProtocolRequest<'_>,
	route_map: &ParamMap,
) -> Option<ProtocolResponse> {
	if request.method != "POST" {
		return Some(ProtocolResponse::json("{\"ok\":false}"));
	}
	let name = query_value(request.query, "name")
		.map(percent_decode)
		.filter(|name| !name.is_empty())
		.unwrap_or_else(|| "sample".to_string());
	// A library sample says which note it is in its name, which is more
	// reliable than reading the pitch back out of the recording. A file that
	// does not say falls back to the fit's own guess.
	let note = query_value(request.query, "note")
		.and_then(|value| value.parse::<u8>().ok())
		.or_else(|| note_of_name(&name));
	let fitted = match i_am_dsp::prelude::fit_sample(request.body, note) {
		Ok(fitted) => fitted,
		Err(error) => {
			return Some(ProtocolResponse::json(format!(
				"{{\"ok\":false,\"error\":\"{}\"}}",
				json_escape(&error)
			)));
		}
	};
	if !import(
		route_map,
		ImportedTable {
			name,
			bytes: fitted.bytes,
		},
	) {
		return Some(ProtocolResponse::json("{\"error\":\"serialize\"}"));
	}
	Some(ProtocolResponse::json(format!(
		"{{\"ok\":true,\"note\":{},\"frequency\":{:.2},\"modes\":{}}}",
		fitted.note, fitted.frequency, fitted.modes
	)))
}

/// Stores a table as the instrument's imported table and selects it.
fn import(route_map: &ParamMap, table: ImportedTable) -> bool {
	let Ok(binary) = i_am_dsp::prelude::to_binary(&Some(table)) else {
		return false;
	};
	route_map.set(
		"imported",
		SetValue::Serialized(binary),
		std::sync::atomic::Ordering::Relaxed,
	);
	route_map.set("instrument", 5i32, std::sync::atomic::Ordering::Relaxed);
	true
}

/// The note a file name carries, in the `Name_A4_127.wav` convention the
/// sample libraries use.
///
/// The octave is the one a DAW writes, which counts middle C as C5, so the
/// name is an octave above the note it sounds.
pub(crate) fn note_of_name(name: &str) -> Option<u8> {
	const NAMES: [&str; 12] = [
		"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
	];
	let body = name.rsplit_once('.').map(|(body, _)| body).unwrap_or(name);
	for token in body.split('_') {
		for (index, pitch) in NAMES.iter().enumerate() {
			let Some(octave) = token.strip_prefix(pitch) else {
				continue;
			};
			// "C" also prefixes "C#", so only the reading whose rest is the
			// octave counts.
			let Ok(octave) = octave.parse::<i32>() else {
				continue;
			};
			let note = (octave + 1) * 12 + index as i32 - 12;
			if (0..=127).contains(&note) {
				return Some(note as u8);
			}
		}
	}
	None
}

/// One value out of a query string like `name=a%20b&note=60`.
fn query_value<'a>(query: Option<&'a str>, key: &str) -> Option<&'a str> {
	query?.split('&').find_map(|part| {
		let (name, value) = part.split_once('=')?;
		(name == key).then_some(value)
	})
}

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

synth_product!(FittedWeb);

impl Plugin for FittedWeb {
	const DESCRIPTOR: Descriptor = Descriptor::new("iamdsp.synths.modal.web", TITLE)
		.with_vendor("iamplugins")
		.with_version(env!("CARGO_PKG_VERSION"))
		.with_description(
			"An instrument fitted from recordings: a measured mode table, played back.",
		)
		.with_tags(&[Tag::Instrument, Tag::Synthesizer]);

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
		let rate = self.processor.sample_rate();
		let map = self.processor.param_map();
		let route_map = map.clone();
		editor_with(
			self,
			faceplate(),
			self.bypass.clone(),
			self.scope.clone(),
			rate,
			move |request: &ProtocolRequest<'_>| match request.path {
				"/table" => table_route(request, &route_map),
				"/sample" => sample_route(request, &route_map),
				_ => None,
			},
			move |_: &str| false,
		)
	}
}
