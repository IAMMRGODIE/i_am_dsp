//! A fitted instrument: a mode table from a recording, with its own faceplate.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use i_am_dsp::prelude::{
	FittedInstrument, GeneratorEffect, ImportedTable, ParamMap, SetValue, TableSlot,
};
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
	slot: &TableSlot,
) -> Option<ProtocolResponse> {
	if request.method != "POST" {
		return Some(ProtocolResponse::json("{\"ok\":false}"));
	}
	import(
		route_map,
		slot,
		ImportedTable {
			name: "imported".to_string(),
			bytes: request.body.to_vec(),
		},
	);
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
	slot: &TableSlot,
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
		.or_else(|| i_am_dsp::prelude::note_of_name(&name));
	let fitted = match i_am_dsp::prelude::fit_sample(request.body, note) {
		Ok(fitted) => fitted,
		Err(error) => {
			return Some(ProtocolResponse::json(format!(
				"{{\"ok\":false,\"error\":\"{}\"}}",
				json_escape(&error)
			)));
		}
	};
	import(
		route_map,
		slot,
		ImportedTable {
			name,
			bytes: fitted.bytes,
		},
	);
	Some(ProtocolResponse::json(format!(
		"{{\"ok\":true,\"note\":{},\"frequency\":{:.2},\"modes\":{}}}",
		fitted.note, fitted.frequency, fitted.modes
	)))
}

/// Stores a table where the instrument plays from, and selects it.
///
/// The table goes into the slot rather than the `imported` parameter: a
/// serialized value cannot travel through the parameter map, which is the
/// reason a fitted sample used to arrive silent. The plug-in copies the slot
/// into the parameter before a save, so the host still stores it.
///
/// `derive` is turned on because a table fitted from one sample holds one
/// note; without it every other key would be silent.
fn import(route_map: &ParamMap, slot: &TableSlot, table: ImportedTable) {
	slot.set(table);
	route_map.set("instrument", 5i32, std::sync::atomic::Ordering::Relaxed);
	route_map.set("derive", true, std::sync::atomic::Ordering::Relaxed);
	route_map.set("transpose", 0i32, std::sync::atomic::Ordering::Relaxed);
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
	/// The table the page fitted, shared with the instrument.
	slot: TableSlot,
	/// The slot version last copied into the `imported` parameter.
	imported_version: u64,
}

impl Default for FittedWeb {
	fn default() -> Self {
		Self::with_slot(TableSlot::default())
	}
}

synth_product!(FittedWeb);

impl FittedWeb {
	/// A product whose instrument plays from a table the host already owns,
	/// which is how the patcher hands a node its fitted table.
	pub fn with_slot(slot: TableSlot) -> Self {
		let build_slot = slot.clone();

		Self {
			processor: EffectProcessor::with_build(move |rate: usize| {
				GeneratorEffect::new(FittedInstrument::with_slot(rate, build_slot.clone()))
			}),
			bypass: Arc::new(AtomicBool::new(false)),
			scope: Arc::new(Scope::new()),
			slot,
			imported_version: 0,
		}
	}

	/// The slot the page's fitted table lives in.
	pub fn slot(&self) -> TableSlot {
		self.slot.clone()
	}
}

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
		// A table the page fitted lives in the slot, because a serialized
		// parameter cannot travel through the parameter map. The host saves the
		// parameter, so it is copied in before the values are pushed.
		if self.slot.version() != self.imported_version
			&& let Some(table) = self.slot.current()
		{
			let Ok(binary) = i_am_dsp::prelude::to_binary(&Some((*table).clone())) else {
				return;
			};
			if self
				.processor
				.set_parameter("imported", SetValue::Serialized(binary))
			{
				self.imported_version = self.slot.version();
			}
		}

		self.processor.sync_params();
	}

	fn on_activate(&mut self, sample_rate: usize) {
		self.processor.activate(sample_rate);
	}

	fn editor(&self) -> Option<Box<dyn i_am_plugin::Editor<Self>>> {
		let rate = self.processor.sample_rate();
		let map = self.processor.param_map();
		let route_map = map.clone();
		let slot = self.slot.clone();
		editor_with(
			self,
			faceplate(),
			self.bypass.clone(),
			self.scope.clone(),
			rate,
			move |request: &ProtocolRequest<'_>| match request.path {
				"/table" => table_route(request, &route_map, &slot),
				"/sample" => sample_route(request, &route_map, &slot),
				_ => None,
			},
			move |_: &str| false,
		)
	}
}
