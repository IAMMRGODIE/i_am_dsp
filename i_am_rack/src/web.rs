//! The routes a rack page owns.
//!
//! The kit serves the page and the scope; these two routes are the rack's own
//! state. A command is a line per edit -- "0.cutoff=1200" -- which keeps the page
//! free of a JSON parser and makes a batch atomic under one lock.
//!
//! Besides module parameters, the page can move a band stage, a crossover point
//! and the contents of a slot:
//!
//! ~~~text
//! band.0.gain=0.5        band.0.pan=-0.25
//! band.0.mute=1          band.0.solo=0
//! split.low=200          split.high=4000
//! slot.0.1.kind=Compressor
//! slot.0.1.clear=1
//! ~~~

use i_am_plugin::wry::{ProtocolRequest, ProtocolResponse};

use crate::shared::RackShared;
use crate::state::{BandState, RackState};

/// The names the three bands and the parallel lane answer to.
/// The lane names a card and the page read. The fourth lane is not a band: it
/// runs beside the crossover on the dry input, which is why it is called what it
/// is rather than "full range".
pub const BAND_NAMES: [&str; 4] = ["Low", "Mid", "High", "Parallel"];

/// The lane at an index, with the parallel full-range lane after the bands.
fn lane(state: &RackState, index: usize) -> Option<&BandState> {
	match index {
		0..=2 => state.bands.get(index),
		3 => Some(&state.parallel),
		_ => None,
	}
}

/// The lane at an index, mutably.
fn lane_mut(state: &mut RackState, index: usize) -> Option<&mut BandState> {
	match index {
		0..=2 => state.bands.get_mut(index),
		3 => Some(&mut state.parallel),
		_ => None,
	}
}

pub(crate) fn route(
	request: &ProtocolRequest<'_>,
	shared: &RackShared,
) -> Option<ProtocolResponse> {
	match (request.method, request.path) {
		("GET", "/rack") => Some(ProtocolResponse::json(rack_json(shared))),
		("GET", "/state") => Some(ProtocolResponse::json(state_json(shared))),
		("GET", "/modulation") => Some(ProtocolResponse::json(modulation_json(shared))),
		("POST", "/rack-command") => Some(command(request.body, shared)),
		("GET", "/module-scope") => module_document(request, shared, false),
		("GET", "/module-note") => module_note(request, shared),
		("GET", "/module-bands") => module_bands(request, shared),
		("GET", "/module-ir") => module_ir(request, shared),
		("POST", "/module-ir") => module_ir(request, shared),
		("GET", "/module-script") => module_script(request, shared),
		("POST", "/module-script") => module_script(request, shared),
		("GET", "/module-curve") => module_curve(request, shared),
		("GET", "/module-stereo") => module_document(request, shared, true),
		_ => None,
	}
}

mod commands;
mod import;
mod json;
mod modules;
mod patch;

use self::commands::command;
pub(crate) use self::json::rack_json;
use self::json::{modulation_json, state_json};
use self::modules::{
	module_bands, module_curve, module_document, module_ir, module_note, module_script,
};
