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

use std::sync::atomic::Ordering;

use i_am_plugin::wry::{ProtocolRequest, ProtocolResponse};
use serde_json::{Value, json};

use crate::crossover::sanitize;
use crate::module::ModuleKind;
use crate::shared::RackShared;
use crate::state::{ModuleState, RackState};

/// The names the three bands answer to.
pub const BAND_NAMES: [&str; 3] = ["Low", "Mid", "High"];

pub(crate) fn route(
	request: &ProtocolRequest<'_>,
	shared: &RackShared,
) -> Option<ProtocolResponse> {
	match (request.method, request.path) {
		("GET", "/rack") => Some(ProtocolResponse::json(rack_json(shared))),
		("POST", "/rack-command") => Some(command(request.body, shared)),
		_ => None,
	}
}

pub(crate) fn rack_json(shared: &RackShared) -> String {
	let state = shared.lock();
	let mut modules = Vec::new();

	for (band_index, band) in state.bands.iter().enumerate() {
		for (slot_index, slot) in band.slots.iter().enumerate() {
			let Some(module) = slot else {
				continue;
			};

			modules.push(module_json(
				shared,
				module,
				band_index,
				slot_index,
			));
		}
	}

	let bands = state
		.bands
		.iter()
		.enumerate()
		.map(|(index, band)| {
			json!({
				"name": BAND_NAMES[index],
				"gain": band.gain,
				"pan": band.pan,
				"mute": band.mute,
				"solo": band.solo,
				"slots": [
					band.slots[0].as_ref().map(|module| module.id),
					band.slots[1].as_ref().map(|module| module.id),
				],
			})
		})
		.collect::<Vec<_>>();

	let kinds = ModuleKind::ALL
		.iter()
		.map(|kind| {
			json!({
				"name": kind.name(),
				"accent": kind.accent(),
				"hero": kind.hero(),
			})
		})
		.collect::<Vec<_>>();

	json!({
		"version": state.version,
		"rate": shared.rate.load(Ordering::Relaxed),
		"split_low": state.split_low,
		"split_high": state.split_high,
		"bands": bands,
		"modules": modules,
		"kinds": kinds,
	})
	.to_string()
}

fn module_json(
	shared: &RackShared,
	module: &ModuleState,
	band: usize,
	slot: usize,
) -> Value {
	json!({
		"id": module.id,
		"band": band,
		"band_name": BAND_NAMES[band],
		"slot": slot,
		"kind": module.kind.name(),
		"accent": module.kind.accent(),
		"hero": module.kind.hero(),
		"mode": module.kind.mode(),
		"header": module.kind.header(),
		"visible": module.kind.visible(),
		"bypass": module.bypass,
		"controls": shared.schemas.get(&module.kind).cloned().unwrap_or_default(),
		"values": &module.params,
	})
}

fn command(body: &[u8], shared: &RackShared) -> ProtocolResponse {
	let text = String::from_utf8_lossy(body);
	let rate = shared.rate.load(Ordering::Relaxed);
	let mut state = shared.lock();
	let mut ok = true;

	for line in text.lines() {
		let line = line.trim();

		if line.is_empty() {
			continue;
		}

		let Some((key, raw)) = line.split_once('=') else {
			ok = false;
			continue;
		};

		let applied = apply_line(&mut state, shared, rate, key.trim(), raw.trim());

		if !applied {
			ok = false;
		}
	}

	shared.dirty.store(true, Ordering::Release);

	ProtocolResponse::json(format!("{{\"ok\":{ok}}}"))
}

fn apply_line(
	state: &mut RackState,
	shared: &RackShared,
	rate: usize,
	key: &str,
	raw: &str,
) -> bool {
	if let Some(rest) = key.strip_prefix("band.") {
		apply_band(state, rest, raw)
	} else if let Some(rest) = key.strip_prefix("split.") {
		apply_split(state, rest, raw, rate)
	} else if let Some(rest) = key.strip_prefix("slot.") {
		apply_slot(state, rest, raw)
	} else {
		apply_module(state, shared, key, raw)
	}
}

fn apply_band(state: &mut RackState, rest: &str, raw: &str) -> bool {
	let Some((band, parameter)) = rest.split_once('.') else {
		return false;
	};

	let Ok(band) = band.parse::<usize>() else {
		return false;
	};

	if band >= state.bands.len() {
		return false;
	}

	let band = &mut state.bands[band];

	match parameter {
		"gain" => {
			let Some(value) = number(raw) else {
				return false;
			};
			band.gain = value.clamp(0.0, 2.0);
			true
		},
		"pan" => {
			let Some(value) = number(raw) else {
				return false;
			};
			band.pan = value.clamp(-1.0, 1.0);
			true
		},
		"mute" => {
			let Some(value) = boolean(raw) else {
				return false;
			};
			band.mute = value;
			true
		},
		"solo" => {
			let Some(value) = boolean(raw) else {
				return false;
			};
			band.solo = value;
			true
		},
		_ => false,
	}
}

fn apply_split(state: &mut RackState, rest: &str, raw: &str, rate: usize) -> bool {
	let Some(value) = number(raw) else {
		return false;
	};

	let (low, high) = match rest {
		"low" => (value as f32, state.split_high),
		"high" => (state.split_low, value as f32),
		_ => return false,
	};

	let (low, high) = sanitize(rate, low, high);
	state.split_low = low;
	state.split_high = high;
	true
}

fn apply_slot(state: &mut RackState, rest: &str, raw: &str) -> bool {
	let Some((band, rest)) = rest.split_once('.') else {
		return false;
	};
	let Some((slot, parameter)) = rest.split_once('.') else {
		return false;
	};

	let Ok(band) = band.parse::<usize>() else {
		return false;
	};
	let Ok(slot) = slot.parse::<usize>() else {
		return false;
	};

	if band >= state.bands.len() || slot >= state.bands[band].slots.len() {
		return false;
	}

	match parameter {
		"kind" => {
			let Some(kind) = module_kind(raw) else {
				return false;
			};

			let (id, bypass, same) = {
				let existing = state.bands[band].slots[slot].as_ref();

				match existing {
					Some(module) => (module.id, module.bypass, module.kind == kind),
					None => (state.next_id(), false, false),
				}
			};

			if same {
				return true;
			}

			let mut module = ModuleState::new(id, kind);
			module.bypass = bypass;
			state.bands[band].slots[slot] = Some(module);
			true
		},
		"clear" => {
			let Some(clear) = boolean(raw) else {
				return false;
			};

			if clear {
				state.bands[band].slots[slot] = None;
			}

			true
		},
		_ => false,
	}
}

fn apply_module(
	state: &mut RackState,
	shared: &RackShared,
	key: &str,
	raw: &str,
) -> bool {
	let Some((id, parameter)) = key.split_once('.') else {
		return false;
	};

	let Ok(id) = id.parse::<u32>() else {
		return false;
	};

	let Some(module) = state.module_mut(id) else {
		return false;
	};

	if parameter == "__bypass" {
		let Some(bypass) = boolean(raw) else {
			return false;
		};
		module.bypass = bypass;
		return true;
	}

	let kind = module.kind;

	let Some(control) = shared
		.schemas
		.get(&kind)
		.and_then(|controls| controls.iter().find(|control| control.id == parameter))
	else {
		return false;
	};

	let Some(value) = number(raw) else {
		return false;
	};

	module.params.insert(
		parameter.to_string(),
		value.clamp(control.min, control.max),
	);
	true
}

fn number(raw: &str) -> Option<f64> {
	raw.parse::<f64>().ok().filter(|value| value.is_finite())
}

fn boolean(raw: &str) -> Option<bool> {
	match raw.to_ascii_lowercase().as_str() {
		"1" | "true" | "on" | "yes" => Some(true),
		"0" | "false" | "off" | "no" => Some(false),
		_ => None,
	}
}

fn module_kind(raw: &str) -> Option<ModuleKind> {
	ModuleKind::ALL
		.into_iter()
		.find(|kind| kind.name().eq_ignore_ascii_case(raw))
}
