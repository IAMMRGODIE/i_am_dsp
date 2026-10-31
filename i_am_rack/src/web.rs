//! The routes a rack page owns.
//!
//! The kit serves the page and the scope; these two routes are the rack's own
//! state. A command is a line per edit -- "0.cutoff=1200" -- which keeps the page
//! free of a JSON parser and makes a batch atomic under one lock.

use std::sync::atomic::Ordering;

use i_am_plugin::wry::{ProtocolRequest, ProtocolResponse};
use serde_json::{Value, json};

use crate::shared::RackShared;
use crate::state::ModuleState;

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

	json!({
		"version": state.version,
		"rate": shared.rate.load(Ordering::Relaxed),
		"split_low": state.split_low,
		"split_high": state.split_high,
		"bands": bands,
		"modules": modules,
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

		let Some((module_id, parameter)) = key.trim().split_once('.') else {
			ok = false;
			continue;
		};

		let Ok(id) = module_id.trim().parse::<u32>() else {
			ok = false;
			continue;
		};

		let position = state.bands.iter().enumerate().find_map(|(band, state)| {
			state.slots.iter().position(|slot| {
				slot.as_ref().is_some_and(|module| module.id == id)
			}).map(|slot| (band, slot))
		});

		let Some((band, slot)) = position else {
			ok = false;
			continue;
		};

		let module = state.bands[band].slots[slot]
			.as_mut()
			.expect("the position came from a populated slot");
		let raw = raw.trim();

		if parameter == "__bypass" {
			module.bypass = matches!(raw, "1" | "true" | "on");
			continue;
		}

		let kind = module.kind;

		let Some(control) = shared
			.schemas
			.get(&kind)
			.and_then(|controls| controls.iter().find(|control| control.id == parameter))
		else {
			ok = false;
			continue;
		};

		let Ok(value) = raw.parse::<f64>() else {
			ok = false;
			continue;
		};

		module.params.insert(
			parameter.to_string(),
			value.clamp(control.min, control.max),
		);
	}

	shared.dirty.store(true, Ordering::Release);

	ProtocolResponse::json(format!("{{\"ok\":{ok}}}"))
}
