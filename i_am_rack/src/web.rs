//! The routes a rack page owns.
//!
//! The kit serves the page and the scope; these two routes are the rack's own
//! state. A command is a line per edit -- "0.cutoff=1200" -- which keeps the page
//! free of a JSON parser and makes a batch atomic under one lock.

use std::sync::atomic::Ordering;

use i_am_plugin::wry::{ProtocolRequest, ProtocolResponse};
use serde_json::json;

use crate::shared::RackShared;

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

	let modules = state
		.modules
		.iter()
		.map(|module| {
			json!({
				"id": module.id,
				"kind": module.kind.name(),
				"hero": module.kind.hero(),
				"bypass": module.bypass,
				"controls": shared.schemas.get(&module.kind).cloned().unwrap_or_default(),
				"values": &module.params,
			})
		})
		.collect::<Vec<_>>();

	json!({
		"version": state.version,
		"rate": shared.rate.load(Ordering::Relaxed),
		"modules": modules,
	})
	.to_string()
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

		let Some(module) = state.modules.iter_mut().find(|module| module.id == id)
		else {
			ok = false;
			continue;
		};

		let raw = raw.trim();

		if parameter == "__bypass" {
			module.bypass = matches!(raw, "1" | "true" | "on");
			continue;
		}

		let Some(control) = shared
			.schemas
			.get(&module.kind)
			.and_then(|controls| controls.iter().find(|control| control.id == parameter))
		else {
			ok = false;
			continue;
		};

		let Ok(value) = raw.parse::<f32>() else {
			ok = false;
			continue;
		};

		module.params.insert(
			parameter.to_string(),
			value.clamp(control.min as f32, control.max as f32),
		);
	}

	shared.dirty.store(true, Ordering::Release);

	ProtocolResponse::json(format!("{{\"ok\":{ok}}}"))
}
