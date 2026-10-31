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
use i_am_web_gui::Kind;
use serde_json::{Value, json};

use crate::crossover::sanitize;
use crate::module::ModuleKind;
use crate::shared::RackShared;
use crate::state::{Assignment, AssignmentMode, ModuleState, RackState, Target, lfo_preset};

/// The names the three bands answer to.
pub const BAND_NAMES: [&str; 3] = ["Low", "Mid", "High"];

pub(crate) fn route(
	request: &ProtocolRequest<'_>,
	shared: &RackShared,
) -> Option<ProtocolResponse> {
	match (request.method, request.path) {
		("GET", "/rack") => Some(ProtocolResponse::json(rack_json(shared))),
		("GET", "/modulation") => Some(ProtocolResponse::json(modulation_json(shared))),
		("POST", "/rack-command") => Some(command(request.body, shared)),
		_ => None,
	}
}

fn modulation_json(shared: &RackShared) -> String {
	let values: [f32; 4] = std::array::from_fn(|index| {
		f32::from_bits(shared.modulation[index].load(Ordering::Relaxed))
	});
	let phases: [f32; 2] = std::array::from_fn(|index| {
		f32::from_bits(shared.phases[index].load(Ordering::Relaxed))
	});

	json!({
		"values": values,
		"phases": phases,
	})
	.to_string()
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
				&state.assignments,
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

	let targets = targets_json(&state, shared);

	let assignments = state
		.assignments
		.iter()
		.enumerate()
		.map(|(slot, assignment)| match assignment {
			Some(assignment) => json!({
				"slot": slot,
				"source": assignment.source,
				"target": assignment.target.as_ref().map(target_id),
				"depth": assignment.depth,
				"mode": assignment.mode,
			}),
			None => Value::Null,
		})
		.collect::<Vec<_>>();

	let sources = json!({
		"lfo": {
			"rate": state.sources.lfo_rate,
			"shape": state.sources.lfo_shape,
			"points": state.sources.lfo_points,
		},
		"env": {
			"attack": state.sources.env_attack,
			"release": state.sources.env_release,
		},
		"random": {
			"rate": state.sources.random_rate,
			"smooth": state.sources.random_smooth,
		},
		"table": {
			"rate": state.sources.table_rate,
			"shape": state.sources.table_shape,
			"morph": state.sources.table_morph,
		},
	});

	json!({
		"version": state.version,
		"rate": shared.rate.load(Ordering::Relaxed),
		"split_low": state.split_low,
		"split_high": state.split_high,
		"bands": bands,
		"modules": modules,
		"kinds": kinds,
		"targets": targets,
		"assignments": assignments,
		"sources": sources,
		"macros": state.macros,
	})
	.to_string()
}

fn module_json(
	shared: &RackShared,
	assignments: &[Option<Assignment>; 8],
	module: &ModuleState,
	band: usize,
	slot: usize,
) -> Value {
	let controls = shared
		.schemas
		.get(&module.kind)
		.map(|controls| {
			controls
				.iter()
				.map(|control| {
					let modulations = assignments
						.iter()
						.flatten()
						.filter_map(|assignment| match &assignment.target {
							Some(Target::Module {
								module: target_module,
								parameter,
							}) if *target_module == module.id && parameter == &control.id => {
								Some(json!({
									"source": assignment.source,
									"depth": assignment.depth,
									"mode": assignment.mode,
								}))
							},
							_ => None,
						})
						.collect::<Vec<_>>();

					let mut value = serde_json::to_value(control).unwrap_or(Value::Null);

					if let Value::Object(object) = &mut value {
						object.insert("modulations".to_string(), json!(modulations));
					}

					value
				})
				.collect::<Vec<_>>()
		})
		.unwrap_or_default();

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
		"controls": controls,
		"values": &module.params,
	})
}

/// The parameters a modulation assignment can point at.
fn targets_json(state: &RackState, shared: &RackShared) -> Vec<Value> {
	let mut targets = Vec::new();

	for (index, _) in state.bands.iter().enumerate() {
		targets.push(json!({
			"id": format!("band.{index}.gain"),
			"label": format!("{} · Gain", BAND_NAMES[index]),
			"kind": "band",
		}));
		targets.push(json!({
			"id": format!("band.{index}.pan"),
			"label": format!("{} · Pan", BAND_NAMES[index]),
			"kind": "band",
		}));
	}

	for (band_index, band) in state.bands.iter().enumerate() {
		for module in band.slots.iter().flatten() {
			let Some(controls) = shared.schemas.get(&module.kind) else {
				continue;
			};

			for control in controls {
				if control.kind != Kind::Float {
					continue;
				}

				targets.push(json!({
					"id": format!("m{}.{}", module.id, control.id),
					"label": format!(
						"{} · {} · {}",
						BAND_NAMES[band_index],
						module.kind.name(),
						control.label
					),
					"kind": "module",
				}));
			}
		}
	}

	targets
}

/// The page's stable name for a target.
fn target_id(target: &Target) -> String {
	match target {
		Target::Module { module, parameter } => format!("m{module}.{parameter}"),
		Target::BandGain(index) => format!("band.{index}.gain"),
		Target::BandPan(index) => format!("band.{index}.pan"),
	}
}

/// Parses a target name the page sent, rejecting a parameter that is not a
/// modulatable float.
fn parse_target(raw: &str, state: &RackState, shared: &RackShared) -> Option<Target> {
	if let Some(rest) = raw.strip_prefix("band.") {
		let (index, parameter) = rest.split_once('.')?;
		let index = index.parse::<usize>().ok()?;

		if index >= state.bands.len() {
			return None;
		}

		return match parameter {
			"gain" => Some(Target::BandGain(index)),
			"pan" => Some(Target::BandPan(index)),
			_ => None,
		};
	}

	let rest = raw.strip_prefix('m')?;
	let (module, parameter) = rest.split_once('.')?;
	let module = module.parse::<u32>().ok()?;
	let state = state.module(module)?;
	let control = shared
		.schemas
		.get(&state.kind)?
		.iter()
		.find(|control| control.id == parameter)?;

	if control.kind != Kind::Float {
		return None;
	}

	Some(Target::Module {
		module,
		parameter: parameter.to_string(),
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
	} else if let Some(rest) = key.strip_prefix("macro.") {
		apply_macro(state, rest, raw)
	} else if let Some(rest) = key.strip_prefix("source.") {
		apply_source(state, rest, raw)
	} else if let Some(rest) = key.strip_prefix("assign.") {
		apply_assignment(state, shared, rest, raw)
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
		"move" => {
			let Some((target_band, target_slot)) = raw.split_once('.') else {
				return false;
			};
			let Ok(target_band) = target_band.parse::<usize>() else {
				return false;
			};
			let Ok(target_slot) = target_slot.parse::<usize>() else {
				return false;
			};

			if target_band >= state.bands.len()
				|| target_slot >= state.bands[target_band].slots.len()
			{
				return false;
			}

			if target_band == band && target_slot == slot {
				return true;
			}

			let Some(module) = state.bands[band].slots[slot].take() else {
				return false;
			};
			let displaced = state.bands[target_band].slots[target_slot].replace(module);
			state.bands[band].slots[slot] = displaced;
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

fn apply_macro(state: &mut RackState, rest: &str, raw: &str) -> bool {
	let Ok(index) = rest.parse::<usize>() else {
		return false;
	};

	let Some(value) = number(raw) else {
		return false;
	};

	let Some(macro_value) = state.macros.get_mut(index) else {
		return false;
	};

	*macro_value = value.clamp(0.0, 1.0) as f32;
	true
}

fn apply_source(state: &mut RackState, rest: &str, raw: &str) -> bool {
	let Some((source, parameter)) = rest.split_once('.') else {
		return false;
	};

	if source == "lfo" && parameter == "points" {
		let Some(points) = parse_lfo_points(raw) else {
			return false;
		};
		state.sources.lfo_points = points;
		state.sources.lfo_shape = 4;
		return true;
	}

	let Some(value) = number(raw) else {
		return false;
	};

	match (source, parameter) {
		("lfo", "rate") => state.sources.lfo_rate = value.clamp(0.01, 20.0) as f32,
		("lfo", "shape") => {
			state.sources.lfo_shape = (value.round() as i32).clamp(0, 4);

			if state.sources.lfo_shape <= 3 {
				state.sources.lfo_points = lfo_preset(state.sources.lfo_shape);
			}
		},
		("env", "attack") => state.sources.env_attack = value.clamp(1.0, 500.0) as f32,
		("env", "release") => {
			state.sources.env_release = value.clamp(1.0, 500.0) as f32;
		},
		("random", "rate") => {
			state.sources.random_rate = value.clamp(0.01, 20.0) as f32;
		},
		("random", "smooth") => {
			state.sources.random_smooth = value.clamp(0.0, 500.0) as f32;
		},
		("table", "rate") => state.sources.table_rate = value.clamp(0.01, 20.0) as f32,
		("table", "shape") => {
			state.sources.table_shape = (value.round() as i32).clamp(0, 3);
		},
		("table", "morph") => {
			state.sources.table_morph = value.clamp(0.0, 1.0) as f32;
		},
		_ => return false,
	}

	true
}

fn parse_lfo_points(raw: &str) -> Option<Vec<(f32, f32, f32)>> {
	let mut points = Vec::new();

	for part in raw.split(';') {
		let part = part.trim();

		if part.is_empty() {
			continue;
		}

		let mut fields = part.split(':');
		let time = fields.next()?.parse::<f32>().ok()?;
		let value = fields.next()?.parse::<f32>().ok()?;
		let bend = fields
			.next()
			.and_then(|bend| bend.parse::<f32>().ok())
			.unwrap_or(0.5);

		if !time.is_finite() || !value.is_finite() || !bend.is_finite() {
			return None;
		}

		points.push((time.clamp(0.0, 1.0), value.clamp(0.0, 1.0), bend.clamp(-10.0, 10.0)));
	}

	points.sort_by(|a, b| a.0.total_cmp(&b.0));
	points.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-4);

	if points.len() < 2 {
		None
	} else {
		Some(points)
	}
}

fn apply_assignment(
	state: &mut RackState,
	shared: &RackShared,
	rest: &str,
	raw: &str,
) -> bool {
	let Some((slot, field)) = rest.split_once('.') else {
		return false;
	};
	let Ok(slot) = slot.parse::<usize>() else {
		return false;
	};

	if slot >= state.assignments.len() {
		return false;
	}

	match field {
		"source" => {
			let Some(source) = source_index(raw) else {
				return false;
			};
			let assignment = state.assignments[slot]
				.get_or_insert_with(Assignment::default);
			assignment.source = source;
			true
		},
		"target" => {
			let Some(target) = parse_target(raw, state, shared) else {
				return false;
			};
			let assignment = state.assignments[slot]
				.get_or_insert_with(Assignment::default);
			assignment.target = Some(target);
			true
		},
		"depth" => {
			let Some(value) = number(raw) else {
				return false;
			};
			let assignment = state.assignments[slot]
				.get_or_insert_with(Assignment::default);
			assignment.depth = value.clamp(-1.0, 1.0) as f32;
			true
		},
		"mode" => {
			let mode = match raw.to_ascii_lowercase().as_str() {
				"center" | "centered" => AssignmentMode::Centered,
				"start" | "frombase" | "from-base" => AssignmentMode::FromBase,
				_ => return false,
			};
			let assignment = state.assignments[slot]
				.get_or_insert_with(Assignment::default);
			assignment.mode = mode;
			true
		},
		"clear" => {
			let Some(clear) = boolean(raw) else {
				return false;
			};
			if clear {
				state.assignments[slot] = None;
			}
			true
		},
		_ => false,
	}
}

fn source_index(raw: &str) -> Option<usize> {
	let lower = raw.to_ascii_lowercase();

	let macro_index = lower
		.strip_prefix("macro")
		.and_then(|index| index.parse::<usize>().ok())
		.filter(|index| (1..=8).contains(index));

	if let Some(index) = macro_index {
		return Some(3 + index);
	}

	match lower.as_str() {
		"lfo" => Some(0),
		"env" | "envelope" => Some(1),
		"random" => Some(2),
		"table" => Some(3),
		_ => None,
	}
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
