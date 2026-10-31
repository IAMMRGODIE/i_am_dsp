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
use crate::state::{Assignment, AssignmentMode, MACRO_BASE, ModuleState, RackState, SourceKind, Target, lfo_preset};

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
	let live = shared
		.live
		.lock()
		.unwrap_or_else(|poisoned| poisoned.into_inner());
	let sources = live
		.iter()
		.map(|(id, value, phase)| {
			json!({
				"id": id,
				"value": value,
				"phase": phase,
			})
		})
		.collect::<Vec<_>>();

	json!({ "sources": sources }).to_string()
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
				"slots": band
					.slots
					.iter()
					.map(|slot| slot.as_ref().map(|module| module.id))
					.collect::<Vec<_>>(),
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

	let sources = state
		.source_slots
		.iter()
		.map(|slot| {
			let settings = &slot.settings;
			let body = match slot.kind {
				SourceKind::Lfo => json!({
					"rate": settings.lfo_rate,
					"shape": settings.lfo_shape,
					"points": settings.lfo_points,
					"sync": settings.lfo_sync,
					"beats": settings.lfo_beats,
				}),
				SourceKind::Envelope => json!({
					"attack": settings.env_attack,
					"release": settings.env_release,
				}),
				SourceKind::Random => json!({
					"rate": settings.random_rate,
					"smooth": settings.random_smooth,
					"sync": settings.random_sync,
					"beats": settings.random_beats,
				}),
				SourceKind::Table => json!({
					"rate": settings.table_rate,
					"shape": settings.table_shape,
					"morph": settings.table_morph,
					"sync": settings.table_sync,
					"beats": settings.table_beats,
				}),
				SourceKind::Pitch => json!({
					"clarity": settings.pitch_clarity,
					"glide": settings.pitch_glide,
				}),
				SourceKind::Midi => json!({
					"mode": settings.midi_mode,
					"note_curve": settings.midi_note_curve,
					"velocity_curve": settings.midi_velocity_curve,
				}),
			};

			json!({
				"id": slot.id,
				"kind": slot.kind,
				"name": slot.kind.name(),
				"bypass": slot.bypass,
				"settings": body,
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
		"targets": targets,
		"assignments": assignments,
		"sources": sources,
		"macros": state.macros,
		"automation": (0..crate::state::AUTOMATION_SLOTS)
			.map(|slot| json!({
				"slot": slot,
				"target": state.automation[slot].as_ref().map(target_id),
				"value": state.automation_values[slot],
			}))
			.collect::<Vec<_>>(),
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
						.enumerate()
						.filter_map(|(slot, assignment)| {
							let assignment = assignment.as_ref()?;

							match &assignment.target {
								Some(Target::Module {
									module: target_module,
									parameter,
								}) if *target_module == module.id && parameter == &control.id => {
									Some(json!({
										"slot": slot,
										"source": assignment.source,
										"depth": assignment.depth,
										"mode": assignment.mode,
									}))
								},
								_ => None,
							}
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

		if applied {
			state.normalize_slots();
		} else {
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
	} else if let Some(rest) = key.strip_prefix("automation.") {
		apply_automation(state, shared, rest, raw)
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

	let Ok(band) = band.parse::<usize>() else {
		return false;
	};

	if band >= state.bands.len() {
		return false;
	}

	if rest == "add" {
		let Some(kind) = module_kind(raw) else {
			return false;
		};

		let id = state.next_id();
		let module = Some(ModuleState::new(id, kind));

		if let Some(empty) = state.bands[band].slots.iter_mut().find(|slot| slot.is_none()) {
			*empty = module;
		} else {
			state.bands[band].slots.push(module);
		}

		return true;
	}

	let Some((slot, parameter)) = rest.split_once('.') else {
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

fn apply_automation(
	state: &mut RackState,
	shared: &RackShared,
	rest: &str,
	raw: &str,
) -> bool {
	let Some((slot, parameter)) = rest.split_once('.') else {
		return false;
	};
	let Ok(slot) = slot.parse::<usize>() else {
		return false;
	};

	if slot >= crate::state::AUTOMATION_SLOTS {
		return false;
	}

	match parameter {
		"target" => {
			let Some(target) = parse_target(raw, state, shared) else {
				return false;
			};
			state.automation[slot] = Some(target);
		},
		"clear" => {
			let Some(clear) = boolean(raw) else {
				return false;
			};
			if clear {
				state.automation[slot] = None;
			}
		},
		"value" => {
			let Some(value) = number(raw) else {
				return false;
			};
			state.automation_values[slot] = value.clamp(0.0, 1.0) as f32;
		},
		_ => return false,
	}

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
	if rest == "add" {
		let Some(kind) = source_kind(raw) else {
			return false;
		};
		let Some(id) = state.next_source_id() else {
			return false;
		};
		state.source_slots.push(crate::state::SourceSlot::new(id, kind));
		return true;
	}

	let Some((source, parameter)) = rest.split_once('.') else {
		return false;
	};

	let id = if let Ok(id) = source.parse::<u32>() {
		id
	} else {
		let Some(kind) = source_kind(source) else {
			return false;
		};
		let Some(slot) = state.source_slots.iter().find(|slot| slot.kind == kind) else {
			return false;
		};
		slot.id
	};

	if parameter == "move" {
		let Ok(target) = raw.parse::<u32>() else {
			return false;
		};

		if target == id || state.source_slot(target).is_none() {
			return target == id;
		}

		let Some(position) = state.source_slots.iter().position(|slot| slot.id == id) else {
			return false;
		};
		let target_position = state
			.source_slots
			.iter()
			.position(|slot| slot.id == target)
			.unwrap_or(state.source_slots.len());
		let slot = state.source_slots.remove(position);
		state
			.source_slots
			.insert(target_position.min(state.source_slots.len()), slot);
		return true;
	}

	if parameter == "remove" {
		state.source_slots.retain(|slot| slot.id != id);

		for assignment in state.assignments.iter_mut() {
			if assignment.as_ref().is_some_and(|assignment| assignment.source == id as usize) {
				*assignment = None;
			}
		}

		return true;
	}

	let Some(kind) = state.source_slot(id).map(|slot| slot.kind) else {
		return false;
	};

	if parameter == "bypass" {
		let Some(flag) = boolean(raw) else {
			return false;
		};
		state.source_slot_mut(id).expect("the slot was just checked").bypass = flag;
		return true;
	}

	let slot = state
		.source_slot_mut(id)
		.expect("the slot was just checked");

	if parameter == "points" {
		if kind != SourceKind::Lfo {
			return false;
		}
		let Some(points) = parse_lfo_points(raw) else {
			return false;
		};
		slot.settings.lfo_points = points;
		slot.settings.lfo_shape = 4;
		return true;
	}

	if parameter == "note_curve" || parameter == "velocity_curve" {
		if kind != SourceKind::Midi {
			return false;
		}
		let Some(points) = parse_curve(raw) else {
			return false;
		};

		if parameter == "note_curve" {
			slot.settings.midi_note_curve = points;
		} else {
			slot.settings.midi_velocity_curve = points;
		}

		return true;
	}

	let Some(value) = number(raw) else {
		return false;
	};

	match (kind, parameter) {
		(SourceKind::Lfo, "rate") => slot.settings.lfo_rate = value.clamp(0.01, 20.0) as f32,
		(SourceKind::Lfo, "shape") => {
			slot.settings.lfo_shape = (value.round() as i32).clamp(0, 4);

			if slot.settings.lfo_shape <= 3 {
				slot.settings.lfo_points = lfo_preset(slot.settings.lfo_shape);
			}
		},
		(SourceKind::Lfo, "sync") => {
			let Some(flag) = boolean(raw) else {
				return false;
			};
			slot.settings.lfo_sync = flag;
		},
		(SourceKind::Lfo, "beats") => {
			slot.settings.lfo_beats = value.clamp(0.03125, 32.0) as f32;
		},
		(SourceKind::Envelope, "attack") => {
			slot.settings.env_attack = value.clamp(1.0, 500.0) as f32;
		},
		(SourceKind::Envelope, "release") => {
			slot.settings.env_release = value.clamp(1.0, 500.0) as f32;
		},
		(SourceKind::Random, "rate") => {
			slot.settings.random_rate = value.clamp(0.01, 20.0) as f32;
		},
		(SourceKind::Random, "smooth") => {
			slot.settings.random_smooth = value.clamp(0.0, 500.0) as f32;
		},
		(SourceKind::Random, "sync") => {
			let Some(flag) = boolean(raw) else {
				return false;
			};
			slot.settings.random_sync = flag;
		},
		(SourceKind::Random, "beats") => {
			slot.settings.random_beats = value.clamp(0.03125, 32.0) as f32;
		},
		(SourceKind::Table, "rate") => slot.settings.table_rate = value.clamp(0.01, 20.0) as f32,
		(SourceKind::Table, "shape") => {
			slot.settings.table_shape = (value.round() as i32).clamp(0, 3);
		},
		(SourceKind::Table, "morph") => {
			slot.settings.table_morph = value.clamp(0.0, 1.0) as f32;
		},
		(SourceKind::Table, "sync") => {
			let Some(flag) = boolean(raw) else {
				return false;
			};
			slot.settings.table_sync = flag;
		},
		(SourceKind::Table, "beats") => {
			slot.settings.table_beats = value.clamp(0.03125, 32.0) as f32;
		},
		(SourceKind::Pitch, "clarity") => {
			slot.settings.pitch_clarity = value.clamp(0.0, 1.0) as f32;
		},
		(SourceKind::Pitch, "glide") => {
			slot.settings.pitch_glide = value.clamp(0.0, 1000.0) as f32;
		},
		(SourceKind::Midi, "mode") => {
			slot.settings.midi_mode = (value.round() as i32).clamp(0, 1);
		},
		_ => return false,
	}

	true
}

fn source_kind(raw: &str) -> Option<SourceKind> {
	match raw.to_ascii_lowercase().as_str() {
		"lfo" => Some(SourceKind::Lfo),
		"env" | "envelope" => Some(SourceKind::Envelope),
		"random" => Some(SourceKind::Random),
		"table" => Some(SourceKind::Table),
		"pitch" => Some(SourceKind::Pitch),
		"midi" => Some(SourceKind::Midi),
		_ => None,
	}
}

fn parse_curve(raw: &str) -> Option<Vec<(f32, f32)>> {
	let mut points = Vec::new();

	for part in raw.split(';') {
		let part = part.trim();

		if part.is_empty() {
			continue;
		}

		let mut fields = part.split(':');
		let x = fields.next()?.parse::<f32>().ok()?;
		let y = fields.next()?.parse::<f32>().ok()?;

		if !x.is_finite() || !y.is_finite() {
			return None;
		}

		points.push((x.clamp(0.0, 1.0), y.clamp(0.0, 1.0)));
	}

	points.sort_by(|a, b| a.0.total_cmp(&b.0));
	points.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-4);

	if points.len() < 2 {
		None
	} else {
		Some(points)
	}
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
			let Some(source) = source_index(state, raw) else {
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

fn source_index(state: &RackState, raw: &str) -> Option<usize> {
	let lower = raw.to_ascii_lowercase();

	let macro_index = lower
		.strip_prefix("macro")
		.and_then(|index| index.parse::<usize>().ok())
		.filter(|index| (1..=8).contains(index));

	if let Some(index) = macro_index {
		return Some(MACRO_BASE + index - 1);
	}

	if let Ok(id) = lower.parse::<u32>() {
		return state.source_slot(id).map(|slot| slot.id as usize);
	}

	let kind = source_kind(&lower)?;
	state
		.source_slots
		.iter()
		.find(|slot| slot.kind == kind)
		.map(|slot| slot.id as usize)
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
