//! The JSON documents the page, the cards and a preset file read.

use std::sync::atomic::Ordering;

use i_am_web_gui::Kind;
use serde_json::{Value, json};

use super::{BAND_NAMES, lane};
use crate::module::ModuleKind;
use crate::shared::RackShared;

use crate::state::{Assignment, ModuleState, RackState, SourceKind, Target};

/// The live state as the JSON a preset file carries.
pub(super) fn state_json(shared: &RackShared) -> String {
	let state = shared.lock();
	serde_json::to_string_pretty(&*state).unwrap_or_else(|_| "{}".to_string())
}

pub(super) fn modulation_json(shared: &RackShared) -> String {
	// The macros and automation slots are host parameters: the DAW writes them
	// and the page has to see that without reloading the whole rack.
	// Read them from the host parameter map rather than the state: the host can
	// write a macro while the transport is stopped, and the page should still
	// follow. The audio thread copies the map into the state every few blocks.
	let host_value = |name: String, fallback: f32| {
		shared
			.params
			.get(&name)
			.map(|param| match param.load(Ordering::Relaxed) {
				i_am_dsp::prelude::SetValue::Float(value) => value,
				_ => fallback,
			})
			.unwrap_or(fallback)
	};

	let state = shared.lock();
	let macros = state
		.macros
		.iter()
		.enumerate()
		.map(|(index, value)| host_value(format!("macro{}", index + 1), *value))
		.collect::<Vec<_>>();
	let automation = state
		.automation_values
		.iter()
		.enumerate()
		.map(|(index, value)| host_value(format!("automation{}", index + 1), *value))
		.collect::<Vec<_>>();
	drop(state);

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

	json!({
		"sources": sources,
		"macros": macros,
		"automation": automation,
	})
	.to_string()
}

pub(crate) fn rack_json(shared: &RackShared) -> String {
	let state = shared.lock();
	let mut modules = Vec::new();

	for (band_index, band) in state
		.bands
		.iter()
		.chain(std::iter::once(&state.parallel))
		.enumerate()
	{
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
		.chain(std::iter::once(&state.parallel))
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

	// The add menu is alphabetical: the catalog's order is a build order, not
	// something a person should have to scan linearly.
	let mut order = ModuleKind::ALL.to_vec();
	order.sort_by_key(|kind| kind.name().to_lowercase());

	let kinds = order
		.into_iter()
		.map(|kind| {
			json!({
				"name": kind.name(),
				"accent": kind.accent(),
				"hero": crate::module::hero(kind),
				"group": crate::module::group(kind),
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

	let history = shared
		.history
		.lock()
		.unwrap_or_else(|poisoned| poisoned.into_inner());
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
					"transpose": settings.midi_transpose,
					"fine": settings.midi_fine,
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
		"macro_names": state.macro_names,
		"history": {
			"undo": history.can_undo(),
			"redo": history.can_redo(),
		},
		"ab": {
			"active": state.ab.letter(),
		},
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

pub(super) fn module_json(
	shared: &RackShared,
	assignments: &[Option<Assignment>],
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
								}) if *target_module == module.id && parameter == &control.id => Some(json!({
									"slot": slot,
									"source": assignment.source,
									"depth": assignment.depth,
									"mode": assignment.mode,
								})),
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
		"note": crate::module::shows_notes(module.kind),
		"meter": crate::module::meter(module.kind),
		"ir": crate::module::needs_ir(module.kind),
		"script": crate::module::needs_script(module.kind),
		"history": module.kind == crate::module::ModuleKind::History,
		"scope": match crate::module::audio_scope(module.kind) {
			i_am_web_gui::Channels::None => serde_json::Value::Null,
			i_am_web_gui::Channels::Mono => "mono".into(),
			i_am_web_gui::Channels::Stereo => "stereo".into(),
		},
		"accent": module.kind.accent(),
		"hero": crate::module::hero(module.kind),
		"mode": module.kind.mode(),
		"header": module.kind.header(),
		"visible": module.kind.visible(),
		"switches": module
			.kind
			.switches()
			.iter()
			.map(|row| json!({
				"control": row.control,
				"on": row.on,
				"off": row.off,
			}))
			.collect::<Vec<_>>(),
		"bypass": module.bypass,
		"group": module.group,
		"controls": controls,
		"values": &module.params,
	})
}

/// The parameters a modulation assignment can point at.
pub(super) fn targets_json(state: &RackState, shared: &RackShared) -> Vec<Value> {
	let mut targets = Vec::new();

	for (index, _) in state
		.bands
		.iter()
		.chain(std::iter::once(&state.parallel))
		.enumerate()
	{
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

	for (band_index, band) in state
		.bands
		.iter()
		.chain(std::iter::once(&state.parallel))
		.enumerate()
	{
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
pub(super) fn target_id(target: &Target) -> String {
	match target {
		Target::Module { module, parameter } => format!("m{module}.{parameter}"),
		Target::BandGain(index) => format!("band.{index}.gain"),
		Target::BandPan(index) => format!("band.{index}.pan"),
	}
}

/// Parses a target name the page sent, rejecting a parameter that is not a
/// modulatable float.
pub(super) fn parse_target(raw: &str, state: &RackState, shared: &RackShared) -> Option<Target> {
	if let Some(rest) = raw.strip_prefix("band.") {
		let (index, parameter) = rest.split_once('.')?;
		let index = index.parse::<usize>().ok()?;

		lane(state, index)?;

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
