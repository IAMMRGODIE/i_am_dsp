//! The edit commands, one family per line prefix.

use super::{lane, lane_mut};
use crate::crossover::sanitize;
use crate::module::ModuleKind;
use crate::shared::RackShared;

use crate::state::{
	Assignment, AssignmentMode, MACRO_BASE, ModuleState, RackState, SourceKind, lfo_preset,
};

use super::import::clean_name;
use super::json::parse_target;

pub(super) fn apply_line(
	state: &mut RackState,
	shared: &RackShared,
	rate: usize,
	key: &str,
	raw: &str,
) -> bool {
	if key == "assign.add" {
		apply_assignment_add(state, shared, raw)
	} else if let Some(rest) = key.strip_prefix("band.") {
		apply_band(state, rest, raw)
	} else if let Some(rest) = key.strip_prefix("split.") {
		apply_split(state, rest, raw, rate)
	} else if let Some(rest) = key.strip_prefix("slot.") {
		apply_slot(state, rest, raw)
	} else if let Some(rest) = key.strip_prefix("group.") {
		apply_group(state, rest)
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

pub(super) fn apply_band(state: &mut RackState, rest: &str, raw: &str) -> bool {
	let Some((band, parameter)) = rest.split_once('.') else {
		return false;
	};

	let Ok(band) = band.parse::<usize>() else {
		return false;
	};

	let Some(band) = lane_mut(state, band) else {
		return false;
	};

	match parameter {
		"gain" => {
			let Some(value) = number(raw) else {
				return false;
			};
			band.gain = value.clamp(0.0, 2.0);
			true
		}
		"pan" => {
			let Some(value) = number(raw) else {
				return false;
			};
			band.pan = value.clamp(-1.0, 1.0);
			true
		}
		"mute" => {
			let Some(value) = boolean(raw) else {
				return false;
			};
			band.mute = value;
			true
		}
		"solo" => {
			let Some(value) = boolean(raw) else {
				return false;
			};
			band.solo = value;
			true
		}
		_ => false,
	}
}

pub(super) fn apply_split(state: &mut RackState, rest: &str, raw: &str, rate: usize) -> bool {
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

pub(super) fn apply_slot(state: &mut RackState, rest: &str, raw: &str) -> bool {
	let Some((band, rest)) = rest.split_once('.') else {
		return false;
	};

	let Ok(band) = band.parse::<usize>() else {
		return false;
	};

	if lane(state, band).is_none() {
		return false;
	}

	if rest == "add" {
		let Some(kind) = module_kind(raw) else {
			return false;
		};

		let id = state.next_id();
		let module = Some(ModuleState::new(id, kind));
		let Some(target) = lane_mut(state, band) else {
			return false;
		};

		if let Some(empty) = target.slots.iter_mut().find(|slot| slot.is_none()) {
			*empty = module;
		} else {
			target.slots.push(module);
		}

		return true;
	}

	let Some((slot, parameter)) = rest.split_once('.') else {
		return false;
	};

	let Ok(slot) = slot.parse::<usize>() else {
		return false;
	};

	let Some(target) = lane(state, band) else {
		return false;
	};

	if slot >= target.slots.len() {
		return false;
	}

	match parameter {
		"kind" => {
			let Some(kind) = module_kind(raw) else {
				return false;
			};

			let new_id = state.next_id();
			let (id, bypass, same) = {
				let existing = lane(state, band)
					.and_then(|lane| lane.slots.get(slot))
					.and_then(|slot| slot.as_ref());

				match existing {
					Some(module) => (module.id, module.bypass, module.kind == kind),
					None => (new_id, false, false),
				}
			};

			if same {
				return true;
			}

			let mut module = ModuleState::new(id, kind);
			module.bypass = bypass;
			let Some(target) = lane_mut(state, band) else {
				return false;
			};
			target.slots[slot] = Some(module);
			true
		}
		"clear" => {
			let Some(clear) = boolean(raw) else {
				return false;
			};

			if clear {
				let Some(target) = lane_mut(state, band) else {
					return false;
				};
				target.slots[slot] = None;
			}

			true
		}
		"duplicate" => {
			let Some(mut copy) = lane(state, band)
				.and_then(|lane| lane.slots.get(slot))
				.and_then(|slot| slot.clone())
			else {
				return false;
			};

			copy.id = state.next_id();
			copy.group = None;
			let Some(target) = lane_mut(state, band) else {
				return false;
			};
			target.slots.insert(slot + 1, Some(copy));
			true
		}
		"copy" => {
			let Some((target_band, target_slot)) = raw.split_once('.') else {
				return false;
			};
			let Ok(target_band) = target_band.parse::<usize>() else {
				return false;
			};
			let Ok(target_slot) = target_slot.parse::<usize>() else {
				return false;
			};

			let Some(mut copy) = lane(state, band)
				.and_then(|lane| lane.slots.get(slot))
				.and_then(|slot| slot.clone())
			else {
				return false;
			};
			let Some(target_lane) = lane(state, target_band) else {
				return false;
			};

			if target_slot >= target_lane.slots.len() {
				return false;
			}

			copy.id = state.next_id();
			copy.group = None;

			let Some(target) = lane_mut(state, target_band) else {
				return false;
			};
			target.slots.insert(target_slot, Some(copy));
			true
		}
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

			let Some(target_lane) = lane(state, target_band) else {
				return false;
			};

			if target_slot >= target_lane.slots.len() {
				return false;
			}

			if target_band == band && target_slot == slot {
				return true;
			}

			let Some(module) = lane_mut(state, band)
				.and_then(|lane| lane.slots.get_mut(slot))
				.and_then(|slot| slot.take())
			else {
				return false;
			};
			let Some(displaced) = lane_mut(state, target_band)
				.and_then(|lane| lane.slots.get_mut(target_slot))
				.map(|slot| slot.replace(module))
			else {
				return false;
			};

			if let Some(source) = lane_mut(state, band).and_then(|lane| lane.slots.get_mut(slot)) {
				*source = displaced;
			}

			true
		}
		_ => false,
	}
}

/// Joins a module to the group on its left, or leaves the group it is in.
pub(super) fn apply_group(state: &mut RackState, rest: &str) -> bool {
	let Some((band, slot)) = rest.split_once('.') else {
		return false;
	};

	let (Ok(band), Ok(slot)) = (band.parse::<usize>(), slot.parse::<usize>()) else {
		return false;
	};

	let Some(lane) = lane(state, band) else {
		return false;
	};

	if slot >= lane.slots.len() {
		return false;
	}

	let Some(module) = lane.slots[slot].as_ref() else {
		return false;
	};

	let next_group = state.next_group_id();

	if let Some(group) = module.group {
		let members = lane
			.slots
			.iter()
			.filter(|module| {
				module
					.as_ref()
					.is_some_and(|module| module.group == Some(group))
			})
			.count();

		let Some(lane) = lane_mut(state, band) else {
			return false;
		};

		if members <= 2 {
			for module in lane.slots.iter_mut().flatten() {
				if module.group == Some(group) {
					module.group = None;
				}
			}
		} else if let Some(module) = lane.slots[slot].as_mut() {
			module.group = None;
		}

		return true;
	}

	if slot == 0 {
		return false;
	}

	let previous = lane.slots[slot - 1]
		.as_ref()
		.and_then(|module| module.group);
	let Some(lane) = lane_mut(state, band) else {
		return false;
	};

	let group = match previous {
		Some(group) => group,
		None => {
			if let Some(module) = lane.slots[slot - 1].as_mut() {
				module.group = Some(next_group);
			}

			next_group
		}
	};

	if let Some(module) = lane.slots[slot].as_mut() {
		module.group = Some(group);
	}

	true
}

pub(super) fn apply_module(
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

	module
		.params
		.insert(parameter.to_string(), value.clamp(control.min, control.max));
	true
}

pub(super) fn apply_automation(
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
		}
		"clear" => {
			let Some(clear) = boolean(raw) else {
				return false;
			};
			if clear {
				state.automation[slot] = None;
			}
		}
		"value" => {
			let Some(value) = number(raw) else {
				return false;
			};
			state.automation_values[slot] = value.clamp(0.0, 1.0) as f32;
		}
		_ => return false,
	}

	true
}

pub(super) fn apply_macro(state: &mut RackState, rest: &str, raw: &str) -> bool {
	if let Some((index, "name")) = rest.split_once('.') {
		let Ok(index) = index.parse::<usize>() else {
			return false;
		};

		let Some(name) = state.macro_names.get_mut(index) else {
			return false;
		};

		*name = clean_name(raw);
		return true;
	}

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

pub(super) fn apply_source(state: &mut RackState, rest: &str, raw: &str) -> bool {
	if rest == "add" {
		let Some(kind) = source_kind(raw) else {
			return false;
		};
		let Some(id) = state.next_source_id() else {
			return false;
		};
		state
			.source_slots
			.push(crate::state::SourceSlot::new(id, kind));
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
			if assignment
				.as_ref()
				.is_some_and(|assignment| assignment.source == id as usize)
			{
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
		state
			.source_slot_mut(id)
			.expect("the slot was just checked")
			.bypass = flag;
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
		}
		(SourceKind::Lfo, "sync") => {
			let Some(flag) = boolean(raw) else {
				return false;
			};
			slot.settings.lfo_sync = flag;
		}
		(SourceKind::Lfo, "beats") => {
			slot.settings.lfo_beats = value.clamp(0.03125, 32.0) as f32;
		}
		(SourceKind::Envelope, "attack") => {
			slot.settings.env_attack = value.clamp(1.0, 500.0) as f32;
		}
		(SourceKind::Envelope, "release") => {
			slot.settings.env_release = value.clamp(1.0, 500.0) as f32;
		}
		(SourceKind::Random, "rate") => {
			slot.settings.random_rate = value.clamp(0.01, 20.0) as f32;
		}
		(SourceKind::Random, "smooth") => {
			slot.settings.random_smooth = value.clamp(0.0, 500.0) as f32;
		}
		(SourceKind::Random, "sync") => {
			let Some(flag) = boolean(raw) else {
				return false;
			};
			slot.settings.random_sync = flag;
		}
		(SourceKind::Random, "beats") => {
			slot.settings.random_beats = value.clamp(0.03125, 32.0) as f32;
		}
		(SourceKind::Table, "rate") => slot.settings.table_rate = value.clamp(0.01, 20.0) as f32,
		(SourceKind::Table, "shape") => {
			slot.settings.table_shape = (value.round() as i32).clamp(0, 3);
		}
		(SourceKind::Table, "morph") => {
			slot.settings.table_morph = value.clamp(0.0, 1.0) as f32;
		}
		(SourceKind::Table, "sync") => {
			let Some(flag) = boolean(raw) else {
				return false;
			};
			slot.settings.table_sync = flag;
		}
		(SourceKind::Table, "beats") => {
			slot.settings.table_beats = value.clamp(0.03125, 32.0) as f32;
		}
		(SourceKind::Pitch, "clarity") => {
			slot.settings.pitch_clarity = value.clamp(0.0, 1.0) as f32;
		}
		(SourceKind::Pitch, "glide") => {
			slot.settings.pitch_glide = value.clamp(0.0, 1000.0) as f32;
		}
		(SourceKind::Midi, "mode") => {
			slot.settings.midi_mode = (value.round() as i32).clamp(0, 1);
		}
		(SourceKind::Midi, "transpose") => {
			slot.settings.midi_transpose = (value.round() as i32).clamp(-24, 24);
		}
		(SourceKind::Midi, "fine") => {
			slot.settings.midi_fine = value.clamp(-100.0, 100.0) as f32;
		}
		_ => return false,
	}

	true
}

pub(super) fn source_kind(raw: &str) -> Option<SourceKind> {
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

pub(super) fn parse_curve(raw: &str) -> Option<Vec<(f32, f32)>> {
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

	if points.len() < 2 { None } else { Some(points) }
}

pub(super) fn parse_lfo_points(raw: &str) -> Option<Vec<(f32, f32, f32)>> {
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

		points.push((
			time.clamp(0.0, 1.0),
			value.clamp(0.0, 1.0),
			bend.clamp(-10.0, 10.0),
		));
	}

	points.sort_by(|a, b| a.0.total_cmp(&b.0));
	points.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-4);

	if points.len() < 2 { None } else { Some(points) }
}

pub(super) fn apply_assignment_add(state: &mut RackState, shared: &RackShared, raw: &str) -> bool {
	let mut assignment = Assignment::default();

	if raw != "1" && !raw.is_empty() {
		let Some((source, target)) = raw.split_once('|') else {
			return false;
		};
		let Some(source) = source_index(state, source) else {
			return false;
		};
		let Some(target) = parse_target(target, state, shared) else {
			return false;
		};

		assignment.source = source;
		assignment.target = Some(target);
	}

	// One source per target: a second row for the same pair would double the
	// same modulation and make the knob's arc meaningless.
	if let Some(target) = assignment.target.as_ref() {
		let duplicate = state.assignments.iter().flatten().any(|other| {
			other.source == assignment.source && other.target.as_ref() == Some(target)
		});

		if duplicate {
			return true;
		}
	}

	state.assignments.push(Some(assignment));
	true
}

/// After an edit, no other row may address the same target from the same
/// source: the newest edit wins and the older row loses its target.
pub(super) fn release_duplicates(state: &mut RackState, slot: usize) {
	let Some(Some(edited)) = state.assignments.get(slot) else {
		return;
	};
	let Some(target) = edited.target.clone() else {
		return;
	};
	let source = edited.source;

	for (index, assignment) in state.assignments.iter_mut().enumerate() {
		if index == slot {
			continue;
		}

		if let Some(other) = assignment
			&& other.source == source
			&& other.target.as_ref() == Some(&target)
		{
			other.target = None;
		}
	}
}

pub(super) fn apply_assignment(
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
			let assignment = state.assignments[slot].get_or_insert_with(Assignment::default);
			assignment.source = source;
			release_duplicates(state, slot);
			true
		}
		"target" => {
			let Some(target) = parse_target(raw, state, shared) else {
				return false;
			};
			let assignment = state.assignments[slot].get_or_insert_with(Assignment::default);
			assignment.target = Some(target);
			release_duplicates(state, slot);
			true
		}
		"depth" => {
			let Some(value) = number(raw) else {
				return false;
			};
			let assignment = state.assignments[slot].get_or_insert_with(Assignment::default);
			assignment.depth = value.clamp(-1.0, 1.0) as f32;
			true
		}
		"mode" => {
			let mode = match raw.to_ascii_lowercase().as_str() {
				"center" | "centered" => AssignmentMode::Centered,
				"start" | "frombase" | "from-base" => AssignmentMode::FromBase,
				_ => return false,
			};
			let assignment = state.assignments[slot].get_or_insert_with(Assignment::default);
			assignment.mode = mode;
			true
		}
		"clear" => {
			let Some(clear) = boolean(raw) else {
				return false;
			};
			if clear {
				state.assignments[slot] = None;
			}
			true
		}
		_ => false,
	}
}

pub(super) fn source_index(state: &RackState, raw: &str) -> Option<usize> {
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

pub(super) fn number(raw: &str) -> Option<f64> {
	raw.parse::<f64>().ok().filter(|value| value.is_finite())
}

pub(super) fn boolean(raw: &str) -> Option<bool> {
	match raw.to_ascii_lowercase().as_str() {
		"1" | "true" | "on" | "yes" => Some(true),
		"0" | "false" | "off" | "no" => Some(false),
		_ => None,
	}
}

pub(super) fn module_kind(raw: &str) -> Option<ModuleKind> {
	ModuleKind::ALL
		.into_iter()
		.find(|kind| kind.name().eq_ignore_ascii_case(raw))
}
