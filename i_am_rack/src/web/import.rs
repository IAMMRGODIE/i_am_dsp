//! Importing a whole state: the migration and the clamp.

use std::sync::atomic::Ordering;

use crate::crossover::sanitize;
use crate::shared::RackShared;
use crate::state::{
	AbState, BandState, MACRO_BASE, RackState, STATE_VERSION, SourceSettings, Target, lfo_preset,
};

/// Applies a state that replaced the whole rack: its scripts reach their
/// workers first, then the audio thread gets the state.
pub(super) fn commit_whole(shared: &RackShared, state: &RackState) {
	shared.apply_scripts(state);
	commit(shared, state);
}

/// Applies whatever the live state just became to the audio thread.
pub(super) fn commit(shared: &RackShared, state: &RackState) {
	shared.sync_params(state);
	shared.dirty.store(true, Ordering::Release);
}

/// Validates a state read from a preset file before it becomes the live rack.
///
/// A hand-edited or older preset may miss parameters the current build has, or
/// carry values outside a control's range; unknown fields are dropped and every
/// known one is clamped, which is also the migration policy for saved projects.
pub(super) fn import_state(
	mut state: RackState,
	rate: usize,
	shared: &RackShared,
) -> Option<RackState> {
	if state.version > STATE_VERSION {
		return None;
	}

	state.migrate();
	normalize_state(&mut state, rate, shared)?;

	// The other A/B side is a full rack too; it gets the same treatment, but
	// never keeps a comparison of its own.
	if let Some(mut other) = state.ab.other.take() {
		if other.version > STATE_VERSION {
			return None;
		}

		other.migrate();
		normalize_state(&mut other, rate, shared)?;
		other.ab = AbState::default();
		state.ab.other = Some(other);
	}

	state.normalize_slots();
	Some(state)
}

/// Clamps every editable field of a (possibly old) state to what this build
/// accepts, and drops references that no longer resolve.
pub(super) fn normalize_state(
	state: &mut RackState,
	rate: usize,
	shared: &RackShared,
) -> Option<()> {
	let (low, high) = sanitize(rate, state.split_low, state.split_high);
	state.split_low = low;
	state.split_high = high;

	for band in &mut state.bands {
		normalize_band(band, shared)?;
	}

	normalize_band(&mut state.parallel, shared)?;

	for value in &mut state.macros {
		*value = value.clamp(0.0, 1.0);
	}

	state.ab.active = state.ab.active.min(1);

	for name in &mut state.macro_names {
		*name = clean_name(name);
	}

	for value in &mut state.automation_values {
		*value = value.clamp(0.0, 1.0);
	}

	for slot in &mut state.source_slots {
		sanitize_settings(&mut slot.settings);
	}

	let lane_count = state.bands.len() + 1;
	let module_ids = state
		.bands
		.iter()
		.chain(std::iter::once(&state.parallel))
		.flat_map(|band| band.slots.iter().flatten())
		.map(|module| module.id)
		.collect::<Vec<_>>();
	let source_ids = state
		.source_slots
		.iter()
		.map(|slot| slot.id as usize)
		.chain(MACRO_BASE..MACRO_BASE + 8)
		.collect::<Vec<_>>();

	for assignment in &mut state.assignments {
		let Some(entry) = assignment else {
			continue;
		};

		entry.depth = entry.depth.clamp(-1.0, 1.0);

		let source_ok = source_ids.contains(&entry.source);
		let target_ok = match &entry.target {
			None => true,
			Some(Target::Module { module, .. }) => module_ids.contains(module),
			Some(Target::BandGain(index) | Target::BandPan(index)) => *index < lane_count,
		};

		if !source_ok || !target_ok {
			*assignment = None;
		}
	}

	Some(())
}

/// Clamps one lane's gain stage and module parameters after an import.
pub(super) fn normalize_band(band: &mut BandState, shared: &RackShared) -> Option<()> {
	band.gain = band.gain.clamp(0.0, 2.0);
	band.pan = band.pan.clamp(-1.0, 1.0);

	for module in band.slots.iter_mut().flatten() {
		let controls = shared.schemas.get(&module.kind)?;

		module
			.params
			.retain(|id, _| controls.iter().any(|control| &control.id == id));

		for (id, value) in crate::module::default_params(module.kind) {
			module.params.entry(id).or_insert(value);
		}

		for (id, value) in &mut module.params {
			if let Some(control) = controls.iter().find(|control| &control.id == id) {
				*value = value.clamp(control.min, control.max);
			}
		}
	}

	Some(())
}

/// The label and preset form of a macro name: trimmed, printable and short.
pub(super) fn clean_name(raw: &str) -> String {
	raw.trim()
		.chars()
		.filter(|character| !character.is_control())
		.take(24)
		.collect()
}

/// Keeps an imported source's knobs inside the ranges the commands enforce.
pub(super) fn sanitize_settings(settings: &mut SourceSettings) {
	settings.lfo_rate = settings.lfo_rate.clamp(0.01, 20.0);
	settings.lfo_shape = settings.lfo_shape.clamp(0, 4);
	settings.lfo_beats = settings.lfo_beats.clamp(0.03125, 32.0);
	settings.env_attack = settings.env_attack.clamp(1.0, 500.0);
	settings.env_release = settings.env_release.clamp(1.0, 500.0);
	settings.random_rate = settings.random_rate.clamp(0.01, 20.0);
	settings.random_smooth = settings.random_smooth.clamp(0.0, 500.0);
	settings.random_beats = settings.random_beats.clamp(0.03125, 32.0);
	settings.table_rate = settings.table_rate.clamp(0.01, 20.0);
	settings.table_shape = settings.table_shape.clamp(0, 3);
	settings.table_morph = settings.table_morph.clamp(0.0, 1.0);
	settings.table_beats = settings.table_beats.clamp(0.03125, 32.0);
	settings.pitch_clarity = settings.pitch_clarity.clamp(0.0, 1.0);
	settings.pitch_glide = settings.pitch_glide.clamp(0.0, 1_000.0);
	settings.midi_mode = settings.midi_mode.clamp(0, 1);
	settings.midi_transpose = settings.midi_transpose.clamp(-24, 24);
	settings.midi_fine = settings.midi_fine.clamp(-100.0, 100.0);

	for (time, value, bend) in &mut settings.lfo_points {
		*time = time.clamp(0.0, 1.0);
		*value = value.clamp(0.0, 1.0);
		*bend = bend.clamp(-10.0, 10.0);
	}

	if settings.lfo_points.len() < 2 {
		settings.lfo_points = lfo_preset(settings.lfo_shape);
	}

	for curve in [
		&mut settings.midi_note_curve,
		&mut settings.midi_velocity_curve,
	] {
		for (x, y) in curve.iter_mut() {
			*x = x.clamp(0.0, 1.0);
			*y = y.clamp(0.0, 1.0);
		}

		if curve.len() < 2 {
			*curve = vec![(0.0, 0.0), (1.0, 1.0)];
		}
	}
}
