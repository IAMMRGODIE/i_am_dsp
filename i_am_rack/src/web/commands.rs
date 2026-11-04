//! The command route: one line per edit, applied under one lock.

use std::sync::atomic::Ordering;

use i_am_plugin::wry::ProtocolResponse;

use crate::shared::RackShared;
use crate::state::{RackState, STATE_VERSION};

use super::import::{commit, commit_whole, import_state};
use super::patch::apply_line;

pub(super) fn command(body: &[u8], shared: &RackShared) -> ProtocolResponse {
	let text = String::from_utf8_lossy(body);
	let rate = shared.rate.load(Ordering::Relaxed);
	let mut state = shared.lock();
	shared.pull_params(&mut state);

	if let Some(response) = special_command(text.trim(), shared, &mut state, rate) {
		return response;
	}

	let before = state.clone();
	let mut ok = true;
	let mut signature = Vec::new();

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
			signature.push(key.trim().to_string());
			state.normalize_slots();
		} else {
			ok = false;
		}
	}

	// A batch that changed nothing (a repeated slider value, a rejected line)
	// must not add an undo step nobody can get back to.
	if *state != before {
		signature.sort();
		signature.dedup();

		shared
			.history
			.lock()
			.unwrap_or_else(|poisoned| poisoned.into_inner())
			.record(before, signature.join(","));
	}

	commit(shared, &state);

	command_response(ok)
}

/// The commands that replace the state as a whole instead of editing a field.
///
/// They travel alone in the body: undo, redo, A/B switching, and a preset file
/// (a compact JSON document after `preset.load=`).
pub(super) fn special_command(
	text: &str,
	shared: &RackShared,
	state: &mut RackState,
	rate: usize,
) -> Option<ProtocolResponse> {
	let (key, raw) = text.split_once('=')?;

	match key.trim() {
		"history.undo" => {
			let changed = shared
				.history
				.lock()
				.unwrap_or_else(|poisoned| poisoned.into_inner())
				.undo(state);

			if changed {
				commit_whole(shared, state);
			}

			Some(command_response(true))
		}
		"history.redo" => {
			let changed = shared
				.history
				.lock()
				.unwrap_or_else(|poisoned| poisoned.into_inner())
				.redo(state);

			if changed {
				commit_whole(shared, state);
			}

			Some(command_response(true))
		}
		"ab.toggle" => {
			state.toggle_ab();

			// A/B is a compare view, not an edit: an undo step from the other
			// side's timeline would restore a state the user cannot see.
			shared.clear_history();
			commit_whole(shared, state);
			Some(command_response(true))
		}
		"preset.init" => {
			let before = std::mem::replace(state, RackState::initial());
			shared
				.history
				.lock()
				.unwrap_or_else(|poisoned| poisoned.into_inner())
				.record_replacement(before);
			commit_whole(shared, state);
			Some(command_response(true))
		}
		"preset.load" => {
			let imported = serde_json::from_str::<RackState>(raw.trim())
				.ok()
				.filter(|incoming| incoming.version <= STATE_VERSION)
				.and_then(|incoming| import_state(incoming, rate, shared));

			let Some(imported) = imported else {
				return Some(command_response(false));
			};

			let before = std::mem::replace(state, imported);
			shared
				.history
				.lock()
				.unwrap_or_else(|poisoned| poisoned.into_inner())
				.record_replacement(before);
			commit_whole(shared, state);
			Some(command_response(true))
		}
		_ => None,
	}
}

pub(super) fn command_response(ok: bool) -> ProtocolResponse {
	ProtocolResponse::json(format!("{{\"ok\":{ok}}}"))
}
