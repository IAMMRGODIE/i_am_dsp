//! Undo, A/B and the preset file.

use super::*;

/// One command is one undo step; redo walks the same stack forward again.
#[test]
fn history_undoes_and_redoes_an_edit() {
	let rack = Rack::default();
	let before = rack.state().bands[0].slots[0]
		.as_ref()
		.expect("the low band's filter")
		.params["cutoff"];

	command(&rack, "0.cutoff=3000");

	assert_eq!(
		rack.state().bands[0].slots[0].as_ref().unwrap().params["cutoff"],
		3_000.0
	);

	command(&rack, "history.undo=1");

	assert_eq!(
		rack.state().bands[0].slots[0].as_ref().unwrap().params["cutoff"],
		before
	);

	command(&rack, "history.redo=1");

	assert_eq!(
		rack.state().bands[0].slots[0].as_ref().unwrap().params["cutoff"],
		3_000.0
	);
}

/// An undo with nothing to undo leaves the rack alone and still answers ok.
#[test]
fn an_empty_undo_is_a_no_op() {
	let rack = Rack::default();
	let before = rack.state();

	let response = command(&rack, "history.undo=1");

	assert_eq!(response.body, b"{\"ok\":true}");
	assert_eq!(rack.state(), before);
}

/// A/B keeps both sides; switching stores the live one and loads the other.
#[test]
fn ab_compare_keeps_both_sides() {
	let rack = Rack::default();

	command(&rack, "0.cutoff=4000");
	command(&rack, "ab.toggle=1");

	let json: serde_json::Value = serde_json::from_slice(&get(&rack, "/rack").body).expect("JSON");
	assert_eq!(json["ab"]["active"], "B");

	command(&rack, "0.cutoff=800");
	command(&rack, "ab.toggle=1");

	let json: serde_json::Value = serde_json::from_slice(&get(&rack, "/rack").body).expect("JSON");
	assert_eq!(json["ab"]["active"], "A");
	assert_eq!(
		rack.state().bands[0].slots[0].as_ref().unwrap().params["cutoff"],
		4_000.0
	);

	command(&rack, "ab.toggle=1");

	assert_eq!(
		rack.state().bands[0].slots[0].as_ref().unwrap().params["cutoff"],
		800.0
	);
}

/// A saved state loaded as a preset becomes the live rack.
#[test]
fn a_preset_file_replaces_the_rack() {
	let rack = Rack::default();
	let mut preset = RackState::initial();
	preset.split_low = 200.0;
	preset.split_high = 3_000.0;
	preset.bands[0].slots[0]
		.as_mut()
		.expect("the low band's filter")
		.params
		.insert("cutoff".to_string(), 4_321.0);
	preset.macros[2] = 0.75;

	let body = format!(
		"preset.load={}",
		serde_json::to_string(&preset).expect("the preset JSON")
	);
	let response = command(&rack, &body);

	assert_eq!(response.body, b"{\"ok\":true}");

	let state = rack.state();

	assert_eq!(state.split_low, 200.0);
	assert_eq!(state.split_high, 3_000.0);
	assert_eq!(
		state.bands[0].slots[0].as_ref().unwrap().params["cutoff"],
		4_321.0
	);
	assert_eq!(state.macros[2], 0.75);
}

/// A file that is not a state is rejected without touching the rack.
#[test]
fn a_broken_preset_is_rejected() {
	let rack = Rack::default();
	let before = rack.state();

	let response = command(&rack, "preset.load=not a state");

	assert_eq!(response.body, b"{\"ok\":false}");
	assert_eq!(rack.state(), before);
}

/// Init resets the rack, and that reset is itself undoable.
#[test]
fn preset_init_returns_to_the_default_rack() {
	let rack = Rack::default();

	command(&rack, "0.cutoff=3000");
	command(&rack, "preset.init=1");

	assert_eq!(rack.state(), RackState::initial());

	command(&rack, "history.undo=1");

	assert_eq!(
		rack.state().bands[0].slots[0].as_ref().unwrap().params["cutoff"],
		3_000.0
	);
}

/// The page's Save button reads exactly the state the host would save.
#[test]
fn the_state_route_returns_a_preset_document() {
	let rack = Rack::default();

	let response = get(&rack, "/state");
	let state: RackState = serde_json::from_slice(&response.body).expect("the state JSON");

	assert_eq!(state, rack.state());
}

/// The page learns whether undo/redo are available and which side is live.
#[test]
fn the_rack_route_reports_undo_and_ab() {
	let rack = Rack::default();

	let json: serde_json::Value = serde_json::from_slice(&get(&rack, "/rack").body).expect("JSON");
	assert_eq!(json["history"]["undo"], false);
	assert_eq!(json["history"]["redo"], false);
	assert_eq!(json["ab"]["active"], "A");

	command(&rack, "0.cutoff=3000");

	let json: serde_json::Value = serde_json::from_slice(&get(&rack, "/rack").body).expect("JSON");
	assert_eq!(json["history"]["undo"], true);
	assert_eq!(json["history"]["redo"], false);

	command(&rack, "history.undo=1");

	let json: serde_json::Value = serde_json::from_slice(&get(&rack, "/rack").body).expect("JSON");
	assert_eq!(json["history"]["undo"], false);
	assert_eq!(json["history"]["redo"], true);
}

/// A knob drag lands as many commands; undo goes back to before the drag.
#[test]
fn a_drag_is_one_undo_step() {
	let rack = Rack::default();
	let before = rack.state().bands[0].slots[0]
		.as_ref()
		.expect("the low band's filter")
		.params["cutoff"];

	command(&rack, "0.cutoff=2000");
	command(&rack, "0.cutoff=2500");
	command(&rack, "0.cutoff=3000");

	command(&rack, "history.undo=1");

	assert_eq!(
		rack.state().bands[0].slots[0].as_ref().unwrap().params["cutoff"],
		before
	);
}

/// A macro can be renamed, and the name reaches the page and the parameter
/// metadata while the stable host identifier stays macroN.
#[test]
fn macro_names_can_be_changed() {
	let rack = Rack::default();

	command(&rack, "macro.0.name=Kick");

	assert_eq!(rack.state().macro_names[0], "Kick");

	let json: serde_json::Value = serde_json::from_slice(&get(&rack, "/rack").body).expect("JSON");
	assert_eq!(json["macro_names"][0], "Kick");

	let parameters = rack.get_parameters();
	let macro1 = parameters
		.iter()
		.find(|parameter| parameter.identifier == "macro1")
		.expect("macro1");

	assert_eq!(macro1.meta.display_name.as_deref(), Some("Kick"));
}

/// An unnamed macro falls back to the page's "M1" label.
#[test]
fn clearing_a_macro_name_restores_the_host_name() {
	let rack = Rack::default();

	command(&rack, "macro.0.name=Kick");
	command(&rack, "macro.0.name=");

	assert_eq!(rack.state().macro_names[0], "");

	let parameters = rack.get_parameters();
	let macro1 = parameters
		.iter()
		.find(|parameter| parameter.identifier == "macro1")
		.expect("macro1");

	assert_eq!(macro1.meta.display_name, None);
}

/// Both A/B sides travel in the host state, so reopening a project keeps them.
#[test]
fn ab_is_saved_with_the_state() {
	let rack = Rack::default();

	command(&rack, "0.cutoff=4000");
	command(&rack, "ab.toggle=1");
	command(&rack, "0.cutoff=800");

	let state = rack.state();

	assert_eq!(state.ab.letter(), "B");
	assert_eq!(
		state.ab.other.as_ref().expect("the A side").bands[0].slots[0]
			.as_ref()
			.unwrap()
			.params["cutoff"],
		4_000.0
	);

	let mut reopened = Rack::default();
	let bytes = to_binary(&state).expect("the saved state");
	assert!(reopened.set_parameter("rack_state", SetValue::Serialized(bytes)));

	let reopened = reopened.state();

	assert_eq!(reopened.ab.letter(), "B");
	assert_eq!(
		reopened.ab.other.as_ref().expect("the A side").bands[0].slots[0]
			.as_ref()
			.unwrap()
			.params["cutoff"],
		4_000.0
	);
}

/// A version-8 file without macro names or A/B still loads and migrates.
#[test]
fn a_version_8_state_migrates() {
	let rack = Rack::default();
	let mut preset = RackState::initial();
	preset.version = 8;

	let mut value = serde_json::to_value(&preset).expect("the state value");
	let object = value.as_object_mut().expect("the state object");
	object.remove("macro_names");
	object.remove("ab");

	let body = format!(
		"preset.load={}",
		serde_json::to_string(&value).expect("the preset JSON")
	);
	let response = command(&rack, &body);

	assert_eq!(response.body, b"{\"ok\":true}");

	let state = rack.state();

	assert_eq!(state.version, crate::state::STATE_VERSION);
	assert_eq!(state.macro_names[0], "");
	assert_eq!(state.ab.letter(), "A");
}

/// A hand-edited state cannot put the A/B index outside its two sides.
#[test]
fn an_out_of_range_ab_side_is_clamped() {
	let mut state = RackState::initial();
	state.ab.active = 5;

	let mut rack = Rack::default();
	let bytes = to_binary(&state).expect("the saved state");
	assert!(rack.set_parameter("rack_state", SetValue::Serialized(bytes)));
	assert_eq!(rack.state().ab.active, 1);

	command(&rack, "ab.toggle=1");
	assert_eq!(rack.state().ab.letter(), "A");
}

/// Transpose moves the MIDI note in semitones; fine moves it in cents.
#[test]
fn midi_transpose_and_fine_shift_the_note() {
	let mut rack = Rack::default();
	assert_eq!(command(&rack, "source.add=midi").body, b"{\"ok\":true}");

	let note = NoteEvent::NoteOn {
		channel: 0,
		note: 60,
		velocity: 0.75,
	};

	assert_eq!(
		command(&rack, "source.4.transpose=12").body,
		b"{\"ok\":true}"
	);

	for _ in 0..128 {
		process_with(&mut rack, [0.0, 0.0], vec![note.clone()]);
	}

	assert!((modulation_value(&rack, 4) - 72.0 / 127.0).abs() < 1e-4);

	assert_eq!(command(&rack, "source.4.fine=50").body, b"{\"ok\":true}");

	for _ in 0..128 {
		process_with(&mut rack, [0.0, 0.0], vec![note.clone()]);
	}

	assert!((modulation_value(&rack, 4) - 72.5 / 127.0).abs() < 1e-4);

	assert_eq!(
		command(&rack, "source.4.transpose=-24").body,
		b"{\"ok\":true}"
	);
	assert_eq!(command(&rack, "source.4.fine=0").body, b"{\"ok\":true}");

	for _ in 0..128 {
		process_with(&mut rack, [0.0, 0.0], vec![note.clone()]);
	}

	assert!((modulation_value(&rack, 4) - 36.0 / 127.0).abs() < 1e-4);
}

/// The transpose and fine knobs are clamped to their published ranges.
#[test]
fn midi_transpose_and_fine_are_clamped() {
	let rack = Rack::default();
	assert_eq!(command(&rack, "source.add=midi").body, b"{\"ok\":true}");
	assert_eq!(
		command(&rack, "source.4.transpose=99").body,
		b"{\"ok\":true}"
	);
	assert_eq!(command(&rack, "source.4.fine=500").body, b"{\"ok\":true}");

	let response = get(&rack, "/rack");
	let json: serde_json::Value = serde_json::from_slice(&response.body).expect("JSON");
	let midi = json["sources"]
		.as_array()
		.expect("sources")
		.iter()
		.find(|source| source["id"] == 4)
		.expect("the MIDI source");

	assert_eq!(midi["settings"]["transpose"], 24);
	assert_eq!(midi["settings"]["fine"], 100.0);
}
