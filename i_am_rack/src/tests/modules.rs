//! Slots, lanes and module copying.

use super::*;

/// Duplicating a module keeps its kind and values and gives it a fresh id.
#[test]
fn a_module_can_be_duplicated() {
	let rack = Rack::default();

	command(&rack, "0.cutoff=2500");
	command(&rack, "slot.0.0.duplicate=1");

	let state = rack.state();
	let band = &state.bands[0];

	assert_eq!(band.slots.len(), 3);

	let original = band.slots[0].as_ref().expect("the original");
	let copy = band.slots[1].as_ref().expect("the copy");

	assert_eq!(copy.kind, original.kind);
	assert_ne!(copy.id, original.id);
	assert_eq!(copy.params["cutoff"], 2_500.0);
	assert_eq!(band.slots[2], None);
}

/// An empty slot has nothing to copy.
#[test]
fn duplicating_an_empty_slot_is_rejected() {
	let rack = Rack::default();
	let before = rack.state();

	let response = command(&rack, "slot.0.1.duplicate=1");

	assert_eq!(response.body, b"{\"ok\":false}");
	assert_eq!(rack.state(), before);
}

/// The copy owns its parameters: editing it leaves the original alone.
#[test]
fn a_duplicate_has_its_own_parameters() {
	let rack = Rack::default();

	command(&rack, "0.cutoff=2500");
	command(&rack, "slot.0.0.duplicate=1");

	let copy_id = rack.state().bands[0].slots[1]
		.as_ref()
		.expect("the copy")
		.id;

	command(&rack, &format!("{copy_id}.cutoff=800"));

	let state = rack.state();

	assert_eq!(
		state.bands[0].slots[0].as_ref().unwrap().params["cutoff"],
		2_500.0
	);
	assert_eq!(
		state.bands[0].slots[1].as_ref().unwrap().params["cutoff"],
		800.0
	);
}

/// Duplicating a bypassed module keeps the bypass on the copy.
#[test]
fn a_duplicate_keeps_the_bypass_state() {
	let rack = Rack::default();

	command(&rack, "0.__bypass=1");
	command(&rack, "slot.0.0.duplicate=1");

	let state = rack.state();

	assert!(state.bands[0].slots[0].as_ref().unwrap().bypass);
	assert!(state.bands[0].slots[1].as_ref().unwrap().bypass);
}

/// An empty parallel lane adds nothing: the three bands stay flat.
#[test]
fn an_empty_parallel_lane_is_silent() {
	let mut rack = Rack::default();
	let mut last = [0.0_f32; 2];

	for _ in 0..2_048 {
		last = process(&mut rack, [0.25, 0.25]);
	}

	assert!(
		(last[0] - 0.25).abs() < 0.01 && (last[1] - 0.25).abs() < 0.01,
		"the parallel lane doubled the signal: {last:?}"
	);
}

/// A module in the parallel lane processes the full-range input once.
#[test]
fn a_parallel_lane_holds_a_full_range_module() {
	let mut rack = Rack::default();

	let response = command(
		&rack,
		"band.0.mute=1
band.1.mute=1
band.2.mute=1
slot.3.add=Gain",
	);
	assert_eq!(response.body, b"{\"ok\":true}");

	let mut last = [0.0_f32; 2];

	for _ in 0..2_048 {
		last = process(&mut rack, [0.25, 0.25]);
	}

	assert!(
		(last[0] - 0.25).abs() < 0.01 && (last[1] - 0.25).abs() < 0.01,
		"the parallel lane output {last:?}"
	);

	let state = rack.state();

	assert!(state.parallel.slots[0].is_some());
	assert!(state.parallel.slots[1].is_none());
}

/// The parallel lane takes the same band commands and modulation targets.
#[test]
fn the_parallel_lane_answers_band_commands() {
	let rack = Rack::default();

	assert_eq!(command(&rack, "band.3.gain=0.5").body, b"{\"ok\":true}");
	assert_eq!(command(&rack, "band.3.pan=-0.5").body, b"{\"ok\":true}");
	assert_eq!(command(&rack, "band.3.mute=1").body, b"{\"ok\":true}");

	let state = rack.state();

	assert_eq!(state.parallel.gain, 0.5);
	assert_eq!(state.parallel.pan, -0.5);
	assert!(state.parallel.mute);

	assert_eq!(
		command(&rack, "assign.add=0|band.3.gain").body,
		b"{\"ok\":true}"
	);

	let json: serde_json::Value = serde_json::from_slice(&get(&rack, "/rack").body).expect("JSON");
	assert!(
		json["targets"]
			.as_array()
			.expect("targets")
			.iter()
			.any(|target| target["id"] == "band.3.gain")
	);
}

/// Automation targets address the full-range lane like any band.
#[test]
fn automation_can_move_the_parallel_gain() {
	let mut rack = Rack::default();

	let response = command(
		&rack,
		"band.0.mute=1
band.1.mute=1
band.2.mute=1
slot.3.add=Gain
automation.0.target=band.3.gain
automation.0.value=0.25",
	);
	assert_eq!(response.body, b"{\"ok\":true}");

	let mut last = [0.0_f32; 2];

	for _ in 0..2_048 {
		last = process(&mut rack, [0.25, 0.25]);
	}

	assert!(
		(last[0] - 0.125).abs() < 0.01 && (last[1] - 0.125).abs() < 0.01,
		"the automated parallel gain left {last:?}"
	);
}

/// The Delay module holds the signal back by its 250 ms default.
#[test]
fn a_delay_module_holds_back_the_signal() {
	let mut rack = Rack::default();

	let response = command(
		&rack,
		"band.0.mute=1
band.1.mute=1
band.2.mute=1
slot.3.add=Delay",
	);
	assert_eq!(response.body, b"{\"ok\":true}");

	process(&mut rack, [1.0, 1.0]);

	let mut peak = 0.0_f32;
	let mut peak_at = 0usize;

	for index in 0..14_000 {
		let out = process(&mut rack, [0.0, 0.0]);

		if out[0].abs() > peak {
			peak = out[0].abs();
			peak_at = index;
		}
	}

	assert!(peak > 0.2, "the delayed impulse was only {peak}");
	assert!(
		(peak_at as i32 - 11_025).abs() < 100,
		"the impulse came back at {peak_at} instead of 250 ms"
	);
}

/// A module joins the group on its left, and leaves it on the next toggle.
#[test]
fn modules_can_be_grouped_and_ungrouped() {
	let rack = Rack::default();

	assert_eq!(command(&rack, "slot.0.add=Gain").body, b"{\"ok\":true}");
	assert_eq!(command(&rack, "group.0.1=1").body, b"{\"ok\":true}");

	let state = rack.state();
	let group = state.bands[0].slots[0].as_ref().expect("the filter").group;
	assert!(group.is_some());
	assert_eq!(group, state.bands[0].slots[1].as_ref().unwrap().group);

	let json: serde_json::Value = serde_json::from_slice(&get(&rack, "/rack").body).expect("JSON");
	assert_eq!(
		json["modules"][0]["group"].as_u64(),
		Some(group.unwrap() as u64)
	);

	assert_eq!(command(&rack, "group.0.1=1").body, b"{\"ok\":true}");

	let state = rack.state();
	assert!(state.bands[0].slots[0].as_ref().unwrap().group.is_none());
	assert!(state.bands[0].slots[1].as_ref().unwrap().group.is_none());
}

/// The first module in a lane has nothing on its left to group with.
#[test]
fn the_first_module_cannot_start_a_group() {
	let rack = Rack::default();

	assert_eq!(command(&rack, "group.0.0=1").body, b"{\"ok\":false}");
}

/// Moving one member to another lane breaks the run, so both halves ungroup.
#[test]
fn a_broken_group_is_cleared() {
	let rack = Rack::default();

	assert_eq!(command(&rack, "slot.0.add=Gain").body, b"{\"ok\":true}");
	assert_eq!(command(&rack, "group.0.1=1").body, b"{\"ok\":true}");

	assert!(
		rack.state().bands[0].slots[0]
			.as_ref()
			.unwrap()
			.group
			.is_some()
	);

	assert_eq!(command(&rack, "slot.0.1.move=1.1").body, b"{\"ok\":true}");

	let state = rack.state();
	assert!(state.bands[0].slots[0].as_ref().unwrap().group.is_none());
	assert!(state.bands[1].slots[1].as_ref().unwrap().group.is_none());
}

/// Copying a module to another lane keeps the source and clones the settings.
#[test]
fn a_module_can_be_copied_to_another_lane() {
	let rack = Rack::default();

	assert_eq!(command(&rack, "slot.0.add=Gain").body, b"{\"ok\":true}");

	let source_id = rack.state().bands[0].slots[1]
		.as_ref()
		.expect("the added gain")
		.id;

	assert_eq!(
		command(&rack, &format!("{source_id}.gain=0.5")).body,
		b"{\"ok\":true}"
	);

	let source = rack.state().bands[0].slots[1]
		.as_ref()
		.expect("the source")
		.clone();

	assert_eq!(command(&rack, "slot.0.1.copy=1.1").body, b"{\"ok\":true}");

	let state = rack.state();
	let original = state.bands[0].slots[1].as_ref().expect("the original");
	let copy = state.bands[1].slots[1].as_ref().expect("the copy");

	assert_eq!(original.id, source.id);
	assert_eq!(original.params["gain"], 0.5);
	assert_ne!(copy.id, source.id);
	assert_eq!(copy.kind, source.kind);
	assert_eq!(copy.params["gain"], 0.5);
	assert!(copy.group.is_none());
}

/// The Crusher kind brings the disperser/saturator controls into a rack slot.
#[test]
fn a_crusher_module_exposes_its_controls() {
	let mut rack = Rack::default();

	assert_eq!(command(&rack, "slot.3.add=Crusher").body, b"{\"ok\":true}");

	let json: serde_json::Value = serde_json::from_slice(&get(&rack, "/rack").body).expect("JSON");
	let crusher = json["modules"]
		.as_array()
		.expect("modules")
		.iter()
		.find(|module| module["kind"] == "Crusher")
		.expect("the Crusher module");
	let controls = crusher["controls"].as_array().expect("controls");

	assert!(controls.iter().any(|control| control["id"] == "amount"));
	assert!(controls.iter().any(|control| control["id"] == "frequency"));

	for _ in 0..256 {
		let out = process(&mut rack, [0.25, 0.25]);
		assert!(out[0].is_finite() && out[1].is_finite());
	}
}

/// The Chorus kind exposes its delay/rate/depth controls.
#[test]
fn a_chorus_module_exposes_its_controls() {
	let mut rack = Rack::default();

	assert_eq!(command(&rack, "slot.3.add=Chorus").body, b"{\"ok\":true}");

	let json: serde_json::Value = serde_json::from_slice(&get(&rack, "/rack").body).expect("JSON");
	let chorus = json["modules"]
		.as_array()
		.expect("modules")
		.iter()
		.find(|module| module["kind"] == "Chorus")
		.expect("the Chorus module");
	let controls = chorus["controls"].as_array().expect("controls");

	for id in ["center_delay_time", "lfo_frequency", "lfo_amplitude"] {
		assert!(
			controls.iter().any(|control| control["id"] == id),
			"missing {id}"
		);
	}

	for _ in 0..256 {
		let out = process(&mut rack, [0.25, 0.25]);
		assert!(out[0].is_finite() && out[1].is_finite());
	}
}

/// The Phaser kind exposes its sweep, feedback and mix controls.
#[test]
fn a_phaser_module_exposes_its_controls() {
	let mut rack = Rack::default();

	assert_eq!(command(&rack, "slot.3.add=Phaser").body, b"{\"ok\":true}");

	let json: serde_json::Value = serde_json::from_slice(&get(&rack, "/rack").body).expect("JSON");
	let phaser = json["modules"]
		.as_array()
		.expect("modules")
		.iter()
		.find(|module| module["kind"] == "Phaser")
		.expect("the Phaser module");
	let controls = phaser["controls"].as_array().expect("controls");

	for id in ["lfo_freq", "min_freq", "max_freq", "feedback_gain"] {
		assert!(
			controls.iter().any(|control| control["id"] == id),
			"missing {id}"
		);
	}

	for _ in 0..256 {
		let out = process(&mut rack, [0.25, 0.25]);
		assert!(out[0].is_finite() && out[1].is_finite());
	}
}
