//! M1 acceptance tests.

use i_am_dsp::Processor;
use i_am_dsp::prelude::{
	Parameters, SetValue, Value, from_binary, to_binary,
};
use i_am_plugin::Plugin;
use i_am_plugin::wry::{ProtocolRequest, ProtocolResponse};

use super::*;
use crate::state::{Assignment, AssignmentMode, Target};

fn command(rack: &Rack, body: &str) -> ProtocolResponse {
	let request = ProtocolRequest {
		method: "POST",
		path: "/rack-command",
		query: None,
		body: body.as_bytes(),
	};

	web::route(&request, &rack.shared).expect("the rack command route")
}

fn get(rack: &Rack, path: &str) -> ProtocolResponse {
	let request = ProtocolRequest {
		method: "GET",
		path,
		query: None,
		body: b"",
	};

	web::route(&request, &rack.shared).expect("the rack route")
}

fn process(rack: &mut Rack, frame: [f32; 2]) -> [f32; 2] {
	let mut frame = frame;
	let mut context: Box<dyn ProcessContext> = Box::new(());

	Processor::process(rack, &mut frame, &[], &mut context);

	frame
}

/// The state parameter is the host's only handle on the rack; a save followed
/// by a load has to come back with the same bands and values.
#[test]
fn the_state_round_trips_through_the_host_parameter() {
	let mut rack = Rack::default();

	let parameters = rack.get_parameters();
	let Value::Serialized(bytes) = &parameters[0].value else {
		panic!("the rack must expose its state as a serialized parameter");
	};

	let mut state: RackState = from_binary(bytes.clone()).expect("the initial state");
	state.bands[0].slots[0]
		.as_mut()
		.expect("the low band's filter")
		.params
		.insert("cutoff".to_string(), 2_400.0);

	let encoded = to_binary(&state).expect("the edited state");
	assert!(rack.set_parameter("rack_state", SetValue::Serialized(encoded)));

	let stored = rack.get_parameters();
	let Value::Serialized(bytes) = &stored[0].value else {
		panic!("the state parameter must stay serialized");
	};
	let decoded: RackState = from_binary(bytes.clone()).expect("the saved state");

	let cutoff = decoded.bands[0].slots[0]
		.as_ref()
		.expect("the low band's filter")
		.params["cutoff"];

	assert_eq!(cutoff, 2_400.0);
}

/// A command line edits the state the audio thread will apply.
#[test]
fn a_command_edits_the_state() {
	let rack = Rack::default();

	let response = command(&rack, "0.cutoff=3000
2.gain=0.5
4.gain=0.5");

	assert_eq!(response.status, 200);
	assert_eq!(response.body, b"{\"ok\":true}");

	let state = rack.state();

	assert_eq!(state.bands[0].slots[0].as_ref().unwrap().params["cutoff"], 3_000.0);
	assert_eq!(state.bands[1].slots[0].as_ref().unwrap().params["gain"], 0.5);
	assert_eq!(state.bands[2].slots[0].as_ref().unwrap().params["gain"], 0.5);
}

/// The page gets the bands and a flat list of the modules they hold.
#[test]
fn the_rack_route_lists_bands_and_modules() {
	let rack = Rack::default();

	let response = get(&rack, "/rack");
	let json: serde_json::Value = serde_json::from_slice(&response.body).expect("JSON");

	let modules = json["modules"].as_array().expect("modules");
	let bands = json["bands"].as_array().expect("bands");

	assert_eq!(bands.len(), 3);
	assert_eq!(bands[0]["name"], "Low");
	assert_eq!(bands[0]["slots"][0], 0);
	assert_eq!(bands[0]["slots"][1], serde_json::Value::Null);

	assert_eq!(modules.len(), 3);
	assert_eq!(modules[0]["kind"], "Filter");
	assert_eq!(modules[0]["band_name"], "Low");
	assert_eq!(modules[1]["kind"], "Distortion");
	assert_eq!(modules[2]["kind"], "Gain");
	assert_eq!(modules[0]["hero"], "spectrum");

	let kinds = json["kinds"].as_array().expect("kinds");
	assert_eq!(kinds.len(), 4);
	assert!(kinds.iter().any(|kind| kind["name"] == "Compressor"));
	assert!(
		modules[0]["controls"]
			.as_array()
			.expect("controls")
			.iter()
			.any(|control| control["id"] == "cutoff")
	);
}

/// A whole-plug-in bypass is the input, untouched.
#[test]
fn the_plug_in_bypass_passes_the_signal_through() {
	let mut rack = Rack::default();

	rack.set_bypassed(true);

	let input = [0.25, -0.5];

	assert_eq!(process(&mut rack, input), input);
}

/// A lowpass at 20 Hz takes a 50 Hz tone out of the low band.
#[test]
fn the_low_band_filter_changes_a_signal_above_its_cutoff() {
	let mut rack = Rack::default();

	command(&rack, "0.cutoff=20");

	let mut input_energy = 0.0;
	let mut output_energy = 0.0;

	for i in 0..8_192 {
		let time = i as f32 / 44_100.0;
		let sample = (2.0 * std::f32::consts::PI * 50.0 * time).sin() * 0.5;
		let output = process(&mut rack, [sample, sample]);

		if i >= 4_096 {
			input_energy += sample * sample;
			output_energy += output[0] * output[0];
		}
	}

	assert!(
		output_energy < input_energy * 0.1,
		"the lowpass left {output_energy} of {input_energy}"
	);
}

/// Muting the low band takes its energy out of the sum.
#[test]
fn a_muted_band_leaves_the_sum() {
	let mut rack = Rack::default();

	{
		let mut state = rack.shared.lock();
		state.bands[0].mute = true;
	}
	rack.shared.dirty.store(true, std::sync::atomic::Ordering::Release);

	let mut input_energy = 0.0;
	let mut output_energy = 0.0;

	for i in 0..8_192 {
		let time = i as f32 / 44_100.0;
		let sample = (2.0 * std::f32::consts::PI * 50.0 * time).sin() * 0.5;
		let output = process(&mut rack, [sample, sample]);

		if i >= 4_096 {
			input_energy += sample * sample;
			output_energy += output[0] * output[0];
		}
	}

	assert!(
		output_energy < input_energy * 0.01,
		"the muted band left {output_energy} of {input_energy}"
	);
}

/// Activating at the host rate rebuilds the effects and keeps the state.
#[test]
fn activation_rebuilds_for_the_new_rate() {
	let mut rack = Rack::default();

	rack.on_activate(48_000);

	assert_eq!(rack.sample_rate(), 48_000);
	assert_eq!(
		rack.state().bands[0].slots[0].as_ref().unwrap().params["cutoff"],
		1_000.0
	);
	assert_eq!(
		rack.engine.kinds(),
		vec![ModuleKind::Filter, ModuleKind::Distortion, ModuleKind::Gain]
	);
}

/// Band gain/pan/mute/solo and the two crossover points are all page commands.
#[test]
fn band_and_crossover_commands_edit_the_state() {
	let rack = Rack::default();

	let response = command(
		&rack,
		"band.0.gain=0.5
band.0.pan=-0.25
band.1.mute=1
band.2.solo=1
split.low=240
split.high=3600",
	);

	assert_eq!(response.body, b"{\"ok\":true}");

	let state = rack.state();

	assert_eq!(state.bands[0].gain, 0.5);
	assert_eq!(state.bands[0].pan, -0.25);
	assert!(state.bands[1].mute);
	assert!(state.bands[2].solo);
	assert_eq!(state.split_low, 240.0);
	assert_eq!(state.split_high, 3_600.0);
}

/// An empty slot can take a module, and the compressor schema is the curated one.
#[test]
fn a_slot_can_hold_and_clear_a_compressor() {
	let rack = Rack::default();

	let response = command(&rack, "slot.1.1.kind=Compressor");

	assert_eq!(response.body, b"{\"ok\":true}");

	let state = rack.state();
	let module = state.bands[1].slots[1]
		.as_ref()
		.expect("the new compressor");

	assert_eq!(module.kind, ModuleKind::Compressor);
	assert_eq!(module.id, 5);
	assert!(module.params.contains_key("threshold"));
	assert!(module.params.contains_key("ratio"));
	assert!(module.params.contains_key("smoother.attack_time"));
	assert!(module.params.contains_key("smoother.release_time"));
	assert!(module.params.contains_key("lookahead"));
	assert!(!module.params.contains_key("enveloper.cutoff"));

	let response = command(&rack, "slot.1.1.clear=1");

	assert_eq!(response.body, b"{\"ok\":true}");
	assert!(rack.state().bands[1].slots[1].is_none());
}

/// A module can be dragged into another slot, swapping with whatever is there.
#[test]
fn a_slot_can_move_and_swap_modules() {
	let rack = Rack::default();

	let response = command(&rack, "slot.0.0.move=0.1");

	assert_eq!(response.body, b"{\"ok\":true}");

	let state = rack.state();

	assert!(state.bands[0].slots[0].is_none());
	assert_eq!(state.bands[0].slots[1].as_ref().map(|module| module.id), Some(0));

	let response = command(&rack, "slot.1.0.move=0.1");

	assert_eq!(response.body, b"{\"ok\":true}");

	let state = rack.state();

	assert_eq!(state.bands[0].slots[1].as_ref().map(|module| module.id), Some(2));
	assert_eq!(state.bands[1].slots[0].as_ref().map(|module| module.id), Some(0));
}

/// Replacing a module keeps the id the page addresses.
#[test]
fn changing_a_slot_type_keeps_its_id() {
	let rack = Rack::default();

	command(&rack, "slot.0.0.kind=Compressor");

	let first = rack.state();
	let first = first.bands[0].slots[0].as_ref().expect("the compressor");

	assert_eq!(first.id, 0);
	assert_eq!(first.kind, ModuleKind::Compressor);

	command(&rack, "slot.0.0.kind=Gain");

	let second = rack.state();
	let second = second.bands[0].slots[0].as_ref().expect("the gain");

	assert_eq!(second.id, 0);
	assert_eq!(second.kind, ModuleKind::Gain);
	assert!(second.params.contains_key("gain"));
}

/// A lookahead compressor delays its band; the other bands are padded to match.
#[test]
fn compressor_latency_is_reported_and_compensated() {
	let mut state = RackState::initial();
	let mut compressor = ModuleState::new(6, ModuleKind::Compressor);

	compressor.params.insert("lookahead".to_string(), 64.0);
	state.bands[0].slots[1] = Some(compressor);

	let engine = crate::engine::Engine::new(48_000, &state);

	assert_eq!(engine.delay(), 64);
	assert_eq!(engine.lane_delays(), [64, 0, 0]);
	assert_eq!(engine.lane_compensations(), [0, 64, 64]);

	state.bands[0].slots[1]
		.as_mut()
		.expect("the compressor")
		.bypass = true;

	let engine = crate::engine::Engine::new(48_000, &state);

	assert_eq!(engine.delay(), 0);
	assert_eq!(engine.lane_delays(), [0, 0, 0]);
	assert_eq!(engine.lane_compensations(), [0, 0, 0]);
}

fn process_engine(engine: &mut crate::engine::Engine, frame: [f32; 2]) -> [f32; 2] {
	let mut frame = frame;
	let mut context: Box<dyn ProcessContext> = Box::new(());

	engine.process(&mut frame, &[], &mut context);
	frame
}

/// The page can set the three sources and the eight assignment slots.
#[test]
fn source_and_assignment_commands_edit_the_state() {
	let rack = Rack::default();

	let response = command(
		&rack,
		"source.lfo.rate=2
source.lfo.shape=2
source.env.attack=5
source.random.smooth=120
source.table.rate=0.5
source.table.shape=1
source.table.morph=0.4
assign.0.source=lfo
assign.0.target=m0.cutoff
assign.0.depth=0.5
assign.0.mode=start",
	);

	assert_eq!(response.body, b"{\"ok\":true}");

	let state = rack.state();

	assert!((state.sources.lfo_rate - 2.0).abs() < 1e-6);
	assert_eq!(state.sources.lfo_shape, 2);
	assert!((state.sources.env_attack - 5.0).abs() < 1e-6);
	assert!((state.sources.random_smooth - 120.0).abs() < 1e-6);
	assert!((state.sources.table_rate - 0.5).abs() < 1e-6);
	assert_eq!(state.sources.table_shape, 1);
	assert!((state.sources.table_morph - 0.4).abs() < 1e-6);

	let assignment = state.assignments[0]
		.as_ref()
		.expect("the first assignment");

	assert_eq!(assignment.source, 0);
	assert_eq!(assignment.depth, 0.5);
	assert_eq!(assignment.mode, AssignmentMode::FromBase);
	assert_eq!(
		assignment.target,
		Some(Target::Module {
			module: 0,
			parameter: "cutoff".to_string(),
		})
	);
}

/// Macros are values, and assignments can use them as sources.
#[test]
fn macro_commands_and_assignments_edit_the_state() {
	let rack = Rack::default();

	let response = command(
		&rack,
		"macro.0=0.75
assign.2.source=macro1
assign.2.target=band.0.gain
assign.2.depth=0.4",
	);

	assert_eq!(response.body, b"{\"ok\":true}");

	let state = rack.state();

	assert!((state.macros[0] - 0.75).abs() < 1e-6);

	let assignment = state.assignments[2]
		.as_ref()
		.expect("the third assignment");

	assert_eq!(assignment.source, 4);
	assert_eq!(assignment.target, Some(Target::BandGain(0)));
	assert!((assignment.depth - 0.4).abs() < 1e-6);
}

/// The page can replace the EditableLfo points, and choosing a preset resets them.
#[test]
fn lfo_points_and_presets_edit_the_state() {
	let rack = Rack::default();

	let response = command(
		&rack,
		"source.lfo.points=0:0.5:-2.5;0.25:0.1:3.5;0.5:0.9:0;0.75:0.2:0;1:0.5:0",
	);

	assert_eq!(response.body, b"{\"ok\":true}");

	let state = rack.state();

	assert_eq!(state.sources.lfo_shape, 4);
	assert_eq!(state.sources.lfo_points.len(), 5);
	assert!((state.sources.lfo_points[1].1 - 0.1).abs() < 1e-6);
	assert!((state.sources.lfo_points[0].2 + 2.5).abs() < 1e-6);

	command(&rack, "source.lfo.shape=1");

	let state = rack.state();

	assert_eq!(state.sources.lfo_shape, 1);
	assert_eq!(state.sources.lfo_points.len(), 3);
	assert_eq!(state.sources.lfo_points[1], (0.5, 1.0, 0.0));
}

/// The page gets the list of parameters an assignment can target.
#[test]
fn the_rack_route_lists_modulation_targets() {
	let rack = Rack::default();
	let response = get(&rack, "/rack");
	let json: serde_json::Value = serde_json::from_slice(&response.body).expect("JSON");

	let targets = json["targets"].as_array().expect("targets");

	assert!(targets.iter().any(|target| target["id"] == "m0.cutoff"));
	assert!(targets.iter().any(|target| target["id"] == "band.0.gain"));
	assert_eq!(json["assignments"].as_array().expect("assignments").len(), 8);
	assert_eq!(json["sources"]["lfo"]["shape"], 0);
	assert_eq!(json["sources"]["table"]["shape"], 0);
	assert_eq!(json["macros"].as_array().expect("macros").len(), 8);
	assert!(json["sources"]["table"]["morph"].is_number());
}

/// The page can read the live source values and phases.
#[test]
fn the_modulation_route_reports_live_values() {
	let rack = Rack::default();
	let response = get(&rack, "/modulation");
	let json: serde_json::Value = serde_json::from_slice(&response.body).expect("JSON");

	assert_eq!(json["values"].as_array().expect("values").len(), 4);
	assert_eq!(json["phases"].as_array().expect("phases").len(), 2);
}

/// An LFO assignment moves the filter cutoff at the control rate.
#[test]
fn an_lfo_modulates_a_filter_cutoff() {
	let mut state = RackState::initial();

	state.sources.lfo_rate = 8.0;
	state.assignments[0] = Some(Assignment {
		source: 0,
		target: Some(Target::Module {
			module: 0,
			parameter: "cutoff".to_string(),
		}),
		depth: 0.8,
		..Assignment::default()
	});

	let mut engine = crate::engine::Engine::new(48_000, &state);
	let base = engine
		.module_parameter(0, "cutoff")
		.expect("the filter cutoff");
	let mut lowest = base;
	let mut highest = base;

	for _ in 0..4_096 {
		process_engine(&mut engine, [0.0, 0.0]);

		let cutoff = engine
			.module_parameter(0, "cutoff")
			.expect("the filter cutoff");

		lowest = lowest.min(cutoff);
		highest = highest.max(cutoff);
	}

	assert!(
		highest - lowest > 100.0,
		"the cutoff stayed between {lowest} and {highest}"
	);
}

/// A table assignment moves the filter cutoff too.
#[test]
fn a_table_modulates_a_filter_cutoff() {
	let mut state = RackState::initial();

	state.sources.table_rate = 10.0;
	state.sources.table_shape = 1;
	state.sources.table_morph = 0.75;
	state.assignments[0] = Some(Assignment {
		source: 3,
		target: Some(Target::Module {
			module: 0,
			parameter: "cutoff".to_string(),
		}),
		depth: 0.8,
		..Assignment::default()
	});

	let mut engine = crate::engine::Engine::new(48_000, &state);
	let base = engine
		.module_parameter(0, "cutoff")
		.expect("the filter cutoff");
	let mut lowest = base;
	let mut highest = base;

	for _ in 0..4_096 {
		process_engine(&mut engine, [0.0, 0.0]);

		let cutoff = engine
			.module_parameter(0, "cutoff")
			.expect("the filter cutoff");

		lowest = lowest.min(cutoff);
		highest = highest.max(cutoff);
	}

	assert!(
		highest - lowest > 100.0,
		"the table cutoff stayed between {lowest} and {highest}"
	);
}

/// A macro value can drive a module parameter just like a source.
#[test]
fn a_macro_modulates_a_filter_cutoff() {
	let mut state = RackState::initial();

	state.macros[0] = 1.0;
	state.assignments[0] = Some(Assignment {
		source: 4,
		target: Some(Target::Module {
			module: 0,
			parameter: "cutoff".to_string(),
		}),
		depth: 0.5,
		..Assignment::default()
	});

	let mut engine = crate::engine::Engine::new(48_000, &state);
	let base = engine
		.module_parameter(0, "cutoff")
		.expect("the filter cutoff");

	for _ in 0..128 {
		process_engine(&mut engine, [0.0, 0.0]);
	}

	let cutoff = engine
		.module_parameter(0, "cutoff")
		.expect("the filter cutoff");

	assert!(
		cutoff > base + 100.0,
		"the macro left the cutoff at {cutoff} (base {base})"
	);
}

/// An envelope assignment moves the low band's gain.
#[test]
fn an_envelope_modulates_a_band_gain() {
	let mut state = RackState::initial();

	state.sources.env_attack = 1.0;
	state.sources.env_release = 50.0;
	state.assignments[0] = Some(Assignment {
		source: 1,
		target: Some(Target::BandGain(0)),
		depth: 1.0,
		..Assignment::default()
	});

	let mut engine = crate::engine::Engine::new(48_000, &state);
	let mut lowest: f64 = 1.0;
	let mut highest: f64 = 1.0;

	for index in 0..4_096 {
		let phase = 2.0 * std::f32::consts::PI * 220.0 * index as f32 / 48_000.0;
		let sample = phase.sin() * 0.5;

		process_engine(&mut engine, [sample, sample]);

		let gain = engine.band_gain(0).expect("the low band");
		lowest = lowest.min(gain);
		highest = highest.max(gain);
	}

	assert!(
		highest - lowest > 0.15,
		"the low band gain stayed between {lowest} and {highest}"
	);
}
