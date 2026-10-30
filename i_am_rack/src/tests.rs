//! M0 acceptance tests.

use i_am_dsp::Processor;
use i_am_dsp::prelude::{
	Parameters, SetValue, Value, from_binary, to_binary,
};
use i_am_plugin::Plugin;
use i_am_plugin::wry::{ProtocolRequest, ProtocolResponse};

use super::*;

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
/// by a load has to come back with the same modules and values.
#[test]
fn the_state_round_trips_through_the_host_parameter() {
	let mut rack = Rack::default();

	let parameters = rack.get_parameters();
	let Value::Serialized(bytes) = &parameters[0].value else {
		panic!("the rack must expose its state as a serialized parameter");
	};

	let mut state: RackState = from_binary(bytes.clone()).expect("the initial state");
	state.modules[0]
		.params
		.insert("cutoff".to_string(), 2_400.0);

	let encoded = to_binary(&state).expect("the edited state");
	assert!(rack.set_parameter("rack_state", SetValue::Serialized(encoded)));

	let stored = rack.get_parameters();
	let Value::Serialized(bytes) = &stored[0].value else {
		panic!("the state parameter must stay serialized");
	};
	let decoded: RackState = from_binary(bytes.clone()).expect("the saved state");

	assert_eq!(decoded.modules[0].params["cutoff"], 2_400.0);
}

/// A command line edits the state the audio thread will apply.
#[test]
fn a_command_edits_the_state() {
	let rack = Rack::default();

	let response = command(&rack, "0.cutoff=3000\n2.gain=0.5");

	assert_eq!(response.status, 200);
	assert_eq!(response.body, b"{\"ok\":true}");

	let state = rack.state();

	assert_eq!(state.modules[0].params["cutoff"], 3_000.0);
	assert_eq!(state.modules[2].params["gain"], 0.5);
}

/// The page gets a card per module, with the controls the effect declares.
#[test]
fn the_rack_route_lists_modules_and_controls() {
	let rack = Rack::default();

	let response = get(&rack, "/rack");
	let json: serde_json::Value = serde_json::from_slice(&response.body).expect("JSON");

	let modules = json["modules"].as_array().expect("modules");

	assert_eq!(modules.len(), 3);
	assert_eq!(modules[0]["kind"], "Filter");
	assert_eq!(modules[1]["kind"], "Drive");
	assert_eq!(modules[2]["kind"], "Gain");
	assert!(
		modules[0]["controls"]
			.as_array()
			.expect("controls")
			.iter()
			.any(|control| control["id"] == "cutoff")
	);
}

/// Bypassing every module is the input, untouched.
#[test]
fn a_bypassed_chain_passes_the_signal_through() {
	let mut rack = Rack::default();

	command(&rack, "0.__bypass=1\n1.__bypass=1\n2.__bypass=1");

	let input = [0.25, -0.5];

	assert_eq!(process(&mut rack, input), input);
}

/// A lowpass at 200 Hz takes most of a 5 kHz tone out.
#[test]
fn the_filter_changes_a_signal_above_its_cutoff() {
	let mut rack = Rack::default();

	command(&rack, "0.cutoff=200");

	let mut input_energy = 0.0;
	let mut output_energy = 0.0;

	for i in 0..4_800 {
		let time = i as f32 / 44_100.0;
		let sample = (2.0 * std::f32::consts::PI * 5_000.0 * time).sin() * 0.5;
		let output = process(&mut rack, [sample, sample]);

		input_energy += sample * sample;
		output_energy += output[0] * output[0];
	}

	assert!(
		output_energy < input_energy * 0.05,
		"the lowpass left {output_energy} of {input_energy}"
	);
}

/// Activating at the host rate rebuilds the effects and keeps the state.
#[test]
fn activation_rebuilds_for_the_new_rate() {
	let mut rack = Rack::default();

	rack.on_activate(48_000);

	assert_eq!(rack.sample_rate(), 48_000);
	assert_eq!(rack.state().modules[0].params["cutoff"], 1_000.0);
	assert_eq!(rack.engine.kinds(), ModuleKind::ALL.to_vec());
}
