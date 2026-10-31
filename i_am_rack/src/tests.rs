//! M1 acceptance tests.

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
	assert_eq!(rack.engine.kinds(), ModuleKind::ALL.to_vec());
}
