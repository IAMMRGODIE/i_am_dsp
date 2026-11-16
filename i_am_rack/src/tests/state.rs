//! The state the host saves, the route the page reads and the catalog.

use super::*;

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

	let response = command(
		&rack,
		"0.cutoff=3000
2.gain=0.5
4.gain=0.5",
	);

	assert_eq!(response.status, 200);
	assert_eq!(response.body, b"{\"ok\":true}");

	let state = rack.state();

	assert_eq!(
		state.bands[0].slots[0].as_ref().unwrap().params["cutoff"],
		3_000.0
	);
	assert_eq!(
		state.bands[1].slots[0].as_ref().unwrap().params["gain"],
		0.5
	);
	assert_eq!(
		state.bands[2].slots[0].as_ref().unwrap().params["gain"],
		0.5
	);
}

/// The page gets the bands and a flat list of the modules they hold.
#[test]
fn the_rack_route_lists_bands_and_modules() {
	let rack = Rack::default();

	let response = get(&rack, "/rack");
	let json: serde_json::Value = serde_json::from_slice(&response.body).expect("JSON");

	let modules = json["modules"].as_array().expect("modules");
	let bands = json["bands"].as_array().expect("bands");

	assert_eq!(bands.len(), 4);
	assert_eq!(bands[0]["name"], "Low");
	assert_eq!(bands[0]["slots"][0], 0);
	assert_eq!(bands[0]["slots"][1], serde_json::Value::Null);
	assert_eq!(bands[3]["name"], "Parallel");
	assert_eq!(bands[3]["slots"][0], serde_json::Value::Null);

	assert_eq!(modules.len(), 3);
	assert_eq!(modules[0]["kind"], "Filter");
	assert_eq!(modules[0]["band_name"], "Low");
	assert_eq!(modules[1]["kind"], "Distortion");
	assert_eq!(modules[2]["kind"], "Gain");
	assert_eq!(modules[0]["hero"], "spectrum");

	let kinds = json["kinds"].as_array().expect("kinds");
	assert_eq!(kinds.len(), 40);

	let names = kinds
		.iter()
		.map(|kind| kind["name"].as_str().expect("a name"))
		.collect::<Vec<_>>();
	let mut sorted = names.clone();
	sorted.sort_by_key(|name| name.to_lowercase());
	assert_eq!(names, sorted, "the add menu is not alphabetical");
	assert!(kinds.iter().any(|kind| kind["name"] == "Compressor"));
	assert!(kinds.iter().any(|kind| kind["name"] == "Reverb"));
	assert!(kinds.iter().any(|kind| kind["name"] == "Pitch Shifter"));
	assert!(
		modules[0]["controls"]
			.as_array()
			.expect("controls")
			.iter()
			.any(|control| control["id"] == "cutoff")
	);
}

/// The shared catalog can add a product wrapper without a second schema
/// definition in the rack.
#[test]
fn the_catalog_adds_reverb_and_pitch_modules() {
	let reverb = module::schema(ModuleKind::Reverb);
	let pitch = module::schema(ModuleKind::Pitch);

	assert!(
		reverb
			.iter()
			.any(|parameter| parameter.identifier == "decay_time")
	);
	assert!(
		pitch
			.iter()
			.any(|parameter| parameter.identifier == "pitch_shift_factor")
	);
	assert_eq!(ModuleKind::Pitch.mode(), Some("algorithm"));
	assert_eq!(ModuleKind::Pitch.header(), &["window_size"]);
	assert!(ModuleKind::Pitch.visible().is_some());
	// The MIDI switches live in the cards' headers, next to the window size.
	assert_eq!(ModuleKind::Resonator.header(), &["midi"]);
	assert_eq!(ModuleKind::Autotune.header(), &["window_size", "midi"]);
	assert_eq!(
		ModuleKind::SpectralRetune.header(),
		&["window_size", "midi"]
	);
	assert_eq!(ModuleKind::Unstablizer.header(), &["midi"]);
	assert_eq!(module::hero(ModuleKind::Reverb), "tail");
	assert_eq!(module::hero(ModuleKind::Chorus), "lfo");
	assert_eq!(module::hero(ModuleKind::Flanger), "lfo");
	assert_eq!(module::hero(ModuleKind::Shifter), "spectra");
	assert_eq!(module::hero(ModuleKind::Disperser), "phase");
	assert_eq!(module::hero(ModuleKind::Delay), "taps");
	assert_eq!(module::hero(ModuleKind::Filter), "spectrum");

	// The shifter's FIR engine takes any odd kernel length, and the rack card
	// sees the same range the product does.
	let shifter = module::schema(ModuleKind::Shifter);
	let order = shifter
		.iter()
		.find(|parameter| parameter.identifier == "order")
		.expect("the shifter order");
	match &order.value {
		i_am_dsp::prelude::Value::Int { range, .. } => assert_eq!(*range, 63..=511),
		other => panic!("unexpected order value {other:?}"),
	}
}

/// The host sees eight macros and 32 fixed automation parameters.
#[test]
fn host_parameters_cover_macros_and_automation() {
	let mut rack = Rack::default();
	let parameters = rack.get_parameters();

	assert_eq!(parameters.len(), 41);
	assert_eq!(parameters[0].identifier, "rack_state");
	assert_eq!(parameters[1].identifier, "macro1");
	assert_eq!(parameters[9].identifier, "automation1");

	let map = rack.param_map();
	assert_eq!(map.len(), 40);
	assert_eq!(map.query_param_id(0), Some("macro1"));

	assert!(rack.set_parameter("macro1", SetValue::Float(0.75)));
	assert!(rack.set_parameter("automation1", SetValue::Float(0.5)));

	let state = rack.state();
	assert!((state.macros[0] - 0.75).abs() < 1e-6);
	assert!((state.automation_values[0] - 0.5).abs() < 1e-6);

	let macro_value = map
		.get("macro1")
		.unwrap()
		.load(std::sync::atomic::Ordering::Relaxed)
		.float()
		.unwrap();
	assert!((macro_value - 0.75).abs() < 1e-6);

	assert_eq!(command(&rack, "macro.0=0.25").body, b"{\"ok\":true}");
	let macro_value = map
		.get("macro1")
		.unwrap()
		.load(std::sync::atomic::Ordering::Relaxed)
		.float()
		.unwrap();
	assert!((macro_value - 0.25).abs() < 1e-6);

	assert_eq!(
		command(&rack, "automation.0.target=m0.cutoff").body,
		b"{\"ok\":true}"
	);

	for _ in 0..2_048 {
		process(&mut rack, [0.0, 0.0]);
	}

	let cutoff = rack.engine.module_parameter(0, "cutoff").unwrap();
	// A linear mapping would put 0.5 near 5 kHz; the logarithmic mapping is much lower.
	assert!(cutoff < 1_000.0, "automated cutoff {cutoff}");
}

/// Host and page edits meet in the same map without reporting the whole table.
#[test]
fn host_and_page_parameter_changes_are_bidirectional() {
	let mut rack = Rack::default();
	let map = rack.param_map();

	assert!(!map.get("macro1").unwrap().is_chanegd());
	assert!(!map.get("macro2").unwrap().is_chanegd());

	assert_eq!(command(&rack, "macro.0=0.5").body, b"{\"ok\":true}");
	assert!(map.get("macro1").unwrap().is_chanegd());
	assert!(!map.get("macro2").unwrap().is_chanegd());

	map.get("macro1")
		.unwrap()
		.store(0.25_f32, std::sync::atomic::Ordering::SeqCst);

	for _ in 0..64 {
		process(&mut rack, [0.0, 0.0]);
	}

	assert!((rack.state().macros[0] - 0.25).abs() < 1e-6);
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
	rack.shared
		.dirty
		.store(true, std::sync::atomic::Ordering::Release);

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
	assert!(module.params.contains_key("attack"));
	assert!(module.params.contains_key("release"));
	assert!(module.params.contains_key("lookahead"));
	assert!(!module.params.contains_key("enveloper.cutoff"));

	let response = command(&rack, "slot.1.1.clear=1");

	assert_eq!(response.body, b"{\"ok\":true}");
	assert!(rack.state().bands[1].slots[1].is_none());
}

/// A band can grow past its initial two slots.
#[test]
fn a_band_can_add_more_than_two_modules() {
	let rack = Rack::default();

	let response = command(&rack, "slot.0.add=Filter");
	assert_eq!(response.body, b"{\"ok\":true}");

	let state = rack.state();
	assert_eq!(state.bands[0].slots.len(), 3);
	assert!(state.bands[0].slots[1].is_some());
	assert!(state.bands[0].slots[2].is_none());

	let response = command(&rack, "slot.0.add=Gain");
	assert_eq!(response.body, b"{\"ok\":true}");

	let state = rack.state();
	assert_eq!(state.bands[0].slots.len(), 4);
	assert!(state.bands[0].slots[2].is_some());
	assert!(state.bands[0].slots[3].is_none());
}

/// A module can be dragged into another slot, swapping with whatever is there.
#[test]
fn a_slot_can_move_and_swap_modules() {
	let rack = Rack::default();

	let response = command(&rack, "slot.0.1.kind=Gain");
	assert_eq!(response.body, b"{\"ok\":true}");

	let state = rack.state();
	let gain = state.bands[0].slots[1].as_ref().expect("the new gain").id;

	let response = command(&rack, "slot.0.0.move=0.1");
	assert_eq!(response.body, b"{\"ok\":true}");

	let state = rack.state();
	assert_eq!(
		state.bands[0].slots[0].as_ref().map(|module| module.id),
		Some(gain)
	);
	assert_eq!(
		state.bands[0].slots[1].as_ref().map(|module| module.id),
		Some(0)
	);
	assert!(state.bands[0].slots[2].is_none());

	let response = command(&rack, "slot.0.1.move=1.1");
	assert_eq!(response.body, b"{\"ok\":true}");

	let state = rack.state();
	assert_eq!(state.bands[0].slots.len(), 2);
	assert_eq!(
		state.bands[0].slots[0].as_ref().map(|module| module.id),
		Some(gain)
	);
	assert!(state.bands[0].slots[1].is_none());
	assert_eq!(
		state.bands[1].slots[1].as_ref().map(|module| module.id),
		Some(0)
	);
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
///
/// The product measures its lookahead in milliseconds and caps it at 50, so the
/// host has to compensate the 2400 samples that lookahead adds at 48 kHz.
#[test]
fn compressor_latency_is_reported_and_compensated() {
	let mut state = RackState::initial();
	let mut compressor = ModuleState::new(6, ModuleKind::Compressor);

	compressor.params.insert("lookahead".to_string(), 50.0);
	state.bands[0].slots[1] = Some(compressor);

	let engine = crate::engine::Engine::new(48_000, &state);

	// The second band starts with a distortion, and its four-times shaper
	// reports the oversampler's latency.
	let distortion = i_am_dsp::tools::oversampling::LATENCY;

	assert_eq!(engine.delay(), 2400);
	assert_eq!(engine.lane_delays(), [2400, distortion, 0]);
	assert_eq!(engine.lane_compensations(), [0, 2400 - distortion, 2400]);

	state.bands[0].slots[1]
		.as_mut()
		.expect("the compressor")
		.bypass = true;

	let engine = crate::engine::Engine::new(48_000, &state);

	assert_eq!(engine.delay(), distortion);
	assert_eq!(engine.lane_delays(), [0, distortion, 0]);
	assert_eq!(engine.lane_compensations(), [distortion, 0, distortion]);
}
