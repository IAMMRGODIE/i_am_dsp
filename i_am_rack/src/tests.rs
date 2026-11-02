//! M1 acceptance tests.

use i_am_dsp::Processor;
use i_am_web_gui::Channels;
use i_am_dsp::prelude::{
	NoteEvent, Parameters, ProcessContext, ProcessInfos, SetValue, Value, from_binary,
	to_binary,
};
use i_am_plugin::Plugin;
use i_am_plugin::wry::{ProtocolRequest, ProtocolResponse};

use super::*;
use crate::state::{Assignment, AssignmentMode, MACRO_BASE, SourceKind, Target};

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

/// The live value of one modulation source, as the page would read it.
fn modulation_value(rack: &Rack, id: u32) -> f64 {
	let response = get(rack, "/modulation");
	let json: serde_json::Value = serde_json::from_slice(&response.body).expect("JSON");

	json["sources"]
		.as_array()
		.expect("sources")
		.iter()
		.find(|source| source["id"] == id)
		.expect("the source")["value"]
		.as_f64()
		.expect("a value")
}

fn process(rack: &mut Rack, frame: [f32; 2]) -> [f32; 2] {
	process_with(rack, frame, Vec::new())
}

struct EventContext {
	infos: ProcessInfos,
	events: Vec<NoteEvent>,
}

impl ProcessContext for EventContext {
	fn infos(&self) -> &ProcessInfos {
		&self.infos
	}

	fn next_event(&mut self) -> Option<NoteEvent> {
		if self.events.is_empty() {
			None
		} else {
			Some(self.events.remove(0))
		}
	}

	fn send_event(&mut self, event: NoteEvent) {
		self.events.push(event);
	}

	fn events(&self) -> &[NoteEvent] {
		&self.events
	}

	fn clear_events(&mut self) {
		self.events.clear();
	}
}

fn process_with(rack: &mut Rack, frame: [f32; 2], events: Vec<NoteEvent>) -> [f32; 2] {
	let mut frame = frame;
	let mut context: Box<dyn ProcessContext> = Box::new(EventContext {
		infos: ProcessInfos::default(),
		events,
	});

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
	assert_eq!(kinds.len(), 36);

	let names = kinds
		.iter()
		.map(|kind| kind["name"].as_str().expect("a name"))
		.collect::<Vec<_>>();
	let mut sorted = names.clone();
	sorted.sort_by_key(|name| name.to_lowercase());
	assert_eq!(names, sorted, "the add menu is not alphabetical");
	assert!(kinds.iter().any(|kind| kind["name"] == "Compressor"));
	assert!(kinds.iter().any(|kind| kind["name"] == "Reverb"));
	assert!(kinds.iter().any(|kind| kind["name"] == "Pitch"));
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

	assert!(reverb.iter().any(|parameter| parameter.identifier == "decay_time"));
	assert!(pitch.iter().any(|parameter| parameter.identifier == "pitch_shift_factor"));
	assert_eq!(ModuleKind::Pitch.mode(), Some("algorithm"));
	assert_eq!(ModuleKind::Pitch.header(), &["window_size"]);
	assert!(ModuleKind::Pitch.visible().is_some());
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

	assert_eq!(command(&rack, "automation.0.target=m0.cutoff").body, b"{\"ok\":true}");

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
	let gain = state.bands[0].slots[1]
		.as_ref()
		.expect("the new gain")
		.id;

	let response = command(&rack, "slot.0.0.move=0.1");
	assert_eq!(response.body, b"{\"ok\":true}");

	let state = rack.state();
	assert_eq!(state.bands[0].slots[0].as_ref().map(|module| module.id), Some(gain));
	assert_eq!(state.bands[0].slots[1].as_ref().map(|module| module.id), Some(0));
	assert!(state.bands[0].slots[2].is_none());

	let response = command(&rack, "slot.0.1.move=1.1");
	assert_eq!(response.body, b"{\"ok\":true}");

	let state = rack.state();
	assert_eq!(state.bands[0].slots.len(), 2);
	assert_eq!(state.bands[0].slots[0].as_ref().map(|module| module.id), Some(gain));
	assert!(state.bands[0].slots[1].is_none());
	assert_eq!(state.bands[1].slots[1].as_ref().map(|module| module.id), Some(0));
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

	assert_eq!(engine.delay(), 2400);
	assert_eq!(engine.lane_delays(), [2400, 0, 0]);
	assert_eq!(engine.lane_compensations(), [0, 2400, 2400]);

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

	let lfo = state
		.source_slots
		.iter()
		.find(|slot| slot.kind == SourceKind::Lfo)
		.expect("the LFO slot");
	assert!((lfo.settings.lfo_rate - 2.0).abs() < 1e-6);
	assert_eq!(lfo.settings.lfo_shape, 2);
	assert!((state.source_slots[1].settings.env_attack - 5.0).abs() < 1e-6);
	assert!((state.source_slots[2].settings.random_smooth - 120.0).abs() < 1e-6);
	assert!((state.source_slots[3].settings.table_rate - 0.5).abs() < 1e-6);
	assert_eq!(state.source_slots[3].settings.table_shape, 1);
	assert!((state.source_slots[3].settings.table_morph - 0.4).abs() < 1e-6);

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

	assert_eq!(assignment.source, MACRO_BASE);
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

	let lfo = state
		.source_slots
		.iter()
		.find(|slot| slot.kind == SourceKind::Lfo)
		.expect("the LFO slot");
	assert_eq!(lfo.settings.lfo_shape, 4);
	assert_eq!(lfo.settings.lfo_points.len(), 5);
	assert!((lfo.settings.lfo_points[1].1 - 0.1).abs() < 1e-6);
	assert!((lfo.settings.lfo_points[0].2 + 2.5).abs() < 1e-6);

	command(&rack, "source.lfo.shape=1");

	let state = rack.state();
	let lfo = state
		.source_slots
		.iter()
		.find(|slot| slot.kind == SourceKind::Lfo)
		.expect("the LFO slot");

	assert_eq!(lfo.settings.lfo_shape, 1);
	assert_eq!(lfo.settings.lfo_points.len(), 3);
	assert_eq!(lfo.settings.lfo_points[1], (0.5, 1.0, 0.0));
}

/// A version-5 flat state becomes four source slots without losing its values.
#[test]
fn legacy_sources_migrate_to_slots() {
	let mut state = RackState::initial();
	state.source_slots.clear();
	state.version = 5;
	state.sources.lfo_rate = 3.0;
	state.assignments[0] = Some(Assignment {
		source: 4,
		target: Some(Target::BandGain(0)),
		depth: 0.5,
		..Assignment::default()
	});

	state.migrate();

	assert_eq!(state.version, crate::state::STATE_VERSION);
	assert_eq!(state.source_slots.len(), 4);
	assert_eq!(state.source_slots[0].kind, SourceKind::Lfo);
	assert!((state.source_slots[0].settings.lfo_rate - 3.0).abs() < 1e-6);
	assert_eq!(state.assignments[0].as_ref().unwrap().source, MACRO_BASE);
}

/// Sources can be added with a free id and removed with their assignments.
#[test]
fn modulation_sources_can_be_added_and_removed() {
	let rack = Rack::default();

	let response = command(&rack, "source.add=lfo");
	assert_eq!(response.body, b"{\"ok\":true}");

	let state = rack.state();
	assert_eq!(state.source_slots.len(), 5);
	assert_eq!(state.source_slots[4].id, 4);
	assert_eq!(state.source_slots[4].kind, SourceKind::Lfo);

	let response = command(
		&rack,
		"source.4.rate=2
assign.0.source=4
assign.0.target=band.0.gain
assign.0.depth=0.4",
	);

	assert_eq!(response.body, b"{\"ok\":true}");

	let state = rack.state();
	assert!((state.source_slots[4].settings.lfo_rate - 2.0).abs() < 1e-6);
	assert_eq!(state.assignments[0].as_ref().unwrap().source, 4);

	let response = command(&rack, "source.4.move=0");
	assert_eq!(response.body, b"{\"ok\":true}");

	let state = rack.state();
	assert_eq!(state.source_slots[0].id, 4);
	assert_eq!(state.source_slots[1].id, 0);
	assert_eq!(state.assignments[0].as_ref().unwrap().source, 4);

	let response = command(&rack, "source.4.remove=1");
	assert_eq!(response.body, b"{\"ok\":true}");

	let state = rack.state();
	assert_eq!(state.source_slots.len(), 4);
	assert!(state.assignments[0].is_none());
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

	let sources = json["sources"].as_array().expect("sources");

	assert_eq!(sources.len(), 4);
	assert_eq!(sources[0]["kind"], "lfo");
	assert_eq!(sources[0]["settings"]["shape"], 0);
	assert_eq!(sources[3]["kind"], "table");
	assert!(sources[3]["settings"]["morph"].is_number());
	assert_eq!(json["macros"].as_array().expect("macros").len(), 8);
	assert_eq!(json["automation"].as_array().expect("automation").len(), 32);
}

/// A Pitch source follows a 440 Hz tone as a normalized MIDI note.
#[test]
fn a_pitch_source_tracks_a_tone() {
	let mut rack = Rack::default();

	let response = command(&rack, "source.add=pitch");
	assert_eq!(response.body, b"{\"ok\":true}");

	for index in 0..8_000 {
		let sample = (index as f32 * 440.0 * std::f32::consts::TAU / 48_000.0).sin() * 0.5;
		process(&mut rack, [sample, sample]);
	}

	let response = get(&rack, "/modulation");
	let json: serde_json::Value = serde_json::from_slice(&response.body).expect("JSON");
	let pitch = json["sources"]
		.as_array()
		.expect("sources")
		.iter()
		.find(|source| source["id"] == 4)
		.expect("the pitch source");
	let value = pitch["value"].as_f64().expect("a value");

	// 440 Hz is MIDI 69; normalized over 127 notes.
	assert!((value - 69.0 / 127.0).abs() < 0.08, "pitch value {value}");
}

/// Matrix rows can be added without a fixed ceiling, including from a drop link.
#[test]
fn matrix_rows_can_be_added_dynamically() {
	let rack = Rack::default();

	assert_eq!(command(&rack, "assign.add=0|m0.cutoff").body, b"{\"ok\":true}");
	assert_eq!(command(&rack, "assign.add=1").body, b"{\"ok\":true}");

	let state = rack.state();
	assert_eq!(state.assignments.len(), 10);
	assert_eq!(state.assignments[8].as_ref().unwrap().source, 0);
	assert_eq!(state.assignments[9].as_ref().unwrap().mode, AssignmentMode::FromBase);
	assert_eq!(
		state.assignments[8].as_ref().unwrap().target,
		Some(Target::Module {
			module: 0,
			parameter: "cutoff".to_string(),
		})
	);
}

/// A MIDI source can follow the note number or its velocity.
#[test]
fn a_midi_source_follows_note_and_velocity() {
	let mut rack = Rack::default();

	assert_eq!(command(&rack, "source.add=midi").body, b"{\"ok\":true}");

	let note = NoteEvent::NoteOn {
		channel: 0,
		note: 60,
		velocity: 0.75,
	};

	for _ in 0..128 {
		process_with(&mut rack, [0.0, 0.0], vec![note.clone()]);
	}

	let response = get(&rack, "/modulation");
	let json: serde_json::Value = serde_json::from_slice(&response.body).expect("JSON");
	let midi = json["sources"]
		.as_array()
		.expect("sources")
		.iter()
		.find(|source| source["id"] == 4)
		.expect("the MIDI source");

	assert!((midi["value"].as_f64().unwrap() - 60.0 / 127.0).abs() < 1e-4);

	assert_eq!(command(&rack, "source.4.mode=1").body, b"{\"ok\":true}");

	for _ in 0..128 {
		process_with(&mut rack, [0.0, 0.0], vec![note.clone()]);
	}

	let response = get(&rack, "/modulation");
	let json: serde_json::Value = serde_json::from_slice(&response.body).expect("JSON");
	let midi = json["sources"]
		.as_array()
		.expect("sources")
		.iter()
		.find(|source| source["id"] == 4)
		.expect("the MIDI source");

	assert!((midi["value"].as_f64().unwrap() - 0.75).abs() < 1e-4);
}

/// MIDI note and velocity curves shape the source output.
#[test]
fn midi_curves_shape_the_output() {
	let mut rack = Rack::default();
	assert_eq!(command(&rack, "source.add=midi").body, b"{\"ok\":true}");
	assert_eq!(
		command(&rack, "source.4.note_curve=0:0;0.5:1;1:0").body,
		b"{\"ok\":true}"
	);

	let note = NoteEvent::NoteOn {
		channel: 0,
		note: 64,
		velocity: 0.25,
	};

	for _ in 0..128 {
		process_with(&mut rack, [0.0, 0.0], vec![note.clone()]);
	}

	let response = get(&rack, "/modulation");
	let json: serde_json::Value = serde_json::from_slice(&response.body).expect("JSON");
	let value = json["sources"]
		.as_array()
		.expect("sources")
		.iter()
		.find(|source| source["id"] == 4)
		.expect("the MIDI source")["value"]
		.as_f64()
		.unwrap();

	assert!(value > 0.9, "curve output {value}");

	assert_eq!(command(&rack, "source.4.mode=1").body, b"{\"ok\":true}");
	assert_eq!(
		command(&rack, "source.4.velocity_curve=0:1;1:0").body,
		b"{\"ok\":true}"
	);

	for _ in 0..128 {
		process_with(&mut rack, [0.0, 0.0], vec![note.clone()]);
	}

	let response = get(&rack, "/modulation");
	let json: serde_json::Value = serde_json::from_slice(&response.body).expect("JSON");
	let value = json["sources"]
		.as_array()
		.expect("sources")
		.iter()
		.find(|source| source["id"] == 4)
		.expect("the MIDI source")["value"]
		.as_f64()
		.unwrap();

	assert!((value - 0.75).abs() < 0.05, "velocity curve output {value}");
}

/// Moving a source forward lands it after the target, not one slot short.
#[test]
fn source_reorder_moves_forward() {
	let rack = Rack::default();

	assert_eq!(command(&rack, "source.0.move=2").body, b"{\"ok\":true}");

	let ids = rack
		.state()
		.source_slots
		.iter()
		.map(|slot| slot.id)
		.collect::<Vec<_>>();

	assert_eq!(ids, vec![1, 2, 0, 3]);
}

/// The source list has no fixed eight-source limit.
#[test]
fn source_slots_have_no_fixed_limit() {
	let rack = Rack::default();

	for _ in 0..10 {
		assert_eq!(command(&rack, "source.add=lfo").body, b"{\"ok\":true}");
	}

	assert_eq!(rack.state().source_slots.len(), 14);
}

/// The page can read the live source values and phases.
#[test]
fn the_modulation_route_reports_live_values() {
	let mut rack = Rack::default();

	for _ in 0..128 {
		process(&mut rack, [0.0, 0.0]);
	}

	let response = get(&rack, "/modulation");
	let json: serde_json::Value = serde_json::from_slice(&response.body).expect("JSON");
	let sources = json["sources"].as_array().expect("sources");

	assert_eq!(sources.len(), 4);
	assert_eq!(sources[0]["id"], 0);
	assert!(sources[0]["value"].is_number());
	assert!(sources[0]["phase"].is_number());
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
		source: MACRO_BASE,
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

	let json: serde_json::Value =
		serde_json::from_slice(&get(&rack, "/rack").body).expect("JSON");
	assert_eq!(json["ab"]["active"], "B");

	command(&rack, "0.cutoff=800");
	command(&rack, "ab.toggle=1");

	let json: serde_json::Value =
		serde_json::from_slice(&get(&rack, "/rack").body).expect("JSON");
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

	let json: serde_json::Value =
		serde_json::from_slice(&get(&rack, "/rack").body).expect("JSON");
	assert_eq!(json["history"]["undo"], false);
	assert_eq!(json["history"]["redo"], false);
	assert_eq!(json["ab"]["active"], "A");

	command(&rack, "0.cutoff=3000");

	let json: serde_json::Value =
		serde_json::from_slice(&get(&rack, "/rack").body).expect("JSON");
	assert_eq!(json["history"]["undo"], true);
	assert_eq!(json["history"]["redo"], false);

	command(&rack, "history.undo=1");

	let json: serde_json::Value =
		serde_json::from_slice(&get(&rack, "/rack").body).expect("JSON");
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

	let json: serde_json::Value =
		serde_json::from_slice(&get(&rack, "/rack").body).expect("JSON");
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
		state
			.ab
			.other
			.as_ref()
			.expect("the A side")
			.bands[0]
			.slots[0]
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
		reopened
			.ab
			.other
			.as_ref()
			.expect("the A side")
			.bands[0]
			.slots[0]
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

	assert_eq!(command(&rack, "source.4.transpose=12").body, b"{\"ok\":true}");

	for _ in 0..128 {
		process_with(&mut rack, [0.0, 0.0], vec![note.clone()]);
	}

	assert!((modulation_value(&rack, 4) - 72.0 / 127.0).abs() < 1e-4);

	assert_eq!(command(&rack, "source.4.fine=50").body, b"{\"ok\":true}");

	for _ in 0..128 {
		process_with(&mut rack, [0.0, 0.0], vec![note.clone()]);
	}

	assert!((modulation_value(&rack, 4) - 72.5 / 127.0).abs() < 1e-4);

	assert_eq!(command(&rack, "source.4.transpose=-24").body, b"{\"ok\":true}");
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
	assert_eq!(command(&rack, "source.4.transpose=99").body, b"{\"ok\":true}");
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

	let json: serde_json::Value =
		serde_json::from_slice(&get(&rack, "/rack").body).expect("JSON");
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
	let group = state.bands[0].slots[0]
		.as_ref()
		.expect("the filter")
		.group;
	assert!(group.is_some());
	assert_eq!(group, state.bands[0].slots[1].as_ref().unwrap().group);

	let json: serde_json::Value =
		serde_json::from_slice(&get(&rack, "/rack").body).expect("JSON");
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

	assert!(rack.state().bands[0].slots[0].as_ref().unwrap().group.is_some());

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

	let json: serde_json::Value =
		serde_json::from_slice(&get(&rack, "/rack").body).expect("JSON");
	let crusher = json["modules"]
		.as_array()
		.expect("modules")
		.iter()
		.find(|module| module["kind"] == "Crusher")
		.expect("the Crusher module");
	let controls = crusher["controls"].as_array().expect("controls");

	assert!(
		controls
			.iter()
			.any(|control| control["id"] == "amount")
	);
	assert!(
		controls
			.iter()
			.any(|control| control["id"] == "frequency")
	);

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

	let json: serde_json::Value =
		serde_json::from_slice(&get(&rack, "/rack").body).expect("JSON");
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

	let json: serde_json::Value =
		serde_json::from_slice(&get(&rack, "/rack").body).expect("JSON");
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

/// The rack page loads the same card assets the product pages use.
#[test]
fn the_rack_page_loads_the_shared_controls() {
	let html = include_str!("page/rack.html");

	assert!(html.contains("/card.css"));
	assert!(html.contains("/card.js"));
	assert!(html.contains("/kit.css"));
	assert!(html.contains("/kit.js"));
	assert!(
		html.find("/card.js").unwrap() < html.find("/kit.js").unwrap(),
		"card.js must load before the page script"
	);
}

/// The band header keeps its trims on the same line as the band name instead of
/// taking a full knob column of vertical space.
#[test]
fn the_band_header_keeps_its_trim_controls_compact() {
	let css = include_str!("page/rack.css");
	let js = include_str!("page/rack.js");

	assert!(css.contains("min-height:52px"), "band header height changed");
	assert!(css.contains(".band-controls .knob-wrap{"), "band control wrapper missing");
	assert!(css.contains("flex-direction:row"), "band controls are not inline");
	assert!(css.contains(".band-controls .knob{width:36px"), "band knob size changed");
	assert!(js.contains("format: function (value)"), "band trims lost their readout");
	assert!(js.contains("name.title"), "the hidden narrow range needs a tooltip");
}

/// The LFO appends points deliberately on double-click, leaving a single click
/// for dragging an existing point or bend handle.
#[test]
fn the_lfo_adds_points_on_double_click() {
	let js = include_str!("page/rack.js");

	assert!(js.contains("svg.ondblclick"), "the LFO needs a double-click handler");
	assert!(js.contains("Double-click to add a point"), "the LFO hint is stale");
	assert!(js.contains("if (!drag) return;"), "an empty single click should not capture the pointer");
}

/// Bypass dims the editable surfaces, while the top bar stays available so the
/// user can turn it back on.
#[test]
fn bypass_dims_the_rack_surfaces() {
	let css = include_str!("page/rack.css");
	let js = include_str!("page/rack.js");

	assert!(css.contains("body.bypassed .rack"), "the rack is not dimmed");
	assert!(css.contains("body.bypassed .bottom"), "the bottom panel is not dimmed");
	assert!(js.contains("document.body.classList.toggle(\"bypassed\""), "bypass does not reach the page");
}

/// The compressor can key off the rack's second input instead of the signal it
/// is processing: a silent sidechain leaves a loud band alone, while the same
/// band keyed off itself is turned down.
#[test]
fn the_compressor_can_key_off_the_sidechain_input() {
	fn render(keyed: bool) -> f32 {
		let mut rack = Rack::default();

		{
			let mut state = rack.shared.lock();
			// Only the parallel lane is audible, so the measurement is the
			// compressor's own output and not the dry bands beside it.
			for band in &mut state.bands {
				band.mute = true;
			}

			let mut compressor = ModuleState::new(0, ModuleKind::Compressor);
			compressor.params.insert("threshold".to_string(), -40.0);
			compressor.params.insert("ratio".to_string(), 10.0);
			compressor.params.insert("attack".to_string(), 1.0);
			compressor.params.insert("release".to_string(), 10.0);
			compressor
				.params
				.insert("sidechain".to_string(), if keyed { 1.0 } else { 0.0 });
			state.parallel.slots[0] = Some(compressor);
		}

		rack.shared.dirty.store(true, Ordering::Release);

		let mut energy = 0.0;

		for i in 0..8_192 {
			let time = i as f32 / 44_100.0;
			let sample = (2.0 * std::f32::consts::PI * 220.0 * time).sin() * 0.5;
			let mut frame = [sample, sample];
			// Port 1 is the sidechain and nobody is feeding it.
			let other = [[0.0_f32; 2], [0.0_f32; 2]];
			let mut context: Box<dyn ProcessContext> = Box::new(());
			rack.process(&mut frame, &other, &mut context);

			if i >= 4_096 {
				energy += frame[0] * frame[0];
			}
		}

		energy
	}

	let unkeyed = render(false);
	let keyed = render(true);

	assert!(
		keyed > unkeyed * 1.5,
		"a silent sidechain left {keyed} of the signal, the keyed input gave {unkeyed}"
	);
}

/// The tempo-controlled modules carry the product's switches, so the free-time
/// knob gives way to the beats knob in the card.
#[test]
fn the_tempo_modules_carry_the_product_switches() {
	assert_eq!(ModuleKind::Delay.switches().len(), 2);
	assert_eq!(ModuleKind::Chorus.switches().len(), 1);
	assert_eq!(ModuleKind::Phaser.switches().len(), 1);
	assert!(ModuleKind::Gain.switches().is_empty());

	let delay = ModuleKind::Delay.switches()[0];
	assert_eq!(delay.control, "tempo.sync");
	assert!(delay.on.contains(&"tempo.beats"));
	assert!(delay.off.contains(&"delay_time"));
}

/// Every product in the catalog can sit in a slot and run. The rack offers the
/// whole line, not a subset it special-cases.
#[test]
fn every_catalog_product_can_run_in_a_slot() {
	for kind in ModuleKind::ALL {
		let mut state = RackState::initial();
		state.bands[0].slots[0] = Some(ModuleState::new(0, kind));

		let mut engine = crate::engine::Engine::new(48_000, &state);
		let mut context: Box<dyn ProcessContext> = Box::new(());
		let mut frame = [0.0_f32; 2];

		for i in 0..256 {
			let time = i as f32 / 48_000.0;
			let sample = (2.0 * std::f32::consts::PI * 220.0 * time).sin() * 0.1;
			frame = [sample, sample];
			engine.process(&mut frame, &[], &mut context);
		}

		assert!(
			frame[0].is_finite() && frame[1].is_finite(),
			"{} produced a non-finite frame",
			kind.name()
		);
	}
}

/// The rack page reads the same /scope document the products do: the heroes
/// that draw audio are useless without it.
#[test]
fn the_page_reads_the_audio_scope() {
	let js = include_str!("page/rack.js");

	assert!(
		js.contains("fetch(\"/module-scope?id=\""),
		"the rack does not read a module's own audio"
	);
	assert!(
		js.contains("fetch(\"/module-stereo?id=\""),
		"the rack does not read a module's stereo pair"
	);
	assert!(js.contains("card.repaintHero"), "a live hero is never redrawn");
	assert!(js.contains("cardContext()"), "the cards share one context");
	assert!(
		js.contains("fetch(\"/module-note?id=\""),
		"the cards never read the note their module hears"
	);
	assert!(
		js.contains("fetch(\"/module-bands?id=\""),
		"the cards never read their band meter"
	);
	assert!(js.contains("spectra: { input: null, output: null }"), "no per-card spectra");
	assert!(
		js.contains("channels: module.scope"),
		"the page polls cards that have no audio reader"
	);
}

/// The scope the page polls is fed by the audio thread, not by the UI thread.
#[test]
fn the_scope_receives_the_signal() {
	let mut rack = Rack::default();
	rack.on_activate(48_000);

	for i in 0..512 {
		let sample = if i % 2 == 0 { 0.5 } else { -0.5 };
		let mut frame = [sample, sample];
		let mut context: Box<dyn ProcessContext> = Box::new(());
		rack.process(&mut frame, &[], &mut context);
	}

	let snapshot = rack.scope.snapshot();
	assert_eq!(snapshot.len(), 4096 * 12, "the scope document has the wrong size");

	// The document is zero-padded up to POINTS, so the newest frame is the one
	// just before the padding starts.
	let newest = 511 * 4;
	let value = f32::from_le_bytes([
		snapshot[newest],
		snapshot[newest + 1],
		snapshot[newest + 2],
		snapshot[newest + 3],
	]);
	assert!(value.abs() > 0.01, "the scope has no signal: {value}");
}

/// A card shows what entered and left its own module: the ring behind the
/// second module holds exactly what the first one produced.
#[test]
fn a_module_scope_holds_what_that_module_heard() {
	let mut rack = Rack::default();
	rack.on_activate(48_000);

	{
		let mut state = rack.shared.lock();
		state.bands[0].slots[0] = Some(ModuleState::new(10, ModuleKind::Gain));
		state.bands[0].slots[1] = Some(ModuleState::new(11, ModuleKind::Gain));
	}
	rack.shared.dirty.store(true, Ordering::Release);

	for i in 0..512 {
		let time = i as f32 / 48_000.0;
		let sample = (2.0 * std::f32::consts::PI * 220.0 * time).sin() * 0.25;
		let mut frame = [sample, sample];
		let mut context: Box<dyn ProcessContext> = Box::new(());
		rack.process(&mut frame, &[], &mut context);
	}

	let first = rack.shared.scope(10, Channels::None);
	let second = rack.shared.scope(11, Channels::None);
	assert!(
		!std::sync::Arc::ptr_eq(&first, &second),
		"two modules share one scope"
	);

	let produced = first.snapshot();
	let heard = second.snapshot();
	let points = 4096 * 4;

	for frame in 0..512 {
		let offset = frame * 4;
		let out = &produced[points + offset..points + offset + 4];
		let into = &heard[offset..offset + 4];
		assert_eq!(out, into, "frame {frame} differs between the modules");
	}
}

/// The rings are allocated from what the card draws: a plot of parameters keeps
/// none, a waveform keeps the mono ones, a goniometer keeps the pair.
#[test]
fn a_card_without_an_audio_plot_keeps_no_rings() {
	let mut rack = Rack::default();
	rack.on_activate(48_000);

	{
		let mut state = rack.shared.lock();
		state.bands[0].slots[0] = Some(ModuleState::new(20, ModuleKind::Delay));
		state.bands[0].slots[1] = Some(ModuleState::new(21, ModuleKind::Filter));
		state.parallel.slots[0] = Some(ModuleState::new(22, ModuleKind::Width));
	}
	rack.shared.dirty.store(true, Ordering::Release);

	for i in 0..64 {
		let time = i as f32 / 48_000.0;
		let sample = (2.0 * std::f32::consts::PI * 220.0 * time).sin() * 0.1;
		let mut frame = [sample, sample];
		let mut context: Box<dyn ProcessContext> = Box::new(());
		rack.process(&mut frame, &[], &mut context);
	}

	assert_eq!(
		rack.shared.scope(20, Channels::None).channels(),
		Channels::None,
		"a delay card reads no audio"
	);
	assert_eq!(
		rack.shared.scope(21, Channels::None).channels(),
		Channels::Mono,
		"a filter card draws its spectrum"
	);
	assert_eq!(
		rack.shared.scope(22, Channels::None).channels(),
		Channels::Stereo,
		"a width card draws the stereo pair"
	);
}

/// The rack offers a note input, which is what the pitch-tracking modules read
/// from the process context; a host that never sends notes still loads it.
#[test]
fn the_rack_declares_a_note_input() {
	use i_am_plugin::Plugin;

	assert_eq!(Rack::INPUT_MIDI_PORTS.len(), 1);
	assert_eq!(Rack::INPUT_MIDI_PORTS[0].name, "Notes");
	assert!(
		ModuleKind::ALL
			.iter()
			.any(|kind| kind.name() == "Autotune"),
		"the autotune is not offered"
	);
}

/// The note a pitch-tracking module hears lands on its scope, which is what the
/// card's badge reads, and the mask it tunes to comes with it.
#[test]
fn a_note_reaches_the_module_that_follows_it() {
	let mut rack = Rack::default();
	rack.on_activate(48_000);

	{
		let mut state = rack.shared.lock();
		state.bands[0].slots[0] = Some(ModuleState::new(30, ModuleKind::Autotune));
	}
	rack.shared.dirty.store(true, Ordering::Release);

	process_with(
		&mut rack,
		[0.0, 0.0],
		vec![NoteEvent::NoteOn {
			channel: 0,
			note: 60,
			velocity: 0.8,
		}],
	);

	let scope = rack.shared.existing_scope(30).expect("the module scope");
	assert_eq!(scope.note(), 60, "the card badge has no note");
	assert!(
		scope.allowed_notes() >= 0,
		"the autotune did not publish its note mask"
	);
}

/// A pitch-tracking module has notes but no audio document, so the page has to
/// poll one without waiting for the other.
#[test]
fn a_note_module_reads_notes_without_audio() {
	assert!(module::shows_notes(ModuleKind::Autotune));
	assert_eq!(
		module::audio_scope(ModuleKind::Autotune),
		Channels::None,
		"the autotune card draws no audio"
	);

	let js = include_str!("page/rack.js");
	assert!(
		js.contains("if (entry.channels) {"),
		"the note poll is still gated behind the audio one"
	);
	assert!(
		!js.contains("if (!entry.channels) return;"),
		"a card without audio never gets its notes"
	);
}

/// A chord lands on the module's scope note by note, which is what the card's
/// badge draws.
#[test]
fn a_chord_reaches_the_module_card() {
	let mut rack = Rack::default();
	rack.on_activate(48_000);

	{
		let mut state = rack.shared.lock();
		state.bands[0].slots[0] = Some(ModuleState::new(31, ModuleKind::Resonator));
	}
	rack.shared.dirty.store(true, Ordering::Release);

	process_with(
		&mut rack,
		[0.0, 0.0],
		[48usize, 55, 64]
			.iter()
			.map(|note| NoteEvent::NoteOn {
				channel: 0,
				note: *note,
				velocity: 0.9,
			})
			.collect(),
	);

	let scope = rack.shared.existing_scope(31).expect("the module scope");
	let held = scope.notes();

	for pitch in [48usize, 55, 64] {
		assert!(
			held & (1u128 << pitch) != 0,
			"note {pitch} is missing from the card"
		);
	}
}

/// The OTT card draws gain reduction, and the vocoder card draws bands: both
/// publish through the module meter, not through the audio scope.
#[test]
fn a_metered_module_publishes_its_bands() {
	let mut rack = Rack::default();
	rack.on_activate(48_000);

	{
		let mut state = rack.shared.lock();
		state.bands[0].slots[0] = Some(ModuleState::new(40, ModuleKind::Ott));
		state.bands[1].slots[0] = Some(ModuleState::new(41, ModuleKind::Vocoder));
	}
	rack.shared.dirty.store(true, Ordering::Release);

	for i in 0..512 {
		let time = i as f32 / 48_000.0;
		let sample = (2.0 * std::f32::consts::PI * 220.0 * time).sin() * 0.3;
		let mut frame = [sample, sample];
		let mut context: Box<dyn ProcessContext> = Box::new(());
		rack.process(&mut frame, &[], &mut context);
	}

	let ott = rack.shared.existing_meters(40).expect("the ott meter");
	let (bands, gains, input, _) = ott.snapshot();
	assert_eq!(bands, 3);
	assert!(gains, "the OTT reports gain reduction");
	assert!(input.iter().all(|value| value.is_finite()));

	let vocoder = rack.shared.existing_meters(41).expect("the vocoder meter");
	let (bands, gains, input, output) = vocoder.snapshot();
	assert!(!gains, "the vocoder reports levels");
	assert!(bands >= 4, "the vocoder runs a band bank");
	assert_eq!(input.len(), bands);
	assert_eq!(output.len(), bands);
}

/// A convolver module owns an impulse response slot, and the page can load one
/// through the same route the product uses.
#[test]
fn a_convolver_module_takes_an_impulse_response() {
	let mut rack = Rack::default();
	rack.on_activate(48_000);

	{
		let mut state = rack.shared.lock();
		state.bands[0].slots[0] = Some(ModuleState::new(42, ModuleKind::Convolver));
	}
	rack.shared.dirty.store(true, Ordering::Release);

	// One block builds the module and, with it, the slot the page talks to.
	process_with(&mut rack, [0.0, 0.0], Vec::new());

	assert!(
		rack.shared.existing_ir(42).is_some(),
		"the convolver has no impulse slot"
	);

	let wav = tiny_wav();
	let request = i_am_web_gui::ProtocolRequest {
		method: "POST",
		path: "/module-ir",
		query: Some("id=42&name=Test.wav"),
		body: &wav,
	};

	assert!(
		crate::web::route(&request, &rack.shared).is_some(),
		"the impulse route refused the file"
	);

	let json = rack
		.shared
		.existing_ir(42)
		.expect("the slot")
		.json();
	assert!(json.contains("Test.wav"), "{json}");
	assert!(!json.contains("\"peaks\":[]"), "the impulse has no picture: {json}");
}

/// A one-sample impulse as a 16-bit mono wav, which is enough to load.
fn tiny_wav() -> Vec<u8> {
	let samples = [30_000i16, 15_000, 7_500, 3_000, 1_000];
	let data_len = (samples.len() * 2) as u32;
	let mut bytes = Vec::new();

	bytes.extend_from_slice(b"RIFF");
	bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
	bytes.extend_from_slice(b"WAVEfmt ");
	bytes.extend_from_slice(&16u32.to_le_bytes());
	bytes.extend_from_slice(&1u16.to_le_bytes());
	bytes.extend_from_slice(&1u16.to_le_bytes());
	bytes.extend_from_slice(&44_100u32.to_le_bytes());
	bytes.extend_from_slice(&(44_100u32 * 2).to_le_bytes());
	bytes.extend_from_slice(&2u16.to_le_bytes());
	bytes.extend_from_slice(&16u16.to_le_bytes());
	bytes.extend_from_slice(b"data");
	bytes.extend_from_slice(&data_len.to_le_bytes());

	for sample in samples {
		bytes.extend_from_slice(&sample.to_le_bytes());
	}

	bytes
}
