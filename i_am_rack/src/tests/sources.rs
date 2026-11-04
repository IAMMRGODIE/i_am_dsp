//! Impulses, scripts and the macro writers.

use super::*;

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

	let json = rack.shared.existing_ir(42).expect("the slot").json();
	assert!(json.contains("Test.wav"), "{json}");
	assert!(
		!json.contains("\"peaks\":[]"),
		"the impulse has no picture: {json}"
	);
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

/// A preset carries the convolver's impulse response: loading the state into a
/// fresh rack puts the room back into the module's slot.
#[test]
fn a_preset_keeps_the_impulse_response() {
	let mut rack = Rack::default();
	rack.on_activate(48_000);

	{
		let mut state = rack.shared.lock();
		state.bands[0].slots[0] = Some(ModuleState::new(43, ModuleKind::Convolver));
	}
	rack.shared.dirty.store(true, Ordering::Release);
	process_with(&mut rack, [0.0, 0.0], Vec::new());

	let wav = tiny_wav();
	let request = i_am_web_gui::ProtocolRequest {
		method: "POST",
		path: "/module-ir",
		query: Some("id=43&name=Room.wav"),
		body: &wav,
	};
	assert!(crate::web::route(&request, &rack.shared).is_some());

	let saved = serde_json::to_string(&rack.state()).expect("the preset");
	assert!(saved.contains("Room.wav"), "the preset dropped the impulse");

	let mut restored = Rack::default();
	let parsed: RackState = serde_json::from_str(&saved).expect("the preset parses");
	*restored.shared.lock() = parsed;
	restored.shared.dirty.store(true, Ordering::Release);
	process_with(&mut restored, [0.0, 0.0], Vec::new());

	let json = restored
		.shared
		.existing_ir(43)
		.expect("the restored slot")
		.json();
	assert!(
		json.contains("Room.wav"),
		"the room did not come back: {json}"
	);
	assert!(
		!json.contains("\"peaks\":[]"),
		"the restored impulse has no picture: {json}"
	);
}

/// A scripted module hands its source to a worker, which renders the curve the
/// card draws. The route is the same shape the product page uses.
#[test]
fn a_scripted_module_takes_a_script() {
	let mut rack = Rack::default();
	rack.on_activate(48_000);

	{
		let mut state = rack.shared.lock();
		state.bands[0].slots[0] = Some(ModuleState::new(50, ModuleKind::ScriptDistortion));
	}
	rack.shared.dirty.store(true, Ordering::Release);
	process_with(&mut rack, [0.0, 0.0], Vec::new());

	let source = "tanh(x * 2.0)";
	let request = i_am_web_gui::ProtocolRequest {
		method: "POST",
		path: "/module-script",
		query: Some("id=50"),
		body: source.as_bytes(),
	};
	assert!(
		crate::web::route(&request, &rack.shared).is_some(),
		"the script route refused the source"
	);

	let script = rack.shared.existing_script(50).expect("the worker");
	assert!(
		script.json().contains("tanh(x * 2.0)"),
		"the worker did not take the source"
	);

	// The render happens on the worker's thread.
	for _ in 0..200 {
		if !script.curve_json().contains("\"points\":[]") {
			return;
		}

		std::thread::sleep(std::time::Duration::from_millis(10));
	}

	panic!("the script never rendered a curve");
}

/// A script that does not compile reports its error, which the card shows under
/// the editor instead of only in a tooltip.
#[test]
fn a_broken_script_reports_its_error() {
	let mut rack = Rack::default();
	rack.on_activate(48_000);

	{
		let mut state = rack.shared.lock();
		state.bands[0].slots[0] = Some(ModuleState::new(51, ModuleKind::ScriptMapper));
	}
	rack.shared.dirty.store(true, Ordering::Release);
	process_with(&mut rack, [0.0, 0.0], Vec::new());

	let request = i_am_web_gui::ProtocolRequest {
		method: "POST",
		path: "/module-script",
		query: Some("id=51"),
		body: b"x * (1.0 +",
	};
	assert!(crate::web::route(&request, &rack.shared).is_some());

	let script = rack.shared.existing_script(51).expect("the worker");

	for _ in 0..200 {
		let json = script.json();

		if json.contains("\"error\":\"") {
			return;
		}

		std::thread::sleep(std::time::Duration::from_millis(10));
	}

	panic!("the broken script reported no error: {}", script.json());
}

/// A source links to a target once: a second drag of the same source onto the
/// same knob does not add a second row.
#[test]
fn a_source_links_to_a_target_once() {
	let rack = Rack::default();

	command(&rack, "assign.add=macro1|band.0.gain");
	command(&rack, "assign.add=macro1|band.0.gain");

	let state = rack.state();
	let active = state
		.assignments
		.iter()
		.flatten()
		.filter(|assignment| assignment.target == Some(Target::BandGain(0)))
		.count();

	assert_eq!(active, 1, "the same link was added twice");
}

/// Editing a row into a duplicate releases the older row instead of leaving two
/// rows on one knob.
#[test]
fn a_duplicate_edit_releases_the_older_row() {
	let rack = Rack::default();

	command(&rack, "assign.add=macro1|band.0.gain");
	command(&rack, "assign.add=macro2|band.1.gain");
	command(&rack, "assign.1.source=macro1");
	command(&rack, "assign.1.target=band.0.gain");

	let state = rack.state();
	let rows = &state.assignments;

	assert_eq!(
		rows[0]
			.as_ref()
			.and_then(|assignment| assignment.target.clone()),
		None,
		"the older row still holds the same target"
	);
	assert_eq!(
		rows[1]
			.as_ref()
			.and_then(|assignment| assignment.target.clone()),
		Some(Target::BandGain(0)),
		"the edited row lost its target"
	);
}

/// Dragging a macro onto a module knob links it, the same way a source does.
#[test]
fn a_macro_links_to_a_module_knob() {
	let rack = Rack::default();

	command(&rack, "assign.add=macro1|m0.cutoff");

	let state = rack.state();
	let linked = state.assignments.iter().flatten().any(|assignment| {
		assignment.target
			== Some(Target::Module {
				module: 0,
				parameter: "cutoff".to_string(),
			})
	});

	assert!(linked, "the macro link was not stored");
}

/// The live route is the only thing the page polls often, so the macro and
/// automation values the host writes have to ride on it.
#[test]
fn the_live_route_carries_the_macro_values() {
	let rack = Rack::default();

	command(&rack, "macro.0=0.25");

	// What the host writes: the map is the truth the page reads.
	rack.shared.params.set(
		"automation4",
		i_am_dsp::prelude::SetValue::Float(0.75),
		std::sync::atomic::Ordering::SeqCst,
	);

	let request = i_am_web_gui::ProtocolRequest {
		method: "GET",
		path: "/modulation",
		query: None,
		body: b"",
	};
	let response = crate::web::route(&request, &rack.shared).expect("the live route");
	let body = String::from_utf8_lossy(&response.body);

	assert!(
		body.contains("\"macros\""),
		"no macros on the live route: {body}"
	);
	assert!(body.contains("0.25"), "the macro value is missing: {body}");
	assert!(
		body.contains("0.75"),
		"the automation value is missing: {body}"
	);
}

/// The host writes a parameter by storing into the map; the plug-in has to pick
/// that up the way Paramed's sync does.
#[test]
fn a_host_macro_write_reaches_the_state() {
	use i_am_dsp::prelude::SetValue;

	let mut rack = Rack::default();
	let map = i_am_plugin::Plugin::param_map(&rack);

	map.set(
		"macro1",
		SetValue::Float(0.8),
		std::sync::atomic::Ordering::SeqCst,
	);

	for _ in 0..70 {
		process_with(&mut rack, [0.0, 0.0], Vec::new());
	}

	assert_eq!(
		rack.state().macros[0],
		0.8,
		"the host's value never reached the state"
	);
}

/// The page writes a macro; the map has to carry the new value *and* say it
/// changed, or the host is never told.
#[test]
fn a_page_macro_write_reaches_the_host_map() {
	use i_am_dsp::prelude::SetValue;

	let rack = Rack::default();
	command(&rack, "macro.0=0.6");

	let map = i_am_plugin::Plugin::param_map(&rack);
	let param = map.get("macro1").expect("macro1 in the map");

	assert_eq!(
		param.load(std::sync::atomic::Ordering::SeqCst),
		SetValue::Float(0.6)
	);
	assert!(
		param.is_chanegd(),
		"the host was never told the macro moved"
	);
}

/// A macro the host writes has to reach the DSP, not only the state: the engine
/// is where the modulation is applied.
#[test]
fn a_host_macro_write_modulates_the_engine() {
	use i_am_dsp::prelude::SetValue;

	let mut rack = Rack::default();

	command(&rack, "assign.add=macro1|m0.cutoff");
	command(&rack, "assign.0.depth=0.5");

	let base = rack
		.module_parameter(0, "cutoff")
		.expect("the filter cutoff");

	let map = i_am_plugin::Plugin::param_map(&rack);
	map.set(
		"macro1",
		SetValue::Float(1.0),
		std::sync::atomic::Ordering::SeqCst,
	);

	// One block: the host's value is the one the very next block runs with, the
	// way Paramed syncs it.
	process_with(&mut rack, [0.0, 0.0], Vec::new());

	assert_eq!(
		rack.engine_macros()[0],
		1.0,
		"the host's macro never reached the engine"
	);
	assert_eq!(
		rack.state().macros[0],
		1.0,
		"the state still has the old macro"
	);

	// The modulation itself is smoothed, so it needs a few blocks to show.
	for _ in 0..128 {
		process_with(&mut rack, [0.0, 0.0], Vec::new());
	}

	let cutoff = rack
		.module_parameter(0, "cutoff")
		.expect("the filter cutoff");

	assert!(
		cutoff > base + 100.0,
		"the host's macro left the cutoff at {cutoff} (base {base})"
	);
}

/// A scripted module's source is part of the preset: the workers outlive the
/// module, so a project has to carry what they should be running.
#[test]
fn a_preset_keeps_the_script_source() {
	let mut rack = Rack::default();
	rack.on_activate(48_000);

	{
		let mut state = rack.shared.lock();
		state.bands[0].slots[0] = Some(ModuleState::new(60, ModuleKind::ScriptMapper));
	}
	rack.shared.dirty.store(true, Ordering::Release);
	process_with(&mut rack, [0.0, 0.0], Vec::new());

	let source = "x * 0.5";
	let write = i_am_web_gui::ProtocolRequest {
		method: "POST",
		path: "/module-script",
		query: Some("id=60"),
		body: source.as_bytes(),
	};
	assert!(
		crate::web::route(&write, &rack.shared).is_some(),
		"the script route refused the source"
	);

	// What the page saves: the whole state as the preset document.
	let saved = crate::web::route(
		&i_am_web_gui::ProtocolRequest {
			method: "GET",
			path: "/state",
			query: None,
			body: b"",
		},
		&rack.shared,
	)
	.expect("the state route");
	let document = String::from_utf8_lossy(&saved.body).to_string();

	let loaded = Rack::default();
	let body = format!("preset.load={document}");
	let load = i_am_web_gui::ProtocolRequest {
		method: "POST",
		path: "/rack-command",
		query: None,
		body: body.as_bytes(),
	};
	assert!(
		crate::web::route(&load, &loaded.shared).is_some(),
		"the preset was refused"
	);

	let script = loaded.shared.existing_script(60).expect("the worker");
	let json = script.json();

	assert!(
		json.contains("x * 0.5"),
		"the preset did not restore the script: {json}"
	);
}
