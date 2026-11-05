use super::*;
use crate::canvas::{CanvasFeed, canvas_route, default_canvas_text, parse_canvas};
use crate::script::{ScriptFeed, script_route};
use i_am_dsp::prelude::SetValue;
use i_am_plugin::{Plugin, Tag};
use i_am_web_gui::ProtocolRequest;

use i_am_dsp::ProcessInfos;

struct TestContext {
	events: Vec<NoteEvent>,
	infos: ProcessInfos,
}

impl TestContext {
	fn new() -> Self {
		let mut infos = ProcessInfos::default();
		infos.sample_rate = 48_000;
		infos.playing = true;
		Self {
			events: Vec::new(),
			infos,
		}
	}
}

impl ProcessContext for TestContext {
	fn infos(&self) -> &ProcessInfos {
		&self.infos
	}

	fn next_event(&mut self) -> Option<NoteEvent> {
		self.events.pop()
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

fn value(map: &ParamMap, identifier: &str) -> SetValue {
	map.get(identifier)
		.expect(identifier)
		.load(Ordering::Relaxed)
}

#[test]
fn a_synthesizer_has_a_note_port_and_no_audio_input() {
	assert!(<StringWeb as Plugin>::INPUT_PORTS.is_empty());
	assert_eq!(<StringWeb as Plugin>::INPUT_MIDI_PORTS.len(), 1);
	assert_eq!(<StringWeb as Plugin>::INPUT_MIDI_PORTS[0].name, "Notes");

	let tags = <StringWeb as Plugin>::DESCRIPTOR.tags;
	assert!(tags.iter().any(|tag| matches!(tag, Tag::Instrument)));
	assert!(tags.iter().any(|tag| matches!(tag, Tag::Synthesizer)));
}

#[test]
fn the_face_offers_the_four_instruments_and_five_sections() {
	let face = i_am_web_gui::face_json(&StringWeb::new(), &faceplate());

	assert!(face.contains("\"hero\":\"string\""), "{face}");
	assert!(face.contains("\"hero\":\"partials\""), "{face}");
	assert!(face.contains("\"hero\":\"vibrato\""), "{face}");
	assert!(
		face.contains("\"choices\":[\"Violin\",\"Cello\"]"),
		"{face}"
	);
	assert!(
		face.contains(
			"\"pages\":[{\"name\":\"Voice\",\"sections\":[\"String\"]},\
			{\"name\":\"Body\",\"sections\":[\"Body\"]},\
			{\"name\":\"Performance\",\"sections\":[\"Performance\"]}]"
		),
		"{face}"
	);

	for name in ["String", "Body", "Performance"] {
		assert!(face.contains(&format!("\"name\":\"{name}\"")), "{face}");
	}
}

#[test]
fn choosing_an_instrument_moves_the_controls() {
	let plugin = StringWeb::new();
	let map = plugin.param_map();

	map.set("transpose", 5.0f32, Ordering::Relaxed);
	apply_preset_command(&map, 48_000, "preset=1");

	assert!(matches!(value(&map, "preset"), SetValue::Int(1)));
	assert!(
		matches!(value(&map, "gain"), SetValue::Float(gain) if (gain - 0.8).abs() < 1e-6),
		"the cello's gain did not arrive: {:?}",
		value(&map, "gain")
	);
	assert!(
		matches!(value(&map, "transpose"), SetValue::Float(shift) if (shift - 5.0).abs() < 1e-6),
		"the performance transpose was overwritten: {:?}",
		value(&map, "transpose")
	);
}

#[test]
fn a_note_plays_through_the_plugin() {
	let mut plugin = StringWeb::new();
	Plugin::on_activate(&mut plugin, 48_000);

	let mut context: Box<dyn ProcessContext> = Box::new(TestContext {
		events: vec![NoteEvent::NoteOn {
			channel: 0,
			note: 57,
			velocity: 1.0,
		}],
		infos: TestContext::new().infos,
	});
	let mut peak = 0.0f32;

	for _ in 0..24_000 {
		let mut frame = [0.0f32; 2];
		Processor::process(&mut plugin, &mut frame, &[], &mut context);
		peak = peak.max(frame[0].abs());
	}

	assert!(peak > 1e-3, "the instrument was silent: {peak}");
	assert!(peak < 4.0, "the instrument ran away: {peak}");
}

#[test]
fn a_shapes_synth_has_a_note_port_and_no_audio_input() {
	assert!(<ShapesWeb as Plugin>::INPUT_PORTS.is_empty());
	assert_eq!(<ShapesWeb as Plugin>::INPUT_MIDI_PORTS.len(), 1);

	let tags = <ShapesWeb as Plugin>::DESCRIPTOR.tags;
	assert!(tags.iter().any(|tag| matches!(tag, Tag::Instrument)));
	assert!(tags.iter().any(|tag| matches!(tag, Tag::Synthesizer)));
}

#[test]
fn the_shapes_face_is_three_pages_with_one_sync_switch() {
	let face = i_am_web_gui::face_json(&ShapesWeb::new(), &shapes_faceplate());

	assert!(face.contains("\"hero\":\"shape\""), "{face}");
	assert!(face.contains("\"hero\":\"envelope\""), "{face}");
	assert!(face.contains("\"hero\":\"waveform\""), "{face}");
	assert!(
		face.contains("\"id\":\"oscillator.smooth_factor\""),
		"{face}"
	);
	assert!(
		face.contains(
			"\"pages\":[{\"name\":\"Shape\",\"sections\":[\"Shape\"]},\
			{\"name\":\"Envelope\",\"sections\":[\"Envelope\"]},\
			{\"name\":\"Voice\",\"sections\":[\"Voice\"]}]"
		),
		"{face}"
	);
	// One Sync switch in the header, trading the free lengths for beats.
	assert!(face.contains("\"header\":[\"sync\"]"), "{face}");
	assert!(
		face.contains("\"switches\":[{\"control\":\"sync\",\"on\":[\"delay.beats\""),
		"{face}"
	);
	// The per-stage switches stay the host's; the beats lengths are on the face.
	assert!(
		face.contains(
			"\"hidden\":[\"delay.sync\",\"attack.sync\",\"hold.sync\",\"decay.sync\",\"release.sync\"]"
		),
		"{face}"
	);

	for name in ["Shape", "Envelope", "Voice"] {
		assert!(face.contains(&format!("\"name\":\"{name}\"")), "{face}");
	}
}

#[test]
fn a_note_plays_through_the_shapes_plugin() {
	let mut plugin = ShapesWeb::new();
	Plugin::on_activate(&mut plugin, 48_000);

	let mut context: Box<dyn ProcessContext> = Box::new(TestContext {
		events: vec![NoteEvent::NoteOn {
			channel: 0,
			note: 57,
			velocity: 1.0,
		}],
		infos: TestContext::new().infos,
	});
	// A host shows a note only at the sample it arrived at, and the Adsr reads
	// the events without consuming them: clear them after the first frame, as
	// the host would.
	let mut frame = [0.0f32; 2];
	Processor::process(&mut plugin, &mut frame, &[], &mut context);
	context.clear_events();

	let mut peak = frame[0].abs();

	for _ in 1..24_000 {
		let mut frame = [0.0f32; 2];
		Processor::process(&mut plugin, &mut frame, &[], &mut context);
		peak = peak.max(frame[0].abs());
	}

	assert!(peak > 1e-3, "the instrument was silent: {peak}");
	assert!(peak < 4.0, "the instrument ran away: {peak}");
}
#[test]
fn an_additive_synth_has_a_note_port_and_no_audio_input() {
	assert!(<AdditiveWeb as Plugin>::INPUT_PORTS.is_empty());
	assert_eq!(<AdditiveWeb as Plugin>::INPUT_MIDI_PORTS.len(), 1);

	let tags = <AdditiveWeb as Plugin>::DESCRIPTOR.tags;
	assert!(tags.iter().any(|tag| matches!(tag, Tag::Instrument)));
	assert!(tags.iter().any(|tag| matches!(tag, Tag::Synthesizer)));
}

#[test]
fn the_additive_face_has_three_pages_and_a_partial_hero() {
	let face = i_am_web_gui::face_json(&AdditiveWeb::new(), &additive_faceplate());

	assert!(face.contains("\"hero\":\"additive\""), "{face}");
	for id in [
		"oscillator.num_sines",
		"oscillator.freq_gen.bend_amount",
		"oscillator.freq_gen.center_space",
		"oscillator.freq_gen.total_bend_amount",
		"oscillator.freq_gen.scale_amount",
		"oscillator.gain",
	] {
		assert!(face.contains(&format!("\"id\":\"{id}\"")), "{face}");
	}
	for label in ["Bend", "Center", "Overall", "Scale", "Partials"] {
		assert!(face.contains(&format!("\"label\":\"{label}\"")), "{face}");
	}
	assert!(
		face.contains(
			"\"pages\":[{\"name\":\"Partials\",\"sections\":[\"Shape\"]},\
			{\"name\":\"Envelope\",\"sections\":[\"Envelope\"]},\
			{\"name\":\"Voice\",\"sections\":[\"Voice\"]}]"
		),
		"{face}"
	);
	assert!(
		face.contains("\"hidden\":[\"oscillator.gain\"]"),
		"the oscillator's own level stays off the face: {face}"
	);
}

#[test]
fn a_note_plays_through_the_additive_plugin() {
	let mut plugin = AdditiveWeb::new();
	Plugin::on_activate(&mut plugin, 48_000);

	let mut context: Box<dyn ProcessContext> = Box::new(TestContext {
		events: vec![NoteEvent::NoteOn {
			channel: 0,
			note: 57,
			velocity: 1.0,
		}],
		infos: TestContext::new().infos,
	});
	let mut frame = [0.0f32; 2];
	Processor::process(&mut plugin, &mut frame, &[], &mut context);
	context.clear_events();

	let mut peak = frame[0].abs();

	for _ in 1..24_000 {
		let mut frame = [0.0f32; 2];
		Processor::process(&mut plugin, &mut frame, &[], &mut context);
		peak = peak.max(frame[0].abs());
	}

	assert!(peak > 1e-3, "the instrument was silent: {peak}");
	assert!(peak < 4.0, "the instrument ran away: {peak}");
}

#[test]
fn a_spectrum_synth_has_a_note_port_and_no_audio_input() {
	assert!(<SpectrumWeb as Plugin>::INPUT_PORTS.is_empty());
	assert_eq!(<SpectrumWeb as Plugin>::INPUT_MIDI_PORTS.len(), 1);

	let tags = <SpectrumWeb as Plugin>::DESCRIPTOR.tags;
	assert!(tags.iter().any(|tag| matches!(tag, Tag::Instrument)));
	assert!(tags.iter().any(|tag| matches!(tag, Tag::Synthesizer)));
}

#[test]
fn the_spectrum_face_is_three_pages_with_the_partial_hero() {
	let face = i_am_web_gui::face_json(&SpectrumWeb::new(), &spectrum_faceplate());

	assert!(face.contains("\"hero\":\"additive\""), "{face}");
	for id in [
		"oscillator.num_sines",
		"oscillator.freq_gen.tilt",
		"oscillator.freq_gen.odd",
		"oscillator.freq_gen.formant",
		"oscillator.freq_gen.formant_center",
		"oscillator.gain",
	] {
		assert!(face.contains(&format!("\"id\":\"{id}\"")), "{face}");
	}
	for label in ["Tilt", "Odd", "Formant", "Centre", "Partials"] {
		assert!(face.contains(&format!("\"label\":\"{label}\"")), "{face}");
	}
	assert!(
		face.contains(
			"\"pages\":[{\"name\":\"Spectrum\",\"sections\":[\"Shape\"]},\
			{\"name\":\"Envelope\",\"sections\":[\"Envelope\"]},\
			{\"name\":\"Voice\",\"sections\":[\"Voice\"]}]"
		),
		"{face}"
	);
	assert!(
		face.contains("\"hidden\":[\"oscillator.gain\"]"),
		"the oscillator's own level stays off the face: {face}"
	);
}

#[test]
fn a_note_plays_through_the_spectrum_plugin() {
	let mut plugin = SpectrumWeb::new();
	Plugin::on_activate(&mut plugin, 48_000);

	let mut context: Box<dyn ProcessContext> = Box::new(TestContext {
		events: vec![NoteEvent::NoteOn {
			channel: 0,
			note: 57,
			velocity: 1.0,
		}],
		infos: TestContext::new().infos,
	});
	let mut frame = [0.0f32; 2];
	Processor::process(&mut plugin, &mut frame, &[], &mut context);
	context.clear_events();

	let mut peak = frame[0].abs();

	for _ in 1..24_000 {
		let mut frame = [0.0f32; 2];
		Processor::process(&mut plugin, &mut frame, &[], &mut context);
		peak = peak.max(frame[0].abs());
	}

	assert!(peak > 1e-3, "the instrument was silent: {peak}");
	assert!(peak < 4.0, "the instrument ran away: {peak}");
}

#[test]
fn the_script_face_shows_the_script_panel() {
	let face = i_am_web_gui::face_json(&ScriptWeb::new(), &script_faceplate());

	assert!(face.contains("\"script\":true"), "{face}");
	assert!(face.contains("\"hero\":\"additive\""), "{face}");
	for id in [
		"oscillator.freq_gen.a",
		"oscillator.freq_gen.b",
		"oscillator.freq_gen.c",
		"oscillator.freq_gen.d",
		"oscillator.num_sines",
		"oscillator.gain",
	] {
		assert!(face.contains(&format!("\"id\":\"{id}\"")), "{face}");
	}
	assert!(
		face.contains(
			"\"pages\":[{\"name\":\"Partials\",\"sections\":[\"Shape\"]},\
			{\"name\":\"Envelope\",\"sections\":[\"Envelope\"]},\
			{\"name\":\"Voice\",\"sections\":[\"Voice\"]}]"
		),
		"{face}"
	);
	assert!(
		face.contains("\"hidden\":[\"oscillator.gain\",\"oscillator.freq_gen.sample_rate\"]"),
		"the oscillator's level and the host's rate stay off the face: {face}"
	);
}

#[test]
fn the_script_routes_round_trip() {
	let feed = ScriptFeed::new();
	let get = ProtocolRequest {
		method: "GET",
		path: "/script",
		query: None,
		body: &[],
	};
	let body = String::from_utf8(script_route(&get, &feed).expect("the script route").body)
		.expect("utf-8");
	assert!(body.contains("\"source\""), "{body}");
	assert!(body.contains("amplitude"), "{body}");

	let source = "ratio = index + 1.0; amplitude = 1.0; damping = 0.0; phase = 0.0;";
	let post = ProtocolRequest {
		method: "POST",
		path: "/script",
		query: None,
		body: source.as_bytes(),
	};
	let body =
		String::from_utf8(script_route(&post, &feed).expect("the route").body).expect("utf-8");
	assert!(body.contains("\"ok\":true"), "{body}");

	let mut seen = 0;
	assert!(
		feed.take(&mut seen).is_some(),
		"the audio thread should pick the script up"
	);
	assert!(feed.take(&mut seen).is_none(), "and only once");
}

#[test]
fn a_note_plays_through_the_script_plugin() {
	let mut plugin = ScriptWeb::new();
	Plugin::on_activate(&mut plugin, 48_000);

	let mut context: Box<dyn ProcessContext> = Box::new(TestContext {
		events: vec![NoteEvent::NoteOn {
			channel: 0,
			note: 57,
			velocity: 1.0,
		}],
		infos: TestContext::new().infos,
	});
	let mut frame = [0.0f32; 2];
	Processor::process(&mut plugin, &mut frame, &[], &mut context);
	context.clear_events();

	let mut peak = frame[0].abs();

	for _ in 1..24_000 {
		let mut frame = [0.0f32; 2];
		Processor::process(&mut plugin, &mut frame, &[], &mut context);
		peak = peak.max(frame[0].abs());
	}

	assert!(peak > 1e-3, "the instrument was silent: {peak}");
	assert!(peak < 4.0, "the instrument ran away: {peak}");
}
#[test]
fn the_canvas_face_is_three_pages_with_the_drawing_hero() {
	let face = i_am_web_gui::face_json(&CanvasWeb::new(), &canvas_faceplate());

	assert!(face.contains("\"hero\":\"canvas\""), "{face}");
	assert!(
		face.contains("\"id\":\"oscillator.smooth_factor\""),
		"{face}"
	);
	assert!(face.contains("\"label\":\"Morph\""), "{face}");
	assert!(
		face.contains(
			"\"pages\":[{\"name\":\"Canvas\",\"sections\":[\"Shape\"]},\
			{\"name\":\"Envelope\",\"sections\":[\"Envelope\"]},\
			{\"name\":\"Voice\",\"sections\":[\"Voice\"]}]"
		),
		"{face}"
	);
}

#[test]
fn the_canvas_keeps_as_many_pages_and_nodes_as_it_is_given() {
	let mut text = "5".to_string();
	for _ in 0..5 {
		text.push_str(" 0 0 1 0.1 0.1 0.2 0.2 0.3 0.3");
	}

	let pages = parse_canvas(&text).expect("five pages parse");
	assert_eq!(pages.len(), 5);
	assert_eq!(pages[0].graph.len(), 1);

	let mut nodes = "1 0 0 40".to_string();
	for index in 0..40 {
		let value = index as f32 * 0.01;
		nodes.push_str(&format!(" {value} {value} {value} {value} {value} {value}"));
	}

	let pages = parse_canvas(&nodes).expect("forty nodes parse");
	assert_eq!(pages[0].graph.len(), 40);
}

#[test]
fn the_canvas_route_round_trips() {
	let feed = CanvasFeed::new();
	let get = ProtocolRequest {
		method: "GET",
		path: "/canvas",
		query: None,
		body: &[],
	};
	let body = String::from_utf8(canvas_route(&get, &feed).expect("the canvas route").body)
		.expect("utf-8");
	assert!(body.contains("\"shape\""), "{body}");

	let text = default_canvas_text();
	let post = ProtocolRequest {
		method: "POST",
		path: "/canvas",
		query: None,
		body: text.as_bytes(),
	};
	let body =
		String::from_utf8(canvas_route(&post, &feed).expect("the route").body).expect("utf-8");
	assert!(body.contains("\"ok\":true"), "{body}");

	let mut seen = 0;
	assert!(
		feed.take(&mut seen).is_some(),
		"the audio thread should pick the drawing up"
	);
	assert!(feed.take(&mut seen).is_none(), "and only once");
}

#[test]
fn a_note_plays_through_the_canvas_plugin() {
	let mut plugin = CanvasWeb::new();
	Plugin::on_activate(&mut plugin, 48_000);

	let mut context: Box<dyn ProcessContext> = Box::new(TestContext {
		events: vec![NoteEvent::NoteOn {
			channel: 0,
			note: 57,
			velocity: 1.0,
		}],
		infos: TestContext::new().infos,
	});
	let mut frame = [0.0f32; 2];
	Processor::process(&mut plugin, &mut frame, &[], &mut context);
	context.clear_events();

	let mut peak = frame[0].abs();

	for _ in 1..24_000 {
		let mut frame = [0.0f32; 2];
		Processor::process(&mut plugin, &mut frame, &[], &mut context);
		peak = peak.max(frame[0].abs());
	}

	assert!(peak > 1e-3, "the instrument was silent: {peak}");
	assert!(peak < 4.0, "the instrument ran away: {peak}");
}
#[test]
fn the_fitted_instrument_plays_through_the_plugin() {
	let mut plugin = FittedWeb::new();
	Plugin::on_activate(&mut plugin, 48_000);

	let mut context: Box<dyn ProcessContext> = Box::new(TestContext {
		events: vec![NoteEvent::NoteOn {
			channel: 0,
			note: 57,
			velocity: 1.0,
		}],
		infos: TestContext::new().infos,
	});
	let mut peak = 0.0f32;
	for _ in 0..24_000 {
		let mut frame = [0.0f32; 2];
		Processor::process(&mut plugin, &mut frame, &[], &mut context);
		peak = peak.max(frame[0].abs());
	}
	assert!(peak > 1e-3, "the fitted instrument was silent: {peak}");
}
#[test]
fn the_modal_face_has_its_hero() {
	let face = i_am_web_gui::face_json(&FittedWeb::new(), &fitted_faceplate());
	assert!(face.contains("\"hero\":\"modal\""), "{face}");
	assert!(face.contains("\"id\":\"attack\""), "{face}");
	// The instrument picker lives in the header and offers every table.
	assert!(face.contains("\"header\":["), "{face}");
	assert!(face.contains("\"instrument\""), "{face}");
	for name in ["The Gentleman", "Peel Guitar", "Bass", "Celeste", "Harp"] {
		assert!(face.contains(name), "missing {name}: {face}");
	}
}