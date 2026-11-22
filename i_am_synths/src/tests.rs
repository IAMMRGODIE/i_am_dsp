use super::*;
use crate::canvas::{CanvasFeed, canvas_route, default_canvas_text, parse_canvas};
use crate::script::{ScriptFeed, script_route};
use crate::spectrum::spectrum_route;
use i_am_dsp::prelude::{ImportedTable, SetValue, SpectralFeed, SpectralFrame, TableSlot};
use i_am_plugin::{Plugin, Tag};
use i_am_web_gui::ProtocolRequest;
use std::sync::Arc;

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
	assert!(face.contains("\"hero\":\"unison\""), "{face}");
	assert!(
		face.contains("\"id\":\"oscillator.smooth_factor\""),
		"{face}"
	);
	// The edge crossfade sits beside the morph, where the imported table is.
	assert!(face.contains("\"id\":\"oscillator.x_fade\""), "{face}");
	assert!(face.contains("\"label\":\"X-Fade\""), "{face}");
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
	// One Sync switch in the header, trading the free lengths for beats.
	assert!(face.contains("\"header\":[\"sync\"]"), "{face}");
	assert!(
		face.contains("\"switches\":[{\"control\":\"sync\",\"on\":[\"delay.beats\""),
		"{face}"
	);
	// The per-stage switches stay the host's; the oscillator's own level stays off
	// the face.
	assert!(
		face.contains(
			"\"hidden\":[\"delay.sync\",\"attack.sync\",\"hold.sync\",\"decay.sync\",\"release.sync\",\"oscillator.gain\"]"
		),
		"{face}"
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
fn the_spectrum_face_draws_the_drawn_pages() {
	let face = i_am_web_gui::face_json(&SpectrumWeb::new(), &spectrum_faceplate());

	// The page is the picture, and Morph is the one control the drawing adds.
	assert!(face.contains("\"hero\":\"morph-spectrum\""), "{face}");
	assert!(face.contains("\"id\":\"oscillator.morph\""), "{face}");
	assert!(face.contains("\"label\":\"Morph\""), "{face}");

	// The pages are state, not a control: the host saves them and the face does
	// not draw them as a knob.
	assert!(!face.contains("\"id\":\"oscillator.frames\""), "{face}");

	assert!(
		face.contains(
			"\"pages\":[{\"name\":\"Spectrum\",\"sections\":[\"Shape\"]},\
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
	// The per-stage switches stay the host's.
	assert!(
		face.contains(
			"\"hidden\":[\"delay.sync\",\"attack.sync\",\"hold.sync\",\"decay.sync\",\"release.sync\"]"
		),
		"{face}"
	);
}

/// A project save carries the drawn pages: the parameter map only sees the
/// morph knob, so the pages ride the serialized state and are restored into the
/// same feed the oscillator renders from.
#[test]
fn the_drawn_pages_survive_a_state_round_trip() {
	use i_am_dsp::prelude::{Value, from_binary, to_binary};

	let source = SpectrumWeb::new();
	let saved = source.get_parameters();
	let value = saved
		.iter()
		.find(|parameter| parameter.identifier == "oscillator.frames")
		.expect("the frames parameter")
		.value
		.clone();
	let Value::Serialized(data) = value else {
		panic!("the pages are not serialized")
	};
	let mut pages = from_binary::<Vec<SpectralFrame>>(data).expect("the pages decode");
	pages[0].levels[0] = 0.25;

	let mut restored = SpectrumWeb::new();
	assert!(restored.set_parameter(
		"oscillator.frames",
		SetValue::Serialized(to_binary(&pages).expect("the pages encode")),
	));
	assert_eq!(restored.feed().frames()[0].levels[0], 0.25);
}

/// The editor asks for the pages and posts new ones; both go through the feed
/// the voice renders from.
#[test]
fn the_spectrum_route_round_trips_the_pages() {
	let feed = SpectralFeed::new(vec![SpectralFrame::saw()]);
	let get = ProtocolRequest {
		method: "GET",
		path: "/spectrum",
		query: None,
		body: &[],
	};
	let body = String::from_utf8(
		spectrum_route(&get, &feed)
			.expect("the spectrum route")
			.body,
	)
	.expect("utf-8");

	assert!(body.contains("\"generation\""), "{body}");
	assert!(body.contains("\"levels\""), "{body}");

	let page = SpectralFrame::square();
	let json = serde_json::to_string(&vec![page.clone()]).expect("the pages serialize");
	let post = ProtocolRequest {
		method: "POST",
		path: "/spectrum",
		query: None,
		body: json.as_bytes(),
	};
	let body = String::from_utf8(spectrum_route(&post, &feed).expect("the post route").body)
		.expect("utf-8");

	assert!(body.contains("\"ok\":true"), "{body}");
	assert_eq!(*feed.frames(), vec![page]);
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
fn a_page_edit_reaches_a_held_spectrum_note() {
	let handle = SpectrumHandle::new();
	let mut plugin = SpectrumWeb::with_handle(handle.clone());
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

	for _ in 0..2_000 {
		let mut frame = [0.0f32; 2];
		Processor::process(&mut plugin, &mut frame, &[], &mut context);
		peak = peak.max(frame[0].abs());
		context.clear_events();
	}

	assert!(peak > 1e-3, "the held note was silent: {peak}");

	let silent = vec![SpectralFrame::new(vec![0.0; 512], vec![0.0; 512])];
	let result = handle.set_json(&serde_json::to_string(&silent).expect("silent pages"));
	assert!(result.contains("\"ok\":true"), "{result}");

	// Let the renderer publish the new tables, then judge the tail rather than
	// the first frames, which still belong to the old bank.
	for _ in 0..4_000 {
		let mut frame = [0.0f32; 2];
		Processor::process(&mut plugin, &mut frame, &[], &mut context);
	}

	let mut after = 0.0f32;

	for _ in 0..1_000 {
		let mut frame = [0.0f32; 2];
		Processor::process(&mut plugin, &mut frame, &[], &mut context);
		after = after.max(frame[0].abs());
	}

	assert!(
		after < peak * 0.1,
		"the held note kept the old tables: {after} vs {peak}"
	);
}

#[test]
fn a_sample_edit_reaches_a_held_sampler_note() {
	use i_am_dsp::prelude::{SampleData, SampleSlot};
	use std::f32::consts::PI;
	use std::sync::Mutex;

	let slot = Arc::new(Mutex::new(SampleSlot::default()));
	{
		let mut guard = slot.lock().expect("slot");
		guard.sample = Some(Arc::new(SampleData {
			name: "held".to_string(),
			rate: 48_000,
			pcm: (0..24_000)
				.map(|index| ((index as f32) * 2.0 * PI * 440.0 / 48_000.0).sin())
				.collect(),
			right: None,
		}));
	}

	let mut plugin = SamplerWeb::with_slot(slot.clone());
	Plugin::on_activate(&mut plugin, 48_000);

	let mut context: Box<dyn ProcessContext> = Box::new(TestContext {
		events: vec![NoteEvent::NoteOn {
			channel: 0,
			note: 60,
			velocity: 1.0,
		}],
		infos: TestContext::new().infos,
	});
	let mut peak = 0.0f32;

	for _ in 0..2_000 {
		let mut frame = [0.0f32; 2];
		Processor::process(&mut plugin, &mut frame, &[], &mut context);
		peak = peak.max(frame[0].abs());
		context.clear_events();
	}

	assert!(peak > 1e-3, "the held sample was silent: {peak}");

	slot.lock().expect("slot").sample = None;

	for _ in 0..4_000 {
		let mut frame = [0.0f32; 2];
		Processor::process(&mut plugin, &mut frame, &[], &mut context);
	}

	let mut after = 0.0f32;

	for _ in 0..1_000 {
		let mut frame = [0.0f32; 2];
		Processor::process(&mut plugin, &mut frame, &[], &mut context);
		after = after.max(frame[0].abs());
	}

	assert!(
		after < peak * 0.1,
		"the held sampler kept the old sample: {after} vs {peak}"
	);
}

#[test]
fn a_table_edit_reaches_a_held_shapes_note() {
	use i_am_dsp::prelude::{WAVE_FRAME_LEN, WaveBank, WaveSlot};
	use std::sync::Mutex;

	let slot = Arc::new(Mutex::new(WaveSlot::default()));
	let mut plugin = ShapesWeb::with_slot(slot.clone());
	Plugin::on_activate(&mut plugin, 48_000);

	let mut context: Box<dyn ProcessContext> = Box::new(TestContext {
		events: vec![NoteEvent::NoteOn {
			channel: 0,
			note: 60,
			velocity: 1.0,
		}],
		infos: TestContext::new().infos,
	});
	let mut peak = 0.0f32;

	for _ in 0..2_000 {
		let mut frame = [0.0f32; 2];
		Processor::process(&mut plugin, &mut frame, &[], &mut context);
		peak = peak.max(frame[0].abs());
		context.clear_events();
	}

	assert!(peak > 1e-3, "the classic shapes were silent: {peak}");

	slot.lock().expect("slot").bank = Some(WaveBank::new(None, vec![vec![0.0; WAVE_FRAME_LEN]]));

	for _ in 0..4_000 {
		let mut frame = [0.0f32; 2];
		Processor::process(&mut plugin, &mut frame, &[], &mut context);
	}

	let mut after = 0.0f32;

	for _ in 0..1_000 {
		let mut frame = [0.0f32; 2];
		Processor::process(&mut plugin, &mut frame, &[], &mut context);
		after = after.max(frame[0].abs());
	}

	assert!(
		after < peak * 0.1,
		"the held shapes kept the old table: {after} vs {peak}"
	);
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
	// One Sync switch in the header, trading the free lengths for beats.
	assert!(face.contains("\"header\":[\"sync\"]"), "{face}");
	assert!(
		face.contains("\"switches\":[{\"control\":\"sync\",\"on\":[\"delay.beats\""),
		"{face}"
	);
	// The per-stage switches stay the host's; the oscillator's level and the
	// host's rate stay off the face.
	assert!(
		face.contains(
			"\"hidden\":[\"delay.sync\",\"attack.sync\",\"hold.sync\",\"decay.sync\",\"release.sync\",\"oscillator.gain\",\"oscillator.freq_gen.sample_rate\"]"
		),
		"{face}"
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

	assert!(
		feed.current_script().is_some(),
		"every voice should see the script the page applied"
	);
	assert_eq!(feed.source(), source);
}

/// The body the Script page starts with has to run: a script that fails to
/// evaluate leaves the instrument silent.
#[test]
fn the_default_script_evaluates() {
	use i_am_dsp::prelude::{GenFreqInfo, RhaiFreqGen, RhaiScript};

	let source = crate::script::SCRIPT_DEFAULT;
	let script = RhaiScript::compile(source).expect("the default script compiles");
	let mut generator = RhaiFreqGen::new(48_000);
	generator.load_compiled(script);

	let fundamental = generator.gen_info(0, 2);

	assert_eq!(fundamental.ratio, 1.0);
	assert!((fundamental.amplitude - 1.0).abs() < 1e-6);
	assert!(
		generator.get_eval_error().is_ok(),
		"the default script must evaluate"
	);
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
	// A sample fitted an octave off is fixed with the transpose, so it has to be
	// on the face, and it has to step in semitones from a centre of zero.
	assert!(face.contains("\"id\":\"transpose\""), "{face}");
	assert!(
		face.contains("\"min\":-24,\"max\":24,\"default\":0"),
		"{face}"
	);
	assert!(face.contains("\"bipolar\":true"), "{face}");
	assert!(face.contains("\"step\":1,"), "{face}");
	// The instrument picker lives in the header and offers every table.
	assert!(face.contains("\"header\":["), "{face}");
	assert!(face.contains("\"instrument\""), "{face}");
	for name in [
		"The Gentleman",
		"Peel Guitar",
		"Bass",
		"Celeste",
		"Harp",
		"Custom",
	] {
		assert!(face.contains(name), "missing {name}: {face}");
	}
}
#[test]
fn an_imported_table_plays_and_survives_the_state() {
	use i_am_dsp::prelude::{ImportedTable, SetValue, to_binary};

	let path = concat!(
		env!("CARGO_MANIFEST_DIR"),
		"/../i_am_string/assets/peel.bin"
	);
	let bytes = std::fs::read(path).expect("read the table");
	let binary = to_binary(&Some(ImportedTable {
		name: "peel".to_string(),
		bytes,
	}))
	.expect("serialize the table");

	let mut plugin = FittedWeb::new();
	Plugin::on_activate(&mut plugin, 48_000);
	assert!(Parameters::set_parameter(
		&mut plugin,
		"imported",
		SetValue::Serialized(binary.clone())
	));
	assert!(Parameters::set_parameter(
		&mut plugin,
		"instrument",
		SetValue::Int(5)
	));

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
	assert!(peak > 1e-4, "the imported table was silent: {peak}");
}
#[test]
fn the_drummer_plays_the_kick_on_note_36() {
	let face = i_am_web_gui::face_json(&DrummerWeb::new(), &drummer_faceplate());
	for id in ["tune", "click", "click_tone", "drive", "gain", "level_kick"] {
		assert!(
			face.contains(&format!("\"id\":\"{id}\"")),
			"missing {id}: {face}"
		);
	}
	assert!(face.contains("\"hero\":\"kick\""), "{face}");

	let tags = <DrummerWeb as Plugin>::DESCRIPTOR.tags;
	assert!(tags.iter().any(|tag| matches!(tag, Tag::Drum)));

	let mut plugin = DrummerWeb::new();
	Plugin::on_activate(&mut plugin, 48_000);
	let mut context: Box<dyn ProcessContext> = Box::new(TestContext {
		events: vec![NoteEvent::NoteOn {
			channel: 0,
			note: 36,
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
	assert!(peak > 1e-3, "the kick was silent: {peak}");
}

#[test]
fn the_drummer_ignores_a_note_it_has_no_voice_for() {
	let mut plugin = DrummerWeb::new();
	Plugin::on_activate(&mut plugin, 48_000);
	let mut context: Box<dyn ProcessContext> = Box::new(TestContext {
		events: vec![NoteEvent::NoteOn {
			channel: 0,
			note: 60,
			velocity: 1.0,
		}],
		infos: TestContext::new().infos,
	});

	let mut peak = 0.0f32;
	for _ in 0..4_800 {
		let mut frame = [0.0f32; 2];
		Processor::process(&mut plugin, &mut frame, &[], &mut context);
		peak = peak.max(frame[0].abs());
	}

	assert_eq!(peak, 0.0, "an unmapped note made a sound: {peak}");
}

#[test]
fn the_pitch_curve_survives_the_state() {
	use i_am_dsp::prelude::{Curve, SetValue, Value, to_binary};

	let curve = Curve::from_triples([(0.0, 6.0, -1.0), (0.04, 1.0, 0.0)]);
	let bytes = to_binary(&curve).expect("serialize the curve");

	let mut plugin = DrummerWeb::new();
	assert!(Parameters::set_parameter(
		&mut plugin,
		"pitch_curve",
		SetValue::Serialized(bytes.clone())
	));

	let stored = Parameters::get_parameters(&plugin)
		.into_iter()
		.find(|parameter| parameter.identifier == "pitch_curve")
		.expect("the curve is a parameter");

	match stored.value {
		Value::Serialized(data) => assert_eq!(data, bytes),
		other => panic!("unexpected value: {other:?}"),
	}
}

#[test]
fn the_drummer_kit_choice_writes_the_knobs() {
	let mut plugin = DrummerWeb::new();
	assert!(Parameters::set_parameter(
		&mut plugin,
		"kit",
		SetValue::Int(1)
	));

	let tune = Parameters::get_parameters(&plugin)
		.into_iter()
		.find(|parameter| parameter.identifier == "tune")
		.expect("the kick's tune");

	let applied = tune.value.to_set_value();

	assert!(
		matches!(applied, SetValue::Float(value) if (value - 48.0).abs() < 1e-4),
		"the 808 kit did not write the tune: {applied:?}"
	);
}

#[test]
fn the_drummer_plays_in_stereo() {
	// The generator used to hand the host two copies of its mono sum, so a pan
	// or a width never left the plug-in. This is the host's own path: process a
	// real frame and check the channels differ.
	let mut plugin = DrummerWeb::new();
	Plugin::on_activate(&mut plugin, 48_000);

	let mut context: Box<dyn ProcessContext> = Box::new(TestContext {
		events: vec![NoteEvent::NoteOn {
			channel: 0,
			note: 42,
			velocity: 1.0,
		}],
		infos: TestContext::new().infos,
	});
	let mut spread = 0.0f32;

	for _ in 0..4_800 {
		let mut frame = [0.0f32; 2];
		Processor::process(&mut plugin, &mut frame, &[], &mut context);
		spread += (frame[0] - frame[1]).abs();
	}

	assert!(spread > 0.01, "the host only saw the mono sum: {spread}");
}

/// A 16 bit mono WAV holding one note, as a page would post it.
fn wav_of(samples: &[f32], rate: u32) -> Vec<u8> {
	let data = (samples.len() * 2) as u32;
	let mut out = Vec::new();
	out.extend_from_slice(b"RIFF");
	out.extend_from_slice(&(36 + data).to_le_bytes());
	out.extend_from_slice(b"WAVEfmt ");
	out.extend_from_slice(&16u32.to_le_bytes());
	out.extend_from_slice(&1u16.to_le_bytes());
	out.extend_from_slice(&1u16.to_le_bytes());
	out.extend_from_slice(&rate.to_le_bytes());
	out.extend_from_slice(&(rate * 2).to_le_bytes());
	out.extend_from_slice(&2u16.to_le_bytes());
	out.extend_from_slice(&16u16.to_le_bytes());
	out.extend_from_slice(b"data");
	out.extend_from_slice(&data.to_le_bytes());
	for sample in samples {
		let value = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
		out.extend_from_slice(&value.to_le_bytes());
	}
	out
}

/// A parameter map with the four values the sample route writes.
fn modal_map() -> ParamMap {
	let int = |value: i32, range: std::ops::RangeInclusive<i32>| {
		Arc::new(
			i_am_dsp::prelude::Value::Int {
				value,
				range,
				logarithmic: false,
			}
			.to_atomic_value(),
		)
	};

	ParamMap::new_with([
		("instrument".to_string(), int(0, 0..=5)),
		("transpose".to_string(), int(0, -24..=24)),
		(
			"derive".to_string(),
			Arc::new(i_am_dsp::prelude::Value::Bool(false).to_atomic_value()),
		),
		(
			"imported".to_string(),
			Arc::new(i_am_dsp::prelude::Value::Nothing.to_atomic_value()),
		),
	])
}

/// One recorded note at 220 Hz, as a page would post it.
fn recorded_note(rate: u32) -> Vec<u8> {
	let samples: Vec<f32> = (0..rate)
		.map(|index| {
			let time = index as f32 / rate as f32;
			let decay = (-3.0 * time).exp();
			decay
				* (0.5 * (std::f32::consts::TAU * 220.0 * time).sin()
					+ 0.25 * (std::f32::consts::TAU * 330.0 * time).sin())
		})
		.collect();

	wav_of(&samples, rate)
}

/// The page posts one recorded note; the product fits it, puts it where the
/// instrument plays from and selects it.
#[test]
fn a_sample_is_fitted_into_the_instrument() {
	let wav = recorded_note(44_100);
	let map = modal_map();
	let slot = TableSlot::default();
	let request = ProtocolRequest {
		method: "POST",
		path: "/sample",
		query: Some("name=note_A4_127.wav"),
		body: &wav,
	};
	let body = String::from_utf8(
		crate::fitted::sample_route(&request, &map, &slot)
			.expect("the sample route")
			.body,
	)
	.expect("utf-8");

	assert!(body.contains("\"ok\":true"), "{body}");
	assert!(body.contains("\"note\":57"), "{body}");
	assert!(body.contains("\"modes\":"), "{body}");
	assert_eq!(
		value(&map, "instrument"),
		SetValue::Int(5),
		"the fitted table was not selected"
	);
	assert_eq!(value(&map, "derive"), SetValue::Bool(true));
	assert_eq!(value(&map, "transpose"), SetValue::Int(0));
	assert!(
		slot.current().is_some(),
		"the table never reached the instrument"
	);
}

/// A sample posted to the running plug-in is what the instrument plays, which
/// is the path a page takes: the route, the slot, the parameter map, the
/// processor, the note.
#[test]
fn a_posted_sample_plays_through_the_plugin() {
	let mut plugin = FittedWeb::new();
	Plugin::on_activate(&mut plugin, 48_000);

	let wav = recorded_note(44_100);
	let map = plugin.param_map();

	crate::fitted::sample_route(
		&ProtocolRequest {
			method: "POST",
			path: "/sample",
			query: Some("name=note_A4_127.wav"),
			body: &wav,
		},
		&map,
		&plugin.slot(),
	);

	// The fitted table holds one note; the other key is derived from it, which
	// is what a single-sample instrument has to do to play a scale.
	for note in [57, 69] {
		let mut context: Box<dyn ProcessContext> = Box::new(TestContext {
			events: vec![NoteEvent::NoteOn {
				channel: 0,
				note,
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

		assert!(peak > 1e-4, "note {note} was silent: {peak}");
	}
}

/// A save carries the fitted table, which the parameter map itself cannot hold.
#[test]
fn a_fitted_table_is_saved_with_the_parameters() {
	let mut plugin = FittedWeb::new();
	let slot = plugin.slot();
	let table = ImportedTable {
		name: "note.wav".to_string(),
		bytes: vec![1, 2, 3],
	};

	slot.set(table.clone());
	Plugin::sync_params(&mut plugin);

	let parameters = <FittedWeb as i_am_dsp::prelude::Parameters>::get_parameters(&plugin);
	let imported = parameters
		.into_iter()
		.find(|parameter| parameter.identifier == "imported")
		.expect("the imported parameter");
	let SetValue::Serialized(bytes) = imported.value.to_set_value() else {
		panic!("the table was not serialized");
	};
	let saved: Option<ImportedTable> =
		i_am_dsp::prelude::from_binary(bytes).expect("the table decodes");

	assert_eq!(saved, Some(table));
}

/// A library file name says which note it holds, in the octave a DAW writes.
#[test]
fn a_file_name_says_which_note_it_holds() {
	assert_eq!(
		i_am_dsp::prelude::note_of_name("The Gentleman_C4_127.wav"),
		Some(48)
	);
	assert_eq!(i_am_dsp::prelude::note_of_name("Bass_A#1_51.wav"), Some(22));
	assert_eq!(
		i_am_dsp::prelude::note_of_name("Celeste_C#8_127.wav"),
		Some(97)
	);
	assert_eq!(i_am_dsp::prelude::note_of_name("kick_1.wav"), None);
	assert_eq!(i_am_dsp::prelude::note_of_name("mypad.wav"), None);
}

/// The wrapper skips a block it knows is silent, so an instrument has to be honest about
/// having nothing left: nothing before the first note, a voice while one plays, and
/// nothing again once its release has run out.
#[test]
fn an_instrument_says_when_it_has_nothing_to_play() {
	let mut plugin = AdditiveWeb::new();
	assert!(
		i_am_dsp::Processor::is_idle(&plugin, true),
		"an instrument nothing was ever played to is silent"
	);

	let mut context: Box<dyn ProcessContext> = Box::new(TestContext::new());
	context.send_event(NoteEvent::NoteOn {
		channel: 0,
		note: 60,
		velocity: 0.8,
	});
	let mut frame = [0.0f32; 2];
	let mut played = false;

	for sample in 0..4_800 {
		i_am_dsp::Processor::process(&mut plugin, &mut frame, &[], &mut context);

		if sample == 0 {
			context.clear_events();
		}

		if !i_am_dsp::Processor::is_idle(&plugin, true) {
			played = true;
		}
	}

	assert!(played, "a held note is something to play");

	context.send_event(NoteEvent::NoteOff {
		channel: 0,
		note: 60,
		velocity: 0.0,
	});

	for sample in 0..480_000 {
		i_am_dsp::Processor::process(&mut plugin, &mut frame, &[], &mut context);

		if sample == 0 {
			context.clear_events();
		}

		if i_am_dsp::Processor::is_idle(&plugin, true) {
			return;
		}
	}

	panic!("the release never ran out");
}
