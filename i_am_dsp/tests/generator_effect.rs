//! The seam that lets a generator be hosted where an effect is expected.

use std::sync::atomic::Ordering;

use i_am_dsp::prelude::*;
use i_am_dsp::{NoteEvent, ProcessContext, ProcessInfos};

const FS: usize = 48_000;

struct TestContext {
	info: ProcessInfos,
	events: Vec<NoteEvent>,
}

impl TestContext {
	fn new() -> Self {
		let mut info = ProcessInfos::default();
		info.sample_rate = FS;
		info.playing = true;
		Self {
			info,
			events: Vec::new(),
		}
	}
}

impl ProcessContext for TestContext {
	fn infos(&self) -> &ProcessInfos {
		&self.info
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

fn note_on(note: usize) -> NoteEvent {
	NoteEvent::NoteOn {
		channel: 0,
		note,
		velocity: 1.0,
	}
}

#[test]
fn a_string_instrument_plays_through_the_adapter() {
	let mut effect = Paramed::new(GeneratorEffect::new(StringInstrument::new(FS)));
	let mut context: Box<dyn ProcessContext> = Box::new(TestContext::new());
	context.send_event(note_on(57));

	let mut audible = 0usize;
	let mut peak = 0.0f32;
	for _ in 0..FS {
		// Whatever the host had in the buffer has to be replaced, not mixed in.
		let mut samples = [17.0f32, -17.0];
		effect.process(&mut samples, &[], &mut context);
		peak = peak.max(samples[0].abs());
		if samples[0].abs() > 1e-4 {
			audible += 1;
		}
	}

	assert!(
		audible > FS / 10,
		"the instrument was silent through the adapter: {audible} audible samples"
	);
	assert!(peak < 4.0, "the instrument ran away: {peak}");
}

#[test]
fn a_host_parameter_reaches_the_generator() {
	let mut effect = Paramed::new(GeneratorEffect::new(StringInstrument::new(FS)));
	let map = effect.param_map();

	assert!(map.set("gain", 0.25f32, Ordering::Relaxed));

	let mut context: Box<dyn ProcessContext> = Box::new(TestContext::new());
	let mut samples = [0.0f32; 2];
	effect.process(&mut samples, &[], &mut context);

	let gain = effect.value.generator().gain;
	assert!((gain - 0.25).abs() < 1e-6, "the gain stayed at {gain}");
}
