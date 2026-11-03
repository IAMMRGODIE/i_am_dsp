//! Rough timing for the two synth voices, for a release build.
//!
//! ```text
//! cargo run --release -p i_am_synths --example bench
//! ```
//!
//! It renders five seconds of one note and prints how many milliseconds of CPU
//! that took per second of audio, which is the number a host's buffer has to
//! stay under.

use std::time::Instant;

use i_am_dsp::prelude::{
	Adsr, EqualTemperament, Generator, SawWave, SineWave, SquareWave, StringInstrument, TableOsc,
	TriangleWave, WaveTableSmoother,
};
use i_am_dsp::{NoteEvent, ProcessContext, ProcessInfos};

struct Ctx {
	events: Vec<NoteEvent>,
	infos: ProcessInfos,
}

impl Ctx {
	fn new(sample_rate: usize) -> Self {
		let mut infos = ProcessInfos::default();
		infos.sample_rate = sample_rate;
		infos.playing = true;
		Self {
			events: Vec::new(),
			infos,
		}
	}
}

impl ProcessContext for Ctx {
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

fn shapes(unisons: usize, notes: usize, rate: usize, seconds: usize) -> (f64, f32) {
	let mut voice: Adsr<TableOsc<WaveTableSmoother>, EqualTemperament, 2> = Adsr::new(
		TableOsc(WaveTableSmoother::new(
			vec![
				Box::new(SineWave),
				Box::new(TriangleWave),
				Box::new(SawWave),
				Box::new(SquareWave),
			],
			0.0,
		)),
		EqualTemperament,
		rate,
	);
	voice.unisons = unisons;

	let mut context = Ctx::new(rate);
	for index in 0..notes {
		context.events.push(NoteEvent::NoteOn {
			channel: 0,
			note: 60 + index * 4,
			velocity: 0.8,
		});
	}
	let mut context: Box<dyn ProcessContext> = Box::new(context);
	let mut peak = 0.0f32;
	let start = Instant::now();

	for frame in 0..rate * seconds {
		if frame == 1 {
			context.clear_events();
		}
		let sample = voice.generate(&mut context);
		peak = peak.max(sample[0].abs());
	}

	(start.elapsed().as_secs_f64(), peak)
}

fn strings(
	strings: i32,
	notes: usize,
	preset: i32,
	excitation: i32,
	core: i32,
	pitch_step: usize,
	rate: usize,
	seconds: usize,
) -> (f64, f32) {
	let mut instrument = StringInstrument::new(rate);
	// The preset's other controls matter: a violin string with a piano's
	// damping and brightness is a different, much longer ringing object.
	instrument.select_preset(preset);
	instrument.excitation = excitation;
	instrument.core = core;
	instrument.strings = strings;
	instrument.bow_force = 0.4;
	instrument.bow_speed = 0.35;

	let mut context: Box<dyn ProcessContext> = Box::new(Ctx::new(rate));
	// One frame with no events lets the instrument push the preset and the
	// excitation into the engine before the note starts; otherwise the note is
	// excited by whatever the engine was built with.
	instrument.generate(&mut context);
	for index in 0..notes {
		context.send_event(NoteEvent::NoteOn {
			channel: 0,
			note: 55 + index * pitch_step,
			velocity: 0.8,
		});
	}
	let mut peak = 0.0f32;
	let start = Instant::now();

	for _ in 0..rate * seconds {
		let sample = instrument.generate(&mut context);
		peak = peak.max(sample[0].abs());
	}

	(start.elapsed().as_secs_f64(), peak)
}

fn main() {
	let rate = 48_000;
	let seconds = 5;

	for unisons in [1usize, 8, 16, 32] {
		let (elapsed, peak) = shapes(unisons, 1, rate, seconds);
		print(
			"shapes unison",
			&format!("{unisons:2}, 1 note"),
			elapsed,
			seconds,
			peak,
		);
	}

	for notes in [1, 4] {
		let (elapsed, peak) = shapes(16, notes, rate, seconds);
		print(
			"shapes unison",
			&format!("16, {notes} notes"),
			elapsed,
			seconds,
			peak,
		);
	}

	for excitation in [0, 1, 2] {
		let (elapsed, peak) = strings(3, 1, 2, excitation, 0, 5, rate, seconds);
		print(
			"strings violin",
			&format!("excitation {excitation}, 3 strings, 1 note"),
			elapsed,
			seconds,
			peak,
		);
	}

	for pitch_step in [0, 5] {
		let (elapsed, peak) = strings(3, 4, 2, 0, 0, pitch_step, rate, seconds);
		print(
			"strings violin",
			&format!("pluck, 4 notes, step {pitch_step}"),
			elapsed,
			seconds,
			peak,
		);
	}

	for excitation in [0, 2] {
		let (elapsed, peak) = strings(3, 4, 2, excitation, 0, 5, rate, seconds);
		print(
			"strings violin",
			&format!("excitation {excitation}, 3 strings, 4 notes"),
			elapsed,
			seconds,
			peak,
		);
	}

	for core in [0, 1] {
		let (elapsed, peak) = strings(3, 4, 2, 2, core, 5, rate, seconds);
		print(
			"strings violin",
			&format!("core {core}, bow, 3 strings, 4 notes"),
			elapsed,
			seconds,
			peak,
		);
	}
}

fn print(what: &str, detail: &str, elapsed: f64, seconds: usize, peak: f32) {
	let per_second = elapsed / seconds as f64;
	println!(
		"{what} {detail}: {:6.1} ms of CPU per second of audio, {:4.1}% of one core, peak {peak:.3}",
		per_second * 1000.0,
		per_second * 100.0
	);
}
