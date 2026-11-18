//! How much the contact solver's tolerance is worth: render a bowed note at
//! several tolerances and compare the level and the coarse spectrum, not the
//! samples. A bowed string is a stick-slip oscillator: two runs diverge sample
//! by sample however tight the solver is, and what a listener hears is the
//! level and the spectrum.
//!
//! ```text
//! cargo run --release -p i_am_synths --example tolerance
//! ```

use i_am_dsp::NoteEvent;
use i_am_dsp::prelude::{
	Generator, ProcessContext, ProcessInfos, StringInstrument, set_solver_tolerance,
};

const RATE: usize = 48_000;
const SECONDS: usize = 2;
const WINDOW: usize = 8192;
const BINS: usize = 64;

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

fn render(tolerance: f32) -> Vec<f32> {
	set_solver_tolerance(tolerance);

	let mut instrument = StringInstrument::new(RATE);
	instrument.select_preset(2);
	instrument.excitation = 2;
	instrument.strings = 3;
	instrument.bow_force = 0.4;
	instrument.bow_speed = 0.35;

	let mut context: Box<dyn ProcessContext> = Box::new(Ctx::new(RATE));
	instrument.generate(&mut context);
	context.send_event(NoteEvent::NoteOn {
		channel: 0,
		note: 57,
		velocity: 0.8,
	});

	let mut output = Vec::with_capacity(RATE * SECONDS);

	for _ in 0..RATE * SECONDS {
		output.push(instrument.generate(&mut context)[0]);
	}

	output
}

fn level(samples: &[f32]) -> f32 {
	(samples.iter().map(|value| value * value).sum::<f32>() / samples.len() as f32).sqrt()
}

/// The first bins, averaged over the note: a single window of a stick-slip
/// string shows whichever slips landed in it, while the average over the note is
/// the timbre a listener hears.
fn spectrum(samples: &[f32]) -> Vec<f32> {
	let windows = 16;
	let stride = (samples.len() - WINDOW) / windows;
	let mut magnitudes = vec![0.0f32; BINS];

	for window_index in 0..windows {
		let start = window_index * stride;

		for (bin, magnitude) in magnitudes.iter_mut().enumerate() {
			let (mut real, mut imaginary) = (0.0f32, 0.0f32);

			for (index, sample) in samples[start..start + WINDOW].iter().enumerate() {
				let window =
					0.5 - 0.5 * (std::f32::consts::TAU * index as f32 / WINDOW as f32).cos();
				let angle =
					std::f32::consts::TAU * ((bin + 1) * index) as f32 / WINDOW as f32;
				let value = sample * window;

				real += value * angle.cos();
				imaginary -= value * angle.sin();
			}

			*magnitude += (real * real + imaginary * imaginary).sqrt() / windows as f32;
		}
	}

	magnitudes
}

fn main() {
	// The first render is the reference; the second is the same tolerance again,
	// which is the model's own run-to-run floor: a bowed string diverges if
	// anything in it differs at all, so a difference at or below this row says
	// nothing about audibility.
	let ticks = [1e-7f32, 1e-7, 1e-9, 1e-5];
	let renders: Vec<Vec<f32>> = ticks.iter().map(|tolerance| render(*tolerance)).collect();
	let levels: Vec<f32> = renders.iter().map(|samples| level(samples)).collect();
	let spectra: Vec<Vec<f32>> = renders.iter().map(|samples| spectrum(samples)).collect();

	println!("bowed violin, {SECONDS} s, one note, level and first {BINS} bins");
	println!("  tolerance   level dB      mean bin dB   worst bin dB");

	for (index, tolerance) in ticks.iter().enumerate() {
		let level_db = 20.0 * (levels[index] / levels[0].max(1e-12)).log10();
		let mut mean = 0.0f32;
		let mut worst = 0.0f32;

		for (a, b) in spectra[0].iter().zip(spectra[index].iter()) {
			let difference = 20.0 * ((a.max(1e-9)) / b.max(1e-9)).log10().abs();

			mean += difference / BINS as f32;
			worst = worst.max(difference);
		}

		println!(
			"  {tolerance:8.0e}   {level_db:6.2}        {mean:6.2}        {worst:6.2}"
		);
	}
}
