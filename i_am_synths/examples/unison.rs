//! Where a unison's CPU goes: a heatmap over copy count and table path.
//!
//! ```text
//! cargo run --release -p i_am_synths --example unison
//! ```
//!
//! Rows are the unison count, columns are the table a voice reads: a sine (a
//! transcendental per copy), a triangle (a few adds), and the drawn frames the
//! Shapes product actually plays, plain, band limited, morphed and all three.
//! Each cell is the CPU time one second of audio costs, shaded by intensity.

use std::time::Instant;

use i_am_dsp::NoteEvent;
use i_am_dsp::prelude::{
	Adsr, EqualTemperament, Generator, Oscillator, ProcessContext, ProcessInfos, SineWave,
	TableOsc, TriangleWave, Tuning, WaveFramesOsc, WaveTableSmoother,
};

const RATE: usize = 48_000;
const SECONDS: usize = 2;

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

/// One voice, ready to render, of whatever oscillator the column names.
fn smoother(tables: Vec<Box<dyn i_am_dsp::prelude::WaveTable + Send + Sync>>) -> TableOsc<WaveTableSmoother> {
	TableOsc(WaveTableSmoother::new(tables, 0.0))
}

fn sunions(unisons: usize, rate: usize) -> Adsr<TableOsc<WaveTableSmoother>, EqualTemperament, 2> {
	let mut voice = Adsr::new(smoother(vec![Box::new(SineWave)]), EqualTemperament, rate);
	voice.unisons = unisons;
	voice
}

fn triangles(unisons: usize, rate: usize) -> Adsr<TableOsc<WaveTableSmoother>, EqualTemperament, 2> {
	let mut voice = Adsr::new(smoother(vec![Box::new(TriangleWave)]), EqualTemperament, rate);
	voice.unisons = unisons;
	voice
}

fn frames(
	unisons: usize,
	rate: usize,
	note: usize,
	morph: f32,
	fade: f32,
) -> Adsr<WaveFramesOsc, EqualTemperament, 2> {
	let mut voice = Adsr::new(WaveFramesOsc::new(), EqualTemperament, rate);
	voice.unisons = unisons;
	voice.oscillator.set_smooth_factor(morph);
	voice.oscillator.set_x_fade(fade);
	voice
		.oscillator
		.set_pitch(EqualTemperament.get_frequency(note as f32), rate);
	voice
}

/// The same voice with its morph moving, the way a hand sweeps it.
fn measure_moving(unisons: usize, rate: usize, note: usize) -> f64 {
	let mut voice = frames(unisons, rate, note, 0.0, 0.0);
	let mut context = Ctx::new(rate);
	context.events.push(NoteEvent::NoteOn {
		channel: 0,
		note,
		velocity: 0.8,
	});
	let mut context: Box<dyn ProcessContext> = Box::new(context);
	let frames = rate * SECONDS;
	let start = Instant::now();

	for frame in 0..frames {
		if frame % 64 == 0 {
			let sweep = (frame as f32 / frames as f32 * 8.0).fract();
			voice.oscillator.set_smooth_factor(sweep);
		}

		if frame == 1 {
			context.clear_events();
		}

		let _ = voice.generate(&mut context);
	}

	start.elapsed().as_secs_f64() * 1000.0 / SECONDS as f64
}

fn measure(mut voice: Box<dyn Generator<2>>, rate: usize, note: usize) -> f64 {
	let mut context = Ctx::new(rate);
	context.events.push(NoteEvent::NoteOn {
		channel: 0,
		note,
		velocity: 0.8,
	});
	let mut context: Box<dyn ProcessContext> = Box::new(context);
	let start = Instant::now();

	for frame in 0..rate * SECONDS {
		if frame == 1 {
			context.clear_events();
		}

		let _ = voice.generate(&mut context);
	}

	start.elapsed().as_secs_f64() * 1000.0 / SECONDS as f64
}

fn shade(value: f64, peak: f64) -> char {
	let levels = [' ', '░', '▒', '▓', '█'];
	let ratio = if peak > 0.0 { value / peak } else { 0.0 };
	let index = (ratio * (levels.len() - 1) as f64).round() as usize;

	levels[index.min(levels.len() - 1)]
}

fn main() {
	let columns = [
		"sine", "triangle", "frames", "band", "morph", "all", "live",
	];
	let unisons = [1usize, 2, 4, 8, 16, 32];
	let mut grid = vec![vec![0.0f64; columns.len()]; unisons.len()];

	for (row, &count) in unisons.iter().enumerate() {
		grid[row][0] = measure(Box::new(sunions(count, RATE)), RATE, 60);
		grid[row][1] = measure(Box::new(triangles(count, RATE)), RATE, 60);
		grid[row][2] = measure(Box::new(frames(count, RATE, 48, 0.0, 0.0)), RATE, 48);
		grid[row][3] = measure(Box::new(frames(count, RATE, 96, 0.0, 0.0)), RATE, 96);
		grid[row][4] = measure(Box::new(frames(count, RATE, 48, 0.5, 0.0)), RATE, 48);
		grid[row][5] = measure(Box::new(frames(count, RATE, 96, 0.5, 0.5)), RATE, 96);
		grid[row][6] = measure_moving(count, RATE, 48);
	}

	let peak = grid.iter().flatten().fold(0.0f64, |peak, value| peak.max(*value));

	println!("CPU milliseconds per second of audio (lower is better)");
	print!("{:>8}", "unison");
	for column in columns {
		print!(" {column:>9}");
	}
	println!();

	for (row, &count) in unisons.iter().enumerate() {
		print!("{count:>8}");

		for &value in &grid[row] {
			print!(" {}{:>8.1}", shade(value, peak), value);
		}

		println!();
	}

	println!();
	println!("Nanoseconds per copy per sample (what one voice costs)");
	print!("{:>8}", "unison");

	for column in columns {
		print!(" {column:>9}");
	}

	println!();

	for (row, &count) in unisons.iter().enumerate() {
		print!("{count:>8}");

		for &value in &grid[row] {
			let voice = value * 1e6 / (RATE as f64 * count as f64);
			print!(" {:>9.1}", voice);
		}

		println!();
	}

	// What it would cost to keep a morphed copy of the pages: the two frames'
	// band-limited levels mixed once per block, which is what a premix would pay
	// while a hand is moving the morph.
	let levels = 4096;
	let mut morphed = vec![0.0f32; levels];
	let mut other = (0..levels).map(|index| index as f32).collect::<Vec<f32>>();
	let blocks = RATE * SECONDS / 64;
	let start = Instant::now();

	for _ in 0..blocks {
		for (slot, value) in morphed.iter_mut().zip(other.iter_mut()) {
			*slot += (*value - *slot) * 0.37;
		}
	}

	let premix = start.elapsed().as_secs_f64() * 1000.0 / SECONDS as f64;

	println!();
	println!(
		"premix of {levels} level samples, rebuilt every 64-sample block: {premix:.1} ms per second of audio"
	);
	println!(
		"  -> frames at 32 copies, one read + premix: {:.1} ms/s (two reads: 81.3, two reads + morph: 139.4)",
		premix + 81.3
	);

	// The fixed part: what one copy costs when the note is alone says how much of
	// the total is per copy and how much is the note's own bookkeeping.
	println!();
	println!("Marginal cost of one more copy (ns/sample), from the row above");
	print!("{:>8}", "unison");

	for column in columns {
		print!(" {column:>9}");
	}

	println!();

	for row in 1..unisons.len() {
		let count = unisons[row];
		print!("{count:>8}");

		for column in 0..columns.len() {
			let added = (grid[row][column] - grid[row - 1][column]) * 1e6
				/ (RATE as f64 * (count as f64 - unisons[row - 1] as f64));
			print!(" {:>9.1}", added);
		}

		println!();
	}
}
