//! An instrument fitted from recordings.
//!
//! A recording is decomposed into decaying sinusoids - modes - and the modes
//! are kept as data: where each partial sits, how loud it is, how it decays and
//! where it starts in phase. Playing a note is looking the note up and summing
//! its modes. The same shape of data fits any instrument whose notes are a set
//! of decaying partials, which is why this says "fitted instrument" and not
//! "piano"; the table comes from `gentleman-modal/pack.py`.
//!
//! Velocity changes brightness, not just level: the fundamental barely moves
//! while the high partials open by tens of decibels. That turned out to be one
//! profile over partial order times one brightness number per velocity, so the
//! table stores the profile and five weights rather than five copies of itself.

use std::sync::OnceLock;

/// One mode of a fitted note.
#[derive(Clone, Copy, Debug, Default)]
pub struct Mode {
	/// Harmonic order, used to look up the velocity profile.
	pub order: f32,
	/// Where the mode sits, in hertz.
	pub frequency: f32,
	/// Amplitude of the prompt decay.
	pub prompt: f32,
	/// Amplitude of the tail.
	pub tail: f32,
	/// T60 of the prompt decay, in seconds.
	pub prompt_t60: f32,
	/// T60 of the tail, in seconds.
	pub tail_t60: f32,
	/// Phase of the prompt decay, in radians.
	pub prompt_phase: f32,
	/// Phase of the tail, in radians.
	pub tail_phase: f32,
}

/// One note of the table.
#[derive(Clone, Debug, Default)]
pub struct Note {
	/// The MIDI note.
	pub note: u8,
	/// Its modes, sorted by frequency.
	pub modes: Vec<Mode>,
}

/// A whole fitted instrument.
#[derive(Debug)]
pub struct Instrument {
	notes: Vec<Note>,
	/// Velocity gain in dB, one row per partial order and one column per layer.
	gain: Vec<f32>,
	rows: usize,
	cols: usize,
}

impl Instrument {
	/// Look a note up.
	pub fn note(&self, note: u8) -> Option<&Note> {
		self.notes.iter().find(|entry| entry.note == note)
	}

	/// The gain for one mode at one velocity, as a linear multiplier.
	///
	/// Bilinear in partial order and velocity: the table is the measured level
	/// of every order at every layer, relative to the hardest layer.
	pub fn mode_gain(&self, mode: &Mode, velocity: f32) -> f32 {
		if self.rows == 0 || self.cols == 0 {
			return 1.0;
		}
		let row = (mode.order - 1.0).clamp(0.0, (self.rows - 1) as f32);
		let row0 = row.floor() as usize;
		let row1 = (row0 + 1).min(self.rows - 1);
		let row_fraction = row - row0 as f32;

		let last = (self.cols - 1) as f32;
		let position = if last > 0.0 {
			(velocity.clamp(0.0, 1.0) * 127.0 / (127.0 / last)).clamp(0.0, last)
		} else {
			0.0
		};
		let col0 = position.floor() as usize;
		let col1 = (col0 + 1).min(self.cols - 1);
		let col_fraction = position - col0 as f32;

		let sample = |r: usize, c: usize| self.gain[r * self.cols + c];
		let top = sample(row0, col0) * (1.0 - col_fraction) + sample(row0, col1) * col_fraction;
		let bottom = sample(row1, col0) * (1.0 - col_fraction) + sample(row1, col1) * col_fraction;
		let db = top * (1.0 - row_fraction) + bottom * row_fraction;
		10f32.powf(db / 20.0)
	}
}

/// The instrument that ships with this crate.
pub fn instrument() -> &'static Instrument {
	static TABLE: OnceLock<Instrument> = OnceLock::new();
	TABLE.get_or_init(parse)
}

fn parse() -> Instrument {
	let data: &[u8] = include_bytes!("../assets/gentleman.bin");
	let mut cursor = 0usize;
	let mut take = |count: usize| {
		let slice = &data[cursor..cursor + count];
		cursor += count;
		slice
	};
	assert_eq!(take(4), b"MODE");
	assert_eq!(take(1)[0], 1);
	let _reserved = take(1);
	let note_count = take(1)[0] as usize;
	let mut notes = Vec::with_capacity(note_count);
	for _ in 0..note_count {
		let note = take(1)[0];
		let count = take(1)[0] as usize;
		let mut modes = Vec::with_capacity(count);
		for _ in 0..count {
			let raw = take(30);
			let order = u16::from_le_bytes([raw[0], raw[1]]) as f32;
			let mut values = [0.0f32; 7];
			for (index, value) in values.iter_mut().enumerate() {
				let start = 2 + index * 4;
				*value = f32::from_le_bytes([
					raw[start],
					raw[start + 1],
					raw[start + 2],
					raw[start + 3],
				]);
			}
			modes.push(Mode {
				order,
				frequency: values[0],
				prompt: values[1],
				tail: values[2],
				prompt_t60: values[3],
				tail_t60: values[4],
				prompt_phase: values[5],
				tail_phase: values[6],
			});
		}
		notes.push(Note { note, modes });
	}
	let rows = take(1)[0] as usize;
	let cols = take(1)[0] as usize;
	let mut gain = Vec::with_capacity(rows * cols);
	for _ in 0..rows * cols {
		let raw = take(4);
		gain.push(f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]));
	}
	Instrument { notes, gain, rows, cols }
}
/// One decaying sinusoid.
struct Oscillator {
	q: f32,
	v: f32,
	cos: f32,
	sin: f32,
	decay: f32,
}

impl Oscillator {
	fn new(amplitude: f32, phase: f32, omega: f32, sigma: f32, rate: f32) -> Self {
		let step = omega / rate;
		Self {
			q: amplitude * phase.sin(),
			v: amplitude * phase.cos(),
			cos: step.cos(),
			sin: step.sin(),
			decay: (-sigma / rate).exp(),
		}
	}

	fn step(&mut self) -> f32 {
		let q = self.decay * (self.cos * self.q - self.sin * self.v);
		let v = self.decay * (self.sin * self.q + self.cos * self.v);
		self.q = q;
		self.v = v;
		q
	}
}

/// One sounding note.
pub struct Voice {
	oscillators: Vec<Oscillator>,
}

impl Voice {
	/// Start a note. `velocity` is 0..1 and `gain` the note's overall level.
	pub fn start(note: u8, velocity: f32, gain: f32, rate: f32) -> Self {
		let instrument = instrument();
		let mut oscillators = Vec::new();
		if let Some(entry) = instrument.note(note) {
			for mode in &entry.modes {
				let level = gain * instrument.mode_gain(mode, velocity);
				let omega = std::f32::consts::TAU * mode.frequency;
				if mode.prompt > 0.0 {
					oscillators.push(Oscillator::new(
						level * mode.prompt,
						mode.prompt_phase,
						omega,
						6.907_755 / mode.prompt_t60.max(0.02),
						rate,
					));
				}
				if mode.tail > 0.0 {
					oscillators.push(Oscillator::new(
						level * mode.tail,
						mode.tail_phase,
						omega,
						6.907_755 / mode.tail_t60.max(0.03),
						rate,
					));
				}
			}
		}
		Self { oscillators }
	}

	/// One sample of the note.
	pub fn next_sample(&mut self) -> f32 {
		self.oscillators.iter_mut().map(|osc| osc.step()).sum()
	}
}