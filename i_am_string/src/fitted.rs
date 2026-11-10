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

/// One note of the table, with one set of modes per channel.
#[derive(Clone, Debug, Default)]
pub struct Note {
	/// The MIDI note.
	pub note: u8,
	/// What the left ear hears.
	pub left: Vec<Mode>,
	/// What the right ear hears. The two are fitted from the two channels of
	/// the recording, so the stereo image is the recording's.
	pub right: Vec<Mode>,
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
	/// The notes the table holds.
	pub fn notes(&self) -> &[Note] {
		&self.notes
	}

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

	/// What a note plays: its own table, a derivation from the measured notes
	/// around it, or nothing.
	pub fn modes_for(&self, note: u8, derive: bool) -> Option<(Vec<Mode>, Vec<Mode>)> {
		if let Some(entry) = self.note(note) {
			return Some((entry.left.clone(), entry.right.clone()));
		}
		if !derive {
			return None;
		}
		let target = note_frequency(note);
		let mut low: Option<&Note> = None;
		let mut high: Option<&Note> = None;
		for entry in &self.notes {
			if entry.note <= note && low.is_none_or(|best| entry.note > best.note) {
				low = Some(entry);
			} else if entry.note > note && high.is_none_or(|best| entry.note < best.note) {
				high = Some(entry);
			}
		}
		let (low, high) = match (low, high) {
			(Some(low), Some(high)) => (low, high),
			(Some(single), None) | (None, Some(single)) => {
				let ratio = target / note_frequency(single.note);
				let scale = |modes: &[Mode]| {
					modes
						.iter()
						.map(|mode| Mode {
							frequency: mode.frequency * ratio,
							..*mode
						})
						.collect()
				};
				return Some((scale(&single.left), scale(&single.right)));
			}
			(None, None) => return None,
		};
		let low_f0 = note_frequency(low.note);
		let high_f0 = note_frequency(high.note);
		let span = (high_f0 / low_f0).ln().max(1e-6);
		let t = ((target / low_f0).ln() / span).clamp(0.0, 1.0);
		Some((
			couple(&low.left, &high.left, t),
			couple(&low.right, &high.right, t),
		))
	}
}

macro_rules! fitted_table {
	($name:ident, $path:literal) => {
		/// One of the fitted instruments.
		pub fn $name() -> &'static Instrument {
			static TABLE: OnceLock<Instrument> = OnceLock::new();
			TABLE.get_or_init(|| parse(include_bytes!($path)))
		}
	};
}

fitted_table!(instrument, "../assets/gentleman.bin");
fitted_table!(peel, "../assets/peel.bin");
fitted_table!(bass, "../assets/bass.bin");
fitted_table!(celeste, "../assets/celeste.bin");
fitted_table!(harp, "../assets/harp.bin");

/// Read a table from bytes, or nothing if they are not one.
pub fn parse_table(data: &[u8]) -> Option<Instrument> {
	try_parse(data)
}

fn try_parse(data: &[u8]) -> Option<Instrument> {
	if data.len() < 8 || &data[0..4] != b"MODE" || data[4] != 3 {
		return None;
	}
	let note_count = data[6] as usize;
	let mut cursor = 7usize;
	let mut notes = Vec::with_capacity(note_count);
	for _ in 0..note_count {
		if cursor + 2 > data.len() {
			return None;
		}
		let note = data[cursor];
		cursor += 2;
		let mut channels = [Vec::new(), Vec::new()];
		for channel in &mut channels {
			if cursor >= data.len() {
				return None;
			}
			let count = data[cursor] as usize;
			cursor += 1;
			if cursor + count * 30 > data.len() {
				return None;
			}
			let mut modes = Vec::with_capacity(count);
			for _ in 0..count {
				let raw = &data[cursor..cursor + 30];
				cursor += 30;
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
			*channel = modes;
		}
		notes.push(Note {
			note,
			left: channels[0].clone(),
			right: channels[1].clone(),
		});
	}
	if cursor + 2 > data.len() {
		return None;
	}
	let rows = data[cursor] as usize;
	cursor += 1;
	let cols = data[cursor] as usize;
	cursor += 1;
	if cursor + rows * cols * 4 > data.len() {
		return None;
	}
	let mut gain = Vec::with_capacity(rows * cols);
	for _ in 0..rows * cols {
		let raw = &data[cursor..cursor + 4];
		cursor += 4;
		gain.push(f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]));
	}
	Some(Instrument {
		notes,
		gain,
		rows,
		cols,
	})
}

/// Install a table the user brought, from bytes. Only one can be installed per
/// session: the slot is written once and read forever after.
pub fn install(bytes: &[u8]) -> bool {
	let Some(instrument) = try_parse(bytes) else {
		return false;
	};
	USER.set(instrument).is_ok()
}

/// The user's table, if one was installed.
pub fn installed() -> Option<&'static Instrument> {
	USER.get()
}

/// Try to install a table from the usual place.
pub fn install_from_config() -> bool {
	if USER.get().is_some() {
		return true;
	}
	for path in config_paths() {
		if let Ok(bytes) = std::fs::read(&path)
			&& install(&bytes)
		{
			return true;
		}
	}
	false
}

fn config_paths() -> Vec<std::path::PathBuf> {
	let mut out = Vec::new();
	if let Ok(path) = std::env::var("IAM_MODAL_TABLE") {
		out.push(std::path::PathBuf::from(path));
	}
	if let Ok(appdata) = std::env::var("APPDATA") {
		out.push(
			std::path::PathBuf::from(appdata)
				.join("i_am_dsp")
				.join("modal.bin"),
		);
	}
	if let Ok(home) = std::env::var("HOME") {
		out.push(
			std::path::PathBuf::from(home)
				.join(".config")
				.join("i_am_dsp")
				.join("modal.bin"),
		);
	}
	out
}

/// The table the user installed, written once per session.
static USER: OnceLock<Instrument> = OnceLock::new();

fn parse(data: &'static [u8]) -> Instrument {
	let mut cursor = 0usize;
	let mut take = |count: usize| {
		let slice = &data[cursor..cursor + count];
		cursor += count;
		slice
	};
	assert_eq!(take(4), b"MODE");
	assert_eq!(take(1)[0], 3);
	let _reserved = take(1);
	let note_count = take(1)[0] as usize;
	let mut notes = Vec::with_capacity(note_count);
	for _ in 0..note_count {
		let note = take(1)[0];
		let _reserved = take(1);
		let mut channels = [Vec::new(), Vec::new()];
		for channel in &mut channels {
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
			*channel = modes;
		}
		notes.push(Note {
			note,
			left: channels[0].clone(),
			right: channels[1].clone(),
		});
	}
	let rows = take(1)[0] as usize;
	let cols = take(1)[0] as usize;
	let mut gain = Vec::with_capacity(rows * cols);
	for _ in 0..rows * cols {
		let raw = take(4);
		gain.push(f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]));
	}
	Instrument {
		notes,
		gain,
		rows,
		cols,
	}
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

fn note_frequency(note: u8) -> f32 {
	440.0 * 2f32.powf((note as f32 - 69.0) / 12.0)
}

/// Interpolate two measured channels partial by partial, in log pitch.
fn couple(low: &[Mode], high: &[Mode], t: f32) -> Vec<Mode> {
	let lerp = |a: f32, b: f32| ((a.max(1e-9)).ln() * (1.0 - t) + (b.max(1e-9)).ln() * t).exp();
	let mut out = Vec::with_capacity(low.len().max(high.len()));
	let mut used = vec![false; high.len()];
	for left in low {
		let mut found = None;
		for (index, right) in high.iter().enumerate() {
			if used[index] {
				continue;
			}
			if (right.order - left.order).abs() < 0.5 {
				found = Some(index);
				break;
			}
		}
		match found {
			Some(index) => {
				used[index] = true;
				let right = &high[index];
				out.push(Mode {
					order: left.order,
					frequency: lerp(left.frequency, right.frequency),
					prompt: lerp(left.prompt, right.prompt),
					tail: lerp(left.tail, right.tail),
					prompt_t60: lerp(left.prompt_t60, right.prompt_t60),
					tail_t60: lerp(left.tail_t60, right.tail_t60),
					prompt_phase: left.prompt_phase,
					tail_phase: left.tail_phase,
				});
			}
			None => out.push(*left),
		}
	}
	for (index, right) in high.iter().enumerate() {
		if !used[index] {
			out.push(*right);
		}
	}
	out
}

/// One sounding note, with one oscillator bank per ear.
pub struct Voice {
	left: Vec<Oscillator>,
	right: Vec<Oscillator>,
}

impl Voice {
	/// Start a note from one of the fitted tables.
	pub fn start_from(
		instrument: &Instrument,
		note: u8,
		velocity: f32,
		gain: f32,
		rate: f32,
		derive: bool,
	) -> Self {
		let mut voice = Self {
			left: Vec::new(),
			right: Vec::new(),
		};
		if let Some((left, right)) = instrument.modes_for(note, derive) {
			for (modes, ear) in [(left, &mut voice.left), (right, &mut voice.right)] {
				for mode in modes {
					let level = gain * instrument.mode_gain(&mode, velocity);
					let omega = std::f32::consts::TAU * mode.frequency;
					if mode.prompt > 0.0 {
						ear.push(Oscillator::new(
							level * mode.prompt,
							mode.prompt_phase,
							omega,
							6.907_755 / mode.prompt_t60.max(0.02),
							rate,
						));
					}
					if mode.tail > 0.0 {
						ear.push(Oscillator::new(
							level * mode.tail,
							mode.tail_phase,
							omega,
							6.907_755 / mode.tail_t60.max(0.03),
							rate,
						));
					}
				}
			}
		}
		voice
	}

	/// Start a note from the default table.
	pub fn start(note: u8, velocity: f32, gain: f32, rate: f32) -> Self {
		Self::start_from(instrument(), note, velocity, gain, rate, true)
	}

	/// One frame, one ear each.
	pub fn next_frame(&mut self) -> [f32; 2] {
		let left = self.left.iter_mut().map(|osc| osc.step()).sum();
		let right = self.right.iter_mut().map(|osc| osc.step()).sum();
		[left, right]
	}

	/// One sample, mixed down to mono.
	pub fn next_sample(&mut self) -> f32 {
		let frame = self.next_frame();
		0.5 * (frame[0] + frame[1])
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn a_measured_note_makes_a_sound() {
		let instrument = instrument();
		let note = instrument.notes()[0].note;
		let mut voice = Voice::start_from(instrument, note, 1.0, 0.8, 48_000.0, false);
		let mut peak = 0.0f32;
		for _ in 0..4_800 {
			peak = peak.max(voice.next_sample().abs());
		}
		assert!(peak > 1e-4, "silent: {peak}");
	}

	#[test]
	fn the_second_table_plays_too() {
		let table = peel();
		let note = table.notes()[0].note;
		let mut voice = Voice::start_from(table, note, 1.0, 0.8, 48_000.0, false);
		let mut peak = 0.0f32;
		for _ in 0..4_800 {
			peak = peak.max(voice.next_sample().abs());
		}
		assert!(peak > 1e-4, "silent: {peak}");
	}

	#[test]
	fn every_table_plays() {
		for table in [instrument(), peel(), bass(), celeste(), harp()] {
			let note = table.notes()[0].note;
			let mut voice = Voice::start_from(table, note, 1.0, 0.8, 48_000.0, false);
			let mut peak = 0.0f32;
			for _ in 0..4_800 {
				peak = peak.max(voice.next_sample().abs());
			}
			assert!(peak > 1e-4, "a table was silent: {peak}");
		}
	}

	#[test]
	fn a_table_can_be_installed_from_bytes() {
		assert!(!install(b"not a table"), "junk was accepted");
		let path = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/peel.bin");
		let bytes = std::fs::read(path).expect("read the table");
		assert!(install(&bytes), "a real table was rejected");
		let custom = installed().expect("the table is installed");
		assert!(!custom.notes().is_empty());
		assert!(!install(&bytes), "the slot was written twice");
	}

	#[test]
	fn an_unmeasured_note_is_derived_or_silent() {
		let instrument = instrument();
		let held: Vec<u8> = instrument.notes().iter().map(|note| note.note).collect();
		let missing = (0u8..=127u8)
			.find(|note| !held.contains(note))
			.expect("a note the table does not hold");
		let quiet = Voice::start_from(instrument, missing, 1.0, 0.8, 48_000.0, false);
		let mut quiet = quiet;
		let mut peak = 0.0f32;
		for _ in 0..4_800 {
			peak = peak.max(quiet.next_sample().abs());
		}
		assert!(peak < 1e-6, "silent note was not silent: {peak}");

		let mut derived = Voice::start_from(instrument, missing, 1.0, 0.8, 48_000.0, true);
		let mut peak = 0.0f32;
		for _ in 0..4_800 {
			peak = peak.max(derived.next_sample().abs());
		}
		assert!(peak > 1e-4, "derived note was silent: {peak}");
	}

	#[test]
	fn the_two_ears_are_not_the_same() {
		let instrument = instrument();
		let note = instrument.notes()[0].note;
		let mut voice = Voice::start_from(instrument, note, 1.0, 0.8, 48_000.0, false);
		let mut difference = 0.0f32;
		let mut total = 0.0f32;
		for _ in 0..4_800 {
			let [left, right] = voice.next_frame();
			difference += (left - right).abs();
			total += left.abs() + right.abs();
		}
		assert!(difference > 0.002 * total, "the channels are identical");
	}
}
