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

/// Write a table in the format [parse_table] reads.
///
/// The notes are expected in the order the engine plays them and their modes
/// sorted by frequency; the velocity matrix is `rows` partial orders by
/// `cols` velocity layers, in dB relative to the hardest layer. A table
/// fitted from one sample has a single cell, which says every velocity plays the
/// same levels.
pub fn write_table(notes: &[Note], gain: &[f32], rows: u8, cols: u8) -> Vec<u8> {
	let mut out = Vec::new();
	out.extend_from_slice(b"MODE");
	out.push(3);
	out.push(0);
	out.push(notes.len() as u8);
	for note in notes {
		out.push(note.note);
		out.push(0);
		for channel in [&note.left, &note.right] {
			out.push(channel.len().min(u8::MAX as usize) as u8);
			for mode in channel.iter().take(u8::MAX as usize) {
				let order = mode.order.max(1.0) as u16;
				out.extend_from_slice(&order.to_le_bytes());
				for value in [
					mode.frequency,
					mode.prompt,
					mode.tail,
					mode.prompt_t60.max(0.02),
					mode.tail_t60.max(0.03),
					mode.prompt_phase,
					mode.tail_phase,
				] {
					out.extend_from_slice(&value.to_le_bytes());
				}
			}
		}
	}
	out.push(rows);
	out.push(cols);
	for value in gain.iter().take(rows as usize * cols as usize) {
		out.extend_from_slice(&value.to_le_bytes());
	}
	out
}

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
	/// The phase advance per sample at the note's own pitch, in radians. Kept so
	/// a moved pitch rebuilds the increment instead of approximating it.
	step: f32,
	/// The increment at the note's own pitch, restored when a glide lands, so
	/// the note ends exactly tuned rather than one rounding away from it.
	base: (f32, f32),
	/// What the decay is heading for, and the factor it moves by each sample
	/// while a glide ramps it.
	decay_target: f32,
	decay_step: f32,
	/// A level multiplier, ramping from where the same partial of the note the
	/// glide started on was to this one's. One when nothing is ramping.
	level: f32,
	level_target: f32,
	level_step: f32,
	/// Which partial this is and which of its two decays, so a note that takes
	/// the voice over can find the same partial of the note it moves to.
	order: f32,
	part: Part,
	/// The amplitude the note was built with, before the level multiplier. A
	/// takeover measures the note it moves to against this.
	built: f32,
}

impl Oscillator {
	fn new(
		amplitude: f32,
		phase: f32,
		omega: f32,
		sigma: f32,
		rate: f32,
		order: f32,
		part: Part,
	) -> Self {
		let step = omega / rate;
		let (sin, cos) = step.sin_cos();
		let decay = (-sigma / rate).exp();
		Self {
			q: amplitude * phase.sin(),
			v: amplitude * phase.cos(),
			cos,
			sin,
			decay,
			step,
			base: (sin, cos),
			decay_target: decay,
			decay_step: 1.0,
			level: 1.0,
			level_target: 1.0,
			level_step: 0.0,
			order,
			part,
			built: amplitude,
		}
	}

	/// Start this partial where the same partial of another note was, and ramp
	/// to this note's own decay and level over `samples`.
	///
	/// `level` is where the ramp starts, relative to this note's own amplitude:
	/// one says the two notes are equally loud there, zero says the other note
	/// has no such partial and it has to fade in.
	fn couple(&mut self, level: f32, t60: f32, rate: f32, samples: u32) {
		let decay = (-6.907_755 / t60.max(0.01) / rate).exp();
		let samples = samples.max(1) as f32;
		self.decay = decay;
		self.decay_step = (self.decay_target / decay).powf(1.0 / samples);
		self.level = level;
		self.level_step = (1.0 - level) / samples;
	}

	/// Rebuild the increment for the pitch multiplier `pitch`.
	fn retune(&mut self, pitch: f32) {
		if pitch == 1.0 {
			(self.sin, self.cos) = self.base;
			return;
		}
		// A partial pushed past Nyquist is held there rather than folded back
		// down: a stuck top partial is inaudible, an alias is not.
		let step = (self.step * pitch).min(std::f32::consts::PI);
		(self.sin, self.cos) = step.sin_cos();
	}

	/// One sample of the ramps a glide drives.
	fn ramp(&mut self) {
		self.decay *= self.decay_step;
		self.level += self.level_step;
	}

	/// Aim this partial at the level and the decay it is moving to.
	///
	/// A decay of `None` leaves the one it has: a slide that only moves the
	/// tuning must not quietly change how fast the note dies away.
	fn aim(&mut self, level: f32, decay: Option<f32>, samples: u32) {
		let samples = samples.max(1) as f32;
		self.level_target = level;
		self.level_step = (level - self.level) / samples;

		if let Some(decay) = decay {
			self.decay_target = decay;
			self.decay_step = (decay / self.decay).powf(1.0 / samples);
		}
	}

	/// Land exactly on what the glide was heading for.
	fn settle(&mut self, pitch: f32) {
		self.decay = self.decay_target;
		self.level = self.level_target;
		self.retune(pitch);
	}

	fn step(&mut self) -> f32 {
		let q = self.decay * (self.cos * self.q - self.sin * self.v);
		let v = self.decay * (self.sin * self.q + self.cos * self.v);
		self.q = q;
		self.v = v;
		q * self.level
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

/// How many samples apart the increments are rebuilt while a glide runs.
///
/// A pitch multiplies every partial's phase advance, so it cannot be applied to
/// the state without rebuilding: changing the increment keeps the phase, which
/// is what makes a slide smooth instead of a click. Sixteen samples is three
/// kilohertz, far above anything a slide needs, and it keeps the `sin_cos` off
/// fifteen samples in sixteen.
const RETUNE_INTERVAL: u32 = 16;

/// How a glide treats the partials while the pitch moves.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Glide {
	/// Only the partial frequencies follow the pitch; the decay and the level
	/// stay the ones the note was built with.
	///
	/// This is what a finger sliding along a string does: the speaking length
	/// changes and the damping does not, so every partial moves together and the
	/// ratios between them are untouched.
	#[default]
	Stretch,
	/// The decay and the level travel with the pitch, from the note the slide
	/// starts on to the note it ends on, so the timbre arrives with the tuning.
	///
	/// A table fitted from one recording holds one timbre, and there this says
	/// the same as [Glide::Stretch]. The difference shows up where the table
	/// holds the notes around the one being played: the note at the far end of
	/// the slide is brighter or longer than the one it left, and the partials
	/// have to travel with it.
	Couple,
}

/// How a note should slide in from the note before it.
#[derive(Clone, Copy, Debug)]
pub struct Slide {
	/// The note the slide starts on, numbered the way the table is.
	pub from: u8,
	/// How long the slide takes, in seconds. Zero or less starts on pitch.
	pub seconds: f32,
	/// What the slide does to the partials.
	pub mode: Glide,
}

/// Which of a mode's two decays an oscillator is.
#[derive(Clone, Copy)]
enum Part {
	/// The prompt decay: the sound of the excitation itself.
	Prompt,
	/// The tail: what the instrument gives back after it.
	Tail,
}

impl Part {
	/// This part's amplitude, decay time and phase.
	fn of(&self, mode: &Mode) -> (f32, f32, f32) {
		match self {
			Part::Prompt => (mode.prompt, mode.prompt_t60, mode.prompt_phase),
			Part::Tail => (mode.tail, mode.tail_t60, mode.tail_phase),
		}
	}
}

/// Where the partials of a slide start.
#[derive(Clone, Copy)]
struct Coupling<'a> {
	/// The note the slide starts on.
	source: &'a [Mode],
	/// How many samples the ramps take.
	samples: u32,
}

/// The partial of the note a slide starts on with the same order, if it has one.
fn partner_of<'a>(source: &'a [Mode], mode: &Mode) -> Option<&'a Mode> {
	source
		.iter()
		.find(|other| (other.order - mode.order).abs() < 0.5)
}

/// Fill one ear with the oscillators of a note.
fn build_ear(
	ear: &mut Vec<Oscillator>,
	modes: &[Mode],
	coupling: Option<Coupling<'_>>,
	instrument: &Instrument,
	velocity: f32,
	gain: f32,
	rate: f32,
) {
	for mode in modes {
		// A partial at or above Nyquist is not in the output: playing it would
		// fold it back down as an alias of itself.
		if mode.frequency <= 0.0 || mode.frequency >= rate * 0.5 {
			continue;
		}

		let level = gain * instrument.mode_gain(mode, velocity);
		let omega = std::f32::consts::TAU * mode.frequency;
		let partner = coupling.and_then(|coupling| partner_of(coupling.source, mode));

		for part in [Part::Prompt, Part::Tail] {
			let (amplitude, t60, phase) = part.of(mode);
			if amplitude <= 0.0 {
				continue;
			}

			let mut oscillator = Oscillator::new(
				level * amplitude,
				phase,
				omega,
				6.907_755 / t60.max(0.02),
				rate,
				mode.order,
				part,
			);

			// A partial the other note does not have starts where this one has
			// it rather than fading in: the voice starts from silence, so it
			// cannot click, and fading it in would cost the attack its brightness
			// for the whole slide.
			if let (Some(partner), Some(coupling)) = (partner, coupling) {
				let (start, start_t60, _) = part.of(partner);
				oscillator.couple(start / amplitude, start_t60, rate, coupling.samples);
			}

			ear.push(oscillator);
		}
	}
}

/// What a note taking a ringing voice over has to say.
///
/// A legato note does not start a sound of its own: the string that is already
/// vibrating is retuned to the new note, so the pitch (and, when coupling, the
/// timbre) has to travel from where it is to where the new note would be.
pub struct Takeover<'a> {
	/// The table the new note is played from.
	pub instrument: &'a Instrument,
	/// The note being played now, numbered the way the table is.
	pub note: u8,
	/// How hard it was pressed.
	pub velocity: f32,
	/// The instrument's own gain at this press.
	pub gain: f32,
	/// Whether a note the table does not hold may be derived from its neighbours.
	pub derive: bool,
	/// What the slide does to the partials.
	pub mode: Glide,
	/// How long the slide takes, in seconds.
	pub seconds: f32,
}

/// The same thing, with what the bank itself knows: the rate and the length of
/// the ramp in samples.
struct Aim<'a> {
	instrument: &'a Instrument,
	velocity: f32,
	gain: f32,
	mode: Glide,
	rate: f32,
	samples: u32,
}

/// Aim one ear's partials at a note, matched partial for partial by order.
fn aim_ear(ear: &mut [Oscillator], modes: &[Mode], aim: &Aim<'_>) {
	for oscillator in ear {
		let Some(target) = modes
			.iter()
			.find(|other| (other.order - oscillator.order).abs() < 0.5)
		else {
			continue;
		};

		let (amplitude, t60, _) = oscillator.part.of(target);
		let loudness = aim.gain * aim.instrument.mode_gain(target, aim.velocity) * amplitude;
		let level = if oscillator.built > 1e-12 {
			loudness / oscillator.built
		} else {
			1.0
		};
		// Only a coupled slide moves the decay: a stretch changes the speaking
		// length and leaves the damping where it was.
		let decay =
			(aim.mode == Glide::Couple).then(|| (-6.907_755 / t60.max(0.02) / aim.rate).exp());

		oscillator.aim(level, decay, aim.samples);
	}
}

/// One sounding note, with one oscillator bank per ear.
pub struct Voice {
	left: Vec<Oscillator>,
	right: Vec<Oscillator>,
	/// The sample rate the bank was built for; a glide is measured in samples.
	rate: f32,
	/// The note the bank was built from. A takeover measures the note it moves
	/// to against this, however many notes the voice has already passed through.
	base_note: u8,
	/// The frequency multiplier the partials are ringing at.
	pitch: f32,
	/// Where a glide is heading, what it multiplies the pitch by each sample,
	/// and how many samples it has left.
	pitch_target: f32,
	pitch_step: f32,
	glide_left: u32,
	/// Samples until the increments are rebuilt again.
	retune_left: u32,
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
		let mut voice = Self::empty(rate, note);
		if let Some((left, right)) = instrument.modes_for(note, derive) {
			build_ear(
				&mut voice.left,
				&left,
				None,
				instrument,
				velocity,
				gain,
				rate,
			);
			build_ear(
				&mut voice.right,
				&right,
				None,
				instrument,
				velocity,
				gain,
				rate,
			);
		}
		voice
	}

	/// Start a note that is already on its way from the pitch of another.
	///
	/// The note sounds at the pitch of `slide.from` and arrives at its own over
	/// `slide.seconds`. Its oscillators start from the phases the fit measured,
	/// so what the slide changes is the tuning, not the attack.
	pub fn start_sliding(
		instrument: &Instrument,
		note: u8,
		velocity: f32,
		gain: f32,
		rate: f32,
		derive: bool,
		slide: Slide,
	) -> Self {
		let ratio = note_frequency(slide.from) / note_frequency(note);
		// Nothing to slide from, or the pitch would already be there.
		if slide.seconds <= 0.0
			|| slide.from == note
			|| !ratio.is_finite()
			|| (ratio - 1.0).abs() < 1e-6
		{
			return Self::start_from(instrument, note, velocity, gain, rate, derive);
		}

		let Some((left, right)) = instrument.modes_for(note, derive) else {
			return Self::empty(rate, note);
		};

		let samples = (slide.seconds * rate).round().max(1.0) as u32;
		let source = match slide.mode {
			Glide::Stretch => None,
			Glide::Couple => instrument.modes_for(slide.from, derive),
		};

		let mut voice = Self::empty(rate, note);
		let coupling = source
			.as_ref()
			.map(|(source, _)| Coupling { source, samples });
		build_ear(
			&mut voice.left,
			&left,
			coupling,
			instrument,
			velocity,
			gain,
			rate,
		);
		let coupling = source
			.as_ref()
			.map(|(_, source)| Coupling { source, samples });
		build_ear(
			&mut voice.right,
			&right,
			coupling,
			instrument,
			velocity,
			gain,
			rate,
		);

		voice.pitch = ratio;
		for oscillator in voice.left.iter_mut().chain(voice.right.iter_mut()) {
			oscillator.retune(ratio);
		}
		voice.start_ramp(1.0, samples);
		voice
	}

	/// Start a note from the default table.
	pub fn start(note: u8, velocity: f32, gain: f32, rate: f32) -> Self {
		Self::start_from(instrument(), note, velocity, gain, rate, true)
	}

	/// The frequency multiplier the partials are ringing at.
	pub fn pitch_ratio(&self) -> f32 {
		self.pitch
	}

	/// Set the frequency multiplier at once, with no slide.
	pub fn set_pitch_ratio(&mut self, ratio: f32) {
		let ratio = ratio.clamp(1.0 / 64.0, 64.0);
		self.pitch = ratio;
		self.pitch_target = ratio;
		self.pitch_step = 1.0;
		self.glide_left = 0;
		for oscillator in self.left.iter_mut().chain(self.right.iter_mut()) {
			oscillator.retune(ratio);
		}
	}

	/// Slide the pitch to a frequency multiplier over `seconds`.
	pub fn glide_to(&mut self, ratio: f32, seconds: f32) {
		let samples = (seconds * self.rate).round().max(1.0) as u32;
		self.start_ramp(ratio.clamp(1.0 / 64.0, 64.0), samples);
	}

	/// Retarget a voice that is already ringing onto another note.
	///
	/// This is what a legato note does: the hand slides, the string keeps
	/// vibrating, and only the tuning travels -- plus the decay and the level
	/// when the slide is coupled. The partials keep the ratios the note the bank
	/// was built from gave them, which is what a real string does when its
	/// speaking length changes: it stretches, it does not become another string.
	pub fn retarget(&mut self, takeover: Takeover<'_>) {
		let samples = (takeover.seconds * self.rate).round().max(1.0) as u32;
		// Measured against the note the bank was built from, not against the one
		// it played last: after two takeovers the second would otherwise aim at
		// the wrong interval.
		let ratio = note_frequency(takeover.note) / note_frequency(self.base_note);

		if let Some((left, right)) = takeover
			.instrument
			.modes_for(takeover.note, takeover.derive)
		{
			let aim = Aim {
				instrument: takeover.instrument,
				velocity: takeover.velocity,
				gain: takeover.gain,
				mode: takeover.mode,
				rate: self.rate,
				samples,
			};
			aim_ear(&mut self.left, &left, &aim);
			aim_ear(&mut self.right, &right, &aim);
		}

		self.start_ramp(ratio, samples);
	}

	/// One frame, one ear each.
	pub fn next_frame(&mut self) -> [f32; 2] {
		if self.glide_left > 0 {
			self.advance();
		}

		let left = self.left.iter_mut().map(Oscillator::step).sum();
		let right = self.right.iter_mut().map(Oscillator::step).sum();
		[left, right]
	}

	/// One sample, mixed down to mono.
	pub fn next_sample(&mut self) -> f32 {
		let frame = self.next_frame();
		0.5 * (frame[0] + frame[1])
	}

	/// A silent bank, tuned to the note it is about to be given.
	fn empty(rate: f32, note: u8) -> Self {
		Self {
			left: Vec::new(),
			right: Vec::new(),
			rate,
			base_note: note,
			pitch: 1.0,
			pitch_target: 1.0,
			pitch_step: 1.0,
			glide_left: 0,
			retune_left: RETUNE_INTERVAL,
		}
	}

	/// Aim the pitch at `target`, arriving in `samples`.
	fn start_ramp(&mut self, target: f32, samples: u32) {
		self.pitch_target = target;
		self.pitch_step = (target / self.pitch.max(1e-6)).powf(1.0 / samples as f32);
		self.glide_left = samples;
		self.retune_left = RETUNE_INTERVAL;
	}

	/// One sample of a glide: the pitch, and the decay and level with it.
	fn advance(&mut self) {
		self.pitch *= self.pitch_step;
		self.glide_left -= 1;

		// The pitch is exponential, so it lands one rounding short of where it
		// was aimed; the last sample puts it exactly there and restores the
		// note's own increment, which is the one the fit measured.
		let landing = self.glide_left == 0;
		if landing {
			self.pitch = self.pitch_target;
		}

		self.retune_left = self.retune_left.saturating_sub(1);
		let retune = landing || self.retune_left == 0;
		if retune {
			self.retune_left = RETUNE_INTERVAL;
		}

		let pitch = self.pitch;
		for oscillator in self.left.iter_mut().chain(self.right.iter_mut()) {
			if landing {
				oscillator.settle(pitch);
			} else {
				oscillator.ramp();
				if retune {
					oscillator.retune(pitch);
				}
			}
		}
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

	/// One partial, so a window of the output is a single frequency.
	fn partial(frequency: f32, amplitude: f32, t60: f32) -> Mode {
		Mode {
			order: 1.0,
			frequency,
			prompt: amplitude,
			prompt_t60: t60,
			tail_t60: t60,
			..Default::default()
		}
	}

	/// A table of two notes an octave apart: the lower one soft and short, the
	/// upper one loud and long, so a slide can be heard to arrive.
	fn an_octave() -> Instrument {
		let note = |note: u8, frequency: f32, amplitude: f32, t60: f32| Note {
			note,
			left: vec![partial(frequency, amplitude, t60)],
			right: vec![partial(frequency, amplitude, t60)],
		};
		let bytes = write_table(
			&[note(57, 220.0, 0.5, 0.2), note(69, 440.0, 1.0, 5.0)],
			&[0.0],
			1,
			1,
		);
		parse_table(&bytes).expect("the table parses")
	}

	/// The frequency a window of samples is ringing at, from its crossings.
	///
	/// Counted between the first crossing and the last rather than against the
	/// window, so a window that does not hold a whole number of cycles does not
	/// round the answer off. A window over a slide measures the frequency
	/// averaged over the time it covers, which is what the pitch is doing there.
	fn frequency_of(samples: &[f32], rate: f32) -> f32 {
		let mut first = None;
		let mut last = None;
		let mut crossings = 0.0f32;

		for (index, pair) in samples.windows(2).enumerate() {
			if pair[0] <= 0.0 && pair[1] > 0.0 {
				// Where between the two samples the signal crossed, which is what
				// makes the count sub-sample.
				let position = index as f32 + -pair[0] / (pair[1] - pair[0]);
				first.get_or_insert(position);
				last = Some(position);
				crossings += 1.0;
			}
		}

		match (first, last) {
			(Some(first), Some(last)) if crossings > 1.0 => {
				(crossings - 1.0) * rate / (last - first)
			}
			_ => 0.0,
		}
	}

	/// Play a number of samples and keep them.
	fn render(voice: &mut Voice, samples: usize) -> Vec<f32> {
		(0..samples).map(|_| voice.next_sample()).collect()
	}

	#[test]
	fn a_glide_starts_on_the_other_note_and_arrives() {
		let rate = 48_000.0;
		let table = an_octave();
		let mut voice = Voice::start_sliding(
			&table,
			69,
			1.0,
			0.8,
			rate,
			true,
			Slide {
				from: 57,
				seconds: 0.5,
				mode: Glide::Stretch,
			},
		);
		// Half a second of slide: over the first hundred milliseconds the pitch
		// sweeps 220 Hz to 236 Hz, so a window there averages 236; a third of a
		// second later it has arrived.
		let early = frequency_of(&render(&mut voice, 4_800), rate);
		render(&mut voice, 24_000);
		let late = frequency_of(&render(&mut voice, 4_800), rate);
		println!("glide from 220 Hz: {early:.1} Hz over the first 100 ms, {late:.1} Hz after");
		assert!(
			(230.0..243.0).contains(&early),
			"the slide did not leave the note before it: {early:.1} Hz"
		);
		assert!(
			(late - 440.0).abs() < 2.0,
			"the slide did not arrive: {late:.1} Hz"
		);
		assert_eq!(voice.pitch_ratio(), 1.0, "the pitch was left off its note");
	}

	#[test]
	fn a_glide_does_not_click() {
		let rate = 48_000.0;
		let table = an_octave();
		for mode in [Glide::Stretch, Glide::Couple] {
			let mut voice = Voice::start_sliding(
				&table,
				69,
				1.0,
				0.8,
				rate,
				true,
				Slide {
					from: 57,
					seconds: 0.05,
					mode,
				},
			);
			let samples = render(&mut voice, 9_600);
			let peak = samples
				.iter()
				.fold(0.0f32, |peak, sample| peak.max(sample.abs()));
			let worst = samples
				.windows(3)
				.map(|window| (window[2] - 2.0 * window[1] + window[0]).abs())
				.fold(0.0f32, f32::max);
			println!("{mode:?}: worst second difference {worst:.5} of a peak {peak:.5}");
			// A sine turns by (2 pi f / rate)^2 of its amplitude per sample
			// squared, about three thousandths here; a phase jump would be of the
			// order of the amplitude itself.
			assert!(worst < 0.05 * peak, "{mode:?} clicks: {worst} of {peak}");
		}
	}

	#[test]
	fn coupling_takes_the_decay_of_the_note_it_came_from() {
		let rate = 48_000.0;
		let table = an_octave();
		let decay = |t60: f32| (-6.907_755 / t60 / rate).exp();
		let slide = |mode| Slide {
			from: 57,
			seconds: 0.05,
			mode,
		};

		let stretch = Voice::start_sliding(&table, 69, 1.0, 0.8, rate, true, slide(Glide::Stretch));
		assert!(
			(stretch.left[0].decay - decay(5.0)).abs() < 1e-9,
			"stretching moved the decay"
		);
		assert_eq!(stretch.left[0].level, 1.0, "stretching moved the level");

		let mut coupled =
			Voice::start_sliding(&table, 69, 1.0, 0.8, rate, true, slide(Glide::Couple));
		assert!(
			(coupled.left[0].decay - decay(0.2)).abs() < 1e-9,
			"coupling did not start on the other note's decay"
		);
		assert!(
			(coupled.left[0].level - 0.5).abs() < 1e-6,
			"coupling did not start on the other note's level: {}",
			coupled.left[0].level
		);

		// The ramps land on the note's own values when the pitch does.
		render(&mut coupled, 2_401);
		assert!(
			(coupled.left[0].decay - decay(5.0)).abs() < 1e-6,
			"the decay did not arrive: {}",
			coupled.left[0].decay
		);
		assert!(
			(coupled.left[0].level - 1.0).abs() < 1e-6,
			"the level did not arrive: {}",
			coupled.left[0].level
		);
	}

	#[test]
	fn a_ringing_note_can_be_bent() {
		let rate = 48_000.0;
		let table = an_octave();
		let mut voice = Voice::start_from(&table, 69, 1.0, 0.8, rate, false);
		voice.glide_to(0.5, 0.05);
		render(&mut voice, 12_000);
		let late = frequency_of(&render(&mut voice, 4_800), rate);
		println!("bent down an octave: {late:.1} Hz");
		assert!(
			(late - 220.0).abs() < 3.0,
			"the bend did not arrive: {late:.1} Hz"
		);
		assert!((voice.pitch_ratio() - 0.5).abs() < 1e-6);
	}

	#[test]
	fn a_slide_of_no_seconds_starts_on_pitch() {
		let rate = 48_000.0;
		let table = an_octave();
		let mut voice = Voice::start_sliding(
			&table,
			69,
			1.0,
			0.8,
			rate,
			true,
			Slide {
				from: 57,
				seconds: 0.0,
				mode: Glide::Stretch,
			},
		);
		assert_eq!(voice.pitch_ratio(), 1.0);
		let early = frequency_of(&render(&mut voice, 4_800), rate);
		assert!((early - 440.0).abs() < 2.0, "started at {early:.1} Hz");
	}

	/// The tables the crate ships hold up to forty-eight partials, so a slide
	/// here rebuilds nearly a hundred increments at a time. It must neither
	/// blow up nor drift into a value f32 cannot hold.
	#[test]
	fn a_slide_on_a_measured_table_stays_sane() {
		let rate = 48_000.0;
		let table = instrument();
		let mut plain = Voice::start_from(table, 60, 1.0, 0.8, rate, true);
		let plain_peak = render(&mut plain, 9_600)
			.iter()
			.fold(0.0f32, |peak, sample| peak.max(sample.abs()));

		for mode in [Glide::Stretch, Glide::Couple] {
			let mut voice = Voice::start_sliding(
				table,
				60,
				1.0,
				0.8,
				rate,
				true,
				Slide {
					from: 72,
					seconds: 0.3,
					mode,
				},
			);
			let samples = render(&mut voice, 19_200);
			let peak = samples
				.iter()
				.fold(0.0f32, |peak, sample| peak.max(sample.abs()));
			println!(
				"{mode:?} down an octave: peak {peak:.3} against {plain_peak:.3} plain, {} finite",
				samples.iter().filter(|sample| sample.is_finite()).count()
			);
			assert!(
				samples.iter().all(|sample| sample.is_finite()),
				"{mode:?} left the range of f32"
			);
			assert!(
				peak < 4.0 * plain_peak,
				"{mode:?} exploded: {peak} against {plain_peak}"
			);
		}
	}

	#[test]
	fn a_takeover_retunes_the_ringing_note() {
		let rate = 48_000.0;
		let table = an_octave();
		let mut voice = Voice::start_from(&table, 69, 1.0, 0.8, rate, true);
		render(&mut voice, 2_400);

		voice.retarget(Takeover {
			instrument: &table,
			note: 57,
			velocity: 1.0,
			gain: 0.8,
			derive: true,
			mode: Glide::Stretch,
			seconds: 0.05,
		});
		// The pitch is still the one it was sounding at when the hand landed.
		assert_eq!(voice.pitch_ratio(), 1.0, "the takeover jumped the pitch");

		let samples = render(&mut voice, 7_200);
		let late = frequency_of(&samples[2_400..], rate);
		let peak = samples
			.iter()
			.fold(0.0f32, |peak, sample| peak.max(sample.abs()));
		let worst = samples
			.windows(3)
			.map(|window| (window[2] - 2.0 * window[1] + window[0]).abs())
			.fold(0.0f32, f32::max);
		println!("takeover down an octave: {late:.1} Hz, worst step {worst:.5} of {peak:.5}");

		assert!(
			(late - 220.0).abs() < 2.0,
			"the takeover did not retune the note: {late:.1} Hz"
		);
		assert!((voice.pitch_ratio() - 0.5).abs() < 1e-6);
		assert!(
			worst < 0.05 * peak,
			"the takeover clicked: {worst} of {peak}"
		);
	}

	#[test]
	fn a_coupled_takeover_moves_the_decay_and_the_level() {
		let rate = 48_000.0;
		let table = an_octave();
		let decay = |t60: f32| (-6.907_755 / t60 / rate).exp();
		let takeover = |mode| Takeover {
			instrument: &table,
			note: 57,
			velocity: 1.0,
			gain: 0.8,
			derive: true,
			mode,
			seconds: 0.05,
		};

		let mut stretch = Voice::start_from(&table, 69, 1.0, 0.8, rate, true);
		stretch.retarget(takeover(Glide::Stretch));
		render(&mut stretch, 2_401);
		assert!(
			(stretch.left[0].decay - decay(5.0)).abs() < 1e-6,
			"a stretch moved the decay: {}",
			stretch.left[0].decay
		);

		let mut coupled = Voice::start_from(&table, 69, 1.0, 0.8, rate, true);
		coupled.retarget(takeover(Glide::Couple));
		render(&mut coupled, 2_401);
		assert!(
			(coupled.left[0].decay - decay(0.2)).abs() < 1e-6,
			"the decay did not follow: {}",
			coupled.left[0].decay
		);
		// The note below is half as loud in the table, so the level lands at a
		// half of what the voice was built with.
		assert!(
			(coupled.left[0].level - 0.5).abs() < 1e-5,
			"the level did not follow: {}",
			coupled.left[0].level
		);
	}

	#[test]
	fn a_partial_past_nyquist_is_not_played() {
		let bytes = write_table(
			&[Note {
				note: 69,
				left: vec![partial(6_000.0, 1.0, 1.0)],
				right: vec![partial(6_000.0, 1.0, 1.0)],
			}],
			&[0.0],
			1,
			1,
		);
		let table = parse_table(&bytes).expect("the table parses");
		let voice = Voice::start_from(&table, 69, 1.0, 0.8, 8_000.0, false);
		assert!(
			voice.left.is_empty() && voice.right.is_empty(),
			"a partial at 6 kHz was played at an 8 kHz rate"
		);
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
