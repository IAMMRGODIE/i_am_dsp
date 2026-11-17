//! Sympathetic resonance: all the other strings on the instrument.
//!
//! Every string of a real instrument is driven through the bridge by every
//! other string, which is why a piano blooms when the sustain pedal is down and
//! why the undamped top octave answers a note played two octaves below it.
//!
//! The model here is a bank of high-Q resonators, one per note and per low
//! partial, fed by the mixed string output. Two things make it cheap: it is one
//! resonator per partial rather than a whole waveguide string, and it is
//! deliberately one-way. The resonators are driven by the strings but do not
//! load them, so the bank cannot feed anything back and cannot run away.
//!
//! The same bank is what implements the damper. Each note carries an engagement
//! between zero (felt lifted, string free to ring on) and one (felt resting on
//! the string), which is set from the key and the pedal together.

use wide::f32x4;

use crate::modal::LANES;

/// A two pole resonator with an adjustable decay rate.
#[derive(Clone, Copy, Debug)]
struct Resonator {
	cosine: f32,
	sine: f32,
	sample_rate: f32,
	/// The decay rate currently programmed in, in 1/s.
	decay: f32,
	a1: f32,
	a2: f32,
	gain: f32,
	y1: f32,
	y2: f32,
}

impl Resonator {
	fn new(frequency: f32, decay: f32, sample_rate: f32) -> Self {
		let omega = std::f32::consts::TAU * frequency / sample_rate;
		let mut resonator = Self {
			cosine: omega.cos(),
			sine: omega.sin(),
			sample_rate,
			decay: f32::NAN,
			a1: 0.0,
			a2: 0.0,
			gain: 0.0,
			y1: 0.0,
			y2: 0.0,
		};
		resonator.set_decay(decay);
		resonator
	}

	/// Reprogram the decay, cheaply doing nothing when it has not moved.
	fn set_decay(&mut self, decay: f32) {
		if (decay - self.decay).abs() <= 1e-4 * (1.0 + decay.abs()) {
			return;
		}
		self.decay = decay;
		let radius = (-decay / self.sample_rate).exp().min(0.999_99);
		self.a1 = -2.0 * radius * self.cosine;
		self.a2 = radius * radius;
		// Normalised so that the resonant peak has roughly unit gain, which
		// keeps the bank's level independent of how long the strings ring.
		self.gain = (1.0 - radius) * 2.0 * self.sine;
	}

	#[inline]
	fn process(&mut self, x: f32) -> f32 {
		let y = self.gain * x - self.a1 * self.y1 - self.a2 * self.y2;
		self.y2 = self.y1;
		self.y1 = y;
		y
	}

	fn clear(&mut self) {
		self.y1 = 0.0;
		self.y2 = 0.0;
	}
}

/// Below this a string is inaudible: -120 dB against a full scale note.
const SILENCE: f32 = 1e-6;

/// One sympathetic string: a few of its lowest partials.
#[derive(Clone, Debug)]
struct SympatheticString {
	note: u8,
	partials: Vec<Resonator>,
	open: Vec<f32>,
	damped: Vec<f32>,
	/// The smoothed damper engagement, in 0..1.
	engagement: f32,
	target: f32,
	/// How long the string has been inaudible, in samples.
	quiet: u32,
	/// Whether it is skipped entirely until something drives it again.
	silent: bool,
}

impl SympatheticString {
	#[inline]
	fn process(&mut self, x: f32) -> f32 {
		// A string that has been inaudible for ten milliseconds is skipped
		// until something drives it again. Its state is zeroed when it goes
		// quiet, so waking it up starts from rest rather than from an old tail.
		if self.silent {
			if x.abs() < SILENCE {
				return 0.0;
			}

			self.silent = false;
		}

		// The partials of one string do not talk to each other, so four of them
		// run in one vector: the recurrence is the same three multiply-adds.
		let mut lanes = f32x4::ZERO;
		let mut index = 0;

		while LANES && index + 4 <= self.partials.len() {
			let partials = &self.partials[index..index + 4];
			let a1 = f32x4::from([
				partials[0].a1,
				partials[1].a1,
				partials[2].a1,
				partials[3].a1,
			]);
			let a2 = f32x4::from([
				partials[0].a2,
				partials[1].a2,
				partials[2].a2,
				partials[3].a2,
			]);
			let gain = f32x4::from([
				partials[0].gain,
				partials[1].gain,
				partials[2].gain,
				partials[3].gain,
			]);
			let y1 = f32x4::from([
				partials[0].y1,
				partials[1].y1,
				partials[2].y1,
				partials[3].y1,
			]);
			let y2 = f32x4::from([
				partials[0].y2,
				partials[1].y2,
				partials[2].y2,
				partials[3].y2,
			]);
			let y = gain * f32x4::splat(x) - a1 * y1 - a2 * y2;

			lanes += y;

			let next = y.to_array();
			let previous = y1.to_array();

			for lane in 0..4 {
				let partial = &mut self.partials[index + lane];
				partial.y2 = previous[lane];
				partial.y1 = next[lane];
			}

			index += 4;
		}

		let mut sum = lanes.reduce_add();

		while index < self.partials.len() {
			sum += self.partials[index].process(x);
			index += 1;
		}

		if sum.abs() < SILENCE && x.abs() < SILENCE {
			self.quiet += 1;

			if self.quiet > 480 {
				for partial in &mut self.partials {
					partial.clear();
				}
				self.quiet = 0;
				self.silent = true;
			}
		} else {
			self.quiet = 0;
		}

		sum
	}
}

/// The complete set of undamped strings.
#[derive(Clone, Debug, Default)]
pub struct SympatheticBank {
	strings: Vec<SympatheticString>,
	level: f32,
	/// How fast the felt moves, in 1/s.
	damper_rate: f32,
}

impl SympatheticBank {
	/// An empty bank.
	pub fn new() -> Self {
		Self {
			strings: Vec::new(),
			level: 1.0,
			damper_rate: 24.0,
		}
	}

	/// Throw away every string.
	pub fn clear(&mut self) {
		self.strings.clear();
	}

	/// Install one string's partials.
	///
	/// Each partial is `(frequency, decay when open, decay when damped)`.
	pub fn set_string(&mut self, note: u8, partials: Vec<(f32, f32, f32)>, sample_rate: f32) {
		let mut resonators = Vec::with_capacity(partials.len());
		let mut open = Vec::with_capacity(partials.len());
		let mut damped = Vec::with_capacity(partials.len());
		for (frequency, open_decay, damped_decay) in partials {
			resonators.push(Resonator::new(frequency, open_decay, sample_rate));
			open.push(open_decay);
			damped.push(damped_decay);
		}
		let existing = self.strings.iter_mut().find(|entry| entry.note == note);
		match existing {
			Some(entry) => {
				entry.partials = resonators;
				entry.open = open;
				entry.damped = damped;
			}
			None => self.strings.push(SympatheticString {
				note,
				partials: resonators,
				open,
				damped,
				engagement: 0.0,
				target: 0.0,
				quiet: 0,
				silent: false,
			}),
		}
	}

	/// How many notes the bank holds.
	pub fn len(&self) -> usize {
		self.strings.len()
	}

	/// Whether the bank is empty.
	pub fn is_empty(&self) -> bool {
		self.strings.is_empty()
	}

	/// The output gain of the bank.
	pub fn set_level(&mut self, level: f32) {
		self.level = level;
	}

	/// How damped a note is: zero for a free string, one for felt resting on it.
	pub fn set_engagement(&mut self, note: u8, engagement: f32) {
		if let Some(entry) = self.strings.iter_mut().find(|entry| entry.note == note) {
			entry.target = engagement.clamp(0.0, 1.0);
		}
	}

	/// Set every damper from a function of its note.
	///
	/// This is how the engine applies the key and pedal state: one sweep over
	/// the bank instead of a lookup per note.
	pub fn set_engagement_by(&mut self, engagement: impl Fn(u8) -> f32) {
		for string in &mut self.strings {
			string.target = engagement(string.note).clamp(0.0, 1.0);
		}
	}

	/// Move every damper to its target and reprogram the resonators.
	pub fn update(&mut self, dt: f32) {
		let step = (dt * self.damper_rate).min(1.0);
		for string in &mut self.strings {
			string.engagement += (string.target - string.engagement) * step;
			for index in 0..string.partials.len() {
				let decay = string.open[index]
					+ string.engagement * (string.damped[index] - string.open[index]);
				string.partials[index].set_decay(decay);
			}
		}
	}

	/// Run one sample of the whole bank.
	#[inline]
	pub fn process(&mut self, input: f32) -> f32 {
		if self.strings.is_empty() || self.level == 0.0 {
			return 0.0;
		}
		let mut sum = 0.0;
		for string in &mut self.strings {
			sum += string.process(input);
		}
		sum * self.level
	}

	/// Silence every resonator without changing its tuning.
	pub fn reset(&mut self) {
		for string in &mut self.strings {
			for partial in &mut string.partials {
				partial.clear();
			}
			string.engagement = string.target;
		}
	}
}

impl Default for Resonator {
	fn default() -> Self {
		Self {
			cosine: 1.0,
			sine: 0.0,
			sample_rate: 48_000.0,
			decay: f32::NAN,
			a1: 0.0,
			a2: 0.0,
			gain: 0.0,
			y1: 0.0,
			y2: 0.0,
		}
	}
}
