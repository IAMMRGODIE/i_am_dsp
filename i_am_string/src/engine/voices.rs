use super::*;

/// The thump of a hammer, the click of a plectrum.
///
/// Every real instrument makes a noise at the moment of excitation that is not
/// part of the string at all: the hammer and the mechanism, the finger leaving
/// the string, the plectrum snapping off it. Leaving it out is one of the
/// clearest reasons a physical model is heard as synthetic, because a string
/// excited by a perfectly smooth force is a thing that does not exist.
pub(super) struct Attack {
	pub(super) low: NoiseBand,
	pub(super) high: NoiseBand,
	/// The felt dragging across the string as the hammer lets go.
	pub(super) scrape: NoiseBand,
	/// The keybed and soundboard knock: five fixed low resonances, rung by
	/// the impact and gone in two thirds of a second.
	pub(super) knock: Vec<f32>,
	pub(super) knock_index: usize,
	pub(super) knock_gain: f32,
	pub(super) rng: Rng,
}

/// One band of the attack noise, with its own envelope and filter.
pub(super) struct NoiseBand {
	pub(super) envelope: f32,
	pub(super) decay: f32,
	pub(super) filter: crate::body::Biquad,
}

impl NoiseBand {
	pub(super) fn new(centre: f32) -> Self {
		Self {
			envelope: 0.0,
			decay: 0.0,
			filter: crate::body::Biquad::bandpass(centre, 0.8, 48_000.0),
		}
	}

	pub(super) fn trigger(
		&mut self,
		centre: f32,
		q: f32,
		level: f32,
		seconds: f32,
		sample_rate: f32,
	) {
		self.filter = crate::body::Biquad::bandpass(centre, q, sample_rate);
		self.envelope = self.envelope.max(level);
		self.decay = (-1.0 / (seconds * sample_rate).max(1.0)).exp();
	}

	/// Trigger a highpassed burst: the scrape has no bottom end.
	pub(super) fn trigger_highpass(
		&mut self,
		centre: f32,
		q: f32,
		level: f32,
		seconds: f32,
		sample_rate: f32,
	) {
		self.filter = crate::body::Biquad::highpass(centre, q, sample_rate);
		self.envelope = self.envelope.max(level);
		self.decay = (-1.0 / (seconds * sample_rate).max(1.0)).exp();
	}

	#[inline]
	pub(super) fn process(&mut self, noise: f32) -> f32 {
		if self.envelope < 1e-7 {
			return 0.0;
		}
		let value = self.filter.process(noise) * self.envelope;
		self.envelope *= self.decay;
		value
	}
}

impl Attack {
	pub(super) fn new() -> Self {
		Self {
			low: NoiseBand::new(1000.0),
			high: NoiseBand::new(3000.0),
			scrape: NoiseBand::new(4000.0),
			knock: Vec::new(),
			knock_index: 0,
			knock_gain: 0.0,
			rng: Rng::new(0x2F9A_1C43),
		}
	}

	/// Build the knock waveform for this sample rate.
	///
	/// The keybed and the board answer a blow with their own low resonances,
	/// not with a noise burst: five fixed modes between 89 and 277 Hz, each
	/// rising over three milliseconds and dying with the board. A piano's
	/// attack is a knock first and a note second, and a noise burst alone
	/// never sounds like wood.
	pub(super) fn prepare(&mut self, sample_rate: f32) {
		let frequencies = [89.0, 117.0, 163.0, 231.0, 277.0];
		let weights = [1.0, 0.65, 0.32, 0.20, 0.28];
		let length = (0.65 * sample_rate) as usize;
		let mut knock = vec![0.0; length];
		for (index, sample) in knock.iter_mut().enumerate() {
			let t = index as f32 / sample_rate;
			let rise = 1.0 - (-t / 0.003).exp();
			let end = ((0.65 - t) / 0.04).clamp(0.0, 1.0);
			for (frequency, weight) in frequencies.iter().zip(weights.iter()) {
				let decay = 8.0 + 0.01 * frequency;
				*sample += weight
					* rise * rise * (-decay * t).exp()
					* (std::f32::consts::TAU * frequency * t).sin()
					* end * end;
			}
		}
		let peak = knock
			.iter()
			.fold(0.0f32, |peak, value| peak.max(value.abs()));
		if peak > 0.0 {
			for value in &mut knock {
				*value /= peak;
			}
		}
		self.knock = knock;
		self.knock_index = self.knock.len();
	}

	/// Ring the knock once, as the key bottoming out does.
	pub(super) fn trigger_knock(&mut self, gain: f32) {
		if !self.knock.is_empty() && gain > 0.0 {
			self.knock_index = 0;
			self.knock_gain = gain;
		}
	}

	/// The low thump: the soundboard being hit, the finger leaving the string.
	pub(super) fn trigger(
		&mut self,
		centre: f32,
		q: f32,
		level: f32,
		seconds: f32,
		sample_rate: f32,
	) {
		self.low.trigger(centre, q, level, seconds, sample_rate);
	}

	/// The bright tick: the hammer's hard edge, the plectrum snapping off.
	///
	/// A piano's impact noise is not one band. The felt gives a thump of a
	/// few milliseconds around a few hundred hertz, and the hard parts give
	/// a tick of under two milliseconds that reaches several kilohertz;
	/// without the tick the note starts like a filter being excited rather
	/// than like a hammer being thrown at it.
	pub(super) fn trigger_tick(
		&mut self,
		centre: f32,
		q: f32,
		level: f32,
		seconds: f32,
		sample_rate: f32,
	) {
		self.high.trigger(centre, q, level, seconds, sample_rate);
	}

	/// The felt dragging across the string, which is a highpassed noise and
	/// not a thump.
	pub(super) fn trigger_scrape(
		&mut self,
		centre: f32,
		q: f32,
		level: f32,
		seconds: f32,
		sample_rate: f32,
	) {
		self.scrape
			.trigger_highpass(centre, q, level, seconds, sample_rate);
	}

	#[inline]
	pub(super) fn process(&mut self) -> f32 {
		let noise = self.rng.bipolar();
		let mut value =
			self.low.process(noise) + self.high.process(noise) + self.scrape.process(noise);
		if self.knock_index < self.knock.len() {
			value += self.knock[self.knock_index] * self.knock_gain;
			self.knock_index += 1;
		}
		value
	}
}

/// One sounding note.
pub(super) struct Voice {
	pub(super) note: u8,
	pub(super) frequency: f32,
	pub(super) strings: Vec<StringCore>,
	pub(super) damper: f32,
	pub(super) damper_target: f32,
	pub(super) gain: f32,
	pub(super) bowed: bool,
	pub(super) released: bool,
	/// The damper engagement the release settles at, 0..1.
	///
	/// This is what makes a stab stop harder than a held note: lifting a
	/// hand that was already resting on the string is not the same gesture
	/// as letting a long note go into the board.
	pub(super) release_engagement: f32,
	/// How fast the felt falls, per sample.
	pub(super) damper_rate: f32,
	pub(super) age: u64,
	pub(super) samples: u64,
	pub(super) peak: f32,
	/// Equal power stereo position for each string of the unison.
	pub(super) pans: Vec<(f32, f32)>,
	/// How long the note has been sounding, in seconds.
	pub(super) age_seconds: f32,
	/// Where the left hand is in its shake.
	pub(super) vibrato_phase: f32,
	/// The pitch ratio the last bend update settled on.
	pub(super) ratio: f32,
	/// How many samples the note fades in over; zero leaves it alone.
	pub(super) fade_in: u64,
	/// Per channel DC blocker state, so every note loses its own static force.
	pub(super) dc: [[f32; 2]; 2],
	/// Whether the blocker has been started on this voice's first sample.
	pub(super) dc_primed: bool,
}
