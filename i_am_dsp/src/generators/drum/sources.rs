//! The pieces a drum voice is made of: decaying partials, noise bands, and a
//! tiny random source that keeps one hit from being the previous hit again.

use std::f32::consts::{PI, TAU};

/// A small xorshift. A voice needs random phase and noise, but it must not
/// reach for a global RNG: two hits in the same block have to differ, and a
/// voice must never block.
#[derive(Clone, Debug)]
pub struct Rng(u32);

impl Rng {
	/// A generator with a seed; zero is avoided because xorshift would stay there.
	pub fn new(seed: u32) -> Self {
		Self(seed | 1)
	}

	fn next_u32(&mut self) -> u32 {
		let mut x = self.0;
		x ^= x << 13;
		x ^= x >> 17;
		x ^= x << 5;
		self.0 = x;
		x
	}

	/// A number in `[0, 1)`.
	pub fn unit(&mut self) -> f32 {
		(self.next_u32() >> 8) as f32 / 16_777_216.0
	}

	/// A number in `[-1, 1)`.
	pub fn bipolar(&mut self) -> f32 {
		self.unit() * 2.0 - 1.0
	}
}

impl Default for Rng {
	fn default() -> Self {
		Self::new(0x9e37_79b9)
	}
}

/// One decaying partial, before the voice starts: the body of a drum is a bank
/// of these, and so is the click, only shorter.
#[derive(Clone, Copy, Debug)]
pub struct Partial {
	/// Frequency in Hz at the nominal pitch.
	pub frequency: f32,
	/// Linear amplitude at the start.
	pub amplitude: f32,
	/// Time to -60 dB, in seconds.
	pub t60: f32,
}

/// One partial that is running: its phase, and its decay counted per sample.
pub struct Running {
	frequency: f32,
	gain: f32,
	phase: f32,
	envelope: f32,
	step: f32,
}

impl Running {
	/// Start a partial, with a random phase so two hits never line up.
	pub fn new(partial: Partial, rate: f32, rng: &mut Rng) -> Self {
		let t60 = partial.t60.max(0.0005);

		Self {
			frequency: partial.frequency,
			gain: partial.amplitude,
			phase: rng.unit(),
			envelope: 1.0,
			step: (-6.907_755 / (t60 * rate)).exp(),
		}
	}

	/// The next sample, with the note's pitch curve applied.
	pub fn next(&mut self, pitch_ratio: f32, rate: f32) -> f32 {
		self.next_fm(pitch_ratio, rate, 0.0)
	}

	/// The next sample with a phase modulator added.
	///
	/// A bank of sines is a comb; phase modulation turns every sine into a
	/// cluster of sidebands, which is how a cymbal stops being a chord.
	pub fn next_fm(&mut self, pitch_ratio: f32, rate: f32, modulation: f32) -> f32 {
		let step = self.frequency * pitch_ratio / rate;

		// A kick can start ten octaves up, where a low shell partial would fold
		// back down the spectrum. Fade it out as it reaches Nyquist instead.
		let ceiling = if step > 0.45 {
			((0.5 - step) / 0.05).clamp(0.0, 1.0)
		} else {
			1.0
		};
		let sample = (self.phase * TAU + modulation).sin() * self.gain * self.envelope * ceiling;

		self.phase = (self.phase + step).fract();
		self.envelope *= self.step;

		sample
	}
}

/// One band of noise, before the voice starts.
#[derive(Clone, Copy, Debug)]
pub struct NoiseBand {
	/// Centre frequency in Hz.
	pub frequency: f32,
	/// Resonance; 1 is broad, higher is narrower.
	pub q: f32,
	/// Linear amplitude at the start.
	pub gain: f32,
	/// Time to -60 dB, in seconds.
	pub t60: f32,
}

/// A band of noise that is running.
pub struct Band {
	svf: Svf,
	gain: f32,
	envelope: f32,
	step: f32,
	g: f32,
	k: f32,
}

impl Band {
	/// Start a noise band at a fixed frequency.
	pub fn new(band: NoiseBand, rate: f32) -> Self {
		let frequency = band.frequency.clamp(20.0, rate * 0.45);
		let t60 = band.t60.max(0.0002);

		Self {
			svf: Svf::default(),
			gain: band.gain,
			envelope: 1.0,
			step: (-6.907_755 / (t60 * rate)).exp(),
			g: (PI * frequency / rate).tan(),
			k: 1.0 / band.q.max(0.05),
		}
	}

	/// The next sample of filtered noise.
	pub fn next(&mut self, rng: &mut Rng) -> f32 {
		let sample = self.svf.band(rng.bipolar(), self.g, self.k) * self.gain * self.envelope;

		self.envelope *= self.step;

		sample
	}
}

/// A two-pole state-variable filter, TPT form: stable at any cutoff, and its
/// band output is what a click's colour wants.
#[derive(Clone, Copy, Default)]
struct Svf {
	a: f32,
	b: f32,
}

impl Svf {
	fn band(&mut self, input: f32, g: f32, k: f32) -> f32 {
		let a1 = 1.0 / (1.0 + g * (g + k));
		let a2 = g * a1;
		let a3 = g * a2;
		let v3 = input - self.b;
		let v1 = a1 * self.a + a2 * v3;
		let v2 = self.b + a2 * self.a + a3 * v3;

		self.a = 2.0 * v1 - self.a;
		self.b = 2.0 * v2 - self.b;

		v1
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn a_partial_decays_to_nothing() {
		let mut rng = Rng::new(7);
		let mut partial = Running::new(
			Partial {
				frequency: 100.0,
				amplitude: 1.0,
				t60: 0.01,
			},
			48_000.0,
			&mut rng,
		);

		let early: f32 = (0..240)
			.map(|_| partial.next(1.0, 48_000.0).abs())
			.sum();
		for _ in 0..4_320 {
			partial.next(1.0, 48_000.0);
		}
		let late: f32 = (0..240)
			.map(|_| partial.next(1.0, 48_000.0).abs())
			.sum();

		assert!(early > late * 10.0, "{early} vs {late}");
	}

	#[test]
	fn a_noise_band_stays_bounded() {
		let mut rng = Rng::new(11);
		let mut band = Band::new(
			NoiseBand {
				frequency: 5_000.0,
				q: 2.0,
				gain: 1.0,
				t60: 0.05,
			},
			48_000.0,
		);

		let mut peak = 0.0f32;
		for _ in 0..4_800 {
			peak = peak.max(band.next(&mut rng).abs());
		}

		assert!(peak > 0.01, "the band was silent: {peak}");
		assert!(peak < 20.0, "the band ran away: {peak}");
	}

	#[test]
	fn the_rng_does_not_repeat() {
		let mut rng = Rng::new(3);

		assert_ne!(rng.bipolar(), rng.bipolar());
	}
}
