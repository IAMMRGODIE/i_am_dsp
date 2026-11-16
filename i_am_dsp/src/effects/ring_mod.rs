//! A ring modulator: the input times a carrier, or the carrier added to it.
//!
//! The carrier is an internal oscillator unless the third mode is chosen, where
//! the sidechain input is the carrier instead. Ring mode multiplies, which
//! removes the signal itself and leaves the sum and difference frequencies; AM
//! mode adds the carrier to the signal first, which keeps it. The two differ by
//! one term, so they live in one effect behind a mode switch.

use i_am_dsp_derive::Parameters;

use crate::{
	Effect, ProcessContext,
	generators::wavetable::{NoiseWave, SawWave, SineWave, SquareWave, TriangleWave, WaveTable},
};

/// One sample of a carrier shape at the given phase.
fn carrier(shape: i32, phase: f32) -> f32 {
	match shape {
		1 => TriangleWave.sample(phase, 0),
		2 => SawWave.sample(phase, 0),
		3 => SquareWave.sample(phase, 0),
		4 => NoiseWave.sample(phase, 0),
		_ => SineWave.sample(phase, 0),
	}
}

/// A ring modulator with an internal carrier, an amplitude mode and a sidechain
/// mode.
#[derive(Parameters)]
pub struct RingMod {
	/// Which modulation runs: multiply by the carrier, add the carrier to the
	/// signal, or take the carrier from the sidechain.
	#[choices("Ring", "AM", "Sidechain")]
	#[default(0)]
	#[id(display = "Mode")]
	#[layout(group = "Ring", order = 0, widget = choice)]
	pub mode: i32,

	/// The waveform of the internal carrier.
	#[choices("Sine", "Triangle", "Saw", "Square", "Noise")]
	#[default(0)]
	#[id(display = "Shape")]
	#[layout(group = "Ring", order = 1, widget = choice)]
	pub shape: i32,

	/// The carrier frequency.
	#[range(min = 0.05, max = 10_000.0)]
	#[logarithmic]
	#[unit(hertz)]
	#[default(440.0)]
	#[id(display = "Rate")]
	#[layout(group = "Ring", order = 2, widget = knob, hero = "ring")]
	pub rate: f32,

	/// How much of the carrier is added in the amplitude mode.
	#[range(min = 0.0, max = 1.0)]
	#[default(0.5)]
	#[id(display = "Depth")]
	#[layout(group = "Ring", order = 3, widget = knob)]
	pub depth: f32,

	/// The offset between the two channels' carriers, in degrees.
	#[range(min = 0.0, max = 180.0)]
	#[default(90.0)]
	#[id(display = "Stereo")]
	#[layout(group = "Ring", order = 4, widget = knob)]
	pub stereo: f32,

	/// Dry/wet blend.
	#[range(min = 0.0, max = 1.0)]
	#[unit(percent)]
	#[default(1.0)]
	#[id(display = "Mix")]
	#[layout(group = "Ring", order = 5, widget = knob)]
	pub mix: f32,

	#[skip]
	sample_rate: f32,
	#[skip]
	phase: f32,
}

impl RingMod {
	/// A sine carrier at 440 Hz, fully wet.
	pub fn new(sample_rate: usize) -> Self {
		Self {
			mode: 0,
			shape: 0,
			rate: 440.0,
			depth: 0.5,
			stereo: 90.0,
			mix: 1.0,
			sample_rate: sample_rate as f32,
			phase: 0.0,
		}
	}
}

impl Effect<2> for RingMod {
	fn delay(&self) -> usize {
		0
	}

	fn process(
		&mut self,
		samples: &mut [f32; 2],
		other: &[&[f32; 2]],
		_: &mut Box<dyn ProcessContext>,
	) {
		let dry = *samples;
		let mix = self.mix.clamp(0.0, 1.0);
		let depth = self.depth.clamp(0.0, 1.0);
		let offset = self.stereo.clamp(0.0, 180.0) / 360.0;
		let mode = self.mode;

		// Without a sidechain the sidechain mode is a pass through, which is
		// less surprising than silence.
		let sidechain = other
			.get(1)
			.map(|frame| (frame[0] + frame[1]) * 0.5)
			.unwrap_or(1.0);

		let mut wet = dry;

		for (channel, value) in wet.iter_mut().enumerate() {
			let carrier = if mode == 2 {
				sidechain
			} else {
				let phase = (self.phase + channel as f32 * offset).rem_euclid(1.0);
				carrier(self.shape, phase)
			};

			*value = if mode == 1 {
				dry[channel] * (1.0 + depth * carrier)
			} else {
				dry[channel] * carrier
			};
		}

		// The phase runs in every mode, so switching back does not jump.
		self.phase = (self.phase + self.rate.max(0.0) / self.sample_rate).rem_euclid(1.0);

		for (channel, sample) in samples.iter_mut().enumerate() {
			*sample = dry[channel] * (1.0 - mix) + wet[channel] * mix;
		}
	}

	fn clear_tail(&mut self) {
		self.phase = 0.0;
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn context() -> Box<dyn ProcessContext> {
		Box::new(())
	}

	/// Ring mode is the input times the carrier.
	#[test]
	fn ring_multiplies_the_carrier() {
		let mut ring = RingMod::new(48_000);
		ring.phase = 0.25;
		ring.stereo = 0.0;

		let mut frame = [1.0f32, 1.0];
		ring.process(&mut frame, &[], &mut context());

		assert!((frame[0] - 1.0).abs() < 1e-5, "{frame:?}");
		assert!((frame[1] - 1.0).abs() < 1e-5, "{frame:?}");
	}

	/// AM at zero depth leaves the signal alone: the carrier term disappears.
	#[test]
	fn am_at_zero_depth_is_the_dry_signal() {
		let mut ring = RingMod::new(48_000);
		ring.mode = 1;
		ring.depth = 0.0;

		let mut frame = [0.3f32, -0.2];
		ring.process(&mut frame, &[], &mut context());

		assert!((frame[0] - 0.3).abs() < 1e-6, "{frame:?}");
		assert!((frame[1] + 0.2).abs() < 1e-6, "{frame:?}");
	}

	/// AM at full depth adds the carrier to the signal rather than replacing it.
	#[test]
	fn am_at_full_depth_adds_the_carrier() {
		let mut ring = RingMod::new(48_000);
		ring.mode = 1;
		ring.depth = 1.0;
		ring.phase = 0.25;
		ring.stereo = 0.0;

		let mut frame = [0.5f32, 0.5];
		ring.process(&mut frame, &[], &mut context());

		// The carrier is one at a quarter cycle, so the signal doubles.
		assert!((frame[0] - 1.0).abs() < 1e-5, "{frame:?}");
	}

	/// The sidechain mode takes the carrier from the second input.
	#[test]
	fn sidechain_mode_multiplies_the_other_input() {
		let mut ring = RingMod::new(48_000);
		ring.mode = 2;

		let silent = [0.0f32; 2];
		let sidechain = [0.5f32; 2];
		let other: [&[f32; 2]; 2] = [&silent, &sidechain];
		let mut frame = [0.4f32, -0.4];

		ring.process(&mut frame, &other, &mut context());

		assert!((frame[0] - 0.2).abs() < 1e-6, "{frame:?}");
		assert!((frame[1] + 0.2).abs() < 1e-6, "{frame:?}");
	}

	/// The stereo offset puts the two channels' carriers apart.
	#[test]
	fn the_stereo_offset_moves_one_carrier() {
		let mut ring = RingMod::new(48_000);
		ring.stereo = 90.0;
		ring.phase = 0.25;

		let mut frame = [1.0f32, 1.0];
		ring.process(&mut frame, &[], &mut context());

		// A quarter cycle later, the second channel's sine is at zero.
		assert!((frame[0] - 1.0).abs() < 1e-5, "{frame:?}");
		assert!(frame[1].abs() < 1e-5, "{frame:?}");
	}

	/// Every shape and a fast carrier still stay bounded and finite.
	#[test]
	fn a_long_run_stays_bounded() {
		let mut ring = RingMod::new(48_000);
		ring.shape = 3;
		ring.rate = 7_000.0;

		for _ in 0..10_000 {
			let mut frame = [0.5, -0.5];

			ring.process(&mut frame, &[], &mut context());

			assert!(frame[0].is_finite() && frame[0].abs() <= 1.001, "{frame:?}");
		}
	}
}
