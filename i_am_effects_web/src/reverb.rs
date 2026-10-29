//! The web FDN reverb: a curated set of controls over the library's Reverb.
//!
//! The library network is wet-only and its pre-delay lives in a private
//! sub-parameter; this wrapper adds the pre-delay in milliseconds and a
//! dry/wet pair, and only resizes the network's lines when size actually moves.

use i_am_dsp::prelude::{Effect, Lowpass, ProcessContext, Reverb};
use i_am_dsp_derive::Parameters;

/// The FDN reverb with the controls a person turns.
#[derive(Parameters)]
pub struct ReverbEffect {
	/// The centre delay of the network; the early reflection spacing.
	#[range(min = 1.0, max = 50.0)]
	#[unit(milliseconds)]
	#[default(10.0)]
	#[id(display = "Size")]
	#[layout(group = "Reverb", order = 0, widget = knob)]
	pub size: f32,

	/// The T60 decay time.
	#[range(min = 100.0, max = 10_000.0)]
	#[logarithmic]
	#[unit(milliseconds)]
	#[default(2_200.0)]
	#[id(display = "Decay")]
	#[layout(group = "Reverb", order = 1, widget = knob, hero = "tail")]
	pub decay_time: f32,

	/// The cutoff of the damping in the feedback path.
	#[range(min = 200.0, max = 20_000.0)]
	#[logarithmic]
	#[unit(hertz)]
	#[default(8_000.0)]
	#[id(display = "Damping")]
	#[layout(group = "Reverb", order = 2, widget = knob)]
	pub damping: f32,

	/// How much the input diffusion smears the input.
	#[range(min = 0.0, max = 1.0)]
	#[unit(percent)]
	#[default(0.7)]
	#[id(display = "Diffusion")]
	#[layout(group = "Reverb", order = 3, widget = knob)]
	pub diffusion: f32,

	/// How much of the feedback the channels share.
	#[range(min = 0.0, max = 1.0)]
	#[unit(percent)]
	#[default(0.5)]
	#[id(display = "Spread")]
	#[layout(group = "Reverb", order = 4, widget = knob)]
	pub spread: f32,

	/// The gap in front of the reverb network.
	#[range(min = 0.0, max = 200.0)]
	#[unit(milliseconds)]
	#[default(20.0)]
	#[id(display = "Pre-delay")]
	#[layout(group = "Reverb", order = 5, widget = knob)]
	pub pre_delay: f32,

	/// The dry gain of the effect.
	#[range(min = 0.0, max = 1.0)]
	#[unit(percent)]
	#[default(0.35)]
	#[id(display = "Dry")]
	#[layout(group = "Reverb", order = 6, widget = knob)]
	pub dry_gain: f32,

	/// The wet gain of the effect.
	#[range(min = 0.0, max = 1.0)]
	#[unit(percent)]
	#[default(0.65)]
	#[id(display = "Wet")]
	#[layout(group = "Reverb", order = 7, widget = knob)]
	pub wet_gain: f32,

	#[skip]
	inner: Reverb<Lowpass<2>, 8, 2>,

	/// The last size and pre-delay sent to the network. Both setters resize
	/// buffers, so they are only called when the value really moved.
	#[skip]
	last_size: f32,
	#[skip]
	last_pre_delay: f32,
}

impl ReverbEffect {
	/// A musical starting point: 10 ms size, 2.2 s decay, 8 kHz damping, 70 %
	/// diffusion, 50 % spread, 20 ms pre-delay and a 35/65 dry/wet.
	pub fn new(sample_rate: usize) -> Self {
		let mut inner = Reverb::new(
			Lowpass::new(sample_rate, 20_000.0, 0.707),
			sample_rate,
			Default::default(),
			10,
			10.0,
		);
		inner.set_pre_delay(20.0);

		Self {
			size: 10.0,
			decay_time: 2_200.0,
			damping: 8_000.0,
			diffusion: 0.7,
			spread: 0.5,
			pre_delay: 20.0,
			dry_gain: 0.35,
			wet_gain: 0.65,
			inner,
			last_size: 10.0,
			last_pre_delay: 20.0,
		}
	}
}

impl Effect<2> for ReverbEffect {
	fn delay(&self) -> usize {
		self.inner.delay()
	}

	fn process(
		&mut self,
		samples: &mut [f32; 2],
		other: &[&[f32; 2]],
		context: &mut Box<dyn ProcessContext>,
	) {
		self.inner.decay_time = self.decay_time;
		self.inner.damping_cutoff = self.damping;
		self.inner.diffusion = self.diffusion;
		self.inner.spread = self.spread;
		self.inner.reverbed_gain = 1.0;

		if (self.size - self.last_size).abs() > 1e-4 {
			self.inner.set_center_delay(self.size);
			self.last_size = self.size;
		}

		if (self.pre_delay - self.last_pre_delay).abs() > 1e-4 {
			self.inner.set_pre_delay(self.pre_delay);
			self.last_pre_delay = self.pre_delay;
		}

		let dry = *samples;
		self.inner.process(samples, other, context);

		for (index, sample) in samples.iter_mut().enumerate() {
			*sample = dry[index] * self.dry_gain + *sample * self.wet_gain;
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// An impulse has to leave a finite tail, and the tail has to get quieter.
	#[test]
	fn the_reverb_tail_decays_and_stays_finite() {
		let mut effect = ReverbEffect::new(48_000);
		effect.decay_time = 500.0;
		effect.dry_gain = 0.0;
		effect.wet_gain = 1.0;
		effect.pre_delay = 0.0;

		let mut context: Box<dyn ProcessContext> = Box::new(());
		let mut early = 0.0;
		let mut late = 0.0;

		for index in 0..48_000 {
			let input = if index == 0 { 0.5 } else { 0.0 };
			let mut samples = [input, input];
			effect.process(&mut samples, &[], &mut context);

			assert!(samples.iter().all(|sample| sample.is_finite()), "{samples:?}");

			let energy = samples[0] * samples[0];
			if index < 4_800 {
				early += energy;
			} else if index >= 43_200 {
				late += energy;
			}
		}

		assert!(early > 0.0, "the network should produce a tail");
		assert!(late < early * 1e-3, "the tail did not decay: {early:e} -> {late:e}");
	}
}
