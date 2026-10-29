//! The ray-traced IR reverb: a generated room impulse response convolved with
//! an FFT convolver.
//!
//! The generator normalises the response, so its first sample can be a unit
//! Dirac; the generator fades that first few milliseconds. The convolver is
//! configured wet-only here, and the wrapper owns the dry/wet pair.

use i_am_dsp::prelude::{Effect, IrBasedReverb, ProcessContext, RayTracing};
use i_am_dsp_derive::Parameters;

/// The IR reverb with a room, a decay and a dry/wet pair.
#[derive(Parameters)]
pub struct IrReverbEffect {
	/// The room size in metres. The generated IR follows it.
	#[range(min = 5.0, max = 20.0)]
	#[default(10.0)]
	#[id(display = "Room")]
	#[layout(group = "IR Reverb", order = 0, widget = knob, hero = "tail")]
	pub size: f32,

	/// The T60 decay time.
	#[range(min = 200.0, max = 5_000.0)]
	#[unit(milliseconds)]
	#[default(2_200.0)]
	#[id(display = "Decay")]
	#[layout(group = "IR Reverb", order = 1, widget = knob)]
	pub decay_time: f32,

	/// The dry gain of the effect.
	#[range(min = 0.0, max = 1.0)]
	#[unit(percent)]
	#[default(0.0)]
	#[id(display = "Dry")]
	#[layout(group = "IR Reverb", order = 2, widget = knob)]
	pub dry_gain: f32,

	/// The wet gain of the effect.
	#[range(min = 0.0, max = 1.0)]
	#[unit(percent)]
	#[default(1.0)]
	#[id(display = "Wet")]
	#[layout(group = "IR Reverb", order = 3, widget = knob)]
	pub wet_gain: f32,

	#[skip]
	inner: IrBasedReverb<RayTracing, 2>,
	#[skip]
	sample_rate: usize,
	#[skip]
	last_size: f32,
	#[skip]
	last_decay: f32,
}

impl IrReverbEffect {
	/// A 10 m room with a 2.2 s decay and a fully wet convolver.
	pub fn new(sample_rate: usize) -> Self {
		let generator = Self::generator(10.0, 2_200.0, sample_rate);
		let inner = IrBasedReverb::new(generator, sample_rate);

		Self {
			size: 10.0,
			decay_time: 2_200.0,
			dry_gain: 0.0,
			wet_gain: 1.0,
			inner,
			sample_rate,
			last_size: 10.0,
			last_decay: 2_200.0,
		}
	}

	fn generator(size: f32, decay_time: f32, sample_rate: usize) -> RayTracing {
		// The ray tracer decays by exp(-decay * distance); this puts T60 at the
		// requested time using the speed of sound.
		let decay = (6.9 / ((decay_time.max(1.0) / 1_000.0) * 343.0)).clamp(1e-4, 1.0);

		RayTracing::rectangular_room(size, size * 0.7, size * 0.9, decay, 4.0, sample_rate)
	}

	fn regenerate(&mut self) {
		let generator = Self::generator(self.size, self.decay_time, self.sample_rate);
		let stereo = generator.generate_stereo_ir(Some(48_000));

		self.inner.convolver.replace_ir(core::array::from_fn(|channel| {
			let mut ir = stereo[channel % 2].clone();
			ir.truncate(48_000);
			ir
		}));
	}
}

impl Effect<2> for IrReverbEffect {
	fn delay(&self) -> usize {
		self.inner.delay()
	}

	fn process(
		&mut self,
		samples: &mut [f32; 2],
		other: &[&[f32; 2]],
		context: &mut Box<dyn ProcessContext>,
	) {
		if (self.size - self.last_size).abs() > 1e-4
			|| (self.decay_time - self.last_decay).abs() > 1e-4
		{
			self.regenerate();
			self.last_size = self.size;
			self.last_decay = self.decay_time;
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

	/// With the default fully wet setting, the impulse must not come straight
	/// back as the dry signal.
	#[test]
	fn the_ir_reverb_does_not_pass_the_dry() {
		let mut effect = IrReverbEffect::new(48_000);
		let mut context: Box<dyn ProcessContext> = Box::new(());
		let mut first = 0.0;

		for index in 0..512 {
			let input = if index == 0 { 1.0 } else { 0.0 };
			let mut samples = [input, input];
			effect.process(&mut samples, &[], &mut context);

			if index == 0 {
				first = samples[0];
			}

			assert!(samples.iter().all(|sample| sample.is_finite()), "{samples:?}");
		}

		assert!(first.abs() < 0.01, "dry leaked at sample 0: {first}");
	}
}
