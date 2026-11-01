//! The Schroeder product surface around the shared DSP in `i_am_dsp`.

use i_am_dsp::prelude::{Effect, ProcessContext, Schroeder};
use i_am_dsp_derive::Parameters;

#[derive(Parameters)]
pub struct SchroederEffect {
	/// The delay scale.
	#[range(min = 0.5, max = 2.0)]
	#[default(1.0)]
	#[id(display = "Size")]
	#[layout(group = "Schroeder", order = 0, widget = knob)]
	pub size: f32,

	/// The T60 decay time.
	#[range(min = 200.0, max = 5_000.0)]
	#[unit(milliseconds)]
	#[default(2_200.0)]
	#[id(display = "Decay")]
	#[layout(group = "Schroeder", order = 1, widget = knob, hero = "tail")]
	pub decay_time: f32,

	/// The damping lowpass cutoff.
	#[range(min = 500.0, max = 20_000.0)]
	#[logarithmic]
	#[unit(hertz)]
	#[default(8_000.0)]
	#[id(display = "Damping")]
	#[layout(group = "Schroeder", order = 2, widget = knob)]
	pub damping: f32,

	/// The dry gain.
	#[range(min = 0.0, max = 1.0)]
	#[unit(percent)]
	#[default(0.2)]
	#[id(display = "Dry")]
	#[layout(group = "Schroeder", order = 3, widget = knob)]
	pub dry_gain: f32,

	/// The wet gain.
	#[range(min = 0.0, max = 1.0)]
	#[unit(percent)]
	#[default(0.8)]
	#[id(display = "Wet")]
	#[layout(group = "Schroeder", order = 4, widget = knob)]
	pub wet_gain: f32,

	#[skip]
	inner: Schroeder,
}

impl SchroederEffect {
	pub fn new(sample_rate: usize) -> Self {
		Self {
			size: 1.0,
			decay_time: 2_200.0,
			damping: 8_000.0,
			dry_gain: 0.2,
			wet_gain: 0.8,
			inner: Schroeder::new(sample_rate),
		}
	}
}

impl Effect<2> for SchroederEffect {
	fn delay(&self) -> usize {
		0
	}

	fn clear_tail(&mut self) {
		self.inner.clear_tail();
	}

	fn process(
		&mut self,
		samples: &mut [f32; 2],
		other: &[&[f32; 2]],
		context: &mut Box<dyn ProcessContext>,
	) {
		self.inner
			.set_parameters(self.size, self.decay_time, self.damping);

		let dry = *samples;
		let wet = self.inner.process(dry, other, context);

		for (index, sample) in samples.iter_mut().enumerate() {
			*sample = dry[index] * self.dry_gain + wet[index] * self.wet_gain;
		}
	}
}
