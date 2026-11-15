//! The web product's disperser: the library's allpass chain with an amount of
//! its own.
//!
//! The library's Disperser leaves the number of allpass sections to the caller,
//! because an effect that sweeps it (the crusher) owns that decision. This
//! product wants it as a knob, so it keeps the section count in step with the
//! amount and only redesigns the prototype when the cutoff or bandwidth moved.

use i_am_dsp::prelude::{Disperser, Effect, ProcessContext};
use i_am_dsp_derive::Parameters;

/// The amount of 1.0 means this many allpass sections.
const MAX_STAGES: usize = 100;

/// A disperser with an amount, a cutoff and a bandwidth.
#[derive(Parameters)]
pub struct Dispersion {
	/// How many allpass sections the signal passes through.
	#[range(min = 0.0, max = 1.0)]
	#[unit(percent)]
	#[default(0.5)]
	#[id(display = "Amount")]
	#[layout(group = "Dispersion", order = 0, widget = knob, hero = "phase")]
	pub amount: f32,
	/// The center of the dispersion, in Hz.
	#[range(min = 10.0, max = 30000.0)]
	#[unit(hertz)]
	#[default(440.0)]
	#[id(display = "Cutoff")]
	#[layout(group = "Dispersion", order = 1, widget = knob)]
	#[logarithmic]
	pub cutoff: f32,
	/// How wide the dispersed region is, in Hz.
	#[range(min = 10.0, max = 10000.0)]
	#[unit(hertz)]
	#[default(1000.0)]
	#[id(display = "Spread")]
	#[layout(group = "Dispersion", order = 2, widget = knob)]
	#[logarithmic]
	pub bandwidth: f32,

	#[skip]
	disperser: Disperser<2>,
	#[skip]
	stages: usize,
}

impl Dispersion {
	/// A disperser with half of the sections, which leaves room in both
	/// directions.
	pub fn new(sample_rate: usize) -> Self {
		let mut disperser = Disperser::<2>::new(sample_rate);
		disperser.set_biquad_count(0);

		Self {
			amount: 0.5,
			cutoff: 440.0,
			bandwidth: 1000.0,
			disperser,
			stages: 0,
		}
	}
}

impl Effect<2> for Dispersion {
	fn delay(&self) -> usize {
		0
	}

	fn process(
		&mut self,
		samples: &mut [f32; 2],
		other: &[&[f32; 2]],
		context: &mut Box<dyn ProcessContext>,
	) {
		let stages = (self.amount.clamp(0.0, 1.0) * MAX_STAGES as f32).round() as usize;

		if stages != self.stages {
			self.disperser.set_biquad_count(stages);
			self.stages = stages;
		}

		// The chain skips a design that would change nothing, so this can run
		// every sample and still cost the design once per knob move.
		self.disperser
			.set_filter_parameters(self.cutoff, self.bandwidth);

		self.disperser.process(samples, other, context);
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// Half the amount is half the chain, and moving it grows or shrinks the
	/// vector rather than only changing a coefficient.
	#[test]
	fn the_amount_sets_the_number_of_sections() {
		let mut dispersion = Dispersion::new(48_000);
		let mut context: Box<dyn ProcessContext> = Box::new(());
		let mut frame = [0.0; 2];

		dispersion.amount = 0.5;
		dispersion.process(&mut frame, &[], &mut context);
		assert_eq!(dispersion.stages, 50);

		dispersion.amount = 1.0;
		dispersion.process(&mut frame, &[], &mut context);
		assert_eq!(dispersion.stages, MAX_STAGES);

		dispersion.amount = 0.0;
		dispersion.process(&mut frame, &[], &mut context);
		assert_eq!(dispersion.stages, 0);
	}
}
