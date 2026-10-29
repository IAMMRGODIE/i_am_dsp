//! A single comb filter as a product, with a delay and a feedback.

use i_am_dsp::prelude::{CombsFilter, Effect, ProcessContext};
use i_am_dsp_derive::Parameters;

fn delay_samples(rate: usize, milliseconds: f32) -> usize {
	((milliseconds.max(0.1) / 1_000.0) * rate as f32)
		.round()
		.max(16.0) as usize
}

#[derive(Parameters)]
pub struct CombEffect {
	/// The feedback delay.
	#[range(min = 1.0, max = 50.0)]
	#[unit(milliseconds)]
	#[default(31.0)]
	#[id(display = "Delay")]
	#[layout(group = "Comb", order = 0, widget = knob, hero = "waveform")]
	pub delay_ms: f32,

	/// The feedback amount.
	#[range(min = 0.01, max = 1.0)]
	#[default(0.8)]
	#[id(display = "Feedback")]
	#[layout(group = "Comb", order = 1, widget = knob)]
	pub feedback: f32,

	#[skip]
	filter: CombsFilter<2>,
	#[skip]
	sample_rate: usize,
	#[skip]
	last_delay: f32,
}

impl CombEffect {
	pub fn new(sample_rate: usize) -> Self {
		let delay_ms = 31.0;
		let filter = CombsFilter::new(0.8, delay_samples(sample_rate, delay_ms));

		Self {
			delay_ms,
			feedback: 0.8,
			filter,
			sample_rate,
			last_delay: delay_ms,
		}
	}
}

impl Effect<2> for CombEffect {
	fn delay(&self) -> usize {
		0
	}

	fn process(
		&mut self,
		samples: &mut [f32; 2],
		other: &[&[f32; 2]],
		context: &mut Box<dyn ProcessContext>,
	) {
		if (self.delay_ms - self.last_delay).abs() > 1e-4 {
			self.filter.resize_history(delay_samples(self.sample_rate, self.delay_ms));
			self.last_delay = self.delay_ms;
		}

		self.filter.feedback = self.feedback;
		self.filter.process(samples, other, context);
	}
}