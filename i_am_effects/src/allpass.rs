//! A single allpass filter as a product, with a delay and a feedback.

use i_am_dsp::prelude::{AllpassFilter, Effect, ProcessContext};
use i_am_dsp_derive::Parameters;

fn delay_samples(rate: usize, milliseconds: f32) -> usize {
	((milliseconds.max(0.1) / 1_000.0) * rate as f32)
		.round()
		.max(16.0) as usize
}

#[derive(Parameters)]
pub struct AllpassEffect {
	/// The feedback delay.
	#[range(min = 1.0, max = 50.0)]
	#[unit(milliseconds)]
	#[default(23.0)]
	#[id(display = "Delay")]
	#[layout(group = "Allpass", order = 0, widget = knob, hero = "waveform")]
	pub delay_ms: f32,

	/// The allpass coefficient.
	#[range(min = 0.01, max = 0.99)]
	#[default(0.7)]
	#[id(display = "Feedback")]
	#[layout(group = "Allpass", order = 1, widget = knob)]
	pub feedback: f32,

	#[skip]
	filter: AllpassFilter<2>,
	#[skip]
	sample_rate: usize,
	#[skip]
	last_delay: f32,
}

impl AllpassEffect {
	pub fn new(sample_rate: usize) -> Self {
		let delay_ms = 23.0;
		let filter = AllpassFilter::new(0.7, delay_samples(sample_rate, delay_ms));

		Self {
			delay_ms,
			feedback: 0.7,
			filter,
			sample_rate,
			last_delay: delay_ms,
		}
	}
}

impl Effect<2> for AllpassEffect {
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