//! A classic Schroeder reverb: four parallel combs feeding two allpass stages,
//! with a damping lowpass. The comb delays are scaled by Size; their feedback
//! is derived from the requested T60.
//!
//! This is the DSP; the product surface with its Parameters lives in
//! `i_am_effects`.

use i_am_dsp::prelude::{AllpassFilter, CombsFilter, Effect, Lowpass, ProcessContext};

const COMB_MS: [f32; 4] = [29.7, 37.1, 41.1, 43.7];
const ALLPASS_MS: [f32; 2] = [5.0, 1.7];

fn delay_samples(rate: usize, milliseconds: f32) -> usize {
	((milliseconds.max(0.1) / 1_000.0) * rate as f32)
		.round()
		.max(16.0) as usize
}

/// The reverb algorithm: four combs, two allpasses and a damping lowpass.
pub struct Schroeder {
	combs: [CombsFilter<2>; 4],
	allpasses: [AllpassFilter<2>; 2],
	damping_filter: Lowpass<2>,
	sample_rate: usize,
	size: f32,
	decay_time: f32,
	damping: f32,
	last_size: f32,
	last_decay: f32,
	last_damping: f32,
}

impl Schroeder {
	/// Builds the reverb for a sample rate.
	pub fn new(sample_rate: usize) -> Self {
		let combs = core::array::from_fn(|index| {
			CombsFilter::new(0.8, delay_samples(sample_rate, COMB_MS[index]))
		});
		let allpasses = core::array::from_fn(|index| {
			AllpassFilter::new(0.5, delay_samples(sample_rate, ALLPASS_MS[index]))
		});
		let damping_filter = Lowpass::new(sample_rate, 8_000.0, 0.7);

		Self {
			combs,
			allpasses,
			damping_filter,
			sample_rate,
			size: 1.0,
			decay_time: 2_200.0,
			damping: 8_000.0,
			last_size: 1.0,
			last_decay: 2_200.0,
			last_damping: 8_000.0,
		}
	}

	/// Sets the size, T60 and damping; the filters update on the next frame.
	pub fn set_parameters(&mut self, size: f32, decay_time: f32, damping: f32) {
		self.size = size;
		self.decay_time = decay_time;
		self.damping = damping;
	}

	fn update(&mut self) {
		let t60 = (self.decay_time / 1_000.0).max(0.05);

		for (comb, milliseconds) in self.combs.iter_mut().zip(COMB_MS) {
			let delay = delay_samples(self.sample_rate, milliseconds * self.size);
			comb.resize_history(delay);
			let seconds = delay as f32 / self.sample_rate as f32;
			comb.feedback = 10f32.powf(-3.0 * seconds / t60).clamp(0.01, 0.99);
		}

		for (allpass, milliseconds) in self.allpasses.iter_mut().zip(ALLPASS_MS) {
			allpass.resize_history(delay_samples(self.sample_rate, milliseconds * self.size));
			allpass.feedback = (0.45 + self.size * 0.05).clamp(0.3, 0.75);
		}

		self.damping_filter.cutoff = self.damping;
		self.last_size = self.size;
		self.last_decay = self.decay_time;
		self.last_damping = self.damping;
	}

	/// Clears the reverb tail.
	pub fn clear_tail(&mut self) {
		for comb in &mut self.combs {
			comb.clear_history();
		}
		for allpass in &mut self.allpasses {
			allpass.clear_history();
		}
		self.damping_filter.clear_state();
	}

	/// Runs the wet path for one stereo frame.
	pub fn process(
		&mut self,
		dry: [f32; 2],
		other: &[&[f32; 2]],
		context: &mut Box<dyn ProcessContext>,
	) -> [f32; 2] {
		if (self.size - self.last_size).abs() > 1e-4
			|| (self.decay_time - self.last_decay).abs() > 1e-4
			|| (self.damping - self.last_damping).abs() > 1e-4
		{
			self.update();
		}

		let mut wet = [0.0; 2];

		for comb in &mut self.combs {
			let mut channel = dry;
			comb.process(&mut channel, other, context);
			for (index, sample) in channel.iter().enumerate() {
				wet[index] += sample;
			}
		}

		for sample in &mut wet {
			*sample *= 0.25;
		}

		for allpass in &mut self.allpasses {
			allpass.process(&mut wet, other, context);
		}

		self.damping_filter.process(&mut wet, other, context);
		wet
	}
}
