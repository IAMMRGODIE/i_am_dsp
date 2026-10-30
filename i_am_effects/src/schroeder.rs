//! A classic Schroeder reverb: four parallel combs feeding two allpass stages,
//! with a damping lowpass. The comb delays are scaled by Size; their feedback
//! is derived from the requested T60.

use i_am_dsp::prelude::{
	AllpassFilter, CombsFilter, Effect, Lowpass, ProcessContext,
};
use i_am_dsp_derive::Parameters;

const COMB_MS: [f32; 4] = [29.7, 37.1, 41.1, 43.7];
const ALLPASS_MS: [f32; 2] = [5.0, 1.7];

fn delay_samples(rate: usize, milliseconds: f32) -> usize {
	((milliseconds.max(0.1) / 1_000.0) * rate as f32)
		.round()
		.max(16.0) as usize
}

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
	combs: [CombsFilter<2>; 4],
	#[skip]
	allpasses: [AllpassFilter<2>; 2],
	#[skip]
	damping_filter: Lowpass<2>,
	#[skip]
	sample_rate: usize,
	#[skip]
	last_size: f32,
	#[skip]
	last_decay: f32,
	#[skip]
	last_damping: f32,
}

impl SchroederEffect {
	pub fn new(sample_rate: usize) -> Self {
		let combs = core::array::from_fn(|index| {
			CombsFilter::new(0.8, delay_samples(sample_rate, COMB_MS[index]))
		});
		let allpasses = core::array::from_fn(|index| {
			AllpassFilter::new(0.5, delay_samples(sample_rate, ALLPASS_MS[index]))
		});
		let damping_filter = Lowpass::new(sample_rate, 8_000.0, 0.7);

		Self {
			size: 1.0,
			decay_time: 2_200.0,
			damping: 8_000.0,
			dry_gain: 0.2,
			wet_gain: 0.8,
			combs,
			allpasses,
			damping_filter,
			sample_rate,
			last_size: 1.0,
			last_decay: 2_200.0,
			last_damping: 8_000.0,
		}
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
}

impl Effect<2> for SchroederEffect {
	fn delay(&self) -> usize {
		0
	}

	fn clear_tail(&mut self) {
		for comb in &mut self.combs {
			comb.clear_history();
		}
		for allpass in &mut self.allpasses {
			allpass.clear_history();
		}
		self.damping_filter.clear_state();
	}

	fn process(
		&mut self,
		samples: &mut [f32; 2],
		other: &[&[f32; 2]],
		context: &mut Box<dyn ProcessContext>,
	) {
		if (self.size - self.last_size).abs() > 1e-4
			|| (self.decay_time - self.last_decay).abs() > 1e-4
			|| (self.damping - self.last_damping).abs() > 1e-4
		{
			self.update();
		}

		let dry = *samples;
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

		for (index, sample) in samples.iter_mut().enumerate() {
			*sample = dry[index] * self.dry_gain + wet[index] * self.wet_gain;
		}
	}
}