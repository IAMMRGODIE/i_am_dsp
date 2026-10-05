//! A disperser whose cutoff is swept by an LFO, a saturator and an amplitude modulation.
//!
//! This is the DSP of the `i_am_crusher` plug-in. The signal is pushed through
//! a stack of allpass sections (a disperser) whose cutoff an LFO sweeps across
//! the audible range, then through a saturating curve and an amplitude
//! modulation. The stack is the library's [`Disperser`], the curve is the one
//! [`Saturator`](crate::effects::distortion::Saturator) uses evaluated with
//! `p = 2` (see [`saturate`]), and the two gains are [`Gain`] effects; only the
//! LFO, the way the amount drives the three of them and the parameter ranges
//! are new.

use std::f32::consts::PI;

use i_am_dsp_derive::Parameters;

use crate::{
	effects::{
		disperser::Disperser,
		distortion::saturate,
		stereo_control::Gain,
	},
	Effect, ProcessContext,
};

/// The lowest cutoff the LFO sweeps the disperser to, in Hz.
const MIN_CUTOFF: f32 = 1.0;
/// The highest cutoff the LFO sweeps the disperser to, in Hz.
const MAX_CUTOFF: f32 = 30000.0;
/// The bandwidth of the disperser sections at an amount of 1, in Hz.
const MAX_BANDWIDTH: f32 = 10000.0;
/// The number of allpass sections an amount of 1 applies.
const MAX_STACK_AMOUNT: usize = 100;

/// A disperser with an LFO driven cutoff, a saturator and an amplitude modulation.
///
/// The amount drives all three stages at once: it decides how many allpass
/// sections are applied, how wide they are, and how hard the saturator is
/// driven, while the LFO sweeps the cutoff of those sections between
/// [`MIN_CUTOFF`] and [`MAX_CUTOFF`], scaled by the amount.
#[derive(Parameters)]
pub struct Crusher<const CHANNELS: usize = 2> {
	/// How much dispersion and saturation is applied, from 0 to 1.
	#[range(min = 0.0, max = 1.0)]
	pub amount: f32,
	/// The rate of the LFO that sweeps the disperser cutoff, in Hz.
	#[range(min = 1.0, max = 2000.0)]
	#[logarithmic]
	pub frequency: f32,
	/// How much the output is amplitude modulated, from 0 to 1.
	#[range(min = 0.0, max = 1.0)]
	pub am_amount: f32,
	/// The rate of the amplitude modulation, in Hz.
	#[range(min = 1.0, max = 2000.0)]
	#[logarithmic]
	pub am_frequency: f32,
	/// The gain applied before the effect.
	#[sub_param]
	pub input_gain: Gain,
	/// The gain applied after the effect.
	#[sub_param]
	pub output_gain: Gain,
	#[skip]
	sample_rate: usize,
	#[skip]
	disperser: Disperser<CHANNELS>,
	#[skip]
	phase: f32,
	#[skip]
	am_phase: f32,
	#[skip]
	biquad_count: usize,
}

impl<const CHANNELS: usize> Crusher<CHANNELS> {
	/// Creates a new crusher for the given sample rate.
	///
	/// # Panics
	///
	/// Panics if `CHANNELS` is 0.
	pub fn new(sample_rate: usize) -> Self {
		assert!(CHANNELS > 0, "CHANNELS must be greater than 0");

		let mut disperser = Disperser::new(sample_rate);
		// Growing the section stack allocates, so the room for the whole stack is
		// reserved here instead of when the amount is first turned up while
		// processing audio.
		disperser.set_biquad_count(MAX_STACK_AMOUNT);
		disperser.set_biquad_count(0);

		Self {
			amount: 0.0,
			frequency: 299.0,
			am_amount: 0.0,
			am_frequency: 486.0,
			input_gain: Gain::new(1.0),
			output_gain: Gain::new(1.0),
			sample_rate,
			disperser,
			phase: 0.0,
			am_phase: 0.0,
			biquad_count: 0,
		}
	}
}

impl<const CHANNELS: usize> Effect<CHANNELS> for Crusher<CHANNELS> {
	fn delay(&self) -> usize {
		0
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"Crusher"
	}

	fn process(
		&mut self,
		samples: &mut [f32; CHANNELS],
		other: &[&[f32; CHANNELS]],
		process_context: &mut Box<dyn ProcessContext>,
	) {
		self.input_gain.process(samples, other, process_context);

		let amount = self.amount.clamp(0.0, 1.0);
		let sample_rate = self.sample_rate as f32;

		let lfo = self.phase.sin();
		self.phase = (self.phase + 2.0 * PI * self.frequency / sample_rate).rem_euclid(2.0 * PI);

		let am_lfo = self.am_phase.sin();
		self.am_phase = (self.am_phase + 2.0 * PI * self.am_frequency / sample_rate).rem_euclid(2.0 * PI);

		// Only the cutoff is modulated: the amount alone decides how many sections
		// are applied, so the stack is only resized when that changes.
		let count = (amount * MAX_STACK_AMOUNT as f32).round() as usize;
		if count != self.biquad_count {
			self.disperser.set_biquad_count(count);
			self.biquad_count = count;
		}

		if count > 0 {
			let bandwidth = amount * MAX_BANDWIDTH;
			let cutoff = amount * (lfo * 0.5 + 0.5) * (MAX_CUTOFF - MIN_CUTOFF) + MIN_CUTOFF;
			self.disperser.set_filter_parameters(cutoff, bandwidth);
			self.disperser.process(samples, other, process_context);
		}

		// The curve of `Saturator` with `p = 2`, driven by the amount.
		let a = (1.0 - amount) * 4.5 + 0.5;
		let am_gain = 1.0 - self.am_amount + self.am_amount * am_lfo;

		for sample in samples.iter_mut() {
			*sample = saturate(*sample, a, 2.0) * am_gain;
		}

		self.output_gain.process(samples, other, process_context);
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, _: String) {
		use crate::tools::ui_tools::gain_ui;
		use egui::Slider;

		ui.add(Slider::new(&mut self.amount, 0.0..=1.0).text("Amount"));
		ui.add(Slider::new(&mut self.frequency, 1.0..=2000.0).text("LFO Frequency (Hz)").logarithmic(true));
		ui.add(Slider::new(&mut self.am_amount, 0.0..=1.0).text("AM Amount"));
		ui.add(Slider::new(&mut self.am_frequency, 1.0..=2000.0).text("AM Frequency (Hz)").logarithmic(true));
		gain_ui(ui, &mut self.input_gain.gain, Some("Input Gain".to_string()), false);
		gain_ui(ui, &mut self.output_gain.gain, Some("Output Gain".to_string()), false);
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// The saturating curve peaks at `x = a`, where it is `(a + 1 / a) / 2`, so
	/// that bounds the absolute output for any input.
	fn peak_bound(a: f32) -> f32 {
		(a + 1.0 / a) / 2.0
	}

	fn run(amount: f32, am_amount: f32, input: &[f32]) -> Vec<f32> {
		let mut crusher = Crusher::<1>::new(48000);
		crusher.amount = amount;
		crusher.am_amount = am_amount;

		let mut context: Box<dyn crate::ProcessContext> = Box::new(());
		let mut output = Vec::with_capacity(input.len());
		for sample in input {
			let mut frame = [*sample];
			crusher.process(&mut frame, &[], &mut context);
			output.push(frame[0]);
		}
		output
	}

	#[test]
	fn silence_stays_silent() {
		for amount in [0.0, 0.5, 1.0] {
			let output = run(amount, 1.0, &[0.0; 1024]);
			assert!(output.iter().all(|sample| *sample == 0.0), "amount {amount} made silence ring");
		}
	}

	#[test]
	fn saturator_bounds_a_hot_input() {
		// A loud, sign alternating input, so the allpass stack is excited as well.
		let input: Vec<f32> = (0..8192)
			.map(|i| if i % 2 == 0 { 8.0 } else { -8.0 })
			.collect();

		for amount in [0.0, 0.25, 0.5, 1.0] {
			let output = run(amount, 0.0, &input);
			let bound = peak_bound((1.0 - amount) * 4.5 + 0.5) * 1.001;
			for sample in &output {
				assert!(sample.is_finite(), "amount {amount} produced a non finite sample");
				assert!(sample.abs() <= bound, "amount {amount} produced {sample}, above {bound}");
			}
		}
	}

	#[test]
	fn the_parameters_are_reachable_by_their_identifiers() {
		use crate::prelude::{Parameters, SetValue};

		let mut crusher = Crusher::<2>::new(48000);
		let identifiers: Vec<String> = crusher
			.get_parameters()
			.into_iter()
			.map(|parameter| parameter.identifier)
			.collect();
		assert_eq!(identifiers, vec![
			"amount",
			"frequency",
			"am_amount",
			"am_frequency",
			"input_gain.gain",
			"output_gain.gain",
		]);

		assert!(crusher.set_parameter("amount", SetValue::Float(0.5)));
		assert_eq!(crusher.amount, 0.5);
		assert!(crusher.set_parameter("output_gain.gain", SetValue::Float(2.0)));
		assert_eq!(crusher.output_gain.gain, 2.0);
	}

	#[test]
	fn the_amount_changes_the_sound() {
		let input: Vec<f32> = (0..8192)
			.map(|i| (2.0 * PI * 440.0 * i as f32 / 48000.0).sin() * 0.5)
			.collect();

		let dry = run(0.0, 0.0, &input);
		let wet = run(1.0, 0.0, &input);

		let difference: f32 = dry
			.iter()
			.zip(wet.iter())
			.map(|(dry, wet)| (dry - wet).abs())
			.sum();
		assert!(difference > 1.0, "the wet signal is identical to the dry one");
	}
}
