//! Smooth the effect like a compressor did.

use i_am_dsp_derive::Parameters;

use crate::{tools::smoother::DoubleTimeConstant, Effect, ProcessContext};

/// Smooth the effect like a compressor did.
#[derive(Parameters)]
pub struct SmoothedEffect<Effector: Effect<CHANNELS>, const CHANNELS: usize> {
	#[sub_param]
	effect: Effector,
	/// Smoother for rapidly changing signals.
	#[sub_param]
	pub smoother: DoubleTimeConstant<CHANNELS>,
}

impl<Effector: Effect<CHANNELS>, const CHANNELS: usize> SmoothedEffect<Effector, CHANNELS> {
	/// Create a new `SmoothedEffect`.
	/// 
	/// # Panics
	/// 
	/// Panics if `CHANNELS` is less than or equal to 0.
	pub fn new(effect: Effector, attack_time: f32, release_time: f32, sample_rate: usize) -> Self {
		assert!(CHANNELS > 0, "CHANNELS must be greater than 0");

		Self {
			effect,
			smoother: DoubleTimeConstant::new(attack_time, release_time, 1.0, sample_rate),
		}
	}
}

impl<Effector: Effect<CHANNELS>, const CHANNELS: usize> Effect<CHANNELS> for SmoothedEffect<Effector, CHANNELS> {
	fn delay(&self) -> usize {
		self.effect.delay()
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"Smoothed Effect"
	}

	fn process(&mut self, samples: &mut [f32; CHANNELS], other: &[&[f32; CHANNELS]], ctx: &mut Box<dyn ProcessContext>) {
		let mut input_backup = *samples;
		self.effect.process(&mut input_backup, other, ctx);
		let mut ratio = [1.0; CHANNELS];
		for (i, (input_value, value)) in input_backup.iter().zip(samples.iter()).enumerate() {
			ratio[i] = *input_value / *value;
		}
		self.smoother.input_value(&ratio);
		let smoothed_gain = self.smoother.get_smoothed_result();
		for (i, (input_value, value)) in input_backup.iter().zip(samples.iter_mut()).enumerate() {
			*value = *input_value * smoothed_gain[i];
		}
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		self.effect.demo_ui(ui, format!("{}_smoothed_effect", id_prefix));
		ui.add(egui::Slider::new(&mut self.smoother.attack_time, 0.0..=1000.0).text("Attack Time (ms)"));
		ui.add(egui::Slider::new(&mut self.smoother.release_time, 0.0..=1000.0).text("Release Time (ms)"));
	}
}