//! A simple effect that stacks multiple allpass filters to create a dispersion effect.

use std::f32::consts::PI;

use i_am_dsp_derive::Parameters;

use crate::{effects::filter::Biquad, Effect, ProcessContext};

/// A simple effect that stacks multiple allpass filters to create a dispersion effect.
#[derive(Parameters)]
pub struct Disperser<const CHANNELS: usize = 2> {
	#[skip]
	sample_rate: usize,
	#[range(min = 10.0, max = 30000.0)]
	cutoff: f32,
	#[range(min = 10.0, max = 10000.0)]
	bandwidth: f32,
	#[sub_param]
	biquads: Vec<Biquad<CHANNELS>>,
}

impl<const CHANNELS: usize> Disperser<CHANNELS> {
	/// Create a new disperser with the given sample rate.
	/// 
	/// # Panics
	/// 
	/// Panics if `CHANNELS` is less than or equal to 0.
	pub const fn new(sample_rate: usize) -> Self {
		assert!(CHANNELS > 0, "CHANNELS must be greater than 0");

		Self {
			sample_rate,
			cutoff: 440.0,
			bandwidth: 1000.0,
			biquads: Vec::new(),
		}
	}

	/// Set the number of allpass filters in the disperser.
	pub fn set_biquad_count(&mut self, count: usize) {
		if count > self.biquads.len() {
			let mut biquad = Biquad::new(self.sample_rate);
			biquad.set_to_bandpass(self.cutoff, self.bandwidth);
			biquad.transform_to_allpass();
			self.biquads.resize(count, biquad);
		} else {
			self.biquads.truncate(count);
		}
	}

	/// Set the filter parameters for the disperser.
	pub fn set_filter_parameters(&mut self, cutoff: f32, bandwidth: f32) {
		self.cutoff = cutoff;
		self.bandwidth = bandwidth;
		for biquad in &mut self.biquads {
			biquad.set_to_bandpass(cutoff, bandwidth);
			biquad.transform_to_allpass();
		}
	}

	/// Compute the complex response of the disperser at the given frequency.
	/// 
	/// Returns a tuple of the amplitude and phase of the response.
	pub fn complex_response(&self, frequency: f32) -> (f32, f32) {
		let mut norm = 1.0;
		let mut phase = 0.0;
		for biquad in &self.biquads {
			let (amplitude, biquad_phase) = biquad.complex_response(frequency);
			norm *= amplitude;
			phase += biquad_phase;
			phase = phase.rem_euclid(2.0 * PI);
		}
		(norm, phase)
	}
}

impl<const CHANNELS: usize> Effect<CHANNELS> for Disperser<CHANNELS> {
	fn delay(&self) -> usize {
		0
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"Disperser"
	}

	fn process(&mut self, samples: &mut [f32; CHANNELS], other: &[&[f32; CHANNELS]], ctx: &mut Box<dyn ProcessContext>) {
		for biquad in &mut self.biquads {
			biquad.process(samples, other, ctx);
		}
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		use egui::*;
		use crate::effects::filter::MIN_FREQUENCY;
		use crate::tools::ui_tools::draw_complex_response;

		egui::Resize::default().resizable([false, true])
			.min_width(ui.available_width())
			.max_width(ui.available_width())
			.id_salt(format!("{id_prefix}_disperser"))
			.show(ui, |ui| 
		{
			draw_complex_response(ui, self.sample_rate, |freq| self.complex_response(freq));
		});

		let max_freq = self.sample_rate as f32 / 2.0;

		ScrollArea::horizontal().show(ui, |ui| {
			ui.horizontal(|ui| {
				let cutoff_backup = self.cutoff;
				let bandwidth_backup = self.bandwidth;
				let mut current_filters = self.biquads.len();
				ui.add(Slider::new(&mut self.cutoff, MIN_FREQUENCY..=max_freq - MIN_FREQUENCY)
					.text("Cutoff Frequency (Hz)")
					.logarithmic(true)
				);
				ui.add(Slider::new(&mut self.bandwidth, MIN_FREQUENCY..=10000.0)
					.text("Bandwidth (Hz)")
					.logarithmic(true)
				);
				ui.add(Slider::new(&mut current_filters, 0..=200)
					.text("Amount")
				);
				if cutoff_backup != self.cutoff || bandwidth_backup != self.bandwidth {
					self.set_filter_parameters(self.cutoff, self.bandwidth);
				}
				if current_filters != self.biquads.len() {
					self.set_biquad_count(current_filters);
				}
			});
		});
	}
}