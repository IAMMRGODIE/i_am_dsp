//! A simple effect that stacks multiple allpass filters to create a dispersion effect.

use std::f32::consts::PI;

use i_am_dsp_derive::Parameters;

use crate::{Effect, ProcessContext, effects::filter::Biquad};

/// A simple effect that stacks multiple allpass filters to create a dispersion effect.
#[derive(Parameters)]
pub struct Disperser<const CHANNELS: usize = 2> {
	#[skip]
	sample_rate: usize,
	#[range(min = 10.0, max = 30000.0)]
	#[unit(hertz)]
	#[default(440.0)]
	#[id(display = "Cutoff")]
	#[layout(group = "Dispersion", order = 0, widget = knob, hero = "phase")]
	#[logarithmic]
	cutoff: f32,
	#[range(min = 10.0, max = 10000.0)]
	#[unit(hertz)]
	#[default(1000.0)]
	#[id(display = "Spread")]
	#[layout(group = "Dispersion", order = 1, widget = knob)]
	#[logarithmic]
	bandwidth: f32,
	#[sub_param]
	biquads: Vec<Biquad<CHANNELS>>,
	/// The cutoff the sections were designed from, so a parameter write that
	/// bypasses the setter is noticed on the next block.
	#[skip]
	seen_cutoff: f32,
	#[skip]
	seen_bandwidth: f32,
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
			seen_cutoff: 440.0,
			seen_bandwidth: 1000.0,
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
	///
	/// Every section shares the same coefficients, so the prototype is designed
	/// once and then copied. An effect that sweeps the cutoff every sample (see
	/// [`Crusher`](crate::effects::crusher::Crusher)) would otherwise run the
	/// filter design once per section.
	pub fn set_filter_parameters(&mut self, cutoff: f32, bandwidth: f32) {
		// Designing the prototype is not free, and an effect that calls this
		// every sample (the wider does) should cost the design once per move.
		if (cutoff, bandwidth) == (self.seen_cutoff, self.seen_bandwidth) {
			return;
		}

		self.cutoff = cutoff;
		self.bandwidth = bandwidth;
		self.seen_cutoff = cutoff;
		self.seen_bandwidth = bandwidth;

		if self.biquads.is_empty() {
			return;
		}

		let mut prototype: Biquad<CHANNELS> = Biquad::new(self.sample_rate);
		prototype.set_to_bandpass(cutoff, bandwidth);
		prototype.transform_to_allpass();

		for biquad in &mut self.biquads {
			biquad.b = prototype.b;
			biquad.a = prototype.a;
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

	fn process(
		&mut self,
		samples: &mut [f32; CHANNELS],
		other: &[&[f32; CHANNELS]],
		ctx: &mut Box<dyn ProcessContext>,
	) {
		// A parameter write changes the field, not the coefficients; rebuilding
		// here is what makes a host's cutoff actually reach the audio.
		if self.cutoff != self.seen_cutoff || self.bandwidth != self.seen_bandwidth {
			self.set_filter_parameters(self.cutoff, self.bandwidth);
		}

		for biquad in &mut self.biquads {
			biquad.process(samples, other, ctx);
		}
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		use crate::effects::filter::MIN_FREQUENCY;
		use crate::tools::ui_tools::draw_complex_response;
		use egui::*;

		egui::Resize::default()
			.resizable([false, true])
			.min_width(ui.available_width())
			.max_width(ui.available_width())
			.id_salt(format!("{id_prefix}_disperser"))
			.show(ui, |ui| {
				draw_complex_response(ui, self.sample_rate, |freq| self.complex_response(freq));
			});

		let max_freq = self.sample_rate as f32 / 2.0;

		ScrollArea::horizontal().show(ui, |ui| {
			ui.horizontal(|ui| {
				let cutoff_backup = self.cutoff;
				let bandwidth_backup = self.bandwidth;
				let mut current_filters = self.biquads.len();
				ui.add(
					Slider::new(&mut self.cutoff, MIN_FREQUENCY..=max_freq - MIN_FREQUENCY)
						.text("Cutoff Frequency (Hz)")
						.logarithmic(true),
				);
				ui.add(
					Slider::new(&mut self.bandwidth, MIN_FREQUENCY..=10000.0)
						.text("Bandwidth (Hz)")
						.logarithmic(true),
				);
				ui.add(Slider::new(&mut current_filters, 0..=200).text("Amount"));
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
#[cfg(test)]
mod tests {
	use super::*;

	/// A cutoff at the top of the range used to make a single allpass section
	/// diverge: 30 kHz is above Nyquist at 44.1 kHz, the coefficients land on the
	/// unit circle, and the time-domain output becomes NaN even though the frequency
	/// response still reads as unity.
	#[test]
	fn a_high_cutoff_allpass_stays_finite_and_passes_a_signal() {
		let mut disperser = Disperser::<2>::new(44_100);
		disperser.set_biquad_count(100);
		disperser.set_filter_parameters(30_000.0, 10_000.0);

		let mut context: Box<dyn crate::ProcessContext> = Box::new(());
		let samples = 44_100 * 2;
		let mut sum = 0.0f64;

		for index in 0..samples {
			let x = (2.0 * PI * 1_000.0 * index as f32 / 44_100.0).sin() * 0.5;
			let mut frame = [x, x];
			disperser.process(&mut frame, &[], &mut context);

			assert!(frame[0].is_finite(), "non-finite sample at {index}");

			if index > samples / 2 {
				sum += (frame[0] as f64) * (frame[0] as f64);
			}
		}

		let rms = (sum / (samples / 2) as f64).sqrt();
		assert!(rms > 0.2, "the allpass chain attenuated to {rms}");
	}

	/// A host writing the cutoff has to hear it: the chain rebuilds itself when
	/// a value it was designed from moves, rather than only when a setter runs.
	#[test]
	fn a_parameter_write_rebuilds_the_chain() {
		let mut disperser = Disperser::<2>::new(48_000);
		disperser.set_biquad_count(4);
		disperser.set_filter_parameters(1_000.0, 1_000.0);
		let before = disperser.complex_response(1_000.0).1;

		assert!(crate::prelude::Parameters::set_parameter(
			&mut disperser,
			"cutoff",
			crate::prelude::SetValue::Float(4_000.0)
		));

		// The field moved; the design catches up on the next block.
		let mut context: Box<dyn crate::ProcessContext> = Box::new(());
		let mut frame = [0.0f32, 0.0];
		disperser.process(&mut frame, &[], &mut context);

		let after = disperser.complex_response(1_000.0).1;

		assert!(
			(before - after).abs() > 1e-3,
			"phase {before} against {after}"
		);
	}
}
