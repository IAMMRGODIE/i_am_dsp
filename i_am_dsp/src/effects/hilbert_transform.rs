//! Hilbert transform effect

use i_am_dsp_derive::Parameters;

use crate::{Effect, ProcessContext};

use std::f32::consts::PI;

/// An IIR Hilbert transform effect, for FIR version see [`crate::effects::convolver`]
#[derive(Parameters)]
pub struct HilbertTransform<const ORDER: usize, const CHANNELS: usize = 2> {
	#[skip]
	x: [[f32; ORDER]; CHANNELS],
	#[skip]
	y: [[f32; ORDER]; CHANNELS],

	#[skip]
	sample_rate: usize,
}

impl<const ORDER: usize, const CHANNELS: usize> HilbertTransform<ORDER, CHANNELS> {
	const BIQUAD_ORDER_2: [[f32; 2]; 1] = [
		[-1.979_999_9, 0.980_1],
	];
	const BIQUAD_ORDER_4: [[f32; 2]; 2] = [
		[-1.979_999_9, 0.980_1],
		[0.99999995, 0.25],
	];
	const BIQUAD_ORDER_6: [[f32; 2]; 3] = [
		[-1.979_999_9, 0.980_1],
		[-0.00000000, 0.722_5],
		[0.99999995, 0.25],
	];
	const BIQUAD_ORDER_8: [[f32; 2]; 4] = [
		[-1.979_999_9, 0.980_1],
		[-0.499_909_3, 0.25],
		[0.499_909_3, 0.25],
		[0.99999995, 0.25],
	];
	const BIQUAD_ORDER_10: [[f32; 2]; 5] = [
		[-1.979_999_9, 0.980_1],
		[-0.706_995_7, 0.25],
		[-0.00000000, 0.722_5],
		[0.706_995_7, 0.25],
		[0.99999995, 0.25],
	];
	const BIQUAD_ORDER_12: [[f32; 2]; 6] = [
		[-1.979_999_9, 0.980_1],
		[-0.808_906_2, 0.25],
		[-0.30895724, 0.25],
		[0.30895724, 0.25],
		[0.808_906_2, 0.25],
		[0.99999995, 0.25],
	];
	const BIQUAD_ORDER_14: [[f32; 2]; 7] = [
		[-1.979_999_9, 0.980_1],
		[-0.86592067, 0.25],
		[-0.499_909_3, 0.25],
		[-0.00000000, 0.722_5],
		[0.499_909_3, 0.25],
		[0.86592067, 0.25],
		[0.99999995, 0.25],
	];
	const BIQUAD_ORDER_16: [[f32; 2]; 8] = [
		[-1.979_999_9, 0.980_1],
		[-0.90087148, 0.25],
		[-0.62338453, 0.25],
		[-0.22247718, 0.25],
		[0.22247718, 0.25],
		[0.62338453, 0.25],
		[0.90087148, 0.25],
		[0.99999995, 0.25],
	];
	
	/// Create a new HilbertTransform instance.
	/// 
	/// # Panics
	/// 
	/// Panics 
	/// 1. `ORDER` is not a positive even integer less than or equal to 16.
	/// 2. `CHANNELS` is not a positive integer.
	pub const fn new(sample_rate: usize) -> Self {
		assert!(ORDER > 0 && CHANNELS > 0 && ORDER <= 16 && ORDER.is_multiple_of(2));

		Self {
			x: [[0.0; ORDER]; CHANNELS],
			y: [[0.0; ORDER]; CHANNELS],
			sample_rate,
		}
	}

	#[inline]
	const fn biquad_coefficients() -> &'static [[f32; 2]] {
		match ORDER {
			2 => &Self::BIQUAD_ORDER_2,
			4 => &Self::BIQUAD_ORDER_4,
			6 => &Self::BIQUAD_ORDER_6,
			8 => &Self::BIQUAD_ORDER_8,
			10 => &Self::BIQUAD_ORDER_10,
			12 => &Self::BIQUAD_ORDER_12,
			14 => &Self::BIQUAD_ORDER_14,
			16 => &Self::BIQUAD_ORDER_16,
			_ => panic!("Invalid order"),
		}
	}

	/// Apply the Hilbert transform to the given samples.
	pub fn apply_transform(&mut self, samples: &mut [f32; CHANNELS]) {
		let biquad_coefficients = Self::biquad_coefficients();

		for (ch, input) in samples.iter_mut().enumerate() {
			for (section, [a1, a2]) in biquad_coefficients.iter().enumerate() {
				let idx = section * 2;
				let x_hist1 = self.x[ch][idx];
				let x_hist2 = self.x[ch][idx + 1];
				let y_hist1 = self.y[ch][idx];
				let y_hist2 = self.y[ch][idx + 1];
				
				let output = 
					a2 * *input +
					a1 * x_hist1 +
					x_hist2 -
					a1 * y_hist1 -
					a2 * y_hist2;
				
				self.x[ch][idx + 1] = x_hist1;
				self.x[ch][idx] = *input;
				self.y[ch][idx + 1] = y_hist1;
				self.y[ch][idx] = output;
				
				*input = output;
			}
		}
	}

	/// Calculate the complex response of the filter at the given frequency.
	/// 
	/// Returns a tuple of the amplitude and phase in radians.
	pub fn complex_response(&self, freq: f32) -> (f32, f32) {
		let frequency = 2.0 * PI * freq / self.sample_rate as f32;

		let cos_f = frequency.cos();
		let sin_f = frequency.sin();
		let cos_2f = 2.0 * cos_f * cos_f - 1.0;
		let sin_2f = 2.0 * cos_f * sin_f;

		let mut amplitude_out = 1.0;
		let mut phase_out = 0.0;

		let biquad_coefficients = Self::biquad_coefficients();

		for [a1, a2] in biquad_coefficients.iter() {
			let real_n = a2 + a1 * cos_f + cos_2f;
			let imag_n = - a1 * sin_f - sin_2f;

			let real_d = 1.0 + a1 * cos_f + a2 * cos_2f;
			let imag_d = - a1 * sin_f - a2 * sin_2f;

			let amplitude = real_n.hypot(imag_n) / real_d.hypot(imag_d);
			let phase = (imag_n.atan2(real_n) - imag_d.atan2(real_d)).rem_euclid(2.0 * PI);

			amplitude_out *= amplitude;
			phase_out += phase;
			phase_out = phase_out.rem_euclid(2.0 * PI)
		}

		(amplitude_out, phase_out)
	}
}

impl<const ORDER: usize, const CHANNELS: usize> Effect<CHANNELS> for HilbertTransform<ORDER, CHANNELS> {
	fn process(&mut self, samples: &mut [f32; CHANNELS], _other: &[&[f32; CHANNELS]], _: &mut Box<dyn ProcessContext>) {
		self.apply_transform(samples);
	}

	fn delay(&self) -> usize {
		0
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"Hilbert Transform"
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		use crate::tools::ui_tools::draw_complex_response;

		egui::Resize::default().resizable([false, true])
			.min_width(ui.available_width())
			.max_width(ui.available_width())
			.id_salt(format!("{id_prefix}_hilbert_transform_demo"))
			.show(ui, |ui| 
		{
			draw_complex_response(ui, self.sample_rate, |freq| self.complex_response(freq));
		});
	}
}