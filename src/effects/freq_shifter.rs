//! Frequency shifter using Hilbert transform.

use std::f32::consts::PI;

use i_am_parameters_derive::Parameters;

use crate::{effects::{filter::{Biquad, MIN_FREQUENCY}, prelude::{hilbert_transform, Convolver, HilbertTransform}}, Effect, ProcessContext};

/// A frequency shifter using an FIR Hilbert transform.
#[derive(Parameters)]
pub struct FIRFreqShifter<const CHANNELS: usize = 2> {
	/// The sample rate of the audio, saves in Hz.
	#[skip]
	pub sample_rate: usize,
	#[sub_param]
	hilbert_transform: Convolver<CHANNELS>,
	#[skip]
	phase_state: f32,
	#[range(min = -20000.0, max = 20000.0)]
	shift_freq: f32,
	#[sub_param]
	filter: Biquad<CHANNELS>,
	
	#[cfg(feature = "real_time_demo")]
	#[serde]
	order_of_transform: usize,
}

impl<const CHANNELS: usize> FIRFreqShifter<CHANNELS> {
	/// Creates a new frequency shifter with the given sample rate and shift frequency.
	/// 
	/// Panics if `CHANNELS` is 0.
	pub fn new(sample_rate: usize, shift_freq: f32) -> Self {
		assert!(CHANNELS > 0, "CHANNELS must be greater than 0");

		Self {
			sample_rate,
			hilbert_transform: Convolver::new(
				hilbert_transform::<CHANNELS>(127),
				&super::prelude::DelyaCaculateMode::Fir
			),
			phase_state: 0.0,
			shift_freq,
			filter: Biquad::new(sample_rate),

			#[cfg(feature = "real_time_demo")]
			order_of_transform: 127
		}
	}

	/// Sets the shift frequency of the frequency shifter.
	pub fn set_shift_freq(&mut self, shift_freq: f32) {
		let sample_rate = self.sample_rate as f32;
		let shift_freq = shift_freq.clamp(- sample_rate / 2.0 + MIN_FREQUENCY, sample_rate / 2.0 - MIN_FREQUENCY);

		self.shift_freq = shift_freq;
		if shift_freq > MIN_FREQUENCY {
			self.filter.set_to_butterworth_lowpass(sample_rate / 2.0 - shift_freq);
		}else if shift_freq <= MIN_FREQUENCY && shift_freq > -MIN_FREQUENCY {
			self.filter.set_to_default_state()
		}else {
			self.filter.set_to_butterworth_highpass(shift_freq.abs());
		}
	}
}

impl<const CHANNELS: usize> Effect<CHANNELS> for FIRFreqShifter<CHANNELS> {
	fn delay(&self) -> usize {
		self.hilbert_transform.delay()
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"FIR Freq Shifter"
	}

	fn process(&mut self, samples: &mut [f32; CHANNELS], other: &[&[f32; CHANNELS]], ctx: &mut Box<dyn ProcessContext>) {
		if self.shift_freq == 0.0 {
			return;
		}

		self.filter.process(samples, other, ctx);

		let phase_increment = 2.0 * PI * self.shift_freq / self.sample_rate as f32;
		let mut imag_parts = *samples;
		self.hilbert_transform.process(&mut imag_parts, other, ctx);
		let phase_real = self.phase_state.cos();
		let phase_imag = self.phase_state.sin();

		for (i, real_part) in samples.iter_mut().enumerate() {
			*real_part = imag_parts[i] * phase_real - imag_parts[i] * phase_imag;
		}

		self.phase_state = (self.phase_state + phase_increment) % (2.0 * PI);
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, _: String) {
		let mut shift_freq = self.shift_freq;
		ui.add(egui::Slider::new(&mut shift_freq, -20000.0..=20000.0)
			.text("Shift Frequency (Hz)")
		);

		if self.shift_freq != shift_freq {
			self.set_shift_freq(shift_freq);
		}

		let mut order_of_transform = self.order_of_transform;
		ui.add(egui::Slider::new(&mut order_of_transform, 15..=511)
			.text("Order of Transform")
		);

		if order_of_transform != self.order_of_transform {
			if order_of_transform.is_multiple_of(2) {
				order_of_transform += 1;
			}
			self.order_of_transform = order_of_transform;
			self.hilbert_transform.replace_ir(
				hilbert_transform::<CHANNELS>(order_of_transform),
				&super::prelude::DelyaCaculateMode::Fir
			);
		}
	}
}

/// A frequency shifter using an IIR Hilbert transform.
#[derive(Parameters)]
pub struct IIRFreqShifter<const ORDER: usize, const CHANNELS: usize = 2> {
	/// The sample rate of the audio, saves in Hz.
	#[skip]
	pub sample_rate: usize,
	#[sub_param]
	hilbert_transform: HilbertTransform<ORDER, CHANNELS>,
	#[skip]
	phase_state: f32,
	#[range(min = -20000.0, max = 20000.0)]
	shift_freq: f32,
	#[sub_param]
	filter: Biquad<CHANNELS>
}

impl<const ORDER: usize, const CHANNELS: usize> IIRFreqShifter<ORDER, CHANNELS> {
	/// Creates a new frequency shifter with the given sample rate and shift frequency.
	/// 
	/// Panics if `CHANNELS` is 0.
	pub const fn new(sample_rate: usize, shift_freq: f32) -> Self {
		assert!(CHANNELS > 0, "CHANNELS must be greater than 0");
		
		Self {
			sample_rate,
			hilbert_transform: HilbertTransform::new(sample_rate),
			phase_state: 0.0,
			shift_freq,
			filter: Biquad::new(sample_rate)
		}
	}

	/// Sets the shift frequency of the frequency shifter.
	pub fn set_shift_freq(&mut self, shift_freq: f32) {
		let sample_rate = self.sample_rate as f32;
		let shift_freq = shift_freq.clamp(- sample_rate / 2.0 + MIN_FREQUENCY, sample_rate / 2.0 - MIN_FREQUENCY);

		self.shift_freq = shift_freq;
		if shift_freq > MIN_FREQUENCY {
			self.filter.set_to_butterworth_lowpass(sample_rate / 2.0 - shift_freq);
		}else if shift_freq <= MIN_FREQUENCY && shift_freq > -MIN_FREQUENCY {
			self.filter.set_to_default_state()
		}else {
			self.filter.set_to_butterworth_highpass(shift_freq.abs());
		}
	}
}

impl<const ORDER: usize, const CHANNELS: usize> Effect<CHANNELS> for IIRFreqShifter<ORDER, CHANNELS> {
	fn delay(&self) -> usize {
		self.hilbert_transform.delay()
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"IIR Freq Shifter"
	}

	fn process(&mut self, samples: &mut [f32; CHANNELS], other: &[&[f32; CHANNELS]], ctx: &mut Box<dyn ProcessContext>) {
		if self.shift_freq == 0.0 {
			return;
		}
		self.filter.process(samples, other, ctx);

		let phase_increment = 2.0 * PI * self.shift_freq / self.sample_rate as f32;
		let mut imag_parts = *samples;
		self.hilbert_transform.process(&mut imag_parts, other, ctx);
		let phase_real = self.phase_state.cos();
		let phase_imag = self.phase_state.sin();

		for (i, real_part) in samples.iter_mut().enumerate() {
			*real_part = imag_parts[i] * phase_real - imag_parts[i] * phase_imag;
		}

		self.phase_state = (self.phase_state + phase_increment) % (2.0 * PI);
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, _: String) {
		let mut shift_freq = self.shift_freq;
		ui.add(egui::Slider::new(&mut shift_freq, -20000.0..=20000.0)
			.text("Shift Frequency (Hz)")
		);

		if self.shift_freq != shift_freq {
			self.set_shift_freq(shift_freq);
		}
	}
}