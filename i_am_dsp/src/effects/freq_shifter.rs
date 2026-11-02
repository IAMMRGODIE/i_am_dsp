//! Frequency shifter using Hilbert transform.

use std::f32::consts::PI;

use i_am_dsp_derive::Parameters;

use crate::{
	Effect, ProcessContext,
	effects::{
		filter::{Biquad, MIN_FREQUENCY},
		prelude::{Convolver, hilbert_transform},
	},
	tools::ring_buffer::RingBuffer,
};

use super::hilbert_transform::HilbertTransform as IIRHilbert;

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
	#[unit(hertz)]
	#[default(0.0)]
	#[id(display = "Shift")]
	#[layout(group = "Shift", order = 0, widget = knob, size = large, hero = "spectra", bipolar)]
	shift_freq: f32,
	#[sub_param]
	filter: Biquad<CHANNELS>,
	#[skip]
	history: [RingBuffer<f32>; CHANNELS],
	/// The length of the Hilbert kernel, odd and at most 511.
	#[skip]
	order: usize,
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
				&super::prelude::DelyaCaculateMode::Fir,
			),
			phase_state: 0.0,
			shift_freq,
			filter: Biquad::new(sample_rate),
			history: core::array::from_fn(|_| RingBuffer::new(63)),
			order: 127,
		}
	}

	/// The Hilbert kernel length this shifter runs at.
	pub fn order(&self) -> usize {
		self.order
	}

	/// Rebuilds the Hilbert transform for a new order.
	///
	/// The kernel is a half-band impulse response of odd length, so the order is
	/// rounded up to an odd number. Rebuilding allocates, which is why this is a
	/// setter rather than something the shifter does per block.
	pub fn set_order(&mut self, order: usize) {
		let order = order.clamp(15, 511) | 1;

		if order == self.order {
			return;
		}

		self.order = order;
		self.hilbert_transform.replace_ir(
			hilbert_transform::<CHANNELS>(order),
			&super::prelude::DelyaCaculateMode::Fir,
		);
		self.history = core::array::from_fn(|_| RingBuffer::new(self.hilbert_transform.delay()));
	}

	/// Sets the shift frequency of the frequency shifter.
	pub fn set_shift_freq(&mut self, shift_freq: f32) {
		let sample_rate = self.sample_rate as f32;
		let shift_freq = shift_freq.clamp(
			-sample_rate / 2.0 + MIN_FREQUENCY,
			sample_rate / 2.0 - MIN_FREQUENCY,
		);

		self.shift_freq = shift_freq;
		if shift_freq > MIN_FREQUENCY {
			self.filter
				.set_to_butterworth_lowpass(sample_rate / 2.0 - shift_freq);
		} else if shift_freq <= MIN_FREQUENCY && shift_freq > -MIN_FREQUENCY {
			self.filter.set_to_default_state()
		} else {
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

	fn process(
		&mut self,
		samples: &mut [f32; CHANNELS],
		other: &[&[f32; CHANNELS]],
		ctx: &mut Box<dyn ProcessContext>,
	) {
		if self.shift_freq == 0.0 {
			return;
		}

		self.filter.process(samples, other, ctx);

		let phase_increment = 2.0 * PI * self.shift_freq / self.sample_rate as f32;
		let mut imag_parts = *samples;
		self.hilbert_transform.process(&mut imag_parts, other, ctx);
		let phase_real = self.phase_state.cos();
		let phase_imag = self.phase_state.sin();

		for (i, sample) in samples.iter_mut().enumerate() {
			self.history[i].push(*sample);
			*sample = self.history[i][0] * phase_real - imag_parts[i] * phase_imag;
		}

		self.phase_state = (self.phase_state + phase_increment) % (2.0 * PI);
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, _: String) {
		let mut shift_freq = self.shift_freq;
		ui.add(egui::Slider::new(&mut shift_freq, -20000.0..=20000.0).text("Shift Frequency (Hz)"));

		if self.shift_freq != shift_freq {
			self.set_shift_freq(shift_freq);
		}

		let mut order = self.order;
		ui.add(egui::Slider::new(&mut order, 15..=511).text("Order of Transform"));

		if order != self.order {
			self.set_order(order);
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
	hilbert_transform: IIRHilbert<ORDER, CHANNELS>,
	/// Unit carrier vector (rotated per sample instead of calling cos/sin,
	/// which is a measurable cost at high per-voice counts).
	#[skip]
	phase_real: f32,
	#[skip]
	phase_imag: f32,
	/// Fixed unit rotation per sample.
	#[skip]
	step_c: f32,
	#[skip]
	step_s: f32,
	/// The shift the carrier step was built from, so a parameter write that
	/// bypasses the setter is noticed on the next block.
	#[skip]
	last_shift: f32,
	#[range(min = -20000.0, max = 20000.0)]
	#[unit(hertz)]
	#[default(0.0)]
	#[id(display = "Shift")]
	#[layout(group = "Shift", order = 0, widget = knob, size = large, hero = "spectra", bipolar)]
	shift_freq: f32,
	#[sub_param]
	filter: Biquad<CHANNELS>,
}

impl<const ORDER: usize, const CHANNELS: usize> IIRFreqShifter<ORDER, CHANNELS> {
	/// Creates a new frequency shifter with the given sample rate and shift frequency.
	///
	/// Panics if `CHANNELS` is 0.
	pub fn new(sample_rate: usize, shift_freq: f32) -> Self {
		assert!(CHANNELS > 0, "CHANNELS must be greater than 0");

		let phase_inc = 2.0 * PI * shift_freq / sample_rate as f32;
		Self {
			sample_rate,
			hilbert_transform: IIRHilbert::new(sample_rate),
			phase_real: 1.0,
			phase_imag: 0.0,
			step_c: phase_inc.cos(),
			step_s: phase_inc.sin(),
			last_shift: shift_freq,
			shift_freq,
			filter: Biquad::new(sample_rate),
		}
	}

	/// Sets the shift frequency of the frequency shifter.
	pub fn set_shift_freq(&mut self, shift_freq: f32) {
		let sample_rate = self.sample_rate as f32;
		let shift_freq = shift_freq.clamp(
			-sample_rate / 2.0 + MIN_FREQUENCY,
			sample_rate / 2.0 - MIN_FREQUENCY,
		);

		self.shift_freq = shift_freq;
		self.last_shift = shift_freq;

		let phase_increment = 2.0 * PI * shift_freq / sample_rate;
		self.step_c = phase_increment.cos();
		self.step_s = phase_increment.sin();

		if shift_freq > MIN_FREQUENCY {
			self.filter
				.set_to_butterworth_lowpass(sample_rate / 2.0 - shift_freq);
		} else if shift_freq <= MIN_FREQUENCY && shift_freq > -MIN_FREQUENCY {
			self.filter.set_to_default_state()
		} else {
			self.filter.set_to_butterworth_highpass(shift_freq.abs());
		}
	}
}

impl<const ORDER: usize, const CHANNELS: usize> Effect<CHANNELS>
	for IIRFreqShifter<ORDER, CHANNELS>
{
	fn delay(&self) -> usize {
		0
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"IIR Freq Shifter"
	}

	fn process(
		&mut self,
		samples: &mut [f32; CHANNELS],
		other: &[&[f32; CHANNELS]],
		ctx: &mut Box<dyn ProcessContext>,
	) {
		// A parameter write changes the field, not the cached step; the setter is
		// what rebuilds the carrier and the image-rejection filter.
		if self.shift_freq != self.last_shift {
			self.set_shift_freq(self.shift_freq);
		}

		if self.shift_freq == 0.0 {
			return;
		}
		self.filter.process(samples, other, ctx);

		let complex_samples = self.hilbert_transform.apply_transform(samples);

		let c = self.phase_real;
		let s = self.phase_imag;
		self.phase_real = c * self.step_c - s * self.step_s;
		self.phase_imag = c * self.step_s + s * self.step_c;
		let mag2 = self.phase_real * self.phase_real + self.phase_imag * self.phase_imag;
		if (mag2 - 1.0).abs() > 1e-4 {
			let n = mag2.sqrt();
			self.phase_real /= n;
			self.phase_imag /= n;
		}

		for (i, complex_sample) in complex_samples.iter().enumerate() {
			samples[i] =
				complex_sample.real * self.phase_real - complex_sample.imag * self.phase_imag;
		}
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, _: String) {
		let mut shift_freq = self.shift_freq;
		ui.add(egui::Slider::new(&mut shift_freq, -20000.0..=20000.0).text("Shift Frequency (Hz)"));

		if self.shift_freq != shift_freq {
			self.set_shift_freq(shift_freq);
		}
	}
}
#[cfg(test)]
mod tests {
	use super::*;
	use std::f32::consts::TAU;

	/// The amplitude of one frequency in a block, with a whole number of cycles
	/// in the window when the caller picks the block length.
	fn magnitude(samples: &[f32], frequency: f32, rate: f32) -> f32 {
		let mut real = 0.0;
		let mut imaginary = 0.0;

		for (index, sample) in samples.iter().enumerate() {
			let phase = TAU * frequency * index as f32 / rate;
			real += sample * phase.cos();
			imaginary += sample * phase.sin();
		}

		2.0 * (real * real + imaginary * imaginary).sqrt() / samples.len() as f32
	}

	/// The order decides the Hilbert kernel length, and the delay follows it.
	#[test]
	fn the_fir_order_sets_the_kernel_and_its_delay() {
		let mut shifter = FIRFreqShifter::<2>::new(48_000, 100.0);
		assert_eq!(shifter.order(), 127);
		assert_eq!(shifter.delay(), 63);

		for (order, delay) in [(63, 31), (127, 63), (255, 127), (511, 255)] {
			shifter.set_order(order);
			assert_eq!(shifter.order(), order);
			assert_eq!(shifter.delay(), delay, "order {order}");
		}

		shifter.set_order(300);
		assert_eq!(shifter.order(), 301, "the kernel is always odd");
	}

	/// The parameter map writes the field directly; the carrier has to notice and
	/// rebuild its step, or the shifter stays at zero hertz forever.
	#[test]
	fn the_carrier_follows_a_parameter_write() {
		let mut shifter = IIRFreqShifter::<8, 2>::new(48_000, 0.0);
		shifter.shift_freq = 1_000.0;

		let mut context: Box<dyn ProcessContext> = Box::new(());
		let mut analysed = Vec::new();

		for index in 0..48_000 {
			let phase = TAU * 1_000.0 * index as f32 / 48_000.0;
			let value = 0.5 * phase.sin();
			let mut frame = [value, value];

			shifter.process(&mut frame, &[], &mut context);

			// Let the Hilbert transform and the image filter settle first.
			if index >= 24_000 {
				analysed.push(frame[0]);
			}
		}

		let at_input = magnitude(&analysed, 1_000.0, 48_000.0);
		let shifted = magnitude(&analysed, 2_000.0, 48_000.0);

		assert!(
			shifted > 0.2,
			"the 1 kHz tone should come out at 2 kHz, got {shifted} (input residue {at_input})"
		);
		assert!(
			shifted > at_input * 2.0,
			"the shifted tone should dominate the input residue: {shifted} vs {at_input}"
		);
	}
}
