//! Filter effect implementation

use std::f32::consts::PI;

pub(crate) const MIN_FREQUENCY: f32 = 10.0;

/// The highest cutoff a filter may be designed for, as a fraction of the sample rate.
///
/// At exactly Nyquist the coefficients land on the unit circle and the filter stops
/// being stable: a single allpass section then diverges to NaN in the time domain even
/// though its frequency response still reads as unity. Keeping the cutoff a little
/// below Nyquist is what an allpass stack of a hundred sections needs.
pub(crate) const MAX_FREQUENCY_RATIO: f32 = 0.49;

/// Feedback values below this are flushed to zero.
///
/// Every feedback loop walks through the subnormal range as it decays, and on
/// x86 arithmetic on subnormals is an order of magnitude slower. Anything this
/// small is below -500 dB and inaudible.
const DENORMAL_THRESHOLD: f32 = 1e-25;
use i_am_dsp_derive::Parameters;
use wide::f32x4;

use super::biquad_design as design;
use crate::{
	Effect, ProcessContext,
	prelude::{Parameter, Parameters, Value, from_binary, to_binary},
	tools::ring_buffer::RingBuffer,
};

#[derive(Clone)]
/// A simple biquad filter
pub struct Biquad<const CHANNELS: usize = 2> {
	/// biquad coefficients
	pub b: [f32; 3],
	/// biquad coefficients
	pub a: [f32; 2],
	x: [[f32; CHANNELS]; 2],
	y: [[f32; CHANNELS]; 2],
	sample_rate: usize,

	#[cfg(feature = "real_time_demo")]
	gui_state: GuiState,
}

impl<const CHANNELS: usize> Parameters for Biquad<CHANNELS> {
	fn get_parameters(&self) -> Vec<Parameter> {
		vec![
			Parameter::new(
				"a",
				Value::Serialized(to_binary(&self.a).expect("cannot serialize")),
			),
			Parameter::new(
				"b",
				Value::Serialized(to_binary(&self.b).expect("cannot serialize")),
			),
		]
	}

	fn set_parameter(&mut self, identifier: &str, value: crate::prelude::SetValue) -> bool {
		match identifier {
			"a" => {
				if let crate::prelude::SetValue::Serialized(v) = value {
					let value: [f32; 2] = if let Ok(value) = from_binary(v) {
						value
					} else {
						return false;
					};
					self.a = value;
					return true;
				}
			}
			"b" => {
				if let crate::prelude::SetValue::Serialized(v) = value {
					let value: [f32; 3] = if let Ok(value) = from_binary(v) {
						value
					} else {
						return false;
					};
					self.b = value;
					return true;
				}
			}
			"a1" => {
				if let crate::prelude::SetValue::Float(v) = value {
					self.a[0] = v;
					return true;
				}
			}
			"a2" => {
				if let crate::prelude::SetValue::Float(v) = value {
					self.a[1] = v;
					return true;
				}
			}
			"b0" => {
				if let crate::prelude::SetValue::Float(v) = value {
					self.b[0] = v;
					return true;
				}
			}
			"b1" => {
				if let crate::prelude::SetValue::Float(v) = value {
					self.b[1] = v;
					return true;
				}
			}
			"b2" => {
				if let crate::prelude::SetValue::Float(v) = value {
					self.b[2] = v;
					return true;
				}
			}
			_ => {}
		}
		false
	}
}

#[cfg(feature = "real_time_demo")]
#[derive(Default, Clone)]
struct GuiState {
	filter_type: FilterType,
	allpass_mode: bool,
}

#[cfg(feature = "real_time_demo")]
#[derive(Default, Clone)]
enum FilterType {
	#[default]
	ByPass,
	Butterworth(Butterworth),
	Rbj(RBJFilter),
}

#[cfg(feature = "real_time_demo")]
#[derive(Clone)]
enum Butterworth {
	Lowpass { cutoff: f32 },
	Highpass { cutoff: f32 },
	Bandpass { cutoff: f32 },
	Bandstop { cutoff: f32 },
}

#[cfg(feature = "real_time_demo")]
#[derive(Clone)]
enum RBJFilter {
	Lowpass {
		cutoff: f32,
		q: f32,
	},
	Highpass {
		cutoff: f32,
		q: f32,
	},
	Bandpass {
		cutoff: f32,
		q: f32,
	},
	Bandstop {
		cutoff: f32,
		q: f32,
	},
	Peak {
		gain: f32,
		cutoff: f32,
		bandwidth: f32,
	},
	LowShelf {
		gain: f32,
		cutoff: f32,
		slope: f32,
	},
	HighShelf {
		gain: f32,
		cutoff: f32,
		slope: f32,
	},
}

// butterworth filters
impl<const CHANNELS: usize> Biquad<CHANNELS> {
	/// The default q value for Butterworth filters
	pub const Q1: f32 = 0.707_106_77;

	#[inline(always)]
	fn get_k_and_norm(fs: f32, f0: f32, q: f32) -> (f32, f32) {
		let k = (PI * f0 / fs).tan();
		let norm = 1.0 / (1.0 + k / q + k * k);
		(k, norm)
	}

	#[inline(always)]
	fn butterworth_hp(fs: f32, f0: f32, q: f32) -> ([f32; 3], [f32; 2]) {
		let (k, norm) = Self::get_k_and_norm(fs, f0, q);

		let b0 = 1.0 * norm;
		let b1 = -2.0 * norm;
		let b2 = 1.0 * norm;
		let a1 = 2.0 * (k * k - 1.0) * norm;
		let a2 = (1.0 - k / q + k * k) * norm;

		([b0, b1, b2], [a1, a2])
	}

	#[inline(always)]
	fn butterworth_lp(fs: f32, f0: f32, q: f32) -> ([f32; 3], [f32; 2]) {
		let (k, norm) = Self::get_k_and_norm(fs, f0, q);

		let b0 = k * k * norm;
		let b1 = 2.0 * k * k * norm;
		let b2 = k * k * norm;
		let a1 = 2.0 * (k * k - 1.0) * norm;
		let a2 = (1.0 - k / q + k * k) * norm;

		([b0, b1, b2], [a1, a2])
	}

	#[inline(always)]
	fn butterworth_bp(fs: f32, f0: f32, q: f32) -> ([f32; 3], [f32; 2]) {
		let (k, norm) = Self::get_k_and_norm(fs, f0, q);
		let b0 = k * norm;
		let b1 = 0.0;
		let b2 = -k * norm;
		let a1 = 2.0 * (k * k - 1.0) * norm;
		let a2 = (1.0 - k / q + k * k) * norm;

		([b0, b1, b2], [a1, a2])
	}

	#[inline(always)]
	fn butterworth_bs(fs: f32, f0: f32, q: f32) -> ([f32; 3], [f32; 2]) {
		let (k, norm) = Self::get_k_and_norm(fs, f0, q);
		let b0 = (1.0 + k * k) * norm;
		let b1 = 2.0 * (k * k - 1.0) * norm;
		let b2 = (1.0 + k * k) * norm;
		let a1 = 2.0 * (k * k - 1.0) * norm;
		let a2 = (1.0 - k / q + k * k) * norm;

		([b0, b1, b2], [a1, a2])
	}

	#[inline(always)]
	#[allow(clippy::type_complexity)]
	fn butterworth_filter(
		sample_rate: usize,
		mut cutoff: f32,
		order: usize,
		filter_type: fn(f32, f32, f32) -> ([f32; 3], [f32; 2]),
	) -> Vec<Self> {
		let mut filters = vec![Self::new(sample_rate); order];
		let sample_rate = sample_rate as f32;
		cutoff = cutoff.clamp(MIN_FREQUENCY, sample_rate / 2.0);

		for (k, filter) in filters.iter_mut().enumerate() {
			let q = Self::caculate_butterworth_q_value(k + 1, order);
			let (b, a) = filter_type(sample_rate, cutoff, q);
			filter.b = b;
			filter.a = a;
		}

		filters
	}

	#[inline(always)]
	/// Calculate Q value of a butterworth filter.
	///
	/// Panics if k or order is out of range.
	pub fn caculate_butterworth_q_value(k: usize, order: usize) -> f32 {
		assert!(k <= order && k > 0 && order > 0);
		let denominator = (PI * (2.0 * k as f32 - 1.0) / 2.0 / order as f32).sin() * 2.0;
		1.0 / denominator
	}

	/// Create series connection of butterworth lowpass filter with given order.
	///
	/// Panics if `CHANNELS` is 0.
	pub fn butterworth_lowpass(sample_rate: usize, cutoff: f32, order: usize) -> Vec<Self> {
		Self::butterworth_filter(sample_rate, cutoff, order, Self::butterworth_lp)
	}

	/// Create series connection of butterworth highpass filter with given order.
	///
	/// Panics if `CHANNELS` is 0.
	pub fn butterworth_highpass(sample_rate: usize, cutoff: f32, order: usize) -> Vec<Self> {
		Self::butterworth_filter(sample_rate, cutoff, order, Self::butterworth_hp)
	}

	/// Create series connection of butterworth bandpass filter with given order.
	///
	/// Panics if `CHANNELS` is 0.
	pub fn butterworth_bandpass(sample_rate: usize, cutoff: f32, order: usize) -> Vec<Self> {
		Self::butterworth_filter(sample_rate, cutoff, order, Self::butterworth_bp)
	}

	/// Create series connection of butterworth bandstop filter with given order.
	///
	/// Panics if `CHANNELS` is 0.
	pub fn butterworth_bandstop(sample_rate: usize, cutoff: f32, order: usize) -> Vec<Self> {
		Self::butterworth_filter(sample_rate, cutoff, order, Self::butterworth_bs)
	}

	/// Set the filter to a butterworth lowpass filter with given cutoff frequency(order is fixed to 1).
	pub fn set_to_butterworth_lowpass(&mut self, cutoff: f32) {
		let (b, a) = Self::butterworth_lp(self.sample_rate as f32, cutoff, Self::Q1);
		self.b = b;
		self.a = a;
	}

	/// Set the filter to a butterworth highpass filter with given cutoff frequency(order is fixed to 1).
	pub fn set_to_butterworth_highpass(&mut self, cutoff: f32) {
		let (b, a) = Self::butterworth_hp(self.sample_rate as f32, cutoff, Self::Q1);
		self.b = b;
		self.a = a;
	}

	/// Set the filter to a butterworth bandpass filter with given cutoff frequency(order is fixed to 1).
	pub fn set_to_butterworth_bandpass(&mut self, cutoff: f32) {
		let (b, a) = Self::butterworth_bp(self.sample_rate as f32, cutoff, Self::Q1);
		self.b = b;
		self.a = a;
	}

	/// Set the filter to a butterworth bandstop filter with given cutoff frequency(order is fixed to 1).
	pub fn set_to_butterworth_bandstop(&mut self, cutoff: f32) {
		let (b, a) = Self::butterworth_bs(self.sample_rate as f32, cutoff, Self::Q1);
		self.b = b;
		self.a = a;
	}
}

// RBJ filters
impl<const CHANNELS: usize> Biquad<CHANNELS> {
	/// Set the filter to a RBJ peak filter.
	pub fn set_to_peak(&mut self, cutoff: f32, gain_db: f32, bandwidth: f32) {
		let cutoff = cutoff.clamp(MIN_FREQUENCY, self.sample_rate as f32 / 2.0);
		let sample_rate = self.sample_rate as f32;
		let w0 = design::w0(sample_rate, cutoff);
		let alpha = w0.sin() * (PI * bandwidth / sample_rate).sinh();

		(self.b, self.a) = design::bell(w0, alpha, gain_db);
	}

	/// Set the filter to a RBJ low shelf filter.
	pub fn set_to_low_shelf(&mut self, cutoff: f32, gain_db: f32, slope_param: f32) {
		let cutoff = cutoff.clamp(MIN_FREQUENCY, self.sample_rate as f32 / 2.0);
		let sample_rate = self.sample_rate as f32;
		// RBJ uses A = 10^(dB/40), and its slope S is at most 1: above that the
		// square root below goes negative for a large gain and the coefficients
		// become NaN.
		let a = 10.0_f32.powf(gain_db / 40.0);
		let slope_param = slope_param.clamp(0.1, 1.0);
		let w0 = design::w0(sample_rate, cutoff);
		let alpha = w0.sin() / 2.0 * ((a + 1.0 / a) * (1.0 / slope_param - 1.0) + 2.0).sqrt();

		(self.b, self.a) = design::low_shelf(w0, alpha, gain_db);
	}

	/// Set the filter to a RBJ high shelf filter.
	pub fn set_to_high_shelf(&mut self, cutoff: f32, gain_db: f32, slope_param: f32) {
		let cutoff = cutoff.clamp(MIN_FREQUENCY, self.sample_rate as f32 / 2.0);
		let sample_rate = self.sample_rate as f32;
		// Same RBJ scaling and slope limit as the low shelf.
		let a = 10.0_f32.powf(gain_db / 40.0);
		let slope_param = slope_param.clamp(0.1, 1.0);
		let w0 = design::w0(sample_rate, cutoff);
		let alpha = w0.sin() / 2.0 * ((a + 1.0 / a) * (1.0 / slope_param - 1.0) + 2.0).sqrt();

		(self.b, self.a) = design::high_shelf(w0, alpha, gain_db);
	}

	/// Set the filter to a RBJ bandpass filter.
	pub fn set_to_bandpass(&mut self, cutoff: f32, bandwidth: f32) {
		let cutoff = cutoff.clamp(MIN_FREQUENCY, self.sample_rate as f32 * MAX_FREQUENCY_RATIO);
		let w0 = design::w0(self.sample_rate as f32, cutoff);
		let q = cutoff / bandwidth;
		let alpha = design::alpha_for_q(w0, q);

		(self.b, self.a) = design::band_pass(w0, alpha);
	}

	/// Set the filter to a RBJ bandstop filter.
	pub fn set_to_bandstop(&mut self, cutoff: f32, bandwidth: f32) {
		let cutoff = cutoff.clamp(MIN_FREQUENCY, self.sample_rate as f32 * MAX_FREQUENCY_RATIO);
		let w0 = design::w0(self.sample_rate as f32, cutoff);
		let q = cutoff / bandwidth;
		let alpha = design::alpha_for_q(w0, q);

		(self.b, self.a) = design::band_stop(w0, alpha);
	}

	/// Set the filter to a RBJ lowpass filter. You can use [`Self::Q1`] as a starting point for the Q value.
	pub fn set_to_lowpass(&mut self, cutoff: f32, q: f32) {
		let cutoff = cutoff.clamp(MIN_FREQUENCY, self.sample_rate as f32 / 2.0);
		let q = q.max(0.01);
		let w0 = design::w0(self.sample_rate as f32, cutoff);
		let alpha = design::alpha_for_q(w0, q);

		(self.b, self.a) = design::low_pass(w0, alpha);
	}

	/// Set the filter to a RBJ highpass filter. You can use [`Self::Q1`] as a starting point for the Q value.
	pub fn set_to_highpass(&mut self, cutoff: f32, q: f32) {
		let cutoff = cutoff.clamp(MIN_FREQUENCY, self.sample_rate as f32 / 2.0);
		let q = q.max(0.01);
		let w0 = design::w0(self.sample_rate as f32, cutoff);
		let alpha = design::alpha_for_q(w0, q);

		(self.b, self.a) = design::high_pass(w0, alpha);
	}

	/// Create a RBJ peak filter with given parameters.
	///
	/// Panics if `CHANNELS` is 0
	pub fn peak(sample_rate: usize, cutoff: f32, gain_db: f32, bandwidth: f32) -> Self {
		let mut filter = Self::new(sample_rate);
		filter.set_to_peak(cutoff, gain_db, bandwidth);
		filter
	}

	/// Create a RBJ low shelf filter with given parameters.
	///
	/// Panics if `CHANNELS` is 0
	pub fn low_shelf(sample_rate: usize, cutoff: f32, gain_db: f32, slope_param: f32) -> Self {
		let mut filter = Self::new(sample_rate);
		filter.set_to_low_shelf(cutoff, gain_db, slope_param);
		filter
	}

	/// Create a RBJ high shelf filter with given parameters.
	pub fn high_shelf(sample_rate: usize, cutoff: f32, gain_db: f32, slope_param: f32) -> Self {
		let mut filter = Self::new(sample_rate);
		filter.set_to_high_shelf(cutoff, gain_db, slope_param);
		filter
	}

	/// Create a RBJ bandpass filter with given parameters.
	///
	/// Panics if `CHANNELS` is 0
	pub fn bandpass(sample_rate: usize, cutoff: f32, bandwidth: f32) -> Self {
		let mut filter = Self::new(sample_rate);
		filter.set_to_bandpass(cutoff, bandwidth);
		filter
	}

	/// Create a RBJ bandstop filter with given parameters.
	///
	/// Panics if `CHANNELS` is 0
	pub fn bandstop(sample_rate: usize, cutoff: f32, bandwidth: f32) -> Self {
		let mut filter = Self::new(sample_rate);
		filter.set_to_bandstop(cutoff, bandwidth);
		filter
	}

	/// Create a RBJ lowpass filter with given parameters.
	///
	/// Panics if `CHANNELS` is 0
	pub fn lowpass(sample_rate: usize, cutoff: f32, q: f32) -> Self {
		let mut filter = Self::new(sample_rate);
		filter.set_to_lowpass(cutoff, q);
		filter
	}

	/// Create a RBJ highpass filter with given parameters.
	///
	/// Panics if `CHANNELS` is 0
	pub fn highpass(sample_rate: usize, cutoff: f32, q: f32) -> Self {
		let mut filter = Self::new(sample_rate);
		filter.set_to_highpass(cutoff, q);
		filter
	}
}

impl<const CHANNELS: usize> Biquad<CHANNELS> {
	/// Create a new instance of the Biquad filter.
	///
	/// Panics the program if `CHANNELS` is 0.
	pub const fn new(sample_rate: usize) -> Self {
		assert!(CHANNELS > 0, "CHANNELS must be greater than 0");
		Self {
			b: [1.0, 0.0, 0.0],
			a: [0.0, 0.0],
			x: [[0.0; CHANNELS], [0.0; CHANNELS]],
			y: [[0.0; CHANNELS], [0.0; CHANNELS]],
			sample_rate,

			#[cfg(feature = "real_time_demo")]
			gui_state: GuiState {
				filter_type: FilterType::ByPass,
				allpass_mode: false,
			},
		}
	}

	/// Calculate the amplitude response of the filter.
	///
	/// Returns original value rather than dB value.
	pub fn amplitude_response(&self, frequency: f32) -> f32 {
		let frequency = 2.0 * PI * frequency / self.sample_rate as f32;

		let cos_f = frequency.cos();
		let sin_f = frequency.sin();
		let cos_2f = 2.0 * cos_f * cos_f - 1.0;
		let sin_2f = 2.0 * cos_f * sin_f;

		let n = (self.b[0] + self.b[1] * cos_f + self.b[2] * cos_2f)
			.hypot(self.b[1] * sin_f + self.b[2] * sin_2f);

		let d = (1.0 + self.a[0] * cos_f + self.a[1] * cos_2f)
			.hypot(self.a[0] * sin_f + self.a[1] * sin_2f);

		n / d
	}

	/// Calculate the phase response of the filter.
	///
	/// Returns phase in radians.
	pub fn phase_response(&self, frequency: f32) -> f32 {
		let frequency = 2.0 * PI * frequency / self.sample_rate as f32;

		let cos_f = frequency.cos();
		let sin_f = frequency.sin();
		let cos_2f = 2.0 * cos_f * cos_f - 1.0;
		let sin_2f = 2.0 * cos_f * sin_f;

		let real_n = self.b[0] + self.b[1] * cos_f + self.b[2] * cos_2f;
		let imag_n = -(self.b[1] * sin_f + self.b[2] * sin_2f);

		let real_d = 1.0 + self.a[0] * cos_f + self.a[1] * cos_2f;
		let imag_d = -(self.a[0] * sin_f + self.a[1] * sin_2f);

		(imag_n.atan2(real_n) - imag_d.atan2(real_d)).rem_euclid(2.0 * PI)
	}

	/// Calculate the complex response of the filter.
	///
	/// Returns a tuple of the amplitude and phase in radians.
	pub fn complex_response(&self, frequency: f32) -> (f32, f32) {
		let frequency = 2.0 * PI * frequency / self.sample_rate as f32;

		let cos_f = frequency.cos();
		let sin_f = frequency.sin();
		let cos_2f = 2.0 * cos_f * cos_f - 1.0;
		let sin_2f = 2.0 * cos_f * sin_f;

		let real_n = self.b[0] + self.b[1] * cos_f + self.b[2] * cos_2f;
		let imag_n = -(self.b[1] * sin_f + self.b[2] * sin_2f);

		let real_d = 1.0 + self.a[0] * cos_f + self.a[1] * cos_2f;
		let imag_d = -(self.a[0] * sin_f + self.a[1] * sin_2f);

		let amplitude = real_n.hypot(imag_n) / real_d.hypot(imag_d);
		let phase = (imag_n.atan2(real_n) - imag_d.atan2(real_d)).rem_euclid(2.0 * PI);

		(amplitude, phase)
	}

	/// Set the filter to it's complementary filter.
	pub fn set_to_complementary_filter(&mut self) {
		self.b[0] = 1.0 - self.b[0];
		self.b[1] = self.a[0] - self.b[1];
		self.b[2] = self.a[1] - self.b[2];
	}

	/// Get the filter's complementary filter.
	pub fn complementary_filter(&self) -> Self {
		let mut filter = Self::new(self.sample_rate);
		filter.b[0] = 1.0 - self.b[0];
		filter.b[1] = self.a[0] - self.b[1];
		filter.b[2] = self.a[1] - self.b[2];
		filter.a = self.a;
		filter
	}

	/// Transform the filter to a allpass filter, in most cases....
	///
	/// It's equivalent to the transform $Y\[n\] = 1 - 2 Filter(X\[n\])$
	pub fn transform_to_allpass(&mut self) {
		self.b[0] = 2.0 * self.b[0] - 1.0;
		self.b[1] = 2.0 * self.b[1] - self.a[0];
		self.b[2] = 2.0 * self.b[2] - self.a[1];
	}

	/// Set the filter to default state.
	pub fn set_to_default_state(&mut self) {
		self.b = [1.0, 0.0, 0.0];
		self.a = [0.0, 0.0];
	}

	/// Get the filter's sample rate.
	pub fn sample_rate(&self) -> usize {
		self.sample_rate
	}

	/// Clear the filter's state.
	pub fn clear_state(&mut self) {
		self.x = [[0.0; CHANNELS], [0.0; CHANNELS]];
		self.y = [[0.0; CHANNELS], [0.0; CHANNELS]];
	}

	/// Process one frame with the direct-form I recurrence.
	///
	/// The effect's process loop uses SIMD; a bank of many small filters, like
	/// the vocoder's, calls this one frame at a time instead.
	pub fn process_sample(&mut self, input: [f32; CHANNELS]) -> [f32; CHANNELS] {
		let mut output = [0.0; CHANNELS];

		for channel in 0..CHANNELS {
			let x = input[channel];
			let y = self.b[0] * x + self.b[1] * self.x[0][channel] + self.b[2] * self.x[1][channel]
				- self.a[0] * self.y[0][channel]
				- self.a[1] * self.y[1][channel];

			self.x[1][channel] = self.x[0][channel];
			self.x[0][channel] = x;
			self.y[1][channel] = self.y[0][channel];
			self.y[0][channel] = y;
			output[channel] = y;
		}

		output
	}
}

mod comb;
mod process;
mod simple;

pub use comb::{AllpassFilter, CombsFilter};
pub use simple::{Bandpass, Bandstop, HighShelf, Highpass, LowShelf, Lowpass, Peak};

#[cfg(test)]
mod tests;
