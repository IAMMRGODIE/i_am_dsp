//! Filter effect implementation

use std::f32::consts::PI;

pub(crate) const MIN_FREQUENCY: f32 = 10.0;

use crate::{Effect, ProcessContext};

#[derive(Clone)]
/// A simple biquad filter
pub struct Biquad<const CHANNELS: usize = 2> {
	b: [f32; 3],
	a: [f32; 2],
	x: [[f32; 2]; CHANNELS],
	y: [[f32; 2]; CHANNELS],
	sample_rate: usize,

	#[cfg(feature = "real_time_demo")]
	gui_state: GuiState,
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
	#[default] ByPass,
	Butterworth(Butterworth),
	Rbj(RBJFilter),
}

#[cfg(feature = "real_time_demo")]
#[derive(Clone)]
enum Butterworth {
	Lowpass {
		cutoff: f32,
	},
	Highpass {
		cutoff: f32,
	},
	Bandpass {
		cutoff: f32,
	},
	Bandstop {
		cutoff: f32,
	},
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
	}
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

		let b0 =  k * k * norm;
		let b1 =  2.0 * k * k * norm;
		let b2 =  k * k * norm;
		let a1 =  2.0 * (k * k - 1.0) * norm;
		let a2 =  (1.0 - k / q + k * k) * norm;

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
		filter_type: fn(f32, f32, f32) -> ([f32; 3], [f32; 2])
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
	pub fn set_to_peak(
		&mut self,
		cutoff: f32, 
		gain_db: f32, 
		bandwidth: f32, 
	) {
		let cutoff = cutoff.clamp(MIN_FREQUENCY, self.sample_rate as f32 / 2.0);
		let sample_rate = self.sample_rate as f32;
		let w0 = 2.0 * PI * cutoff / sample_rate;
		let alpha = w0.sin() * (PI * bandwidth / sample_rate).sinh();
		let a = 10.0_f32.powf(gain_db / 20.0);

		let b0 = 1.0 + alpha * a;
		let b1 = -2.0 * w0.cos();
		let b2 = 1.0 - alpha * a;
		let a0 = 1.0 + alpha / a;
		let a1 = -2.0 * w0.cos();
		let a2 = 1.0 - alpha / a;

		let b = [b0 / a0, b1 / a0, b2 / a0];
		let a = [a1 / a0, a2 / a0];

		self.b = b;
		self.a = a;
	}

	/// Set the filter to a RBJ low shelf filter.
	pub fn set_to_low_shelf(
		&mut self,
		cutoff: f32, 
		gain_db: f32, 
		slope_param: f32, 
	) {
		let cutoff = cutoff.clamp(MIN_FREQUENCY, self.sample_rate as f32 / 2.0);
		let sample_rate = self.sample_rate as f32;
		let a = 10.0_f32.powf(gain_db / 20.0);
		let w0 = 2.0 * PI * cutoff / sample_rate;
		let alpha = w0.sin() / 2.0 * ((a + 1.0 / a) * (1.0 / slope_param - 1.0) + 2.0).sqrt();

		let b0 = a * ((a + 1.0) - (a - 1.0) * w0.cos() + 2.0 * a.sqrt() * alpha);
		let b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * w0.cos());
		let b2 = a * ((a + 1.0) - (a - 1.0) * w0.cos() - 2.0 * a.sqrt() * alpha);
		let a0 = (a + 1.0) + (a - 1.0) * w0.cos() + 2.0 * a.sqrt() * alpha;
		let a1 = -2.0 * ((a - 1.0) + (a + 1.0) * w0.cos());
		let a2 = (a + 1.0) + (a - 1.0) * w0.cos() - 2.0 * a.sqrt() * alpha;
		
		let b = [b0 / a0, b1 / a0, b2 / a0];
		let a = [a1 / a0, a2 / a0];

		self.b = b;
		self.a = a;
	}

	/// Set the filter to a RBJ high shelf filter.
	pub fn set_to_high_shelf(
		&mut self,
		cutoff: f32, 
		gain_db: f32, 
		slope_param: f32, 
	) {
		let cutoff = cutoff.clamp(MIN_FREQUENCY, self.sample_rate as f32 / 2.0);
		let sample_rate = self.sample_rate as f32;
		let a = 10.0_f32.powf(gain_db / 20.0);
		let w0 = 2.0 * PI * cutoff / sample_rate;
		let alpha = w0.sin() / 2.0 * ((a + 1.0 / a) * (1.0 / slope_param - 1.0) + 2.0).sqrt();
		
		let b0 = a * ((a + 1.0) + (a - 1.0) * w0.cos() + 2.0 * a.sqrt() * alpha);
		let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * w0.cos());
		let b2 = a * ((a + 1.0) + (a - 1.0) * w0.cos() - 2.0 * a.sqrt() * alpha);
		let a0 = (a + 1.0) - (a - 1.0) * w0.cos() + 2.0 * a.sqrt() * alpha;
		let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * w0.cos());
		let a2 = (a + 1.0) - (a - 1.0) * w0.cos() - 2.0 * a.sqrt() * alpha;
		
		let b = [b0 / a0, b1 / a0, b2 / a0];
		let a = [a1 / a0, a2 / a0];

		self.b = b;
		self.a = a;
	}

	/// Set the filter to a RBJ bandpass filter.
	pub fn set_to_bandpass(
		&mut self,
		cutoff: f32, 
		bandwidth: f32,
	) {
		let omega_0 = 2.0 * PI * cutoff / self.sample_rate as f32;
		let q = cutoff / bandwidth;
		let alpha = omega_0.sin() / (2.0 * q);

		let b0 = alpha;
		let b1 = 0.0;
		let b2 = -alpha;
		let a0 = 1.0 + alpha;
		let a1 = -2.0 * omega_0.cos();
		let a2 = 1.0 - alpha;

		self.b = [b0 / a0, b1 / a0, b2 / a0];
		self.a = [a1 / a0, a2 / a0];
	}

	/// Set the filter to a RBJ bandstop filter.
	pub fn set_to_bandstop(
		&mut self,
		cutoff: f32, 
		bandwidth: f32,
	) {
		let omega_0 = 2.0 * PI * cutoff / self.sample_rate as f32;
		let q = cutoff / bandwidth;
		let alpha = omega_0.sin() / (2.0 * q);

		let b0 = 1.0;
		let b1 = - 2.0 * omega_0.cos();
		let b2 = 1.0;
		let a0 = 1.0 + alpha;
		let a1 = -2.0 * omega_0.cos();
		let a2 = 1.0 - alpha;

		self.b = [b0 / a0, b1 / a0, b2 / a0];
		self.a = [a1 / a0, a2 / a0];
	}

	/// Set the filter to a RBJ lowpass filter. You can use [`Self::Q1`] as a starting point for the Q value.
	pub fn set_to_lowpass(
		&mut self,
		cutoff: f32,
		q: f32,
	) {
		let omega_0 = 2.0 * PI * cutoff / self.sample_rate as f32;
		let alpha = omega_0.sin() / (2.0 * q);

		let b0 = (1.0 - omega_0.cos()) / 2.0;
		let b1 = b0 * 2.0;
		let b2 = b0;
		let a0 = 1.0 + alpha;
		let a1 = - 2.0 * omega_0.cos();
		let a2 = 1.0 - alpha;

		self.b = [b0 / a0, b1 / a0, b2 / a0];
		self.a = [a1 / a0, a2 / a0];
	}

	/// Set the filter to a RBJ highpass filter. You can use [`Self::Q1`] as a starting point for the Q value.
	pub fn set_to_highpass(
		&mut self,
		cutoff: f32,
		q: f32,
	) {
		let omega_0 = 2.0 * PI * cutoff / self.sample_rate as f32;
		let alpha = omega_0.sin() / (2.0 * q);

		let b0 = (1.0 + omega_0.cos()) / 2.0;
		let b1 = - b0 * 2.0;
		let b2 = b0;
		let a0 = 1.0 + alpha;
		let a1 = - 2.0 * omega_0.cos();
		let a2 = 1.0 - alpha;

		self.b = [b0 / a0, b1 / a0, b2 / a0];
		self.a = [a1 / a0, a2 / a0];
	}

	/// Create a RBJ peak filter with given parameters.
	/// 
	/// Panics if `CHANNELS` is 0
	pub fn peak(
		sample_rate: usize, 
		cutoff: f32, 
		gain_db: f32, 
		bandwidth: f32, 
	) -> Self {
		let mut filter = Self::new(sample_rate);
		filter.set_to_peak(cutoff, gain_db, bandwidth);
		filter
	}

	/// Create a RBJ low shelf filter with given parameters.
	/// 
	/// Panics if `CHANNELS` is 0
	pub fn low_shelf(
		sample_rate: usize, 
		cutoff: f32, 
		gain_db: f32, 
		slope_param: f32, 
	) -> Self {
		let mut filter = Self::new(sample_rate);
		filter.set_to_low_shelf(cutoff, gain_db, slope_param);
		filter
	}

	/// Create a RBJ high shelf filter with given parameters.
	pub fn high_shelf(
		sample_rate: usize, 
		cutoff: f32, 
		gain_db: f32, 
		slope_param: f32, 
	) -> Self {
		let mut filter = Self::new(sample_rate);
		filter.set_to_high_shelf(cutoff, gain_db, slope_param);
		filter
	}
	
	/// Create a RBJ bandpass filter with given parameters.
	/// 
	/// Panics if `CHANNELS` is 0
	pub fn bandpass(
		sample_rate: usize, 
		cutoff: f32, 
		bandwidth: f32,
	) -> Self {
		let mut filter = Self::new(sample_rate);
		filter.set_to_bandpass(cutoff, bandwidth);
		filter
	}

	/// Create a RBJ bandstop filter with given parameters.
	/// 
	/// Panics if `CHANNELS` is 0
	pub fn bandstop(
		sample_rate: usize, 
		cutoff: f32, 
		bandwidth: f32,
	) -> Self {
		let mut filter = Self::new(sample_rate);
		filter.set_to_bandstop(cutoff, bandwidth);
		filter
	}

	/// Create a RBJ lowpass filter with given parameters.
	/// 
	/// Panics if `CHANNELS` is 0
	pub fn lowpass(
		sample_rate: usize, 
		cutoff: f32, 
		q: f32,
	) -> Self {
		let mut filter = Self::new(sample_rate);
		filter.set_to_lowpass(cutoff, q);
		filter
	}

	/// Create a RBJ highpass filter with given parameters.
	/// 
	/// Panics if `CHANNELS` is 0
	pub fn highpass(
		sample_rate: usize, 
		cutoff: f32, 
		q: f32,
	) -> Self {
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
			x: [[0.0, 0.0]; CHANNELS],
			y: [[0.0, 0.0]; CHANNELS],
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
		let imag_n = - (self.b[1] * sin_f + self.b[2] * sin_2f);
		
		let real_d = 1.0 + self.a[0] * cos_f + self.a[1] * cos_2f;
		let imag_d = - (self.a[0] * sin_f + self.a[1] * sin_2f);

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
		let imag_n = - (self.b[1] * sin_f + self.b[2] * sin_2f);
		
		let real_d = 1.0 + self.a[0] * cos_f + self.a[1] * cos_2f;
		let imag_d = - (self.a[0] * sin_f + self.a[1] * sin_2f);

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
}

impl<const CHANNELS: usize> Effect<CHANNELS> for Biquad<CHANNELS> {
	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"Biquad Filter"
	}

	fn delay(&self) -> usize {
		0
	}

	fn process(&mut self, samples: &mut [f32; CHANNELS], _: &[&[f32; CHANNELS]], _: &mut Box<dyn ProcessContext>) {
		// TODO: implement SIMD version of this filter
		for (i, sample) in samples.iter_mut().enumerate() {
			let output = 
			self.b[0] * *sample + 
			self.b[1] * self.x[i][0] + 
				self.b[2] * self.x[i][1] - 
				self.a[0] * self.y[i][0] - 
				self.a[1] * self.y[i][1];
			
			let output = if output.is_nan() { 0.0 } else { output };

			self.x[i][1] = self.x[i][0];
			self.x[i][0] = *sample;
			self.y[i][1] = self.y[i][0];
			self.y[i][0] = output;

			*sample = output;
		}
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		use egui::*;
		use crate::tools::ui_tools::draw_complex_response;
		use std::ops::RangeInclusive;

		type ChangeFunc<const CHANNELS: usize> = Box<dyn Fn(&mut Biquad<CHANNELS>, f32)>;
		type ChangeFunc2<const CHANNELS: usize> = Box<dyn Fn(&mut Biquad<CHANNELS>, f32, f32)>;
		type ChangeFunc3<const CHANNELS: usize> = Box<dyn Fn(&mut Biquad<CHANNELS>, f32, f32, f32)>;

		fn cutoff_ui<const CHANNELS: usize>(
			ui: &mut egui::Ui, 
			cutoff: &mut f32, 
			max_freq: f32,
			change_func: ChangeFunc<CHANNELS>
		) -> Option<(ChangeFunc<CHANNELS>, f32)> {
			let mut cutoff_backup = *cutoff;
			let slider = Slider::new(&mut cutoff_backup, 10.0..=max_freq - MIN_FREQUENCY)
				.text("Cutoff")
				.logarithmic(true);
			ui.add(slider);
			if cutoff_backup != *cutoff {
				*cutoff = cutoff_backup;
				Some((change_func, *cutoff))
			} else {
				None
			}
		}

		fn cutoff_q_ui<const CHANNELS: usize>(
			ui: &mut egui::Ui, 
			cutoff: &mut f32,
			q: &mut f32,
			q_range: RangeInclusive<f32>,
			q_text: impl Into<String>,
			max_freq: f32,
			change_func: ChangeFunc2<CHANNELS>
		) -> Option<(ChangeFunc2<CHANNELS>, f32, f32)> {
			let mut cutoff_backup = *cutoff;
			let slider = Slider::new(&mut cutoff_backup, 10.0..=max_freq - MIN_FREQUENCY)
				.text("Cutoff")
				.logarithmic(true);
			ui.add(slider);
			let mut q_backup = *q;
			let q_log = *q_range.end() > 1000.0;
			let q_slider = Slider::new(&mut q_backup, q_range)
				.text(q_text.into())
				.logarithmic(q_log);
			ui.add(q_slider);

			if cutoff_backup != *cutoff || q_backup != *q {
				*cutoff = cutoff_backup;
				*q = q_backup;
				Some((change_func, *cutoff, *q))
			} else {
				None
			}
		}

		fn simulate_ui<const CHANNELS: usize>(
			ui: &mut egui::Ui, 
			params: (&mut f32, &mut f32, &mut f32),
			max_freq: f32,
			factor_range: RangeInclusive<f32>,
			mut texts: Vec<impl Into<String>>,
			change_func: ChangeFunc3<CHANNELS>,
		) -> Option<(ChangeFunc3<CHANNELS>, f32, f32, f32)> {
			let (cutoff, gain, factor) = params;

			let mut cutoff_backup = *cutoff;
			let mut gain_backup = *gain;
			let mut factor_backup = *factor;
			let factor_text = texts.pop().unwrap().into();
			let gain_text = texts.pop().unwrap().into();
			let cutoff_text = texts.pop().unwrap().into();

			let slider_cutoff = Slider::new(&mut cutoff_backup, 10.0..=max_freq - MIN_FREQUENCY)
				.text(cutoff_text)
				.logarithmic(true);
			let slider_gain = Slider::new(&mut gain_backup, -12.0..=6.0)
				.text(gain_text);
			let factor_log = *factor_range.end() > 1000.0;
			let slider_factor = Slider::new(&mut factor_backup, factor_range)
				.text(factor_text)
				.logarithmic(factor_log);

			ui.add(slider_cutoff);
			ui.add(slider_gain);
			ui.add(slider_factor);

			if cutoff_backup != *cutoff || gain_backup != *gain || factor_backup != *factor {
				*cutoff = cutoff_backup;
				*gain = gain_backup;
				*factor = factor_backup;
				Some((change_func, *cutoff, *gain, *factor))
			} else {
				None
			}
		}

		egui::Resize::default().resizable([false, true])
			.min_width(ui.available_width())
			.max_width(ui.available_width())
			.id_salt(format!("{id_prefix}_filter"))
			.show(ui, |ui| 
		{
			draw_complex_response(ui, self.sample_rate, |freq| self.complex_response(freq));
		});

		ScrollArea::horizontal().show(ui, |ui| {
			ui.horizontal(|ui| {
				if ui.selectable_label(matches!(self.gui_state.filter_type, FilterType::ByPass), "Bypass").clicked() {
					self.gui_state.filter_type = FilterType::ByPass;
					self.b = [1.0, 0.0, 0.0];
					self.a = [0.0, 0.0];
				}
				if ui.selectable_label(matches!(self.gui_state.filter_type, FilterType::Butterworth(_)), "Butterworth").clicked() {
					self.gui_state.filter_type = FilterType::Butterworth(Butterworth::Lowpass { cutoff: 1000.0 });
					self.set_to_butterworth_lowpass(1000.0);
				}

				if ui.selectable_label(matches!(self.gui_state.filter_type, FilterType::Rbj(_)), "Rbj").clicked() {
					self.gui_state.filter_type = FilterType::Rbj(RBJFilter::Lowpass { cutoff: 1000.0 , q: Self::Q1 });
					self.set_to_lowpass(1000.0, Self::Q1);
				}

				if ui.selectable_label(self.gui_state.allpass_mode, "Allpass").clicked() {
					self.gui_state.allpass_mode = !self.gui_state.allpass_mode;
					if self.gui_state.allpass_mode {
						self.transform_to_allpass();
					}else {
						self.gui_state.filter_type = FilterType::ByPass;
						self.b = [1.0, 0.0, 0.0];
						self.a = [0.0, 0.0];
					}
				}
			});

			fn butterworth_gui<const CHANNELS: usize>(
				ui: &mut egui::Ui, 
				butterworth: &mut Butterworth,
				sample_rate: usize
			) -> Option<Biquad<CHANNELS>> {
				let mut output = None;
				if ui.selectable_label(matches!(butterworth, Butterworth::Lowpass { .. }), "Lowpass").clicked() {
					*butterworth = Butterworth::Lowpass { cutoff: 1000.0 };
					output = Some(Biquad::butterworth_lowpass(sample_rate, 1000.0, 1).pop().unwrap())
				}
				if ui.selectable_label(matches!(butterworth, Butterworth::Highpass { .. }), "Highpass").clicked() {
					*butterworth = Butterworth::Highpass { cutoff: 1000.0 };
					output = Some(Biquad::butterworth_highpass(sample_rate, 1000.0, 1).pop().unwrap())
				}
				if ui.selectable_label(matches!(butterworth, Butterworth::Bandpass { .. }), "Bandpass").clicked() {
					*butterworth = Butterworth::Bandpass { cutoff: 1000.0 };
					output = Some(Biquad::butterworth_bandpass(sample_rate, 1000.0, 1).pop().unwrap())
				}
				if ui.selectable_label(matches!(butterworth, Butterworth::Bandstop { .. }), "Bandstop").clicked() {
					*butterworth = Butterworth::Bandstop { cutoff: 1000.0 };
					output = Some(Biquad::butterworth_bandstop(sample_rate, 1000.0, 1).pop().unwrap())
				}

				output
			}

			fn rbj_gui<const CHANNELS: usize>(
				ui: &mut egui::Ui, 
				rbj: &mut RBJFilter,
				sample_rate: usize
			) -> Option<Biquad<CHANNELS>> {
				let mut output = None;
				if ui.selectable_label(matches!(rbj, RBJFilter::Lowpass { .. }), "Lowpass").clicked() {
					*rbj = RBJFilter::Lowpass { cutoff: 1000.0, q: Biquad::<CHANNELS>::Q1 };
					output = Some(Biquad::lowpass(sample_rate, 1000.0, Biquad::<CHANNELS>::Q1))
				}
				if ui.selectable_label(matches!(rbj, RBJFilter::Highpass { .. }), "Highpass").clicked() {
					*rbj = RBJFilter::Highpass { cutoff: 1000.0, q: Biquad::<CHANNELS>::Q1 };
					output = Some(Biquad::highpass(sample_rate, 1000.0, Biquad::<CHANNELS>::Q1))
				}
				if ui.selectable_label(matches!(rbj, RBJFilter::Bandpass { .. }), "Bandpass").clicked() {
					*rbj = RBJFilter::Bandpass { cutoff: 1000.0, q: Biquad::<CHANNELS>::Q1 };
					output = Some(Biquad::bandpass(sample_rate, 1000.0, 100.0))
				}
				if ui.selectable_label(matches!(rbj, RBJFilter::Bandstop { .. }), "Bandstop").clicked() {
					*rbj = RBJFilter::Bandstop { cutoff: 1000.0, q: Biquad::<CHANNELS>::Q1 };
					output = Some(Biquad::bandstop(sample_rate, 1000.0, 100.0))
				}
				if ui.selectable_label(matches!(rbj, RBJFilter::Peak { .. }), "Peak").clicked() {
					*rbj = RBJFilter::Peak { cutoff: 1000.0, gain: -3.01, bandwidth: 100.0 };
					output = Some(Biquad::peak(sample_rate, 1000.0, -3.01, 100.0))
				}
				if ui.selectable_label(matches!(rbj, RBJFilter::LowShelf { .. }), "LowShelf").clicked() {
					*rbj = RBJFilter::LowShelf { cutoff: 1000.0, gain: -3.01, slope: 1.5 };
					output = Some(Biquad::low_shelf(sample_rate, 1000.0, -3.01, 1.5))
				}
				if ui.selectable_label(matches!(rbj, RBJFilter::HighShelf { .. }), "HighShelf").clicked() {
					*rbj = RBJFilter::HighShelf { cutoff: 1000.0, gain: -3.01, slope: 1.5 };
					output = Some(Biquad::high_shelf(sample_rate, 1000.0, -3.01, 1.5))
				}

				output
			}

			let mut need_change: Option<Biquad<CHANNELS>> = None;

			ui.horizontal(|ui| {
				match &mut self.gui_state.filter_type {
					FilterType::Butterworth(inner) => {
						need_change = need_change.take().or(butterworth_gui(ui, inner, self.sample_rate));
					}
					FilterType::Rbj(inner) => {
						need_change = need_change.take().or(rbj_gui(ui, inner, self.sample_rate))
					}
					FilterType::ByPass => {},
				}
			});

			if let Some(biquad) = need_change {
				self.a = biquad.a;
				self.b = biquad.b;
				if self.gui_state.allpass_mode {
					self.transform_to_allpass();
				}
			}

			let sample_rate = self.sample_rate as f32;
			let max_freq = sample_rate / 2.0;

			let mut change_cutoff: Option<(ChangeFunc<CHANNELS>, f32)> = None;
			let mut change_cutoff_q: Option<(ChangeFunc2<CHANNELS>, f32, f32)> = None;
			let mut change_simulated: Option<(ChangeFunc3<CHANNELS>, f32, f32, f32)> = None;

			match &mut self.gui_state.filter_type {
				FilterType::ByPass => {
					ui.label("Bypassed");
				},
				FilterType::Butterworth(Butterworth::Lowpass { cutoff }) => {
					change_cutoff = change_cutoff.take().or(cutoff_ui(
						ui, 
						cutoff, 
						max_freq, 
						Box::new(Self::set_to_butterworth_lowpass)
					));
				},
				FilterType::Butterworth(Butterworth::Highpass { cutoff }) => {
					change_cutoff = change_cutoff.take().or(cutoff_ui(
						ui, 
						cutoff, 
						max_freq, 
						Box::new(Self::set_to_butterworth_highpass)
					));
				},
				FilterType::Butterworth(Butterworth::Bandpass { cutoff }) => {
					change_cutoff = change_cutoff.take().or(cutoff_ui(
						ui, 
						cutoff,
						max_freq,
						Box::new(Self::set_to_butterworth_bandpass)
					));
				},
				FilterType::Butterworth(Butterworth::Bandstop { cutoff }) => {
					change_cutoff = change_cutoff.take().or(cutoff_ui(
						ui, 
						cutoff, 
						max_freq, 
						Box::new(Self::set_to_butterworth_bandstop)
					));
				},
				FilterType::Rbj(RBJFilter::Lowpass { cutoff, q }) => {
					change_cutoff_q = change_cutoff_q.take().or(cutoff_q_ui(
						ui, 
						cutoff,
						q,
						0.01..=10.0,
						"Q",
						max_freq,
						Box::new(Self::set_to_lowpass)
					));
				},
				FilterType::Rbj(RBJFilter::Highpass { cutoff, q }) => {
					change_cutoff_q = change_cutoff_q.take().or(cutoff_q_ui(
						ui, 
						cutoff,
						q,
						0.01..=10.0,
						"Q",
						max_freq,
						Box::new(Self::set_to_highpass)
					));
				},
				FilterType::Rbj(RBJFilter::Bandpass { cutoff, q }) => {
					change_cutoff_q = change_cutoff_q.take().or(cutoff_q_ui(
						ui, 
						cutoff,
						q,
						MIN_FREQUENCY..=10000.0,
						"BandWidth",
						max_freq,
						Box::new(Self::set_to_bandpass)
					));
				},
				FilterType::Rbj(RBJFilter::Bandstop { cutoff, q }) => {
					change_cutoff_q = change_cutoff_q.take().or(cutoff_q_ui(
						ui, 
						cutoff,
						q,
						MIN_FREQUENCY..=10000.0,
						"BandWidth",
						max_freq,
						Box::new(Self::set_to_bandstop)
					));
				},
				FilterType::Rbj(RBJFilter::Peak { cutoff, gain, bandwidth }) => {
					change_simulated = change_simulated.take().or(simulate_ui(
						ui, 
						(cutoff, gain, bandwidth),
						max_freq,
						MIN_FREQUENCY..=10000.0,
						vec!["Cutoff", "Gain(dB)", "Bandwidth"],
						Box::new(Self::set_to_peak)
					));
				},
				FilterType::Rbj(RBJFilter::LowShelf { cutoff, gain, slope }) => {
					change_simulated = change_simulated.take().or(simulate_ui(
						ui, 
						(cutoff, gain, slope),
						max_freq,
						0.5..=2.0,
						vec!["Cutoff", "Gain(dB)", "Slope"],
						Box::new(Self::set_to_low_shelf)
					));
				},
				FilterType::Rbj(RBJFilter::HighShelf { cutoff, gain, slope }) => {
					change_simulated = change_simulated.take().or(simulate_ui(
						ui, 
						(cutoff, gain, slope),
						max_freq,
						0.5..=2.0,
						vec!["Cutoff", "Gain(dB)", "Slope"],
						Box::new(Self::set_to_high_shelf)
					));
				}
			}

			if let Some((change_func, cutoff)) = change_cutoff {
				change_func(self, cutoff);
				if self.gui_state.allpass_mode {
					self.transform_to_allpass();
				}
			}

			if let Some((change_func, cutoff, q)) = change_cutoff_q {
				change_func(self, cutoff, q);
				if self.gui_state.allpass_mode {
					self.transform_to_allpass();
				}
			}

			if let Some((change_func, cutoff, gain, factor)) = change_simulated {
				change_func(self, cutoff, gain, factor);
				if self.gui_state.allpass_mode {
					self.transform_to_allpass();
				}
			}
		});
	}
}