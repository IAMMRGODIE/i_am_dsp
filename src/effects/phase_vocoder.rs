//! Phase Vocoder effect implementation, can be used for frequency remapping.

use std::{f32::consts::PI, sync::Arc};

use i_am_parameters_derive::Parameters;
use rustfft::{num_complex::Complex, Fft, FftPlanner};

use crate::{prelude::Parameters, tools::{ring_buffer::RingBuffer}, Effect};

const OVERLAP_RATIO: usize = 4;

/// A Mapper trait for frequency remapping.
pub trait FrequencyMapper: Parameters {
	/// Maps a frequency and amplitude to a new frequency and amplitude.
	fn map_frequency(&mut self, frequency: f32, amplitude: f32) -> (f32, f32);

	#[cfg(feature = "real_time_demo")]
	/// UI for the frequency mapper.
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String);
}

fn window(window_size: usize, index: usize, offset: usize, window_factor: f32) -> f32 {
	let index = (index + offset) % window_size;
	0.5 * (window_factor - (1.0 - window_factor) * (2.0 * PI * index as f32 / window_size as f32).cos())
}

fn bin_frequencie(sample_rate: usize, window_size: usize, k: usize) -> f32 {
	k as f32 * sample_rate as f32 / window_size as f32
}

/// The Phase Vocoder effect.
/// 
/// The window function is as follows:
/// $$
/// w(n) = \frac{1}{2}(window_factor - (1.0 - window_factor) * \cos(\frac{2 \pi (n + window_offset \% window_size)}{window_size})
/// $$
#[derive(Parameters)]
pub struct PhaseVocoder<Mapper: FrequencyMapper, const CHANNELS: usize> {
	/// The mapper to use for frequency remapping.
	#[sub_param]
	pub mapper: Mapper,
	/// The window factor to use for the window function.
	pub window_factor: f32,
	/// The window offset to use for the window function.
	#[range(min = 0, max = 4096)]
	pub window_offset: usize,
	/// The output gain, saves in linear scale
	#[range(min = 0.0, max = 4.0)]
	pub gain: f32,
	/// The sample rate of the input signal.
	#[skip]
	pub sample_rate: usize,

	#[range(min = 256, max = 4096)]
	#[logarithmic]
	window_size: usize,
	#[skip]
	frame_hop: usize,

	#[skip]
	fft: Arc<dyn Fft<f32>>,
	#[skip]
	ifft: Arc<dyn Fft<f32>>,

	#[skip]
	input_buffer: [RingBuffer<f32>; CHANNELS],
	#[skip]
	output_buffer: [RingBuffer<f32>; CHANNELS],
	#[skip]
	prev_analysis_phase: [Vec<f32>; CHANNELS],

	#[skip]
	temp_buffer: [Vec<Complex<f32>>; CHANNELS],
	#[skip]
	output_temp_buffer: [Vec<Complex<f32>>; CHANNELS],

	#[skip]
	input_count: usize,
	#[skip]
	output_count: usize,
}

impl<Mapper: FrequencyMapper, const CHANNELS: usize> PhaseVocoder<Mapper, CHANNELS> {
	/// Creates a new Phase Vocoder effect with the given mapper and window size.
	/// 
	/// Panics if `CHANNELS` is 0
	pub fn new(mapper: Mapper, window_size: usize, sample_rate: usize) -> Self {
		assert!(CHANNELS > 0, "CHANNELS must be greater than 0");

		let window_size = window_size.next_power_of_two();
		let window_size = window_size.max(OVERLAP_RATIO);

		let frame_hop = window_size / OVERLAP_RATIO;

		let input_buffer = core::array::from_fn(|_| RingBuffer::new(window_size));
		let output_buffer = core::array::from_fn(|_| RingBuffer::new(window_size));

		let mut planner = FftPlanner::new();
		let fft = planner.plan_fft_forward(window_size);
		let ifft = planner.plan_fft_inverse(window_size);

		let prev_analysis_phase = core::array::from_fn(|_| vec![0.0; window_size]);

		let temp_buffer = core::array::from_fn(|_| vec![Complex::ZERO; window_size]);
		let output_temp_buffer = core::array::from_fn(|_| vec![Complex::ZERO; window_size]);

		Self {
			mapper,
			window_factor: 0.54,
			window_offset: 0,
			gain: 1.0,

			window_size,
			frame_hop,
			input_buffer,
			output_buffer,
			fft,
			ifft,
			prev_analysis_phase,
			sample_rate,
			temp_buffer,
			output_temp_buffer,
			input_count: 0,
			output_count: 0,
		}
	}

	/// Renews the window size of the effect.
	pub fn renew_window_size(&mut self, window_size: usize) -> Option<usize> {
		let window_size = window_size.next_power_of_two();
		let window_size = window_size.max(OVERLAP_RATIO);

		if window_size == self.window_size {
			return None;
		}

		self.window_size = window_size;
		self.frame_hop = window_size / OVERLAP_RATIO;

		self.input_buffer = core::array::from_fn(|_| RingBuffer::new(window_size));
		self.output_buffer = core::array::from_fn(|_| RingBuffer::new(window_size));

		let mut planner = FftPlanner::new();
		self.fft = planner.plan_fft_forward(window_size);
		self.ifft = planner.plan_fft_inverse(window_size);

		self.prev_analysis_phase = core::array::from_fn(|_| vec![0.0; window_size]);
		// self.prev_synthesis_phase = vec![0.0; window_size];

		self.temp_buffer = core::array::from_fn(|_| vec![Complex::ZERO; window_size]);
		self.output_temp_buffer = core::array::from_fn(|_| vec![Complex::ZERO; window_size]);

		self.input_count = 0;
		self.output_count = 0;

		Some(window_size)
	}

	fn process_inner(&mut self) {
		for channel in 0..CHANNELS {
			for (i, value) in self.temp_buffer[channel].iter_mut().enumerate() {
				*value = Complex::new(
					window(self.window_size, i, self.window_offset, self.window_factor) * self.input_buffer[channel][i], 
					0.0
				);
				self.output_temp_buffer[channel][i] = Complex::ZERO;
			}
	
			self.fft.process(&mut self.temp_buffer[channel]);
	
			for (k, value) in self.temp_buffer[channel].iter().enumerate().take(self.window_size / 2 + 1) {
				if k == 0 {
					self.output_temp_buffer[channel][0] = *value;
					continue;
				}
	
				let magnitude = value.norm();
				let bin_center_freq = bin_frequencie(self.sample_rate, self.window_size, k);
				let (mapped_freq, magnitude) = self.mapper.map_frequency(bin_center_freq, magnitude);
	
				if mapped_freq < 0.0 || mapped_freq >= self.sample_rate as f32 / 2.0 {
					continue;
				}
	
				let new_phase = 
					self.prev_analysis_phase[channel][k] + 
					2.0 * PI * bin_center_freq * self.frame_hop as f32 / self.sample_rate as f32;
	
				self.prev_analysis_phase[channel][k] = value.arg();
	
				let new_idx = mapped_freq / self.sample_rate as f32 * self.window_size as f32;
				let ratio = new_idx.fract();
				let k_low = new_idx.floor() as usize;
	
				if k_low <= self.window_size / 2 {
					self.output_temp_buffer[channel][k_low] += (1.0 - ratio) * Complex::from_polar(magnitude, new_phase);
				}
				if k_low < self.window_size / 2 {
					self.output_temp_buffer[channel][k_low + 1] += ratio * Complex::from_polar(magnitude, new_phase);
				}
			}
	
			self.output_temp_buffer[channel][0].im = 0.0;
			self.output_temp_buffer[channel][self.window_size / 2].im = 0.0;
			for i in 1..self.window_size / 2 {
				self.output_temp_buffer[channel][self.window_size - i] = self.output_temp_buffer[channel][i].conj();	
			}
	
			self.ifft.process(&mut self.output_temp_buffer[channel]);
	
			for i in 0..self.window_size {
				self.output_buffer[channel][i] += 
					self.output_temp_buffer[channel][i].re * 
					window(self.window_size, i, self.window_offset, self.window_factor) / 
					self.window_size as f32 *
					self.gain;
			}
		}
	}
}

impl<Mapper: FrequencyMapper + Send + Sync, const CHANNELS: usize> Effect<CHANNELS> for PhaseVocoder<Mapper, CHANNELS> {
	fn delay(&self) -> usize {
		self.window_size / 2
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"Frequency Mapper(Phase Vocoder)"
	}

	fn process(
		&mut self, 
		samples: &mut [f32; CHANNELS], 
		_: &[&[f32; CHANNELS]],
		_: &mut Box<dyn crate::ProcessContext>,
	) {
		for (i, sample) in samples.iter_mut().enumerate() {
			self.input_buffer[i].push(*sample);
			*sample = self.output_buffer[i][self.output_count] * 4.0;
		}

		self.output_count = (self.output_count + 1) % self.output_buffer[0].capacity(); 
		self.input_count += 1;

		if self.input_count >= self.frame_hop {
			for buffer in self.output_buffer.iter_mut() {
				buffer.extend_defaults(self.frame_hop);
			}
			self.input_count -= self.frame_hop;
			self.output_count = (self.output_count + self.output_buffer[0].capacity() - self.frame_hop) % self.output_buffer[0].capacity();
			self.process_inner();
		}
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		use crate::tools::ui_tools::gain_ui;
    	use egui::*;

		self.mapper.demo_ui(ui, format!("{}_mapper", id_prefix));

		let window_size_backup = usize::BITS - self.window_size.leading_zeros();
		let mut window_size = window_size_backup;
		ui.add(Slider::new(&mut window_size, 7..=12).text("Window Size")
			.custom_formatter(|val, _| {
				let size = 2usize.pow(val as u32);
				format!("{}", size)
			}));
		
		if window_size_backup != window_size {
			let _ = self.renew_window_size(2usize.pow(window_size));
		}

		ui.add(Slider::new(&mut self.window_factor, 0.0..=1.0).text("Window Factor"));
		ui.add(Slider::new(&mut self.window_offset, 0..=4096).text("Window Offset"));
		gain_ui(ui, &mut self.gain, None, false);
	}
}

/// A pitch shifter frequency mapper.
#[derive(Parameters)]
pub struct PitchShift {
	/// Shift amount in ratio.
	#[range(min = 0.25, max = 4.0)]
	#[logarithmic]
	pub shift: f32,
}

impl FrequencyMapper for PitchShift {
	fn map_frequency(&mut self, frequency: f32, amplitude: f32) -> (f32, f32) {
		let new_freq = frequency * self.shift;
		(new_freq, amplitude)
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, _: String) {
		use egui::*;

		ui.add(Slider::new(&mut self.shift, 0.25..=4.0)
			.text("Pitch Shift")
			.logarithmic(true)
		);
	}
}

impl Default for PitchShift {
	fn default() -> Self {
		Self {
			shift: 1.0,
		}
	}
}

#[derive(Default)]
/// A frequency shifter frequency mapper.
#[derive(Parameters)]
pub struct FreqShift {
	/// Shift amount in Hz.
	#[range(min = -2000.0, max = 2000.0)]
	pub freq: f32,
}


impl FrequencyMapper for FreqShift {
	fn map_frequency(&mut self, frequency: f32, amplitude: f32) -> (f32, f32) {
		let new_freq = frequency + self.freq;
		(new_freq, amplitude)
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, _: String) {
		use egui::*;

		ui.add(Slider::new(&mut self.freq, -2000.0..=2000.0)
			.text("Frequency Shift")
		);
	}
}