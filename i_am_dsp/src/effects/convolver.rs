//! A simple convolver implementation.

use std::{f32::consts::PI, ops::Range, sync::Arc};

use i_am_dsp_derive::Parameters;
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use wide::f32x4;

use crate::{Effect, ProcessContext, prelude::{Biquad, WaveTable}, tools::ring_buffer::RingBuffer};

fn format_ir<const CHANNELS: usize>(ir: &[Vec<f32>; CHANNELS]) -> Vec<u8> {
	assert_eq!(std::mem::size_of::<f32>(), 4);
	
	let mut ir_data = vec![];

	for channel in ir {
		ir_data.extend_from_slice(channel);
	}
	
	for channel in ir.iter().rev() {
		ir_data.push(f32::from_bits(channel.len() as u32))
	}

	let ptr = ir_data.as_mut_ptr();
	let len = ir_data.len() * std::mem::size_of::<f32>();
	let cap = ir_data.capacity() * std::mem::size_of::<f32>();

	std::mem::forget(ir_data);

	unsafe {
		Vec::from_raw_parts(ptr as *mut u8, len, cap)
	}
}

fn parse_ir<const CHANNELS: usize>(mut data: Vec<u8>) -> [Vec<f32>; CHANNELS] {
	if data.is_empty() {
		panic!("Invalid data length");
	}

	assert_eq!(std::mem::size_of::<f32>(), 4);

	if !data.len().is_multiple_of(std::mem::size_of::<f32>()) {
		panic!("Invalid data length");
	}

	let ptr = data.as_mut_ptr() as *mut f32;
	let len = data.len() / std::mem::size_of::<f32>();
	let cap = data.capacity() / std::mem::size_of::<f32>();

	std::mem::forget(data);

	let mut ir_data = unsafe {
		Vec::from_raw_parts(ptr, len, cap)
	};

	// let channels = ir_data.pop().expect("Invalid PCM data: missing channel count").to_bits() as usize;
	let mut data_len = Vec::with_capacity(CHANNELS);

	for _ in 0..CHANNELS {
		data_len.push(ir_data.pop().expect("Invalid PCM data: missing channel length").to_bits() as usize);
	}

	let outputs: [Vec<f32>; CHANNELS] = std::array::from_fn(|i| {
		let mut channel = ir_data.split_off(data_len[i]);
		std::mem::swap(&mut channel, &mut ir_data);
		channel
	});

	outputs
}

#[derive(Debug, Clone)]
/// The mode to calculate the delay.
pub enum DelyaCaculateMode {
	/// Fixed delay
	Custom(usize),
	/// Threshold and min_consecutive
	Epsilon(f32, usize),
	/// Noise area, k value and min_consecutive
	Snr(
		Range<usize>, 
		f32,
		usize
	),
	/// FIR filter, use (N - 1) / 2 as delay, N must be odd
	Fir
}

impl Default for DelyaCaculateMode {
	fn default() -> Self {
		DelyaCaculateMode::Custom(0)
	}
}

impl DelyaCaculateMode {
	/// Calculate the delay based on the given mode and pcm data.
	/// 
	/// Panics 
	/// 1. if the mode is `Fir` and the length of the IR is not odd.
	/// 2. pcm_data is empty.
	pub fn calculate_delay<const CHANNELS: usize>(&self, pcm_data: &[Vec<f32>; CHANNELS]) -> usize {
		match self {
			DelyaCaculateMode::Custom(delay) => *delay,
			DelyaCaculateMode::Epsilon(epsilon, min_consecutive) => {
				let mut consecutive_count  = 0;
				let total_len = pcm_data[0].len();
				for i in 0..total_len {
					let sample = pcm_data
						.iter()
						.map(|x| x[i])
						.min_by_key(|float| (float.abs() * 1000.0) as usize)
						.unwrap_or(0.0);

					if sample.abs() > *epsilon {
						consecutive_count += 1;
						if consecutive_count >= *min_consecutive {
							return i - consecutive_count + 1;
						}
					}else {
						consecutive_count = 0;
					}
				}
				pcm_data.len()
			},
			DelyaCaculateMode::Snr(noise_area, k, min_consecutive) => {
				let total_len = pcm_data[0].len();
				let end = if noise_area.start >= pcm_data.len() {
					return 0;
				}else if noise_area.end > pcm_data.len() {
					pcm_data.len()
				}else {
					noise_area.end
				};
				let start = noise_area.start;

				if end - start <= 1{
					return 0;
				}

				let avg = pcm_data.iter()
					.map(|inner| inner[start..end].iter().sum::<f32>() / (end - start) as f32)
					.min_by_key(|x| (x.abs() * 1000.0) as usize)
					.unwrap_or(0.0);

				let std = (pcm_data.iter()
					.map(|inner| inner[start..end].iter().map(|&x| (x - avg).powi(2)).sum::<f32>() / (end - start - 1) as f32)
					.min_by_key(|x| (x.abs() * 1000.0) as usize)
					.unwrap_or(0.0)
				).sqrt();

				let threshold_std = k * std;
				let mut consecutive_count  = 0;
				for i in 0..total_len {
					let sample = pcm_data
						.iter()
						.map(|x| x[i])
						.min_by_key(|float| (float.abs() * 1000.0) as usize)
						.unwrap_or(0.0);

					if sample.abs() > threshold_std {
						consecutive_count += 1;
						if consecutive_count >= *min_consecutive {
							return i - consecutive_count + 1;
						}
					}else {
						consecutive_count = 0;
					}
				}
				pcm_data.len()
			},
			DelyaCaculateMode::Fir => {
				let n = pcm_data[0].len();
				assert!(n % 2 == 1, "The length of the IR must be odd");
				(n - 1) / 2
			}
		}
	}
}

/// The classical convolver, which is a FIR filter.
/// 
/// Note: The time complexity of this convolver is O(n*m), for o(l log l) implementation, see [`FftConvolver`].
#[derive(Parameters)]
pub struct Convolver<const CHANNELS: usize = 2> {
	#[persist(serialize = "format_ir", deserialize = "parse_ir")]
	ir: [Vec<f32>; CHANNELS],
	#[skip]
	history: [RingBuffer<f32>; CHANNELS],
	#[serde]
	delay: usize,

	#[cfg(feature = "real_time_demo")]
	#[skip]
	gui_state: (DelyaCaculateMode, Option<String>),

	#[cfg(feature = "real_time_demo")]
	#[skip]
	allow_change_ir: bool,

	#[skip]
	#[cfg(feature = "real_time_demo")]
	opened_file: Option<std::path::PathBuf>,
	#[cfg(feature = "real_time_demo")]
	#[skip]
	dialog: Option<egui_file::FileDialog>,
}

impl<const CHANNELS: usize> Convolver<CHANNELS> {
	/// Create a new convolver with the given IR and delta_caulate_mode.
	/// 
	/// # Panics
	/// 
	/// Panics if `CHANNELS` is 0.
	pub fn new(ir: [Vec<f32>; CHANNELS], delta_caulate_mode: &DelyaCaculateMode) -> Self {
		assert!(CHANNELS > 0, "CHANNELS must be greater than 0");
		let delay = delta_caulate_mode.calculate_delay(&ir);
		let history = core::array::from_fn(|_| RingBuffer::new(delay));
		Self { 
			ir, 
			history, 
			delay,

			#[cfg(feature = "real_time_demo")]
			gui_state: (delta_caulate_mode.clone(), None),

			#[cfg(feature = "real_time_demo")]
			allow_change_ir: false,

			#[cfg(feature = "real_time_demo")]
			dialog: None,
			#[cfg(feature = "real_time_demo")]
			opened_file: None,
		}
	}

	/// Replace the IR.
	pub fn replace_ir(&mut self, ir: [Vec<f32>; CHANNELS], delta_caulate_mode: &DelyaCaculateMode) {
		self.delay = delta_caulate_mode.calculate_delay(&ir);
		self.history = core::array::from_fn(|_| RingBuffer::new(ir[0].len()));
		self.ir = ir;

		#[cfg(feature = "real_time_demo")]
		{
			self.gui_state.0 = delta_caulate_mode.clone();
		}
	}

	/// Recalculate the delay based on the given mode.
	pub fn recaculate_delay(&mut self, delta_caulate_mode: &DelyaCaculateMode) {
		self.delay = delta_caulate_mode.calculate_delay(&self.ir);

		#[cfg(feature = "real_time_demo")]
		{
			self.gui_state.0 = delta_caulate_mode.clone();
		}
	}

	/// Get the history of the convolver.
	pub fn get_history(&self) -> &[RingBuffer<f32>; CHANNELS] {
		&self.history
	}
}

impl<const CHANNELS: usize> Effect<CHANNELS> for Convolver<CHANNELS> {
	fn delay(&self) -> usize {
		self.delay
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"Convolver"
	}

	fn process(&mut self, samples: &mut [f32; CHANNELS], _: &[&[f32; CHANNELS]], _: &mut Box<dyn ProcessContext>) {
		let n = self.ir[0].len();
		if n == 0 {
			return;
		}

		for (i, sample) in samples.iter_mut().enumerate() {
			self.history[i].push(*sample);
			*sample = 0.0;
			for j in (0..self.ir[i].len()).step_by(4) {
				let ir_samples = f32x4::from(&self.ir[i][j..(j + 4).min(self.ir[i].len())]);
				let history_samples = f32x4::from([
					self.history[i][n - j],
					self.history[i][n - j - 1],
					self.history[i][n - j - 2],
					self.history[i][n - j - 3],
				]);

				*sample += (ir_samples * history_samples).reduce_add();
			}
		}
	}
	
	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		use egui::*;
		use crate::tools::pcm_data::load_from_file;
		use crate::tools::pcm_data::PcmOutput;
		use crate::tools::ui_tools::draw_waveform;

		let mut clear_error = false;
		if let Some(error) = self.gui_state.1.as_ref() {
			ui.colored_label(Color32::RED, error);

			if ui.button("clear error").clicked() {
				clear_error = true;
			}
		}

		if clear_error {
			self.gui_state.1 = None;
		}

		
		egui::Resize::default().resizable([false, true])
		// .auto_sized()
			.min_width(ui.available_width())
			.max_width(ui.available_width())
			.id_salt(format!("{id_prefix}_convolver"))
			.show(ui, |ui| 
		{
			let ir_ref = self.ir.iter().map(|inner| inner.as_slice()).collect::<Vec<_>>();
			draw_waveform(ui, None, &ir_ref, &None, false, false);
		});

		ScrollArea::horizontal().show(ui, |ui| {
			ui.label(format!("Delay: {}", self.delay));
			ui.horizontal(|ui| {
				if ui.selectable_label(matches!(self.gui_state.0, DelyaCaculateMode::Custom(_)), "Fixed delay").clicked() {
					self.gui_state.0 = DelyaCaculateMode::Custom(self.delay);
					self.recaculate_delay(&DelyaCaculateMode::Custom(self.delay));
				}

				if ui.selectable_label(
					matches!(self.gui_state.0, DelyaCaculateMode::Epsilon(_, _)), 
					"Threshold Method"
				).clicked() {
					self.gui_state.0 = DelyaCaculateMode::Epsilon(0.01, 1);
					self.recaculate_delay(&DelyaCaculateMode::Epsilon(0.01, 1));
				}

				if ui.selectable_label(
					matches!(self.gui_state.0, DelyaCaculateMode::Snr {.. }), 
					"SNR Method"
				).clicked() {
					self.gui_state.0 = DelyaCaculateMode::Snr(0..10, 1.0, 1);
					self.recaculate_delay(&DelyaCaculateMode::Snr(0..10, 1.0, 1));
				}

				if ui.selectable_label(
					matches!(self.gui_state.0, DelyaCaculateMode::Fir), 
					"FIR filter"
				).clicked() {
					self.gui_state.0 = DelyaCaculateMode::Fir;
					self.recaculate_delay(&DelyaCaculateMode::Fir);
				}
			});
			ui.horizontal(|ui| {
				match &mut self.gui_state.0 {
					DelyaCaculateMode::Custom(delay) => {
						ui.add(Slider::new(delay, 0..=self.ir[0].len()).text("Delay"));
					},
					DelyaCaculateMode::Epsilon(epsilon, min_consecutive) => {
						ui.add(Slider::new(epsilon, 0.0..=1.0).text("Epsilon"));
						ui.add(Slider::new(min_consecutive, 1..=10).text("Min consecutive"));
					},
					DelyaCaculateMode::Snr(noise_area, k, min_consecutive) => {
						ui.add(Slider::new(&mut noise_area.start, 0..=self.ir[0].len()).text("Noise start"));
						ui.add(Slider::new(&mut noise_area.end, 0..=self.ir[0].len()).text("Noise end"));
						ui.add(Slider::new(k, 0.0..=10.0).text("K"));
						ui.add(Slider::new(min_consecutive, 1..=10).text("Min consecutive"));
					},
					DelyaCaculateMode::Fir => {
						ui.label("FIR filter, Delay = (N - 1) / 2");
					}
				}
			});
			ui.horizontal(|ui| {
				let mut path = None;

				if self.allow_change_ir {
					ui.input(|input| {
						path = input.raw.dropped_files.first().map(|inner| {
							inner.path.clone()
						}).unwrap_or_default();
					});
				}

				if ui.button("replace ir").clicked() {
					use std::ffi::OsStr;
					use egui_file::FileDialog;

					let filter = Box::new({
						let ext = Some(OsStr::new("wav"));
						move |path: &std::path::Path| -> bool {
							path.extension() == ext
						}
					});
					let mut dialog = FileDialog::open_file(self.opened_file.clone()).show_files_filter(filter);
					dialog.open();

					self.dialog = Some(dialog);
				}
				
				if let Some(dialog) = self.dialog.as_mut() {
					let dialog = dialog.show(ui.ctx());
					if dialog.selected() {
						path = dialog.path().map(|path| path.to_path_buf());
					}
				}

				if let Some(path) = path {
					if path.extension().map(|ext| ext.to_string_lossy().to_lowercase() != "wav").unwrap_or(true) {
						return;
					}

					self.opened_file = Some(path.clone());

					match load_from_file::<CHANNELS>(path) {
						Ok(PcmOutput {
							pcm_data,
							..
						}) => {
							let delta_caculate_mode = self.gui_state.0.clone();
							self.replace_ir(pcm_data, &delta_caculate_mode);
						}
						Err(e) => {
							self.gui_state.1 = Some(format!("Error: {}", e));
						}
					}
				}
				if ui.button("hilbert transform").clicked() {
					self.gui_state.0 = DelyaCaculateMode::Fir;
					self.replace_ir(hilbert_transform(511), &DelyaCaculateMode::Fir);
				}
				if ui.selectable_label(self.allow_change_ir, "Allow Replace IR").clicked() {
					self.allow_change_ir = !self.allow_change_ir;
				}
			});
		});
	}
}

/// Generate a Hilbert transform filter.
/// 
/// Panics if the length of the filter is not odd.
pub fn hilbert_transform<const CHANNELS: usize>(filter_len: usize) -> [Vec<f32>; CHANNELS] {
	assert!(filter_len % 2 == 1, "The length of the filter must be odd");

	let filter_delty = (filter_len - 1) / 2;
	let mut output = core::array::from_fn(|_| vec![0.0; filter_len]);

	for i in 0..filter_len {
		if i != filter_delty {
			let n_val = i as isize - filter_delty as isize;
			let sample = 2.0 / (PI * n_val as f32) * (PI * n_val as f32 / 2.0).sin().powi(2);
			for output_array in output.iter_mut().take(CHANNELS) {
				output_array[i] = sample;
			}
		}
	}

	// println!("{:?}", output[0]);

	output
}

const FFT_CONVOLVER_HISTORY_LEN: usize = 256;

#[derive(Parameters)]
struct FftBuffer<
	const CHANNELS: usize = 2,
	const FFT_SIZE: usize = FFT_CONVOLVER_HISTORY_LEN,
> {
	#[persist(serialize = "format_ir_splited", deserialize = "parse_ir_splited")]
	ir_splited: [Vec<[Complex<f32>; FFT_SIZE]>; CHANNELS],
	#[skip]
	historys: [Vec<RingBuffer<Complex<f32>>>; CHANNELS],
	#[skip]
	outputs: [Vec<[Complex<f32>; FFT_SIZE]>; CHANNELS],
	#[skip]
	history_counts: Vec<usize>,
	#[skip]
	downsample_filters: [Vec<Biquad<1>>; CHANNELS],
	#[skip]
	forward_fft: Arc<dyn Fft<f32>>,
	#[skip]
	inverse_fft: Arc<dyn Fft<f32>>,
	#[skip]
	output_count: usize
}

fn format_ir_splited<const CHANNELS: usize, const FFT_SIZE: usize>(ir_splited: &[Vec<[Complex<f32>; FFT_SIZE]>; CHANNELS]) -> Vec<u8> {
	let mut writer = vec![];
	let ir_splited = ir_splited.iter()
		.map(|inner| {
			inner.iter().map(|inner| {
				inner.iter().map(|complex| (complex.re, complex.im)).collect::<Vec<_>>()
			}).collect::<Vec<_>>()
		}).collect::<Vec<_>>();
	let _ = ciborium::into_writer(&ir_splited, &mut writer);
	writer
}

fn parse_ir_splited<const CHANNELS: usize, const FFT_SIZE: usize>(data: Vec<u8>) -> [Vec<[Complex<f32>; FFT_SIZE]>; CHANNELS] {
	let ir_splited: Vec<Vec<Vec<(f32, f32)>>> = ciborium::from_reader(data.as_slice()).unwrap();

	core::array::from_fn(|channel| {
		ir_splited[channel].iter().map(|inner| {
			core::array::from_fn(|id| {
				Complex::new(inner[id].0, inner[id].1)
			})
		}).collect::<Vec<[Complex<f32>; FFT_SIZE]>>()
	})
}

impl<const CHANNELS: usize, const FFT_SIZE: usize> FftBuffer<CHANNELS, FFT_SIZE> {
	fn new(ir: [Vec<f32>; CHANNELS], sample_rate: usize) -> Self {
		assert!(CHANNELS > 0, "CHANNELS must be greater than 0");

		let mut planner = FftPlanner::new();
		let forward_fft = planner.plan_fft_forward(FFT_SIZE);
		let inverse_fft = planner.plan_fft_inverse(FFT_SIZE);

		let ir_len = ir.iter().map(|inner| inner.len()).min().unwrap_or_default();
		let mut downsample_filters: [Vec<Biquad<1>>; CHANNELS] = core::array::from_fn(|_| {
			let mut filters = vec![];
			let mut current_freq = sample_rate as f32;
			let mut i = 0;
			loop {
				if i >= ir_len {
					break;
				}

				if i == 0 {
					filters.push(Biquad::<1>::new(sample_rate));
					i = FFT_SIZE;
					current_freq = current_freq / 2.0 - 10.0;
					continue;
				}

				filters.push(Biquad::<1>::lowpass(sample_rate, current_freq, Biquad::<1>::Q1));
				i *= 2;
				current_freq /= 2.0;
			}

			filters
		});
		let ir_splited: [Vec<[Complex<f32>; FFT_SIZE]>; CHANNELS] = core::array::from_fn(|channel| {
			let mut ir_splited = vec![];
			let mut i = 0;
			let mut current_freq = sample_rate as f32;
			let mut ctx = Box::new(()) as Box<dyn ProcessContext + 'static>;
			loop {
				let factor = if i == 0 {
					0
				}else {
					2_usize.pow(i as u32 - 1)
				};
				if factor * FFT_SIZE >= ir_len {
					break;
				}
				let mut output = [Complex::ZERO; FFT_SIZE];

				if i == 0 {
					for (id, sample) in ir[channel].iter().take(FFT_SIZE).enumerate() {
						output[id] = Complex::new(*sample, 0.0);
					}
					forward_fft.process(&mut output);
					ir_splited.push(output);
					i = 1;
					current_freq = current_freq / 2.0 - 10.0;
					continue;
				}
				
				let downsample_factor = 2_usize.pow(i as u32 - 1);
				for (id, sample) in ir[channel].iter()
					.skip(FFT_SIZE * downsample_factor)
					.take(FFT_SIZE * downsample_factor * 2)
					.enumerate() 
				{
					let mut input = [*sample];
					downsample_filters[channel][i].process(&mut input, &[], &mut ctx);

					if id % downsample_factor == 0 {
						output[id % downsample_factor] = Complex::new(input[0], 0.0);
					}
				}

				forward_fft.process(&mut output);
				ir_splited.push(output);
				i += 1;
				current_freq /= 2.0;
			}
			ir_splited
		});

		downsample_filters.iter_mut().for_each(|inner| {
			inner.iter_mut().for_each(|filter| {
				filter.clear_state();
			});
		});

		Self {
			historys: core::array::from_fn(|_| vec![RingBuffer::new(FFT_SIZE); ir_splited[0].len()]),
			downsample_filters,
			forward_fft,
			inverse_fft,
			history_counts: vec![0; ir_splited[0].len()],
			outputs: core::array::from_fn(|_| vec![[Complex::ZERO; FFT_SIZE]; ir_splited[0].len()]),
			ir_splited,
			output_count: 0,
		}
	}

	fn frame(&mut self, input: [f32; CHANNELS]) -> [f32; CHANNELS] {
		let mut ctx = Box::new(()) as Box<dyn ProcessContext + 'static>;
		for channel in 0..CHANNELS {
			let input = input[channel];
			for (id, (history, filter)) in self.historys[channel]
				.iter_mut().zip(self.downsample_filters[channel].iter_mut())
				.enumerate() 
			{
				if id == 0 {
					history.push(Complex::new(input, 0.0));
					continue;
				}
				let mut input = [input];
				filter.process(&mut input, &[], &mut ctx);
				if self.history_counts[id] == 0 {
					history.push(Complex::new(input[0], 0.0));
				}
				self.history_counts[id] += 1;
				self.history_counts[id] %= 2_usize.pow(id as u32 - 1);

				if history.current_pos() == 0 {
					let underlying_buffer = history.underlying_buffer_mut();
					self.forward_fft.process(underlying_buffer);

					self.outputs[channel][id].iter_mut().enumerate().for_each(|(i, inner)| {
						*inner = self.ir_splited[channel][id][i] * underlying_buffer[i] / FFT_SIZE as f32;
					});

					self.inverse_fft.process(underlying_buffer);
					self.inverse_fft.process(&mut self.outputs[channel][id]);
				}
			}
		}

		let mut output = [0.0; CHANNELS];

		for (channel, output) in output.iter_mut().enumerate() {
			for (id, output_array) in self.outputs[channel].iter().enumerate() {
				if id == 0 {
					*output += output_array[0].re;
					continue;
				}
				let downsample_factor = 2_usize.pow(id as u32 - 1);
				let count = self.output_count % (downsample_factor * FFT_SIZE);
				let time = count as f32 / (downsample_factor * FFT_SIZE) as f32;
				*output += output_array.as_slice().sample(time, 0);
			}
		}

		output
	}
}

/// The Fft-based convolver, Faster than the classical convolver but may cause lots of memory usage.
#[derive(Parameters)]
pub struct FftConvolver<
	const CHANNELS: usize = 2,
	const FFT_SIZE: usize = FFT_CONVOLVER_HISTORY_LEN,
> {
	#[sub_param]
	buffer: FftBuffer<CHANNELS, FFT_SIZE>,

	#[cfg(feature = "real_time_demo")]
	#[skip]
	ir: [Vec<f32>; CHANNELS],

	// #[skip]
	// other_way_convolver: [fft_convolver::FFTConvolver<f32>; CHANNELS],

	#[range(min = 0.01, max = 4.0)]
	#[logarithmic]
	/// The wet gain of the convolver, saves in linear scale.
	pub wet_gain: f32,
	#[range(min = 0.01, max = 4.0)]
	#[logarithmic]
	/// The dry gain of the convolver, saves in linear scale.
	pub dry_gain: f32,

	#[cfg(feature = "real_time_demo")]
	#[skip]
	allow_change_ir: bool,

	#[cfg(feature = "real_time_demo")]
	#[skip]
	error: Option<String>,

	#[cfg(feature = "real_time_demo")]
	#[skip]
	opened_file: Option<std::path::PathBuf>,
	#[cfg(feature = "real_time_demo")]
	#[skip]
	dialog: Option<egui_file::FileDialog>,

	#[skip]
	sample_rate: usize,

	// #[cfg(feature = "real_time_demo")]
	// #[skip]
	// other_way: bool,
}

impl<const CHANNELS: usize, const FFT_SIZE: usize> FftConvolver<CHANNELS, FFT_SIZE> {
	/// Create a new FftConvolver with the given IR.
	/// 
	/// # Panics
	/// 
	/// Panics if `CHANNELS` is 0.
	pub fn new(ir: [Vec<f32>; CHANNELS], sample_rate: usize) -> Self {
		assert!(CHANNELS > 0, "CHANNELS must be greater than 0");
		Self {
			#[cfg(feature = "real_time_demo")]
			ir: ir.clone(),

			buffer: FftBuffer::new(ir, sample_rate),

			// other_way_convolver,
			
			dry_gain: 1.0,
			wet_gain: 0.0125,

			#[cfg(feature = "real_time_demo")]
			allow_change_ir: false,
			#[cfg(feature = "real_time_demo")]
			error: None,

			#[cfg(feature = "real_time_demo")]
			opened_file: None,
			#[cfg(feature = "real_time_demo")]
			dialog: None,
			sample_rate,

			// #[cfg(feature = "real_time_demo")]
			// other_way: false,
		}
	}

	/// Replace the IR.
	pub fn replace_ir(&mut self, ir: [Vec<f32>; CHANNELS]) {
		#[cfg(feature = "real_time_demo")]
		{
			self.ir = ir.clone();
		}

		self.buffer = FftBuffer::new(ir, self.sample_rate);
	}
}

impl<const CHANNELS: usize, const FFT_SIZE: usize> Effect<CHANNELS> for FftConvolver<CHANNELS, FFT_SIZE> {
	fn delay(&self) -> usize {
		FFT_CONVOLVER_HISTORY_LEN
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"FftConvolver"
	}

	fn process(
		&mut self, 
		samples: &mut [f32; CHANNELS], 
		_: &[&[f32; CHANNELS]],
		_: &mut Box<dyn ProcessContext>,
	) {
		*samples = self.buffer.frame(*samples);
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		use egui::*;
		use crate::tools::pcm_data::load_from_file;
		use crate::tools::pcm_data::PcmOutput;
		use crate::tools::ui_tools::draw_waveform;
		use crate::tools::ui_tools::gain_ui;

		let mut clear_error = false;
		if let Some(error) = self.error.as_ref() {
			ui.colored_label(Color32::RED, error);

			if ui.button("clear error").clicked() {
				clear_error = true;
			}
		}

		if clear_error {
			self.error = None;
		}

		
		egui::Resize::default().resizable([false, true])
		// .auto_sized()
			.min_width(ui.available_width())
			.max_width(ui.available_width())
			.id_salt(format!("{id_prefix}_convolver"))
			.show(ui, |ui| 
		{
			let ir = self.ir.iter().map(|v| v.as_slice()).collect::<Vec<_>>();
			draw_waveform(ui, None, &ir, &None, false, false);
		});

		ui.horizontal(|ui| {
			let mut path = None;

			if self.allow_change_ir {
				ui.input(|input| {
					path = input.raw.dropped_files.first().map(|inner| {
						inner.path.clone()
					}).unwrap_or_default();
				});
			}

			if ui.button("replace ir").clicked() {
				use std::ffi::OsStr;
				use egui_file::FileDialog;

				let filter = Box::new({
					let ext = Some(OsStr::new("wav"));
					move |path: &std::path::Path| -> bool {
						path.extension() == ext
					}
				});
				let mut dialog = FileDialog::open_file(self.opened_file.clone()).show_files_filter(filter);
				dialog.open();

				self.dialog = Some(dialog);
			}
			
			if let Some(dialog) = self.dialog.as_mut() {
				let dialog = dialog.show(ui.ctx());
				if dialog.selected() {
					path = dialog.path().map(|path| path.to_path_buf());
				}
			}

			if let Some(path) = path {
				if path.extension().map(|ext| ext.to_string_lossy().to_lowercase() != "wav").unwrap_or(true) {
					return;
				}

				self.opened_file = Some(path.clone());

				match load_from_file::<CHANNELS>(path) {
					Ok(PcmOutput {
						pcm_data,
						..
					}) => {
						self.replace_ir(pcm_data);
					}
					Err(e) => {
						self.error = Some(format!("Error: {}", e));
					}
				}
			}
			if ui.button("hilbert transform").clicked() {
				self.replace_ir(hilbert_transform(511));
			}
			if ui.selectable_label(self.allow_change_ir, "Allow Replace IR").clicked() {
				self.allow_change_ir = !self.allow_change_ir;
			}

			gain_ui(ui, &mut self.dry_gain, Some("Dry Gain".to_string()), false);
			gain_ui(ui, &mut self.wet_gain, Some("Wet Gain".to_string()), true);

			// if ui.selectable_label(self.other_way, "Other Way").clicked() {
			// 	self.other_way = !self.other_way;
			// }
		});
	}
}

/// Generate a convolve ir that does nothing.
pub fn convolve_identity<const CHANNELS: usize>(len: usize) -> [Vec<f32>; CHANNELS] {
	core::array::from_fn(|_| (0..len).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect())
}