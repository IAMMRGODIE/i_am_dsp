//! A simple convolver implementation.

use std::{f32::consts::PI, ops::Range, sync::Arc};

use crossbeam_channel::{Receiver, Sender};
use i_am_parameters_derive::Parameters;
use rustfft::{num_complex::Complex, FftPlanner};
use threadpool::ThreadPool;

use crate::{tools::ring_buffer::RingBuffer, Effect, ProcessContext};

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

fn format_ir_arc<const CHANNELS: usize>(ir: &[Arc<Vec<f32>>; CHANNELS]) -> Vec<u8> {
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

fn parse_ir_arc<const CHANNELS: usize>(data: Vec<u8>) -> [Arc<Vec<f32>>; CHANNELS] {
	let output: [Vec<f32>; CHANNELS] = parse_ir(data);
	let outputs: [Arc<Vec<f32>>; CHANNELS] = std::array::from_fn(|i| {
		Arc::new(output[i].clone())
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
			for (j, ir) in self.ir[i].iter().enumerate() {
				*sample += ir * self.history[i][n - j] 
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

type TrackInner = (TrackHandle, Sender<(Vec<f32>, usize)>);

/// The Fft-based convolver, Faster than the classical convolver but may cause lots of memory usage.
#[derive(Parameters)]
pub struct FftConvolver<const CHANNELS: usize = 2> {
	#[persist(serialize = "format_ir_arc", deserialize = "parse_ir_arc")]
	ir: [Arc<Vec<f32>>; CHANNELS],

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

	#[skip]
	history: [RingBuffer<f32>; CHANNELS],
	#[skip]
	thread_pool: ThreadPool,
	#[skip]
	track: [Vec<TrackInner>; CHANNELS],

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

	// #[cfg(feature = "real_time_demo")]
	// #[skip]
	// other_way: bool,
}

struct TrackHandle {
	history: Arc<Vec<f32>>,
	output: Vec<f32>,
	start: usize,
	end: usize,
	current_pos: usize,
	rx: Receiver<(Vec<f32>, usize)>,
}

impl TrackHandle {
	fn new(
		history: Arc<Vec<f32>>,
		output: Vec<f32>, 
		desired_size: usize,
		rx: Receiver<(Vec<f32>, usize)>
	) -> Self {
		Self {
			history,
			output,
			start: 0,
			end: desired_size,
			current_pos: 0,
			rx,
		}
	}

	fn is_ended(&self) -> bool {
		if self.current_pos < self.start {
			false
		}else {
			let index = self.current_pos - self.start;
			index >= self.output.len()
		}
	}

	fn get_output(&mut self) -> (f32, Option<usize>) {
		let output_value = if self.current_pos < self.start {
			0.0
		}else {
			let index = self.current_pos - self.start;
			if index >= self.output.len() {
				0.0
			}else {
				self.output[index]
			}
		};

		let new_pos = if self.current_pos >= self.end {
			if let Ok((mut data, desired_size)) = self.rx.try_recv() {
				let desired_len = self.end - self.start;
				for i in 0..self.output.len().saturating_sub(desired_len) {
					if i < data.len() {
						data[i] += self.output[desired_len + i]
					}
				}
				self.output = data;
				self.start = self.end;
				self.end = desired_size;
				Some(self.end)
			}else {
				None
			}
		}else {
			None
		};
		self.current_pos += 1;
		
		(output_value, new_pos)
	}
}

impl<const CHANNELS: usize> FftConvolver<CHANNELS> {
	/// Create a new FftConvolver with the given IR.
	/// 
	/// # Panics
	/// 
	/// Panics if `CHANNELS` is 0.
	pub fn new(ir: [Vec<f32>; CHANNELS]) -> Self {
		assert!(CHANNELS > 0, "CHANNELS must be greater than 0");

		// let max_ir_len = ir.iter().map(|v| v.len()).max().unwrap_or(0);
		let mut ir = ir.into_iter().rev().collect::<Vec<_>>();
		let ir = core::array::from_fn(|_| Arc::new(ir.pop().unwrap()));
		let history = core::array::from_fn(|_| RingBuffer::new(FFT_CONVOLVER_HISTORY_LEN));
	
		// let mut num_threads = CHANNELS;
		// let mut start = FFT_CONVOLVER_HISTORY_LEN;

		// while start < max_ir_len {
		// 	num_threads += CHANNELS;
		// 	start *= 2;
		// }

		let thread_pool = ThreadPool::new(4);
		let track = core::array::from_fn(|_| vec![]);
		// let other_way_convolver =  core::array::from_fn(|i| {
		// 	let mut inner = fft_convolver::FFTConvolver::default();
		// 	inner.init(2048, &ir[i]).unwrap();
		// 	inner
		// });

		Self {
			ir,
			history,
			thread_pool,
			track,

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

			// #[cfg(feature = "real_time_demo")]
			// other_way: false,
		}
	}

	/// Replace the IR.
	pub fn replace_ir(&mut self, ir: [Vec<f32>; CHANNELS]) {
		// let max_ir_len = ir.iter().map(|v| v.len()).max().unwrap_or(0);
		// let mut num_threads = CHANNELS;
		// let mut start = FFT_CONVOLVER_HISTORY_LEN;

		// while start < max_ir_len {
		// 	num_threads += CHANNELS;
		// 	start *= 2;
		// }
		let mut ir = ir.into_iter().rev().collect::<Vec<_>>();
		let ir = core::array::from_fn(|_| Arc::new(ir.pop().unwrap()));

		// for (i, ir) in ir.iter().enumerate() {
		// 	self.other_way_convolver[i].init(2048, ir).unwrap();
		// }

		self.ir = ir;
		// self.thread_pool.set_num_threads(num_threads);
	}

	/// Get the history of the convolver.
	pub fn get_history(&self) -> &[RingBuffer<f32>; CHANNELS] {
		&self.history
	}
}

fn convolve(
	history: &[f32],
	ir: &[f32],
) -> Vec<f32> {
	// use rayon::prelude::*;
	let min_len = history.len() + ir.len() - 1;

	let len = if min_len.is_power_of_two() {
		min_len
	}else {
		min_len.next_power_of_two()
	};

	let mut planer = FftPlanner::new();

	let mut history = (0..len).map(|i| {
		if i < history.len() {
			Complex::new(history[i], 0.0) 
		}else {
			Complex::ZERO
		}
	}).collect::<Vec<_>>();

	let mut ir = (0..len).map(|i| {
		if i < ir.len() {
			Complex::new(ir[i], 0.0) 
		}else {
			Complex::ZERO
		}
	}).collect::<Vec<_>>();

	let forward = planer.plan_fft_forward(len);

	forward.process(&mut history);
	forward.process(&mut ir);

	let mut output = (0..len).map(|i| {
		history[i] * ir[i]
	}).collect::<Vec<_>>();

	let inverse = planer.plan_fft_inverse(len);

	inverse.process(&mut output);

	output.iter().map(|c| c.re / len as f32).collect()
}

impl<const CHANNELS: usize> Effect<CHANNELS> for FftConvolver<CHANNELS> {
	fn delay(&self) -> usize {
		FFT_CONVOLVER_HISTORY_LEN
	}

	fn name(&self) -> &str {
		"FftConvolver"
	}

	fn process(
		&mut self, 
		samples: &mut [f32; CHANNELS], 
		_: &[&[f32; CHANNELS]],
		_: &mut Box<dyn ProcessContext>,
	) {
		// if self.other_way {
		// 	for (i, sample) in samples.iter_mut().enumerate() {
		// 		let mut output = [0.0];
		// 		self.other_way_convolver[i].process(&[*sample], &mut output).unwrap();
		// 		*sample = output[0] * self.wet_gain + *sample * self.dry_gain;
		// 	}
		// 	return;
		// }

		for (i, history) in self.history.iter_mut().enumerate() {
			if history.current_pos() == 0 {
				let under_lying_data = history.underlying_buffer().to_vec();
				let end = FFT_CONVOLVER_HISTORY_LEN.min(self.ir[i].len());
				let convolved = convolve(&under_lying_data, &self.ir[i][0..end]);
				let (tx, rx) = crossbeam_channel::unbounded();
				let tx_in = tx.clone();
				let history = Arc::new(under_lying_data);
				let history_in = history.clone();
				let ir_in = self.ir[i].clone();
				if FFT_CONVOLVER_HISTORY_LEN < ir_in.len() {
					self.thread_pool.execute(move || {
						let start = FFT_CONVOLVER_HISTORY_LEN;
						let end = (start * 2).min(ir_in.len());
						let output = convolve(&history_in, &ir_in[start..end]);
						let _ = tx_in.send((output, end));
					});
				}

				self.track[i].push((TrackHandle::new(history, convolved, FFT_CONVOLVER_HISTORY_LEN, rx), tx));
			}
			history.push(samples[i]);
		}

		for (i, sample) in samples.iter_mut().enumerate() {
			*sample *= self.dry_gain;
			if self.track[i].is_empty() {
				continue;
			}
			self.track[i].retain_mut(|(handle, tx)| {
				let (output, new_end) = handle.get_output();
				let history_in = handle.history.clone();
				let ir_in = self.ir[i].clone();
				let tx_in = tx.clone();
				if let Some(new_end) = new_end {
					if new_end > ir_in.len() {
						return false;
					}

					self.thread_pool.execute(move || {
						let start = new_end;
						let end = (start * 2).min(ir_in.len());
						// println!("start: {}, end: {}, ir_len: {}", start, end, ir_in.len());
						let output = convolve(&history_in, &ir_in[start..end]);
						let _ = tx_in.send((output, end));
					});
					// self.thread_pool.join();
				}
				*sample += output * self.wet_gain;
				!handle.is_ended()
			});
		}
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