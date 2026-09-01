//! A simple convolver implementation.

use std::{f32::consts::PI, ops::Range, sync::Arc};

use i_am_dsp_derive::Parameters;
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use wide::f32x4;

use crate::{Effect, ProcessContext, tools::ring_buffer::RingBuffer};

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
							inner.path().to_path_buf()
						});
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
					let mut dialog = FileDialog::open_file().show_files_filter(filter);
					if let Some(opened_file) = &self.opened_file {
						dialog = dialog.initial_path(opened_file
							.parent()
							.map(|inner| inner.to_path_buf())
							.unwrap_or(std::path::PathBuf::from("."))
						)
					}
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

/// A single partition of the impulse response.
///
/// The IR is split into non-uniformly sized partitions: partition `k` covers
/// `[offset, offset + len)` where the lengths grow by powers of two. Each
/// partition is convolved with the input stream via block-FFT (overlap-add)
/// and its result is added to the output block corresponding to the
/// partition's offset. No downsampling is used; larger partitions simply use
/// larger FFT sizes.
struct FftPartition {
	/// Delay in whole input blocks: the partition consumes the input block
	/// `block_delay` blocks in the past, so its result lines up with the
	/// partition's offset in the impulse response.
	block_delay: usize,
	/// FFT size for this partition (next power of two >= FFT_SIZE + ir_len - 1).
	fft_size: usize,
	/// Precomputed FFT of the zero-padded IR segment.
	ir_fft: Vec<Complex<f32>>,
	/// Overlap-add accumulator for the partition's convolution result.
	acc: Vec<f32>,
	/// Scratch buffer holding the current input block, zero-padded.
	work: Vec<Complex<f32>>,
	/// Forward FFT of size `fft_size`.
	forward_fft: Arc<dyn Fft<f32>>,
	/// Inverse FFT of size `fft_size`.
	inverse_fft: Arc<dyn Fft<f32>>,
}

impl FftPartition {
	fn new(
		ir_segment: &[f32],
		offset: usize,
		block_size: usize,
		planner: &mut FftPlanner<f32>,
	) -> Self {
		let ir_len = ir_segment.len();
		let fft_size = (block_size + ir_len - 1).next_power_of_two();

		let mut ir_fft = vec![Complex::ZERO; fft_size];
		for (i, sample) in ir_segment.iter().enumerate() {
			ir_fft[i] = Complex::new(*sample, 0.0);
		}
		let forward_fft = planner.plan_fft_forward(fft_size);
		let inverse_fft = planner.plan_fft_inverse(fft_size);
		forward_fft.process(&mut ir_fft);

		Self {
			block_delay: offset / block_size,
			fft_size,
			ir_fft,
			acc: vec![0.0; fft_size],
			work: vec![Complex::ZERO; fft_size],
			forward_fft,
			inverse_fft,
		}
	}

	/// Convolve the given (already zero-padded) input block `work` against this
	/// partition and add the result into `out_block`.
	///
	/// `work` must contain the input block in `[0..block_size)` followed by
	/// zeros up to `fft_size`. Both the forward and inverse FFTs used here are
	/// unnormalized, so the result is divided by `fft_size` to obtain the true
	/// linear convolution.
	fn process_block(&mut self, out_block: &mut [f32], block_size: usize) {
		self.forward_fft.process(&mut self.work);
		for (a, b) in self.work.iter_mut().zip(self.ir_fft.iter()) {
			*a *= *b;
		}
		self.inverse_fft.process(&mut self.work);

		for i in 0..self.fft_size {
			self.acc[i] += self.work[i].re / self.fft_size as f32;
		}
		for (i, inner) in out_block.iter_mut().take(block_size).enumerate() {
			*inner += self.acc[i];
		}

		for i in 0..self.fft_size - block_size {
			self.acc[i] = self.acc[i + block_size];
		}
		for i in self.fft_size - block_size..self.fft_size {
			self.acc[i] = 0.0;
		}
	}
}

/// The streaming FFT convolution engine.
///
/// The impulse response is split into non-uniform partitions (doubling
/// lengths). Each partition runs its own block-overlap-add convolution with
/// the input stream, consuming the input block located `block_delay` blocks
/// in the past so its output lands at the partition's offset in the result.
struct FftBuffer<
	const CHANNELS: usize = 2,
	const FFT_SIZE: usize = FFT_CONVOLVER_HISTORY_LEN,
> {
	/// The raw impulse response, kept for serialization.
	ir: [Vec<f32>; CHANNELS],
	/// Per-channel partition structures.
	partitions: [Vec<FftPartition>; CHANNELS],
	/// Completed input blocks, newest at the back.
	history: Vec<[Vec<f32>; CHANNELS]>,
	/// The input block currently being filled.
	current_block: [Vec<f32>; CHANNELS],
	/// The output block currently being played out.
	output_block: [Vec<f32>; CHANNELS],
	/// Position within the current input/output block.
	pos: usize,
	/// Largest block delay over all partitions (history size needed).
	max_block_delay: usize,
}

impl<const CHANNELS: usize, const FFT_SIZE: usize> FftBuffer<CHANNELS, FFT_SIZE> {
	fn new(ir: [Vec<f32>; CHANNELS], _sample_rate: usize) -> Self {
		assert!(CHANNELS > 0, "CHANNELS must be greater than 0");
		let mut buffer = Self {
			ir,
			partitions: core::array::from_fn(|_| vec![]),
			history: vec![],
			current_block: core::array::from_fn(|_| vec![0.0; FFT_SIZE]),
			output_block: core::array::from_fn(|_| vec![0.0; FFT_SIZE]),
			pos: 0,
			max_block_delay: 0,
		};
		buffer.rebuild_partitions();
		buffer
	}

	/// Recompute all partitions from the current IR, clearing all running state.
	fn rebuild_partitions(&mut self) {
		let mut planner = FftPlanner::new();
		self.partitions = core::array::from_fn(|channel| {
			let mut partitions = vec![];
			let ir = &self.ir[channel];
			let mut offset = 0usize;
			let mut len = FFT_SIZE;
			while offset < ir.len() {
				let end = (offset + len).min(ir.len());
				partitions.push(FftPartition::new(&ir[offset..end], offset, FFT_SIZE, &mut planner));
				offset = end;
				len *= 2;
			}
			partitions
		});
		self.max_block_delay = self.partitions.iter()
			.flat_map(|inner| inner.iter().map(|p| p.block_delay))
			.max().unwrap_or(0);
		self.history.clear();
		self.pos = 0;
		for block in self.current_block.iter_mut() { block.fill(0.0); }
		for block in self.output_block.iter_mut() { block.fill(0.0); }
	}

	fn frame(&mut self, input: [f32; CHANNELS]) -> [f32; CHANNELS] {
		// Step 1: emit the oldest sample of the current output block.
		let mut output = [0.0; CHANNELS];

		for (channel, output) in output.iter_mut().enumerate() {
			*output = self.output_block[channel][self.pos];
		}

		// Step 2: collect the current input sample.
		for (channel, input) in input.iter().enumerate() {
			self.current_block[channel][self.pos] = *input;
		}
		self.pos += 1;

		// Step 3: once a full block has been collected, run every partition and
		// synthesize the next output block.
		if self.pos == FFT_SIZE {
			self.pos = 0;
			self.compute_next_block();
		}

		output
	}

	fn compute_next_block(&mut self) {
		let block_size = FFT_SIZE;

		// Push the completed input block into the history.
		let completed: [Vec<f32>; CHANNELS] = core::array::from_fn(|ch| std::mem::take(&mut self.current_block[ch]));
		self.history.push(completed);
		if self.history.len() > self.max_block_delay + 1 {
			self.history.remove(0);
		}
		for ch in 0..CHANNELS {
			self.current_block[ch] = vec![0.0; block_size];
		}

		// Reset the output block, then let every partition add its contribution.
		for ch in 0..CHANNELS {
			self.output_block[ch].iter_mut().for_each(|v| *v = 0.0);
		}

		let num_blocks = self.history.len(); // index of the just-completed block
		for ch in 0..CHANNELS {
			for partition in self.partitions[ch].iter_mut() {
				let delay = partition.block_delay;
				// This partition needs the block `delay` positions in the past.
				if num_blocks <= delay {
					continue;
				}
				let input_index = num_blocks - 1 - delay;
				let input_block = &self.history[input_index][ch];

				partition.work.iter_mut().for_each(|v| *v = Complex::ZERO);
				for (i, &sample) in input_block.iter().enumerate() {
					partition.work[i] = Complex::new(sample, 0.0);
				}
				partition.process_block(&mut self.output_block[ch], block_size);
			}
		}
	}
}

// Manual Parameters implementation so the FftBuffer can be serialized together
// with FftConvolver. The raw impulse response is what gets persisted; the
// FFT partitions are state derived from it and rebuild automatically.
impl<const CHANNELS: usize, const FFT_SIZE: usize> crate::prelude::Parameters for FftBuffer<CHANNELS, FFT_SIZE> {
	fn get_parameters(&self) -> Vec<crate::prelude::Parameter> {
		vec![crate::prelude::Parameter {
			identifier: "ir".to_string(),
			value: crate::prelude::Value::Serialized(format_ir(&self.ir)),
		}]
	}

	fn set_parameter(&mut self, identifier: &str, value: crate::prelude::SetValue) -> bool {
		if identifier != "ir" {
			return false;
		}
		if let crate::prelude::SetValue::Serialized(data) = value {
			self.ir = parse_ir(data);
			self.rebuild_partitions();
			true
		}else {
			false
		}
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

	/// Per-channel delay line delaying the dry signal by FFT_SIZE samples so
	/// it lines up with the (equally delayed) convolved wet signal at mix time.
	#[skip]
	dry_delay: [RingBuffer<f32>; CHANNELS],

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
			dry_delay: core::array::from_fn(|_| RingBuffer::new(FFT_SIZE)),

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
		FFT_SIZE
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
		let dry = *samples;
		// The wet path (FFT convolution) is delayed by FFT_SIZE samples, so
		// delay the dry path by the same amount to keep the mix time-aligned.
		let wet = self.buffer.frame(dry);

		for (i, sample) in samples.iter_mut().enumerate() {
			let dry_delayed = self.dry_delay[i][0];
			self.dry_delay[i].push(dry[i]);
			*sample = dry_delayed * self.dry_gain + wet[i] * self.wet_gain;
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
						inner.path().to_path_buf()
					});
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
				let mut dialog = FileDialog::open_file().show_files_filter(filter);
				if let Some(opened_file) = &self.opened_file {
					dialog = dialog.initial_path(opened_file
						.parent()
						.map(|inner| inner.to_path_buf())
						.unwrap_or(std::path::PathBuf::from("."))
					)
				}
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

#[cfg(test)]
mod fft_convolver_tests {
	use super::*;

	fn direct_conv(x: &[f32], h: &[f32]) -> Vec<f32> {
		let mut y = vec![0.0; x.len()];
		for n in 0..x.len() {
			for k in 0..h.len() {
				if k <= n {
					y[n] += x[n - k] * h[k];
				}
			}
		}
		y
	}

	/// delta IR: the FFT convolver must reproduce the input (with one block of
	/// latency), and match the direct convolution exactly.
	#[test]
	fn delta_ir_is_delayed_passthrough() {
		let ir = convolve_identity::<1>(8);
		let mut conv = FftConvolver::<1, 8>::new(ir, 48_000);
		conv.dry_gain = 0.0;
		conv.wet_gain = 1.0;


		let x: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.1).sin()).collect();
		let mut out = vec![];
		let mut ctx: Box<dyn ProcessContext> = Box::new(());
		for &s in &x {
			let mut block = [s];
			conv.process(&mut block, &[], &mut ctx);
			out.push(block[0]);
		}

		// latency = FFT_SIZE samples
		for n in 8..x.len() {
			let diff = (out[n] - x[n - 8]).abs();
			assert!(diff < 1e-4, "delta passthrough mismatch at {n}: {}", diff);
		}
	}

	/// For a short IR that spans several partitions, the FFT convolver must
	/// agree with a direct (!) convolution (up to the block latency).
	#[test]
	fn matches_direct_convolution() {
		// IR longer than a few blocks, spanning multiple partitions
		let ir_len = 257usize;
		let ir: Vec<f32> = (0..ir_len).map(|i| (i as f32 * 0.3).sin() * (-(i as f32) / 40.0).exp()).collect();
		let ir_channels = [ir.clone()];

		let mut fft_conv = FftConvolver::<1, 8>::new(ir_channels.clone(), 48_000);
		fft_conv.dry_gain = 0.0;
		fft_conv.wet_gain = 1.0;


		let x: Vec<f32> = (0..2048).map(|i| (i as f32 * 0.05).sin() + 0.3 * (i as f32 * 0.9).cos()).collect();

		let mut out = vec![];
		let mut ctx: Box<dyn ProcessContext> = Box::new(());
		for &s in &x {
			let mut block = [s];
			fft_conv.process(&mut block, &[], &mut ctx);
			out.push(block[0]);
		}

		let reference = direct_conv(&x, &ir);

		let block = 8usize; // FFT_SIZE
		let mut max_err = 0.0f32;
		for n in block..x.len() {
			let err = (out[n] - reference[n - block]).abs();
			max_err = max_err.max(err);
		}
		assert!(max_err < 1e-3, "FFT convolver disagrees with direct: max err = {max_err}");
	}

	/// No IR (empty) must not panic and should behave as a passthrough.
	#[test]
	fn empty_ir_does_not_panic() {
		let mut conv = FftConvolver::<1, 8>::new([vec![]], 48_000);
		conv.wet_gain = 0.0;

		let mut ctx: Box<dyn ProcessContext> = Box::new(());
		for i in 0..128usize {
			let mut block = [i as f32 * 0.01];
			conv.process(&mut block, &[], &mut ctx);
		}
	}

	/// dry_gain and wet_gain must both be applied, and the dry path must be
	/// delayed by the block latency so dry and wet stay time-aligned: with a
	/// delta IR and gains of 1.0/1.0 the output is twice the delayed input.
	#[test]
	fn dry_wet_mix_is_time_aligned() {
		let ir = convolve_identity::<1>(8);
		let mut conv = FftConvolver::<1, 8>::new(ir, 48_000);
		conv.dry_gain = 1.0;
		conv.wet_gain = 1.0;

		let x: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.1).sin()).collect();
		let mut out = vec![];
		let mut ctx: Box<dyn ProcessContext> = Box::new(());
		for &s in &x {
			let mut block = [s];
			conv.process(&mut block, &[], &mut ctx);
			out.push(block[0]);
		}

		// both paths are delayed by FFT_SIZE, so out[n] = 2 * x[n - 8]
		for n in 8..x.len() {
			let diff = (out[n] - 2.0 * x[n - 8]).abs();
			assert!(diff < 1e-4, "dry/wet mix mismatch at {n}: {}", diff);
		}
	}
}
