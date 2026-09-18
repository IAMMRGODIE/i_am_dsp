//! A simple convolver implementation.

use std::{f32::consts::PI, ops::Range, sync::Arc};

use i_am_dsp_derive::Parameters;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex;
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

const FFT_CONVOLVER_HISTORY_LEN: usize = 64;

/// The longest hop a partition grows to, in input samples.
///
/// The partition length and hop double together, so that every octave of the
/// impulse response is only transformed at its own rate. Doubling stops here
/// because a partition with hop h is transformed with an FFT of up to 2 * h
/// points, and that single transform has to fit inside one audio block: a longer
/// partition would make the block that evaluates it proportionally more
/// expensive, and unlike the first block it cannot be evaluated directly. The
/// rest of the impulse response is covered by partitions of this size.
const MAX_PARTITION_HOP: usize = 16384;


/// Convolve one sample against the direct head coefficients.
///
/// The block transforms can only produce their first sample once a whole hop of
/// input has been collected, so the taps of the first hop are evaluated in the
/// time domain instead: that is what makes the convolver zero latency, at the
/// price of 'coeffs.len()' multiply-adds per sample per channel. 'coeffs' holds
/// the impulse response taps in reverse, so the window, which is in
/// chronological order, is read forwards like the coefficients.
///
/// Four taps per multiply-add is as far as this is worth taking. The loop is
/// bound by reading the window, and a wider version with avx2 and fma behind a
/// runtime check measured no faster: the loads it saves are not the bottleneck.
/// See 'FftBuffer::frame' for what the bottleneck is.
fn head_convolve(coeffs: &[f32], window: &[f32]) -> f32 {
	let mut sum = f32x4::ZERO;
	let (coefficients, coefficient_tail) = coeffs.as_chunks::<4>();
	let (samples, sample_tail) = window.as_chunks::<4>();
	for (coefficients, samples) in coefficients.iter().zip(samples.iter()) {
		sum = f32x4::from(*coefficients).mul_add(f32x4::from(*samples), sum);
	}
	let mut output = sum.reduce_add();
	for (coefficient, sample) in coefficient_tail.iter().zip(sample_tail) {
		output += coefficient * sample;
	}
	output
}

/// A single partition of the impulse response.
///
/// The partition covers the taps [offset, offset + segment_len) and consumes
/// 'hop' new input samples per evaluation. The linear convolution of one hop of
/// input with the segment uses an FFT of size next_power_of_two(hop +
/// segment_len - 1), so a partition that covers one octave of the impulse
/// response is only transformed once per octave of input instead of once per
/// block. Both the input window and the impulse response segment are real, so
/// the transforms are real-to-complex and complex-to-real and only the N / 2 + 1
/// non-redundant bins are stored and multiplied.
///
/// 'offset' is always at least 'hop', so the result of an evaluation starts at
/// the sample that is played out next and the convolver has no latency.
///
/// 'phase' staggers the partitions inside their 'hop_blocks' cycle so that they
/// do not all run in the same audio block. Any phase is correct: a partition
/// always convolves its most recent 'hop' input samples, so the phase only
/// decides which block pays for which FFT, not what is computed.
struct FftPartition {
	/// Offset of this partition inside the impulse response.
	offset: usize,
	/// Input samples consumed by one evaluation (a multiple of the block size).
	hop: usize,
	/// Number of input blocks between two evaluations (hop / block_size).
	hop_blocks: usize,
	/// Firing phase inside the hop_blocks cycle.
	phase: usize,
	/// Number of impulse response taps covered by this partition.
	segment_len: usize,
	/// FFT size, a power of two >= hop + segment_len - 1.
	fft_size: usize,
	/// Precomputed half spectrum of the zero padded impulse response segment,
	/// with the 1 / fft_size of the unnormalized inverse transform folded in.
	ir_spectrum: Vec<Complex<f32>>,
	/// The real input window of this partition, zero padded. The forward
	/// transform clobbers it, so it is refilled from the block ring every time.
	work: Vec<f32>,
	/// Half spectrum of the current window. The inverse transform clobbers it.
	spectrum: Vec<Complex<f32>>,
	/// Preallocated scratch, so evaluating a partition never allocates.
	scratch: Vec<Complex<f32>>,
	/// Forward transform of size fft_size.
	forward_fft: Arc<dyn RealToComplex<f32>>,
	/// Inverse transform of size fft_size.
	inverse_fft: Arc<dyn ComplexToReal<f32>>,
}

impl FftPartition {
	fn new(
		ir_segment: &[f32],
		offset: usize,
		hop: usize,
		phase: usize,
		block_size: usize,
		planner: &mut RealFftPlanner<f32>,
	) -> Self {
		let segment_len = ir_segment.len();
		// The real transforms need at least two samples, which a one sample
		// window with a one tap segment would otherwise ask for.
		let fft_size = (hop + segment_len - 1).max(2).next_power_of_two();

		let forward_fft = planner.plan_fft_forward(fft_size);
		let inverse_fft = planner.plan_fft_inverse(fft_size);

		let mut work = forward_fft.make_input_vec();
		let mut ir_spectrum = forward_fft.make_output_vec();
		let mut scratch = vec![
			Complex::ZERO;
			forward_fft.get_scratch_len().max(inverse_fft.get_scratch_len())
		];
		work[..segment_len].copy_from_slice(ir_segment);
		forward_fft
			.process_with_scratch(&mut work, &mut ir_spectrum, &mut scratch)
			.expect("the impulse response window has the size the plan asked for");
		// Both transforms are unnormalized, so the 1 / fft_size of the inverse
		// transform is folded into the precomputed spectrum. The overlap-add then
		// only has to add instead of multiply-add.
		let inv_fft_size = 1.0 / fft_size as f32;
		for bin in ir_spectrum.iter_mut() {
			*bin *= inv_fft_size;
		}

		Self {
			offset,
			hop,
			hop_blocks: hop / block_size,
			phase,
			segment_len,
			fft_size,
			ir_spectrum,
			work,
			spectrum: forward_fft.make_output_vec(),
			scratch,
			forward_fft,
			inverse_fft,
		}
	}

	/// Convolve the most recent 'hop' input samples with this partition and
	/// overlap-add the result into the output accumulator.
	///
	/// 'history' holds history_mask + 1 completed input blocks of 'block_size'
	/// samples, 'window_start' is the ring position of the oldest block of this
	/// partition's window, 'acc_start' is where the result lands in the output
	/// accumulator, and the accumulator wraps at its own length.
	fn process(
		&mut self,
		history: &[f32],
		window_start: usize,
		history_mask: usize,
		block_size: usize,
		acc: &mut [f32],
		acc_start: usize,
	) {
		// Gather the input window in chronological order. Ring slots that have
		// not been written yet are still zero, which is exactly what the samples
		// before the start of the stream are, so early windows need no special
		// case.
		for i in 0..self.hop_blocks {
			let source = &history[((window_start + i) & history_mask) * block_size..][..block_size];
			self.work[i * block_size..][..block_size].copy_from_slice(source);
		}
		// The zero padding past the window still holds the previous evaluation
		// and has to be cleared before the forward transform.
		self.work[self.hop..].fill(0.0);

		self.forward_fft
			.process_with_scratch(&mut self.work, &mut self.spectrum, &mut self.scratch)
			.expect("the input window has the size the plan asked for");
		for (spectrum, ir) in self.spectrum.iter_mut().zip(self.ir_spectrum.iter()) {
			*spectrum *= *ir;
		}
		self.inverse_fft
			.process_with_scratch(&mut self.spectrum, &mut self.work, &mut self.scratch)
			.expect("the spectrum has the size the plan asked for");

		// Only hop + segment_len - 1 samples of the circular convolution are the
		// linear convolution, and the spectrum already carries the 1 / fft_size of
		// the unnormalized inverse transform.
		let end = (self.hop + self.segment_len - 1).min(self.fft_size);
		// Splitting the overlap-add where the accumulator ring wraps keeps both
		// halves contiguous, which is what lets the add vectorize.
		let first = (acc.len() - acc_start).min(end);
		for (acc, result) in acc[acc_start..].iter_mut().zip(self.work[..first].iter()) {
			*acc += *result;
		}
		let rest = end - first;
		for (acc, result) in acc[..rest].iter_mut().zip(self.work[first..end].iter()) {
			*acc += *result;
		}
	}
}

/// The streaming FFT convolution engine.
///
/// The convolver is zero latency. The first hop of the impulse response is
/// convolved directly, sample by sample, because a block transform can only
/// produce its first sample after a whole hop of input has been collected. The
/// rest of the impulse response is split into non-uniform partitions whose
/// length and hop both grow by powers of two, so one partition covers one octave
/// of the impulse response and is only evaluated once per octave of input; every
/// evaluation overlap-adds its result into a single output accumulator. Because
/// every partition keeps offset >= hop, its result starts exactly at the sample
/// that is played out next, which is what removes the block of latency a
/// partitioned convolver normally has.
///
/// This keeps the per-sample cost proportional to log(ir_len) plus the direct
/// head. Evaluating every partition on every block instead costs the whole
/// impulse response per block, that is O(ir_len) per sample, which is what a
/// partitioned convolver is supposed to avoid.
struct FftBuffer<
	const CHANNELS: usize = 2,
	const FFT_SIZE: usize = FFT_CONVOLVER_HISTORY_LEN,
> {
	/// The raw impulse response, kept for serialization.
	ir: [Vec<f32>; CHANNELS],
	/// The first FFT_SIZE taps of every channel in reverse, or the whole impulse
	/// response when it is shorter than one block: the direct head.
	head: [Vec<f32>; CHANNELS],
	/// The most recent 2 * FFT_SIZE input samples of every channel. Each sample is
	/// written twice, at 'pos' and at 'pos + FFT_SIZE', so that the head window is
	/// always contiguous and needs no wrapping.
	head_window: [Vec<f32>; CHANNELS],
	/// Per-channel partition structures.
	partitions: [Vec<FftPartition>; CHANNELS],
	/// Completed input blocks as a ring (history_mask + 1 blocks of FFT_SIZE).
	history: [Vec<f32>; CHANNELS],
	/// Ring position the next completed block is written to.
	history_pos: usize,
	/// Mask for the block ring.
	history_mask: usize,
	/// The input block currently being filled.
	current_block: [Vec<f32>; CHANNELS],
	/// Output accumulator ring. The partitions add into it, the samples are
	/// played out one at a time, and every played sample is cleared so that the
	/// next lap starts from silence.
	out_acc: [Vec<f32>; CHANNELS],
	/// Mask for the output accumulator ring.
	acc_mask: usize,
	/// Position within the input block being filled.
	pos: usize,
	/// Number of output samples already played out.
	emit_pos: usize,
	/// Number of completed input blocks.
	blocks_completed: usize,
}

impl<const CHANNELS: usize, const FFT_SIZE: usize> FftBuffer<CHANNELS, FFT_SIZE> {
	fn new(ir: [Vec<f32>; CHANNELS], _sample_rate: usize) -> Self {
		assert!(CHANNELS > 0, "CHANNELS must be greater than 0");
		let mut buffer = Self {
			ir,
			head: core::array::from_fn(|_| vec![]),
			head_window: core::array::from_fn(|_| vec![]),
			partitions: core::array::from_fn(|_| vec![]),
			history: core::array::from_fn(|_| vec![]),
			history_pos: 0,
			history_mask: 0,
			current_block: core::array::from_fn(|_| vec![0.0; FFT_SIZE]),
			out_acc: core::array::from_fn(|_| vec![]),
			acc_mask: 0,
			pos: 0,
			emit_pos: 0,
			blocks_completed: 0,
		};
		buffer.rebuild_partitions();
		buffer
	}

	/// Recompute the direct head and the partition layout from the current
	/// impulse response, and clear all running state.
	fn rebuild_partitions(&mut self) {
		let block_size = FFT_SIZE;
		let ir_len = self.ir.iter().map(|channel| channel.len()).max().unwrap_or(0);
		let max_hop = MAX_PARTITION_HOP.max(block_size);

		// The direct head covers the first block of every channel, in reverse.
		// A channel whose impulse response is shorter than a block is convolved
		// by the head alone and gets no partitions at all.
		self.head = core::array::from_fn(|channel| {
			let ir = &self.ir[channel];
			ir[..ir.len().min(block_size)].iter().rev().copied().collect()
		});
		self.head_window = core::array::from_fn(|_| vec![0.0; 2 * block_size]);

		let mut planner = RealFftPlanner::<f32>::new();
		self.partitions = core::array::from_fn(|channel| {
			let ir = &self.ir[channel];

			// The partitions pick up where the direct head stops, so the first
			// one starts at one block and every later offset is at least the hop
			// that covers it: the doubling lengths keep the coverage contiguous
			// and the offset >= hop invariant keeps the convolver zero latency.
			let mut layout = Vec::new();
			let mut offset = ir.len().min(block_size);
			let mut hop = block_size;
			while offset < ir.len() {
				let end = (offset + hop).min(ir.len());
				layout.push((offset, hop, end - offset));
				offset = end;
				hop = (hop * 2).min(max_hop);
			}

			let count = layout.len().max(1);
			layout.iter().enumerate().map(|(i, (offset, hop, segment_len))| {
				// Stagger the partitions across the longest cycle so that their
				// FFTs are spread over the blocks instead of all landing in the
				// same one. i * hop_blocks / count stays below hop_blocks, which
				// is what the alignment of a staggered partition needs.
				let hop_blocks = hop / block_size;
				let phase = i * hop_blocks / count;
				let segment = &ir[*offset..*offset + *segment_len];
				FftPartition::new(segment, *offset, *hop, phase, block_size, &mut planner)
			}).collect()
		});

		// The block ring only has to hold the longest window any partition
		// consumes. The accumulator has to hold the result span of one evaluation
		// plus the block that is played out next to it, so that a partition can
		// never wrap around onto samples that are still live.
		let max_hop_blocks = self.partitions.iter()
			.flatten()
			.map(|partition| partition.hop_blocks)
			.max()
			.unwrap_or(1);
		let max_span = self.partitions.iter()
			.flatten()
			.map(|partition| partition.hop + partition.segment_len)
			.max()
			.unwrap_or(0);
		let history_blocks = max_hop_blocks.next_power_of_two();
		let acc_len = (ir_len + block_size)
			.max(max_span + block_size + 1)
			.next_power_of_two();

		self.history_mask = history_blocks - 1;
		self.history = core::array::from_fn(|_| vec![0.0; history_blocks * block_size]);
		self.history_pos = 0;
		self.acc_mask = acc_len - 1;
		self.out_acc = core::array::from_fn(|_| vec![0.0; acc_len]);
		self.pos = 0;
		self.emit_pos = 0;
		self.blocks_completed = 0;
		for block in self.current_block.iter_mut() {
			block.fill(0.0);
		}
	}

	fn frame(&mut self, input: [f32; CHANNELS]) -> [f32; CHANNELS] {
		let mut output = [0.0; CHANNELS];
		let pos = self.pos;

		// The direct head convolves the sample that has just arrived against the
		// first block of the impulse response, so its contribution is available
		// immediately: this is the zero latency part of the output.
		for channel in 0..CHANNELS {
			let window = &mut self.head_window[channel];
			let sample = input[channel];

			let head = &self.head[channel];
			if !head.is_empty() {
				// The oldest taps read the window that ends one sample back and the
				// newest tap is applied straight to the sample that just arrived.
				// Reading a window that is a whole sample old is what keeps the
				// vector loads off the store below: loading a sample that was
				// stored a few cycles earlier stalls on store to load forwarding,
				// and that stall measured as expensive as the multiply-adds
				// themselves.
				let recent = &head[..head.len() - 1];
				let start = pos + FFT_SIZE - recent.len();
				output[channel] = head[head.len() - 1] * sample
					+ head_convolve(recent, &window[start..start + recent.len()]);
			}

			// The window is written after the convolution, so the next sample
			// finds this one where it expects it.
			window[pos] = sample;
			window[pos + FFT_SIZE] = sample;
		}

		// Add what the block partitions have already written for this sample.
		// The sample is cleared while it is played, so the next lap of the
		// accumulator ring starts from silence.
		let index = self.emit_pos & self.acc_mask;
		for (channel, output) in output.iter_mut().enumerate() {
			*output += self.out_acc[channel][index];
			self.out_acc[channel][index] = 0.0;
		}
		self.emit_pos += 1;

		// Collect the current input sample.
		for (channel, input) in input.iter().enumerate() {
			self.current_block[channel][pos] = *input;
		}
		self.pos += 1;

		// Once a full block has been collected, store it and run the partitions
		// whose hop period has elapsed. They can only write samples from here on,
		// which is what keeps the output causal.
		if self.pos == FFT_SIZE {
			self.pos = 0;
			self.blocks_completed += 1;
			self.compute_block();
		}

		output
	}

	/// Store the block that was just filled and let every partition whose hop
	/// period has elapsed add its contribution to the output accumulator.
	fn compute_block(&mut self) {
		let block_size = FFT_SIZE;

		// Store the block that was just filled; the ring keeps the most recent
		// blocks, which is exactly the longest window any partition consumes.
		let slot = self.history_pos;
		for channel in 0..CHANNELS {
			let start = slot * block_size;
			self.history[channel][start..start + block_size]
				.copy_from_slice(&self.current_block[channel]);
		}
		self.history_pos = (self.history_pos + 1) & self.history_mask;

		let blocks = self.blocks_completed;
		let emit_pos = self.emit_pos;
		let history_pos = self.history_pos;
		let history_mask = self.history_mask;
		let acc_mask = self.acc_mask;

		for channel in 0..CHANNELS {
			let history = &self.history[channel];
			let acc = &mut self.out_acc[channel];

			for partition in self.partitions[channel].iter_mut() {
				if !(blocks + partition.phase).is_multiple_of(partition.hop_blocks) {
					continue;
				}

				// This partition consumes its most recent hop_blocks blocks.
				let window_start =
					(history_pos + history_mask + 1 - partition.hop_blocks) & history_mask;

				// Convolving that window with the segment produces samples starting
				// offset - hop samples after the sample that is played out next.
				// Offset is never below hop, so those samples are in the future and
				// the convolver stays causal and zero latency.
				let acc_start = emit_pos + partition.offset - partition.hop;
				debug_assert!(partition.offset >= partition.hop);
				debug_assert!(partition.hop + partition.segment_len <= acc.len() - block_size);

				partition.process(
					history,
					window_start,
					history_mask,
					block_size,
					acc,
					acc_start & acc_mask,
				);
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

/// The Fft-based convolver.
///
/// It has no latency: the first 'FFT_SIZE' taps of the impulse response are
/// convolved directly in the time domain, so the wet path is available at the
/// same time as the input and the dry path needs no alignment. The direct head
/// costs 'FFT_SIZE' multiply-adds per sample per channel, which is why a small
/// FFT_SIZE is the cheap choice here: it shortens the head and only makes the
/// partition tail slightly more expensive, since one more octave of the impulse
/// response has to be transformed.
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
		// The first block is convolved directly in the time domain, so the wet
		// signal comes out of the same sample it goes in at.
		0
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
		// Both paths are zero latency, so the dry signal is mixed in as it is.
		let wet = self.buffer.frame(dry);

		for (i, sample) in samples.iter_mut().enumerate() {
			*sample = dry[i] * self.dry_gain + wet[i] * self.wet_gain;
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

	/// delta IR: the FFT convolver must reproduce the input sample for sample.
	/// This is the zero latency check.
	#[test]
	fn delta_ir_is_passthrough() {
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

		for n in 0..x.len() {
			let diff = (out[n] - x[n]).abs();
			assert!(diff < 1e-4, "delta passthrough mismatch at {n}: {}", diff);
		}
	}

	/// For a short IR that spans the direct head and several partitions, the FFT
	/// convolver must agree with a direct (!) convolution.
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

		let mut max_err = 0.0f32;
		for n in 0..x.len() {
			let err = (out[n] - reference[n]).abs();
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

	/// dry_gain and wet_gain must both be applied and stay time-aligned: the wet
	/// path is zero latency, so with a delta IR and gains of 1.0/1.0 the output is
	/// twice the input, sample for sample.
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

		for n in 0..x.len() {
			let diff = (out[n] - 2.0 * x[n]).abs();
			assert!(diff < 1e-4, "dry/wet mix mismatch at {n}: {}", diff);
		}
	}
	/// An impulse response that spans several partitions, including equal sized
	/// and staggered ones, must still match a direct convolution.
	#[test]
	fn staggered_partitions_match_direct_convolution() {
		let ir_len = 1500usize;
		let ir: Vec<f32> = (0..ir_len)
			.map(|i| (i as f32 * 0.11).sin() * (-(i as f32) / 300.0).exp())
			.collect();

		let mut conv = FftConvolver::<1, 8>::new([ir.clone()], 48_000);
		conv.dry_gain = 0.0;
		conv.wet_gain = 1.0;

		let x: Vec<f32> = (0..4096)
			.map(|i| (i as f32 * 0.017).sin() + 0.5 * (i as f32 * 0.31).cos())
			.collect();

		let mut out = vec![];
		let mut ctx: Box<dyn ProcessContext> = Box::new(());
		for &s in &x {
			let mut block = [s];
			conv.process(&mut block, &[], &mut ctx);
			out.push(block[0]);
		}

		let reference = direct_conv(&x, &ir);

		let mut max_err = 0.0f32;
		for n in 0..x.len() {
			max_err = max_err.max((out[n] - reference[n]).abs());
		}
		assert!(max_err < 1e-3, "the partitions disagree with direct: max err = {max_err}");
	}

	/// The production configuration (FFT_SIZE = 256 with the one second impulse
	/// responses the reverb presets use) must run without tripping the alignment
	/// and accumulator bounds assertions, and stay finite.
	#[test]
	fn long_ir_smoke() {
		let ir: Vec<f32> = (0..48_000)
			.map(|i| (i as f32 * 0.003).sin() * (-(i as f32) / 8000.0).exp())
			.collect();

		let mut conv = FftConvolver::<1, 256>::new([ir], 48_000);
		conv.dry_gain = 0.0;
		conv.wet_gain = 1.0;

		let mut ctx: Box<dyn ProcessContext> = Box::new(());
		for i in 0..20_000usize {
			let mut block = [(i as f32 * 0.01).sin()];
			conv.process(&mut block, &[], &mut ctx);
			assert!(block[0].is_finite());
		}
	}

	/// Run 'samples' samples through a convolver with the given block size and
	/// check the result against a direct convolution.
	fn check_block_size<const FS: usize>(ir: &[f32], samples: usize) {
		let mut conv = FftConvolver::<1, FS>::new([ir.to_vec()], 48_000);
		conv.dry_gain = 0.0;
		conv.wet_gain = 1.0;

		let x: Vec<f32> = (0..samples)
			.map(|i| (i as f32 * 0.023).sin() + 0.4 * (i as f32 * 0.7).cos())
			.collect();

		let mut out = vec![];
		let mut ctx: Box<dyn ProcessContext> = Box::new(());
		for &s in &x {
			let mut block = [s];
			conv.process(&mut block, &[], &mut ctx);
			out.push(block[0]);
		}

		let reference = direct_conv(&x, ir);
		let mut max_err = 0.0f32;
		for n in 0..x.len() {
			max_err = max_err.max((out[n] - reference[n]).abs());
		}
		assert!(max_err < 1e-3, "FFT_SIZE = {FS} disagrees with direct: max err = {max_err}");
	}

	/// Block sizes that are not powers of two have to work as well: the block
	/// ring and the output accumulator are rounded up to powers of two
	/// independently of FFT_SIZE, and a partition cycle is only ever a multiple
	/// of the block size.
	#[test]
	fn non_power_of_two_block_size() {
		let ir: Vec<f32> = (0..100)
			.map(|i| (i as f32 * 0.19).sin() * (-(i as f32) / 40.0).exp())
			.collect();
		check_block_size::<12>(&ir, 512);
		check_block_size::<13>(&ir, 512);
	}

	/// A single deep tap must come out at exactly the tap index, with no extra
	/// block of latency. This pins down the alignment of the staggered deep
	/// partitions.
	#[test]
	fn deep_tap_keeps_its_offset() {
		let mut ir = vec![0.0f32; 1500];
		ir[900] = 1.0;

		let mut conv = FftConvolver::<1, 8>::new([ir], 48_000);
		conv.dry_gain = 0.0;
		conv.wet_gain = 1.0;

		let x: Vec<f32> = (0..3000)
			.map(|i| ((i * 37 % 101) as f32) / 101.0 - 0.5)
			.collect();

		let mut out = vec![];
		let mut ctx: Box<dyn ProcessContext> = Box::new(());
		for &s in &x {
			let mut block = [s];
			conv.process(&mut block, &[], &mut ctx);
			out.push(block[0]);
		}

		let tap = 900;
		for n in tap..x.len() {
			let diff = (out[n] - x[n - tap]).abs();
			assert!(diff < 1e-4, "deep tap misaligned at {n}: {}", diff);
		}
	}

	/// The partition cap has to bound the largest transform even for impulse
	/// responses much longer than it, the split has to cover the impulse response
	/// without gaps, and no partition may start before its own hop: that last
	/// invariant is what makes the convolver zero latency.
	#[test]
	fn partition_layout_is_bounded_causal_and_complete() {
		let ir = vec![0.1f32; MAX_PARTITION_HOP * 4];
		let buffer = FftBuffer::<1, 256>::new([ir], 48_000);
		let partitions = &buffer.partitions[0];

		assert_eq!(buffer.head[0].len(), 256, "the head covers one block");
		assert!(
			partitions.len() < 16,
			"the split should stay logarithmic, got {} partitions",
			partitions.len()
		);
		assert_eq!(partitions[0].offset, 256, "the tail starts where the head stops");

		let mut covered = buffer.head[0].len();
		for partition in partitions {
			assert_eq!(partition.offset, covered, "the split has a gap or an overlap");
			assert!(
				partition.offset >= partition.hop,
				"offset {} below hop {}",
				partition.offset,
				partition.hop
			);
			assert!(partition.hop <= MAX_PARTITION_HOP, "hop {} above the cap", partition.hop);
			assert!(
				partition.fft_size <= 2 * MAX_PARTITION_HOP,
				"fft {} above the bound",
				partition.fft_size
			);
			covered += partition.segment_len;
		}
		assert_eq!(covered, MAX_PARTITION_HOP * 4, "the split is incomplete");
	}
}
