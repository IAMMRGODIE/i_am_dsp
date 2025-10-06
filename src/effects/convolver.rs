//! A simple convolver implementation.

use std::{f32::consts::PI, ops::Range};

use i_am_parameters_derive::Parameters;

use crate::{tools::{format_usize, parse_usize, ring_buffer::RingBuffer}, Effect, ProcessContext};

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

	if data.len() % std::mem::size_of::<f32>() != 0 {
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
/// Note: The time complexity of this convolver is O(n*m), for o(l log l) implementation, see [FftConvolver(TODO)].
#[derive(Parameters)]
pub struct Convolver<const CHANNELS: usize = 2> {
	#[persist(serialize = "format_ir", deserialize = "parse_ir")]
	ir: [Vec<f32>; CHANNELS],
	#[skip]
	history: [RingBuffer<f32>; CHANNELS],
	#[persist(serialize = "format_usize", deserialize = "parse_usize")]
	delay: usize,

	#[cfg(feature = "real_time_demo")]
	#[skip]
	gui_state: (DelyaCaculateMode, Option<String>),

	#[cfg(feature = "real_time_demo")]
	#[skip]
	allow_change_ir: bool
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
		use crate::tools::load_pcm_data::load_from_file;
		use crate::tools::load_pcm_data::PcmOutput;
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
			.id_source(format!("{id_prefix}_convolver"))
			.show(ui, |ui| 
		{
			draw_waveform(ui, None, &self.ir, &None, false, false);
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

				#[cfg(feature = "rfd")]
				if ui.button("replace ir").clicked() {
					let dialog = rfd::FileDialog::new().add_filter("Wave files", &["wav"]);
					path = dialog.pick_file();
				}

				if let Some(path) = path {
					if path.extension().map(|ext| ext.to_string_lossy().to_lowercase() != "wav").unwrap_or(true) {
						return;
					}
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