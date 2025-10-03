//! implementation of the Sampler.

use std::{path::Path, thread::JoinHandle};

use crossbeam_channel::Receiver;
use rubato::{Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction};

use crate::{prelude::Oscillator, tools::{interpolate::cubic_interpolate, load_pcm_data::{load_from_file, ReadFileError}}, Generator, ProcessContext};

/// A sampler that plays back audio files.
/// 
/// Note: this sampler cannot accept MIDI input.
/// For MIDI support, use [`crate::prelude::Adsr`] warp it.
pub struct Sampler<const CHANNELS: usize = 2> {
	pcm_data: Option<[Vec<f32>; CHANNELS]>,
	current_position: usize,
	sample_rate: usize,
	/// The loop range of the sample. Saves in samples.
	/// 
	/// None for no loop.
	loop_range: Option<(usize, usize)>,
	/// Whether the sample is currently playing.
	/// 
	/// This param will only be used if current sampler is used as [`crate::prelude::Generator`].
	pub is_playing: bool,
	/// The volume of the sample, saves in linear scale.
	pub volume: f32,
	/// Whether to play the sample in reverse.
	pub reverse: bool,
	/// Whether to reverse the polarity of the sample.
	pub reverse_polarity: bool,
	/// The playback speed of the sample.
	pub speed: f32,
	/// The random start position range of the sample. Saves in milliseconds
	/// 
	/// This param will only be used if current sampler is used as [`crate::prelude::Oscillator`].
	pub phase_range: f32,
	/// Whether to play the sample only once.
	/// 
	/// This param will only be used if current sampler is used as [`crate::prelude::Oscillator`].
	pub one_shot: bool,

	#[cfg(feature = "real_time_demo")]
	gui_state: GuiState<CHANNELS>,
}

/// A handle for the resampling process.
/// 
/// This handle can be used to check the progress of the resampling process.
pub struct ResampleHandle<const CHANNELS: usize> {
	/// The thread handle for the resampling process.
	pub thread_handle: JoinHandle<Result<[Vec<f32>; CHANNELS], ReadFileError>>,
	/// How many samples are processed.
	pub process_receiver: Receiver<usize>,
	/// The total number of samples in the input file.
	pub total_samples: usize,
}

#[derive(Default)]
#[cfg(feature = "real_time_demo")]
struct GuiState<const CHANNELS: usize> {
	error: Option<String>,
	resample_process: Option<ResampleHandle<CHANNELS>>,
	last_position: Option<usize>,
	show_time_as_samples: bool,
	drag_start_at: usize,
	resample_ratio: f32,
}

impl<const CHANNELS: usize> Sampler<CHANNELS> {
	/// Create a new `Sampler` with the given sample rate.
	/// 
	/// # Panics
	/// 
	/// Panics if `CHANNELS` is 0.
	pub fn new(sample_rate: usize) -> Self {
		if CHANNELS == 0 {
			panic!("CHANNELS must be greater than 0");
		}

		Self {
			pcm_data: None,
			current_position: 0,
			volume: 1.0,
			sample_rate,
			is_playing: false,
			loop_range: None,
			reverse: false,
			reverse_polarity: false,
			speed: 1.0,
			one_shot: false,
			phase_range: 150.0 * 1000.0 / sample_rate as f32,

			#[cfg(feature = "real_time_demo")]
			gui_state: Default::default(),
		}
	}

	/// Set the playback speed of the sample.
	pub fn set_speed(&mut self, speed: f32) {
		self.speed = speed;
	}

	/// Get current position in samples, but may not be integer because of the speed.
	pub fn current_position(&self) -> f32 {
		self.current_position as f32 * self.speed
	}

	/// Get current position in seconds.
	pub fn current_time(&self) -> f32 {
		self.current_position as f32 / self.sample_rate as f32 * self.speed
	}

	fn resample(mut pcm_data: [Vec<f32>; CHANNELS], audio_sample_rate: usize, sample_rate: usize) -> ResampleHandle<CHANNELS> {
		let total_samples = pcm_data[0].len();
		let (sender, receiver) = crossbeam_channel::unbounded();
		let ratio = sample_rate as f64 / audio_sample_rate as f64;
		let thread_handle = std::thread::spawn(move || {
			let resample_params = SincInterpolationParameters {
				sinc_len: 256,
				f_cutoff: 0.95,
				interpolation: SincInterpolationType::Linear,
				oversampling_factor: 256,
				window: WindowFunction::BlackmanHarris2,
			};

			let mut resampler = SincFixedIn::<f32>::new(
				ratio,
				1.1,
				resample_params,
				4096,
				CHANNELS,
			)?;

			// println!("In Sample Length: {}", pcm_data[0].len());

			let mut output_buffer = [const { Vec::new() }; CHANNELS];

			let mut temp_buffer = [const { Vec::new() }; CHANNELS];
			let mut input_frames_next = resampler.input_frames_next();
			let max_output = resampler.output_frames_max();
			let mut rest_buffer: Vec<&[f32]> = vec!();
			for buffer in &pcm_data {
				rest_buffer.push(buffer);
			}
			for buffer in &mut temp_buffer {
				*buffer = vec![0.0_f32; max_output];
			}

			let mut processed = 0;

			while rest_buffer[0].len() >= input_frames_next {
				let (nbr_in, nbr_out) = resampler.process_into_buffer(
					&rest_buffer, 
					&mut temp_buffer, 
					None
				)?;

				for buffer in &mut rest_buffer {
					*buffer = &buffer[nbr_in..];
				}

				for (i, data) in temp_buffer.iter().enumerate() {
					output_buffer[i].extend_from_slice(&data[..nbr_out]);
				}
				input_frames_next = resampler.input_frames_next();
				processed += nbr_in;
				let _ = sender.send(processed);
			}

			if !rest_buffer[0].is_empty() {
				let (_nbr_in, nbr_out) = resampler
					.process_partial_into_buffer(Some(&rest_buffer), &mut temp_buffer, None)?;
				for (i, data) in temp_buffer.iter().enumerate() {
					output_buffer[i].extend_from_slice(&data[..nbr_out]);
				}
			}

			
			for (i, data) in output_buffer.into_iter().enumerate() {
				pcm_data[i] = data.to_vec();
			}

			Ok(pcm_data)
		});

		ResampleHandle { 
			thread_handle, 
			process_receiver: receiver, 
			total_samples,
		}
	}

	/// Resample the current sample to the given sample rate.
	/// 
	/// User should manually load pcm data using [`Self::set_pcm_data`] method.
	/// 
	/// None returned if their's no pcm data loaded.
	pub fn resample_current_sample(&mut self, sample_rate: usize) -> Option<ResampleHandle<CHANNELS>> {
		if sample_rate == self.sample_rate {
			return None;
		}
		let pcm_data = self.pcm_data.take()?;
		Some(Self::resample(pcm_data, self.sample_rate, sample_rate))
	}

	/// Set the sample rate of the current sample.
	/// 
	/// Will **not** resample the current sample to the new sample rate.
	pub fn set_sample_rate(&mut self, sample_rate: usize) {
		self.sample_rate = sample_rate;
	}

	/// Only 2-CHANNELS Wav files are supported for now.
	/// 
	/// Will return a handle if the file need to be resample to current sample rate, 
	/// and user should manually load pcm data using [`Self::set_pcm_data`] method.
	/// 
	/// For more details, see [`crate::tools::load_pcm_data::load_from_file`].
	pub fn load_from_file(&mut self, path: impl AsRef<Path>) -> Result<Option<ResampleHandle<CHANNELS>>, ReadFileError> {
		let pcm_data = load_from_file::<CHANNELS>(path)?;
		let audio_sample_rate = pcm_data.sample_rate;

		// println!("audio_sample_rate: {}, self.sample_rate: {}", audio_sample_rate, self.sample_rate);

		if audio_sample_rate != self.sample_rate {
			Ok(Some(Self::resample(pcm_data.pcm_data, audio_sample_rate, self.sample_rate)))
		}else {
			self.set_pcm_data(pcm_data.pcm_data);
			Ok(None)
		}
	}

	/// Load pcm data into the current sampler.
	pub fn set_pcm_data(&mut self, pcm_data: [Vec<f32>; CHANNELS]) {
		self.pcm_data = Some(pcm_data);
		self.current_position = 0;
	}

	/// Drop the current pcm data.
	pub fn drop_pcm_data(&mut self) {
		self.pcm_data = None;
		self.current_position = 0;
	}

	/// Set the loop range of the current sample.
	/// 
	/// Panics if `start` is greater than or equal to `end`.
	pub fn set_loop_range(&mut self, start: usize, end: usize) {
		assert!(start < end);
		self.loop_range = Some((start, end));
	}

	/// Remove the loop range of the current sample.
	pub fn remove_loop_range(&mut self) {
		self.loop_range = None;
	}

	/// Play the current sample.
	pub fn play(&mut self) {
		self.is_playing = true;
	}

	/// Stop the current sample. This will reset the current position to 0.
	pub fn stop(&mut self) {
		self.is_playing = false;
		self.current_position = 0;
	}

	/// Pause the current sample.
	pub fn pause(&mut self) {
		self.is_playing = false;
	}

	/// An auxiliary function to use current sampler as an Oscillator.
	pub fn sample_at(&mut self, mut position: f32, scale: f32, phase: [f32; CHANNELS]) -> [f32; CHANNELS] {
		let delay_range = self.phase_range / 1000.0 * self.sample_rate as f32;
		
		position += self.current_position();

		let Some(pcm_data) = &self.pcm_data else {
			return [0.0; CHANNELS];
		};

		if pcm_data[0].is_empty() {
			return [0.0; CHANNELS];
		}

		if let Some((start, end)) = &self.loop_range {
			if start >= end {
				// self.stop();
				return [0.0; CHANNELS];
			}

			let start = *start as f32 / self.speed;
			let end = *end as f32 / self.speed;

			if position < start {
				position = start;
			}else if position >= end {
				position %= end - start;
				position += start;
			}
		}
		let mut output = [0.0; CHANNELS];

		for (i, vals) in pcm_data.iter().enumerate() {
			let position = (if self.reverse {
				(vals.len() - 1) as f32 - position
			}else {
				position
			} + phase[i] * delay_range) * scale * self.speed;
			let position = if self.one_shot {
				position.clamp(0.0, vals.len() as f32 - 2.0)
			}else {
				position % (vals.len() - 1) as f32
			};
	
			let (start, end) = (position.floor() as usize, position.ceil() as usize);
			let before = if start == 0 { *vals.last().unwrap()  } else { vals[start - 1] };
			let after = if end >= vals.len() { vals[0] } else { vals[end] };

			let val = if position == start as f32 {
				vals[start]
			}else {
				cubic_interpolate(
					position.fract(),
					[before, vals[start], vals[end], after],
				)
			};

			output[i] = val * self.volume * if self.reverse_polarity { -1.0 } else { 1.0 };
		}
			
		output
	}

	#[cfg(feature = "real_time_demo")]
	fn generate_ui(&mut self, ui: &mut egui::Ui, id_prefix: String, is_resampler: bool) {
		use egui::*;
		use crate::tools::ui_tools::draw_waveform;
		// use rayon::prelude::*;

		if let Some(error) = &self.gui_state.error {
			ui.colored_label(egui::Color32::RED, error);
			if ui.button("clear error").clicked() {
				self.gui_state.error = None;
			}
		}

		let mut need_load_pcm = false;
		let mut cancel_resample = false;

		if let Some(handle) = &self.gui_state.resample_process {
			if let Ok(last_position) = handle.process_receiver.try_recv() {
				self.gui_state.last_position = Some(last_position);
			}
			
			ui.label("Resampling...");
			if let Some(last_position) = &self.gui_state.last_position {
				ui.label(format!("prcessed {} / {} samples", last_position, handle.total_samples));
				ui.label(format!("percentage {:.2}%", *last_position as f64 / handle.total_samples as f64 * 100.0));
			}

			if ui.button("Cancel").clicked() {
				cancel_resample = true;
			}

			need_load_pcm = handle.thread_handle.is_finished();
		}

		if cancel_resample {
			self.gui_state.resample_process = None;
		}

		if need_load_pcm {
			if let Some(handle) = self.gui_state.resample_process.take() {
				let handle = match handle.thread_handle.join() {
					Ok(Ok(inner)) => inner,
					Ok(Err(e)) => {
						self.gui_state.error = Some(format!("{}", e));
						return;
					},
					Err(_) => {
						self.gui_state.error = Some("Thread Error Occured".to_string());
						return;
					},
				};
				self.set_pcm_data(handle)
			}
		}

		let Some(pcm_data) = &self.pcm_data else {
			if self.gui_state.resample_process.is_none() {
				ui.label("No PCM data loaded");
			
				if ui.button("Load PCM data").clicked() {
					let dialog = rfd::FileDialog::new().add_filter("Wave files", &["wav"]);
					if let Some(path) = dialog.pick_file() {
						match self.load_from_file(path) {
							Ok(inner) => self.gui_state.resample_process = inner,
							Err(e) => self.gui_state.error = Some(format!("{}", e)),
						}
					}
				}
			}
			
			return;
		};
		let total_samples = pcm_data[0].len();
		// let current_sample = self.current_position;
		let current_position = self.current_position();

		egui::Resize::default().resizable([false, true])
			.min_width(ui.available_width())
			.max_width(ui.available_width())
			.id_salt(format!("{id_prefix}_sampler_waveform"))
			.show(ui, |ui| 
		{
			let response = draw_waveform(
				ui, 
				Some(current_position), 
				pcm_data, 
				&self.loop_range, 
				self.reverse,
				!is_resampler
			);
			let rect = response.rect;

			if response.dragged_by(PointerButton::Primary) || response.clicked() {
				if let Some(position) = ui.ctx().input(|state| {
					state.pointer.hover_pos()
				}) {
					let position = position.x - rect.min.x;
					self.current_position = (position / rect.width() * total_samples as f32 / self.speed) as usize;
					self.loop_range = None;
				}
			}

			if response.dragged_by(PointerButton::Secondary) {
				if let Some(position) = ui.ctx().input(|state| {
					state.pointer.hover_pos()
				}) {
					let position = position.x - rect.min.x;
					let current_position = (position / rect.width() * total_samples as f32) as usize;
					if response.drag_started_by(PointerButton::Secondary) {
						self.gui_state.drag_start_at = current_position;
					}

					let min = current_position.min(self.gui_state.drag_start_at);
					let max = current_position.max(self.gui_state.drag_start_at);
					self.loop_range = Some((min, max));
					if self.current_position >= max {
						self.current_position = (max as f32 / self.speed).floor() as usize;
					}else if self.current_position < min {
						self.current_position = (min as f32 / self.speed).ceil() as usize;
					}
				}
			}
		});

		ScrollArea::horizontal().show(ui, |ui| {		
			ui.horizontal(|ui| {			
				if ui.selectable_label(self.reverse, "Reverse").clicked() {
					self.reverse = !self.reverse;
				}
	
				if ui.selectable_label(self.reverse_polarity, "Reverse Polarity").clicked() {
					self.reverse_polarity = !self.reverse_polarity;
				}

				if is_resampler {
					let label = ui.selectable_label(self.one_shot, "One Shot");
					if label.clicked() {
						self.one_shot = !self.one_shot;
					}
					ui.add(Slider::new(&mut self.phase_range, 0.0..=1000.0).text("Phase Range(ms)"));
					return;
				}

				let play_btn_text = if self.is_playing { "Pause" } else { "Play" };
	
				let btn = ui.button(play_btn_text);
	
				if btn.double_clicked() {
					self.stop();
				}else if btn.clicked() | ui.input(|input| input.key_pressed(Key::Space)) {
					self.is_playing = !self.is_playing;
				}
	
				if ui.selectable_label(self.gui_state.show_time_as_samples, "Show as Samples").clicked() {
					self.gui_state.show_time_as_samples = !self.gui_state.show_time_as_samples;
				}
	
	
				if self.gui_state.show_time_as_samples {
					// let mut current_position = self.current_position().round() as usize;
					// ui.add(DragValue::new(&mut current_position)
					// 	.range(0..=total_samples)
					// 	.custom_formatter(|value, _| {
					// 		format!("{:.0}", value, )
					// 	})
					// );

					// self.current_position = (current_position as f32 / self.speed) as usize;

					ui.label(format!("{:.2} / {} samples", self.current_position(), total_samples));
				}else {
					// let current_samples = self.current_position();
					// let mut current_time = current_samples / self.sample_rate as f32;
					let total_time = total_samples as f32 / self.sample_rate as f32;
					// ui.add(DragValue::new(&mut current_time)
					// 	.range(0.0..=total_time)
					// 	.custom_formatter(|value, _| {
					// 		format!("{:.2}", value)
					// 	})
					// );
	
					ui.label(format!("{:.2}s / {:.2}s", self.current_time(), total_time));
				}
			});
	
			ui.horizontal(|ui| {
				ui.add(egui::Slider::new(&mut self.speed, 0.25..=4.0).text("Speed").logarithmic(true));
				ui.add(egui::Slider::new(&mut self.volume, 0.0..=2.0).text("Volume"));
				if ui.button("Drop PCM data").clicked() {
					self.drop_pcm_data();
				}
	
				if ui.button("Reload PCM data").clicked() {
					let dialog = rfd::FileDialog::new().add_filter("Wave files", &["wav"]);
					if let Some(path) = dialog.pick_file() {
						match self.load_from_file(path) {
							Ok(inner) => self.gui_state.resample_process = inner,
							Err(e) => self.gui_state.error = Some(format!("{}", e)),
						}
					}
				}
			});

			ui.horizontal(|ui| {
				if self.gui_state.resample_ratio <= 0.0 {
					self.gui_state.resample_ratio = 1.0;
				}
				ui.label(format!("Sample Rate: {} Hz", self.sample_rate));
				ui.add(
					Slider::new(&mut self.gui_state.resample_ratio, 2.0_f32.powi(-4)..=2.0_f32.powi(4)).logarithmic(true)
				);
				if ui.button("Resample").clicked() {
					let new_sample_rate = (self.sample_rate as f32 * self.gui_state.resample_ratio) as usize;
					self.gui_state.resample_process = self.resample_current_sample(new_sample_rate);
				}
			})
		});
	}
}

impl<const CHANNELS: usize> Generator<CHANNELS> for Sampler {
	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"Sampler"
	}

	fn generate(&mut self, _: &mut Box<dyn ProcessContext>) -> [f32; CHANNELS] {
		if let Some(pcm_data) = &self.pcm_data {
			if !self.is_playing {
				return [0.0; CHANNELS];
			}

			if let Some((start, end)) = &self.loop_range {
				if start >= end {
					// self.stop();
					return [0.0; CHANNELS];
				}

				let start = (*start as f32 / self.speed).ceil() as usize;
				let end = (*end as f32 / self.speed).floor() as usize;

				if self.current_position >= end || self.current_position < start {
					self.current_position = start;
				}

			}

			let mut output = [0.0; CHANNELS];

			if self.current_position() >= pcm_data[0].len() as f32 {
				self.current_position = 0;
			}

			for (i, vals) in pcm_data.iter().enumerate() {
				let position = if self.reverse {
					vals.len() - 1 - self.current_position
				}else {
					self.current_position
				};
				let position = position as f32 * self.speed;
				if position >= vals.len() as f32 {
					output[i] = 0.0;
					continue;
				}

				let (start, end) = (position.floor() as usize, position.ceil() as usize);
				let before = if start == 0 { 0.0 } else { vals[start - 1] };
				let after = if end >= vals.len() { 0.0 } else { vals[end] };

				let val = if position == start as f32 {
					vals[start]
				}else {
					cubic_interpolate(
						position.fract(),
						[before, vals[start], vals[end], after],
					)
				};

				output[i] = val * self.volume * if self.reverse_polarity { -1.0 } else { 1.0 };
			}

			self.current_position += 1;

			if self.current_position() >= pcm_data[0].len() as f32 {
				self.current_position = 0;
				self.stop();
			}

			output
		}else {
			[0.0; CHANNELS]
		}
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		self.generate_ui(ui, id_prefix, false);
	}
}

impl<const CHANNELS: usize> Oscillator<CHANNELS> for Sampler<CHANNELS> {
	fn play_at(&mut self, frequency: f32, time: f32, phase: [f32; CHANNELS]) -> [f32; CHANNELS] {
		const C5_HZ: f32 = 523.251_16;
		let factor = frequency / C5_HZ;
		let position = time * self.sample_rate as f32;
		self.sample_at(position, factor, phase)
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		self.generate_ui(ui, id_prefix, true);
	}
}