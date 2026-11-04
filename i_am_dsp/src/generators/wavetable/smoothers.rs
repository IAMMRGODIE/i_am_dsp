//! A wave table that morphs between several tables.

use i_am_dsp_derive::Parameters;

use crate::prelude::{Parameter, Parameters};

use super::{Oscillator, WaveTable};

/// A wave table that smoothly transitions between multiple wave tables.
#[derive(Parameters)]
pub struct WaveTableSmoother {
	#[sub_param]
	/// The wave tables to smooth between.
	pub tables: Vec<Box<dyn WaveTable + Send + Sync>>,
	/// should be between 0 and 1, where 0 is first table and 1 is last table.
	#[range(min = 0.0, max = 1.0)]
	#[default(0.0)]
	#[id(display = "Morph")]
	pub smooth_factor: f32,

	#[skip]
	#[cfg(feature = "real_time_demo")]
	split_amount: usize,
	#[skip]
	#[cfg(feature = "real_time_demo")]
	error: Option<String>,
	#[skip]
	#[cfg(feature = "real_time_demo")]
	allow_change_table: bool,
	#[skip]
	#[cfg(feature = "real_time_demo")]
	opened_file: Option<std::path::PathBuf>,
	#[skip]
	#[cfg(feature = "real_time_demo")]
	openfile_dialog: Option<egui_file::FileDialog>,
}

impl Parameters for Vec<Box<dyn WaveTable + Send + Sync>> {
	fn get_parameters(&self) -> Vec<Parameter> {
		let mut output = vec![];
		for (i, table) in self.iter().enumerate() {
			for mut param in table.get_parameters() {
				param.identifier = format!("{}.{}", i, param.identifier);
				output.push(param);
			}
		}
		output
	}

	fn set_parameter(&mut self, identifier: &str, value: crate::prelude::SetValue) -> bool {
		let mut parts = identifier.split(".").collect::<Vec<_>>();
		let index = parts
			.remove(0)
			.parse::<usize>()
			.expect("Invalid identifier");
		let param_name = parts.join(".");
		if let Some(table) = self.get_mut(index) {
			table.set_parameter(&param_name, value)
		} else {
			false
		}
	}
}

impl WaveTable for WaveTableSmoother {
	fn sample(&self, t: f32, channel: usize) -> f32 {
		let count = self.tables.len();

		if count == 0 {
			return 0.0;
		} else if count == 1 {
			return self.tables[0].sample(t, channel);
		}

		let last = count - 1;
		let smooth = (self.smooth_factor * last as f32).clamp(0.0, last as f32);
		let from = smooth.floor() as usize;
		let from_sample = self.tables[from].sample(t, channel);
		let fraction = smooth - from as f32;

		// The common case is a morph sitting exactly on one table: sampling the
		// second table and throwing the result away would double the work.
		if fraction <= f32::EPSILON || from == last {
			return from_sample;
		}

		let to_sample = self.tables[from + 1].sample(t, channel);
		from_sample + (to_sample - from_sample) * fraction
	}
}

impl WaveTableSmoother {
	/// Create a new [`WaveTableSmoother`] from a vector of wave tables and a smooth factor.
	pub fn new(tables: Vec<Box<dyn WaveTable + Send + Sync>>, smooth_factor: f32) -> Self {
		Self {
			tables,
			smooth_factor,

			#[cfg(feature = "real_time_demo")]
			split_amount: 256,
			#[cfg(feature = "real_time_demo")]
			error: None,
			#[cfg(feature = "real_time_demo")]
			allow_change_table: false,
			#[cfg(feature = "real_time_demo")]
			opened_file: None,
			#[cfg(feature = "real_time_demo")]
			openfile_dialog: None,
		}
	}

	/// Sample the wave table at time t, t in the range [0, 1]
	///
	/// Due to rust's limitation, we can't implement [`WaveTable`] for `WaveTableSmoother`.
	pub fn sample(&self, t: f32, channel: usize) -> f32 {
		let count = self.tables.len();

		if count == 0 {
			return 0.0;
		} else if count == 1 {
			return self.tables[0].sample(t, channel);
		}

		let last = count - 1;
		let smooth = (self.smooth_factor * last as f32).clamp(0.0, last as f32);
		let from = smooth.floor() as usize;
		let from_sample = self.tables[from].sample(t, channel);
		let fraction = smooth - from as f32;

		// The common case is a morph sitting exactly on one table: sampling the
		// second table and throwing the result away would double the work.
		if fraction <= f32::EPSILON || from == last {
			return from_sample;
		}

		let to_sample = self.tables[from + 1].sample(t, channel);
		from_sample + (to_sample - from_sample) * fraction
	}

	/// Create a new [`WaveTableSmoother`] from a vector of wave tables.
	///
	/// The wave tables will be split into frames of the given length,
	/// will add zeroes to the end of the last frame if necessary,
	pub fn split_from_vec(vec: Vec<f32>, frame_len: usize) -> Self {
		let tables = Self::split_vec(vec, frame_len);
		Self {
			tables,
			smooth_factor: 0.0,

			#[cfg(feature = "real_time_demo")]
			split_amount: 256,
			#[cfg(feature = "real_time_demo")]
			error: None,
			#[cfg(feature = "real_time_demo")]
			allow_change_table: false,
			#[cfg(feature = "real_time_demo")]
			opened_file: None,
			#[cfg(feature = "real_time_demo")]
			openfile_dialog: None,
		}
	}

	fn split_vec(mut vec: Vec<f32>, frame_len: usize) -> Vec<Box<dyn WaveTable + Send + Sync>> {
		let len = vec.len();
		let frame_count = len / frame_len;
		let frame_count = if frame_count * frame_len < len {
			frame_count + 1
		} else {
			frame_count
		};
		vec.resize(frame_count * frame_len, 0.0);
		let mut tables: Vec<Box<dyn WaveTable + Send + Sync>> = vec![];
		for i in 0..frame_count {
			let start = i * frame_len;
			let end = start + frame_len;
			tables.push(Box::new(vec[start..end].to_vec()));
		}
		tables
	}
}

impl<const CHANNELS: usize> Oscillator<CHANNELS> for WaveTableSmoother {
	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		use crate::tools::pcm_data::load_from_file;
		use crate::tools::ui_tools::draw_wavetable;
		use egui::Slider;

		let mut clear_error = false;
		if let Some(error) = self.error.as_ref() {
			ui.colored_label(egui::Color32::RED, error);
			clear_error = ui.button("Clear Error").clicked();
		}

		if clear_error {
			self.error = None;
		}

		egui::Resize::default()
			.resizable([false, true])
			.min_width(ui.available_width())
			.max_width(ui.available_width())
			.id_salt(format!("{id_prefix}_wavtable_smoother"))
			.show(ui, |ui| {
				draw_wavetable(ui, |t| self.sample(t, 0));
			});

		ui.horizontal(|ui| {
			let mut path = None;

			if self.allow_change_table {
				ui.input(|input| {
					path = input
						.raw
						.dropped_files
						.first()
						.map(|inner| inner.path().to_path_buf());
				});
			}

			if ui.button("Load Sample").clicked() {
				use egui_file::FileDialog;
				use std::ffi::OsStr;

				let filter = Box::new({
					let ext = Some(OsStr::new("wav"));
					move |path: &std::path::Path| -> bool { path.extension() == ext }
				});
				let mut dialog = FileDialog::open_file().show_files_filter(filter);
				if let Some(opened_file) = &self.opened_file {
					dialog = dialog.initial_path(
						opened_file
							.parent()
							.map(|inner| inner.to_path_buf())
							.unwrap_or(std::path::PathBuf::from(".")),
					)
				}
				dialog.open();

				self.openfile_dialog = Some(dialog);
			}

			if let Some(dialog) = self.openfile_dialog.as_mut() {
				let dialog = dialog.show(ui.ctx());
				if dialog.selected() {
					path = dialog.path().map(|path| path.to_path_buf());
				}
			}

			if let Some(path) = path {
				if path
					.extension()
					.map(|ext| ext.to_string_lossy().to_lowercase() != "wav")
					.unwrap_or(true)
				{
					return;
				}

				self.opened_file = Some(path.clone());

				let data = match load_from_file::<CHANNELS>(path) {
					Ok(data) => data,
					Err(e) => {
						self.error = Some(format!("ERR: {}", e));
						return;
					}
				};
				let Some(len) = data.pcm_data.iter().map(|inner| inner.len()).min() else {
					self.error = Some("ERR: No data found in file".to_string());
					return;
				};
				let wave_form_data = (0..len)
					.map(|i| {
						let mut sum = 0.0;
						for channel in data.pcm_data.iter() {
							sum += channel[i];
						}
						sum / data.pcm_data.len() as f32
					})
					.collect::<Vec<_>>();
				self.tables = Self::split_vec(wave_form_data, self.split_amount);
				self.allow_change_table = false;
			}
			ui.add(Slider::new(&mut self.split_amount, 16..=4096).text("Split Size"));
			if ui
				.selectable_label(self.allow_change_table, "Allow Change Table")
				.clicked()
			{
				self.allow_change_table = !self.allow_change_table;
			}
		});

		ui.horizontal(|ui| {
			ui.add(egui::Slider::new(&mut self.smooth_factor, 0.0..=1.0).text("Smooth Factor"));
		});
	}

	fn play_at(&self, frequency: f32, time: f32, phase: [f32; CHANNELS]) -> [f32; CHANNELS] {
		let mut output = [0.0; CHANNELS];

		for i in 0..CHANNELS {
			let t = (time * frequency + phase[i]) % 1.0;
			output[i] = self.sample(t, i);
		}

		output
	}
}
