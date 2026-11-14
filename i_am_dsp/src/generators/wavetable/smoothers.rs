//! A wave table that morphs between several tables.

use i_am_dsp_derive::Parameters;

use crate::prelude::{Parameter, Parameters, page_position};

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

/// Sample a homogeneous slice of tables with one morph value. This is the
/// shared body behind the type-erased smoother and the typed wrapper.
fn sample_tables<T: WaveTable>(tables: &[T], smooth_factor: f32, t: f32, channel: usize) -> f32 {
	if tables.is_empty() {
		return 0.0;
	} else if tables.len() == 1 {
		return tables[0].sample(t, channel);
	}

	let (from, to, fraction) = page_position(smooth_factor, tables.len());
	let from_sample = tables[from].sample(t, channel);

	// Sitting exactly on one table should not sample the next one.
	if fraction <= f32::EPSILON || from == to {
		return from_sample;
	}

	let to_sample = tables[to].sample(t, channel);
	from_sample + (to_sample - from_sample) * fraction
}

/// A smoother over one concrete table type.
///
/// The type-erased [`WaveTableSmoother`] exists for heterogeneous frames; when
/// every frame is the same type this wrapper keeps that type in the signature
/// and stores the tables without a trait object.
pub struct TypedWaveTableSmoother<T: WaveTable> {
	/// The tables to smooth between.
	pub tables: Vec<T>,
	/// 0 is the first table, 1 the last.
	pub smooth_factor: f32,
}

impl<T: WaveTable> TypedWaveTableSmoother<T> {
	/// A smoother over a concrete table slice.
	pub fn new(tables: Vec<T>, smooth_factor: f32) -> Self {
		Self {
			tables,
			smooth_factor,
		}
	}

	/// The value at `t` in [0, 1].
	pub fn sample(&self, t: f32, channel: usize) -> f32 {
		sample_tables(&self.tables, self.smooth_factor, t, channel)
	}
}

impl<T: WaveTable> Parameters for TypedWaveTableSmoother<T> {
	fn get_parameters(&self) -> Vec<Parameter> {
		let mut output = vec![Parameter::new(
			"smooth_factor",
			crate::prelude::Value::Float {
				value: self.smooth_factor,
				range: 0.0..=1.0,
				logarithmic: false,
			},
		)];

		for (index, table) in self.tables.iter().enumerate() {
			for mut parameter in table.get_parameters() {
				parameter.identifier = format!("{}.{}", index, parameter.identifier);
				output.push(parameter);
			}
		}

		output
	}

	fn set_parameter(&mut self, identifier: &str, value: crate::prelude::SetValue) -> bool {
		if identifier == "smooth_factor" {
			if let crate::prelude::SetValue::Float(value) = value {
				self.smooth_factor = value;
				return true;
			}
			return false;
		}

		let Some((index, parameter)) = identifier.split_once('.') else {
			return false;
		};
		let Ok(index) = index.parse::<usize>() else {
			return false;
		};

		self.tables
			.get_mut(index)
			.is_some_and(|table| table.set_parameter(parameter, value))
	}
}

impl<T: WaveTable> WaveTable for TypedWaveTableSmoother<T> {
	fn sample(&self, t: f32, channel: usize) -> f32 {
		sample_tables(&self.tables, self.smooth_factor, t, channel)
	}
}

impl WaveTable for WaveTableSmoother {
	fn sample(&self, t: f32, channel: usize) -> f32 {
		sample_tables(&self.tables, self.smooth_factor, t, channel)
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
		sample_tables(&self.tables, self.smooth_factor, t, channel)
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

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn a_typed_smoother_keeps_the_table_type() {
		let smoother = TypedWaveTableSmoother::new(vec![vec![0.0, 0.0], vec![1.0, 1.0]], 0.5);

		assert!((smoother.sample(0.0, 0) - 0.5).abs() < 1e-6);
		assert!((smoother.sample(1.0, 0) - 0.5).abs() < 1e-6);
	}
}
