//! Wave table and Oscillator implementations.

use std::f32::consts::PI;

use i_am_parameters_derive::Parameters;

use crate::{prelude::{bend, Parameter, Parameters}, tools::{interpolate::cubic_interpolate, ring_buffer::RingBuffer}};

#[cfg(feature = "sin_table")]
lazy_static::lazy_static! {
	static ref SINE_TABLE: Vec<f32> = (0..256).map(|i| (i as f32 / 256.0 * 2.0 * PI).sin()).collect();
}

/// A trait for wave tables that can be used by oscillators or be used as LFO.
/// 
/// Note that `Vec<f32>` and [`RingBuffer<f32>`] is also implemented for this trait.
pub trait WaveTable: Parameters {
	/// Returns the sample value at time t.
	/// 
	/// t in the range [0, 1]
	fn sample(&self, t: f32, channel: usize) -> f32;
}

/// An oscillator that can be used to generate audio signals.
/// 
/// For midi support, use [`crate::prelude::Adsr`]
pub trait Oscillator<const CHANNELS: usize>: Parameters {
	/// Plays the oscillator at the given frequency and time, and returns the output for each channel.
	/// 
	/// frequency in Hz
	/// time in milliseconds
	/// phase in [0.0, 1.0]
	fn play_at(&mut self, frequency: f32, time: f32, phase: [f32; CHANNELS]) -> [f32; CHANNELS];
	
	#[cfg(feature = "real_time_demo")]
	/// Draws the wave table for the oscillator in the demo UI.
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String);
}

impl<T: WaveTable, const CHANNELS: usize> Oscillator<CHANNELS> for T {
	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		use crate::tools::ui_tools::draw_wavetable;

		let width = ui.available_width();
		egui::Resize::default().resizable([false, true])
		// .auto_sized()
		.min_width(width)
		.max_width(width)
		.id_source(format!("{id_prefix}_wavtable"))
		.show(ui, |ui| {
			draw_wavetable(ui, |t| self.sample(t, 0));
		});
	}

	fn play_at(&mut self, frequency: f32, time: f32, phase: [f32; CHANNELS]) -> [f32; CHANNELS] {
		let mut output = [0.0; CHANNELS];

		for i in 0..CHANNELS {
			let t = (time * frequency + phase[i]) % 1.0;
			output[i] = self.sample(t, i);
		}

		output
	}
}

impl WaveTable for Vec<f32> {
	fn sample(&self, t: f32, _: usize) -> f32 {
		if self.is_empty() {
			return 0.0;
		}else if self.len() == 1 {
			return self[0];
		}
		let t = t * (self.len() as f32 - 1.0);
		if t > self.len() as f32 - 1.0 {
			return 0.0;
		}

		let (start, end) = (t.floor() as usize, t.ceil() as usize);
		let before = if start == 0 { 0.0 } else { self[start - 1] };
		let after = if end >= self.len() { 0.0 } else { self[end] };

		if t == start as f32 {
			self[start]
		}else {
			cubic_interpolate(
				t.fract(),
				[before, self[start], self[end], after],
			)
		}
	}
}

impl WaveTable for &mut [f32] {
	fn sample(&self, t: f32, _: usize) -> f32 {
		if self.is_empty() {
			return 0.0;
		}else if self.len() == 1 {
			return self[0];
		}
		let t = t * (self.len() as f32 - 1.0);
		if t > self.len() as f32 - 1.0 {
			return 0.0;
		}

		let (start, end) = (t.floor() as usize, t.ceil() as usize);
		let before = if start == 0 { 0.0 } else { self[start - 1] };
		let after = if end >= self.len() { 0.0 } else { self[end] };

		if t == start as f32 {
			self[start]
		}else {
			cubic_interpolate(
				t.fract(),
				[before, self[start], self[end], after],
			)
		}
	}
}

impl WaveTable for RingBuffer<f32> {
	fn sample(&self, t: f32, _: usize) -> f32 {
		if self.capacity() == 0 {
			return 0.0;
		}else if self.capacity() == 1 {
			return self[0];
		}

		let t = t * (self.capacity() as f32 - 1.0);

		let (start, end) = (t.floor() as isize, t.ceil() as isize);

		if t == start as f32 {
			self[start]
		}else {
			cubic_interpolate(
				t.fract(),
				[self[start - 1], self[start], self[end], self[end + 1]],
			)
		}
	}
}

/// A sine wave wave table.
#[derive(Parameters)]
pub struct SineWave;

impl WaveTable for SineWave {
	fn sample(&self, t: f32, channel: usize) -> f32 {
		let t = t % 1.0;

		#[cfg(feature = "sin_table")]
		{
			SINE_TABLE.sample(t, channel)
		}
		#[cfg(not(feature = "sin_table"))]
		{
			(t * PI * 2.0).sin()
		}
	}
}

/// A square wave wave table.
#[derive(Parameters)]
pub struct SquareWave;

impl WaveTable for SquareWave {
	fn sample(&self, t: f32, _: usize) -> f32 {
		let t = t % 1.0;

		if t < 0.5 {
			1.0
		}else {
			-1.0
		}
	}
}

/// A triangle wave wave table.
#[derive(Parameters)]
pub struct TriangleWave;

impl WaveTable for TriangleWave {
	fn sample(&self, t: f32, _: usize) -> f32 {
		let t = t % 1.0;

		if t < 0.25 {
			4.0 * t
		}else if t < 0.75 {
			4.0 * (0.5 - t)
		}else {
			4.0 * (t - 1.0)
		}
	}
}

/// A saw wave wave table.
#[derive(Parameters)]
pub struct SawWave;

impl WaveTable for SawWave {
	fn sample(&self, t: f32, _: usize) -> f32 {
		let t = t % 1.0;

		if t < 0.5 {
			2.0 * t
		}else {
			2.0 * t - 2.0
		}
	}
}

/// A noise wave wave table.
#[derive(Parameters)]
pub struct NoiseWave;

impl WaveTable for NoiseWave {
	fn sample(&self, _: f32, _: usize) -> f32 {
		rand::random_range(-1.0..=1.0)
	}
}

/// A wave table that smoothly transitions between multiple wave tables.
#[derive(Parameters)]
pub struct WaveTableSmoother {
	#[sub_param]
	/// The wave tables to smooth between.
	pub tables: Vec<Box<dyn WaveTable + Send + Sync>>,
	/// should be between 0 and 1, where 0 is first table and 1 is last table.
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
		let index = parts.remove(0).parse::<usize>().expect("Invalid identifier");
		let param_name = parts.join(".");
		if let Some(table) = self.get_mut(index) {
			table.set_parameter(&param_name, value)
		}else {
			false
		}
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
		}
	}

	/// Sample the wave table at time t, t in the range [0, 1]
	/// 
	/// Due to rust's limitation, we can't implement [`WaveTable`] for `WaveTableSmoother`.
	pub fn sample(&self, t: f32, channel: usize) -> f32 {
		if self.tables.is_empty() {
			return 0.0;
		}else if self.tables.len() == 1 {
			return self.tables[0].sample(t, channel);
		}
		let smooth = self.smooth_factor * (self.tables.len() as f32 - 1.0);
		let (from, to) = (smooth.floor() as usize, smooth.ceil() as usize);
		let from_sample = self.tables[from].sample(t, channel);
		let to_sample = self.tables[to].sample(t, channel);
		let smooth = smooth % 1.0;

		from_sample + (to_sample - from_sample) * smooth
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
		}
	}

	fn split_vec(mut vec: Vec<f32>, frame_len: usize) -> Vec<Box<dyn WaveTable + Send + Sync>> {
		let len = vec.len();
		let frame_count = len / frame_len;
		let frame_count = if frame_count * frame_len < len {
			frame_count + 1
		}else {
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
		use crate::tools::ui_tools::draw_wavetable;
		use crate::tools::load_pcm_data::load_from_file;
		use egui::Slider;

		let mut clear_error = false;
		if let Some(error) = self.error.as_ref() {
			ui.colored_label(egui::Color32::RED, error);
			clear_error = ui.button("Clear Error").clicked();
		}

		if clear_error {
			self.error = None;
		}

		egui::Resize::default().resizable([false, true])
			.min_width(ui.available_width())
			.max_width(ui.available_width())
			.id_source(format!("{id_prefix}_wavtable_smoother"))
			.show(ui, |ui| 
		{
			draw_wavetable(ui, |t| self.sample(t, 0));
		});
		
		ui.horizontal(|ui| {
			let mut path = None;

			if self.allow_change_table {
				ui.input(|input| {
					path = input.raw.dropped_files.first().map(|inner| {
						inner.path.clone()
					}).unwrap_or_default();
				});
			}
			
			#[cfg(feature = "rfd")]
			if ui.button("Load Sample").clicked() {
				let dialog = rfd::FileDialog::new()
					.add_filter("Wav Files", &["wav"]);
				path = dialog.pick_file();
			}

			if let Some(path) = path {
				if path.extension().map(|ext| ext.to_string_lossy().to_lowercase() != "wav").unwrap_or(true) {
					return;
				}

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
				let wave_form_data = (0..len).map(|i| {
					let mut sum = 0.0;
					for channel in data.pcm_data.iter() {
						sum += channel[i];
					}
					sum / data.pcm_data.len() as f32
				}).collect::<Vec<_>>();
				self.tables = Self::split_vec(wave_form_data, self.split_amount);
				self.allow_change_table = false;
			}
			ui.add(Slider::new(&mut self.split_amount, 16..=4096).text("Split Size"));
			if ui.selectable_label(self.allow_change_table, "Allow Change Table").clicked() {
				self.allow_change_table = !self.allow_change_table;
			}
		});

		ui.horizontal(|ui| {
			ui.add(egui::Slider::new(&mut self.smooth_factor, 0.0..=1.0).text("Smooth Factor"));
		});
	}

	fn play_at(&mut self, frequency: f32, time: f32, phase: [f32; CHANNELS]) -> [f32; CHANNELS] {
		let mut output = [0.0; CHANNELS];

		for i in 0..CHANNELS {
			let t = (time * frequency + phase[i]) % 1.0;
			output[i] = self.sample(t, i);
		}

		output
	}
}


/// A wave table that smoothly transitions between multiple wave tables.
#[derive(Parameters)]
pub struct OscillatorSmoother<const CHANNELS: usize> {
	/// The oscillators to smooth between.
	#[sub_param]
	pub oscillators: Vec<Box<dyn Oscillator<CHANNELS> + Send + Sync>>,
	/// should be between 0 and 1, where 0 is first table and 1 is last table.
	pub smooth_factor: f32,
}

impl<const CHANNELS: usize> Parameters for Vec<Box<dyn Oscillator<CHANNELS> + Send + Sync>> {
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
		let index = parts.remove(0).parse::<usize>().expect("Invalid identifier");
		let param_name = parts.join(".");
		if let Some(table) = self.get_mut(index) {
			table.set_parameter(&param_name, value)
		}else {
			false
		}
	}
}

impl<const CHANNELS: usize> OscillatorSmoother<CHANNELS> {
	/// Sample the wave table at time t, t in the range [0, 1]
	pub fn sample(&mut self, t: f32, phase: [f32; CHANNELS]) -> [f32; CHANNELS] {
		if self.oscillators.is_empty() {
			return [0.0; CHANNELS];
		}else if self.oscillators.len() == 1 {
			return self.oscillators[0].play_at(1.0, t, phase);
		}
		let smooth = self.smooth_factor * (self.oscillators.len() as f32 - 1.0);
		let (from, to) = (smooth.floor() as usize, smooth.ceil() as usize);
		let from_sample = self.oscillators[from].play_at(1.0, t, phase);
		let to_sample = self.oscillators[to].play_at(1.0, t, phase);
		let smooth = smooth % 1.0;
		let mut output = [0.0; CHANNELS];

		for i in 0..CHANNELS {
			output[i] = from_sample[i] + (to_sample[i] - from_sample[i]) * smooth;
		}

		output
	}
}

impl<const CHANNELS: usize> Oscillator<CHANNELS> for OscillatorSmoother<CHANNELS> {
	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		use crate::tools::ui_tools::draw_wavetable;
		egui::Resize::default().resizable([false, true])
			.min_width(ui.available_width())
			.max_width(ui.available_width())
			.id_source(format!("{id_prefix}_wavtable_smoother"))
			.show(ui, |ui| 
		{
			draw_wavetable(ui, |t| self.sample(t, [0.0; CHANNELS]).iter().map(|inner| {
				*inner * *inner
			}).sum::<f32>().sqrt());
		});
		
		ui.horizontal(|ui| {
			ui.add(egui::Slider::new(&mut self.smooth_factor, 0.0..=1.0).text("Smooth Factor"));
			// if ui.selectable_label(self.linear_interp, "Linear Interpolation").clicked() {
			// 	self.linear_interp = !self.linear_interp;
			// }
		});
	}

	fn play_at(&mut self, frequency: f32, time: f32, phase: [f32; CHANNELS]) -> [f32; CHANNELS] {
		let t = (time * frequency) % 1.0;
		self.sample(t, phase)
	}
}

/// A editable LFO.
#[derive(Parameters)]
pub struct EditableLfo {
	/// The inner data of current lfo.
	/// 
	/// the format is (time, value, bend)
	#[persist(serialize = "format_lfo_data", deserialize = "parse_lfo_data")]
	pub data: Vec<(f32, f32, f32)>,
	#[range(min = 0.01, max = 100.0)]
	#[logarithmic]
	/// The frequency of the LFO.
	pub lfo_frequency: f32,

	/// If true, the LFO will clamp the time to the range [0, 1] rather than wrapping around.
	pub one_shot: bool,

	#[skip]
	#[cfg(feature = "real_time_demo")]
	dragging: Option<(usize, bool)>,

	#[skip]
	#[cfg(feature = "real_time_demo")]
	grid: Option<(usize, usize)>,
	
}

fn format_lfo_data(data: &[(f32, f32, f32)]) -> Vec<u8> {
	let mut output = vec![];
	for (time, value, bend) in data {
		output.extend_from_slice(&f32::to_le_bytes(*time));
		output.extend_from_slice(&f32::to_le_bytes(*value));
		output.extend_from_slice(&f32::to_le_bytes(*bend));
	}
	output
}

fn parse_lfo_data(data: Vec<u8>) -> Vec<(f32, f32, f32)> {
	let mut output = vec![];
	for i in (0..data.len()).step_by(12) {
		let time = f32::from_le_bytes([data[i], data[i+1], data[i+2], data[i+3]]);
		let value = f32::from_le_bytes([data[i+4], data[i+5], data[i+6], data[i+7]]);
		let bend = f32::from_le_bytes([data[i+8], data[i+9], data[i+10], data[i+11]]);
		output.push((time, value, bend));
	}
	output
}

impl Default for EditableLfo {
	fn default() -> Self {
		Self::new()
	}
}

impl EditableLfo {
	/// Create a new editable LFO.
	pub fn new() -> Self {
		Self {
			data: vec![],
			lfo_frequency: 1.0,
			one_shot: false,

			#[cfg(feature = "real_time_demo")]
			dragging: None,
			#[cfg(feature = "real_time_demo")]
			grid: None,
		}
	}

	/// Get the value of the LFO at the given time.
	/// 
	/// Will clamp the time and value to the range [0, 1].
	pub fn add_point(&mut self, time: f32, value: f32, bend: f32) {
		let time = time.clamp(0.0, 1.0);
		let value = value.clamp(0.0, 1.0);

		self.data.push((time, value, bend));
		self.data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
	}

	/// Remove the point at the given time.
	pub fn remove_point(&mut self, time: f32) {
		self.data.retain(|p| p.0 != time);
	}

	/// Change the value of the point at the given time.
	/// 
	/// Will clamp the value to the range [0, 1].
	pub fn change_point_value(&mut self, time: f32, value: f32) {
		let value = value.clamp(0.0, 1.0);
		if let Some(p) = self.data.iter_mut().find(|p| p.0 == time) {
			p.1 = value;
		}
	}

	/// Change the bend of the point at the given time.
	pub fn change_point_bend(&mut self, time: f32, bend: f32) {
		if let Some(p) = self.data.iter_mut().find(|p| p.0 == time) {
			p.2 = bend;
		}
	}

	/// Change the time of the point at the given time.
	/// 
	/// Will clamp the new time to the range [0, 1].
	pub fn change_point_time(&mut self, old_time: f32, new_time: f32) {
		let new_time = new_time.clamp(0.0, 1.0);
		// assert!((0.0..=1.0).contains(&new_time), "Time must be in the range [0, 1]");

		if old_time == new_time {
			return;
		}

		if let Some(p) = self.data.iter_mut().find(|p| p.0 == old_time) {
			p.0 = new_time;
		}
		self.data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
	}

	#[cfg(feature = "real_time_demo")]
	/// Draws the editable LFO in the demo UI.
	pub fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		use egui::*;
		use crate::tools::ui_tools::draw_wavetable;

		const HEIGHT: f32 = 100.0;
		const RADIUS: f32 = 3.0;

		Resize::default()
			.min_height(HEIGHT)
			.max_height(HEIGHT)
			.id_source(format!("{id_prefix}_editable_lfo"))
			.show(ui, |ui| 
		{
			let response = draw_wavetable(ui, |t| {
				self.sample(t / self.lfo_frequency, 0) * 2.0 - 1.0
			});

			let rect = response.rect;

			let pointer_position = ui.input(|input| {
				input.pointer.latest_pos()
			});

			let to_screen = emath::RectTransform::from_to(
				Rect::from_x_y_ranges(0.0..=1.0, -1.0..=0.0), 
				rect
			);

			let dragging = response.is_pointer_button_down_on();

			if let Some((row, column)) = &self.grid {
				for i in 0..=*row {
					for j in 0..=*column {
						let x = (j as f32) / *column as f32;
						let y = (i as f32) / *row as f32;
						let row_pos_l = to_screen * pos2(0.0, -y);
						let row_pos_r = to_screen * pos2(1.0, -y);
						ui.painter().line_segment([row_pos_l, row_pos_r], (
							1.0, 
							Color32::from_rgba_unmultiplied(50, 50, 50, 10)
						));
						let column_pos_t = to_screen * pos2(x, 0.0);
						let column_pos_b = to_screen * pos2(x, -1.0);
						ui.painter().line_segment([column_pos_t, column_pos_b], (
							1.0, 
							Color32::from_rgba_unmultiplied(50, 50, 50, 10)
						));
					}
				}
			}

			ui.painter_at(rect).extend((0..self.data.len()).flat_map(|i| {
				let (current_time, value, _) = self.data[i];
				let next_time = if i == self.data.len() - 1 {
					1.0
				}else {
					self.data[i + 1].0
				};
				let current_x = current_time;
				let bend_x = (current_x + next_time) / 2.0;
				let current_y = value;
				let bend_y = self.sample(bend_x / self.lfo_frequency, 0);

				let bend_pos = to_screen * pos2(bend_x, -bend_y);
				let current_pos = to_screen * pos2(current_x, -current_y);

				if dragging {
					if let Some(pos) = pointer_position {
						if pos.distance(bend_pos) <= RADIUS * 2.0 && self.dragging.is_none() {
							self.dragging = Some((i, true));
						}else if pos.distance(current_pos) <= RADIUS * 2.0 && self.dragging.is_none() {
							self.dragging = Some((i, false));
						}
					}
				}

				[
					Shape::circle_stroke(current_pos, RADIUS, (1.0, Color32::WHITE)),
					Shape::circle_stroke(bend_pos, RADIUS, (1.0, Color32::WHITE)),
				]
			}));

			let Some(pointer_position) = pointer_position else { return };

			if response.double_clicked() {
				let mut time = (pointer_position.x - rect.left()) / rect.width();
				let mut value = (rect.bottom() - pointer_position.y) / rect.height();
				if let Some((row, column)) = &self.grid {
					if *row != 0 && *column != 0 {
						time = (time * *column as f32).round() / *column as f32;
						value = (value * *row as f32).round() / *row as f32;
					}
				}

				self.add_point(time, value, 0.0);
			}
			
			if response.clicked_by(PointerButton::Secondary) {
				for (current_time, current_value, _) in self.data.iter() {
					let point_time = *current_time;
					let current_time = (current_time * rect.width()) + rect.left();
					let current_value = (current_value * rect.height()) + rect.top();
					if (pointer_position.x - current_time).hypot(pointer_position.y - current_value) <= RADIUS * 2.0 {
						self.remove_point(point_time);
						break;
					}
				}
			}

			if let Some((i, is_bend)) = self.dragging {
				let time = (pointer_position.x - rect.left()) / rect.width();
				let value = (rect.bottom() - pointer_position.y) / rect.height();
				if is_bend {
					self.change_point_bend(self.data[i].0, value * 20.0 - 10.0);
				}else {
					self.change_point_value(self.data[i].0, value);
					self.change_point_time(self.data[i].0, time);
				}
			}

			if !dragging {
				if let Some((row, column)) = self.grid.take() {
					if let Some((i, is_bend)) = self.dragging.take() {
						if !is_bend {
							let (time, mut value, _) = self.data[i];
							value = (value * row as f32).round() / row as f32;
							self.change_point_value(time, value);
							let new_time = (time * column as f32).round() / column as f32;
							self.change_point_time(time, new_time);
						}
						self.dragging = Some((i, is_bend));
					}
					self.grid = Some((row, column));
				}

				self.dragging = None;
			}
		});

		if ui.selectable_label(self.one_shot, "One Shot").clicked() {
			self.one_shot = !self.one_shot;
		}
		ui.add(Slider::new(&mut self.lfo_frequency, 0.01..=100.0).text("LFO Frequency").logarithmic(true));

		if ui.selectable_label(self.grid.is_some(), "Grid").clicked() {
			if self.grid.is_none() {
				self.grid = Some((8, 8));
			}else {
				self.grid = None;
			}
		}

		if let Some((row, column)) = &mut self.grid {
			ui.horizontal(|ui| {
				ui.add(Slider::new(row, 1..=16).text("Grid Rows"));
				ui.add(Slider::new(column, 1..=16).text("Grid Columns"));
			});
		}
	}
}

impl WaveTable for EditableLfo {
	fn sample(&self, t: f32, _: usize) -> f32 {
		let t = t * self.lfo_frequency;
		let t = if self.one_shot {
			t.clamp(0.0, 1.0)
		}else {
			t % 1.0
		};
		
		let mut value = 0.0;
		let mut bend_amount = 0.0;
		let mut last_time = 0.0;
		let mut current_time = None;
		for (time, v, b) in &self.data {
			if *time > t {
				current_time = Some((*time, *v));
				break;
			}
			value = *v;
			bend_amount = *b;
			last_time = *time;
		}

		let (current_time, current_value) = current_time.unwrap_or((1.0, 0.0));
		
		let t = (t - last_time) / (current_time - last_time);
		let bended_t = bend(t, bend_amount);
		(current_value - value) * bended_t + value
	}
}