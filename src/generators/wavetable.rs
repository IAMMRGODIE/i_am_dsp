//! Wave table and Oscillator implementations.

use std::f32::consts::PI;

use crate::tools::{interpolate::cubic_interpolate, ring_buffer::RingBuffer};

#[cfg(feature = "sin_table")]
lazy_static::lazy_static! {
	static ref SINE_TABLE: Vec<f32> = (0..256).map(|i| (i as f32 / 256.0 * 2.0 * PI).sin()).collect();
}

/// A trait for wave tables that can be used by oscillators or be used as LFO.
/// 
/// Note that `Vec<f32>` and [`RingBuffer<f32>`] is also implemented for this trait.
pub trait WaveTable {
	/// Returns the sample value at time t.
	/// 
	/// t in the range [0, 1]
	fn sample(&self, t: f32, channel: usize) -> f32;
}

/// An oscillator that can be used to generate audio signals.
/// 
/// For midi support, use [`crate::prelude::Adsr`]
pub trait Oscillator<const CHANNELS: usize> {
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
		.id_salt(format!("{id_prefix}_wavtable"))
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
		self.as_slice().sample(t, 0)
	}
}

impl WaveTable for &Vec<f32> {
	fn sample(&self, t: f32, _: usize) -> f32 {
		self.as_slice().sample(t, 0)
	}
}

impl WaveTable for &[f32] {
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
pub struct NoiseWave;

impl WaveTable for NoiseWave {
	fn sample(&self, _: f32, _: usize) -> f32 {
		rand::random_range(-1.0..=1.0)
	}
}

/// A wave table that smoothly transitions between multiple wave tables.
pub struct WaveTableSmoother {
	/// The wave tables to smooth between.
	pub tables: Vec<Box<dyn WaveTable + Send + Sync>>,
	/// should be between 0 and 1, where 0 is first table and 1 is last table.
	pub smooth_factor: f32,
}

impl WaveTableSmoother {
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
	pub fn split_from_vec(mut vec: Vec<f32>, frame_len: usize) -> Self {
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
		Self {
			tables,
			smooth_factor: 0.0,
		}
	}
}

impl<const CHANNELS: usize> Oscillator<CHANNELS> for WaveTableSmoother {
	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		use crate::tools::ui_tools::draw_wavetable;
		egui::Resize::default().resizable([false, true])
			.min_width(ui.available_width())
			.max_width(ui.available_width())
			.id_salt(format!("{id_prefix}_wavtable_smoother"))
			.show(ui, |ui| 
		{
			draw_wavetable(ui, |t| self.sample(t, 0));
		});
		ui.horizontal(|ui| {
			ui.add(egui::Slider::new(&mut self.smooth_factor, 0.0..=1.0).text("Smooth Factor"));
			// if ui.selectable_label(self.linear_interp, "Linear Interpolation").clicked() {
			// 	self.linear_interp = !self.linear_interp;
			// }
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
pub struct OscillatorSmoother<const CHANNELS: usize> {
	/// The oscillators to smooth between.
	pub oscillators: Vec<Box<dyn Oscillator<CHANNELS> + Send + Sync>>,
	/// should be between 0 and 1, where 0 is first table and 1 is last table.
	pub smooth_factor: f32,
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
			.id_salt(format!("{id_prefix}_wavtable_smoother"))
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
