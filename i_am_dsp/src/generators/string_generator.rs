//! Collections of string generators

use i_am_dsp_derive::Parameters;

use crate::{prelude::{OscTable, Oscillator, WaveTable}, tools::ring_buffer::RingBuffer};

/// A simple pluck generator based on Karplus-Strong algorithm
#[derive(Parameters)]
pub struct SimplePluck<
	InitialFill: Oscillator<CHANNELS> + Send, 
	Coeff: Oscillator<CHANNELS> + Send, 
	const CHANNELS: usize = 2
> {
	#[skip]
	buffer: [RingBuffer<f32>; CHANNELS],
	#[skip]
	sample_rate: usize,
	#[skip]
	cache: [Vec<f32>; CHANNELS],
	#[sub_param]
	/// The initial fill of the string
	pub initial_fill: OscTable<CHANNELS, InitialFill>,
	#[sub_param]
	/// The filter coefficient of the string
	pub coef: OscTable<CHANNELS, Coeff>,
	#[range(min = 1.0, max = 5000.0)]
	/// The length of the coefficient in milliseconds
	#[logarithmic]
	pub coef_len: f32,
	/// The length of the sample in milliseconds
	#[logarithmic]
	#[range(min = 1.0, max = 5000.0)]
	pub sample_len: f32,
	/// The minimum value of the coefficient
	#[range(min = -1.0, max = 1.0)]
	pub coef_min: f32,
	/// The maximum value of the coefficient
	#[range(min = -1.0, max = 1.0)]
	pub coef_max: f32,
}

impl<
	InitialFill: Oscillator<CHANNELS> + Send,
	Coeff: Oscillator<CHANNELS> + Send, 
	const CHANNELS: usize
> SimplePluck<InitialFill, Coeff, CHANNELS> {
	/// Create a new SimpleString
	/// 
	/// # Panics
	/// 
	/// Panics if `CHANNELS == 0` or `coef_len <= 0.0`
	pub fn new(
		sample_rate: usize, 
		inital_fill: InitialFill, 
		coef: Coeff, 
		coef_len: f32,
		sample_len: f32
	) -> Self {
		assert!(CHANNELS > 0 && coef_len > 0.0, "CHANNELS and coef_len must be greater than 0");
		let buffer = std::array::from_fn(|_| RingBuffer::new(96));
		let cache = std::array::from_fn(|_| Vec::new());
		let mut out = Self {
			buffer,
			sample_rate,
			cache,
			initial_fill: OscTable(inital_fill),
			coef: OscTable(coef),
			sample_len,
			coef_len,
			coef_min: -0.5,
			coef_max: 0.5,
		};
		out.refresh_cache(false);
		out
	}

	/// Refresh the cache to apply changes to the coefficients
	pub fn refresh_cache(&mut self, stop_when_coeff_ends: bool) {
		for (buf, cache) in self.buffer.iter_mut().zip(self.cache.iter_mut()) {
			let buf_capacity = buf.capacity();
			buf.fill_with(|i| {
				let sample_t = i as f32 / buf_capacity as f32;
				self.initial_fill.sample(sample_t, 0)
			});

			for i in 0..buf.capacity() {
				buf[i as isize] -= buf[i as isize - 7];
			}

			cache.clear();
			let mut i = 0;

			let mut abs_max = 0.0;
			loop {
				let sample_t = i as f32 / self.coef_len;
				let coef = (self.coef.sample(sample_t, 0) + 1.0) / 2.0;
				let coef = self.coef_min + (self.coef_max - self.coef_min) * coef;
				let current_sample = (buf[1] + buf[0]) * coef;
				cache.push(current_sample);
				buf.push(current_sample);
				i += 1;
				abs_max = current_sample.abs().max(abs_max);

				if i as f32 >= self.sample_len * self.sample_rate as f32 / 1000.0 {
					break;
				}

				if stop_when_coeff_ends && i as f32 >= self.coef_len * self.sample_rate as f32 / 1000.0 {
					break;
				}
			}

			for val in cache.iter_mut() {
				*val /= abs_max;
			}
		}
	}
}

impl<
	InitialFill: Oscillator<CHANNELS> + Send,
	Coeff: Oscillator<CHANNELS> + Send, 
	const CHANNELS: usize
> Oscillator<CHANNELS> for SimplePluck<InitialFill, Coeff, CHANNELS> {
	fn play_at(&self, frequency: f32, time: f32, phase: [f32; CHANNELS]) -> [f32; CHANNELS] {
		let mut output = [0.0; CHANNELS];

		let actual_freq = self.sample_rate as f32 / self.buffer[0].capacity() as f32;
		let factor = frequency / actual_freq;
		let position = time * self.sample_rate as f32 * factor;

		for (i, (out, cache)) in output.iter_mut().zip(self.cache.iter()).enumerate() {
			let t = position + phase[i] * self.buffer[0].capacity() as f32;
			let cache_len = cache.len() as f32;
			let t = if t >= cache_len {
				let buffer_len = (self.buffer[0].capacity() as f32 / factor).min(cache_len);
				let warpped_t = (t - cache_len) % (2.0 * buffer_len);

				cache_len - if warpped_t >= buffer_len {
					2.0 * buffer_len - warpped_t
				}else {
					warpped_t
				}
			}else {
				t
			};
			*out = cache.sample(t / cache.len() as f32, 0);
		}

		output
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		use crate::tools::ui_tools::draw_waveform;
		
		let width = ui.available_width();
		egui::Resize::default().resizable([false, true])
		// .auto_sized()
		.min_width(width)
		.max_width(width)
		.id_salt(format!("{id_prefix}_wavtable"))
		.show(ui, |ui| {
			draw_waveform(ui, None, &self.cache.iter().map(|x| &x[..]).collect::<Vec<_>>(), &None, false, false);
		});
		ui.collapsing("Initial Fill", |ui| {
			self.initial_fill.0.demo_ui(ui, format!("{id_prefix}_inital_fill"));
		});
		ui.collapsing("Coefficient", |ui| {
			self.coef.0.demo_ui(ui, format!("{id_prefix}_coef"));
		});
		ui.add(egui::Slider::new(&mut self.coef_len, 1.0..=5000.0).text("Coefficient Length (ms)"));
		ui.add(egui::Slider::new(&mut self.sample_len, 1.0..=5000.0).text("Sample Length (ms)"));
		ui.add(egui::Slider::new(&mut self.coef_min, -1.0..=1.0).text("Coefficient Min"));
		ui.add(egui::Slider::new(&mut self.coef_max, -1.0..=1.0).text("Coefficient Max"));
		if ui.button("Refresh Cache").clicked() {
			self.refresh_cache(false);
		}
	}
}