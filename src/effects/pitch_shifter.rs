//! A pitch shifter effect using WSOLA algorithm.

use crate::{prelude::{negative_mean_square_error, wsola, WaveTable}, tools::ring_buffer::RingBuffer, Effect};

/// A pitch shifter effect using WSOLA algorithm.
pub struct PitchShifter<const CHANNELS: usize = 2> {
	/// The pitch shift factor, saves in ratio.
	pub pitch_shift_factor: f32,
	buffer: [RingBuffer<f32>; CHANNELS],
	stretched_buffer: [Vec<f32>; CHANNELS],
}

impl<const CHANNELS: usize> Default for PitchShifter<CHANNELS> {
	fn default() -> Self {
		Self::new(1024)
	}
}

impl<const CHANNELS: usize> PitchShifter<CHANNELS> {
	/// Creates a new pitch shifter with the given buffer size.
	/// 
	/// # Panics 
	/// 
	/// 1. `buffer_size` is less than or equal to 0.
	/// 2. `CHANNELS` is less than or equal to 0.
	pub fn new(buffer_size: usize) -> Self {
		assert!(CHANNELS > 0, "CHANNELS must be greater than 0");
		assert!(buffer_size > 0, "buffer_size must be greater than 0");

		Self {
			pitch_shift_factor: 1.0,
			buffer: core::array::from_fn(|_| RingBuffer::new(buffer_size)),
			stretched_buffer: core::array::from_fn(|_| vec![]),
		}
	}

	/// Resizes the buffer to the given size.
	pub fn resize(&mut self, buffer_size: usize) {
		assert!(buffer_size > 0);

		self.buffer = core::array::from_fn(|_| RingBuffer::new(buffer_size));
		self.stretched_buffer = core::array::from_fn(|_| vec![]);
	}
}

impl<const CHANNELS: usize> Effect<CHANNELS> for PitchShifter<CHANNELS> {
	fn delay(&self) -> usize {
		self.buffer[0].capacity()
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"WSOLA Pitch Shifter"
	}

	fn process(
		&mut self, 
		samples: &mut [f32; CHANNELS], 
		_: &[&[f32; CHANNELS]],
		_: &mut Box<dyn crate::ProcessContext>,
	) {
		let current_pos = self.buffer[0].current_pos();
		let half_len = self.buffer[0].capacity() / 2;
		if current_pos % half_len == 0 {
			for (buffer, stretched_buffer) in self.buffer.iter().zip(self.stretched_buffer.iter_mut()) {
				*stretched_buffer = wsola(
					buffer, 
					stretched_buffer,
					self.pitch_shift_factor, 
					10, 
					buffer.capacity() / 2, 
					300, 
					negative_mean_square_error,
				);


				// println!("stretched_buffer len: {}", stretched_buffer.len());
			}
		}
		// let hop = self.buffer[0].capacity() / 4;

		for (i, sample) in samples.iter_mut().enumerate() {
			if self.stretched_buffer[i].is_empty() {
				continue;
			}

			let current_pos = self.buffer[i].current_pos() % half_len;
			
			let current_t = (current_pos as f32 / self.buffer[i].capacity() as f32) % 0.5;
			
			self.buffer[i].push(*sample);
			*sample = self.stretched_buffer[i].sample(current_t, 0);
		}
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, _: String) {
		ui.add(egui::Slider::new(&mut self.pitch_shift_factor, 0.25..=4.0)
			.text("Pitch Shift Factor")
			.logarithmic(true)
		);
	}
}