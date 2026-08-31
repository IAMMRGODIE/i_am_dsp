//! A pitch shifter effect using WSOLA algorithm.

use i_am_dsp_derive::Parameters;

use crate::{prelude::{negative_mean_square_error, wsola, WaveTable}, tools::ring_buffer::RingBuffer, Effect};

/// A pitch shifter effect using WSOLA algorithm.
#[derive(Parameters)]
pub struct PitchShifter<const CHANNELS: usize = 2> {
	/// The pitch shift factor, saves in ratio.
	#[range(min = 0.25, max = 4.0)]
	#[logarithmic]
	pub pitch_shift_factor: f32,
	#[skip]
	buffer: [RingBuffer<f32>; CHANNELS],
	#[skip]
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
		if current_pos.is_multiple_of(half_len) {
			for (buffer, stretched_buffer) in self.buffer.iter().zip(self.stretched_buffer.iter_mut()) {
				let hop = (buffer.capacity() / 4).max(1);
				let ref_range = (buffer.capacity() / 8).clamp(1, hop);
				let prev: &[f32] = if stretched_buffer.len() >= 2 * hop {
					let end = (stretched_buffer.len() / 2 + hop).min(stretched_buffer.len());
					&stretched_buffer[..end]
				}else {
					&[]
				};
				*stretched_buffer = wsola(
					buffer, 
					prev,
					self.pitch_shift_factor, 
					10, 
					hop, 
					ref_range, 
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
#[cfg(test)]
mod tests {
	use super::*;
	use crate::Effect;

	/// Feeds a sine wave through the pitch shifter and returns the output samples.
	fn run_shifter(alpha: f32, total: usize) -> Vec<f32> {
		let fs = 48_000.0f32;
		let freq = 440.0f32;
		let mut shifter = PitchShifter::<1>::new(1024);
		shifter.pitch_shift_factor = alpha;

		let mut output = Vec::with_capacity(total);
		let mut context: Box<dyn crate::ProcessContext> = Box::new(());
		for i in 0..total {
			let sample = (2.0 * std::f32::consts::PI * freq * i as f32 / fs).sin();
			let mut block = [sample];
			shifter.process(&mut block, &[], &mut context);
			output.push(block[0]);
		}
		output
	}

	/// The effect must not produce clicks at the recompute boundary (every
	/// `capacity / 2` = 512 samples): the maximum sample-to-sample step should
	/// stay in the same order as the signal's own slope. A click shows up as a
	/// step close to the full signal amplitude.
	#[test]
	fn no_click_at_block_boundary() {
		// At stretch factor 1.0 the sine's largest genuine step is
		// 2*pi*f/fs ~= 0.058; a boundary click would be ~1.0 or larger.
		let output = run_shifter(1.0, 8 * 1024);

		let mut max_step = 0.0f32;
		for i in 2048..output.len() {
			max_step = max_step.max((output[i] - output[i - 1]).abs());
		}
		assert!(max_step < 0.3, "click found at block boundary, max step = {max_step}");
	}

	/// Same check for other stretch factors: the discontinuity was independent
	/// of alpha, so it must be gone for alpha != 1 as well.
	#[test]
	fn no_click_at_block_boundary_other_alphas() {
		for alpha in [0.5f32, 1.5, 2.0] {
			let output = run_shifter(alpha, 6 * 1024);
			let mut max_step = 0.0f32;
			for i in 2048..output.len() {
				max_step = max_step.max((output[i] - output[i - 1]).abs());
			}
			assert!(max_step < 0.5, "alpha={alpha}: click found, max step = {max_step}");
		}
	}
}
