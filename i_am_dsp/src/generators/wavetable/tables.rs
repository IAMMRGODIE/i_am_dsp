//! The wave tables: the slices, the buffers and the four basic waves.

#[cfg(not(feature = "sin_table"))]
use std::f32::consts::PI;

use i_am_dsp_derive::Parameters;
use rustfft::num_complex::Complex;

use crate::tools::{
	interpolate::{cubic_interpolate, cubic_interpolate_complex},
	ring_buffer::RingBuffer,
};

#[cfg(feature = "sin_table")]
use super::SINE_TABLE;
use super::WaveTable;

impl WaveTable for Vec<f32> {
	fn sample(&self, t: f32, _: usize) -> f32 {
		if self.is_empty() {
			return 0.0;
		} else if self.len() == 1 {
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
		} else {
			cubic_interpolate(t.fract(), [before, self[start], self[end], after])
		}
	}
}

impl WaveTable for Vec<Complex<f32>> {
	fn sample(&self, t: f32, _: usize) -> f32 {
		if self.is_empty() {
			return 0.0;
		} else if self.len() == 1 {
			return self[0].re;
		}
		let t = t * (self.len() as f32 - 1.0);
		if t > self.len() as f32 - 1.0 {
			return 0.0;
		}

		let (start, end) = (t.floor() as usize, t.ceil() as usize);
		let before = if start == 0 {
			Complex::ZERO
		} else {
			self[start - 1]
		};
		let after = if end >= self.len() {
			Complex::ZERO
		} else {
			self[end]
		};

		if t == start as f32 {
			self[start].re
		} else {
			cubic_interpolate_complex(t.fract(), [before, self[start], self[end], after]).re
		}
	}
}

impl WaveTable for &[f32] {
	fn sample(&self, t: f32, _: usize) -> f32 {
		if self.is_empty() {
			return 0.0;
		} else if self.len() == 1 {
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
		} else {
			cubic_interpolate(t.fract(), [before, self[start], self[end], after])
		}
	}
}

impl WaveTable for &[Complex<f32>] {
	fn sample(&self, t: f32, _: usize) -> f32 {
		if self.is_empty() {
			return 0.0;
		} else if self.len() == 1 {
			return self[0].re;
		}
		let t = t * (self.len() as f32 - 1.0);
		if t > self.len() as f32 - 1.0 {
			return 0.0;
		}

		let (start, end) = (t.floor() as usize, t.ceil() as usize);
		let before = if start == 0 {
			Complex::ZERO
		} else {
			self[start - 1]
		};
		let after = if end >= self.len() {
			Complex::ZERO
		} else {
			self[end]
		};

		if t == start as f32 {
			self[start]
		} else {
			cubic_interpolate_complex(t.fract(), [before, self[start], self[end], after])
		}
		.re
	}
}

impl RingBuffer<f32> {
	/// Sample at a fractional ring index, where 0 is the oldest sample and
	/// `capacity - 1` the newest; the index wraps, so a delay read can cross the
	/// end of the buffer without a branch at the seam.
	pub fn sample_at(&self, position: f64) -> f32 {
		let capacity = self.capacity();
		if capacity == 0 {
			return 0.0;
		} else if capacity == 1 {
			return self[0];
		}

		let position = position.rem_euclid(capacity as f64);
		let index = position.floor() as isize;
		let fraction = (position - index as f64) as f32;

		if fraction == 0.0 {
			self[index]
		} else {
			cubic_interpolate(
				fraction,
				[
					self[index - 1],
					self[index],
					self[index + 1],
					self[index + 2],
				],
			)
		}
	}
}

impl WaveTable for RingBuffer<f32> {
	fn sample(&self, t: f32, _: usize) -> f32 {
		if self.capacity() == 0 {
			return 0.0;
		} else if self.capacity() == 1 {
			return self[0];
		}

		let t = t * (self.capacity() as f32 - 1.0);

		let (start, end) = (t.floor() as isize, t.ceil() as isize);

		if t == start as f32 {
			self[start]
		} else {
			cubic_interpolate(
				t.fract(),
				[self[start - 1], self[start], self[end], self[end + 1]],
			)
		}
	}
}

impl WaveTable for RingBuffer<Complex<f32>> {
	fn sample(&self, t: f32, _: usize) -> f32 {
		if self.capacity() == 0 {
			return 0.0;
		} else if self.capacity() == 1 {
			return self[0_usize].re;
		}

		let t = t * (self.capacity() as f32 - 1.0);

		let (start, end) = (t.floor() as isize, t.ceil() as isize);

		if t == start as f32 {
			self[start].re
		} else {
			cubic_interpolate_complex(
				t.fract(),
				[self[start - 1], self[start], self[end], self[end + 1]],
			)
			.re
		}
	}
}

impl WaveTable for f32 {
	fn sample(&self, _: f32, _: usize) -> f32 {
		*self
	}
}

impl WaveTable for Complex<f32> {
	fn sample(&self, _: f32, _: usize) -> f32 {
		self.re
	}
}

impl WaveTable for () {
	fn sample(&self, _: f32, _: usize) -> f32 {
		0.0
	}
}

/// A sine wave wave table.
#[derive(Parameters)]
pub struct SineWave;

impl WaveTable for SineWave {
	fn sample(&self, t: f32, _channel: usize) -> f32 {
		let t = t % 1.0;

		#[cfg(feature = "sin_table")]
		{
			SINE_TABLE.sample(t, _channel)
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

		if t < 0.5 { 1.0 } else { -1.0 }
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
		} else if t < 0.75 {
			4.0 * (0.5 - t)
		} else {
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

		if t < 0.5 { 2.0 * t } else { 2.0 * t - 2.0 }
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
