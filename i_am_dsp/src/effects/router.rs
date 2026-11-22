//! A stereo router: a 2x2 gain matrix in dB.
//!
//! out_left  = L-to-L * left + R-to-L * right
//! out_right = L-to-R * left + R-to-R * right
//!
//! The cross paths start at -60 dB (silence), so the defaults are an identity
//! matrix. Swap or sum the channels by moving the four gains.

use i_am_dsp_derive::Parameters;

use crate::{Effect, ProcessContext};

/// A stereo routing matrix.
#[derive(Parameters)]
pub struct StereoRouter {
	/// Left input to left output.
	#[range(min = -60.0, max = 12.0)]
	#[unit(decibels)]
	#[default(0.0)]
	#[id(display = "L to L")]
	#[layout(group = "Router", order = 0, widget = knob, hero = "stereo")]
	pub l_to_l: f32,

	/// Left input to right output.
	#[range(min = -60.0, max = 12.0)]
	#[unit(decibels)]
	#[default(-60.0)]
	#[id(display = "L to R")]
	#[layout(group = "Router", order = 1, widget = knob)]
	pub l_to_r: f32,

	/// Right input to left output.
	#[range(min = -60.0, max = 12.0)]
	#[unit(decibels)]
	#[default(-60.0)]
	#[id(display = "R to L")]
	#[layout(group = "Router", order = 2, widget = knob)]
	pub r_to_l: f32,

	/// Right input to right output.
	#[range(min = -60.0, max = 12.0)]
	#[unit(decibels)]
	#[default(0.0)]
	#[id(display = "R to R")]
	#[layout(group = "Router", order = 3, widget = knob)]
	pub r_to_r: f32,

	#[skip]
	sample_rate: f32,
	#[skip]
	gains: [f32; 4],
}

impl StereoRouter {
	/// An identity matrix: left stays left, right stays right.
	pub fn new(sample_rate: usize) -> Self {
		Self {
			l_to_l: 0.0,
			l_to_r: -60.0,
			r_to_l: -60.0,
			r_to_r: 0.0,
			sample_rate: sample_rate as f32,
			gains: [1.0, 0.0, 0.0, 1.0],
		}
	}

	fn linear(db: f32) -> f32 {
		if db <= -59.5 {
			0.0
		} else {
			10.0_f32.powf(db / 20.0)
		}
	}
}

impl Effect<2> for StereoRouter {
	fn delay(&self) -> usize {
		0
	}

	/// Four gains and a sum: nothing is stored between samples but the gains' own
	/// convergence, which is inaudible while the output is silent.
	fn is_idle(&self, input_is_silent: bool) -> bool {
		input_is_silent
	}

	fn process(
		&mut self,
		samples: &mut [f32; 2],
		_other: &[&[f32; 2]],
		_: &mut Box<dyn ProcessContext>,
	) {
		let targets = [
			Self::linear(self.l_to_l),
			Self::linear(self.l_to_r),
			Self::linear(self.r_to_l),
			Self::linear(self.r_to_r),
		];
		let smoothing = (-1.0 / (0.01 * self.sample_rate).max(1.0)).exp();

		for (gain, target) in self.gains.iter_mut().zip(targets) {
			*gain = smoothing * *gain + (1.0 - smoothing) * target;
		}

		let [l_to_l, l_to_r, r_to_l, r_to_r] = self.gains;
		let left = samples[0];
		let right = samples[1];
		samples[0] = l_to_l * left + r_to_l * right;
		samples[1] = l_to_r * left + r_to_r * right;

		for sample in samples.iter_mut() {
			if !sample.is_finite() {
				*sample = 0.0;
			}
		}
	}

	fn clear_tail(&mut self) {
		self.gains = [
			Self::linear(self.l_to_l),
			Self::linear(self.l_to_r),
			Self::linear(self.r_to_l),
			Self::linear(self.r_to_r),
		];
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn render(effect: &mut StereoRouter, frames: usize) -> Vec<[f32; 2]> {
		let mut context: Box<dyn ProcessContext> = Box::new(());
		let mut output = Vec::new();

		for _ in 0..frames {
			let mut frame = [0.25, -0.5];
			effect.process(&mut frame, &[], &mut context);
			output.push(frame);
		}

		output
	}

	/// The default matrix is an identity.
	#[test]
	fn the_default_router_is_an_identity() {
		let mut router = StereoRouter::new(48_000);

		for frame in render(&mut router, 1_000) {
			assert!((frame[0] - 0.25).abs() < 1e-6, "{frame:?}");
			assert!((frame[1] + 0.5).abs() < 1e-6, "{frame:?}");
		}
	}

	/// Moving the cross paths swaps the two channels.
	#[test]
	fn the_cross_paths_swap_the_channels() {
		let mut router = StereoRouter::new(48_000);
		router.l_to_l = -60.0;
		router.l_to_r = 0.0;
		router.r_to_l = 0.0;
		router.r_to_r = -60.0;
		let output = render(&mut router, 4_000);

		for frame in &output[3_000..] {
			assert!((frame[0] + 0.5).abs() < 2e-3, "{frame:?}");
			assert!((frame[1] - 0.25).abs() < 2e-3, "{frame:?}");
		}
	}

	/// All four paths at 0 dB sum both inputs into both outputs.
	#[test]
	fn all_paths_open_sums_to_mono() {
		let mut router = StereoRouter::new(48_000);
		router.l_to_l = 0.0;
		router.l_to_r = 0.0;
		router.r_to_l = 0.0;
		router.r_to_r = 0.0;
		let output = render(&mut router, 4_000);

		for frame in &output[3_000..] {
			assert!((frame[0] + 0.25).abs() < 1e-3, "{frame:?}");
			assert!((frame[1] + 0.25).abs() < 1e-3, "{frame:?}");
		}
	}
}
