//! Polyphase oversampling for the stages that cannot run at the host rate.
//!
//! A shaper that clips, folds or quantizes makes harmonics above the host's
//! Nyquist, and those fold back into the audible band. Running the shaper at
//! four times the rate and filtering the result keeps most of them out.

use std::f32::consts::PI;

/// The taps of the half-band kernel, at twice the rate it is used on.
const TAPS: usize = 33;

/// How many input samples the four-times path delays by, in input samples.
pub const LATENCY: usize = 3 * (TAPS - 1) / 4;

/// The low-pass kernel: a windowed sinc with its cutoff on the input's Nyquist.
///
/// Half of its taps are zero, which is what lets the two phases below share one
/// kernel: the even taps interpolate, the odd taps hold the images.
fn kernel() -> [f32; TAPS] {
	let center = (TAPS - 1) as f32 / 2.0;
	let mut taps = [0.0; TAPS];
	let mut sum = 0.0;

	for (index, tap) in taps.iter_mut().enumerate() {
		let offset = index as f32 - center;
		let sinc = if offset == 0.0 {
			0.5
		} else {
			(PI * 0.5 * offset).sin() / (PI * offset)
		};
		let angle = 2.0 * PI * index as f32 / (TAPS - 1) as f32;
		let window = 0.42 - 0.5 * angle.cos() + 0.08 * (2.0 * angle).cos();

		*tap = sinc * window;
		sum += *tap;
	}

	// A constant has to come out as itself.
	for tap in &mut taps {
		*tap /= sum;
	}

	taps
}

/// One two-times stage: interpolate, let the shaper run, then decimate.
pub struct HalfBand<const CHANNELS: usize> {
	kernel: [f32; TAPS],
	input: [[f32; TAPS]; CHANNELS],
	even: [[f32; TAPS]; CHANNELS],
	odd: [[f32; TAPS]; CHANNELS],
	write: usize,
}

impl<const CHANNELS: usize> HalfBand<CHANNELS> {
	/// A stage with an empty history.
	pub fn new() -> Self {
		assert!(CHANNELS > 0, "CHANNELS must be greater than 0");

		Self {
			kernel: kernel(),
			input: [[0.0; TAPS]; CHANNELS],
			even: [[0.0; TAPS]; CHANNELS],
			odd: [[0.0; TAPS]; CHANNELS],
			write: 0,
		}
	}

	/// The delay the two filters add, in the stage's input samples.
	pub const fn latency() -> usize {
		(TAPS - 1) / 2
	}

	fn history(samples: &[f32; TAPS], write: usize, back: usize) -> f32 {
		samples[(write + TAPS - back) % TAPS]
	}

	/// The two samples of one input frame, at twice its rate.
	pub fn upsample(&mut self, input: [f32; CHANNELS]) -> ([f32; CHANNELS], [f32; CHANNELS]) {
		for (channel, value) in input.iter().enumerate() {
			self.input[channel][self.write] = *value;
		}

		let mut even = [0.0; CHANNELS];
		let mut odd = [0.0; CHANNELS];

		for channel in 0..CHANNELS {
			for tap in 0..=TAPS / 2 {
				even[channel] +=
					self.kernel[2 * tap] * Self::history(&self.input[channel], self.write, tap);
			}

			for tap in 0..TAPS / 2 {
				odd[channel] +=
					self.kernel[2 * tap + 1] * Self::history(&self.input[channel], self.write, tap);
			}
		}

		// The interpolation runs on the zero-stuffed signal, whose average is
		// half of the input's; the factor puts the level back.
		for value in even.iter_mut().chain(odd.iter_mut()) {
			*value *= 2.0;
		}

		(even, odd)
	}

	/// The one sample of the shaped frame pair, at the stage's input rate.
	pub fn downsample(&mut self, even: [f32; CHANNELS], odd: [f32; CHANNELS]) -> [f32; CHANNELS] {
		for (channel, value) in even.iter().enumerate() {
			self.even[channel][self.write] = *value;
			self.odd[channel][self.write] = odd[channel];
		}

		let mut output = [0.0; CHANNELS];

		for (channel, value) in output.iter_mut().enumerate() {
			for tap in 0..=TAPS / 2 {
				*value +=
					self.kernel[2 * tap] * Self::history(&self.even[channel], self.write, tap);
			}

			for tap in 0..TAPS / 2 {
				*value += self.kernel[2 * tap + 1]
					* Self::history(&self.odd[channel], self.write, tap + 1);
			}
		}

		self.write = (self.write + 1) % TAPS;

		output
	}

	/// Forgets the history, so the next frame starts from silence.
	pub fn clear(&mut self) {
		self.input = [[0.0; TAPS]; CHANNELS];
		self.even = [[0.0; TAPS]; CHANNELS];
		self.odd = [[0.0; TAPS]; CHANNELS];
		self.write = 0;
	}
}

impl<const CHANNELS: usize> Default for HalfBand<CHANNELS> {
	fn default() -> Self {
		Self::new()
	}
}

/// Four-times oversampling around a nonlinear stage.
pub struct Oversampler<const CHANNELS: usize> {
	first: HalfBand<CHANNELS>,
	second: HalfBand<CHANNELS>,
}

impl<const CHANNELS: usize> Oversampler<CHANNELS> {
	/// An oversampler with an empty history.
	pub fn new() -> Self {
		Self {
			first: HalfBand::new(),
			second: HalfBand::new(),
		}
	}

	/// The delay the filters add, in input samples.
	pub const fn latency() -> usize {
		LATENCY
	}

	/// Shapes one input frame at four times the input rate.
	pub fn process(
		&mut self,
		input: [f32; CHANNELS],
		shape: &mut impl FnMut([f32; CHANNELS]) -> [f32; CHANNELS],
	) -> [f32; CHANNELS] {
		let (first_even, first_odd) = self.first.upsample(input);

		let (second_even, second_odd) = self.second.upsample(first_even);
		let first_even = self
			.second
			.downsample(shape(second_even), shape(second_odd));

		let (third_even, third_odd) = self.second.upsample(first_odd);
		let first_odd = self.second.downsample(shape(third_even), shape(third_odd));

		self.first.downsample(first_even, first_odd)
	}

	/// Forgets the history, so the next frame starts from silence.
	pub fn clear(&mut self) {
		self.first.clear();
		self.second.clear();
	}
}

impl<const CHANNELS: usize> Default for Oversampler<CHANNELS> {
	fn default() -> Self {
		Self::new()
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn identity<const CHANNELS: usize>(frame: [f32; CHANNELS]) -> [f32; CHANNELS] {
		frame
	}

	/// The filters must not delay more or less than the latency says.
	#[test]
	fn the_latency_is_where_the_impulse_lands() {
		let mut oversampler = Oversampler::<1>::new();
		let mut output = Vec::new();

		for index in 0..128 {
			let input = [if index == 0 { 1.0 } else { 0.0 }];
			output.push(oversampler.process(input, &mut identity)[0]);
		}

		let peak = output
			.iter()
			.enumerate()
			.max_by(|left, right| left.1.abs().partial_cmp(&right.1.abs()).unwrap())
			.map(|(index, _)| index)
			.unwrap();

		assert_eq!(peak, Oversampler::<1>::latency());
	}

	/// The kernel sums to one, so a constant comes out as itself.
	#[test]
	fn a_constant_passes_unchanged() {
		let mut oversampler = Oversampler::<1>::new();
		let mut last = 0.0;

		for _ in 0..256 {
			last = oversampler.process([0.5], &mut identity)[0];
		}

		assert!((last - 0.5).abs() < 1e-3, "{last}");
	}

	/// A tone below the cutoff keeps its amplitude.
	#[test]
	fn a_tone_keeps_its_amplitude() {
		let mut oversampler = Oversampler::<1>::new();
		let mut peak = 0.0f32;

		for index in 0..4_096 {
			let input = [(2.0 * PI * 1_000.0 * index as f32 / 48_000.0).sin()];
			let output = oversampler.process(input, &mut identity)[0];

			if index > 2_048 {
				peak = peak.max(output.abs());
			}
		}

		assert!((peak - 1.0).abs() < 1e-2, "{peak}");
	}

	/// The shaper sees one frame per quarter of an input sample.
	#[test]
	fn the_shaper_runs_four_times_per_sample() {
		let mut oversampler = Oversampler::<2>::new();
		let mut calls = 0;

		for _ in 0..10 {
			oversampler.process([0.0; 2], &mut |frame| {
				calls += 1;
				frame
			});
		}

		assert_eq!(calls, 40);
	}
}
