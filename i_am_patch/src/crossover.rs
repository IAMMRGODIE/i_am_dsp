//! A three-way Linkwitz-Riley 4th-order split.
//!
//! Low, mid and high are LR4 sections at the two crossover points. LR4 is two
//! Butterworth sections in series, so each band is four poles (24 dB/oct).
//!
//! The bands have to sum flat, and they do not if the mid and high are simply
//! the highpass times the next lowpass/highpass: the high band's second split
//! leaves the low band un-compensated. Multiplying the low band by a 2nd-order
//! allpass at the high split restores the identity
//!
//! ```text
//! LP4 + HP4 = allpass
//! ```
//!
//! so the three bands together are two allpass sections: flat magnitude, a
//! constant phase shift, which is what a crossover is allowed to add.

use i_am_dsp::prelude::{Biquad, Effect, ProcessContext};

/// Butterworth Q for one 2nd-order section.
const BUTTERWORTH: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// The three bands a frame is split into.
pub type Bands = [[f32; 2]; 3];

/// The splitter the rack engine keeps between blocks.
pub struct Crossover {
	sample_rate: usize,
	low_hz: f32,
	high_hz: f32,
	low: [Biquad<2>; 2],
	low_allpass: Biquad<2>,
	mid_highpass: [Biquad<2>; 2],
	mid_lowpass: [Biquad<2>; 2],
	high_highpass_low: [Biquad<2>; 2],
	high_highpass_high: [Biquad<2>; 2],
}

impl Crossover {
	/// Builds a splitter at the two crossover points.
	pub fn new(sample_rate: usize, low_hz: f32, high_hz: f32) -> Self {
		let (low_hz, high_hz) = sanitize(sample_rate, low_hz, high_hz);

		let low_allpass = allpass(sample_rate, high_hz);

		Self {
			sample_rate,
			low_hz,
			high_hz,
			low: lowpasses(sample_rate, low_hz),
			low_allpass,
			mid_highpass: highpasses(sample_rate, low_hz),
			mid_lowpass: lowpasses(sample_rate, high_hz),
			high_highpass_low: highpasses(sample_rate, low_hz),
			high_highpass_high: highpasses(sample_rate, high_hz),
		}
	}

	/// Moves the crossover points, rebuilding the sections.
	pub fn set_splits(&mut self, low_hz: f32, high_hz: f32) {
		let (low_hz, high_hz) = sanitize(self.sample_rate, low_hz, high_hz);

		if (low_hz - self.low_hz).abs() < 1e-3 && (high_hz - self.high_hz).abs() < 1e-3 {
			return;
		}

		self.low_hz = low_hz;
		self.high_hz = high_hz;
		self.low = lowpasses(self.sample_rate, low_hz);
		self.mid_highpass = highpasses(self.sample_rate, low_hz);
		self.mid_lowpass = lowpasses(self.sample_rate, high_hz);
		self.high_highpass_low = highpasses(self.sample_rate, low_hz);
		self.high_highpass_high = highpasses(self.sample_rate, high_hz);

		self.low_allpass = allpass(self.sample_rate, high_hz);
	}

	/// The current crossover points, for a test.
	#[cfg(test)]
	pub fn splits(&self) -> (f32, f32) {
		(self.low_hz, self.high_hz)
	}

	/// Splits one frame into low, mid and high.
	pub fn split(
		&mut self,
		input: [f32; 2],
		process_context: &mut Box<dyn ProcessContext>,
	) -> Bands {
		let mut low = input;
		let mut mid = input;
		let mut high = input;

		for filter in &mut self.low {
			filter.process(&mut low, &[], process_context);
		}
		self.low_allpass.process(&mut low, &[], process_context);

		for filter in &mut self.mid_highpass {
			filter.process(&mut mid, &[], process_context);
		}
		for filter in &mut self.mid_lowpass {
			filter.process(&mut mid, &[], process_context);
		}

		for filter in &mut self.high_highpass_low {
			filter.process(&mut high, &[], process_context);
		}
		for filter in &mut self.high_highpass_high {
			filter.process(&mut high, &[], process_context);
		}

		[low, mid, high]
	}
}

/// Keeps the split points inside the band the filters can be designed for.
pub fn sanitize(sample_rate: usize, low_hz: f32, high_hz: f32) -> (f32, f32) {
	let nyquist = sample_rate as f32 * 0.49;
	let low_hz = low_hz.clamp(20.0, nyquist * 0.25);
	let high_hz = high_hz.clamp(low_hz * 1.5, nyquist);

	(low_hz, high_hz)
}

/// A 2nd-order allpass with the same poles as the Butterworth lowpass.
///
/// The numerator is the denominator reversed: a2, a1, 1. This is the phase
/// correction the low band needs after the high split.
fn allpass(sample_rate: usize, cutoff: f32) -> Biquad<2> {
	let low = Biquad::lowpass(sample_rate, cutoff, BUTTERWORTH);
	let mut filter = low.clone();
	filter.b = [low.a[1], low.a[0], 1.0];
	filter
}

fn lowpasses(sample_rate: usize, cutoff: f32) -> [Biquad<2>; 2] {
	[
		Biquad::lowpass(sample_rate, cutoff, BUTTERWORTH),
		Biquad::lowpass(sample_rate, cutoff, BUTTERWORTH),
	]
}

fn highpasses(sample_rate: usize, cutoff: f32) -> [Biquad<2>; 2] {
	[
		Biquad::highpass(sample_rate, cutoff, BUTTERWORTH),
		Biquad::highpass(sample_rate, cutoff, BUTTERWORTH),
	]
}

#[cfg(test)]
mod tests {
	use super::*;

	/// The level the splitter leaves of a sine once the filters have settled.
	fn level(frequency: f32, split_low: f32, split_high: f32) -> f32 {
		let mut crossover = Crossover::new(48_000, split_low, split_high);
		let mut context: Box<dyn ProcessContext> = Box::new(());
		let mut energy = 0.0_f32;

		for index in 0..16_384 {
			let value = (std::f32::consts::TAU * frequency * index as f32 / 48_000.0).sin() * 0.5;
			let bands = crossover.split([value, value], &mut context);
			let sum = bands[0][0] + bands[1][0] + bands[2][0];

			if index >= 8_192 {
				energy += sum * sum;
			}
		}

		(energy / 8_192.0).sqrt()
	}

	/// The three bands must add up to the input, within a tenth of a decibel,
	/// across the whole audible range.
	#[test]
	fn the_three_bands_sum_flat() {
		for frequency in [
			30.0, 60.0, 120.0, 240.0, 1_000.0, 2_000.0, 4_000.0, 10_000.0,
		] {
			let level = level(frequency, 120.0, 2_000.0);
			let reference = 0.5 / 2.0_f32.sqrt();

			let decibels = 20.0 * (level / reference).log10();

			assert!(decibels.abs() < 0.1, "{frequency} Hz: {decibels:.3} dB");
		}
	}

	/// Moving the split points rebuilds the sections.
	#[test]
	fn the_crossover_points_can_move() {
		let mut crossover = Crossover::new(48_000, 120.0, 2_000.0);

		assert_eq!(crossover.splits(), (120.0, 2_000.0));

		crossover.set_splits(200.0, 4_000.0);

		assert_eq!(crossover.splits(), (200.0, 4_000.0));
	}

	/// The allpass by itself is unity gain, which is what makes the low band's
	/// compensation phase-only.
	#[test]
	fn the_compensation_allpass_is_flat() {
		let filter: Biquad<2> = allpass(48_000, 2_000.0);

		for frequency in [20.0, 200.0, 2_000.0, 10_000.0, 20_000.0] {
			let (amplitude, _) = filter.complex_response(frequency);
			assert!(
				(amplitude - 1.0).abs() < 0.01,
				"{frequency} Hz: {amplitude}"
			);
		}
	}
}
