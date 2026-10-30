//! McLeod Pitch Method (MPM) pitch tracking.
//!
//! The tracker follows the paper "A Smarter Way to Find Pitch" by McLeod and
//! Wyvill: a normalised square difference function (NSDF), key maxima between
//! zero crossings, parabolic interpolation and a clarity value. The clarity is
//! the correlation of the chosen peak, so a caller can use it to decide whether
//! the estimate is a tone or just noise.
//!
//! The window is rectangular on purpose: the paper notes that MPM does not need
//! a windowing function, because the NSDF normalisation already handles the
//! shrinking window at large lags. The only preprocessing is removing the DC
//! offset, which would otherwise bias the normalisation.

use std::sync::Arc;

use rustfft::{Fft, FftPlanner, num_complex::Complex};

/// One pitch estimate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PitchEstimate {
	/// The fundamental frequency in hertz.
	pub frequency: f32,
	/// How tone-like the estimate is: 1 is a clear period, 0 is noise.
	pub clarity: f32,
	/// The frequency as a MIDI note number, with a fractional part for cents.
	pub midi_note: f32,
}

/// The lowest frequency the tracker reports.
pub const PITCH_MIN_HZ: f32 = 50.0;
/// The highest frequency the tracker reports.
pub const PITCH_MAX_HZ: f32 = 2_000.0;

/// A monophonic MPM pitch tracker.
pub struct PitchTracker {
	sample_rate: f32,
	window: usize,
	hop: usize,
	ring: Vec<f32>,
	write: usize,
	since_hop: usize,
	fft_len: usize,
	fft: Arc<dyn Fft<f32>>,
	ifft: Arc<dyn Fft<f32>>,
	buffer: Vec<Complex<f32>>,
	prefix: Vec<f32>,
	nsdf: Vec<f32>,
	last: Option<PitchEstimate>,
}

impl PitchTracker {
	/// A tracker with the given window. The window is clamped to a power of two
	/// between 512 and 4096; the default is 2048, which reaches below a low E on
	/// a guitar at 48 kHz.
	pub fn new(sample_rate: usize, window: usize) -> Self {
		let window = window.clamp(512, 4_096).next_power_of_two().min(4_096);
		let hop = (window / 4).max(1);
		let fft_len = window * 2;
		let mut planner = FftPlanner::new();
		let fft = planner.plan_fft_forward(fft_len);
		let ifft = planner.plan_fft_inverse(fft_len);

		Self {
			sample_rate: sample_rate.max(1) as f32,
			window,
			hop,
			ring: vec![0.0; window],
			write: 0,
			since_hop: 0,
			fft_len,
			fft,
			ifft,
			buffer: vec![Complex::new(0.0, 0.0); fft_len],
			prefix: vec![0.0; window + 1],
			nsdf: vec![0.0; window / 2 + 1],
			last: None,
		}
	}

	/// The analysis window in samples.
	pub fn window(&self) -> usize {
		self.window
	}

	/// How many samples between estimates.
	pub fn hop(&self) -> usize {
		self.hop
	}

	/// The last estimate, or nothing before the first window has filled.
	pub fn last(&self) -> Option<PitchEstimate> {
		self.last
	}

	/// Feed one mono sample. Returns an estimate on the hop that completes a
	/// window, and nothing on the samples in between.
	pub fn process_sample(&mut self, sample: f32) -> Option<PitchEstimate> {
		self.ring[self.write] = if sample.is_finite() { sample } else { 0.0 };
		self.write = (self.write + 1) % self.window;
		self.since_hop += 1;

		if self.since_hop >= self.hop {
			self.since_hop = 0;
			self.last = self.estimate();
			return self.last;
		}

		None
	}

	/// Drop the analysis history.
	pub fn reset(&mut self) {
		self.ring.fill(0.0);
		self.write = 0;
		self.since_hop = 0;
		self.last = None;
	}

	fn estimate(&mut self) -> Option<PitchEstimate> {
		let window = self.window;
		let mut mean = 0.0;

		for index in 0..window {
			let value = self.ring[(self.write + index) % window];
			self.buffer[index] = Complex::new(value, 0.0);
			mean += value;
		}

		mean /= window as f32;

		for value in self.buffer.iter_mut().take(window) {
			value.re -= mean;
		}

		for value in self.buffer.iter_mut().skip(window) {
			*value = Complex::new(0.0, 0.0);
		}

		self.prefix[0] = 0.0;

		for index in 0..window {
			let value = self.buffer[index].re;
			self.prefix[index + 1] = self.prefix[index] + value * value;
		}

		let fft = self.fft.clone();
		fft.process(&mut self.buffer);

		// The power spectrum is real and even, so its inverse is the linear
		// autocorrelation once the input was zero-padded to twice the window.
		for value in self.buffer.iter_mut() {
			let power = value.norm_sqr();
			*value = Complex::new(power, 0.0);
		}

		let ifft = self.ifft.clone();
		ifft.process(&mut self.buffer);

		let scale = 1.0 / self.fft_len as f32;
		let max_tau = (window / 2).min((self.sample_rate / PITCH_MIN_HZ).ceil() as usize);
		let min_tau = ((self.sample_rate / PITCH_MAX_HZ).floor() as usize).max(2);

		for tau in 1..=max_tau {
			let first = self.prefix[window - tau];
			let second = self.prefix[window] - self.prefix[tau];
			let energy = first + second;

			self.nsdf[tau] = if energy > 1e-12 {
				2.0 * self.buffer[tau].re * scale / energy
			} else {
				0.0
			};
		}

		// Collect the local maxima of the positive lobes. A maximum between two
		// zero crossings is what the paper calls a key maximum.
		let mut best: Option<(usize, f32, f32)> = None; // (tau, period, value)
		let mut first_above: Option<(f32, f32)> = None;
		let mut highest: f32 = 0.0;

		for tau in (min_tau.max(2))..max_tau {
			let previous = self.nsdf[tau - 1];
			let current = self.nsdf[tau];
			let next = self.nsdf[tau + 1];

			if current <= 0.0 || current <= previous || current < next {
				continue;
			}

			let denominator = previous - 2.0 * current + next;
			let offset = if denominator.abs() > 1e-9 {
				0.5 * (previous - next) / denominator
			} else {
				0.0
			};
			let period = tau as f32 + offset.clamp(-0.5, 0.5);
			let value = current - 0.25 * (previous - next) * offset;

			if let Some((_, _, best_value)) = best {
				if value > best_value {
					best = Some((tau, period, value));
				}
			} else {
				best = Some((tau, period, value));
			}

			highest = highest.max(value);
		}

		if let Some((_, period, value)) = best {
			let threshold = highest * 0.9;

			for tau in (min_tau.max(2))..max_tau {
				let previous = self.nsdf[tau - 1];
				let current = self.nsdf[tau];
				let next = self.nsdf[tau + 1];

				if current <= 0.0 || current <= previous || current < next {
					continue;
				}

				let denominator = previous - 2.0 * current + next;
				let offset = if denominator.abs() > 1e-9 {
					0.5 * (previous - next) / denominator
				} else {
					0.0
				};
				let candidate_period = tau as f32 + offset.clamp(-0.5, 0.5);
				let candidate_value = current - 0.25 * (previous - next) * offset;

				if candidate_value >= threshold {
					first_above = Some((candidate_period, candidate_value));
					break;
				}
			}

			let (period, clarity) = first_above.unwrap_or((period, value));
			let clarity = clarity.clamp(0.0, 1.0);

			if clarity <= 0.05 || period <= 0.0 {
				return None;
			}

			let frequency = self.sample_rate / period;
			let midi_note = 69.0 + 12.0 * (frequency / 440.0).log2();

			return Some(PitchEstimate {
				frequency,
				clarity,
				midi_note,
			});
		}

		None
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn track(frequency: f32, seconds: f32) -> Option<PitchEstimate> {
		let mut tracker = PitchTracker::new(48_000, 2_048);
		let frames = (seconds * 48_000.0) as usize;
		let mut last = None;

		for index in 0..frames {
			let t = index as f32 / 48_000.0;
			let value = 0.5 * (2.0 * std::f32::consts::PI * frequency * t).sin();

			if let Some(estimate) = tracker.process_sample(value) {
				last = Some(estimate);
			}
		}

		last
	}

	/// A 440 Hz sine is a 440 Hz MIDI A.
	#[test]
	fn a_sine_reports_its_frequency_and_note() {
		let estimate = track(440.0, 0.5).expect("a tone has a pitch");

		assert!((estimate.frequency - 440.0).abs() < 1.0, "{estimate:?}");
		assert!((estimate.midi_note - 69.0).abs() < 0.05, "{estimate:?}");
		assert!(estimate.clarity > 0.9, "{estimate:?}");
	}

	/// A bass note below 100 Hz still fits the default window.
	#[test]
	fn a_bass_note_is_tracked() {
		let estimate = track(82.41, 0.8).expect("a low E has a pitch");

		assert!((estimate.frequency - 82.41).abs() < 1.0, "{estimate:?}");
		assert!(estimate.clarity > 0.8, "{estimate:?}");
	}

	/// Silence has no key maximum at all.
	#[test]
	fn silence_has_no_pitch() {
		let mut tracker = PitchTracker::new(48_000, 2_048);

		for _ in 0..48_000 {
			assert!(tracker.process_sample(0.0).is_none());
		}
	}

	/// Noise is not a tone; whatever it reports has low clarity.
	#[test]
	fn noise_has_low_clarity() {
		let mut tracker = PitchTracker::new(48_000, 2_048);
		let mut last = None;
		let mut state = 0x1234_5678u32;

		for _ in 0..48_000 {
			state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
			let value = (state as f32 / u32::MAX as f32) * 2.0 - 1.0;

			if let Some(estimate) = tracker.process_sample(value * 0.3) {
				last = Some(estimate);
			}
		}

		if let Some(estimate) = last {
			assert!(estimate.clarity < 0.6, "{estimate:?}");
		}
	}
}
