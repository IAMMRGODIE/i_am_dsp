//! A pitch-synchronous overlap-add pitch shifter.
//!
//! The pitch tracker gives the current period; grains are two periods long and
//! are replayed at the pitch ratio over a short crossfade at their edges. Reading
//! the grain faster or slower changes pitch while the grain hop stays tied to the
//! output timeline, so duration is preserved. The short edge fade keeps the grain
//! boundaries from clicking without smearing the pitch across grain copies.

use crate::{
	Effect, ProcessContext,
	tools::pitch::PitchTracker,
};

/// Long enough for a six-period grain at the lowest tracked note and a 4x shift.
const HISTORY: usize = 32_768;
/// Output overlap-add ring; the same span plus the scheduling latency.
const OLA: usize = 32_768;
/// Output latency in samples: grains are written this far ahead of the read head.
const LATENCY: usize = 4_096;
/// The crossfade length at each grain edge.
const FADE: usize = 16;

/// A pitch-synchronous OLA pitch shifter.
pub struct Psola<const CHANNELS: usize = 2> {
	input: [Vec<f32>; CHANNELS],
	ola: [Vec<f32>; CHANNELS],
	norm: [Vec<f32>; CHANNELS],
	write: usize,
	read: usize,
	ola_write: usize,
	timer: f32,
	period: f32,
	shift: f32,
	sample_rate: f32,
	tracker: PitchTracker,
}

impl<const CHANNELS: usize> Psola<CHANNELS> {
	/// A shifter at unity pitch.
	pub fn new(sample_rate: usize) -> Self {
		assert!(CHANNELS > 0, "CHANNELS must be greater than 0");

		Self {
			input: core::array::from_fn(|_| vec![0.0; HISTORY]),
			ola: core::array::from_fn(|_| vec![0.0; OLA]),
			norm: core::array::from_fn(|_| vec![0.0; OLA]),
			write: 0,
			read: 0,
			ola_write: LATENCY,
			timer: 0.0,
			period: 512.0,
			shift: 1.0,
			sample_rate: sample_rate.max(1) as f32,
			tracker: PitchTracker::new(sample_rate, 2_048),
		}
	}

	/// Set the pitch ratio. One is no shift.
	pub fn set_shift(&mut self, factor: f32) {
		self.shift = factor.clamp(0.25, 4.0);
	}

	/// The tracker's last estimate, for tests and diagnostics.
	pub fn last_estimate(&self) -> Option<crate::tools::pitch::PitchEstimate> {
		self.tracker.last()
	}

	fn schedule_grain(&mut self) {
		let length = (self.period * 2.0).clamp(64.0, 4_096.0) as usize;
		let fade = FADE.min(length / 8).max(1);
		let hop = (length - fade).max(1);
		let source_span = (length as f32 * self.shift).ceil() as usize;

		if self.write < source_span {
			return;
		}

		let start = (self.write + HISTORY - source_span) % HISTORY;
		let ola_start = self.ola_write;

		for channel in 0..CHANNELS {
			for index in 0..length {
				let source = start as f32 + index as f32 * self.shift;
				let base = source.floor() as usize % HISTORY;
				let next = (base + 1) % HISTORY;
				let fraction = source.fract();
				let sample = self.input[channel][base] * (1.0 - fraction)
					+ self.input[channel][next] * fraction;

				let window = if index < fade {
					0.5 - 0.5 * (core::f32::consts::PI * index as f32 / fade as f32).cos()
				} else if index >= length - fade {
					0.5 - 0.5
						* (core::f32::consts::PI * (length - index) as f32 / fade as f32).cos()
				} else {
					1.0
				};

				let at = (ola_start + index) % OLA;
				self.ola[channel][at] += sample * window;
				self.norm[channel][at] += window;
			}
		}

		self.ola_write = (ola_start + hop) % OLA;
	}

	fn tracker_period(&mut self, mono: f32) {
		if let Some(estimate) = self.tracker.process_sample(mono)
			&& estimate.clarity >= 0.45
			&& estimate.frequency > 0.0
		{
			self.period = (self.sample_rate / estimate.frequency).clamp(64.0, 2_048.0);
		}
	}
}

impl<const CHANNELS: usize> crate::prelude::Parameters for Psola<CHANNELS> {
	fn get_parameters(&self) -> Vec<crate::prelude::Parameter> {
		Vec::new()
	}

	fn set_parameter(&mut self, _: &str, _: crate::prelude::SetValue) -> bool {
		false
	}
}

impl<const CHANNELS: usize> Effect<CHANNELS> for Psola<CHANNELS> {
	fn delay(&self) -> usize {
		LATENCY
	}

	fn process(
		&mut self,
		samples: &mut [f32; CHANNELS],
		_: &[&[f32; CHANNELS]],
		_: &mut Box<dyn ProcessContext>,
	) {
		let mono = samples.iter().copied().sum::<f32>() / CHANNELS as f32;
		let input = *samples;
		self.tracker_period(mono);

		for (channel, sample) in samples.iter_mut().enumerate() {
			let at = self.read;
			let weight = self.norm[channel][at];
			*sample = if weight > 1e-6 {
				self.ola[channel][at] / weight
			} else {
				0.0
			};
			self.ola[channel][at] = 0.0;
			self.norm[channel][at] = 0.0;
		}

		for (channel, sample) in input.iter().enumerate() {
			self.input[channel][self.write] = *sample;
		}
		self.write = (self.write + 1) % HISTORY;

		self.timer -= 1.0;
		if self.timer <= 0.0 {
			self.schedule_grain();
			let length = (self.period * 2.0).clamp(64.0, 4_096.0) as usize;
			let fade = FADE.min(length / 8).max(1);
			self.timer += (length - fade).max(1) as f32;
		}

		self.read = (self.read + 1) % OLA;
	}

	fn clear_tail(&mut self) {
		for channel in 0..CHANNELS {
			self.input[channel].fill(0.0);
			self.ola[channel].fill(0.0);
			self.norm[channel].fill(0.0);
		}
		self.write = 0;
		self.read = 0;
		self.ola_write = LATENCY;
		self.timer = 0.0;
		self.tracker.reset();
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::tools::pitch::PitchTracker;

	fn render(factor: f32) -> Vec<f32> {
		let mut shifter = Psola::<1>::new(48_000);
		shifter.set_shift(factor);
		let mut context: Box<dyn ProcessContext> = Box::new(());
		let mut output = Vec::new();

		for index in 0..48_000 {
			let phase = 2.0 * core::f32::consts::PI * 220.0 * index as f32 / 48_000.0;
			let mut frame = [0.5 * phase.sin()];
			shifter.process(&mut frame, &[], &mut context);
			output.push(frame[0]);
		}

		output
	}

	fn estimate(output: &[f32]) -> f32 {
		let mut tracker = PitchTracker::new(48_000, 2_048);
		let mut last = None;
		for sample in &output[12_000..] {
			if let Some(next) = tracker.process_sample(*sample) && next.clarity > 0.5 {
				last = Some(next.frequency);
			}
		}
		last.unwrap_or(0.0)
	}

	#[test]
	fn the_pitch_ratio_moves_the_tracked_note() {
		let unity = estimate(&render(1.0));
		let up = estimate(&render(1.5));

		assert!((unity - 220.0).abs() < 5.0, "unity {unity}");
		assert!((up - 330.0).abs() < 15.0, "up {up}");
	}

	#[test]
	fn the_output_stays_finite() {
		for factor in [0.5, 1.0, 2.0, 4.0] {
			let output = render(factor);
			assert!(output.iter().all(|sample| sample.is_finite()));
			assert!(output[12_000..].iter().any(|sample| sample.abs() > 1e-3));
		}
	}
}
