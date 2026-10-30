//! A tape stop: the playback head slows to a halt over a set time, so the
//! pitch falls and the sound freezes, then a new start snaps back to the live
//! input with a short fade.

use i_am_dsp_derive::Parameters;

use crate::{Effect, ProcessContext};

/// A tape stop over a stereo (or wider) signal.
#[derive(Parameters)]
pub struct TapeStopEffect<const CHANNELS: usize = 2> {
	/// Stop the tape. Turning this off starts the tape again.
	#[default(false)]
	#[id(display = "Stop")]
	#[layout(group = "Tape Stop", order = 0, widget = toggle, hero = "waveform")]
	pub stop: bool,

	/// How long the tape takes to reach a standstill.
	#[range(min = 0.05, max = 4.0)]
	#[unit(seconds)]
	#[default(0.5)]
	#[id(display = "Time")]
	#[layout(group = "Tape Stop", order = 1, widget = knob)]
	pub time: f32,

	#[skip]
	sample_rate: f32,
	#[skip]
	buffer: [Vec<f32>; CHANNELS],
	#[skip]
	length: usize,
	#[skip]
	write: usize,
	#[skip]
	read: f64,
	#[skip]
	speed: f32,
	#[skip]
	fade: f32,
	#[skip]
	stopping: bool,
}

impl<const CHANNELS: usize> TapeStopEffect<CHANNELS> {
	/// A tape stop with four seconds of history, enough for the longest time.
	pub fn new(sample_rate: usize) -> Self {
		let length = (sample_rate as f32 * 4.0).ceil() as usize + 2;

		Self {
			stop: false,
			time: 0.5,
			sample_rate: sample_rate as f32,
			buffer: core::array::from_fn(|_| vec![0.0; length]),
			length,
			write: 0,
			read: 0.0,
			speed: 1.0,
			fade: 1.0,
			stopping: false,
		}
	}
}

impl<const CHANNELS: usize> Effect<CHANNELS> for TapeStopEffect<CHANNELS> {
	fn delay(&self) -> usize {
		0
	}

	fn process(
		&mut self,
		samples: &mut [f32; CHANNELS],
		_other: &[&[f32; CHANNELS]],
		_: &mut Box<dyn ProcessContext>,
	) {
		let length = self.length;
		let restart_step = 1.0 / (0.01 * self.sample_rate).max(1.0);

		if !self.stop {
			// A new start is the live input faded in over 10 ms; the tape buffer
			// keeps running so a following stop has fresh audio to slow down.
			if self.stopping {
				self.stopping = false;
				self.speed = 1.0;
				self.read = self.write as f64;
				self.fade = 0.0;
			}

			for (channel, sample) in samples.iter_mut().enumerate() {
				self.buffer[channel][self.write] = *sample;
			}

			self.write = (self.write + 1) % length;
			self.read = self.write as f64;
			self.fade = (self.fade + restart_step).min(1.0);

			for sample in samples.iter_mut() {
				*sample *= self.fade;
			}

			return;
		}

		if !self.stopping {
			self.stopping = true;
			self.read = self.write as f64;
		}

		for (channel, sample) in samples.iter().enumerate() {
			self.buffer[channel][self.write] = *sample;
		}

		let position = self.read.rem_euclid(length as f64);
		let index = position.floor() as usize % length;
		let next = (index + 1) % length;
		let fraction = (position - index as f64) as f32;
		let step = 1.0 / (self.time.max(0.05) * self.sample_rate).max(1.0);

		for (channel, sample) in samples.iter_mut().enumerate() {
			let first = self.buffer[channel][index];
			let second = self.buffer[channel][next];
			*sample = (first + (second - first) * fraction) * self.speed;
		}

		self.write = (self.write + 1) % length;
		self.read += self.speed as f64;
		self.speed = (self.speed - step).max(0.0);

		if self.speed == 0.0 {
			for sample in samples.iter_mut() {
				*sample = 0.0;
			}
		}
	}

	fn clear_tail(&mut self) {
		for buffer in self.buffer.iter_mut() {
			buffer.fill(0.0);
		}

		self.write = 0;
		self.read = 0.0;
		self.speed = 1.0;
		self.fade = 1.0;
		self.stopping = false;
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn render(stop: bool, seconds: f32) -> Vec<[f32; 2]> {
		let mut effect = TapeStopEffect::<2>::new(48_000);
		effect.stop = stop;
		effect.time = 0.1;
		let mut context: Box<dyn ProcessContext> = Box::new(());
		let mut output = Vec::new();

		for index in 0..(seconds * 48_000.0) as usize {
			let phase = 2.0 * std::f32::consts::PI * 220.0 * index as f32 / 48_000.0;
			let value = 0.5 * phase.sin();
			let mut frame = [value, value];
			effect.process(&mut frame, &[], &mut context);
			output.push(frame);
		}

		output
	}

	fn energy(frames: &[[f32; 2]]) -> f32 {
		frames
			.iter()
			.flat_map(|frame| frame.iter())
			.map(|sample| sample * sample)
			.sum()
	}

	/// With the tape running, the effect is the input.
	#[test]
	fn a_running_tape_passes_the_input() {
		let output = render(false, 0.1);

		for (index, frame) in output.iter().enumerate().skip(16) {
			let phase = 2.0 * std::f32::consts::PI * 220.0 * index as f32 / 48_000.0;
			let expected = 0.5 * phase.sin();
			assert!((frame[0] - expected).abs() < 1e-6, "{index}: {frame:?}");
		}
	}

	/// Stopping fades the tape to a standstill within the requested time.
	#[test]
	fn stopping_reaches_silence() {
		let output = render(true, 0.3);
		let early = energy(&output[..2_000]);
		let late = energy(&output[output.len() - 4_800..]);

		assert!(early > 0.0);
		assert!(late < 1e-6, "late energy {late}");
	}

	/// A new start after a stop returns to the live input.
	#[test]
	fn a_new_start_returns_to_the_input() {
		let mut effect = TapeStopEffect::<2>::new(48_000);
		effect.time = 0.05;
		let mut context: Box<dyn ProcessContext> = Box::new(());
		let mut last = [0.0; 2];

		for index in 0..4_800 {
			let phase = 2.0 * std::f32::consts::PI * 220.0 * index as f32 / 48_000.0;
			let mut frame = [0.5 * phase.sin(), 0.5 * phase.sin()];
			effect.stop = index < 2_400;
			effect.process(&mut frame, &[], &mut context);
			last = frame;
		}

		// The 10 ms fade-in has finished by now, so the output is the input again.
		let phase = 2.0 * std::f32::consts::PI * 220.0 * 4_799.0 / 48_000.0;
		assert!((last[0] - 0.5 * phase.sin()).abs() < 1e-5, "{last:?}");
	}
}
