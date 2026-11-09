use super::{INTERPOLATION_TAPS, glide_coefficient};
use i_am_dsp_derive::Parameters;

use crate::{
	Effect, ProcessContext, ProcessInfos, prelude::SyncedTime,
	tools::interpolate::cubic_interpolate,
};

/// A pure delay that delays the signal by a fixed amount of time.
#[derive(Parameters)]
pub struct PureDelay<const CHANNELS: usize = 2> {
	/// The delay history.
	///
	/// A plain ring with an explicit position: the fractional read below touches
	/// it five times per sample, and the modulo a ring buffer needs per access is
	/// a measurable part of that.
	#[skip]
	history: [Vec<f32>; CHANNELS],
	/// The slot that was written last, which holds the newest sample.
	#[skip]
	position: usize,
	/// The delay time, saves in milliseconds
	///
	/// Used when [`Self::tempo`] is not syncing. It has no declared default: where a delay
	/// starts is up to whoever builds the effect.
	#[range(min = 0.0, max = 4000.0)]
	#[unit(milliseconds)]
	pub delay_time: f32,
	/// The same delay time in musical units, for when the host has a tempo.
	#[id(display = "Tempo")]
	#[sub_param]
	pub tempo: SyncedTime,
	/// The sample rate of the audio signal, saves in Hz
	#[skip]
	pub sample_rate: usize,
	/// The delay being read right now, in fractional samples.
	#[skip]
	pub(super) smoothed_delay: f32,
	/// The delay the parameters ask for, in samples: where the glide is heading.
	#[skip]
	target_delay: f32,
}

impl<const CHANNELS: usize> PureDelay<CHANNELS> {
	/// Creates a new pure delay with a given delay time and sample rate.
	///
	/// The maximum delay time is limited by the maximum delay length.
	///
	/// # Panics
	///
	/// `CHANNELS` is 0
	pub fn new(maxium_delay_length: usize, delay_time: f32, sample_rate: usize) -> Self {
		assert!(CHANNELS > 0, "CHANNELS must be greater than 0");

		// The glide starts where the parameter is, so a fresh delay has no transient to
		// settle through before it sounds like itself.
		let delay = delay_time / 1000.0 * sample_rate as f32;

		Self {
			history: core::array::from_fn(|_| vec![0.0; maxium_delay_length]),
			position: 0,
			delay_time,
			tempo: SyncedTime::default(),
			sample_rate,
			smoothed_delay: delay,
			target_delay: delay,
		}
	}

	/// Clear delay history.
	pub fn clear_history(&mut self) {
		for buffer in self.history.iter_mut() {
			buffer.fill(0.0);
		}
		self.position = 0;
	}

	/// Resize delay history.
	pub fn resize_history(&mut self, new_capacity: usize) {
		for buffer in self.history.iter_mut() {
			buffer.clear();
			buffer.resize(new_capacity, 0.0);
		}
		self.position = 0;
	}

	/// Returns the maximum delay time that can be set without overflowing the buffer.
	pub fn maxium_delay_time(&self) -> f32 {
		self.history[0].len().saturating_sub(INTERPOLATION_TAPS - 1) as f32
			/ self.sample_rate as f32
			* 1000.0
	}

	/// Returns the history length.
	pub fn history_len(&self) -> usize {
		self.history[0].len()
	}

	/// The delay the parameters ask for, in fractional samples.
	///
	/// The value is fractional because the read below interpolates, and it comes from
	/// [`SyncedTime`] because a synced time needs the tempo the process context carries.
	fn target_delay_samples(&self, infos: &ProcessInfos) -> f32 {
		self.tempo.seconds(infos, self.delay_time) * self.sample_rate as f32
	}

	/// The sample written 'delay' samples before the newest one.
	///
	/// 'delay' must be smaller than the history length.
	#[inline]
	fn tap(&self, channel: usize, delay: usize) -> f32 {
		let buffer = &self.history[channel];
		if delay <= self.position {
			buffer[self.position - delay]
		} else {
			buffer[self.position + buffer.len() - delay]
		}
	}
}

impl<const CHANNELS: usize> Effect<CHANNELS> for PureDelay<CHANNELS> {
	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"PureDelay"
	}

	fn delay(&self) -> usize {
		// The wet signal is delayed by the delay time, so that is what the dry signal has to
		// be aligned against, and it is the target rather than the glide (delay compensation
		// wants where the delay is going, not where it is mid-slide).
		//
		// A free time can be answered straight from the parameter. A synced one cannot:
		// the tempo lives in the process context, so the last processed sample is the only
		// thing that knows it.
		if self.tempo.sync {
			self.target_delay.round().max(0.0) as usize
		} else {
			(self.delay_time / 1000.0 * self.sample_rate as f32)
				.round()
				.max(0.0) as usize
		}
	}

	fn process(
		&mut self,
		samples: &mut [f32; CHANNELS],
		_: &[&[f32; CHANNELS]],
		ctx: &mut Box<dyn ProcessContext>,
	) {
		let capacity = self.history[0].len();
		// The interpolation reads the sample before the integer part of the delay
		// and the two after it, so the last slots cannot be addressed.
		if capacity < INTERPOLATION_TAPS {
			return;
		}
		let max_delay = (capacity - INTERPOLATION_TAPS) as f32;
		let target = self.target_delay_samples(ctx.infos()).clamp(0.0, max_delay);
		self.target_delay = target;
		self.smoothed_delay = (self.smoothed_delay
			+ (target - self.smoothed_delay) * glide_coefficient(self.sample_rate))
		.clamp(0.0, max_delay);

		let delay = self.smoothed_delay;
		let whole = delay.floor() as usize;
		let fraction = delay - whole as f32;

		// Advance first, so that the interpolation can reach the current sample
		// when the delay rounds down to zero.
		self.position += 1;
		if self.position == capacity {
			self.position = 0;
		}

		for (i, sample) in samples.iter_mut().enumerate() {
			self.history[i][self.position] = *sample;
			// The taps straddle the integer part of the delay, so interpolating
			// between the middle two of them lands on the requested delay.
			let taps = [
				self.tap(i, whole.saturating_sub(1)),
				self.tap(i, whole),
				self.tap(i, whole + 1),
				self.tap(i, whole + 2),
			];
			*sample = cubic_interpolate(fraction, taps);
		}

		if ctx.should_stop() {
			self.clear_history();
		}
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, _: String) {
		let delay_range = 1.0..=self.maxium_delay_time();

		crate::tools::ui_tools::synced_time_ui(
			ui,
			"pure_delay",
			&mut self.tempo,
			&mut self.delay_time,
			delay_range,
			"Delay Time (ms)",
		);
	}
}
