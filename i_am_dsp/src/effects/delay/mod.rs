//! Delay related effects.

use i_am_dsp_derive::Parameters;

use crate::{
	Effect, ProcessContext,
	prelude::SyncedTime,
	tools::{interpolate::cubic_interpolate, ring_buffer::RingBuffer},
};

/// How many samples the cubic interpolation of the delay reads.
pub(super) const INTERPOLATION_TAPS: usize = 4;

/// How long a delay time takes to glide to a new value, in seconds.
///
/// Moving a delay time has to move the read head with it, and a read head that jumps is a
/// click. Gliding turns a knob move, or a tempo change under a synced time, into a short
/// tape-like slide instead; the time constant is small enough that a musical change still
/// lands promptly.
const DELAY_GLIDE_SECONDS: f32 = 0.02;

/// The one-pole glide coefficient for [`DELAY_GLIDE_SECONDS`] at a sample rate.
pub(super) fn glide_coefficient(sample_rate: usize) -> f32 {
	1.0 - (-1.0 / (DELAY_GLIDE_SECONDS * sample_rate.max(1) as f32)).exp()
}

/// Equal-power gains for a pan position in [-1, 1]: +1 is left, -1 right.
fn pan_gains(pan: f32) -> (f32, f32) {
	// The UI puts -1 on the left; positive therefore has to be the right side.
	let angle = (1.0 + pan.clamp(-1.0, 1.0)) * std::f32::consts::PI / 4.0;
	(angle.cos(), angle.sin())
}

/// Reads a delayed sample from a ring buffer, with cubic interpolation.
///
/// RingBuffer::sample takes a position from the start of the buffer, but a delay
/// wants a distance from the write head; the Chorus and Flanger used the wrong
/// one and read near the oldest slot instead of their delay time.
pub(super) fn read_delayed(history: &RingBuffer<f32>, delay_samples: f32) -> f32 {
	let capacity = history.capacity() as isize;
	let delay = delay_samples.clamp(0.0, (capacity - 3) as f32);
	let whole = delay.floor() as isize;
	let fraction = delay.fract();

	cubic_interpolate(
		fraction,
		[
			history[capacity + 1 - whole],
			history[capacity - whole],
			history[capacity - 1 - whole],
			history[capacity - 2 - whole],
		],
	)
}

mod chorus;
mod flanger;
mod pure;

pub use chorus::Chorus;
pub use flanger::Flanger;
pub use pure::PureDelay;

/// A delay effect that delays the signal by a variable amount of time.
#[derive(Parameters)]
pub struct Delay<DelayedEffect: Effect<CHANNELS>, const CHANNELS: usize = 2> {
	#[skip]
	history: [RingBuffer<f32>; CHANNELS],
	#[sub_param]
	delayed_effect: DelayedEffect,
	/// The delay time, saves in milliseconds.
	///
	/// Used when [`Self::tempo`] is not syncing. It has no declared default: where a delay
	/// starts is up to whoever builds the effect.
	#[range(min = 0.0, max = 1000.0)]
	#[unit(milliseconds)]
	#[id(display = "Time")]
	#[layout(group = "Delay", order = 0, widget = knob, hero = "taps")]
	pub delay_time: f32,
	/// The same delay time in musical units, for when the host has a tempo.
	#[id(display = "Tempo")]
	#[layout(group = "Delay", order = 1)]
	#[sub_param]
	pub tempo: SyncedTime,
	#[skip]
	/// The sample rate of the audio signal, saves in Hz
	pub sample_rate: usize,
	/// a factor that controls the decay rate of the delay effect
	///
	/// must be between 0 and 1
	#[range(min = 0.0, max = 0.95)]
	#[unit(percent)]
	#[default(0.8)]
	#[id(display = "Feedback")]
	#[layout(group = "Delay", order = 2, widget = knob)]
	pub decay_factor: f32,
	/// The wet gain of the effect, saves in linear scale
	/// The dry gain of the effect.
	#[range(min = 0.0, max = 1.0)]
	#[unit(percent)]
	#[default(1.0)]
	#[id(display = "Dry")]
	#[layout(group = "Delay", order = 3, widget = knob)]
	pub dry_gain: f32,
	/// The wet gain of the effect.
	#[range(min = 0.0, max = 4.0)]
	#[logarithmic]
	#[unit(percent)]
	#[default(1.0)]
	#[id(display = "Wet")]
	#[layout(group = "Delay", order = 4, widget = knob, bipolar)]
	pub wet_gain: f32,
	/// Whether the repeats walk around the channels instead of staying where they started.
	///
	/// The repeats become one chain driven by the sum of the inputs, so a repeat leaves on one
	/// side, the next one on the other, and so on — the alternating pan is the point, and it
	/// holds for a stereo source too. Off, each side repeats into itself and keeps its input.
	///
	/// Summing means out-of-phase input cancels in the repeats, which is the usual trade for
	/// this effect. The dry signal is untouched and keeps its image.
	#[default(false)]
	#[id(display = "Ping-Pong")]
	#[layout(group = "Delay", order = 5, widget = toggle)]
	pub pingpong: bool,
	/// How far a ping-pong repeat is panned: -1 is full left then full right,
	/// +1 starts on the right, and 0 centres both.
	#[range(min = -1.0, max = 1.0)]
	#[unit(percent)]
	#[default(-1.0)]
	#[id(display = "Pan")]
	#[layout(group = "Delay", order = 6, widget = knob, bipolar)]
	pub pingpong_pan: f32,
	/// The delay being read right now, in fractional samples.
	#[skip]
	smoothed_delay: f32,
	/// The delay the parameters ask for, in samples: where the glide is heading.
	#[skip]
	target_delay: f32,
}

impl<DelayedEffect: Effect<CHANNELS>, const CHANNELS: usize> Delay<DelayedEffect, CHANNELS> {
	/// Creates a new delay effect with a given delay time, sample rate, and decay factor.
	///
	/// The maximum delay time is limited by the maximum delay length.
	///
	/// # Panics
	///
	/// `CHANNELS` is 0
	pub fn new(
		delayed_effect: DelayedEffect,
		maxium_delay_length: usize,
		delay_time: f32,
		sample_rate: usize,
	) -> Self {
		assert!(CHANNELS > 0, "CHANNELS must be greater than 0");
		let history = core::array::from_fn(|_| RingBuffer::new(maxium_delay_length));

		// The glide starts where the parameter is, so a fresh delay has no transient to
		// settle through before it sounds like itself.
		let delay = delay_time / 1000.0 * sample_rate as f32;

		Self {
			history,
			delayed_effect,
			delay_time,
			tempo: SyncedTime::default(),
			sample_rate,
			wet_gain: 1.0,
			decay_factor: 0.8,
			dry_gain: 1.0,
			pingpong: false,
			pingpong_pan: -1.0,
			smoothed_delay: delay,
			target_delay: delay,
		}
	}

	/// caculate the maxium delay time that can be set without overflowing the buffer
	///
	/// sample lower than epsilon will treated as zero
	///
	/// panics if the decay factor is less than zero or greater than one
	pub fn max_delay_time(&self, epsilon: f32) -> f32 {
		assert!(self.decay_factor > 0.0 && self.decay_factor < 1.0);

		let max_time = self.history[0].capacity() as f32 / self.sample_rate as f32;
		let zero_time = epsilon.ln() / self.decay_factor.ln();
		max_time * zero_time * 1000.0
	}

	/// Clear delay history.
	pub fn clear_history(&mut self) {
		for buffer in self.history.iter_mut() {
			buffer.clear();
		}

		// A cleared line leaves the glide where it is: the delay time is a parameter, not
		// part of the history being wiped.
		self.smoothed_delay = self.target_delay;
	}

	/// Resize delay history.
	pub fn resize_history(&mut self, new_capacity: usize) {
		for buffer in self.history.iter_mut() {
			buffer.resize(new_capacity);
		}

		// The new capacity may be shorter than the delay that was being read.
		let max_delay = new_capacity.saturating_sub(INTERPOLATION_TAPS) as f32;
		self.smoothed_delay = self.smoothed_delay.min(max_delay);
		self.target_delay = self.target_delay.min(max_delay);
	}

	/// Toggles the ping-pong mode.
	pub fn toggle_pingpong(&mut self) {
		self.pingpong = !self.pingpong;
	}

	/// Returns true if the effect is in ping-pong mode.
	pub fn pingpong(&self) -> bool {
		self.pingpong
	}
}

impl<DelayedEffect: Effect<CHANNELS>, const CHANNELS: usize> Effect<CHANNELS>
	for Delay<DelayedEffect, CHANNELS>
{
	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"Delay"
	}

	fn clear_tail(&mut self) {
		self.clear_history();
	}

	fn delay(&self) -> usize {
		// The dry signal passes through undelayed, so nothing has to be compensated for;
		// the repeats are the effect, not latency.
		0
	}

	fn process(
		&mut self,
		samples: &mut [f32; CHANNELS],
		_: &[&[f32; CHANNELS]],
		ctx: &mut Box<dyn ProcessContext>,
	) {
		let dry = *samples;
		let history_len = self.history[0].capacity();
		// The interpolation reads one sample either side of the delay, so the last few slots
		// cannot be addressed.
		let max_delay = history_len.saturating_sub(INTERPOLATION_TAPS) as f32;

		let target = (self.tempo.seconds(ctx.infos(), self.delay_time) * self.sample_rate as f32)
			.clamp(0.0, max_delay);
		self.target_delay = target;
		self.smoothed_delay = (self.smoothed_delay
			+ (target - self.smoothed_delay) * glide_coefficient(self.sample_rate))
		.clamp(0.0, max_delay);

		let delay_samples = self.smoothed_delay;
		let t = delay_samples.fract();
		let whole = delay_samples.floor() as isize;
		let history_len = history_len as isize;

		// Read every tap before anything is written, because a repeat is fed from another
		// channel's output.
		// `history[capacity - k]` is the sample written `k` samples ago, and the cubic reads
		// from the second tap to the third: the window has to run from `whole - 1` ago to
		// `whole + 2` ago for the interpolation to land on `whole + t`. Reversed, the read sat
		// one sample late, and every trip around the feedback loop added another.
		let mut delayed = [0.0; CHANNELS];
		for (i, history) in self.history.iter().enumerate() {
			let interpolate_parameters = [
				history[history_len + 1 - whole],
				history[history_len - whole],
				history[history_len - 1 - whole],
				history[history_len - 2 - whole],
			];
			delayed[i] = cubic_interpolate(t, interpolate_parameters);
		}

		// Ping-pong is one chain of repeats that walks around the channels, so a repeat comes
		// out of one side at a time and the sides alternate. Feeding each line from the line
		// before it is what does that; letting both sides keep their own input would make them
		// repeat together, and a stereo source would never alternate at all.
		let pingpong_input = if self.pingpong {
			// The chain is driven by the sum: the repeats are the effect, and the dry signal
			// keeps the stereo image.
			Some(samples.iter().sum::<f32>() / CHANNELS as f32)
		} else {
			None
		};

		let mut delayed_sample_output = [0.0; CHANNELS];
		for (i, history) in self.history.iter_mut().enumerate() {
			let input = match pingpong_input {
				Some(mono) => {
					let previous = if i == 0 { CHANNELS - 1 } else { i - 1 };
					let repeat = delayed[previous] * self.wet_gain * self.decay_factor;
					if i == 0 { mono + repeat } else { repeat }
				}
				None => samples[i] + delayed[i] * self.wet_gain * self.decay_factor,
			};

			history.push(input);
			delayed_sample_output[i] = delayed[i] * self.decay_factor;
		}

		self.delayed_effect
			.process(&mut delayed_sample_output, &[], ctx);

		if self.pingpong && CHANNELS == 2 {
			// The chain still alternates the repeats internally; the pan decides how
			// much of each repeat reaches each side. +1 keeps the old full left then
			// full right, -1 starts on the right, 0 centres both.
			let (left_left, left_right) = pan_gains(self.pingpong_pan);
			let (right_left, right_right) = pan_gains(-self.pingpong_pan);
			let left = delayed_sample_output[0] * self.wet_gain;
			let right = delayed_sample_output[1] * self.wet_gain;

			samples[0] = dry[0] * self.dry_gain + left * left_left + right * right_left;
			samples[1] = dry[1] * self.dry_gain + left * left_right + right * right_right;
		} else {
			delayed_sample_output
				.into_iter()
				.enumerate()
				.for_each(|(i, mut inner)| {
					inner *= self.wet_gain;
					samples[i] = dry[i] * self.dry_gain + inner;
				});
		}

		if ctx.should_stop() {
			self.clear_history();
		}
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		use crate::tools::ui_tools::gain_ui;

		// Ping-pong no longer shortens the usable delay: each side runs the whole time and
		// feeds the other, instead of the read offset being stretched across the channels.
		let delay_range = 1.0..=1000.0_f32
			.min(self.history[0].capacity() as f32 / self.sample_rate as f32 * 1000.0);

		ui.checkbox(&mut self.pingpong, "Ping-Pong");
		crate::tools::ui_tools::synced_time_ui(
			ui,
			&id_prefix,
			&mut self.tempo,
			&mut self.delay_time,
			delay_range,
			"Delay Time (ms)",
		);
		ui.add(egui::Slider::new(&mut self.decay_factor, 0.01..=0.8).text("Decay Factor"));
		gain_ui(ui, &mut self.wet_gain, Some("Wet Gain".to_string()), false);

		self.delayed_effect
			.demo_ui(ui, format!("{}_delayed_effect", id_prefix));
	}
}

#[cfg(test)]
mod tests;
