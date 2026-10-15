//! Delay related effects.

use i_am_dsp_derive::Parameters;

use crate::{ProcessInfos, generators::wavetable::WaveTable, prelude::SyncedTime, tools::{interpolate::cubic_interpolate, ring_buffer::RingBuffer}, Effect, ProcessContext};

/// How many samples the cubic interpolation of the delay reads.
const INTERPOLATION_TAPS: usize = 4;

/// How long a delay time takes to glide to a new value, in seconds.
///
/// Moving a delay time has to move the read head with it, and a read head that jumps is a
/// click. Gliding turns a knob move, or a tempo change under a synced time, into a short
/// tape-like slide instead; the time constant is small enough that a musical change still
/// lands promptly.
const DELAY_GLIDE_SECONDS: f32 = 0.02;

/// The one-pole glide coefficient for [`DELAY_GLIDE_SECONDS`] at a sample rate.
fn glide_coefficient(sample_rate: usize) -> f32 {
	1.0 - (-1.0 / (DELAY_GLIDE_SECONDS * sample_rate.max(1) as f32)).exp()
}

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
	/// Used when [`Self::tempo`] is not syncing.
	#[range(min = 0.0, max = 4000.0)]
	pub delay_time: f32,
	/// The same delay time in musical units, for when the host has a tempo.
	#[sub_param]
	pub tempo: SyncedTime,
	/// The sample rate of the audio signal, saves in Hz
	#[skip]
	pub sample_rate: usize,
	/// The delay being read right now, in fractional samples.
	#[skip]
	smoothed_delay: f32,
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
	pub fn new(
		maxium_delay_length: usize,
		delay_time: f32, 
		sample_rate: usize
	) -> Self {
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
		self.history[0].len().saturating_sub(INTERPOLATION_TAPS - 1) as f32 / self.sample_rate as f32 * 1000.0
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
		}else {
			(self.delay_time / 1000.0 * self.sample_rate as f32).round().max(0.0) as usize
		}
	}
	
	fn process(&mut self, samples: &mut [f32; CHANNELS], _: &[&[f32; CHANNELS]], ctx: &mut Box<dyn ProcessContext>) {
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

		crate::tools::ui_tools::synced_time_ui(ui, "pure_delay", &mut self.tempo, &mut self.delay_time, delay_range);
	}
}

/// A delay effect that delays the signal by a variable amount of time.
#[derive(Parameters)]
pub struct Delay<
	DelayedEffect: Effect<CHANNELS>, 
	const CHANNELS: usize = 2
> {
	#[skip]
	history: [RingBuffer<f32>; CHANNELS],
	#[sub_param]
	delayed_effect: DelayedEffect,
	/// The delay time, saves in milliseconds.
	///
	/// Used when [`Self::tempo`] is not syncing.
	#[range(min = 0.0, max = 1000.0)]
	pub delay_time: f32,
	/// The same delay time in musical units, for when the host has a tempo.
	#[sub_param]
	pub tempo: SyncedTime,
	#[skip]
	/// The sample rate of the audio signal, saves in Hz
	pub sample_rate: usize,
	/// a factor that controls the decay rate of the delay effect
	/// 
	/// must be between 0 and 1
	pub decay_factor: f32,
	/// The wet gain of the effect, saves in linear scale
	#[range(min = 0.01, max = 4.0)]
	#[logarithmic]
	pub wet_gain: f32,
	/// Whether the repeats alternate between the channels.
	///
	/// Each side feeds the other, so an echo leaves on one side, comes back on the other,
	/// and so on. Off, each side repeats into itself.
	pub pingpong: bool,
	/// The delay being read right now, in fractional samples.
	#[skip]
	smoothed_delay: f32,
	/// The delay the parameters ask for, in samples: where the glide is heading.
	#[skip]
	target_delay: f32,
}


impl<
	DelayedEffect: Effect<CHANNELS>, 
	const CHANNELS: usize
> Delay<DelayedEffect, CHANNELS> {
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
		sample_rate: usize
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
			pingpong: false,
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

impl<
	DelayedEffect: Effect<CHANNELS>, 
	const CHANNELS: usize
> Effect<CHANNELS> for Delay<DelayedEffect, CHANNELS> {
	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"Delay"
	}

	fn delay(&self) -> usize {
		// The dry signal passes through undelayed, so nothing has to be compensated for;
		// the repeats are the effect, not latency.
		0
	}
	
	fn process(&mut self, samples: &mut [f32; CHANNELS], _: &[&[f32; CHANNELS]], ctx: &mut Box<dyn ProcessContext>) {
		let history_len = self.history[0].capacity();
		// The interpolation reads one sample either side of the delay, so the last few slots
		// cannot be addressed.
		let max_delay = history_len.saturating_sub(INTERPOLATION_TAPS) as f32;

		let target = (self.tempo.seconds(ctx.infos(), self.delay_time)
			* self.sample_rate as f32)
			.clamp(0.0, max_delay);
		self.target_delay = target;
		self.smoothed_delay = (self.smoothed_delay
			+ (target - self.smoothed_delay) * glide_coefficient(self.sample_rate))
			.clamp(0.0, max_delay);

		let delay_samples = self.smoothed_delay;
		let t = delay_samples.fract();
		let whole = delay_samples.floor() as isize;
		let history_len = history_len as isize;

		// Read every tap before anything is written, because ping-pong feeds each side from
		// the other's output.
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

		let mut delayed_sample_output = [0.0; CHANNELS];
		for (i, history) in self.history.iter_mut().enumerate() {
			// Ping-pong crosses the feedback: each side repeats what the *other* side just
			// produced, so the repeats alternate instead of each side echoing into itself.
			let feedback = if self.pingpong {
				delayed[(i + 1) % CHANNELS]
			}else {
				delayed[i]
			};

			history.push(feedback * self.wet_gain * self.decay_factor + samples[i]);
			delayed_sample_output[i] = delayed[i] * self.decay_factor;
		}

		self.delayed_effect.process(&mut delayed_sample_output, &[], ctx);
		delayed_sample_output.into_iter().enumerate().for_each(|(i, mut inner)| {
			inner *= self.wet_gain;
			samples[i] += inner;
		});

		if ctx.should_stop() {
			self.clear_history();	
		}
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
    	use crate::tools::ui_tools::gain_ui;

		// Ping-pong no longer shortens the usable delay: each side runs the whole time and
		// feeds the other, instead of the read offset being stretched across the channels.
		let delay_range = 1.0..=1000.0_f32.min(self.history[0].capacity() as f32
			/ self.sample_rate as f32 * 1000.0);

		ui.checkbox(&mut self.pingpong, "Ping-Pong");
		crate::tools::ui_tools::synced_time_ui(ui, &id_prefix, &mut self.tempo, &mut self.delay_time, delay_range);
		ui.add(egui::Slider::new(&mut self.decay_factor, 0.01..=0.8).text("Decay Factor"));
		gain_ui(ui, &mut self.wet_gain, Some("Wet Gain".to_string()), false);

		self.delayed_effect.demo_ui(ui, format!("{}_delayed_effect", id_prefix));
	}
}

/// A Flanger effect that adds a flanging effect to the signal.
#[derive(Parameters)]
pub struct Flanger<Lfo: WaveTable, const CHANNELS: usize = 2> {
	#[skip]
	history: [RingBuffer<f32>; CHANNELS],
	#[skip]
	processed_history: [RingBuffer<f32>; CHANNELS],
	// pub decay_factor: f32,
	/// the center saves in milliseconds
	#[range(min = 0.0, max = 4000.0)]
	pub center_delay_time: f32,
	/// The sample rate of the audio signal, saves in Hz
	#[skip]
	pub sample_rate: usize,
	/// The LFO waveform
	#[sub_param]
	pub lfo: Lfo,
	/// The frequency of the LFO, saves in Hz
	#[range(min = 0.01, max = 20.0)]
	pub lfo_frequency: f32,
	/// The amplitude of the LFO, saves in milliseconds
	#[range(min = 0.01, max = 10.0)]
	pub lfo_amplitude: f32,
	#[skip]
	phase: f32,
	/// The wet gain of the effect, saves in linear scale
	#[range(min = 0.01, max = 4.0)]
	#[logarithmic]
	pub wet_gain: f32,
	#[range(min = 0.01, max = 1.0)]
	#[logarithmic]
	/// The feed back gain, saves in linear scale
	pub feed_back_gain: f32,
}

impl<Lfo: WaveTable, const CHANNELS: usize> Flanger<Lfo, CHANNELS> {
	/// Creates a new flanger effect with a given sample rate and LFO.
	/// 
	/// Panics if `CHANNELS` is 0.
	pub fn new(
		maxium_delay_length: usize,
		sample_rate: usize,
		lfo: Lfo,
	) -> Self {
		assert!(CHANNELS > 0, "CHANNELS must be greater than 0");

		let history = core::array::from_fn(|_| RingBuffer::new(maxium_delay_length));
		let processed_history = core::array::from_fn(|_| RingBuffer::new(maxium_delay_length));

		Self {
			history,
			processed_history,
			// decay_factor: 0.8,
			sample_rate,
			center_delay_time: 10.0,
			lfo,
			lfo_frequency: 1.0,
			lfo_amplitude: 5.0,
			phase: 0.0,
			wet_gain: 1.0,
			feed_back_gain: 0.1,
		}
	}

	/// Clear delay history.
	pub fn clear_history(&mut self) {
		for buffer in self.history.iter_mut() {
			buffer.clear();
		}

		for buffer in self.processed_history.iter_mut() {
			buffer.clear();
		}
	}

	/// Resize delay history.
	pub fn resize_history(&mut self, new_capacity: usize) {
		for buffer in self.history.iter_mut() {
			buffer.resize(new_capacity);
		}

		for buffer in self.processed_history.iter_mut() {
			buffer.resize(new_capacity);
		}
	}
}

impl<Lfo: WaveTable + Send + Sync, const CHANNELS: usize> Effect<CHANNELS> for Flanger<Lfo, CHANNELS> {
	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"Flanger"
	}

	fn delay(&self) -> usize {
		0
	}
	
	fn process(&mut self, samples: &mut [f32; CHANNELS], _: &[&[f32; CHANNELS]], ctx: &mut Box<dyn ProcessContext>) {
		let delay_time = self.center_delay_time / 1000.0 * self.sample_rate as f32;
		let lfo_amp = self.lfo_amplitude / 1000.0 * self.sample_rate as f32;
		let step = self.lfo_frequency / self.sample_rate as f32;
		let lfo_sample = self.lfo.sample(self.phase, 0) * lfo_amp;

		let sample_pos = lfo_sample + delay_time;
		let uniformed_pos = sample_pos / self.history[0].capacity() as f32;

		for (i, sample) in samples.iter_mut().enumerate() {
			let sampled = self.history[i].sample(uniformed_pos, 0);
			let feed_back = self.processed_history[i].sample(uniformed_pos, 0) * self.feed_back_gain;
			self.history[i].push(*sample);
			*sample += (sampled + feed_back) * self.wet_gain;
		}
		self.phase += step;
		self.phase %= 1.0;

		if ctx.should_stop() {
			self.clear_history();	
		}
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, _: String) {
    	use crate::tools::ui_tools::gain_ui;

		ui.add(egui::Slider::new(&mut self.center_delay_time, 0.01..=10.0).text("Center Delay Time (ms)"));
		// ui.add(egui::Slider::new(&mut self.decay_factor, 0.01..=0.8).text("Decay Factor"));
		ui.add(egui::Slider::new(&mut self.lfo_frequency, 0.01..=20.0).text("LFO Frequency (Hz)"));
		ui.add(egui::Slider::new(&mut self.lfo_amplitude, 0.01..=10.0).text("LFO Amplitude(ms)"));
		gain_ui(ui, &mut self.feed_back_gain, Some("Feed Back Gain".to_string()), true);
		gain_ui(ui, &mut self.wet_gain, Some("Wet Gain".to_string()), false);
	}
}

/// A Chorus effect that adds a chorus effect to the signal.
#[derive(Parameters)]
pub struct Chorus<Lfo: WaveTable, const CHANNELS: usize = 2> {
	#[skip]
	history: [RingBuffer<f32>; CHANNELS],
	/// The sample rate of the audio signal, saves in Hz
	#[skip]
	pub sample_rate: usize,
	/// The center delay time, saves in milliseconds
	#[range(min = 5.0, max = 20.0)]
	pub center_delay_time: f32,
	/// The LFO waveform
	#[sub_param]
	pub lfo: Lfo,
	/// The frequency of the LFO, saves in Hz
	#[range(min = 0.01, max = 2.0)]
	pub lfo_frequency: f32,
	/// The amplitude of the LFO, saves in milliseconds
	#[range(min = 5.0, max = 20.0)]
	pub lfo_amplitude: f32,
	/// The wet_gain of the effect, saves in linear scale
	#[range(min = 0.01, max = 4.0)]
	#[logarithmic]
	pub wet_gain: f32,
	#[range(min = 1, max = 50)]
	delay_lines: usize,
	#[skip]
	phase: f32,
}

impl<Lfo: WaveTable, const CHANNELS: usize> Chorus<Lfo, CHANNELS> {
	/// Creates a new chorus effect with a given sample rate and LFO.
	/// 
	/// Panics if `CHANNELS` is 0.
	pub fn new(
		maxium_delay_length: usize,
		delay_lines: usize,
		sample_rate: usize,
		lfo: Lfo,
	) -> Self {
		assert!(CHANNELS > 0, "CHANNELS must be greater than 0");

		let history = core::array::from_fn(|_| RingBuffer::new(maxium_delay_length));

		Self {
			history,
			sample_rate,
			center_delay_time: 10.0,
			lfo,
			lfo_frequency: 1.0,
			lfo_amplitude: 10.0,
			wet_gain: 1.0,
			phase: 0.0,
			delay_lines,
		}
	}

	/// Clear delay history.
	pub fn clear_history(&mut self) {
		for line in self.history.iter_mut() {
			line.clear();
		}
	}

	/// Resize delay history.
	pub fn resize_history(&mut self, new_capacity: usize) {
		for line in self.history.iter_mut() {
			line.resize(new_capacity);
		}
	}
}

impl<Lfo: WaveTable + Send + Sync, const CHANNELS: usize> Effect<CHANNELS> for Chorus<Lfo, CHANNELS> {
	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"Chorus"
	}

	fn delay(&self) -> usize {
		0
	}
	
	fn process(&mut self, samples: &mut [f32; CHANNELS], _: &[&[f32; CHANNELS]], ctx: &mut Box<dyn ProcessContext>) {
		let delay_time = self.center_delay_time / 1000.0 * self.sample_rate as f32;
		let lfo_amp = self.lfo_amplitude / 1000.0 * self.sample_rate as f32;
		let step = self.lfo_frequency / self.sample_rate as f32;

		let mut output = [0.0; CHANNELS];

		for i in 0..self.delay_lines {
			for (j, output) in output.iter_mut().enumerate() {
				let phase_delta = (i as f32 / (self.delay_lines + j) as f32).fract();
				let lfo_sample = self.lfo.sample((self.phase + phase_delta).fract(), 0) * lfo_amp;
				let sample_pos = lfo_sample + delay_time;
				let uniformed_pos = sample_pos / self.history[0].capacity() as f32;

				let sampled = self.history[j].sample(uniformed_pos, 0);
				*output += sampled * self.wet_gain;
			}
		}

		samples.iter_mut().enumerate().for_each(|(i, sample)| {
			self.history[i].push(*sample);
			*sample += output[i];
			*sample /= self.delay_lines as f32;
		});

		self.phase += step;
		self.phase %= 1.0;

		if ctx.should_stop() {
			self.clear_history();	
		}
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, _: String) {
    	use crate::tools::ui_tools::gain_ui;

		ui.add(egui::Slider::new(&mut self.center_delay_time, 5.0..=20.0).text("Center Delay Time (ms)"));
		ui.add(egui::Slider::new(&mut self.lfo_frequency, 0.01..=2.0).text("LFO Frequency (Hz)"));
		ui.add(egui::Slider::new(&mut self.lfo_amplitude, 5.0..=20.0).text("LFO Amplitude(ms)"));
		ui.add(egui::Slider::new(&mut self.delay_lines, 1..=50).text("Delay Lines"));
		gain_ui(ui, &mut self.wet_gain, Some("Wet Gain".to_string()), false);
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// Renders the impulse response of the delay.
	fn impulse_response(delay: &mut PureDelay<1>, samples: usize) -> Vec<f32> {
		let mut ctx: Box<dyn ProcessContext> = Box::new(());
		let mut output = Vec::with_capacity(samples);
		for i in 0..samples {
			let mut frame = [if i == 0 { 1.0 } else { 0.0 }];
			delay.process(&mut frame, &[], &mut ctx);
			output.push(frame[0]);
		}
		output
	}

	/// The energy of the impulse has to end up where the delay time asks for,
	/// including between two samples.
	#[test]
	fn impulse_lands_at_the_delay_time() {
		for delay_time in [0.0f32, 5.0, 10.0, 10.1, 100.0, 380.0] {
			let mut delay = PureDelay::<1>::new(48_000, delay_time, 48_000);
			let response = impulse_response(&mut delay, 24_000);

			let total: f32 = response.iter().map(|x| x.abs()).sum();
			let centroid: f32 = response
				.iter()
				.enumerate()
				.map(|(i, x)| i as f32 * x.abs())
				.sum::<f32>() / total;
			let expected = delay_time / 1000.0 * 48_000.0;

			assert!((total - 1.0).abs() < 0.2, "delay {delay_time} ms lost energy: {total}");
			assert!(
				(centroid - expected).abs() < 1.5,
				"delay {delay_time} ms landed at {centroid}, expected {expected}"
			);
		}
	}

	/// The reported latency has to be the delay the effect actually applies.
	#[test]
	fn latency_matches_the_delay() {
		let mut delay = PureDelay::<1>::new(48_000, 10.0, 48_000);
		assert_eq!(delay.delay(), 480);
		delay.delay_time = 10.5;
		assert_eq!(delay.delay(), 504);
	}

	/// Resizing and clearing must not panic, and a cleared line is silent.
	#[test]
	fn clear_and_resize() {
		let mut delay = PureDelay::<1>::new(48_000, 10.0, 48_000);
		let _ = impulse_response(&mut delay, 1_000);
		delay.clear_history();
		let response = impulse_response(&mut delay, 10);
		assert!(response[1..].iter().all(|x| *x == 0.0));
		delay.resize_history(128);
		assert_eq!(delay.history_len(), 128);
		delay.resize_history(0);
		let response = impulse_response(&mut delay, 10);
		assert!(response.iter().all(|x| x.is_finite()));
	}
	/// A process context that reports fixed infos, for the synced cases.
	struct FixedInfos(ProcessInfos);

	impl ProcessContext for FixedInfos {
		fn infos(&self) -> &ProcessInfos {
			&self.0
		}

		fn next_event(&mut self) -> Option<crate::NoteEvent> {
			None
		}

		fn send_event(&mut self, _: crate::NoteEvent) {}

		fn clear_events(&mut self) {}

		fn events(&self) -> &[crate::NoteEvent] {
			&[]
		}
	}

	fn infos_with_tempo(tempo: f32) -> ProcessInfos {
		ProcessInfos {
			sample_rate: 48_000,
			trustable: true,
			playing: true,
			tempo: Some(tempo),
			..ProcessInfos::default()
		}
	}

	/// The samples the echoes of an impulse land on, per channel.
	fn echo_samples(delay: &mut Delay<(), 2>, samples: usize) -> [Vec<usize>; 2] {
		let mut ctx: Box<dyn ProcessContext> = Box::new(());
		let mut echoes = [Vec::new(), Vec::new()];

		for i in 0..samples {
			let mut frame = [if i == 0 { 1.0 } else { 0.0 }, 0.0];
			delay.process(&mut frame, &[], &mut ctx);

			// The dry input passes through the effect, so the first samples carry the
			// impulse itself; only what comes later is an echo.
			if i < 100 {
				continue;
			}
			for (channel, echo) in echoes.iter_mut().enumerate() {
				if frame[channel].abs() > 1e-4 {
					echo.push(i);
				}
			}
		}

		echoes
	}

	/// 100 ms at 48 kHz is 4_800 samples, and that is where the first repeat belongs — the
	/// ping-pong read used to stretch it across the channels and land at twice the time.
	#[test]
	fn ping_pong_alternates_at_the_delay_time() {
		let mut delay = Delay::<(), 2>::new((), 1 << 16, 100.0, 48_000);
		delay.decay_factor = 0.5;
		delay.pingpong = true;

		let echoes = echo_samples(&mut delay, 48_000 / 2);

		assert_eq!(echoes[0].first(), Some(&4_800), "left repeats: {:?}", echoes[0]);
		assert_eq!(echoes[1].first(), Some(&9_600), "right repeats: {:?}", echoes[1]);
		assert_eq!(echoes[0].get(1), Some(&14_400), "left repeats: {:?}", echoes[0]);
	}

	/// Without ping-pong both sides repeat into themselves, at the same distance.
	#[test]
	fn a_plain_delay_repeats_on_the_same_side() {
		let mut delay = Delay::<(), 2>::new((), 1 << 16, 100.0, 48_000);
		delay.decay_factor = 0.5;

		let echoes = echo_samples(&mut delay, 48_000 / 2);

		assert_eq!(echoes[0].first(), Some(&4_800));
		assert_eq!(echoes[0].get(1), Some(&9_600));
		assert!(echoes[1].is_empty(), "the other side is silent: {:?}", echoes[1]);
	}

	/// Moving a delay time has to move the read head, and a read head that teleports is a
	/// click; the glide is what turns it into a slide.
	#[test]
	fn a_moved_delay_time_glides_instead_of_jumping() {
		let mut delay = PureDelay::<1>::new(1 << 16, 10.0, 48_000);
		let mut ctx: Box<dyn ProcessContext> = Box::new(());
		let mut frame = [0.0];

		// 10 ms is 480 samples, and a fresh delay starts there rather than gliding to it.
		delay.process(&mut frame, &[], &mut ctx);
		assert!((delay.smoothed_delay - 480.0).abs() < 1e-3);

		// Asked for 100 ms, the read head has to travel: one sample of a 20 ms glide moves it
		// by about 1/960 of the way, not 4_320 samples.
		delay.delay_time = 100.0;
		delay.process(&mut frame, &[], &mut ctx);
		assert!(
			delay.smoothed_delay > 480.0 && delay.smoothed_delay < 490.0,
			"the read head jumped to {}",
			delay.smoothed_delay
		);

		// It does get there.
		for _ in 0..48_000 / 5 {
			delay.process(&mut frame, &[], &mut ctx);
		}
		assert!((delay.smoothed_delay - 4_800.0).abs() < 1.0, "settled at {}", delay.smoothed_delay);
	}

	/// A synced delay takes its time from the tempo, and reports it to whoever asks for the
	/// latency even though only the process context knows the tempo.
	#[test]
	fn a_synced_delay_follows_the_tempo() {
		let mut delay = PureDelay::<1>::new(1 << 16, 10.0, 48_000);
		delay.tempo = SyncedTime::synced(0.5);

		let mut ctx: Box<dyn ProcessContext> = Box::new(FixedInfos(infos_with_tempo(120.0)));
		let mut frame = [0.0];
		for _ in 0..48_000 / 5 {
			delay.process(&mut frame, &[], &mut ctx);
		}

		// An eighth note at 120 bpm is 250 ms, which is 12_000 samples — not the 480 the free
		// value asks for.
		assert!((delay.smoothed_delay - 12_000.0).abs() < 1.0, "read {}", delay.smoothed_delay);
		assert_eq!(delay.delay(), 12_000);
	}
}