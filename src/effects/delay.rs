//! Delay related effects.

use crate::{generators::wavetable::WaveTable, tools::{interpolate::cubic_interpolate, ring_buffer::RingBuffer}, Effect, ProcessContext};

/// A pure delay that delays the signal by a fixed amount of time.
pub struct PureDelay<const CHANNELS: usize = 2> {
	history: [RingBuffer<f32>; CHANNELS],
	/// The delay time,  saves in milliseconds
	pub delay_time: f32,
	/// The sample rate of the audio signal, saves in Hz
	pub sample_rate: usize,
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
		let history = core::array::from_fn(|_| RingBuffer::new(maxium_delay_length));

		Self {
			history,
			delay_time,
			sample_rate,
		}
	}

	/// Clear delay history.
	pub fn clear_history(&mut self) {
		for buffer in self.history.iter_mut() {
			buffer.clear();
		}
	}

	/// Resize delay history.
	pub fn resize_history(&mut self, new_capacity: usize) {
		for buffer in self.history.iter_mut() {
			buffer.resize(new_capacity);
		}
	}

	/// Returns the maximum delay time that can be set without overflowing the buffer.
	pub fn maxium_delay_time(&self) -> f32 {
		self.history[0].capacity() as f32 / self.sample_rate as f32 * 1000.0
	}
}

impl<const CHANNELS: usize> Effect<CHANNELS> for PureDelay<CHANNELS> {
	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"PureDelay"
	}

	fn delay(&self) -> usize {
		0
	}
	
	fn process(&mut self, samples: &mut [f32; CHANNELS], _: &[&[f32; CHANNELS]], ctx: &mut Box<dyn ProcessContext>) {
		let delay_samples = self.delay_time / 1000.0 * self.sample_rate as f32;
		let t = delay_samples.fract();
		let delay_samples = delay_samples.floor() as isize;
		let history_len = self.history[0].capacity() as isize;

		for (i, sample) in samples.iter_mut().enumerate() {
			let interpolate_parameters = [
				self.history[i][history_len - 2 - delay_samples],
				self.history[i][history_len - 1 - delay_samples],
				self.history[i][history_len - delay_samples],
				self.history[i][history_len + 1 - delay_samples],
			];
			self.history[i].push(*sample);
			*sample = cubic_interpolate(t, interpolate_parameters);
		}

		if ctx.should_stop() {
			self.clear_history();	
		}
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, _: String) {
		let delay_range = 1.0..=self.maxium_delay_time();

		ui.add(egui::Slider::new(&mut self.delay_time, delay_range).text("Delay Time (ms)"));
	}
}

/// A delay effect that delays the signal by a variable amount of time.
pub struct Delay<
	DelayedEffect: Effect<CHANNELS>, 
	const CHANNELS: usize = 2
> {
	history: [RingBuffer<f32>; CHANNELS],
	delayed_effect: DelayedEffect,
	/// the delay time, saves in milliseconds
	pub delay_time: f32,
	/// The sample rate of the audio signal, saves in Hz
	pub sample_rate: usize,
	/// a factor that controls the decay rate of the delay effect
	/// 
	/// must be between 0 and 1
	pub decay_factor: f32,
	/// The wet gain of the effect, saves in linear scale
	pub wet_gain: f32,
	pingpong: Option<[PureDelay<1>; CHANNELS]>,
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

		Self {
			history,
			delayed_effect,
			delay_time,
			sample_rate,
			wet_gain: 1.0,
			decay_factor: 0.8,
			pingpong: None,
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
		max_time / zero_time * 1000.0
	}

	/// Clear delay history.
	pub fn clear_history(&mut self) {
		for buffer in self.history.iter_mut() {
			buffer.clear();
		}

		if let Some(pingpong) = self.pingpong.as_mut() {
			for delay in pingpong.iter_mut() {
				delay.clear_history();
			}
		}
	}

	/// Resize delay history.
	pub fn resize_history(&mut self, new_capacity: usize) {
		for buffer in self.history.iter_mut() {
			buffer.resize(new_capacity);
		}

		if let Some(pingpong) = self.pingpong.as_mut() {
			for delay in pingpong.iter_mut() {
				delay.resize_history(new_capacity);
			}
		}
	}

	/// Toggles the ping-pong mode.
	pub fn toggle_pingpong(&mut self) {
		if self.pingpong.is_none() {
			let history_len = self.history[0].capacity();
			// let max_time = self.max_delay_time(0.01);
			// self.delay_time = self.delay_time.clamp(0.0, max_time / CHANNELS as f32);
			self.pingpong = Some(core::array::from_fn(|i| {
				PureDelay::new(
					history_len, 
					i as f32 * self.delay_time,
					self.sample_rate
				)
			}));
		}else {
			self.pingpong = None;
		}
	}

	/// Returns true if the effect is in ping-pong mode.
	pub fn pingpong(&self) -> bool {
		self.pingpong.is_some()
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
		0
	}
	
	fn process(&mut self, samples: &mut [f32; CHANNELS], _: &[&[f32; CHANNELS]], ctx: &mut Box<dyn ProcessContext>) {
		let delay_samples = self.delay_time / 1000.0 * self.sample_rate as f32;
		let t = delay_samples.fract();
		let delay_samples_ori = delay_samples.floor() as isize;
		let history_len = self.history[0].capacity() as isize;

		let delay_samples = if self.pingpong() {
			delay_samples_ori * CHANNELS as isize
		}else {
			delay_samples_ori
		};

		let delay_samples = delay_samples.clamp(0, history_len - 1);
		let mut delayed_sample_output = [0.0; CHANNELS];
		for (i, history) in self.history.iter_mut().enumerate() {
			let interpolate_parameters = [
				history[history_len - 2 - delay_samples],
				history[history_len - 1 - delay_samples],
				history[history_len - delay_samples],
				history[history_len + 1 - delay_samples],
			];
			// self.history[i].push(*sample);
			let mut delayed_sample = [cubic_interpolate(t, interpolate_parameters)];
			let sample = delayed_sample[0] * self.wet_gain * self.decay_factor + samples[i];
			history.push(sample);

			delayed_sample_output[i] = if let Some(pingpong) = &mut self.pingpong {
				pingpong[i].process(&mut delayed_sample, &[], ctx);
				delayed_sample[0]
			}else {
				delayed_sample[0]
			} * self.decay_factor;
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

		let max_time = self.max_delay_time(0.01);

		let max_time = if self.pingpong() {
			max_time / CHANNELS as f32
		}else {
			max_time
		};

		let delay_range = 1.0..=max_time;

		if ui.selectable_label(self.pingpong(), "Ping-Pong").clicked() {
			self.toggle_pingpong();
		}
		ui.add(egui::Slider::new(&mut self.delay_time, delay_range).text("Delay Time (ms)"));
		ui.add(egui::Slider::new(&mut self.decay_factor, 0.01..=0.8).text("Decay Factor"));
		gain_ui(ui, &mut self.wet_gain, Some("Wet Gain".to_string()), false);

		self.delayed_effect.demo_ui(ui, format!("{}_delayed_effect", id_prefix));
	}
}

/// A Flanger effect that adds a flanging effect to the signal.
pub struct Flanger<Lfo: WaveTable, const CHANNELS: usize = 2> {
	history: [RingBuffer<f32>; CHANNELS],
	processed_history: [RingBuffer<f32>; CHANNELS],
	// pub decay_factor: f32,
	/// the center saves in milliseconds
	pub center_delay_time: f32,
	/// The sample rate of the audio signal, saves in Hz
	pub sample_rate: usize,
	/// The LFO waveform
	pub lfo: Lfo,
	/// The frequency of the LFO, saves in Hz
	pub lfo_frequency: f32,
	/// The amplitude of the LFO, saves in milliseconds
	pub lfo_amplitude: f32,
	phase: f32,
	/// The wet gain of the effect, saves in linear scale
	pub wet_gain: f32,
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
pub struct Chorus<Lfo: WaveTable, const CHANNELS: usize = 2> {
	history: [RingBuffer<f32>; CHANNELS],
	/// The sample rate of the audio signal, saves in Hz
	pub sample_rate: usize,
	/// The center delay time, saves in milliseconds
	pub center_delay_time: f32,
	/// The LFO waveform
	pub lfo: Lfo,
	/// The frequency of the LFO, saves in Hz
	pub lfo_frequency: f32,
	/// The amplitude of the LFO, saves in milliseconds
	pub lfo_amplitude: f32,
	/// The wet_gain of the effect, saves in linear scale
	pub wet_gain: f32,
	delay_lines: usize,
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