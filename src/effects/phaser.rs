//! A phaser effect with variable feedback gain and LFO frequency.

use std::f32::consts::PI;

use crate::{prelude::{WaveTable, MIN_FREQUENCY}, Effect, ProcessContext};

/// A phaser effect with variable feedback gain and LFO frequency.
pub struct Phaser<Lfo: WaveTable, const CHANNELS: usize = 2> {
	history_in: Vec<[f32; CHANNELS]>,
	history_out: Vec<[f32; CHANNELS]>,
	feedback_history: [f32; CHANNELS],
	/// the LFO waveform
	pub lfo: Lfo,
	/// LFO frequency in Hz
	pub lfo_freq: f32,
	/// Minimum frequency in Hz
	pub min_freq: f32,
	/// Maximum frequency in Hz
	pub max_freq: f32,
	sample_rate: usize,
	phase: f32,
	/// The feedback gain, saves in linear scale
	pub feedback_gain: f32,
	/// Add phase delta to each allpass
	pub phase_delta: f32,
}

impl<Lfo: WaveTable, const CHANNELS: usize> Phaser<Lfo, CHANNELS> {
	/// Create a new phaser effect with the given LFO waveform and allpass count.
	/// 
	/// Panics if `CHANNELS` is less than or equal to 0.
	pub fn new(lfo: Lfo, allpass_count: usize, sample_rate: usize) -> Self {
		assert!(CHANNELS > 0, "CHANNELS must be greater than 0");

		Self {
			history_in: vec![[0.0; CHANNELS]; allpass_count],
			history_out: vec![[0.0; CHANNELS]; allpass_count],
			feedback_history: [0.0; CHANNELS],
			lfo,
			lfo_freq: 0.5,
			min_freq: MIN_FREQUENCY,
			max_freq: sample_rate as f32 / 2.0 - MIN_FREQUENCY,
			sample_rate,
			phase: 0.0,
			feedback_gain: 0.0,
			phase_delta: 0.0,
			// wet_gain: 1.0,
		}
	}

	/// Set the number of allpasses in the phaser.
	pub fn set_allpass_count(&mut self, allpass_count: usize) {
		self.history_in.resize(allpass_count, [0.0; CHANNELS]);
		self.history_out.resize(allpass_count, [0.0; CHANNELS]);
	}

	/// Get the number of allpasses in the phaser.
	pub fn get_allpass_count(&self) -> usize {
		self.history_in.len()
	}
}

impl<Lfo: WaveTable + Send + Sync, const CHANNELS: usize> Effect<CHANNELS> for Phaser<Lfo, CHANNELS> {
	fn delay(&self) -> usize {
		0
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"Phaser"
	}

	fn process(&mut self, samples: &mut [f32; CHANNELS], _: &[&[f32; CHANNELS]], _: &mut Box<dyn ProcessContext>) {
		let step = self.lfo_freq / self.sample_rate as f32;
		samples.iter_mut().zip(self.feedback_history.iter_mut()).for_each(|(sample, feedback)| {
			*sample += *feedback * self.feedback_gain;
		});

		let max_depth = self.max_freq.min(self.sample_rate as f32 / 2.0 - MIN_FREQUENCY) / self.sample_rate as f32;
		let min_depth = self.min_freq.max(MIN_FREQUENCY) / self.sample_rate as f32;

		let mut output = *samples;

		for (i, (history_in, history_out)) in self.history_in
			.iter_mut()
			.zip(self.history_out.iter_mut())
			.enumerate() 
		{
			for (channel, output) in output.iter_mut().enumerate() {
				let phase = (self.phase + (i + channel) as f32 * self.phase_delta) % 1.0;
				let lfo = ((self.lfo.sample(phase, 0) + 1.0) / 2.0).clamp(0.0, 1.0);
				let freq = (max_depth - min_depth) * lfo + min_depth;
				let a_val = (PI * freq).tan();
				let a = (1.0 - a_val) / (1.0 + a_val);
				let input = *output;
				*output = a * *output + history_in[channel] - a * history_out[channel];
				history_in[channel] = input;
				history_out[channel] = *output;
			}
		}

		self.feedback_history = output;
		// println!("{:?}", output);

		samples.iter_mut().enumerate().for_each(|(i, sample)| {
			*sample = output[i].clamp(-1.0, 1.0);
		});

		self.phase += step;
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, _: String) {
		use crate::tools::ui_tools::gain_ui;

		let mut allpass_count = self.history_in.len();
		ui.add(egui::Slider::new(&mut allpass_count, 1..=100).text("Allpass Count"));
		if allpass_count != self.history_in.len() {
			self.set_allpass_count(allpass_count);
		}

		let max_freq = self.sample_rate as f32 / 2.0 - MIN_FREQUENCY;

		ui.add(egui::Slider::new(&mut self.lfo_freq, 0.01..=20.0).text("LFO Frequency"));
		ui.add(egui::Slider::new(&mut self.min_freq, MIN_FREQUENCY..=max_freq).text("Min Frequency").logarithmic(true));
		ui.add(egui::Slider::new(&mut self.max_freq, MIN_FREQUENCY..=max_freq).text("Max Frequency").logarithmic(true));
		ui.add(egui::Slider::new(&mut self.phase_delta, 0.0..=1.0).text("Phase Delta"));
		gain_ui(ui, &mut self.feedback_gain, Some("Feedback Gain".to_string()), true);
		// ui.add(egui::Slider::new(&mut self.wet_gain, 0.0..=1.0).text("Wet Gain"));
	}
}