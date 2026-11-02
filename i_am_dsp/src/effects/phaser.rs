//! A phaser effect with variable feedback gain and LFO frequency.

use std::f32::consts::PI;

use i_am_dsp_derive::Parameters;

use crate::{
	Effect, ProcessContext,
	prelude::{MIN_FREQUENCY, SyncedPhase, SyncedTime, WaveTable},
};

/// A phaser effect with variable feedback gain and LFO frequency.
#[derive(Parameters)]
pub struct Phaser<Lfo: WaveTable, const CHANNELS: usize = 2> {
	#[skip]
	history_in: Vec<[f32; CHANNELS]>,
	#[skip]
	history_out: Vec<[f32; CHANNELS]>,
	#[skip]
	feedback_history: [f32; CHANNELS],
	/// the LFO waveform
	#[sub_param]
	#[layout(group = "Phaser", order = 0)]
	pub lfo: Lfo,
	/// LFO frequency in Hz, used when [`Self::tempo`] is not syncing.
	#[range(min = 0.05, max = 10.0)]
	#[unit(hertz)]
	#[id(display = "Rate")]
	#[layout(group = "Phaser", order = 0, widget = knob, hero = "lfo")]
	pub lfo_freq: f32,
	/// The same rate in musical time.
	///
	/// Syncing also takes the sweep's phase from the song position while the transport runs,
	/// so it lines up with the grid, and keeps it sweeping at the synced rate while the
	/// transport is stopped.
	#[sub_param]
	#[id(display = "Tempo")]
	#[layout(group = "Phaser", order = 0)]
	pub tempo: SyncedTime,
	#[range(min = 20.0, max = 2000.0)]
	#[logarithmic]
	#[default(200.0)]
	/// Minimum frequency in Hz
	#[unit(hertz)]
	#[id(display = "Min")]
	#[layout(group = "Phaser", order = 1, widget = knob)]
	pub min_freq: f32,
	#[range(min = 200.0, max = 20000.0)]
	#[logarithmic]
	#[default(2000.0)]
	/// Maximum frequency in Hz
	#[unit(hertz)]
	#[id(display = "Max")]
	#[layout(group = "Phaser", order = 2, widget = knob)]
	pub max_freq: f32,
	#[skip]
	sample_rate: usize,
	#[skip]
	phase: SyncedPhase,
	/// The feedback gain, saves in linear scale
	#[range(min = 0.0, max = 0.95)]
	#[unit(percent)]
	#[default(0.2)]
	#[id(display = "Feedback")]
	#[layout(group = "Phaser", order = 3, widget = knob)]
	pub feedback_gain: f32,
	/// Add phase delta to each allpass
	#[range(min = 0.0, max = 1.0)]
	#[unit(percent)]
	#[default(0.1)]
	#[id(display = "Spread")]
	#[layout(group = "Phaser", order = 4, widget = knob)]
	pub phase_delta: f32,
	/// The dry gain of the effect.
	#[range(min = 0.0, max = 1.0)]
	#[unit(percent)]
	#[default(0.5)]
	#[id(display = "Dry")]
	#[layout(group = "Phaser", order = 5, widget = knob)]
	pub dry_gain: f32,
	/// The wet gain of the effect.
	#[range(min = 0.0, max = 1.0)]
	#[unit(percent)]
	#[default(0.5)]
	#[id(display = "Wet")]
	#[layout(group = "Phaser", order = 6, widget = knob)]
	pub wet_gain: f32,
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
			tempo: SyncedTime::default(),
			min_freq: 200.0,
			max_freq: 2000.0,
			sample_rate,
			phase: SyncedPhase::new(),
			feedback_gain: 0.2,
			phase_delta: 0.1,
			dry_gain: 0.5,
			wet_gain: 0.5,
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

impl<Lfo: WaveTable + Send + Sync, const CHANNELS: usize> Effect<CHANNELS>
	for Phaser<Lfo, CHANNELS>
{
	fn delay(&self) -> usize {
		0
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"Phaser"
	}

	fn process(
		&mut self,
		samples: &mut [f32; CHANNELS],
		_: &[&[f32; CHANNELS]],
		ctx: &mut Box<dyn ProcessContext>,
	) {
		let dry = *samples;
		let base_phase =
			self.phase
				.advance_rate(&self.tempo, ctx.infos(), self.lfo_freq, self.sample_rate) as f32;
		samples
			.iter_mut()
			.zip(self.feedback_history.iter_mut())
			.for_each(|(sample, feedback)| {
				*sample += *feedback * self.feedback_gain;
			});

		let nyquist = self.sample_rate as f32 / 2.0 - MIN_FREQUENCY;
		let low = self.min_freq.clamp(MIN_FREQUENCY, nyquist);
		let high = self.max_freq.clamp(low, nyquist);
		let max_depth = high / self.sample_rate as f32;
		let min_depth = low / self.sample_rate as f32;

		let mut output = *samples;

		for (i, (history_in, history_out)) in self
			.history_in
			.iter_mut()
			.zip(self.history_out.iter_mut())
			.enumerate()
		{
			for (channel, output) in output.iter_mut().enumerate() {
				let phase = (base_phase + (i + channel) as f32 * self.phase_delta) % 1.0;
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
			// An allpass alone changes phase but not level; the notches come from
			// summing it with the dry signal, which this used to skip entirely.
			*sample = (dry[i] * self.dry_gain + output[i] * self.wet_gain).clamp(-1.0, 1.0);
		});
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

		crate::tools::ui_tools::synced_time_ui(
			ui,
			"phaser",
			&mut self.tempo,
			&mut self.lfo_freq,
			0.01..=20.0,
			"LFO Frequency (Hz)",
		);
		ui.add(
			egui::Slider::new(&mut self.min_freq, MIN_FREQUENCY..=max_freq)
				.text("Min Frequency")
				.logarithmic(true),
		);
		ui.add(
			egui::Slider::new(&mut self.max_freq, MIN_FREQUENCY..=max_freq)
				.text("Max Frequency")
				.logarithmic(true),
		);
		ui.add(egui::Slider::new(&mut self.phase_delta, 0.0..=1.0).text("Phase Delta"));
		gain_ui(
			ui,
			&mut self.feedback_gain,
			Some("Feedback Gain".to_string()),
			true,
		);
		// ui.add(egui::Slider::new(&mut self.wet_gain, 0.0..=1.0).text("Wet Gain"));
	}
}
#[cfg(test)]
mod tests {
	use super::*;
	use crate::generators::wavetable::SineWave;

	/// A fully dry phaser must pass the input unchanged; an allpass alone changes
	/// phase, and the notches only appear once it is summed with the dry path.
	#[test]
	fn a_fully_dry_phaser_is_the_input() {
		let mut phaser = Phaser::<SineWave, 2>::new(SineWave, 6, 48_000);
		phaser.dry_gain = 1.0;
		phaser.wet_gain = 0.0;

		let mut ctx: Box<dyn ProcessContext> = Box::new(());

		for i in 0..512 {
			let input = (core::f32::consts::TAU * 440.0 * i as f32 / 48_000.0).sin() * 0.5;
			let mut frame = [input, input];
			phaser.process(&mut frame, &[], &mut ctx);
			assert!(
				(frame[0] - input).abs() < 1e-6,
				"dry sample {i}: {}",
				frame[0]
			);
		}
	}

	/// A sweep whose maximum is below its minimum is clamped instead of running
	/// backwards.
	#[test]
	fn a_reversed_sweep_is_clamped() {
		let mut phaser = Phaser::<SineWave, 2>::new(SineWave, 6, 48_000);
		phaser.min_freq = 2_000.0;
		phaser.max_freq = 200.0;

		let mut ctx: Box<dyn ProcessContext> = Box::new(());
		let mut frame = [0.5, 0.5];

		for _ in 0..128 {
			phaser.process(&mut frame, &[], &mut ctx);
			assert!(frame.iter().all(|sample| sample.is_finite()));
		}
	}
}
