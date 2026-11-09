use super::read_delayed;
use i_am_dsp_derive::Parameters;

use crate::{
	Effect, ProcessContext,
	generators::wavetable::WaveTable,
	prelude::{SyncedPhase, SyncedTime},
	tools::ring_buffer::RingBuffer,
};

/// A Flanger effect that adds a flanging effect to the signal.
#[derive(Parameters)]
pub struct Flanger<Lfo: WaveTable, const CHANNELS: usize = 2> {
	#[skip]
	history: [RingBuffer<f32>; CHANNELS],
	#[skip]
	processed_history: [RingBuffer<f32>; CHANNELS],
	// pub decay_factor: f32,
	/// the center saves in milliseconds
	#[range(min = 0.1, max = 10.0)]
	#[unit(milliseconds)]
	#[id(display = "Delay")]
	#[layout(group = "Flanger", order = 1, widget = knob)]
	pub center_delay_time: f32,
	/// The sample rate of the audio signal, saves in Hz
	#[skip]
	pub sample_rate: usize,
	/// The LFO waveform
	#[sub_param]
	pub lfo: Lfo,
	/// The frequency of the LFO, saves in Hz, used when [`Self::tempo`] is not syncing.
	#[range(min = 0.05, max = 10.0)]
	#[unit(hertz)]
	#[id(display = "Rate")]
	#[layout(group = "Flanger", order = 0, widget = knob, hero = "lfo")]
	pub lfo_frequency: f32,
	/// The same rate in musical time.
	///
	/// With it syncing, the sweep takes its phase from the song position while the transport
	/// runs, so it lines up with the grid, and keeps sweeping at the synced rate while the
	/// transport is stopped.
	#[id(display = "Tempo")]
	#[layout(group = "Flanger", order = 0)]
	#[sub_param]
	pub tempo: SyncedTime,
	/// The amplitude of the LFO, saves in milliseconds
	#[range(min = 0.1, max = 5.0)]
	#[unit(milliseconds)]
	#[id(display = "Depth")]
	#[layout(group = "Flanger", order = 2, widget = knob)]
	pub lfo_amplitude: f32,
	#[skip]
	pub(super) phase: SyncedPhase,
	/// The dry gain of the effect.
	#[range(min = 0.0, max = 1.0)]
	#[unit(percent)]
	#[default(0.5)]
	#[id(display = "Dry")]
	#[layout(group = "Flanger", order = 4, widget = knob)]
	pub dry_gain: f32,
	/// The wet gain of the effect.
	#[range(min = 0.0, max = 1.0)]
	#[unit(percent)]
	#[default(0.5)]
	#[id(display = "Wet")]
	#[layout(group = "Flanger", order = 5, widget = knob)]
	pub wet_gain: f32,
	#[range(min = 0.0, max = 0.95)]
	#[unit(percent)]
	#[default(0.3)]
	#[id(display = "Feedback")]
	#[layout(group = "Flanger", order = 3, widget = knob)]
	/// The feed back gain, saves in linear scale
	pub feed_back_gain: f32,
}

impl<Lfo: WaveTable, const CHANNELS: usize> Flanger<Lfo, CHANNELS> {
	/// Creates a new flanger effect with a given sample rate and LFO.
	///
	/// Panics if `CHANNELS` is 0.
	pub fn new(maxium_delay_length: usize, sample_rate: usize, lfo: Lfo) -> Self {
		assert!(CHANNELS > 0, "CHANNELS must be greater than 0");

		let history = core::array::from_fn(|_| RingBuffer::new(maxium_delay_length));
		let processed_history = core::array::from_fn(|_| RingBuffer::new(maxium_delay_length));

		Self {
			history,
			processed_history,
			// decay_factor: 0.8,
			sample_rate,
			center_delay_time: 2.0,
			lfo,
			lfo_frequency: 0.5,
			tempo: SyncedTime::default(),
			lfo_amplitude: 2.0,
			phase: SyncedPhase::new(),
			dry_gain: 0.5,
			wet_gain: 0.5,
			feed_back_gain: 0.3,
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

impl<Lfo: WaveTable + Send + Sync, const CHANNELS: usize> Effect<CHANNELS>
	for Flanger<Lfo, CHANNELS>
{
	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"Flanger"
	}

	fn delay(&self) -> usize {
		0
	}

	fn process(
		&mut self,
		samples: &mut [f32; CHANNELS],
		_: &[&[f32; CHANNELS]],
		ctx: &mut Box<dyn ProcessContext>,
	) {
		let delay_time = self.center_delay_time / 1000.0 * self.sample_rate as f32;
		let lfo_amp = self.lfo_amplitude / 1000.0 * self.sample_rate as f32;
		let phase = self.phase.advance_rate(
			&self.tempo,
			ctx.infos(),
			self.lfo_frequency,
			self.sample_rate,
		) as f32;
		let lfo_sample = self.lfo.sample(phase, 0) * lfo_amp;

		let sample_pos = lfo_sample + delay_time;

		let dry = *samples;

		for (i, sample) in samples.iter_mut().enumerate() {
			let sampled = read_delayed(&self.history[i], sample_pos);
			let feed_back =
				read_delayed(&self.processed_history[i], sample_pos) * self.feed_back_gain;
			let wet = sampled + feed_back;

			self.history[i].push(dry[i]);
			// The processed line is what the feedback reads; it used to stay empty,
			// so the feedback knob did nothing at all.
			self.processed_history[i].push(wet);
			*sample = dry[i] * self.dry_gain + wet * self.wet_gain;
		}

		if ctx.should_stop() {
			self.clear_history();
		}
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, _: String) {
		use crate::tools::ui_tools::gain_ui;

		ui.add(
			egui::Slider::new(&mut self.center_delay_time, 0.01..=10.0)
				.text("Center Delay Time (ms)"),
		);
		// ui.add(egui::Slider::new(&mut self.decay_factor, 0.01..=0.8).text("Decay Factor"));
		crate::tools::ui_tools::synced_time_ui(
			ui,
			"flanger",
			&mut self.tempo,
			&mut self.lfo_frequency,
			0.01..=20.0,
			"LFO Frequency (Hz)",
		);
		ui.add(egui::Slider::new(&mut self.lfo_amplitude, 0.01..=10.0).text("LFO Amplitude(ms)"));
		gain_ui(
			ui,
			&mut self.feed_back_gain,
			Some("Feed Back Gain".to_string()),
			true,
		);
		gain_ui(ui, &mut self.wet_gain, Some("Wet Gain".to_string()), false);
	}
}
