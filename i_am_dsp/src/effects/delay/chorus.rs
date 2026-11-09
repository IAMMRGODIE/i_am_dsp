use super::read_delayed;
use i_am_dsp_derive::Parameters;

use crate::{
	Effect, ProcessContext,
	generators::wavetable::WaveTable,
	prelude::{SyncedPhase, SyncedTime},
	tools::ring_buffer::RingBuffer,
};

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
	#[unit(milliseconds)]
	#[id(display = "Delay")]
	#[layout(group = "Chorus", order = 1, widget = knob)]
	pub center_delay_time: f32,
	/// The LFO waveform
	#[layout(group = "Chorus", order = 0)]
	#[sub_param]
	pub lfo: Lfo,
	/// The frequency of the LFO, saves in Hz, used when [`Self::tempo`] is not syncing.
	#[range(min = 0.01, max = 2.0)]
	#[unit(hertz)]
	#[id(display = "Rate")]
	#[layout(group = "Chorus", order = 0, widget = knob, hero = "lfo")]
	pub lfo_frequency: f32,
	/// The same rate in musical time; see [`Flanger::tempo`] for what syncing does to a
	/// sweep.
	#[id(display = "Tempo")]
	#[layout(group = "Chorus", order = 0)]
	#[sub_param]
	pub tempo: SyncedTime,
	/// The amplitude of the LFO, saves in milliseconds
	#[range(min = 0.0, max = 10.0)]
	#[unit(milliseconds)]
	#[id(display = "Depth")]
	#[layout(group = "Chorus", order = 2, widget = knob)]
	pub lfo_amplitude: f32,
	/// The dry gain of the effect.
	#[range(min = 0.0, max = 1.0)]
	#[unit(percent)]
	#[default(0.5)]
	#[id(display = "Dry")]
	#[layout(group = "Chorus", order = 3, widget = knob)]
	pub dry_gain: f32,
	/// The wet gain of the effect.
	#[range(min = 0.0, max = 1.0)]
	#[unit(percent)]
	#[default(0.5)]
	#[id(display = "Wet")]
	#[layout(group = "Chorus", order = 4, widget = knob)]
	pub wet_gain: f32,
	#[skip]
	delay_lines: usize,
	#[skip]
	phase: SyncedPhase,
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
			tempo: SyncedTime::default(),
			lfo_amplitude: 5.0,
			dry_gain: 0.5,
			wet_gain: 0.5,
			phase: SyncedPhase::new(),
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

impl<Lfo: WaveTable + Send + Sync, const CHANNELS: usize> Effect<CHANNELS>
	for Chorus<Lfo, CHANNELS>
{
	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"Chorus"
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

		let dry = *samples;
		let mut wet = [0.0; CHANNELS];

		for i in 0..self.delay_lines {
			for (j, wet) in wet.iter_mut().enumerate() {
				let phase_delta = (i as f32 / (self.delay_lines + j) as f32).fract();
				let lfo_sample = self.lfo.sample((phase + phase_delta).fract(), 0) * lfo_amp;
				let sample_pos = lfo_sample + delay_time;

				*wet += read_delayed(&self.history[j], sample_pos);
			}
		}

		samples.iter_mut().enumerate().for_each(|(i, sample)| {
			self.history[i].push(dry[i]);
			// Average the taps, then apply the two gains independently. The dry used
			// to be divided by the number of lines as well, which made the bypass
			// path quieter the more lines the chorus used.
			*sample = dry[i] * self.dry_gain + wet[i] / self.delay_lines as f32 * self.wet_gain;
		});

		if ctx.should_stop() {
			self.clear_history();
		}
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, _: String) {
		use crate::tools::ui_tools::gain_ui;

		ui.add(
			egui::Slider::new(&mut self.center_delay_time, 5.0..=20.0)
				.text("Center Delay Time (ms)"),
		);
		crate::tools::ui_tools::synced_time_ui(
			ui,
			"chorus",
			&mut self.tempo,
			&mut self.lfo_frequency,
			0.01..=2.0,
			"LFO Frequency (Hz)",
		);
		ui.add(egui::Slider::new(&mut self.lfo_amplitude, 5.0..=20.0).text("LFO Amplitude(ms)"));
		ui.add(egui::Slider::new(&mut self.delay_lines, 1..=50).text("Delay Lines"));
		gain_ui(ui, &mut self.wet_gain, Some("Wet Gain".to_string()), false);
	}
}
