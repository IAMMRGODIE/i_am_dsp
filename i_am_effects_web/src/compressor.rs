//! The web compressor: a curated set of controls over the library's compressor.
//!
//! The library's Compressor exposes its envelope, smoother and gain-factor
//! internals; this wrapper keeps the ones a person turns, adds a lookahead in
//! milliseconds and a dry/wet pair, and lets the faceplate choose the envelope
//! detector from the header.

use i_am_dsp::prelude::{
	Compressor, Effect, Enveloper, IIRHilbertEnvelope, Parameter, Parameters, PeakDetector,
	ProcessContext, RectifyEnvelope, SetValue, TkeoEnvelope,
};
use i_am_dsp::tools::audio_io_chooser::AudioIoChooser;
use i_am_dsp::tools::ring_buffer::RingBuffer;
use i_am_dsp_derive::Parameters;

/// One of the library's level detectors, selected at runtime.
pub struct SelectableEnvelope {
	/// 0 peak, 1 RMS (rectified), 2 TKeo, 3 Hilbert.
	pub kind: i32,
	peak: PeakDetector<2>,
	rectify: RectifyEnvelope<2>,
	tkeo: TkeoEnvelope<2>,
	hilbert: IIRHilbertEnvelope<6, 2>,
}

impl SelectableEnvelope {
	fn new(sample_rate: usize) -> Self {
		Self {
			kind: 0,
			// The detector is only the level follower; the compressor's own
			// smoother owns the attack and release that a person turns.
			peak: PeakDetector::new(sample_rate, 0.1, 10.0),
			rectify: RectifyEnvelope::new(sample_rate, 30.0),
			tkeo: TkeoEnvelope::new(),
			hilbert: IIRHilbertEnvelope::new(sample_rate),
		}
	}
}

impl Enveloper<2> for SelectableEnvelope {
	fn delay(&self) -> usize {
		match self.kind {
			1 => Enveloper::delay(&self.rectify),
			2 => Enveloper::delay(&self.tkeo),
			3 => Enveloper::delay(&self.hilbert),
			_ => Enveloper::delay(&self.peak),
		}
	}

	fn input_value(&mut self, input: &mut [f32; 2]) {
		match self.kind {
			1 => self.rectify.input_value(input),
			2 => self.tkeo.input_value(input),
			3 => self.hilbert.input_value(input),
			_ => self.peak.input_value(input),
		}
	}

	fn get_current_envelope(&self) -> [f32; 2] {
		match self.kind {
			1 => self.rectify.get_current_envelope(),
			2 => self.tkeo.get_current_envelope(),
			3 => self.hilbert.get_current_envelope(),
			_ => self.peak.get_current_envelope(),
		}
	}

	fn set_infos(&mut self, infos: &i_am_dsp::ProcessInfos) {
		if self.kind == 0 {
			self.peak.set_infos(infos);
		}
	}
}

impl Parameters for SelectableEnvelope {
	fn get_parameters(&self) -> Vec<Parameter> {
		Vec::new()
	}

	fn set_parameter(&mut self, _: &str, _: SetValue) -> bool {
		false
	}
}

/// A compressor with the controls a person turns, plus a dry/wet pair and a
/// sidechain switch.
#[derive(Parameters)]
pub struct CompressorEffect {
	/// The level above which the compressor starts.
	#[range(min = -60.0, max = 0.0)]
	#[unit(decibels)]
	#[default(-24.0)]
	#[id(display = "Threshold")]
	#[layout(group = "Compressor", order = 0, widget = knob, hero = "compressor")]
	pub threshold: f32,

	/// How much the level above the threshold is reduced.
	#[range(min = 1.0, max = 20.0)]
	#[default(4.0)]
	#[id(display = "Ratio")]
	#[layout(group = "Compressor", order = 1, widget = knob)]
	pub ratio: f32,

	/// How quickly the gain falls when the signal gets louder.
	#[range(min = 0.1, max = 200.0)]
	#[unit(milliseconds)]
	#[default(10.0)]
	#[id(display = "Attack")]
	#[layout(group = "Compressor", order = 2, widget = knob)]
	pub attack: f32,

	/// How quickly the gain comes back.
	#[range(min = 10.0, max = 2000.0)]
	#[unit(milliseconds)]
	#[default(120.0)]
	#[id(display = "Release")]
	#[layout(group = "Compressor", order = 3, widget = knob)]
	pub release: f32,

	/// How far ahead the detector looks.
	#[range(min = 0.0, max = 50.0)]
	#[unit(milliseconds)]
	#[default(0.0)]
	#[id(display = "Lookahead")]
	#[layout(group = "Compressor", order = 4, widget = knob)]
	pub lookahead: f32,

	/// Gain added back after compression.
	#[range(min = -24.0, max = 24.0)]
	#[unit(decibels)]
	#[default(0.0)]
	#[id(display = "Makeup")]
	#[layout(group = "Compressor", order = 5, widget = knob)]
	pub makeup_db: f32,

	/// The dry gain of the effect.
	#[range(min = 0.0, max = 1.0)]
	#[unit(percent)]
	#[default(0.0)]
	#[id(display = "Dry")]
	#[layout(group = "Compressor", order = 6, widget = knob)]
	pub dry_gain: f32,

	/// The wet gain of the effect.
	#[range(min = 0.0, max = 1.0)]
	#[unit(percent)]
	#[default(1.0)]
	#[id(display = "Wet")]
	#[layout(group = "Compressor", order = 7, widget = knob)]
	pub wet_gain: f32,

	/// Which level detector the compressor uses.
	#[range(min = 0, max = 3)]
	#[choices("Peak", "RMS", "TKeo", "Hilbert")]
	#[default(0)]
	#[id(display = "Enveloper")]
	#[layout(group = "Compressor", order = 0, widget = choice)]
	pub enveloper: i32,

	/// Whether the detector follows the second input instead of the main one.
	#[default(false)]
	#[id(display = "Sidechain")]
	#[layout(group = "Compressor", order = 8, widget = toggle)]
	pub sidechain: bool,

	#[skip]
	inner: Compressor<SelectableEnvelope, 2>,

	/// The dry signal delayed by the lookahead, so the mix stays aligned.
	#[skip]
	dry_buffer: [RingBuffer<f32>; 2],
}

impl CompressorEffect {
	/// A compressor with a musical starting point: -24 dB, 4:1, 10 ms attack,
	/// 120 ms release, no lookahead, no makeup, 100 % wet.
	pub fn new(sample_rate: usize) -> Self {
		let inner = Compressor::new(
			SelectableEnvelope::new(sample_rate),
			sample_rate,
			10.0,
			120.0,
			-24.0,
			4.0,
		);

		Self {
			threshold: -24.0,
			ratio: 4.0,
			attack: 10.0,
			release: 120.0,
			lookahead: 0.0,
			makeup_db: 0.0,
			dry_gain: 0.0,
			wet_gain: 1.0,
			enveloper: 0,
			sidechain: false,
			inner,
			dry_buffer: core::array::from_fn(|_| RingBuffer::new(0)),
		}
	}
}

impl Effect<2> for CompressorEffect {
	fn delay(&self) -> usize {
		self.inner.delay()
	}

	fn process(
		&mut self,
		samples: &mut [f32; 2],
		other: &[&[f32; 2]],
		context: &mut Box<dyn ProcessContext>,
	) {
		let dry = *samples;
		let sample_rate = self.inner.sample_rate as f32;

		self.inner.threshold = self.threshold;
		self.inner.ratio = self.ratio;
		self.inner.smoother.attack_time = self.attack;
		self.inner.smoother.release_time = self.release;
		self.inner.lookahead = (self.lookahead * sample_rate / 1000.0).round().clamp(0.0, 4096.0)
			as usize;
		self.inner.gain_linear = 10.0_f32.powf(self.makeup_db / 20.0);
		self.inner.audio_io_chooser = if self.sidechain {
			// Port 0 is the main input; the declared sidechain port is 1.
			AudioIoChooser::Other(1)
		} else {
			AudioIoChooser::Current
		};
		self.inner.enveloper.kind = self.enveloper.clamp(0, 3);
		self.inner.enveloper.set_infos(context.infos());

		self.inner.process(samples, other, context);

		// The wet path is already delayed by the lookahead; delay the dry by the
		// same amount so a parallel mix does not comb.
		let lookahead = self.inner.lookahead;

		if self.dry_buffer[0].capacity() != lookahead {
			for buffer in self.dry_buffer.iter_mut() {
				buffer.resize(lookahead);
			}
		}

		for (index, sample) in samples.iter_mut().enumerate() {
			let delayed_dry = if lookahead > 0 {
				let value = self.dry_buffer[index][0];
				self.dry_buffer[index].push(dry[index]);
				value
			} else {
				dry[index]
			};

			*sample = delayed_dry * self.dry_gain + *sample * self.wet_gain;
		}
	}
}
#[cfg(test)]
mod tests {
	use super::*;
	use std::f32::consts::TAU;

	/// A loud main signal with a quiet sidechain: when the sidechain is selected,
	/// the detector should follow the quiet signal and leave the main alone.
	#[test]
	fn the_sidechain_keeps_the_main_signal_up() {
		fn render(sidechain: bool) -> f32 {
			let mut effect = CompressorEffect::new(48_000);
			effect.threshold = -30.0;
			effect.ratio = 4.0;
			effect.attack = 1.0;
			effect.release = 50.0;
			effect.dry_gain = 0.0;
			effect.wet_gain = 1.0;
			effect.sidechain = sidechain;

			let mut context: Box<dyn ProcessContext> = Box::new(());
			let mut energy = 0.0;

			for index in 0..4_096 {
				let phase = TAU * 1_000.0 * index as f32 / 48_000.0;
				let main = 0.5 * phase.sin();
				let side = 0.001 * phase.sin();
				let main_frame = [main, main];
				let side_frame = [side, side];
				let other: [&[f32; 2]; 2] = [&main_frame, &side_frame];
				let mut samples = main_frame;

				effect.process(&mut samples, &other, &mut context);

				if index >= 2_048 {
					energy += samples[0] * samples[0];
				}
			}

			(energy / 2_048.0).sqrt()
		}

		let compressed = render(false);
		let sidechained = render(true);

		assert!(
			sidechained > compressed * 2.0,
			"sidechained {sidechained} should stay above compressed {compressed}"
		);
	}

	/// Every envelope detector has to run without NaN.
	#[test]
	fn every_enveloper_is_finite() {
		for enveloper in 0..4 {
			let mut effect = CompressorEffect::new(48_000);
			effect.enveloper = enveloper;

			let mut context: Box<dyn ProcessContext> = Box::new(());

			for index in 0..512 {
				let value = (TAU * 440.0 * index as f32 / 48_000.0).sin() * 0.5;
				let mut samples = [value, value];
				effect.process(&mut samples, &[], &mut context);
				assert!(
					samples.iter().all(|sample| sample.is_finite()),
					"enveloper {enveloper}: {samples:?}"
				);
			}
		}
	}
}
