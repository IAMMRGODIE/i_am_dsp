//! The Pitch product with three engines: the WSOLA time-domain shifter, the
//! phase-vocoder pitch shifter and the pitch-synchronous PSOLA mode.

use i_am_dsp::prelude::{
	Effect, PhaseVocoder, PitchShift, PitchShifter, ProcessContext, Psola, overlap_add_gain,
};
use i_am_dsp_derive::Parameters;

pub const PITCH_VISIBLE: &[(i32, &[&str])] = &[
	(0, &["pitch_shift_factor"]),
	(1, &["pitch_shift_factor", "window_size"]),
	(2, &["pitch_shift_factor"]),
];

#[derive(Parameters)]
pub struct PitchEffect {
	/// The engine.
	#[choices("WSOLA", "Phase Vocoder", "PSOLA")]
	#[default(0)]
	#[id(display = "Algorithm")]
	#[layout(group = "Pitch", order = 0, widget = choice)]
	pub algorithm: i32,

	/// The pitch shift ratio.
	#[range(min = 0.25, max = 4.0)]
	#[logarithmic]
	#[default(1.0)]
	#[unit(percent)]
	#[id(display = "Shift")]
	#[layout(group = "Pitch", order = 1, widget = knob, size = large, hero = "pitch")]
	pub pitch_shift_factor: f32,

	/// The phase-vocoder window size, picked from the header.
	#[choices("512", "1024", "2048", "4096")]
	#[default(2)]
	#[id(display = "Window Size")]
	#[layout(group = "Pitch", order = 2, widget = choice)]
	pub window_size: i32,

	#[skip]
	time: PitchShifter<2>,
	#[skip]
	vocoder: PhaseVocoder<PitchShift, 2>,
	#[skip]
	psola: Psola<2>,
	#[skip]
	last_window_size: i32,
}

impl PitchEffect {
	pub fn new(sample_rate: usize) -> Self {
		Self {
			algorithm: 0,
			pitch_shift_factor: 1.0,
			window_size: 2,
			time: PitchShifter::<2>::new(1_024),
			vocoder: PhaseVocoder::new(PitchShift::default(), 2_048, sample_rate),
			psola: Psola::<2>::new(sample_rate),
			last_window_size: 2,
		}
	}
}

impl Effect<2> for PitchEffect {
	fn delay(&self) -> usize {
		match self.algorithm {
			0 => self.time.delay(),
			2 => self.psola.delay(),
			_ => self.vocoder.window_size(),
		}
	}

	fn process(
		&mut self,
		samples: &mut [f32; 2],
		other: &[&[f32; 2]],
		context: &mut Box<dyn ProcessContext>,
	) {
		if self.algorithm == 0 {
			self.time.pitch_shift_factor = self.pitch_shift_factor;
			self.time.process(samples, other, context);
			return;
		}

		if self.algorithm == 2 {
			self.psola.set_shift(self.pitch_shift_factor);
			self.psola.process(samples, other, context);
			return;
		}

		if self.window_size != self.last_window_size {
			let size = [512, 1024, 2048, 4096][self.window_size.clamp(0, 3) as usize];
			if self.vocoder.renew_window_size(size).is_some() {
				self.vocoder.gain = 1.0 / overlap_add_gain(self.vocoder.window_factor);
			}
			self.last_window_size = self.window_size;
		}

		self.vocoder.mapper.shift = self.pitch_shift_factor;
		self.vocoder.gain = 1.0 / overlap_add_gain(self.vocoder.window_factor);
		self.vocoder.process(samples, other, context);
	}
}
