//! The Shifter product with two engines: the IIR Hilbert frequency shifter and
//! the phase-vocoder frequency shifter.

use i_am_dsp::prelude::{
	Effect, FreqShift, IIRFreqShifter, PhaseVocoder, ProcessContext, overlap_add_gain,
};
use i_am_dsp_derive::Parameters;

pub const SHIFTER_VISIBLE: &[(i32, &[&str])] = &[
	(0, &["shift_freq"]),
	(1, &["shift_freq", "window_size"]),
];

#[derive(Parameters)]
pub struct ShifterEffect {
	/// The engine.
	#[choices("IIR", "Phase Vocoder")]
	#[default(0)]
	#[id(display = "Algorithm")]
	#[layout(group = "Shift", order = 0, widget = choice)]
	pub algorithm: i32,

	/// The frequency shift, in Hz.
	#[range(min = -20000.0, max = 20000.0)]
	#[logarithmic]
	#[unit(hertz)]
	#[default(0.0)]
	#[id(display = "Shift")]
	#[layout(group = "Shift", order = 1, widget = knob, size = large, hero = "spectra", bipolar)]
	pub shift_freq: f32,

	/// The phase-vocoder window size, picked from the header.
	#[choices("512", "1024", "2048", "4096")]
	#[default(2)]
	#[id(display = "Window Size")]
	#[layout(group = "Shift", order = 2, widget = choice)]
	pub window_size: i32,

	#[skip]
	time: IIRFreqShifter<8, 2>,
	#[skip]
	vocoder: PhaseVocoder<FreqShift, 2>,
	#[skip]
	last_window_size: i32,
}

impl ShifterEffect {
	pub fn new(sample_rate: usize) -> Self {
		Self {
			algorithm: 0,
			shift_freq: 0.0,
			window_size: 2,
			time: IIRFreqShifter::<8, 2>::new(sample_rate, 0.0),
			vocoder: PhaseVocoder::new(FreqShift::default(), 2_048, sample_rate),
			last_window_size: 2,
		}
	}
}

impl Effect<2> for ShifterEffect {
	fn delay(&self) -> usize {
		if self.algorithm == 0 {
			0
		} else {
			self.vocoder.window_size()
		}
	}

	fn process(
		&mut self,
		samples: &mut [f32; 2],
		other: &[&[f32; 2]],
		context: &mut Box<dyn ProcessContext>,
	) {
		if self.algorithm == 0 {
			self.time.set_shift_freq(self.shift_freq);
			self.time.process(samples, other, context);
			return;
		}

		if self.window_size != self.last_window_size {
			let size = [512, 1024, 2048, 4096][self.window_size.clamp(0, 3) as usize];
			if self.vocoder.renew_window_size(size).is_some() {
				self.vocoder.gain = 1.0 / overlap_add_gain(self.vocoder.window_factor);
			}
			self.last_window_size = self.window_size;
		}

		self.vocoder.mapper.freq = self.shift_freq;
		self.vocoder.gain = 1.0 / overlap_add_gain(self.vocoder.window_factor);
		self.vocoder.process(samples, other, context);
	}
}
