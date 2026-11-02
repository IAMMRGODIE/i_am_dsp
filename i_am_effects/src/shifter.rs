//! The Shifter product with three engines: the IIR Hilbert frequency shifter,
//! the phase-vocoder frequency shifter, and the FIR Hilbert one, whose kernel
//! length the user picks.

use i_am_dsp::prelude::{
	Effect, FIRFreqShifter, FreqShift, IIRFreqShifter, PhaseVocoder, ProcessContext,
	overlap_add_gain,
};
use i_am_dsp_derive::Parameters;

pub const SHIFTER_VISIBLE: &[(i32, &[&str])] = &[
	(0, &["shift_freq"]),
	(1, &["shift_freq", "window_size"]),
	(2, &["shift_freq", "order"]),
];

#[derive(Parameters)]
pub struct ShifterEffect {
	/// The engine.
	#[choices("IIR", "Phase Vocoder", "FIR")]
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

	/// The FIR Hilbert kernel length. Longer is a sharper sideband, at the cost
	/// of latency and CPU.
	#[choices("63", "127", "255", "511")]
	#[default(1)]
	#[id(display = "Order")]
	#[layout(group = "Shift", order = 3, widget = choice)]
	pub order: i32,

	#[skip]
	time: IIRFreqShifter<8, 2>,
	#[skip]
	fir: FIRFreqShifter<2>,
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
			fir: FIRFreqShifter::<2>::new(sample_rate, 0.0),
			vocoder: PhaseVocoder::new(FreqShift::default(), 2_048, sample_rate),
			last_window_size: 2,
			order: 1,
		}
	}
}

impl Effect<2> for ShifterEffect {
	fn delay(&self) -> usize {
		match self.algorithm {
			0 => 0,
			2 => self.fir.delay(),
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
			self.time.set_shift_freq(self.shift_freq);
			self.time.process(samples, other, context);
			return;
		}

		if self.algorithm == 2 {
			self.fir.set_order([63, 127, 255, 511][self.order.clamp(0, 3) as usize]);
			self.fir.set_shift_freq(self.shift_freq);
			self.fir.process(samples, other, context);
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

#[cfg(test)]
mod tests {
	use super::*;
	use i_am_dsp::prelude::{Parameters, Value};

	/// The FIR engine is a third algorithm with its own order, which is a choice
	/// on the parameter rather than one type per length.
	#[test]
	fn the_fir_engine_is_a_choice_with_a_selectable_order() {
		let shifter = ShifterEffect::new(48_000);
		let order = shifter
			.get_parameters()
			.into_iter()
			.find(|parameter| parameter.identifier == "order")
			.expect("the order parameter");

		match order.value {
			Value::Int { value, .. } => assert_eq!(value, 1, "127 is the default kernel"),
			other => panic!("unexpected order value {other:?}"),
		}

		assert_eq!(
			order.meta.choices,
			Some(&["63", "127", "255", "511"][..]),
			"the four kernel lengths"
		);

		assert!(
			SHIFTER_VISIBLE
				.iter()
				.any(|(index, ids)| *index == 2 && ids.contains(&"order")),
			"the FIR mode does not show its order"
		);
	}
}
