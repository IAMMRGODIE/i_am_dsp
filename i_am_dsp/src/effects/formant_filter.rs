//! A vowel formant filter with a formant shift.
//!
//! Three parallel band-pass resonators impose a vowel's first three formants on
//! the input. The shift control moves all three formants together, in semitones,
//! which changes the apparent vocal tract size without changing the pitch of the
//! source. It is a formant filter, not a pitch-preserving spectral formant
//! shifter: the input's own resonances are not separated from the vowel bank.

use i_am_dsp_derive::Parameters;

use crate::{Effect, ProcessContext, effects::filter::Biquad};

/// The first three formants of A, E, I, O and U, in Hz.
const FORMANTS: [[f32; 3]; 5] = [
	[800.0, 1_150.0, 2_900.0],
	[400.0, 1_700.0, 2_600.0],
	[350.0, 1_700.0, 2_700.0],
	[450.0, 800.0, 2_830.0],
	[325.0, 700.0, 2_530.0],
];

/// The bandwidths that go with each vowel's formants, in Hz.
const BANDWIDTHS: [[f32; 3]; 5] = [
	[80.0, 90.0, 120.0],
	[70.0, 80.0, 100.0],
	[60.0, 80.0, 100.0],
	[70.0, 80.0, 100.0],
	[50.0, 60.0, 80.0],
];

/// A three-formant vowel filter with a semitone formant shift.
#[derive(Parameters)]
pub struct FormantFilter<const CHANNELS: usize = 2> {
	/// The vowel whose formants are imposed on the input.
	#[range(min = 0, max = 4)]
	#[choices("A", "E", "I", "O", "U")]
	#[default(0)]
	#[id(display = "Vowel")]
	#[layout(group = "Formant", order = 0, widget = choice, hero = "formant")]
	pub vowel: i32,

	/// Moves all three formants together.
	#[range(min = -24.0, max = 24.0)]
	#[default(0.0)]
	#[id(display = "Shift (st)")]
	#[layout(group = "Formant", order = 1, widget = knob, size = large)]
	pub shift: f32,

	/// How narrow the formant peaks are.
	#[range(min = 0.5, max = 12.0)]
	#[default(4.0)]
	#[id(display = "Resonance")]
	#[layout(group = "Formant", order = 2, widget = knob)]
	pub resonance: f32,

	/// How much of the filtered signal is heard.
	#[range(min = 0.0, max = 1.0)]
	#[unit(percent)]
	#[default(0.65)]
	#[id(display = "Mix")]
	#[layout(group = "Formant", order = 3, widget = knob)]
	pub mix: f32,

	#[skip]
	sample_rate: f32,
	#[skip]
	filters: [Biquad<CHANNELS>; 3],
	/// The last settings the filters were built for.
	#[skip]
	settings: Option<(i32, f32, f32)>,
}

impl<const CHANNELS: usize> FormantFilter<CHANNELS> {
	/// A formant filter on A with no shift.
	pub fn new(sample_rate: usize) -> Self {
		assert!(CHANNELS > 0, "CHANNELS must be greater than 0");

		let mut effect = Self {
			vowel: 0,
			shift: 0.0,
			resonance: 4.0,
			mix: 0.65,
			sample_rate: sample_rate as f32,
			filters: core::array::from_fn(|_| Biquad::new(sample_rate)),
			settings: None,
		};
		effect.retune();
		effect
	}

	fn retune(&mut self) {
		let settings = (self.vowel, self.shift, self.resonance);
		if self.settings == Some(settings) {
			return;
		}
		self.settings = Some(settings);

		let vowel = self.vowel.clamp(0, FORMANTS.len() as i32 - 1) as usize;
		let factor = 2.0_f32.powf(self.shift.clamp(-24.0, 24.0) / 12.0);
		let q = self.resonance.clamp(0.5, 12.0);

		for (filter, index) in self.filters.iter_mut().zip(0..3) {
			let frequency = (FORMANTS[vowel][index] * factor).clamp(30.0, self.sample_rate * 0.45);
			let bandwidth = (BANDWIDTHS[vowel][index] * (4.0 / q)).clamp(20.0, 2_000.0);
			filter.set_to_bandpass(frequency, bandwidth);
		}
	}
}

impl<const CHANNELS: usize> Effect<CHANNELS> for FormantFilter<CHANNELS> {
	fn delay(&self) -> usize {
		0
	}

	fn process(
		&mut self,
		samples: &mut [f32; CHANNELS],
		_other: &[&[f32; CHANNELS]],
		_process_context: &mut Box<dyn ProcessContext>,
	) {
		self.retune();

		let input = *samples;
		let mut wet = [0.0_f32; CHANNELS];

		for filter in &mut self.filters {
			let filtered = filter.process_sample(input);
			for (sum, sample) in wet.iter_mut().zip(filtered) {
				*sum += sample;
			}
		}

		let mix = self.mix.clamp(0.0, 1.0);

		for channel in 0..CHANNELS {
			samples[channel] = input[channel] * (1.0 - mix) + wet[channel] * (mix * 0.7);
		}
	}

	fn clear_tail(&mut self) {
		for filter in &mut self.filters {
			filter.clear_state();
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{Effect, NoteEvent, ProcessContext, ProcessInfos};

	struct TestContext;

	impl ProcessContext for TestContext {
		fn infos(&self) -> &ProcessInfos {
			&crate::DEFAULT_PROCESS_INFO
		}

		fn next_event(&mut self) -> Option<NoteEvent> {
			None
		}

		fn send_event(&mut self, _: NoteEvent) {}

		fn events(&self) -> &[NoteEvent] {
			&[]
		}

		fn clear_events(&mut self) {}
	}

	fn render(vowel: i32, shift: f32) -> Vec<f32> {
		let mut shifter = FormantFilter::<1>::new(48_000);
		shifter.vowel = vowel;
		shifter.shift = shift;
		let mut context: Box<dyn ProcessContext> = Box::new(TestContext);
		let mut output = Vec::new();

		for index in 0..4_800 {
			let phase = 2.0 * core::f32::consts::PI * 220.0 * index as f32 / 48_000.0;
			let mut frame = [0.5 * phase.sin()];
			shifter.process(&mut frame, &[], &mut context);
			output.push(frame[0]);
		}

		output
	}

	fn energy(frames: &[f32]) -> f32 {
		frames.iter().map(|sample| sample * sample).sum()
	}

	#[test]
	fn a_vowel_keeps_a_bounded_tone() {
		let output = render(0, 0.0);
		assert!(energy(&output) > 0.0);
		assert!(output.iter().all(|sample| sample.is_finite()));
	}

	#[test]
	fn the_shift_moves_the_formants() {
		let open = render(0, 0.0);
		let shifted = render(0, 12.0);
		assert!(
			open.iter()
				.zip(shifted.iter())
				.any(|(a, b)| (a - b).abs() > 1e-3),
			"the formant shift had no effect"
		);
	}

	#[test]
	fn the_vowel_changes_the_timbre() {
		let a = render(0, 0.0);
		let u = render(4, 0.0);
		assert!(
			a.iter().zip(u.iter()).any(|(a, b)| (a - b).abs() > 1e-3),
			"the vowel had no effect"
		);
	}
}
