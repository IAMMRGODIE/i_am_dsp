//! The filter the web product ships: seven algorithms behind one Biquad.
//!
//! The DSP crate has a lowpass, a highpass and a bandpass, each with its own type;
//! the RBJ setters on Biquad cover the rest. A plug-in that offers them all cannot
//! swap its type at runtime, so this wraps one Biquad and changes what it is set
//! to when the mode moves. The parameters are shared where they overlap and the
//! faceplate hides the ones a mode does not read.

use i_am_dsp::prelude::{Biquad, Effect, ProcessContext};
use i_am_dsp_derive::Parameters;

/// The controls each mode reads; the page hides the rest.
pub const FILTER_VISIBLE: &[(i32, &[&str])] = &[
	(0, &["cutoff", "q"]),
	(1, &["cutoff", "q"]),
	(2, &["cutoff", "q"]),
	(3, &["cutoff", "q"]),
	(4, &["cutoff", "q", "gain_db"]),
	(5, &["cutoff", "gain_db", "slope"]),
	(6, &["cutoff", "gain_db", "slope"]),
];

/// A filter that is a lowpass, highpass, bandpass, notch, peak or a shelf.
#[derive(Parameters)]
pub struct Multimode {
	/// Which algorithm the filter runs.
	#[range(min = 0, max = 6)]
	#[choices(
		"Lowpass",
		"Highpass",
		"Bandpass",
		"Notch",
		"Peak",
		"Low Shelf",
		"High Shelf"
	)]
	#[default(0)]
	#[id(display = "Mode")]
	#[layout(group = "Filter", order = 0, widget = choice)]
	pub mode: i32,

	/// The cutoff of the filter.
	#[range(min = 10.0, max = 21000.0)]
	#[logarithmic]
	#[unit(hertz)]
	#[id(display = "Cutoff")]
	#[layout(group = "Filter", order = 1, widget = knob, size = large, hero = "spectrum")]
	pub cutoff: f32,

	/// How much the filter rings at its cutoff, and how narrow a band is.
	#[range(min = 0.01, max = 10.0)]
	#[id(display = "Resonance")]
	#[layout(group = "Filter", order = 2, widget = knob)]
	pub q: f32,

	/// How much a peak or a shelf lifts or cuts, in dB.
	#[range(min = -24.0, max = 24.0)]
	#[unit(db)]
	#[default(0.0)]
	#[id(display = "Gain")]
	#[layout(group = "Filter", order = 3, widget = knob)]
	pub gain_db: f32,

	/// How steep the shelf transition is.
	#[range(min = 0.5, max = 2.0)]
	#[default(1.0)]
	#[id(display = "Slope")]
	#[layout(group = "Filter", order = 4, widget = knob)]
	pub slope: f32,

	#[sub_param]
	filter: Biquad<2>,

	/// The values the coefficients were computed from, so a block that changed
	/// nothing does not recompute them.
	#[skip]
	cached: Option<(i32, f32, f32, f32, f32)>,
}

impl Multimode {
	/// A lowpass at the given cutoff, which is where a fresh filter starts.
	pub fn new(sample_rate: usize, cutoff: f32, q: f32) -> Self {
		Self {
			mode: 0,
			cutoff,
			q,
			gain_db: 0.0,
			slope: 1.0,
			filter: Biquad::lowpass(sample_rate, cutoff, q),
			cached: None,
		}
	}
}

impl Effect<2> for Multimode {
	fn delay(&self) -> usize {
		0
	}

	fn process(
		&mut self,
		samples: &mut [f32; 2],
		other: &[&[f32; 2]],
		context: &mut Box<dyn ProcessContext>,
	) {
		let key = (self.mode, self.cutoff, self.q, self.gain_db, self.slope);

		if self.cached != Some(key) {
			// The RBJ band setters take a bandwidth in hertz, so Q has to be
			// turned back into one: a high Q is a narrow band.
			let bandwidth = (self.cutoff / self.q.max(0.05)).max(1.0);

			match self.mode {
				1 => self.filter.set_to_highpass(self.cutoff, self.q),
				2 => self.filter.set_to_bandpass(self.cutoff, bandwidth),
				3 => self.filter.set_to_bandstop(self.cutoff, bandwidth),
				4 => self.filter.set_to_peak(self.cutoff, self.gain_db, bandwidth),
				5 => self.filter.set_to_low_shelf(self.cutoff, self.gain_db, self.slope),
				6 => self.filter.set_to_high_shelf(self.cutoff, self.gain_db, self.slope),
				_ => self.filter.set_to_lowpass(self.cutoff, self.q),
			}

			self.cached = Some(key);
		}

		self.filter.process(samples, other, context);
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// The level a mode leaves of a sine at a frequency, after the filter settles.
	fn level_with_gain(mode: i32, frequency: f32, gain_db: f32) -> f32 {
		let mut filter = Multimode::new(48_000, 1_000.0, 0.707);
		filter.mode = mode;
		filter.gain_db = gain_db;

		let mut context: Box<dyn ProcessContext> = Box::new(());
		let mut energy = 0.0_f32;

		for index in 0..4_096 {
			let value = (std::f32::consts::TAU * frequency * index as f32 / 48_000.0).sin() * 0.5;
			let mut samples = [value, value];

			filter.process(&mut samples, &[], &mut context);

			if index >= 2_048 {
				energy += samples[0] * samples[0];
			}
		}

		(energy / 2_048.0).sqrt()
	}

	fn level(mode: i32, frequency: f32) -> f32 {
		level_with_gain(mode, frequency, 0.0)
	}

	/// A mode is not a label: each one has to pass its own band and reject the other.
	#[test]
	fn the_three_basic_modes_pass_different_bands() {
		assert!(level(0, 100.0) > 0.3, "the lowpass passes what is below it");
		assert!(level(0, 10_000.0) < 0.05, "and rejects what is above it");

		assert!(level(1, 100.0) < 0.05, "the highpass rejects what is below it");
		assert!(level(1, 10_000.0) > 0.3, "and passes what is above it");

		assert!(level(2, 1_000.0) > 0.2, "the bandpass passes its centre");
		assert!(level(2, 100.0) < 0.1, "and rejects both sides of it");
		assert!(level(2, 10_000.0) < 0.1);
	}

	/// A notch rejects its centre and passes both ends.
	#[test]
	fn the_notch_rejects_its_centre() {
		assert!(level(3, 1_000.0) < 0.1);
		assert!(level(3, 100.0) > 0.3);
		assert!(level(3, 10_000.0) > 0.3);
	}

	/// A peak lifts its centre and leaves the far ends alone.
	#[test]
	fn the_peak_boosts_its_centre() {
		let flat = level_with_gain(4, 1_000.0, 0.0);
		let boosted = level_with_gain(4, 1_000.0, 12.0);

		assert!(boosted > flat * 1.5, "a +12 dB peak should lift the centre");
		assert!(
			level_with_gain(4, 100.0, 12.0) < boosted,
			"and leave the far low end lower"
		);
	}

	/// The two shelves tilt opposite ends of the spectrum.
	#[test]
	fn the_shelves_tilt_opposite_ends() {
		let low = level_with_gain(5, 100.0, 12.0);
		let low_high = level_with_gain(5, 10_000.0, 12.0);
		assert!(low > low_high * 1.5, "a low shelf lifts the low end");

		let high = level_with_gain(6, 10_000.0, 12.0);
		let high_low = level_with_gain(6, 100.0, 12.0);
		assert!(high > high_low * 1.5, "a high shelf lifts the high end");
	}
}
