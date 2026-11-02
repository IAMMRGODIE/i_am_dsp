//! A filter with seven algorithms, each available as a cascade of up to four
//! biquads.
//!
//! The DSP crate has a lowpass, a highpass and a bandpass, each with its own type;
//! the RBJ setters on Biquad cover the rest. A plug-in that offers them all cannot
//! swap its type at runtime, so this owns four Biquads, sets the first stages to
//! the current mode and leaves the rest as pass-through. One biquad is 12 dB/oct;
//! four in series are 48. The parameters are shared where they overlap and the
//! faceplate hides the ones a mode does not read.

use crate::prelude::{Biquad, Effect, ProcessContext};
use i_am_dsp_derive::Parameters;

/// The parameters the page draws in the header, beside the mode.
pub const FILTER_HEADER: &[&str] = &["stages"];

/// The controls each mode reads; the page hides the rest.
pub const FILTER_VISIBLE: &[(i32, &[&str])] = &[
	(0, &["cutoff", "q", "stages"]),
	(1, &["cutoff", "q", "stages"]),
	(2, &["cutoff", "q", "stages"]),
	(3, &["cutoff", "q", "stages"]),
	(4, &["cutoff", "q", "gain_db"]),
	(5, &["cutoff", "gain_db", "slope"]),
	(6, &["cutoff", "gain_db", "slope"]),
];

/// How many biquads a cascade can hold.
const STAGES: usize = 4;

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

	/// How many 12 dB/oct sections are in series.
	#[range(min = 0, max = 3)]
	#[choices("12 dB/oct", "24 dB/oct", "36 dB/oct", "48 dB/oct")]
	#[default(0)]
	#[id(display = "Slope")]
	#[layout(group = "Filter", order = 3, widget = choice)]
	pub stages: i32,

	/// How much a peak or a shelf lifts or cuts, in dB.
	#[range(min = -24.0, max = 24.0)]
	#[unit(db)]
	#[default(0.0)]
	#[id(display = "Gain")]
	#[layout(group = "Filter", order = 4, widget = knob)]
	pub gain_db: f32,

	/// How steep the shelf transition is; RBJ's slope is at most 1.
	#[range(min = 0.5, max = 1.0)]
	#[default(1.0)]
	#[id(display = "Shape")]
	#[layout(group = "Filter", order = 5, widget = knob)]
	pub slope: f32,

	/// The sections; the first stages plus one carry the mode, the rest pass
	/// through.
	#[skip]
	filters: [Biquad<2>; STAGES],

	/// The values the coefficients were computed from, so a block that changed
	/// nothing does not recompute them.
	#[skip]
	cached: Option<(i32, f32, f32, i32, f32, f32)>,
}

impl Multimode {
	/// A lowpass at the given cutoff, which is where a fresh filter starts.
	pub fn new(sample_rate: usize, cutoff: f32, q: f32) -> Self {
		Self {
			mode: 0,
			cutoff,
			q,
			stages: 0,
			gain_db: 0.0,
			slope: 1.0,
			filters: core::array::from_fn(|_| Biquad::lowpass(sample_rate, cutoff, q)),
			cached: None,
		}
	}

	/// The number of sections the current mode actually uses. Peak and shelves
	/// are one section: repeating them would multiply the gain as well as the
	/// slope.
	fn section_count(&self) -> usize {
		if self.mode >= 4 {
			1
		} else {
			(self.stages.clamp(0, 3) + 1) as usize
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
		let key = (
			self.mode,
			self.cutoff,
			self.q,
			self.stages,
			self.gain_db,
			self.slope,
		);
		let sections = self.section_count();

		if self.cached != Some(key) {
			// The RBJ band setters take a bandwidth in hertz, so Q has to be
			// turned back into one: a high Q is a narrow band.
			let bandwidth = (self.cutoff / self.q.max(0.05)).max(1.0);

			for (index, filter) in self.filters.iter_mut().enumerate() {
				if index >= sections {
					// A pass-through section keeps the inactive slots harmless and
					// their state fed with the current signal, so raising the slope
					// does not start from silence.
					filter.set_to_default_state();
					continue;
				}

				match self.mode {
					1 => filter.set_to_highpass(self.cutoff, self.q),
					2 => filter.set_to_bandpass(self.cutoff, bandwidth),
					3 => filter.set_to_bandstop(self.cutoff, bandwidth),
					4 => filter.set_to_peak(self.cutoff, self.gain_db, bandwidth),
					5 => filter.set_to_low_shelf(self.cutoff, self.gain_db, self.slope),
					6 => filter.set_to_high_shelf(self.cutoff, self.gain_db, self.slope),
					_ => filter.set_to_lowpass(self.cutoff, self.q),
				}
			}

			self.cached = Some(key);
		}

		for filter in self.filters.iter_mut() {
			filter.process(samples, other, context);
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// The level a mode leaves of a sine at a frequency, after the filter settles.
	fn level_with(mode: i32, frequency: f32, gain_db: f32, stages: i32) -> f32 {
		let mut filter = Multimode::new(48_000, 1_000.0, 0.707);
		filter.mode = mode;
		filter.gain_db = gain_db;
		filter.stages = stages;

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

	fn level_with_gain(mode: i32, frequency: f32, gain_db: f32) -> f32 {
		level_with(mode, frequency, gain_db, 0)
	}

	fn level(mode: i32, frequency: f32) -> f32 {
		level_with_gain(mode, frequency, 0.0)
	}

	/// A mode is not a label: each one has to pass its own band and reject the other.
	#[test]
	fn the_three_basic_modes_pass_different_bands() {
		assert!(level(0, 100.0) > 0.3, "the lowpass passes what is below it");
		assert!(level(0, 10_000.0) < 0.05, "and rejects what is above it");

		assert!(
			level(1, 100.0) < 0.05,
			"the highpass rejects what is below it"
		);
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

	/// Four sections really are steeper than one, and the passband stays there.
	#[test]
	fn the_slope_stacks_sections() {
		let shallow = level_with(0, 8_000.0, 0.0, 0);
		let steep = level_with(0, 8_000.0, 0.0, 3);

		assert!(steep < shallow * 0.1, "48 dB/oct rejects far more than 12");
		assert!(
			level_with(0, 100.0, 0.0, 3) > 0.3,
			"and still passes the band"
		);
	}
	/// A large shelf gain at the old slope limit used to take the square root of
	/// a negative number; it has to stay finite for every shelf extreme.
	#[test]
	fn the_shelf_extremes_stay_finite() {
		for mode in [5, 6] {
			for gain in [-24.0_f32, 0.0, 24.0] {
				for slope in [0.5_f32, 1.0, 2.0] {
					let mut filter = Multimode::new(48_000, 1_000.0, 0.707);
					filter.mode = mode;
					filter.gain_db = gain;
					filter.slope = slope;

					let mut context: Box<dyn ProcessContext> = Box::new(());

					for index in 0..512 {
						let input =
							(std::f32::consts::TAU * 440.0 * index as f32 / 48_000.0).sin() * 0.5;
						let mut samples = [input, input];
						filter.process(&mut samples, &[], &mut context);

						assert!(
							samples.iter().all(|sample| sample.is_finite()),
							"mode {mode}, gain {gain}, slope {slope}: {samples:?}"
						);
					}
				}
			}
		}
	}
}
