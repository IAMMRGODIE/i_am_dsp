//! The gain computers behind the dynamics effects.
//!
//! A dynamics stage is two halves: an envelope that turns a signal into a level,
//! and a curve that turns that level into a gain. The envelope belongs to the
//! effect that owns the detector; the curve is the same arithmetic everywhere,
//! and this module is where it lives, so a time-domain compressor and a spectral
//! one that compresses each bin call the same function and cannot drift apart.
//!
//! Every curve works in decibels: a level in, a gain in. A gain of 0 dB is a
//! pass through, and no curve asks for a gain that would undo what another curve
//! in the chain asked for.

/// Linear amplitude for a level in dB.
pub fn db_to_linear(db: f32) -> f32 {
	10.0_f32.powf(db / 20.0)
}

/// Level in dB for a linear amplitude, with a floor instead of -infinity.
pub fn linear_to_db(value: f32) -> f32 {
	20.0 * value.max(1e-9).log10()
}

/// The gain a compressor applies to `level_db`, in dB.
///
/// Above the threshold the output level climbs by `1 / ratio` of the input,
/// which is the usual `(x - T) / R + T` written as a gain. A ratio of one is
/// the identity; a ratio of infinity pins everything above the threshold to it.
/// A ratio at or below zero would divide by zero, and reads as infinity: full
/// attenuation above the threshold is the only useful meaning of "compress by
/// nothing".
pub fn compressor_gain_db(level_db: f32, threshold_db: f32, ratio: f32) -> f32 {
	if level_db <= threshold_db {
		return 0.0;
	}

	let above = level_db - threshold_db;

	if ratio <= 0.0 || ratio.is_infinite() {
		return -above;
	}

	above * (1.0 / ratio - 1.0)
}

/// The gain a downward expander applies to `level_db`, in dB.
///
/// Below the threshold the level falls by `ratio` times its distance to the
/// threshold, so quiet detail gets quieter instead of being cut off like a gate.
/// `range_db` is the deepest attenuation the curve may ask for.
pub fn expander_gain_db(level_db: f32, threshold_db: f32, ratio: f32, range_db: f32) -> f32 {
	if level_db >= threshold_db {
		return 0.0;
	}

	let below = threshold_db - level_db;
	let slope = (below * (ratio.max(1.0) - 1.0)).min(range_db.max(0.0));

	-slope.max(0.0)
}

/// The gain a limiter applies to `level_db`, in dB: just enough to stay under
/// the ceiling, and nothing at all below it.
pub fn limiter_gain_db(level_db: f32, ceiling_db: f32) -> f32 {
	if level_db > ceiling_db {
		ceiling_db - level_db
	} else {
		0.0
	}
}

/// The compressor's slope above one threshold and the expander's slope below
/// another, read from the same measured level.
///
/// A processor that follows one level can run both stages without one feeding
/// the other: both curves see the level that was measured, so the gain is the
/// sum of what each would apply on its own. That is the difference between this
/// and chaining two effects, where the compressor's attenuation becomes the
/// expander's input and the two curves fight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompanderCurve {
	/// The level above which the compressor acts.
	pub compress_threshold_db: f32,
	/// How much the compressor reduces a level above its threshold; 1 is off.
	pub compress_ratio: f32,
	/// The level below which the expander acts.
	pub expand_threshold_db: f32,
	/// How much the expander pushes a level below its threshold down; 1 is off.
	pub expand_ratio: f32,
	/// The deepest attenuation the expander may apply.
	pub range_db: f32,
}

impl CompanderCurve {
	/// The compressor's gain and the expander's gain, in dB, for one level.
	///
	/// They are kept apart because a processor smooths them with different time
	/// constants: a compressor wants a fast attack, while lifting a quiet level
	/// wants a slow one so the noise floor does not swell.
	pub fn gains_db(&self, level_db: f32) -> (f32, f32) {
		(
			compressor_gain_db(level_db, self.compress_threshold_db, self.compress_ratio),
			expander_gain_db(
				level_db,
				self.expand_threshold_db,
				self.expand_ratio,
				self.range_db,
			),
		)
	}

	/// The combined gain, in dB, for one level.
	pub fn gain_db(&self, level_db: f32) -> f32 {
		let (compress, expand) = self.gains_db(level_db);

		compress + expand
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// A ratio of one has to leave the level alone: a bypassed curve that still
	/// shaves a tenth of a decibel is a bug nobody finds for a year.
	#[test]
	fn a_compressor_at_one_to_one_is_the_identity() {
		for level in [-60.0, -20.5, -20.0, -0.5, 0.0] {
			assert_eq!(compressor_gain_db(level, -20.0, 1.0), 0.0);
		}
	}

	/// Four to one turns 20 dB above the threshold into 5, so the gain is -15 dB.
	#[test]
	fn a_four_to_one_compressor_takes_three_quarters_of_the_distance() {
		let gain = compressor_gain_db(0.0, -20.0, 4.0);

		assert!((gain + 15.0).abs() < 1e-5, "gain {gain}");
	}

	/// An infinite ratio is the hard ceiling the curve has always promised.
	#[test]
	fn an_infinite_ratio_pins_the_level_to_the_threshold() {
		let gain = compressor_gain_db(-6.0, -20.0, f32::INFINITY);

		assert!((gain + 14.0).abs() < 1e-5, "gain {gain}");
	}

	/// A ratio at or below zero reads as infinity instead of dividing by zero.
	#[test]
	fn a_degenerate_ratio_stays_finite() {
		for ratio in [0.0, -2.0] {
			let gain = compressor_gain_db(0.0, -20.0, ratio);

			assert!(gain.is_finite(), "ratio {ratio} gave {gain}");
			assert!((gain + 20.0).abs() < 1e-5, "ratio {ratio} gave {gain}");
		}
	}

	/// Below the threshold the expander has no opinion.
	#[test]
	fn an_expander_leaves_a_level_above_its_threshold_alone() {
		assert_eq!(expander_gain_db(-10.0, -40.0, 4.0, 30.0), 0.0);
	}

	/// Five dB under the threshold at four to one falls 15 dB; further down the
	/// range is what stops it, not the slope.
	#[test]
	fn the_expander_range_caps_how_far_it_pushes() {
		let shallow = expander_gain_db(-45.0, -40.0, 4.0, 30.0);

		assert!((shallow + 15.0).abs() < 1e-5, "gain {shallow}");

		let deep = expander_gain_db(-60.0, -40.0, 4.0, 30.0);

		assert!((deep + 30.0).abs() < 1e-5, "gain {deep}");
	}

	/// A ratio of one on the expander is off, not a gate at unity.
	#[test]
	fn an_expander_at_one_to_one_is_the_identity() {
		for level in [-80.0, -41.0, -40.0] {
			assert_eq!(expander_gain_db(level, -40.0, 1.0, 30.0), 0.0);
		}
	}

	/// The limiter only ever pushes down, and only past the ceiling.
	#[test]
	fn the_limiter_only_cuts_above_the_ceiling() {
		assert_eq!(limiter_gain_db(-9.0, -6.0), 0.0);

		let gain = limiter_gain_db(0.0, -6.0);

		assert!((gain + 6.0).abs() < 1e-5, "gain {gain}");
	}

	/// A loud level is only compressed, a quiet one only expanded, and a level
	/// between the two thresholds passes through.
	#[test]
	fn the_compander_runs_both_slopes_on_the_same_measured_level() {
		let curve = CompanderCurve {
			compress_threshold_db: -18.0,
			compress_ratio: 3.0,
			expand_threshold_db: -45.0,
			expand_ratio: 2.0,
			range_db: 30.0,
		};

		let loud = curve.gain_db(0.0);
		assert!((loud + 12.0).abs() < 1e-5, "loud {loud}");

		let quiet = curve.gain_db(-60.0);
		assert!((quiet + 15.0).abs() < 1e-5, "quiet {quiet}");

		let between = curve.gain_db(-30.0);
		assert_eq!(between, 0.0);
	}
}
