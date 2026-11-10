//! The RBJ cookbook biquad designs, in one place.
//!
//! Every second order filter in this crate is designed from the same handbook
//! formulas, and before this module each caller carried its own copy: the
//! public Biquad setters, the EQ bands and the OTT crossover. The copies had
//! already drifted apart in how they clamp a cutoff, and a fix in one could not
//! reach the others.
//!
//! The functions take the normalised frequency w0 and the alpha term rather
//! than a "kind", because callers parameterise alpha differently -- a Q, a
//! bandwidth, or a shelf slope -- and that choice belongs to the caller. What is
//! shared is the algebra after alpha is known.

/// A pair of numerator and denominator coefficients, normalised by a0.
pub(crate) type Coefficients = ([f32; 3], [f32; 2]);

/// The normalised cutoff frequency of a design, in radians per sample.
pub(crate) fn w0(sample_rate: f32, cutoff: f32) -> f32 {
	2.0 * std::f32::consts::PI * cutoff / sample_rate
}

/// The alpha of a design that is parameterised by Q.
pub(crate) fn alpha_for_q(w0: f32, q: f32) -> f32 {
	w0.sin() / (2.0 * q)
}

fn normalise(b0: f32, b1: f32, b2: f32, a0: f32, a1: f32, a2: f32) -> Coefficients {
	([b0 / a0, b1 / a0, b2 / a0], [a1 / a0, a2 / a0])
}

/// A second order lowpass.
pub(crate) fn low_pass(w0: f32, alpha: f32) -> Coefficients {
	let cos = w0.cos();

	let b0 = (1.0 - cos) / 2.0;
	let b1 = b0 * 2.0;
	let b2 = b0;

	normalise(b0, b1, b2, 1.0 + alpha, -2.0 * cos, 1.0 - alpha)
}

/// A second order highpass.
pub(crate) fn high_pass(w0: f32, alpha: f32) -> Coefficients {
	let cos = w0.cos();

	let b0 = (1.0 + cos) / 2.0;
	let b1 = -b0 * 2.0;
	let b2 = b0;

	normalise(b0, b1, b2, 1.0 + alpha, -2.0 * cos, 1.0 - alpha)
}

/// A bandpass with a peak gain of one.
pub(crate) fn band_pass(w0: f32, alpha: f32) -> Coefficients {
	normalise(
		alpha,
		0.0,
		-alpha,
		1.0 + alpha,
		-2.0 * w0.cos(),
		1.0 - alpha,
	)
}

/// A bandstop, the reciprocal of a peaking band at unity gain.
pub(crate) fn band_stop(w0: f32, alpha: f32) -> Coefficients {
	let cos = w0.cos();

	normalise(1.0, -2.0 * cos, 1.0, 1.0 + alpha, -2.0 * cos, 1.0 - alpha)
}

/// A peaking EQ band: gain in decibels at the cutoff, flat either side.
///
/// The linear gain is the square root of the amplitude ratio, which is what the
/// RBJ reference writes as A = 10^(dB/40). Using 20 here doubled the gain.
pub(crate) fn bell(w0: f32, alpha: f32, gain_db: f32) -> Coefficients {
	let a = 10.0_f32.powf(gain_db / 40.0);
	let cos = w0.cos();

	normalise(
		1.0 + alpha * a,
		-2.0 * cos,
		1.0 - alpha * a,
		1.0 + alpha / a,
		-2.0 * cos,
		1.0 - alpha / a,
	)
}

/// A low shelf.
///
/// alpha carries the slope: the RBJ shelf formula reads the slope through
/// alpha, and clamping it is the caller's business because the useful range
/// depends on the gain.
pub(crate) fn low_shelf(w0: f32, alpha: f32, gain_db: f32) -> Coefficients {
	let a = 10.0_f32.powf(gain_db / 40.0);
	let cos = w0.cos();
	let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;

	normalise(
		a * ((a + 1.0) - (a - 1.0) * cos + two_sqrt_a_alpha),
		2.0 * a * ((a - 1.0) - (a + 1.0) * cos),
		a * ((a + 1.0) - (a - 1.0) * cos - two_sqrt_a_alpha),
		(a + 1.0) + (a - 1.0) * cos + two_sqrt_a_alpha,
		-2.0 * ((a - 1.0) + (a + 1.0) * cos),
		(a + 1.0) + (a - 1.0) * cos - two_sqrt_a_alpha,
	)
}

/// A high shelf.
pub(crate) fn high_shelf(w0: f32, alpha: f32, gain_db: f32) -> Coefficients {
	let a = 10.0_f32.powf(gain_db / 40.0);
	let cos = w0.cos();
	let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;

	normalise(
		a * ((a + 1.0) + (a - 1.0) * cos + two_sqrt_a_alpha),
		-2.0 * a * ((a - 1.0) + (a + 1.0) * cos),
		a * ((a + 1.0) + (a - 1.0) * cos - two_sqrt_a_alpha),
		(a + 1.0) - (a - 1.0) * cos + two_sqrt_a_alpha,
		2.0 * ((a - 1.0) - (a + 1.0) * cos),
		(a + 1.0) - (a - 1.0) * cos - two_sqrt_a_alpha,
	)
}

#[cfg(test)]
mod tests {
	use super::*;

	/// The magnitude of a normalised biquad at one frequency.
	fn gain_at((b, a): Coefficients, at: f32) -> f32 {
		let [b0, b1, b2] = b;
		let [a1, a2] = a;
		let cos = at.cos();
		let cos2 = (2.0 * at).cos();

		let numerator =
			(b0 * b0 + b1 * b1 + b2 * b2 + 2.0 * (b0 * b1 + b1 * b2) * cos + 2.0 * b0 * b2 * cos2)
				.max(0.0)
				.sqrt();
		let denominator = (1.0 + a1 * a1 + a2 * a2 + 2.0 * (a1 + a1 * a2) * cos + 2.0 * a2 * cos2)
			.max(0.0)
			.sqrt();

		numerator / denominator
	}

	#[test]
	fn a_lowpass_passes_dc_and_stops_nyquist() {
		let cutoff = super::w0(48_000.0, 1_000.0);
		let low = low_pass(cutoff, alpha_for_q(cutoff, std::f32::consts::FRAC_1_SQRT_2));

		assert!((gain_at(low, 0.0) - 1.0).abs() < 1e-3);
		assert!(gain_at(low, std::f32::consts::PI) < 1e-3);
	}

	#[test]
	fn a_highpass_stops_dc_and_passes_nyquist() {
		let cutoff = super::w0(48_000.0, 1_000.0);
		let high = high_pass(cutoff, alpha_for_q(cutoff, std::f32::consts::FRAC_1_SQRT_2));

		assert!(gain_at(high, 0.0) < 1e-3);
		assert!((gain_at(high, std::f32::consts::PI) - 1.0).abs() < 1e-2);
	}

	#[test]
	fn a_bell_at_zero_db_is_flat() {
		let cutoff = super::w0(48_000.0, 1_000.0);
		let flat = bell(cutoff, alpha_for_q(cutoff, 0.707), 0.0);

		for frequency in [100.0, 1_000.0, 5_000.0] {
			let at = super::w0(48_000.0, frequency);
			assert!((gain_at(flat, at) - 1.0).abs() < 1e-3);
		}
	}

	#[test]
	fn a_low_shelf_raises_the_low_end_by_the_gain() {
		let cutoff = super::w0(48_000.0, 1_000.0);
		let shelf = low_shelf(cutoff, cutoff.sin(), 6.0);

		let dc = gain_at(shelf, 0.0);
		assert!(
			(dc - 10.0_f32.powf(6.0 / 20.0)).abs() < 1e-2,
			"dc gain {dc}"
		);

		let high = gain_at(shelf, std::f32::consts::PI);
		assert!((high - 1.0).abs() < 1e-2, "high gain {high}");
	}
}
