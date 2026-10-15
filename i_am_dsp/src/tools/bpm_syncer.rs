//! A free-running beat accumulator driven by a tempo, for when there is no host timeline.
//!
//! See [`crate::tools::beat_clock::BeatClock`] for the other half of the problem: a clock
//! anchored to what a host reports. Reach for this one only when there is no host to be
//! anchored to — a standalone build, or a plugin that deliberately keeps moving while the
//! transport is stopped — because a free-running accumulator cannot know about a seek, a
//! loop or a tempo change until someone tells it.
//!
//! The type itself is deprecated; its own implementation is allowed to keep using it.
#![allow(deprecated)]

use crate::parameters::Parameters;

/// Accumulates musical time from a tempo, one sample worth at a time.
///
/// The position is kept in `f64` on purpose. It used to be an `f32`, which stops advancing
/// for good once the position reaches a power of two it can no longer add a single sample's
/// worth of a beat to: at 120 bpm and 48 kHz that is beat 1024, about eight minutes in, and
/// every per-sample caller froze there.
///
/// # Deprecated
///
/// Prefer [`crate::tools::beat_clock::BeatClock`], which does everything this does and more:
/// it computes from an absolute sample count instead of accumulating (so the same rounding
/// cannot bite), it takes a tempo *ramp* rather than a single rate, it can be anchored to
/// what a host reports and re-anchored when the host speaks again, and it knows the
/// difference between a stopped transport and a running one. [`BeatClock::advance_local`] is
/// the direct replacement for a free-running grid:
///
/// ```
/// # use i_am_dsp::prelude::BeatClock;
/// # use i_am_dsp::tools::bpm_syncer::BpmSyncer;
/// # #[allow(deprecated)] {
/// // BpmSyncer::new(48_000) + next_k(bpm, 1) per sample
/// let mut syncer = BpmSyncer::new(48_000);
/// syncer.next_k(120.0, 1);
/// let old = syncer.read();
///
/// // becomes one call with the absolute sample count the grid has reached (the first
/// // call only sets the origin)
/// let mut clock = BeatClock::new(48_000);
/// clock.advance_local(0, 120.0);
/// let new = clock.advance_local(1, 120.0);
/// assert_eq!(old, new as f32);
/// # }
/// ```
#[deprecated(note = "use `BeatClock`, which is anchorable, ramp-aware and free of the accumulation rounding")]
pub struct BpmSyncer {
	current_signature: f64,
	sample_rate: usize
}

impl BpmSyncer {
	/// Create a new BpmSyncer with the given sample rate
	pub fn new(sample_rate: usize) -> Self {
		Self {
			current_signature: 0.0,
			sample_rate
		}
	}

	/// Update the current signature by one sample
	pub fn next(&mut self, current_bpm: f32) {
		self.next_k(current_bpm, 1);
	}

	/// Update the current signature by the given number of samples
	pub fn next_k(&mut self, current_bpm: f32, samples: usize) {
		let beat_per_second = current_bpm as f64 / 60.0;
		let beat_per_sample = beat_per_second / self.sample_rate as f64;
		self.current_signature += beat_per_sample * samples as f64;
	}

	/// Read the current signature
	///
	/// Rounded to `f32` for the callers that predate the wider accumulator; the rounding is
	/// at most half an `f32` ulp (some 0.06 ms of music at 120 bpm).
	pub fn read(&self) -> f32 {
		self.current_signature as f32
	}

	/// Reset the current signature to 0.0
	pub fn reset(&mut self) {
		self.current_signature = 0.0;
	}

	/// Set the current signature to the given value
	pub fn set_current(&mut self, signature: f32) {
		self.current_signature = signature as f64;
	}
}
impl Parameters for BpmSyncer {
	fn get_parameters(&self) -> Vec<crate::prelude::Parameter> {
		vec![]
	}

	fn set_parameter(&mut self, _: &str, _: crate::prelude::SetValue) -> bool {
		false
	}

	fn set_parameter_by_index(&mut self, _: usize, _: crate::prelude::SetValue) -> bool {
		false
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// The accumulator has to keep adding once the position is large. An `f32` one does not:
	/// at beat 1024 the increment (120 bpm at 48 kHz is 4.17e-5 beats per sample) is below
	/// half an `f32` ulp (6.1e-5), so every addition rounded back to the same value and the
	/// position stopped for good — about eight minutes into a run.
	#[test]
	fn the_position_keeps_advancing_past_where_an_f32_accumulator_stalls() {
		let mut syncer = BpmSyncer::new(48_000);
		syncer.set_current(1024.0);

		for _ in 0..1_000 {
			syncer.next(120.0);
		}

		let expected = 1024.0 + 1_000.0 * 120.0 / (60.0 * 48_000.0);
		let position = syncer.read() as f64;
		// `read` rounds to `f32` (half an ulp there is 6.1e-5), so the tolerance is a little
		// wider than that and far tighter than the 0.0417 beats the stall lost.
		assert!(
			(position - expected).abs() < 1e-3,
			"position {position} should have advanced to {expected}"
		);
	}

	#[test]
	fn a_bulk_step_matches_the_same_number_of_single_steps() {
		let mut bulk = BpmSyncer::new(48_000);
		bulk.next_k(120.0, 4_800);

		let mut single = BpmSyncer::new(48_000);
		for _ in 0..4_800 {
			single.next(120.0);
		}

		assert!((bulk.read() - single.read()).abs() < 1e-4);
	}

	#[test]
	fn reset_and_set_current_move_the_position() {
		let mut syncer = BpmSyncer::new(48_000);
		syncer.next_k(120.0, 48_000);
		assert!((syncer.read() - 2.0).abs() < 1e-6);

		syncer.set_current(16.0);
		assert_eq!(syncer.read(), 16.0);

		syncer.reset();
		assert_eq!(syncer.read(), 0.0);
	}
}
