//! The phase warps: functions that decide where in a table a position reads.

use i_am_dsp_derive::Parameters;

use crate::prelude::{Parameter, Parameters, SetValue};

/// A phase warp: maps a position in the cycle to the position the table is
/// read at.
pub trait Warp: Parameters + Send {
	/// The position to read at for the given position of the cycle.
	fn apply(&self, t: f32) -> f32;
}

/// Bends the phase towards the start or the end of the cycle.
#[derive(Parameters)]
pub struct BendWarp {
	/// The exponent of the bend; 1 leaves the table alone.
	#[range(min = 0.25, max = 4.0)]
	#[default(1.0)]
	#[id(display = "Bend")]
	pub power: f32,
}

impl BendWarp {
	/// A bend that leaves the table alone.
	pub fn new() -> Self {
		Self { power: 1.0 }
	}
}

impl Default for BendWarp {
	fn default() -> Self {
		Self::new()
	}
}

impl Warp for BendWarp {
	fn apply(&self, t: f32) -> f32 {
		t.powf(self.power).clamp(0.0, 1.0)
	}
}

/// Folds the second half of the cycle back over the first.
#[derive(Parameters)]
pub struct MirrorWarp {
	/// How far the second half folds back; 0 leaves the table alone.
	#[range(min = 0.0, max = 1.0)]
	#[default(0.0)]
	#[id(display = "Mirror")]
	pub amount: f32,
}

impl MirrorWarp {
	/// A fold that leaves the table alone.
	pub fn new() -> Self {
		Self { amount: 0.0 }
	}
}

impl Default for MirrorWarp {
	fn default() -> Self {
		Self::new()
	}
}

impl Warp for MirrorWarp {
	fn apply(&self, t: f32) -> f32 {
		let folded = 1.0 - (2.0 * t - 1.0).abs();
		(t + (folded - t) * self.amount).clamp(0.0, 1.0)
	}
}

/// Repeats the table inside one cycle.
#[derive(Parameters)]
pub struct SyncWarp {
	/// How many table cycles fit in one cycle; 1 leaves the table alone.
	#[range(min = 1.0, max = 8.0)]
	#[default(1.0)]
	#[layout(group = "Warp", order = 0, widget = knob, step = 1.0)]
	#[id(display = "Sync")]
	pub ratio: f32,
}

impl SyncWarp {
	/// A sync that leaves the table alone.
	pub fn new() -> Self {
		Self { ratio: 1.0 }
	}
}

impl Default for SyncWarp {
	fn default() -> Self {
		Self::new()
	}
}

impl Warp for SyncWarp {
	fn apply(&self, t: f32) -> f32 {
		// `fract` keeps the position in the cycle; the ratio is clamped so a
		// control at zero cannot divide by it.
		(t * self.ratio.max(1.0)).fract()
	}
}

/// Snaps the read position to a grid, which adds a staircase to the waveform.
#[derive(Parameters)]
pub struct QuantizeWarp {
	/// The number of grid steps; 1 leaves the table alone.
	#[range(min = 1.0, max = 64.0)]
	#[default(1.0)]
	#[layout(group = "Warp", order = 0, widget = knob, step = 1.0)]
	#[id(display = "Quantize")]
	pub steps: f32,
}

impl QuantizeWarp {
	/// A quantize that leaves the table alone.
	pub fn new() -> Self {
		Self { steps: 1.0 }
	}
}

impl Default for QuantizeWarp {
	fn default() -> Self {
		Self::new()
	}
}

impl Warp for QuantizeWarp {
	fn apply(&self, t: f32) -> f32 {
		let steps = self.steps.max(1.0);

		if steps <= 1.0 {
			t
		} else {
			((t * steps).floor() / steps).clamp(0.0, 1.0)
		}
	}
}

impl Warp for Box<dyn Warp + Send + Sync> {
	fn apply(&self, t: f32) -> f32 {
		self.as_ref().apply(t)
	}
}

impl Parameters for Box<dyn Warp + Send + Sync> {
	fn get_parameters(&self) -> Vec<Parameter> {
		self.as_ref().get_parameters()
	}

	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		self.as_mut().set_parameter(identifier, value)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// Every warp at its default leaves the position where it was.
	#[test]
	fn default_warps_are_the_identity() {
		let warps: [Box<dyn Warp + Send + Sync>; 4] = [
			Box::new(BendWarp::new()),
			Box::new(MirrorWarp::new()),
			Box::new(SyncWarp::new()),
			Box::new(QuantizeWarp::new()),
		];

		for warp in warps {
			for index in 0..64 {
				let t = index as f32 / 64.0;
				let value = warp.apply(t);

				assert!((value - t).abs() < 1e-5, "{t} -> {value}");
			}
		}
	}

	/// A bend with a larger exponent reads the table earlier in the cycle.
	#[test]
	fn bend_moves_the_middle_of_the_cycle() {
		let mut warp = BendWarp::new();
		warp.power = 2.0;

		assert!((warp.apply(0.5) - 0.25).abs() < 1e-5);
	}

	/// A full fold maps the two halves onto the first one.
	#[test]
	fn mirror_folds_the_second_half() {
		let mut warp = MirrorWarp::new();
		warp.amount = 1.0;

		assert!((warp.apply(0.25) - 0.5).abs() < 1e-5);
		assert!((warp.apply(0.75) - 0.5).abs() < 1e-5);
	}

	/// Two table cycles fit in one cycle at ratio 2.
	#[test]
	fn sync_repeats_the_cycle() {
		let mut warp = SyncWarp::new();
		warp.ratio = 2.0;

		assert!((warp.apply(0.1) - 0.2).abs() < 1e-5);
		assert!((warp.apply(0.6) - 0.2).abs() < 1e-5);
	}

	/// Four steps snap the position to quarter marks.
	#[test]
	fn quantize_snaps_to_the_grid() {
		let mut warp = QuantizeWarp::new();
		warp.steps = 4.0;

		assert!((warp.apply(0.3) - 0.25).abs() < 1e-5);
		assert!((warp.apply(0.99) - 0.75).abs() < 1e-5);
	}

	/// Whatever the control says, a warp has to land inside one cycle.
	#[test]
	fn every_warp_stays_in_the_unit_cycle() {
		let warps: [Box<dyn Warp + Send + Sync>; 4] = [
			Box::new(BendWarp::new()),
			Box::new(MirrorWarp::new()),
			Box::new(SyncWarp::new()),
			Box::new(QuantizeWarp::new()),
		];

		for warp in warps {
			for index in 0..=100 {
				let t = index as f32 / 100.0;
				let value = warp.apply(t);

				assert!(
					value.is_finite() && (0.0..=1.0).contains(&value),
					"{t} -> {value}"
				);
			}
		}
	}
}
