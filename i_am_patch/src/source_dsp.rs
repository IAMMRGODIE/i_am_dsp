//! The control-source maths the graph nodes and the rack slots share.
//!
//! The patcher carries its control sources as graph nodes, and the rack reads
//! the same six sources from a slot runtime (modulation.rs). Both need the same
//! shapes, the same synced period and the same one-pole coefficient; before this
//! module each had its own copy, and the shape function existed three times.

/// One LFO sample in -1..1.
pub(crate) fn shape(kind: i32, phase: f32) -> f32 {
	match kind.rem_euclid(4) {
		1 => 1.0 - 4.0 * (phase - 0.5).abs(),
		2 => phase * 2.0 - 1.0,
		3 => {
			if phase < 0.5 {
				1.0
			} else {
				-1.0
			}
		}
		_ => (phase * std::f32::consts::TAU).sin(),
	}
}

/// One table sample, morphing from a shape toward the next one.
pub(crate) fn morph(kind: i32, blend: f32, phase: f32) -> f32 {
	let first = self::shape(kind, phase);
	let second = self::shape(kind + 1, phase);

	first + (second - first) * blend.clamp(0.0, 1.0)
}

/// A synced musical period, in hertz.
pub(crate) fn sync_hz(tempo: f32, beats: f32) -> f32 {
	tempo / 60.0 / beats.max(0.01)
}

/// The one-pole coefficient for a time constant in milliseconds.
pub(crate) fn time_coefficient(milliseconds: f32, sample_rate: f32) -> f32 {
	if milliseconds <= 0.0 {
		1.0
	} else {
		1.0 - (-1000.0 / (milliseconds * sample_rate)).exp()
	}
}
