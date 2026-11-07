//! A breakpoint envelope: points in seconds, each carrying the bend of the
//! segment that starts at it.
//!
//! The bend is the one the ADSR already uses, so a curve drawn in an editor and
//! a curve evaluated on the audio thread agree sample for sample.

use serde::{Deserialize, Serialize};

use crate::generators::adsr::bend;

/// One point: time in seconds, value, and the bend of the segment that starts
/// here. A negative bend spends the segment early (a fast drop), a positive one
/// spends it late (a swell).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Point {
	/// Seconds from the start of the note.
	pub time: f32,
	/// The value at that time.
	pub value: f32,
	/// The shape of the segment that starts here.
	pub bend: f32,
}

/// A breakpoint curve, sorted by time.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Curve {
	points: Vec<Point>,
}

impl Curve {
	/// A curve through the points, sorted by time.
	pub fn new(mut points: Vec<Point>) -> Self {
		points.sort_by(|a, b| a.time.total_cmp(&b.time));
		Self { points }
	}

	/// The points, in time order.
	pub fn points(&self) -> &[Point] {
		&self.points
	}

	/// Build from `(time, value, bend)` triples, which is how the editor sends
	/// a curve over the wire.
	pub fn from_triples(points: impl IntoIterator<Item = (f32, f32, f32)>) -> Self {
		Self::new(
			points
				.into_iter()
				.map(|(time, value, bend)| Point { time, value, bend })
				.collect(),
		)
	}

	/// The time of the last point, or zero for an empty curve.
	pub fn end(&self) -> f32 {
		self.points.last().map_or(0.0, |point| point.time)
	}

	/// The value at `time`, interpolated in octaves.
	///
	/// Pitch moves by octaves, not by hertz: a curve from 1024x to 1x should
	/// spend half its time around 32x, and a straight line on the editor's log
	/// axis should sound like a straight line. The bend applies to the octave
	/// ramp, so the picture and the voice agree.
	pub fn octaves(&self, time: f32) -> f32 {
		let Some(first) = self.points.first() else {
			return 0.0;
		};

		let octave = |value: f32| value.max(1e-4).log2();

		if time <= first.time {
			return octave(first.value);
		}

		let last = *self.points.last().unwrap_or(first);
		if time >= last.time {
			return octave(last.value);
		}

		for pair in self.points.windows(2) {
			let start = pair[0];
			let end = pair[1];

			if time >= start.time && time <= end.time {
				let span = (end.time - start.time).max(1e-9);
				let unit = ((time - start.time) / span).clamp(0.0, 1.0);
				let from = octave(start.value);
				let to = octave(end.value);

				return from + (to - from) * bend(unit, start.bend);
			}
		}

		octave(last.value)
	}

	/// The value at `time`.
	///
	/// Before the first point and after the last the curve holds its end value,
	/// so a voice never jumps when a curve runs out.
	pub fn value(&self, time: f32) -> f32 {
		let Some(first) = self.points.first() else {
			return 0.0;
		};

		if time <= first.time {
			return first.value;
		}

		let last = *self.points.last().unwrap_or(first);
		if time >= last.time {
			return last.value;
		}

		for pair in self.points.windows(2) {
			let start = pair[0];
			let end = pair[1];
			if time >= start.time && time <= end.time {
				let span = (end.time - start.time).max(1e-9);
				let unit = ((time - start.time) / span).clamp(0.0, 1.0);
				return start.value + (end.value - start.value) * bend(unit, start.bend);
			}
		}

		last.value
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn the_curve_holds_its_ends() {
		let curve = Curve::new(vec![
			Point {
				time: 0.0,
				value: 3.0,
				bend: 0.0,
			},
			Point {
				time: 0.1,
				value: 1.0,
				bend: 0.0,
			},
		]);

		assert_eq!(curve.value(-1.0), 3.0);
		assert_eq!(curve.value(0.0), 3.0);
		assert_eq!(curve.value(0.1), 1.0);
		assert_eq!(curve.value(10.0), 1.0);
		assert!(curve.value(0.05) < 2.5);
		assert!(curve.value(0.05) > 1.0);
	}

	#[test]
	fn a_negative_bend_spends_the_segment_early() {
		let fast = Curve::new(vec![
			Point {
				time: 0.0,
				value: 0.0,
				bend: -2.0,
			},
			Point {
				time: 1.0,
				value: 1.0,
				bend: 0.0,
			},
		]);
		let slow = Curve::new(vec![
			Point {
				time: 0.0,
				value: 0.0,
				bend: 2.0,
			},
			Point {
				time: 1.0,
				value: 1.0,
				bend: 0.0,
			},
		]);

		assert!(fast.value(0.5) > slow.value(0.5));
	}

	#[test]
	fn the_octave_ramp_is_straight() {
		let curve = Curve::new(vec![
			Point {
				time: 0.0,
				value: 1_024.0,
				bend: 0.0,
			},
			Point {
				time: 0.1,
				value: 1.0,
				bend: 0.0,
			},
		]);

		// Half the time is half the octaves: 1024 -> 32 -> 1.
		assert!((curve.octaves(0.05) - 5.0).abs() < 1e-4);
		assert!((2f32.powf(curve.octaves(0.05)) - 32.0).abs() < 1e-2);
		// The same curve in the linear domain lands near 512, which is the
		// bounce a hertz ramp gives a listener.
		assert!(curve.value(0.05) > 400.0);
	}

	#[test]
	fn an_empty_curve_is_silent() {
		let curve = Curve::default();

		assert_eq!(curve.value(0.5), 0.0);
		assert_eq!(curve.end(), 0.0);
	}
}
