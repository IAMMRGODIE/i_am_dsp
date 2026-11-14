//! A wavetable used as a filter: one frame becomes a short FIR kernel, and the
//! cutoff moves the frame's twenty-fourth harmonic.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use i_am_dsp_derive::Parameters;

use crate::{Effect, ProcessContext};

/// One breakpoint of the frame the page draws. The DSP reads a dense frame
/// built from these, so editing stays cheap and the audio side never parses UI
/// data.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FilterPoint {
	/// Position in the frame, 0..1.
	pub time: f32,
	/// Amplitude, 0..1 (the DSP maps it to -1..1).
	pub value: f32,
	/// Segment bend.
	pub bend: f32,
}

/// The frame the page edits, shared with the effect.
pub struct FilterFrame {
	generation: AtomicU64,
	points: Mutex<Arc<Vec<FilterPoint>>>,
}

impl FilterFrame {
	/// A saw-like default with enough harmonics for the cutoff mapping to
	/// mean something.
	pub fn new() -> Arc<Self> {
		Arc::new(Self {
			generation: AtomicU64::new(0),
			points: Mutex::new(Arc::new(default_points())),
		})
	}

	/// The points the page edits.
	pub fn points(&self) -> Arc<Vec<FilterPoint>> {
		self.points
			.lock()
			.unwrap_or_else(|error| error.into_inner())
			.clone()
	}

	/// Replaces the points and moves the generation.
	pub fn set_points(&self, points: Vec<FilterPoint>) {
		let points = if points.len() < 2 {
			default_points()
		} else {
			points
		};
		*self
			.points
			.lock()
			.unwrap_or_else(|error| error.into_inner()) = Arc::new(points);
		self.generation.fetch_add(1, Ordering::Release);
	}

	/// Moves whenever the points change.
	pub fn generation(&self) -> u64 {
		self.generation.load(Ordering::Acquire)
	}
}

impl Default for FilterFrame {
	fn default() -> Self {
		Self {
			generation: AtomicU64::new(0),
			points: Mutex::new(Arc::new(default_points())),
		}
	}
}

fn default_points() -> Vec<FilterPoint> {
	(0..=32)
		.map(|index| {
			let time = index as f32 / 32.0;
			FilterPoint {
				time,
				value: time,
				bend: 0.0,
			}
		})
		.collect()
}

/// The kernel length. Long enough for a usable low cutoff at 48 kHz, short
/// enough that a direct FIR is cheap.
const TAPS: usize = 1024;
/// The dense frame the points are sampled into before the kernel resample.
const FRAME: usize = 2048;

/// A convolution filter whose kernel is a wavetable frame.
#[derive(Parameters)]
pub struct FilterTableEffect {
	/// The frequency the frame's twenty-fourth harmonic lands on.
	#[range(min = 20.0, max = 20000.0)]
	#[logarithmic]
	#[unit(hertz)]
	#[default(1000.0)]
	#[id(display = "Cutoff")]
	#[layout(group = "Filter", order = 0, widget = knob, hero = "filtertable")]
	pub cutoff: f32,
	/// How much the peaks of the frame are emphasised.
	#[range(min = 0.0, max = 1.0)]
	#[default(0.25)]
	#[id(display = "Resonance")]
	#[layout(group = "Filter", order = 1, widget = knob)]
	pub resonance: f32,
	/// Dry/wet blend.
	#[range(min = 0.0, max = 1.0)]
	#[default(1.0)]
	#[id(display = "Mix")]
	#[layout(group = "Filter", order = 2, widget = knob)]
	pub mix: f32,
	#[skip]
	frame: Arc<FilterFrame>,
	#[skip]
	sample_rate: f32,
	#[skip]
	kernel: [f32; TAPS],
	#[skip]
	history: [Vec<f32>; 2],
	#[skip]
	write: usize,
	#[skip]
	seen_generation: u64,
	#[skip]
	seen_cutoff: f32,
	#[skip]
	seen_resonance: f32,
}

impl FilterTableEffect {
	/// A filter over a frame the page owns.
	pub fn with_frame(sample_rate: usize, frame: Arc<FilterFrame>) -> Self {
		let mut effect = Self {
			cutoff: 1000.0,
			resonance: 0.25,
			mix: 1.0,
			frame,
			sample_rate: sample_rate as f32,
			kernel: [0.0; TAPS],
			history: core::array::from_fn(|_| vec![0.0; TAPS]),
			write: 0,
			seen_generation: u64::MAX,
			seen_cutoff: f32::NAN,
			seen_resonance: f32::NAN,
		};
		effect.rebuild();
		effect
	}

	/// The frame this effect reads.
	pub fn frame(&self) -> Arc<FilterFrame> {
		self.frame.clone()
	}

	fn rebuild(&mut self) {
		self.seen_generation = self.frame.generation();
		self.seen_cutoff = self.cutoff;
		self.seen_resonance = self.resonance;

		let points = self.frame.points();
		let mut dense = [0.0f32; FRAME];

		for (index, sample) in dense.iter_mut().enumerate() {
			let time = index as f32 / FRAME as f32;
			*sample = sample_frame(&points, time) * 2.0 - 1.0;
		}

		// A saw has a DC step at the wrap; removing it keeps the kernel from
		// adding an impulse every cycle.
		let mean = dense.iter().sum::<f32>() / FRAME as f32;
		for sample in &mut dense {
			*sample -= mean;
		}

		// The frame's twenty-fourth harmonic sits at 24 * rate / FRAME; scaling
		// the read position by this ratio puts it on the cutoff.
		let ratio = self.cutoff * FRAME as f32 / (24.0 * self.sample_rate);
		let peak = self.resonance.clamp(0.0, 1.0);

		for (index, kernel) in self.kernel.iter_mut().enumerate() {
			let position = index as f32 * ratio;
			let wrapped = position.rem_euclid(FRAME as f32);
			let first = dense[wrapped.floor() as usize % FRAME];
			let second = dense[(wrapped.floor() as usize + 1) % FRAME];
			let fraction = wrapped - wrapped.floor();
			let mut value = first + (second - first) * fraction;
			let window = 0.5 - 0.5 * (std::f32::consts::TAU * index as f32 / TAPS as f32).cos();
			value *= window;
			value *= 1.0
				+ peak
					* (std::f32::consts::TAU * self.cutoff * index as f32 / self.sample_rate).cos();
			*kernel = value;
		}

		// Keep the wet level in the same neighbourhood as the dry signal.
		let energy = self
			.kernel
			.iter()
			.map(|value| value * value)
			.sum::<f32>()
			.sqrt();
		if energy > 1e-6 {
			let scale = (TAPS as f32).sqrt() / energy;
			for kernel in &mut self.kernel {
				*kernel *= scale;
			}
		}
	}
}

/// The point curve at `x`, with the same bend shape the editors use.
fn sample_frame(points: &[FilterPoint], x: f32) -> f32 {
	if points.len() < 2 {
		return 0.5;
	}

	let mut sorted = points.to_vec();
	sorted.sort_by(|a, b| {
		a.time
			.partial_cmp(&b.time)
			.unwrap_or(core::cmp::Ordering::Equal)
	});

	if x <= sorted[0].time {
		return sorted[0].value;
	}

	for pair in sorted.windows(2) {
		let left = pair[0];
		let right = pair[1];
		if x > right.time {
			continue;
		}
		let span = (right.time - left.time).max(1e-6);
		let unit = ((x - left.time) / span).clamp(0.0, 1.0);
		let bend = left.bend;
		let shaped = if bend.abs() < 1e-4 {
			unit
		} else {
			((unit * bend).exp() - 1.0) / (bend.exp() - 1.0)
		};
		return left.value + (right.value - left.value) * shaped;
	}

	sorted.last().map(|point| point.value).unwrap_or(0.5)
}

impl Effect<2> for FilterTableEffect {
	fn delay(&self) -> usize {
		0
	}

	fn process(
		&mut self,
		samples: &mut [f32; 2],
		_other: &[&[f32; 2]],
		_context: &mut Box<dyn ProcessContext>,
	) {
		if self.frame.generation() != self.seen_generation
			|| self.cutoff != self.seen_cutoff
			|| self.resonance != self.seen_resonance
		{
			self.rebuild();
		}

		for (channel, sample) in samples.iter().enumerate() {
			self.history[channel][self.write] = *sample;
		}

		let mut wet = [0.0f32; 2];

		for (channel, wet_sample) in wet.iter_mut().enumerate() {
			let mut position = self.write;

			for tap in &self.kernel {
				*wet_sample += self.history[channel][position] * *tap;
				position = if position == 0 {
					TAPS - 1
				} else {
					position - 1
				};
			}
		}

		let mix = self.mix.clamp(0.0, 1.0);
		for (channel, sample) in samples.iter_mut().enumerate() {
			*sample = *sample * (1.0 - mix) + wet[channel] * mix;
		}

		self.write = (self.write + 1) % TAPS;
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{ProcessContext, ProcessInfos};

	struct TestContext(ProcessInfos);

	impl ProcessContext for TestContext {
		fn infos(&self) -> &ProcessInfos {
			&self.0
		}
		fn next_event(&mut self) -> Option<crate::NoteEvent> {
			None
		}
		fn send_event(&mut self, _: crate::NoteEvent) {}
		fn events(&self) -> &[crate::NoteEvent] {
			&[]
		}
		fn clear_events(&mut self) {}
	}

	#[test]
	fn a_silent_kernel_passes_the_dry_signal_at_zero_mix() {
		let frame = FilterFrame::new();
		let mut effect = FilterTableEffect::with_frame(48_000, frame);
		effect.mix = 0.0;
		let mut context: Box<dyn ProcessContext> = Box::new(TestContext(ProcessInfos {
			sample_rate: 48_000,
			..ProcessInfos::default()
		}));
		let mut frame = [0.25f32, -0.5f32];

		effect.process(&mut frame, &[], &mut context);

		assert!((frame[0] - 0.25).abs() < 1e-5);
		assert!((frame[1] + 0.5).abs() < 1e-5);
	}

	#[test]
	fn the_kernel_is_finite_and_changes_with_cutoff() {
		let frame = FilterFrame::new();
		let mut effect = FilterTableEffect::with_frame(48_000, frame);
		effect.cutoff = 200.0;
		effect.rebuild();
		let low = effect.kernel;
		effect.cutoff = 8000.0;
		effect.rebuild();
		let high = effect.kernel;
		assert!(low.iter().all(|value| value.is_finite()));
		assert!(high.iter().all(|value| value.is_finite()));
		assert_ne!(low, high);
	}

	#[test]
	fn a_frame_edit_moves_the_kernel() {
		let frame = FilterFrame::new();
		let mut effect = FilterTableEffect::with_frame(48_000, frame.clone());
		let before = effect.kernel;
		frame.set_points(vec![
			FilterPoint {
				time: 0.0,
				value: 0.0,
				bend: 0.0,
			},
			FilterPoint {
				time: 0.5,
				value: 1.0,
				bend: 0.0,
			},
			FilterPoint {
				time: 1.0,
				value: 0.0,
				bend: 0.0,
			},
		]);
		effect.rebuild();
		assert_ne!(before, effect.kernel);
	}
}
