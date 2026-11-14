//! A wavetable used as a filter: one frame becomes a short FIR kernel, and the
//! cutoff moves the frame's twenty-fourth harmonic.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use i_am_dsp_derive::Parameters;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;
use wide::f32x4;

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

/// The frames the page edits, shared with the effect. A writer swaps a whole
/// snapshot; the audio thread loads it with one atomic and never locks or
/// copies the points.
pub struct FilterFrame {
	current: std::sync::atomic::AtomicPtr<FrameData>,
}

struct FrameData {
	generation: u64,
	pages: Vec<Vec<FilterPoint>>,
}

impl FilterFrame {
	/// A saw-like default with enough harmonics for the cutoff mapping to
	/// mean something.
	pub fn new() -> Arc<Self> {
		Arc::new(Self::default())
	}

	/// The first page, for a caller that only knows about one frame.
	pub fn points(&self) -> Vec<FilterPoint> {
		self.pages().first().cloned().unwrap_or_default()
	}

	/// Every page, for the card and the state save.
	pub fn pages(&self) -> Vec<Vec<FilterPoint>> {
		self.snapshot().pages.clone()
	}

	/// A shared snapshot. The audio thread reads this once per rebuild.
	fn snapshot(&self) -> Arc<FrameData> {
		let pointer = self.current.load(Ordering::Acquire);
		// SAFETY: every pointer stored in `current` came from `Arc::into_raw`,
		// and the writer only drops its own reference after the swap. Taking a
		// new strong count here keeps the data alive while we read it.
		unsafe {
			Arc::increment_strong_count(pointer);
			Arc::from_raw(pointer)
		}
	}

	/// Moves whenever the pages change.
	pub fn generation(&self) -> u64 {
		self.snapshot().generation
	}

	/// Replaces one page and keeps the rest.
	pub fn set_points(&self, points: Vec<FilterPoint>) {
		self.set_pages(vec![points]);
	}

	/// Replaces all pages; the swap is safe for the audio thread to read.
	pub fn set_pages(&self, mut pages: Vec<Vec<FilterPoint>>) {
		if pages.is_empty() {
			pages.push(default_points());
		}

		for page in &mut pages {
			if page.len() < 2 {
				*page = default_points();
			}

			page.sort_by(|left, right| {
				left.time
					.partial_cmp(&right.time)
					.unwrap_or(core::cmp::Ordering::Equal)
			});
		}

		let generation = self.snapshot().generation + 1;
		let data = Arc::new(FrameData { generation, pages });
		let pointer = Arc::into_raw(data) as *mut FrameData;
		let old = self.current.swap(pointer, Ordering::AcqRel);
		// SAFETY: the pointer was created by `Arc::into_raw` in this struct.
		unsafe {
			drop(Arc::from_raw(old));
		}
	}
}

impl Drop for FilterFrame {
	fn drop(&mut self) {
		let pointer = *self.current.get_mut();
		// SAFETY: the pointer was created by `Arc::into_raw` in this struct.
		unsafe {
			drop(Arc::from_raw(pointer));
		}
	}
}

impl Default for FilterFrame {
	fn default() -> Self {
		let data = Arc::new(FrameData {
			generation: 0,
			pages: vec![default_points()],
		});
		Self {
			current: std::sync::atomic::AtomicPtr::new(Arc::into_raw(data) as *mut FrameData),
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
	/// The phase response of the kernel.
	#[choices("Raw", "Linear", "Minimum")]
	#[default(0)]
	#[id(display = "Phase")]
	#[layout(group = "Filter", order = 3, widget = choice)]
	pub phase: i32,
	/// The page the frame plays, interpolated between neighbours.
	#[range(min = 0.0, max = 1.0)]
	#[default(0.0)]
	#[id(display = "Morph")]
	#[layout(group = "Filter", order = 4, widget = knob)]
	pub morph: f32,
	#[skip]
	frame: Arc<FilterFrame>,
	#[skip]
	sample_rate: f32,
	#[skip]
	kernel: [f32; TAPS],
	#[skip]
	kernel_rev: [f32; TAPS],
	#[skip]
	previous: [f32; TAPS],
	#[skip]
	previous_rev: [f32; TAPS],
	#[skip]
	next_rev: [f32; TAPS],
	#[skip]
	next: [f32; TAPS],
	#[skip]
	blend: f32,
	#[skip]
	initialized: bool,
	#[skip]
	forward: Arc<dyn rustfft::Fft<f32>>,
	#[skip]
	inverse: Arc<dyn rustfft::Fft<f32>>,
	#[skip]
	dry: [Vec<f32>; 2],
	#[skip]
	dry_write: usize,
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
	#[skip]
	seen_phase: i32,
	#[skip]
	seen_morph: f32,
}

impl FilterTableEffect {
	/// A filter over a frame the page owns.
	pub fn with_frame(sample_rate: usize, frame: Arc<FilterFrame>) -> Self {
		let mut effect = Self {
			cutoff: 1000.0,
			resonance: 0.25,
			mix: 1.0,
			phase: 0,
			morph: 0.0,
			frame,
			sample_rate: sample_rate as f32,
			kernel: [0.0; TAPS],
			kernel_rev: [0.0; TAPS],
			previous: [0.0; TAPS],
			previous_rev: [0.0; TAPS],
			next_rev: [0.0; TAPS],
			next: [0.0; TAPS],
			blend: 1.0,
			initialized: false,
			forward: {
				let mut planner = FftPlanner::<f32>::new();
				planner.plan_fft_forward(TAPS)
			},
			inverse: {
				let mut planner = FftPlanner::<f32>::new();
				planner.plan_fft_inverse(TAPS)
			},
			dry: core::array::from_fn(|_| vec![0.0; TAPS / 2 + 1]),
			dry_write: 0,
			history: core::array::from_fn(|_| vec![0.0; TAPS * 2]),
			write: 0,
			seen_generation: u64::MAX,
			seen_cutoff: f32::NAN,
			seen_resonance: f32::NAN,
			seen_phase: i32::MIN,
			seen_morph: f32::NAN,
		};
		effect.rebuild();
		effect
	}

	/// The frame this effect reads.
	pub fn frame(&self) -> Arc<FilterFrame> {
		self.frame.clone()
	}

	fn rebuild(&mut self) {
		let snapshot = self.frame.snapshot();
		self.seen_generation = snapshot.generation;
		self.seen_cutoff = self.cutoff;
		self.seen_resonance = self.resonance;
		self.seen_phase = self.phase;
		self.seen_morph = self.morph;

		let mut raw = [0.0f32; TAPS];
		let pages = &snapshot.pages;
		let last = pages.len().saturating_sub(1);
		let position = self.morph.clamp(0.0, 1.0) * last as f32;
		let from = position.floor() as usize;
		let to = (from + 1).min(last);
		let blend = position - from as f32;
		let mut first = [0.0f32; FRAME];
		let mut second = [0.0f32; FRAME];

		for index in 0..FRAME {
			let time = index as f32 / FRAME as f32;
			first[index] = sample_frame(&pages[from], time) * 2.0 - 1.0;
			second[index] = sample_frame(&pages[to], time) * 2.0 - 1.0;
		}

		let mut dense = [0.0f32; FRAME];
		for index in 0..FRAME {
			dense[index] = first[index] + (second[index] - first[index]) * blend;
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

		for (index, kernel) in raw.iter_mut().enumerate() {
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

		normalize(&mut raw);

		let target = match self.phase {
			1 => self.linear(&raw),
			2 => self.minimum(&raw),
			_ => raw,
		};

		if !self.initialized {
			self.initialized = true;
			self.previous = target;
			self.kernel = target;
			self.next = target;
			self.blend = 1.0;
		} else {
			self.previous = self.kernel;
			self.previous_rev = self.kernel_rev;
			self.next = target;
			self.blend = 0.0;
		}

		self.kernel = target;
		for index in 0..TAPS {
			self.kernel_rev[index] = target[TAPS - 1 - index];
		}
		self.next_rev = self.kernel_rev;

		if self.previous == self.kernel {
			self.previous_rev = self.kernel_rev;
		}
	}

	fn linear(&self, raw: &[f32; TAPS]) -> [f32; TAPS] {
		let mut buffer = [Complex::new(0.0, 0.0); TAPS];
		for (slot, sample) in buffer.iter_mut().zip(raw) {
			*slot = Complex::new(*sample, 0.0);
		}
		self.forward.process(&mut buffer);

		for value in &mut buffer {
			*value = Complex::new(value.norm(), 0.0);
		}
		self.inverse.process(&mut buffer);

		let mut out = [0.0f32; TAPS];
		for (index, value) in buffer.iter().enumerate() {
			out[(index + TAPS / 2) % TAPS] = value.re / TAPS as f32;
		}
		normalize(&mut out);
		out
	}

	fn minimum(&self, raw: &[f32; TAPS]) -> [f32; TAPS] {
		let mut buffer = [Complex::new(0.0, 0.0); TAPS];
		for (slot, sample) in buffer.iter_mut().zip(raw) {
			*slot = Complex::new(*sample, 0.0);
		}
		self.forward.process(&mut buffer);

		let mut cepstrum = [Complex::new(0.0, 0.0); TAPS];
		for (index, value) in cepstrum.iter_mut().enumerate() {
			*value = Complex::new((buffer[index].norm() + 1e-9).ln(), 0.0);
		}
		self.inverse.process(&mut cepstrum);

		for value in &mut cepstrum {
			*value /= TAPS as f32;
		}

		for value in cepstrum.iter_mut().take(TAPS / 2).skip(1) {
			*value *= 2.0;
		}
		for value in cepstrum.iter_mut().skip(TAPS / 2) {
			*value = Complex::new(0.0, 0.0);
		}

		self.forward.process(&mut cepstrum);
		for value in &mut cepstrum {
			let magnitude = value.re.clamp(-20.0, 20.0).exp();
			*value = Complex::new(magnitude * value.im.cos(), magnitude * value.im.sin());
		}
		self.inverse.process(&mut cepstrum);

		let mut out = [0.0f32; TAPS];
		for (index, value) in cepstrum.iter().enumerate() {
			out[index] = value.re / TAPS as f32;
		}
		normalize(&mut out);
		out
	}
}

fn dot(history: &[f32], kernel: &[f32; TAPS]) -> f32 {
	let mut sum = f32x4::ZERO;

	for index in (0..TAPS).step_by(4) {
		sum += f32x4::from(&history[index..index + 4]) * f32x4::from(&kernel[index..index + 4]);
	}

	sum.reduce_add()
}

fn normalize(kernel: &mut [f32; TAPS]) {
	let energy = kernel.iter().map(|value| value * value).sum::<f32>().sqrt();
	if energy > 1e-6 {
		let scale = (TAPS as f32).sqrt() / energy;
		for value in kernel {
			*value *= scale;
		}
	}
}

/// The point curve at `x`, with the same bend shape the editors use.
fn sample_frame(points: &[FilterPoint], x: f32) -> f32 {
	if points.len() < 2 {
		return 0.5;
	}

	if x <= points[0].time {
		return points[0].value;
	}

	for pair in points.windows(2) {
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

	points.last().map(|point| point.value).unwrap_or(0.5)
}

impl Effect<2> for FilterTableEffect {
	fn delay(&self) -> usize {
		if self.phase == 1 { TAPS / 2 } else { 0 }
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
			|| self.phase != self.seen_phase
			|| self.morph != self.seen_morph
		{
			self.rebuild();
		}

		let latency = self.delay();
		let dry_length = TAPS / 2 + 1;

		let mut dry = [0.0f32; 2];
		for (channel, sample) in samples.iter().enumerate() {
			self.dry[channel][self.dry_write] = *sample;
			dry[channel] = if latency == 0 {
				*sample
			} else {
				self.dry[channel][(self.dry_write + dry_length - latency) % dry_length]
			};
			self.history[channel][self.write] = *sample;
			self.history[channel][self.write + TAPS] = *sample;
		}

		let mut wet = [0.0f32; 2];
		let blend = self.blend;
		let start = (self.write + 1) % TAPS;

		for (channel, wet_sample) in wet.iter_mut().enumerate() {
			let history = &self.history[channel][start..start + TAPS];

			if blend >= 1.0 {
				*wet_sample = dot(history, &self.kernel_rev);
			} else {
				let old = dot(history, &self.previous_rev);
				let next = dot(history, &self.next_rev);
				*wet_sample = old + (next - old) * blend;
			}
		}

		self.blend = (blend + 1.0 / 128.0).min(1.0);
		self.dry_write = (self.dry_write + 1) % dry_length;

		let mix = self.mix.clamp(0.0, 1.0);
		for (channel, sample) in samples.iter_mut().enumerate() {
			*sample = dry[channel] * (1.0 - mix) + wet[channel] * mix;
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

	#[test]
	fn linear_phase_centres_the_kernel_and_reports_its_latency() {
		let frame = FilterFrame::new();
		let mut effect = FilterTableEffect::with_frame(48_000, frame);
		effect.phase = 1;
		effect.rebuild();

		assert_eq!(effect.delay(), TAPS / 2);
		assert!(effect.kernel.iter().all(|value| value.is_finite()));

		for offset in 1..TAPS / 2 {
			let left = effect.kernel[(TAPS / 2 + offset) % TAPS];
			let right = effect.kernel[(TAPS / 2 - offset + TAPS) % TAPS];
			assert!(
				(left - right).abs() < 1e-3,
				"offset {offset}: {left} vs {right}"
			);
		}
	}

	#[test]
	fn minimum_phase_is_causal() {
		let frame = FilterFrame::new();
		let mut effect = FilterTableEffect::with_frame(48_000, frame);
		effect.phase = 2;
		effect.rebuild();

		assert_eq!(effect.delay(), 0);
		let front: f32 = effect.kernel[..TAPS / 2]
			.iter()
			.map(|value| value * value)
			.sum();
		let back: f32 = effect.kernel[TAPS / 2..]
			.iter()
			.map(|value| value * value)
			.sum();
		assert!(
			front > back,
			"energy is not front-loaded: {front} vs {back}"
		);
	}

	#[test]
	fn a_one_sample_kernel_delays_by_one() {
		let frame = FilterFrame::new();
		let mut effect = FilterTableEffect::with_frame(48_000, frame);
		effect.mix = 1.0;
		effect.blend = 1.0;
		effect.kernel = [0.0; TAPS];
		effect.kernel[1] = 1.0;
		effect.kernel_rev = [0.0; TAPS];
		effect.kernel_rev[TAPS - 2] = 1.0;
		effect.previous = effect.kernel;
		effect.previous_rev = effect.kernel_rev;
		effect.next = effect.kernel;
		effect.next_rev = effect.kernel_rev;
		effect.seen_generation = effect.frame.generation();
		effect.seen_cutoff = effect.cutoff;
		effect.seen_resonance = effect.resonance;
		effect.seen_phase = effect.phase;
		let mut context: Box<dyn ProcessContext> = Box::new(TestContext(ProcessInfos {
			sample_rate: 48_000,
			..ProcessInfos::default()
		}));
		let mut first = [1.0f32, 1.0f32];
		let mut second = [0.0f32; 2];

		effect.process(&mut first, &[], &mut context);
		effect.process(&mut second, &[], &mut context);

		assert!((second[0] - 1.0).abs() < 1e-5, "{second:?}");
		assert!((second[1] - 1.0).abs() < 1e-5, "{second:?}");
	}

	#[test]
	fn morph_moves_between_pages() {
		let frame = FilterFrame::new();
		frame.set_pages(vec![
			vec![
				FilterPoint {
					time: 0.0,
					value: 0.0,
					bend: 0.0,
				},
				FilterPoint {
					time: 1.0,
					value: 1.0,
					bend: 0.0,
				},
			],
			vec![
				FilterPoint {
					time: 0.0,
					value: 1.0,
					bend: 0.0,
				},
				FilterPoint {
					time: 1.0,
					value: 0.0,
					bend: 0.0,
				},
			],
		]);
		let mut effect = FilterTableEffect::with_frame(48_000, frame);
		effect.morph = 0.0;
		effect.rebuild();
		let first = effect.kernel;
		effect.morph = 1.0;
		effect.rebuild();
		let second = effect.kernel;

		assert_ne!(first, second);
	}
}
