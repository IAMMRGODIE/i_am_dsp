//! A wavetable used as a filter: one frame becomes a short FIR kernel, and the
//! cutoff moves the frame's twenty-fourth harmonic.

use std::sync::Arc;

use i_am_dsp_derive::Parameters;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;
use wide::f32x4;

use crate::prelude::{page_position, segment_bend};
use crate::tools::curve_pages::{CurvePages, PageSnapshot};
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
	pages: CurvePages<FilterPoint>,
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
		self.pages.pages()
	}

	/// A shared snapshot. The audio thread reads this once per rebuild.
	fn snapshot(&self) -> Arc<PageSnapshot<FilterPoint>> {
		self.pages.snapshot()
	}

	/// Moves whenever the pages change.
	pub fn generation(&self) -> u64 {
		self.pages.generation()
	}

	/// Replaces the only page.
	pub fn set_points(&self, points: Vec<FilterPoint>) {
		self.set_pages(vec![points]);
	}

	/// Replaces one page and keeps the rest.
	pub fn set_page(&self, index: usize, points: Vec<FilterPoint>) {
		let mut pages = self.pages();
		if pages.is_empty() {
			pages.push(default_points());
		}
		let index = index.min(pages.len() - 1);
		pages[index] = points;
		self.set_pages(pages);
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

		self.pages.replace(pages);
	}
}

impl Default for FilterFrame {
	fn default() -> Self {
		Self {
			pages: CurvePages::new(default_points()),
		}
	}
}

fn default_points() -> Vec<FilterPoint> {
	(0..=64)
		.map(|index| {
			let time = index as f32 / 64.0;
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
	/// Whether the kernel comes from the frame or from the sidechain input.
	#[choices("Table", "Sidechain")]
	#[default(0)]
	#[id(display = "Mode")]
	#[layout(group = "Filter", order = 5, widget = choice)]
	pub mode: i32,
	/// What the sidechain becomes: its magnitude response, or the kernel itself.
	#[choices("Response", "Kernel")]
	#[default(0)]
	#[id(display = "Sidechain")]
	#[layout(group = "Filter", order = 6, widget = choice)]
	pub sidechain_mode: i32,
	#[skip]
	frame: Arc<FilterFrame>,
	#[skip]
	sidechain: [f32; TAPS],
	#[skip]
	sidechain_write: usize,
	#[skip]
	sidechain_count: usize,
	#[skip]
	seen_mode: i32,
	#[skip]
	spectrum: [Complex<f32>; TAPS],
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
	low_kernel_rev: [f32; TAPS],
	#[skip]
	low_previous_rev: [f32; TAPS],
	#[skip]
	low_next_rev: [f32; TAPS],
	#[skip]
	low_history: [Vec<f32>; 2],
	#[skip]
	low_write: usize,
	#[skip]
	low_count: usize,
	#[skip]
	low_sum: [f32; 2],
	#[skip]
	low_out: [f32; 2],
	#[skip]
	low_previous_out: [f32; 2],
	#[skip]
	low_interp: f32,
	#[skip]
	low_lp: [[f32; 2]; 2],
	#[skip]
	low_alpha: f32,
	#[skip]
	fade: f32,
	#[skip]
	fade_step: f32,
	#[skip]
	low_initialized: bool,
	#[skip]
	seen_factor: usize,
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
			low_kernel_rev: [0.0; TAPS],
			low_previous_rev: [0.0; TAPS],
			low_next_rev: [0.0; TAPS],
			low_history: core::array::from_fn(|_| vec![0.0; TAPS * 2]),
			low_write: 0,
			low_count: 0,
			low_sum: [0.0; 2],
			low_out: [0.0; 2],
			low_previous_out: [0.0; 2],
			low_interp: 0.0,
			low_lp: [[0.0; 2]; 2],
			low_alpha: 0.0,
			fade: 1.0,
			fade_step: 1.0 / (sample_rate as f32 * 0.03).max(1.0),
			low_initialized: false,
			seen_factor: 0,
			dry: core::array::from_fn(|_| vec![0.0; TAPS / 2 * 8 + 1]),
			dry_write: 0,
			history: core::array::from_fn(|_| vec![0.0; TAPS * 2]),
			write: 0,
			seen_generation: u64::MAX,
			seen_cutoff: f32::NAN,
			seen_resonance: f32::NAN,
			seen_phase: i32::MIN,
			seen_morph: f32::NAN,
			mode: 0,
			sidechain_mode: 0,
			sidechain: [0.0; TAPS],
			sidechain_write: 0,
			sidechain_count: 0,
			seen_mode: 0,
			spectrum: [Complex::new(0.0, 0.0); TAPS],
		};
		effect.rebuild();
		effect
	}

	/// The frame this effect reads.
	pub fn frame(&self) -> Arc<FilterFrame> {
		self.frame.clone()
	}

	fn factor(&self) -> usize {
		if self.mode == 1 {
			// The sidechain path has no cutoff to decimate for.
			return 1;
		}

		if self.cutoff < 120.0 {
			8
		} else if self.cutoff < 240.0 {
			4
		} else if self.cutoff < 480.0 {
			2
		} else {
			1
		}
	}

	fn build_target(&self, snapshot: &PageSnapshot<FilterPoint>, rate: f32) -> [f32; TAPS] {
		let mut raw = [0.0f32; TAPS];
		let pages = snapshot.pages();
		let (from, to, blend) = page_position(self.morph, pages.len());
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

		// A quiet frame must not make a quiet filter: normalise its peak before
		// the spectral work, then normalise the kernel's energy below.
		let peak = dense
			.iter()
			.fold(0.0f32, |peak, sample| peak.max(sample.abs()));
		if peak > 1e-6 {
			for sample in &mut dense {
				*sample /= peak;
			}
		}

		// The frame's twenty-fourth harmonic sits at 24 * rate / FRAME; scaling
		// the read position by this ratio puts it on the cutoff.
		let ratio = self.cutoff * FRAME as f32 / (24.0 * rate);
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
			value *= 1.0 + peak * (std::f32::consts::TAU * self.cutoff * index as f32 / rate).cos();
			*kernel = value;
		}

		normalize(&mut raw);

		match self.phase {
			1 => self.linear(&raw),
			2 => self.minimum(&raw),
			_ => raw,
		}
	}

	fn rebuild(&mut self) {
		let snapshot = self.frame.snapshot();
		self.seen_generation = snapshot.generation();
		self.seen_cutoff = self.cutoff;
		self.seen_resonance = self.resonance;
		self.seen_phase = self.phase;
		self.seen_morph = self.morph;
		let factor = self.factor();
		if factor != self.seen_factor && self.initialized {
			self.fade = 0.0;
		}
		self.seen_factor = factor;

		if self.seen_factor == 1 {
			let target = self.build_target(&snapshot, self.sample_rate);
			self.apply_target(target);
		} else {
			let rate = self.sample_rate / self.seen_factor as f32;
			let target = self.build_target(&snapshot, rate);

			if !self.low_initialized {
				self.low_initialized = true;
				self.low_previous_rev = reverse(&target);
				self.low_kernel_rev = self.low_previous_rev;
				self.low_next_rev = self.low_previous_rev;
				self.blend = 1.0;
			} else {
				let mix = self.blend;
				for index in 0..TAPS {
					self.low_previous_rev[index] = self.low_previous_rev[index]
						+ (self.low_next_rev[index] - self.low_previous_rev[index]) * mix;
				}
				self.low_next_rev = reverse(&target);
				self.blend = 0.0;
			}

			self.low_kernel_rev = self.low_next_rev;
			self.low_history = core::array::from_fn(|_| vec![0.0; TAPS * 2]);
			self.low_write = 0;
			self.low_count = 0;
			self.low_sum = [0.0; 2];
			self.low_out = [0.0; 2];
			self.low_previous_out = [0.0; 2];
			self.low_interp = 0.0;
			self.low_lp = [[0.0; 2]; 2];
			self.low_alpha = 1.0
				- (-std::f32::consts::TAU * 0.45 * self.sample_rate
					/ (self.sample_rate * self.seen_factor as f32))
					.exp();
		}
	}

	/// Crossfades the kernel to `target` over the effect's fade.
	fn apply_target(&mut self, target: [f32; TAPS]) {
		if !self.initialized {
			self.initialized = true;
			self.previous = target;
			self.kernel = target;
			self.next = target;
			self.blend = 1.0;
		} else {
			// Start the new fade from the kernel that is actually playing, not
			// from the previous target. Otherwise a control move during a fade
			// jumps from the mix to the old target in one sample.
			let mix = self.blend;
			for index in 0..TAPS {
				self.previous[index] =
					self.previous[index] + (self.next[index] - self.previous[index]) * mix;
			}
			self.previous_rev = reverse(&self.previous);
			self.next = target;
			self.blend = 0.0;
		}

		self.kernel = target;
		for index in 0..TAPS {
			self.kernel_rev[index] = target[TAPS - 1 - index];
		}
		self.next_rev = self.kernel_rev;
	}

	/// Captures the sidechain and rebuilds the kernel once per window.
	fn follow_sidechain(&mut self, other: &[&[f32; 2]]) {
		let Some(frame) = other.get(1) else {
			return;
		};

		let mono = (frame[0] + frame[1]) * 0.5;
		self.sidechain[self.sidechain_write] = mono;
		self.sidechain_write = (self.sidechain_write + 1) % TAPS;
		self.sidechain_count += 1;

		if self.sidechain_count < TAPS {
			return;
		}

		self.sidechain_count = 0;

		let target = if self.sidechain_mode == 1 {
			self.sidechain_kernel_target()
		} else {
			self.sidechain_target()
		};

		if let Some(target) = target {
			self.apply_target(target);
		}
	}

	/// The kernel a window of the sidechain asks for: its magnitude, minimum
	/// phase, so the filter stays causal and reports no latency.
	fn sidechain_target(&mut self) -> Option<[f32; TAPS]> {
		let mut energy = 0.0;

		for index in 0..TAPS {
			let sample = self.sidechain[(self.sidechain_write + index) % TAPS];
			let window = 0.5 - 0.5 * (std::f32::consts::TAU * index as f32 / TAPS as f32).cos();
			energy += sample * sample;
			self.spectrum[index] = Complex::new(sample * window, 0.0);
		}

		// Silence has no spectrum to follow; keeping the last kernel is better
		// than fading to nothing every time the sidechain pauses.
		if energy < 1e-9 {
			return None;
		}

		self.forward.process(&mut self.spectrum);

		for value in &mut self.spectrum {
			*value = Complex::new(value.norm(), 0.0);
		}

		self.inverse.process(&mut self.spectrum);

		let mut raw = [0.0f32; TAPS];
		for (index, value) in self.spectrum.iter().enumerate() {
			raw[(index + TAPS / 2) % TAPS] = value.re / TAPS as f32;
		}

		normalize(&mut raw);
		Some(self.minimum(&raw))
	}

	/// The kernel the sidechain's own waveform asks for: the window is the
	/// impulse response, so the main input is convolved with the sidechain.
	fn sidechain_kernel_target(&mut self) -> Option<[f32; TAPS]> {
		let mut raw = [0.0f32; TAPS];
		let mut energy = 0.0;

		for (index, target) in raw.iter_mut().enumerate() {
			let sample = self.sidechain[(self.sidechain_write + index) % TAPS];
			let window = 0.5 - 0.5 * (std::f32::consts::TAU * index as f32 / TAPS as f32).cos();
			energy += sample * sample;
			*target = sample * window;
		}

		// A silent window would make the filter vanish; keep the last kernel.
		if energy < 1e-9 {
			return None;
		}

		normalize(&mut raw);
		Some(raw)
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

fn reverse(kernel: &[f32; TAPS]) -> [f32; TAPS] {
	let mut out = [0.0; TAPS];
	for index in 0..TAPS {
		out[index] = kernel[TAPS - 1 - index];
	}
	out
}

fn normalize(kernel: &mut [f32; TAPS]) {
	let energy = kernel.iter().map(|value| value * value).sum::<f32>().sqrt();
	if energy > 1e-6 {
		// Unit energy, as an IR should be: white noise comes out at the same RMS
		// instead of changing level with the kernel's length or shape.
		let scale = 1.0 / energy;
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
		let shaped = segment_bend(unit, left.bend);

		return left.value + (right.value - left.value) * shaped;
	}

	points.last().map(|point| point.value).unwrap_or(0.5)
}

impl Effect<2> for FilterTableEffect {
	fn delay(&self) -> usize {
		if self.mode == 1 {
			// The sidechain kernel is minimum phase: nothing is held back.
			return 0;
		}

		let factor = self.factor();
		if self.phase == 1 {
			TAPS / 2 * factor
		} else {
			0
		}
	}

	fn process(
		&mut self,
		samples: &mut [f32; 2],
		other: &[&[f32; 2]],
		_context: &mut Box<dyn ProcessContext>,
	) {
		if self.mode != self.seen_mode {
			self.seen_mode = self.mode;
			self.sidechain = [0.0; TAPS];
			self.sidechain_write = 0;
			self.sidechain_count = 0;

			// Coming back to the table has to rebuild from the frame; going to
			// the sidechain waits for its first window and keeps the old kernel
			// until then.
			if self.mode == 0 {
				self.rebuild();
			}
		}

		if self.mode == 1 {
			self.follow_sidechain(other);
		} else if self.frame.generation() != self.seen_generation
			|| self.cutoff != self.seen_cutoff
			|| self.resonance != self.seen_resonance
			|| self.phase != self.seen_phase
			|| self.morph != self.seen_morph
			|| self.factor() != self.seen_factor
		{
			self.rebuild();
		}

		let latency = self.delay();
		let dry_length = TAPS / 2 * 8 + 1;

		let mut dry = [0.0f32; 2];
		for (channel, sample) in samples.iter().enumerate() {
			self.dry[channel][self.dry_write] = *sample;
			dry[channel] = if latency == 0 {
				*sample
			} else {
				self.dry[channel][(self.dry_write + dry_length - latency) % dry_length]
			};
		}
		self.dry_write = (self.dry_write + 1) % dry_length;

		let mix = self.mix.clamp(0.0, 1.0);
		let factor = self.factor();

		if factor == 1 {
			for (channel, sample) in samples.iter().enumerate() {
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

			self.blend = (blend + self.fade_step).min(1.0);

			let fade = self.fade;
			for (channel, sample) in samples.iter_mut().enumerate() {
				*sample = dry[channel] * (1.0 - mix) + wet[channel] * fade * mix;
			}

			self.fade = (fade + self.fade_step).min(1.0);
			self.write = (self.write + 1) % TAPS;
			return;
		}

		for (channel, sample) in samples.iter().enumerate() {
			self.low_lp[0][channel] += self.low_alpha * (*sample - self.low_lp[0][channel]);
			self.low_lp[1][channel] +=
				self.low_alpha * (self.low_lp[0][channel] - self.low_lp[1][channel]);
			self.low_sum[channel] = self.low_lp[1][channel];
		}
		self.low_count += 1;

		if self.low_count >= factor {
			let decimated = self.low_sum;

			for (channel, sample) in decimated.iter().enumerate() {
				self.low_history[channel][self.low_write] = *sample;
				self.low_history[channel][self.low_write + TAPS] = *sample;
			}

			let blend = self.blend;
			let start = (self.low_write + 1) % TAPS;

			self.low_previous_out = self.low_out;
			for channel in 0..2 {
				let history = &self.low_history[channel][start..start + TAPS];
				self.low_out[channel] = if blend >= 1.0 {
					dot(history, &self.low_kernel_rev)
				} else {
					let old = dot(history, &self.low_previous_rev);
					let next = dot(history, &self.low_next_rev);
					old + (next - old) * blend
				};
			}
			self.low_interp = 0.0;

			self.blend = (blend + self.fade_step * factor as f32).min(1.0);
			self.low_write = (self.low_write + 1) % TAPS;
			self.low_count = 0;
			self.low_sum = [0.0; 2];
		}

		self.low_interp = (self.low_interp + 1.0 / factor as f32).min(1.0);
		let fade = self.fade;
		for (channel, sample) in samples.iter_mut().enumerate() {
			let wet = self.low_previous_out[channel]
				+ (self.low_out[channel] - self.low_previous_out[channel]) * self.low_interp;
			*sample = dry[channel] * (1.0 - mix) + wet * fade * mix;
		}
		self.fade = (fade + self.fade_step).min(1.0);
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

	#[test]
	fn a_low_cutoff_runs_the_decimated_path() {
		let frame = FilterFrame::new();
		let mut effect = FilterTableEffect::with_frame(48_000, frame);
		effect.cutoff = 60.0;
		effect.phase = 1;
		effect.rebuild();
		assert_eq!(effect.factor(), 8);
		assert_eq!(effect.delay(), TAPS / 2 * 8);
		let mut context: Box<dyn ProcessContext> = Box::new(TestContext(ProcessInfos {
			sample_rate: 48_000,
			..ProcessInfos::default()
		}));

		for index in 0..10_000 {
			let mut sample = [(index as f32 * 0.01).sin(), 0.0];
			effect.process(&mut sample, &[], &mut context);
			assert!(sample[0].is_finite());
		}
	}

	#[test]
	fn the_kernel_has_unit_energy() {
		let frame = FilterFrame::new();
		let effect = FilterTableEffect::with_frame(48_000, frame);
		let energy = effect.kernel.iter().map(|value| value * value).sum::<f32>();
		assert!((energy - 1.0).abs() < 1e-3, "energy {energy}");
	}

	/// The sidechain's spectrum becomes the kernel's magnitude response.
	#[test]
	fn the_sidechain_shapes_the_kernel() {
		let frame = FilterFrame::new();
		let mut effect = FilterTableEffect::with_frame(48_000, frame);
		effect.mode = 1;
		let mut context: Box<dyn ProcessContext> = Box::new(());

		for index in 0..8_192 {
			let phase = std::f32::consts::TAU * 1_000.0 * index as f32 / 48_000.0;
			let sidechain = [phase.sin() * 0.5; 2];
			let other: [&[f32; 2]; 1] = [&sidechain];
			let mut samples = [0.0f32; 2];

			effect.process(&mut samples, &other, &mut context);
		}

		let magnitude = |frequency: f32| {
			let bin = (frequency * TAPS as f32 / 48_000.0).round() as usize;
			let mut buffer: Vec<Complex<f32>> = effect
				.kernel
				.iter()
				.map(|sample| Complex::new(*sample, 0.0))
				.collect();

			FftPlanner::<f32>::new()
				.plan_fft_forward(TAPS)
				.process(&mut buffer);

			buffer[bin].norm()
		};

		assert!(
			magnitude(1_000.0) > magnitude(4_000.0) * 4.0,
			"1 kHz {}, 4 kHz {}",
			magnitude(1_000.0),
			magnitude(4_000.0)
		);
	}

	/// The sidechain kernel is minimum phase, so it reports no latency even when
	/// the table's phase choice says linear.
	#[test]
	fn the_sidechain_mode_reports_no_latency() {
		let mut effect = FilterTableEffect::with_frame(48_000, FilterFrame::new());
		effect.phase = 1;
		effect.mode = 1;

		assert_eq!(effect.delay(), 0);
	}

	/// A paused sidechain keeps the kernel it had: fading to silence every time
	/// the sidechain rests would pump the mix.
	#[test]
	fn a_silent_sidechain_keeps_the_kernel() {
		let frame = FilterFrame::new();
		let mut effect = FilterTableEffect::with_frame(48_000, frame);
		effect.mode = 1;
		let mut context: Box<dyn ProcessContext> = Box::new(());

		for index in 0..4_096 {
			let phase = std::f32::consts::TAU * 500.0 * index as f32 / 48_000.0;
			let sidechain = [phase.sin() * 0.5; 2];
			let other: [&[f32; 2]; 1] = [&sidechain];
			let mut samples = [0.0f32; 2];

			effect.process(&mut samples, &other, &mut context);
		}

		let before = effect.kernel;

		for _ in 0..4_096 {
			let sidechain = [0.0f32; 2];
			let other: [&[f32; 2]; 1] = [&sidechain];
			let mut samples = [0.0f32; 2];

			effect.process(&mut samples, &other, &mut context);
		}

		assert_eq!(before, effect.kernel);
	}

	/// In kernel mode the sidechain's own waveform is the impulse response.
	#[test]
	fn the_sidechain_kernel_is_the_input_window() {
		let frame = FilterFrame::new();
		let mut effect = FilterTableEffect::with_frame(48_000, frame);
		effect.mode = 1;
		effect.sidechain_mode = 1;
		let mut context: Box<dyn ProcessContext> = Box::new(());

		for index in 0..TAPS {
			let phase = std::f32::consts::TAU * 1_000.0 * index as f32 / 48_000.0;
			let sidechain = [phase.sin() * 0.5; 2];
			let zero = [0.0f32; 2];
			let other: [&[f32; 2]; 2] = [&zero, &sidechain];
			let mut samples = [0.0f32; 2];

			effect.process(&mut samples, &other, &mut context);
		}

		// The window the sidechain filled, hann weighted and normalised.
		let mut expected = [0.0f32; TAPS];
		for (index, value) in expected.iter_mut().enumerate() {
			let phase = std::f32::consts::TAU * 1_000.0 * index as f32 / 48_000.0;
			let window = 0.5 - 0.5 * (std::f32::consts::TAU * index as f32 / TAPS as f32).cos();
			*value = phase.sin() * 0.5 * window;
		}
		normalize(&mut expected);

		for (index, value) in effect.kernel.iter().enumerate() {
			assert!(
				(value - expected[index]).abs() < 1e-5,
				"sample {index}: {value} against {}",
				expected[index]
			);
		}
	}

	/// The two sidechain modes have to sound different: one keeps the phase, the
	/// other drops it.
	#[test]
	fn the_two_sidechain_modes_build_different_kernels() {
		fn kernel(mode: i32) -> [f32; TAPS] {
			let mut effect = FilterTableEffect::with_frame(48_000, FilterFrame::new());
			effect.mode = 1;
			effect.sidechain_mode = mode;
			let mut context: Box<dyn ProcessContext> = Box::new(());

			for index in 0..TAPS {
				let phase = std::f32::consts::TAU * 1_000.0 * index as f32 / 48_000.0;
				let sidechain = [phase.sin() * 0.5; 2];
				let zero = [0.0f32; 2];
				let other: [&[f32; 2]; 2] = [&zero, &sidechain];
				let mut samples = [0.0f32; 2];

				effect.process(&mut samples, &other, &mut context);
			}

			effect.kernel
		}

		let response = kernel(0);
		let direct = kernel(1);
		let difference = response
			.iter()
			.zip(direct.iter())
			.map(|(left, right)| (left - right) * (left - right))
			.sum::<f32>()
			.sqrt();

		assert!(difference > 0.1, "the kernels are too close: {difference}");
	}

	/// Switching into the sidechain keeps the signal finite, table kernel and all.
	#[test]
	fn switching_to_the_sidechain_stays_finite() {
		let frame = FilterFrame::new();
		let mut effect = FilterTableEffect::with_frame(48_000, frame);
		let mut context: Box<dyn ProcessContext> = Box::new(());

		for index in 0..2_048 {
			let phase = std::f32::consts::TAU * 220.0 * index as f32 / 48_000.0;
			let mut samples = [phase.sin() * 0.3; 2];

			effect.process(&mut samples, &[], &mut context);
			assert!(samples[0].is_finite() && samples[1].is_finite());
		}

		effect.mode = 1;

		for index in 0..4_096 {
			let phase = std::f32::consts::TAU * 1_000.0 * index as f32 / 48_000.0;
			let sidechain = [phase.sin() * 0.5; 2];
			let other: [&[f32; 2]; 1] = [&sidechain];
			let mut samples = [0.0f32; 2];

			effect.process(&mut samples, &other, &mut context);
			assert!(
				samples[0].is_finite() && samples[1].is_finite(),
				"non-finite at {index}"
			);
		}
	}
}
