//! Classic distortion effects.

use std::f32::consts::PI;

use i_am_dsp_derive::Parameters;

use crate::{
	Effect, ProcessContext,
	prelude::MIN_FREQUENCY,
	tools::oversampling::{LATENCY, Oversampler},
};

/// A Hard Clipper which clamps the signal between -threshold and threshold.
#[derive(Parameters)]
pub struct HardClipper {
	/// saves in linear scale
	#[unit(percent)]
	#[id(display = "Threshold")]
	#[layout(group = "Drive", order = 0, widget = knob, hero = "transfer")]
	pub threshold: f32,
	#[range(min = 0.0, max = 4.0)]
	#[logarithmic]
	#[unit(percent)]
	#[id(display = "Output")]
	/// saves in linear scale
	#[layout(group = "Drive", order = 1, widget = knob)]
	pub gain: f32,
}

impl HardClipper {
	/// Creates a new HardClipper with the given threshold and gain.
	pub fn new(threshold: f32, gain: f32) -> Self {
		Self { threshold, gain }
	}
}

impl<const CHANNELS: usize> Effect<CHANNELS> for HardClipper {
	fn process(
		&mut self,
		input: &mut [f32; CHANNELS],
		_: &[&[f32; CHANNELS]],
		_: &mut Box<dyn ProcessContext>,
	) {
		for val in input.iter_mut() {
			*val = val.clamp(-self.threshold, self.threshold) * self.gain;
		}
	}

	fn delay(&self) -> usize {
		0
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"Hard Clipper"
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, _: String) {
		use crate::tools::ui_tools::gain_ui;

		ui.add(egui::Slider::new(&mut self.threshold, 0.0..=1.0).text("Threshold"));
		gain_ui(ui, &mut self.gain, None, false);
	}
}

/// A Soft Clipper which saturates the signal at the threshold.
///
/// The function is defined as: $f(x) := \tanh(x / \alpha)$
#[derive(Parameters)]
pub struct SoftClipper {
	/// a parameter in the range [0, 1]
	pub alpha: f32,
	/// The output gain, saved in linear scale.
	#[range(min = 0.0, max = 4.0)]
	#[logarithmic]
	pub gain: f32,
}

impl SoftClipper {
	/// Creates a new SoftClipper with the given alpha and gain.
	pub fn new(alpha: f32, gain: f32) -> Self {
		Self { alpha, gain }
	}
}

impl<const CHANNELS: usize> Effect<CHANNELS> for SoftClipper {
	fn process(
		&mut self,
		input: &mut [f32; CHANNELS],
		_: &[&[f32; CHANNELS]],
		_: &mut Box<dyn ProcessContext>,
	) {
		for val in input.iter_mut() {
			*val = (*val / self.alpha).tanh() * self.gain;
		}
	}

	fn delay(&self) -> usize {
		0
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"Soft Clipper"
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, _: String) {
		use crate::tools::ui_tools::gain_ui;

		ui.add(
			egui::Slider::new(&mut self.alpha, 0.01..=1.0)
				.text("Alpha")
				.logarithmic(true),
		);
		gain_ui(ui, &mut self.gain, None, false);
	}
}

/// A clipper with two shapes: a hard one and a soft one.
///
/// The two used to be separate effects with different controls, which made a
/// "switch the algorithm" control impossible without changing the parameter set.
/// Here the amount means the same thing in both modes -- how hard the signal is
/// driven into the shaper -- so a mode switch can keep its knobs.
#[derive(Parameters)]
pub struct Clipper {
	/// Hard or soft.
	#[range(min = 0, max = 1)]
	#[choices("Hard", "Soft")]
	#[default(0)]
	#[id(display = "Mode")]
	#[layout(group = "Drive", order = 0, widget = choice)]
	pub mode: i32,

	/// How hard the signal is driven into the shaper.
	#[range(min = 0.01, max = 1.0)]
	#[unit(percent)]
	#[id(display = "Amount")]
	#[layout(group = "Drive", order = 1, widget = knob, hero = "transfer")]
	pub threshold: f32,

	/// The gain after the shaper.
	#[range(min = 0.0, max = 4.0)]
	#[logarithmic]
	#[unit(percent)]
	#[id(display = "Output")]
	#[layout(group = "Drive", order = 2, widget = knob)]
	pub gain: f32,
}

impl Clipper {
	/// A clipper in the given mode, amount and output gain.
	pub fn new(mode: i32, threshold: f32, gain: f32) -> Self {
		Self {
			mode,
			threshold,
			gain,
		}
	}

	/// The shape one sample gets, before the output gain.
	pub fn shape(&self, sample: f32) -> f32 {
		match self.mode {
			1 => (sample / self.threshold.max(0.01)).tanh(),
			_ => sample.clamp(-self.threshold, self.threshold),
		}
	}
}

impl<const CHANNELS: usize> Effect<CHANNELS> for Clipper {
	fn process(
		&mut self,
		input: &mut [f32; CHANNELS],
		_: &[&[f32; CHANNELS]],
		_: &mut Box<dyn ProcessContext>,
	) {
		for value in input.iter_mut() {
			*value = self.shape(*value) * self.gain;
		}
	}

	fn delay(&self) -> usize {
		0
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"Clipper"
	}
}

/// An overdrive effect.
///
/// The function is defined as: $f(x) := 2 / (1 + e^{-k x}) - 1$
#[derive(Parameters)]
pub struct Overdrive {
	#[range(min = 0.01, max = 10.0)]
	/// The k parameter of the overdrive function.
	pub k: f32,
	#[range(min = 0.0, max = 4.0)]
	#[logarithmic]
	/// The output gain, saved in linear scale.
	pub gain: f32,
}

impl Overdrive {
	/// Creates a new Overdrive with the given k and gain.
	pub fn new(k: f32, gain: f32) -> Self {
		Self { k, gain }
	}
}

impl<const CHANNELS: usize> Effect<CHANNELS> for Overdrive {
	fn process(
		&mut self,
		input: &mut [f32; CHANNELS],
		_: &[&[f32; CHANNELS]],
		_: &mut Box<dyn ProcessContext>,
	) {
		for val in input.iter_mut() {
			*val = 2.0 / (1.0 + (-*val * self.k).exp()) - 1.0;
			*val *= self.gain;
		}
	}

	fn delay(&self) -> usize {
		0
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"Overdrive"
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, _: String) {
		use crate::tools::ui_tools::gain_ui;

		ui.add(
			egui::Slider::new(&mut self.k, 0.01..=10.0)
				.text("k")
				.logarithmic(true),
		);
		gain_ui(ui, &mut self.gain, None, false);
	}
}

/// A Bit Crusher effect.
///
/// The function is defined as: $f(x) := \operatorname{round}(x * bits) / bits$
#[derive(Parameters)]
pub struct BitCrusher {
	#[range(min = 1, max = 128)]
	/// The number of bits to keep in the output.
	pub bits: usize,
	#[range(min = 0.0, max = 4.0)]
	#[logarithmic]
	/// The output gain, saved in linear scale.
	pub gain: f32,
}

impl BitCrusher {
	/// Creates a new BitCrusher with the given bits and gain.
	pub fn new(bits: usize, gain: f32) -> Self {
		Self { bits, gain }
	}
}

impl<const CHANNELS: usize> Effect<CHANNELS> for BitCrusher {
	fn process(
		&mut self,
		input: &mut [f32; CHANNELS],
		_: &[&[f32; CHANNELS]],
		_: &mut Box<dyn ProcessContext>,
	) {
		for val in input.iter_mut() {
			*val = (self.bits as f32 * *val).round() / (self.bits as f32);
			*val *= self.gain;
		}
	}

	fn delay(&self) -> usize {
		0
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"Bit Crusher"
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, _: String) {
		use crate::tools::ui_tools::gain_ui;

		ui.add(
			egui::Slider::new(&mut self.bits, 1..=128)
				.text("Bits")
				.logarithmic(true),
		);
		gain_ui(ui, &mut self.gain, None, false);
	}
}

/// A Saturator effect.
///
/// The function is defined as: $f(x) := \frac{x}{1 + \frac{|x|}{a}^p} \cdot (1 + \frac{1}{a}^p)$
#[derive(Parameters)]
pub struct Saturator {
	#[range(min = 0.01, max = 5.0)]
	/// a parameter controlling the slope of the saturating function
	pub a: f32,
	#[range(min = 1.0, max = 10.0)]
	/// a parameter controlling the power of the saturating function
	pub p: f32,
	#[range(min = 0.0, max = 4.0)]
	#[logarithmic]
	/// saves in linear scale
	pub gain: f32,
}

impl Saturator {
	/// Creates a new Saturator with the given a, p, and gain.
	pub fn new(a: f32, p: f32, gain: f32) -> Self {
		Self { a, p, gain }
	}
}

/// The saturating curve used by [`Saturator`]:
/// `f(x) = x / (1 + (|x| / a)^p) * (1 + 1 / a^p)`.
///
/// It is exposed on its own so that an effect whose curve is a special case of
/// it (see [`Crusher`](crate::effects::crusher::Crusher), which is this curve
/// with `p = 2` and `a` driven by its amount) shares the implementation instead
/// of copying the formula.
pub fn saturate(sample: f32, a: f32, p: f32) -> f32 {
	let factor = 1.0 + 1.0 / a.powf(p);
	let norm = (sample.abs() / a).powf(p);
	(sample / (1.0 + norm)) * factor
}

impl<const CHANNELS: usize> Effect<CHANNELS> for Saturator {
	fn process(
		&mut self,
		input: &mut [f32; CHANNELS],
		_: &[&[f32; CHANNELS]],
		_: &mut Box<dyn ProcessContext>,
	) {
		for val in input.iter_mut() {
			*val = saturate(*val, self.a, self.p) * self.gain;
		}
	}

	fn delay(&self) -> usize {
		0
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"Saturator"
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, _: String) {
		use crate::tools::ui_tools::gain_ui;

		ui.add(
			egui::Slider::new(&mut self.a, 0.01..=5.0)
				.text("a")
				.logarithmic(true),
		);
		ui.add(egui::Slider::new(&mut self.p, 1.0..=10.0).text("p"));
		gain_ui(ui, &mut self.gain, None, false);
	}
}

const EMPTY_COUNT_THRESHOLD: usize = 50;

/// A Downsampler effect.
#[derive(Parameters)]
pub struct Downsampler<const CHANNELS: usize = 2> {
	#[range(min = 10.0, max = 192000.0)]
	target_sample_rate: f32,
	#[skip]
	sample_rate: usize,
	/// The output gain, saved in linear scale.
	#[range(min = 0.0, max = 4.0)]
	#[logarithmic]
	pub gain: f32,
	#[skip]
	history: [f32; CHANNELS],
	#[skip]
	output: [f32; CHANNELS],
	#[skip]
	phase: f32,
	#[skip]
	empty_count: usize,
}

impl<const CHANNELS: usize> Downsampler<CHANNELS> {
	/// Creates a new Downsampler with the given sample rate.
	pub fn new(sample_rate: usize) -> Self {
		Self {
			target_sample_rate: sample_rate as f32,
			sample_rate,
			gain: 1.0,
			output: [0.0; CHANNELS],
			history: [0.0; CHANNELS],
			phase: 2.0 * PI,
			empty_count: 0,
		}
	}

	/// Sets the target sample rate for the downsampler.
	///
	/// Will clamp the target sample rate to the range [10.0, sample_rate].
	pub fn set_target_sample_rate(&mut self, sample_rate: f32) {
		self.target_sample_rate = sample_rate.clamp(MIN_FREQUENCY, self.sample_rate as f32);
	}

	/// Returns the target sample rate for the downsampler.
	pub fn target_sample_rate(&self) -> f32 {
		self.target_sample_rate
	}
}

impl<const CHANNELS: usize> Effect<CHANNELS> for Downsampler<CHANNELS> {
	fn delay(&self) -> usize {
		0
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"Downsampler"
	}

	fn process(
		&mut self,
		samples: &mut [f32; CHANNELS],
		_: &[&[f32; CHANNELS]],
		_: &mut Box<dyn ProcessContext>,
	) {
		if samples.iter().all(|&x| x == 0.0) {
			self.empty_count += 1;
		} else {
			self.empty_count = 0;
		}

		let phase_increment = 2.0 * PI * self.target_sample_rate / self.sample_rate as f32;
		self.phase += phase_increment;

		if self.phase > 2.0 * PI {
			self.phase = self.phase % 2.0 * PI;
			let t = self.phase / (2.0 * PI);
			for (i, sample) in samples.iter().enumerate().take(CHANNELS) {
				self.output[i] = (1.0 - t) * self.history[i] + t * *sample;
			}
		}

		self.history = *samples;
		if self.empty_count > EMPTY_COUNT_THRESHOLD {
			self.output = [0.0; CHANNELS];
			self.phase = 2.0 * PI;
		}
		*samples = self.output;
		samples.iter_mut().for_each(|x| *x *= self.gain);
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, _: String) {
		use crate::tools::ui_tools::gain_ui;

		ui.add(
			egui::Slider::new(
				&mut self.target_sample_rate,
				MIN_FREQUENCY..=self.sample_rate as f32,
			)
			.text("Target Sample Rate")
			.logarithmic(true),
		);
		gain_ui(ui, &mut self.gain, None, false);
	}
}
/// A distortion with every shaper from this module behind one mode switch.
///
/// The algorithms have different controls — a clipper has a threshold, a
/// saturator has a curve, a bit crusher has a bit count — so the faceplate shows
/// one set at a time. The DSP keeps them all: a parameter that the current mode
/// does not read is simply ignored.
#[derive(Parameters)]
pub struct Distortion<const CHANNELS: usize = 2> {
	/// Which shaper is active.
	#[range(min = 0, max = 5)]
	#[choices("Hard", "Soft", "Overdrive", "Saturator", "Bitcrush", "Downsample")]
	#[default(0)]
	#[id(display = "Mode")]
	#[layout(group = "Shape", order = 0, widget = choice)]
	pub mode: i32,

	/// The clip level of the hard shaper, as a fraction of full scale.
	#[range(min = 0.01, max = 1.0)]
	#[unit(percent)]
	#[default(0.8)]
	#[id(display = "Threshold")]
	#[layout(group = "Shape", order = 1, widget = knob, hero = "transfer")]
	pub threshold: f32,

	/// How soft the soft shaper is; smaller is harder.
	#[range(min = 0.01, max = 1.0)]
	#[unit(percent)]
	#[default(0.5)]
	#[id(display = "Softness")]
	#[layout(group = "Shape", order = 2, widget = knob)]
	pub alpha: f32,

	/// The slope of the overdrive curve.
	#[range(min = 0.01, max = 10.0)]
	#[default(2.0)]
	#[id(display = "Drive")]
	#[layout(group = "Shape", order = 3, widget = knob)]
	pub k: f32,

	/// The knee of the saturator.
	#[range(min = 0.01, max = 5.0)]
	#[default(1.0)]
	#[id(display = "Amount")]
	#[layout(group = "Shape", order = 4, widget = knob)]
	pub a: f32,

	/// The power of the saturator curve.
	#[range(min = 1.0, max = 10.0)]
	#[default(2.0)]
	#[id(display = "Curve")]
	#[layout(group = "Shape", order = 5, widget = knob)]
	pub p: f32,

	/// How many steps the bit crusher keeps.
	#[range(min = 1, max = 128)]
	#[default(8)]
	#[id(display = "Bits")]
	#[layout(group = "Shape", order = 6, widget = knob)]
	pub bits: usize,

	/// The rate the downsampler holds its samples at.
	#[range(min = 10.0, max = 192000.0)]
	#[logarithmic]
	#[unit(hertz)]
	#[default(48000.0)]
	#[id(display = "Rate")]
	#[layout(group = "Shape", order = 7, widget = knob)]
	pub target_sample_rate: f32,

	/// The gain after the shaper.
	#[range(min = 0.0, max = 4.0)]
	#[logarithmic]
	#[unit(percent)]
	#[default(1.0)]
	#[id(display = "Output")]
	#[layout(group = "Shape", order = 8, widget = knob)]
	pub gain: f32,

	#[skip]
	sample_rate: usize,
	#[skip]
	history: [f32; CHANNELS],
	#[skip]
	hold: [f32; CHANNELS],
	#[skip]
	phase: f32,
	#[skip]
	empty_count: usize,
	#[skip]
	last_mode: i32,
	#[skip]
	oversampler: Oversampler<CHANNELS>,
	/// The delay line the sample-rate crushers run through, so every mode has
	/// the same timing as the oversampled ones.
	#[skip]
	align: [[f32; CHANNELS]; LATENCY],
	#[skip]
	align_write: usize,
}

impl<const CHANNELS: usize> Distortion<CHANNELS> {
	/// A hard clipper at 80 %, which is where a fresh distortion starts.
	pub fn new(sample_rate: usize) -> Self {
		Self {
			mode: 0,
			threshold: 0.8,
			alpha: 0.5,
			k: 2.0,
			a: 1.0,
			p: 2.0,
			bits: 8,
			target_sample_rate: sample_rate as f32,
			gain: 1.0,
			sample_rate,
			history: [0.0; CHANNELS],
			hold: [0.0; CHANNELS],
			phase: 2.0 * PI,
			empty_count: 0,
			last_mode: 0,
			oversampler: Oversampler::new(),
			align: [[0.0; CHANNELS]; LATENCY],
			align_write: 0,
		}
	}

	/// The downsampler keeps one sample until its fractional phase wraps.
	fn downsample(&mut self, samples: &mut [f32; CHANNELS]) {
		if samples.iter().all(|value| *value == 0.0) {
			self.empty_count += 1;
		} else {
			self.empty_count = 0;
		}

		// The parameter is a rate in hertz and may be set above what the host is
		// actually running; above the host rate there is nothing to hold.
		let target = self.target_sample_rate.clamp(10.0, self.sample_rate as f32);
		let increment = 2.0 * PI * target / self.sample_rate as f32;
		self.phase += increment;

		if self.phase > 2.0 * PI {
			self.phase %= 2.0 * PI;
			let fraction = self.phase / (2.0 * PI);

			for (index, sample) in samples.iter().enumerate() {
				self.hold[index] = (1.0 - fraction) * self.history[index] + fraction * *sample;
			}
		}

		self.history = *samples;

		if self.empty_count > EMPTY_COUNT_THRESHOLD {
			self.hold = [0.0; CHANNELS];
			self.phase = 2.0 * PI;
		}

		*samples = self.hold;
	}
}

impl<const CHANNELS: usize> Effect<CHANNELS> for Distortion<CHANNELS> {
	fn delay(&self) -> usize {
		// The shaper runs four times the host rate, and the filters around it
		// are linear phase, so the host has to pull the result back by this much.
		LATENCY
	}

	fn process(
		&mut self,
		samples: &mut [f32; CHANNELS],
		_: &[&[f32; CHANNELS]],
		_: &mut Box<dyn ProcessContext>,
	) {
		if self.mode != self.last_mode {
			// A mode change starts the sample-and-hold clean rather than with the
			// last value of whatever mode came before.
			self.last_mode = self.mode;
			self.phase = 2.0 * PI;
			self.hold = [0.0; CHANNELS];
			self.empty_count = 0;
		}

		// Every mode runs through the same delay, so switching between them does
		// not move the effect against the rest of the graph.
		let input = *samples;
		let aligned = self.align[self.align_write];
		self.align[self.align_write] = input;
		self.align_write = (self.align_write + 1) % LATENCY;

		if self.mode >= 4 {
			// The crushers are sample-rate effects; filtering their hold or their
			// steps away would remove the sound they exist for.
			let mut frame = aligned;

			match self.mode {
				4 => {
					let steps = self.bits.max(1) as f32;

					for value in frame.iter_mut() {
						*value = (steps * *value).round() / steps;
					}
				}
				_ => self.downsample(&mut frame),
			}

			for value in frame.iter_mut() {
				*value *= self.gain;
			}

			*samples = frame;
			return;
		}

		let mode = self.mode;
		let threshold = self.threshold;
		let alpha = self.alpha.max(0.01);
		let k = self.k;
		let a = self.a.max(0.01);
		let p = self.p;

		let mut shape = |frame: [f32; CHANNELS]| {
			let mut shaped = frame;

			match mode {
				1 => {
					for value in shaped.iter_mut() {
						*value = (*value / alpha).tanh();
					}
				}
				2 => {
					for value in shaped.iter_mut() {
						*value = 2.0 / (1.0 + (-*value * k).exp()) - 1.0;
					}
				}
				3 => {
					for value in shaped.iter_mut() {
						*value = saturate(*value, a, p);
					}
				}
				_ => {
					for value in shaped.iter_mut() {
						*value = value.clamp(-threshold, threshold);
					}
				}
			}

			shaped
		};

		let mut shaped = self.oversampler.process(input, &mut shape);

		for value in shaped.iter_mut() {
			*value *= self.gain;
		}

		*samples = shaped;
	}

	fn clear_tail(&mut self) {
		self.oversampler.clear();
		self.align = [[0.0; CHANNELS]; LATENCY];
		self.align_write = 0;
		self.history = [0.0; CHANNELS];
		self.hold = [0.0; CHANNELS];
		self.phase = 2.0 * PI;
		self.empty_count = 0;
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"Distortion"
	}
}

#[cfg(test)]
mod clipper_tests {
	use super::*;

	/// The two modes are not the same curve with a different name.
	#[test]
	fn the_modes_shape_a_hot_sample_differently() {
		let hard = Clipper::new(0, 0.5, 1.0);
		let soft = Clipper::new(1, 0.5, 1.0);

		assert_eq!(hard.shape(2.0), 0.5);
		assert!((soft.shape(2.0) - (2.0_f32 / 0.5).tanh()).abs() < 1e-6);
		assert!(soft.shape(2.0) > 0.5);
	}

	/// The output gain is applied after the shape, as its name says.
	#[test]
	fn the_output_gain_follows_the_shape() {
		let mut clipper = Clipper::new(0, 0.5, 2.0);
		let mut frame = [1.0_f32; 2];
		let mut context: Box<dyn ProcessContext> = Box::new(());

		clipper.process(&mut frame, &[], &mut context);

		assert_eq!(frame, [1.0, 1.0]);
	}
}

#[cfg(test)]
mod distortion_tests {
	use super::*;

	fn shaped(mode: i32, sample: f32) -> f32 {
		let mut effect = Distortion::<1>::new(48_000);
		effect.mode = mode;

		let mut context: Box<dyn ProcessContext> = Box::new(());
		let mut output = sample;

		// A shaper inside an oversampler has to settle; the first frames out are
		// the filter's start-up, not the mode's shape.
		for _ in 0..1_024 {
			let mut frame = [sample];

			effect.process(&mut frame, &[], &mut context);
			output = frame[0];
		}

		output
	}

	/// The six shapers are not six names for the same curve.
	#[test]
	fn each_mode_uses_its_own_shape() {
		assert!(
			(shaped(0, 2.0) - 0.8).abs() < 1e-6,
			"the hard clipper honours its threshold"
		);
		assert!(
			(shaped(1, 2.0) - (2.0_f32 / 0.5).tanh()).abs() < 1e-5,
			"the soft clipper uses its softness"
		);
		assert!(shaped(2, 2.0) > 0.9, "overdrive is a saturating curve");
		assert!(shaped(3, 2.0) > 0.0, "saturator is finite and positive");
		assert!((shaped(4, 0.33) - 0.375).abs() < 1e-6, "bitcrush quantises");
		assert!(
			shaped(5, 0.5).is_finite(),
			"downsample starts from silence, not NaN"
		);
	}

	/// A downsample rate above the host rate has nothing to hold, so it is clamped
	/// to the host rate instead of running the hold phase backwards.
	#[test]
	fn a_downsample_rate_above_the_host_is_clamped() {
		fn render(target: f32) -> Vec<f32> {
			let mut effect = Distortion::<1>::new(48_000);
			effect.mode = 5;
			effect.target_sample_rate = target;

			let mut context: Box<dyn ProcessContext> = Box::new(());
			let mut output = Vec::with_capacity(512);

			for index in 0..512 {
				let input = (core::f32::consts::TAU * 440.0 * index as f32 / 48_000.0).sin();
				let mut frame = [input];
				effect.process(&mut frame, &[], &mut context);
				output.push(frame[0]);
			}

			output
		}

		assert_eq!(render(48_000.0), render(192_000.0));
	}

	/// Four-times shaping keeps the clip's fold-back out of the audible band.
	#[test]
	fn the_hard_clip_aliases_less_than_a_direct_clip() {
		fn magnitude(samples: &[f32], frequency: f32) -> f32 {
			let mut real = 0.0;
			let mut imaginary = 0.0;

			for (index, sample) in samples.iter().enumerate() {
				let phase = core::f32::consts::TAU * frequency * index as f32 / 48_000.0;
				real += sample * phase.cos();
				imaginary -= sample * phase.sin();
			}

			(real * real + imaginary * imaginary).sqrt() / samples.len() as f32
		}

		let mut distortion = Distortion::<1>::new(48_000);
		distortion.mode = 0;
		distortion.threshold = 0.5;

		let mut context: Box<dyn ProcessContext> = Box::new(());
		let mut shaped = Vec::new();
		let mut direct = Vec::new();

		for index in 0..4_800 {
			let input = (core::f32::consts::TAU * 5_000.0 * index as f32 / 48_000.0).sin();
			let mut frame = [input];

			distortion.process(&mut frame, &[], &mut context);
			shaped.push(frame[0]);
			direct.push(input.clamp(-0.5, 0.5));
		}

		// The fifth harmonic of 5 kHz folds back onto 13 kHz at 48 kHz.
		let skip = 512;
		let alias = magnitude(&shaped[skip..], 13_000.0);
		let raw_alias = magnitude(&direct[skip..], 13_000.0);

		assert!(alias < raw_alias * 0.5, "{alias} against {raw_alias}");
	}

	/// The sample-rate crushers run on the delayed input, so every mode lands at
	/// the same time.
	#[test]
	fn the_crushers_land_at_the_oversampler_latency() {
		let mut distortion = Distortion::<1>::new(48_000);
		distortion.mode = 4;
		distortion.bits = 32;

		let mut context: Box<dyn ProcessContext> = Box::new(());
		let mut output = Vec::new();

		for index in 0..96 {
			let mut frame = [if index == 0 { 1.0 } else { 0.0 }];

			distortion.process(&mut frame, &[], &mut context);
			output.push(frame[0]);
		}

		let peak = output
			.iter()
			.enumerate()
			.max_by(|left, right| left.1.abs().partial_cmp(&right.1.abs()).unwrap())
			.map(|(index, _)| index)
			.unwrap();

		assert_eq!(peak, LATENCY);
	}
}
