//! Classic distortion effects.

use std::f32::consts::PI;

use i_am_dsp_derive::Parameters;

use crate::{prelude::MIN_FREQUENCY, Effect, ProcessContext};

/// A Hard Clipper which clamps the signal between -threshold and threshold.
#[derive(Parameters)]
pub struct HardClipper {
	/// saves in linear scale
	#[unit(percent)]
	#[id(display = "Threshold")]
	#[layout(group = "Drive", order = 0, widget = knob, hero = "transfer")]
	pub threshold: f32,
	#[range(min = 0.01, max = 4.0)]
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
	fn process(&mut self, input: &mut [f32; CHANNELS], _: &[&[f32; CHANNELS]], _: &mut Box<dyn ProcessContext>) {
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
	#[range(min = 0.01, max = 4.0)]
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
	fn process(&mut self, input: &mut [f32; CHANNELS], _: &[&[f32; CHANNELS]], _: &mut Box<dyn ProcessContext>) {
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

		ui.add(egui::Slider::new(&mut self.alpha, 0.01..=1.0).text("Alpha").logarithmic(true));
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
	#[range(min = 0.01, max = 4.0)]
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
	#[range(min = 0.01, max = 4.0)]
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
	fn process(&mut self, input: &mut [f32; CHANNELS], _: &[&[f32; CHANNELS]], _: &mut Box<dyn ProcessContext>) {
		for val in input.iter_mut() {
			*val = 2.0 / (1.0 + (- *val * self.k).exp()) - 1.0;
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

		ui.add(egui::Slider::new(&mut self.k, 0.01..=10.0).text("k").logarithmic(true));
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
	#[range(min = 0.01, max = 4.0)]
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
	fn process(&mut self, input: &mut [f32; CHANNELS], _: &[&[f32; CHANNELS]], _: &mut Box<dyn ProcessContext>) {
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

		ui.add(egui::Slider::new(&mut self.bits, 1..=128).text("Bits").logarithmic(true));
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
	#[range(min = 0.01, max = 4.0)]
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
	fn process(&mut self, input: &mut [f32; CHANNELS], _: &[&[f32; CHANNELS]], _: &mut Box<dyn ProcessContext>) {
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

		ui.add(egui::Slider::new(&mut self.a, 0.01..=5.0).text("a").logarithmic(true));
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
	#[range(min = 0.01, max = 4.0)]
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

	fn process(&mut self, samples: &mut [f32; CHANNELS], _: &[&[f32; CHANNELS]], _: &mut Box<dyn ProcessContext>) {
		if samples.iter().all(|&x| x == 0.0) {
			self.empty_count += 1;
		}else {
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

		ui.add(egui::Slider::new(
			&mut self.target_sample_rate, 
			MIN_FREQUENCY..=self.sample_rate as f32
		).text("Target Sample Rate").logarithmic(true));
		gain_ui(ui, &mut self.gain, None, false);
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

