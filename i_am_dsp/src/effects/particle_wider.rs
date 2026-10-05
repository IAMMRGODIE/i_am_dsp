//! A spectral widener that scatters the magnitudes of the partials of a signal.
//!
//! This is the DSP of the `i_am_particle_wider` plug-in: the signal is
//! transformed frame by frame, every bin whose frequency is inside a band has
//! its magnitude multiplied by a random factor, and the frames are transformed
//! back and overlap added. The transform, the windowing, the overlap add and
//! the phase handling are the library's [`PhaseVocoder`] with a
//! [`ParticleRandomizer`] mapper; the gains are [`Gain`] effects and the
//! mid/side stage the plug-in applies afterwards is the library's
//! [`StereoController`].

use i_am_dsp_derive::Parameters;
use rand::random_range;

use crate::{
	effects::{
		phase_vocoder::{overlap_add_gain, FrequencyMapper, PhaseVocoder},
		stereo_control::{Gain, StereoController},
	},
	Effect, ProcessContext,
};

/// The window size the plug-in starts with, in samples.
const DEFAULT_WINDOW_SIZE: usize = 2048;

/// The frequency mapper behind [`ParticleWider`].
///
/// Every bin keeps its frequency, and a bin whose frequency lies inside the
/// band between [`Self::low_frequency`] and [`Self::high_frequency`] has its
/// magnitude multiplied by a random factor: the partials of the signal are
/// scattered independently, which decorrelates them and widens the sound. The
/// phase is kept as it was measured, because the mapper does not move the bins
/// and the overlapping frames would otherwise not add back up.
#[derive(Parameters)]
pub struct ParticleRandomizer {
	/// How far a magnitude may deviate, from 0 to 1.
	#[range(min = 0.0, max = 1.0)]
	pub depth: f32,
	/// The lowest frequency whose magnitude is randomized, in Hz.
	#[range(min = 1.0, max = 30000.0)]
	#[logarithmic]
	pub low_frequency: f32,
	/// The highest frequency whose magnitude is randomized, in Hz.
	#[range(min = 1.0, max = 30000.0)]
	#[logarithmic]
	pub high_frequency: f32,
	/// Whether a magnitude may be inverted as well, which flips the phase of the bin.
	pub allow_negative: bool,
}

impl Default for ParticleRandomizer {
	fn default() -> Self {
		Self {
			depth: 0.0,
			low_frequency: 1.0,
			high_frequency: 30000.0,
			allow_negative: false,
		}
	}
}

impl FrequencyMapper for ParticleRandomizer {
	fn map_frequency(&mut self, frequency: f32, amplitude: f32) -> (f32, f32) {
		if frequency <= self.low_frequency || frequency >= self.high_frequency {
			return (frequency, amplitude);
		}

		let factor = if self.allow_negative {
			random_range(-self.depth..=self.depth) * 2.0 + 1.0 - self.depth
		} else {
			1.0 + random_range(-self.depth..=self.depth)
		};

		(frequency, amplitude * factor)
	}

	fn synthesis_phase(&mut self, _: f32, analysis_phase: f32, _: f32) -> f32 {
		analysis_phase
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, _: String) {
		use egui::Slider;

		ui.add(Slider::new(&mut self.depth, 0.0..=1.0).text("Depth"));
		ui.add(Slider::new(&mut self.low_frequency, 1.0..=30000.0).text("Low Frequency (Hz)").logarithmic(true));
		ui.add(Slider::new(&mut self.high_frequency, 1.0..=30000.0).text("High Frequency (Hz)").logarithmic(true));
		ui.checkbox(&mut self.allow_negative, "Allow Negative Magnitudes");
	}
}

/// A spectral widener, stereo only, because of its mid/side stage.
#[derive(Parameters)]
pub struct ParticleWider {
	/// The gain applied before the widener.
	#[sub_param]
	pub input_gain: Gain,
	/// The mid/side controller applied after the widener.
	#[sub_param]
	pub stereo_control: StereoController,
	/// The gain applied after the widener.
	#[sub_param]
	pub output_gain: Gain,
	/// The phase vocoder that scatters the partials.
	#[sub_param]
	pub vocoder: PhaseVocoder<ParticleRandomizer, 2>,
}

impl ParticleWider {
	/// Creates a new particle wider for the given sample rate.
	pub fn new(sample_rate: usize) -> Self {
		let vocoder = PhaseVocoder::new(ParticleRandomizer::default(), DEFAULT_WINDOW_SIZE, sample_rate);
		let mut wider = Self {
			input_gain: Gain::new(1.0),
			stereo_control: StereoController::new(),
			output_gain: Gain::new(1.0),
			vocoder,
		};
		wider.compensate_overlap_gain();
		wider
	}

	/// Cancels the overlap add gain of the vocoder for its current window factor.
	///
	/// A depth of zero then passes the signal through unchanged. The gain the
	/// overlap add applies depends on the window factor, so this has to be called
	/// again after changing [`PhaseVocoder::window_factor`], and it overwrites
	/// [`PhaseVocoder::gain`].
	pub fn compensate_overlap_gain(&mut self) {
		self.vocoder.gain = 1.0 / overlap_add_gain(self.vocoder.window_factor);
	}
}

impl Effect<2> for ParticleWider {
	fn delay(&self) -> usize {
		self.vocoder.window_size()
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"Particle Wider"
	}

	fn process(
		&mut self,
		samples: &mut [f32; 2],
		other: &[&[f32; 2]],
		process_context: &mut Box<dyn ProcessContext>,
	) {
		self.input_gain.process(samples, other, process_context);
		self.vocoder.process(samples, other, process_context);
		self.stereo_control.process(samples, other, process_context);
		self.output_gain.process(samples, other, process_context);
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		use crate::tools::ui_tools::gain_ui;

		self.vocoder.demo_ui(ui, format!("{}_vocoder", id_prefix));
		self.stereo_control.demo_ui(ui, format!("{}_stereo", id_prefix));
		gain_ui(ui, &mut self.input_gain.gain, Some("Input Gain".to_string()), false);
		gain_ui(ui, &mut self.output_gain.gain, Some("Output Gain".to_string()), false);
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn root_mean_square(samples: &[f32]) -> f32 {
		(samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len() as f32).sqrt()
	}

	fn run(wider: &mut ParticleWider, input: &[f32], window_size: usize) -> Vec<f32> {
		let mut context: Box<dyn crate::ProcessContext> = Box::new(());
		let mut output = Vec::with_capacity(input.len());
		for sample in input {
			let mut frame = [*sample, *sample];
			wider.process(&mut frame, &[], &mut context);
			output.push(frame[0]);
		}

		// The overlapping frames delay the signal by one window.
		output.split_off(window_size)
	}

	fn sine(frequency: f32, sample_rate: f32, len: usize) -> Vec<f32> {
		(0..len)
			.map(|i| (2.0 * std::f32::consts::PI * frequency * i as f32 / sample_rate).sin() * 0.5)
			.collect()
	}

	#[test]
	fn a_depth_of_zero_passes_the_signal_through() {
		let sample_rate = 48000.0;
		let input = sine(1000.0, sample_rate, 8192 * 4);

		for window_factor in [0.0, 0.25, 0.54, 0.75, 1.0] {
			let mut wider = ParticleWider::new(48000);
			wider.vocoder.window_factor = window_factor;
			wider.compensate_overlap_gain();

			let window_size = wider.vocoder.window_size();
			let output = run(&mut wider, &input, window_size);
			let gain = root_mean_square(&output) / root_mean_square(&input[..output.len()]);
			assert!(
				(gain - 1.0).abs() < 0.02,
				"window factor {window_factor}: the signal came out at {gain} times its level",
			);

			// The frames delay the signal by exactly one window, and the phase is
			// kept, so the samples come back one by one.
			for (i, sample) in output.iter().enumerate() {
				assert!(
					(sample - input[i]).abs() < 0.001,
					"window factor {window_factor}: sample {i} came out as {sample}, not {}",
					input[i],
				);
			}
		}
	}

	#[test]
	fn the_parameters_are_reachable_by_their_identifiers() {
		use crate::prelude::{Parameters, SetValue};

		let mut wider = ParticleWider::new(48000);
		assert!(wider.set_parameter("vocoder.mapper.depth", SetValue::Float(0.75)));
		assert_eq!(wider.vocoder.mapper.depth, 0.75);
		assert!(wider.set_parameter("vocoder.mapper.allow_negative", SetValue::Bool(true)));
		assert!(wider.vocoder.mapper.allow_negative);
		assert!(wider.set_parameter("stereo_control.side_gain", SetValue::Float(1.5)));
		assert_eq!(wider.stereo_control.side_gain, 1.5);
		assert!(wider.set_parameter("input_gain.gain", SetValue::Float(1.25)));
		assert_eq!(wider.input_gain.gain, 1.25);
	}

	#[test]
	fn the_randomization_changes_the_partials() {
		let sample_rate = 48000.0;
		let input = sine(1000.0, sample_rate, 8192 * 4);

		let mut wider = ParticleWider::new(48000);
		wider.vocoder.mapper.depth = 1.0;
		let window_size = wider.vocoder.window_size();
		let output = run(&mut wider, &input, window_size);

		let difference: f32 = output
			.iter()
			.zip(input.iter())
			.map(|(output, input)| (output - input).abs())
			.sum();
		assert!(difference > 1.0, "the randomized signal is identical to the input");
		assert!(output.iter().all(|sample| sample.is_finite()), "the output is not finite");
	}
}
