//! Stereo control effects

use crate::{Effect, ProcessContext};

/// Stereo controller effect
pub struct StereoController {
	/// saves in linear scale
	pub mid_gain: f32,
	/// saves in linear scale
	pub side_gain: f32,
}

impl Default for StereoController {
	fn default() -> Self {
		Self::new()
	}
}

impl StereoController {
	/// Creates a new stereo controller with default values
	pub fn new() -> Self {
		Self {
			mid_gain: 1.0,
			side_gain: 1.0,
		}
	}
}

impl Effect<2> for StereoController {
	fn delay(&self) -> usize {
		0
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"Stereo Controller"
	}

	fn process(&mut self, samples: &mut [f32; 2], _: &[&[f32; 2]], _: &mut Box<dyn ProcessContext>) {
		let mid = (samples[0] + samples[1]) / 2.0;
		let side = (samples[0] - samples[1]) / 2.0;
		samples[0] = mid * self.mid_gain + side * self.side_gain;
		samples[1] = mid * self.mid_gain - side * self.side_gain;
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, _: String) {
    	use crate::tools::ui_tools::gain_ui;
		gain_ui(ui, &mut self.mid_gain, Some("Mid Gain".to_string()),false);
		gain_ui(ui, &mut self.side_gain, Some("Side Gain".to_string()),false);
	}
}

#[derive(Debug, Clone, Copy)]
/// Gain effect
pub struct Gain {
	/// saves in linear scale
	pub gain: f32,
}

impl Default for Gain {
	fn default() -> Self {
		Self::new(1.0)
	}
}

impl Gain {
	/// Creates a new gain effect
	pub fn new(gain: f32) -> Self {
		Self { gain }
	}
}

impl<const CHANNELS: usize> Effect<CHANNELS> for Gain {
	fn delay(&self) -> usize {
		0
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"Gain"
	}

	fn process(&mut self, samples: &mut [f32; CHANNELS], _: &[&[f32; CHANNELS]], _: &mut Box<dyn ProcessContext>) {
		for sample in samples.iter_mut() {
			*sample *= self.gain;
		}
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, _: String) {
		use crate::tools::ui_tools::gain_ui;
		gain_ui(ui, &mut self.gain, None, false);
	}
}

#[derive(Debug, Clone, Copy)]
/// L/R controller effect
pub struct LrControl {
	/// saves in linear scale
	pub left_gain: f32,
	/// saves in linear scale
	pub right_gain: f32,
}

impl Default for LrControl {
	fn default() -> Self {
		Self { left_gain: 1.0, right_gain: 1.0 }
	}
}

impl Effect<2> for LrControl {
	fn delay(&self) -> usize {
		0
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"LR Controller"
	}

	fn process(&mut self, samples: &mut [f32; 2], _: &[&[f32; 2]], _: &mut Box<dyn ProcessContext>) {
		samples[0] *= self.left_gain;
		samples[1] *= self.right_gain;
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, _: String) {
		use crate::tools::ui_tools::gain_ui;
		gain_ui(ui, &mut self.left_gain, Some("Left Gain".to_string()), false);
		gain_ui(ui, &mut self.right_gain, Some("Right Gain".to_string()), false);
	}
}

