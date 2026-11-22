//! Stereo control effects

use i_am_dsp_derive::Parameters;

use crate::{Effect, ProcessContext};

/// Stereo controller effect
#[derive(Parameters)]
#[default_float_range(min = 0.01, max = 4.0)]
pub struct StereoController {
	#[logarithmic]
	#[default(1.0)]
	#[unit(percent)]
	#[id(display = "Mid")]
	#[layout(group = "Width", order = 0, widget = knob, hero = "stereo")]
	/// saves in linear scale
	pub mid_gain: f32,
	#[logarithmic]
	#[default(1.0)]
	#[unit(percent)]
	#[id(display = "Side")]
	#[layout(group = "Width", order = 1, widget = knob)]
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

	/// Mid and side gains: two multiplications and no state, so silence comes out.
	fn is_idle(&self, input_is_silent: bool) -> bool {
		input_is_silent
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"Stereo Controller"
	}

	fn process(
		&mut self,
		samples: &mut [f32; 2],
		_: &[&[f32; 2]],
		_: &mut Box<dyn ProcessContext>,
	) {
		let mid = (samples[0] + samples[1]) / 2.0;
		let side = (samples[0] - samples[1]) / 2.0;
		samples[0] = mid * self.mid_gain + side * self.side_gain;
		samples[1] = mid * self.mid_gain - side * self.side_gain;
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, _: String) {
		use crate::tools::ui_tools::gain_ui;
		gain_ui(ui, &mut self.mid_gain, Some("Mid Gain".to_string()), false);
		gain_ui(
			ui,
			&mut self.side_gain,
			Some("Side Gain".to_string()),
			false,
		);
	}
}

#[derive(Debug, Clone, Copy)]
/// Gain effect
#[derive(Parameters)]
#[default_float_range(min = 0.01, max = 4.0)]
pub struct Gain {
	#[logarithmic]
	#[unit(percent)]
	#[id(display = "Gain")]
	#[layout(group = "Gain", order = 0, widget = knob, size = large, hero = "waveform")]
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

	/// A gain of nothing is nothing, and a gain keeps no memory of what it multiplied:
	/// the wrapper can leave this one out of a silent block.
	fn is_idle(&self, input_is_silent: bool) -> bool {
		input_is_silent
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"Gain"
	}

	fn process(
		&mut self,
		samples: &mut [f32; CHANNELS],
		_: &[&[f32; CHANNELS]],
		_: &mut Box<dyn ProcessContext>,
	) {
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
#[derive(Parameters)]
#[default_float_range(min = 0.01, max = 4.0)]
pub struct LrControl {
	#[logarithmic]
	#[default(1.0)]
	#[unit(percent)]
	#[id(display = "Left")]
	#[layout(group = "Balance", order = 0, widget = knob, hero = "balance")]
	/// saves in linear scale
	pub left_gain: f32,
	#[logarithmic]
	#[default(1.0)]
	#[unit(percent)]
	#[id(display = "Right")]
	#[layout(group = "Balance", order = 1, widget = knob)]
	/// saves in linear scale
	pub right_gain: f32,
}

impl Default for LrControl {
	fn default() -> Self {
		Self {
			left_gain: 1.0,
			right_gain: 1.0,
		}
	}
}

impl Effect<2> for LrControl {
	fn delay(&self) -> usize {
		0
	}

	/// One multiplication per channel, and no state at all.
	fn is_idle(&self, input_is_silent: bool) -> bool {
		input_is_silent
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"LR Controller"
	}

	fn process(
		&mut self,
		samples: &mut [f32; 2],
		_: &[&[f32; 2]],
		_: &mut Box<dyn ProcessContext>,
	) {
		samples[0] *= self.left_gain;
		samples[1] *= self.right_gain;
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, _: String) {
		use crate::tools::ui_tools::gain_ui;
		gain_ui(
			ui,
			&mut self.left_gain,
			Some("Left Gain".to_string()),
			false,
		);
		gain_ui(
			ui,
			&mut self.right_gain,
			Some("Right Gain".to_string()),
			false,
		);
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::Effect;

	/// The wrapper skips a block it can prove is silent, so these effects have to say
	/// when they may be left out: silence in is silence out, and nothing is remembered
	/// between samples that could keep ringing.
	#[test]
	fn a_gain_is_idle_exactly_when_its_input_is() {
		let gain = Gain::default();
		assert!(<Gain as Effect<2>>::is_idle(&gain, true));
		assert!(!<Gain as Effect<2>>::is_idle(&gain, false));

		let balance = LrControl::default();
		assert!(<LrControl as Effect<2>>::is_idle(&balance, true));
		assert!(!<LrControl as Effect<2>>::is_idle(&balance, false));

		let width = StereoController::default();
		assert!(<StereoController as Effect<2>>::is_idle(&width, true));
		assert!(!<StereoController as Effect<2>>::is_idle(&width, false));
	}
}
