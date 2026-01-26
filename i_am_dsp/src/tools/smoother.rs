//! Smoother for rapidly changing signals.

use i_am_dsp_derive::Parameters;

/// Double time constant smoother.
#[derive(Parameters)]
#[default_float_range(min = 0.0, max = 4000.0)]
#[default_int_range(min = 0, max = 2147483647)]
pub struct DoubleTimeConstant<const CHANNELS: usize = 2> {
	/// Attack time in milliseconds.
	pub attack_time: f32,
	/// Release time in milliseconds.
	pub release_time: f32,
	#[skip]
	sample_rate: usize,
	#[skip]
	value: [f32; CHANNELS],
}

impl<const CHANNELS: usize> DoubleTimeConstant<CHANNELS> {
	/// Create a new smoother with the given attack and release times in milliseconds,
	pub fn new(
		attack_time: f32, 
		release_time: f32, 
		default_value: f32,
		sample_rate: usize,
	) -> Self {
		Self {
			attack_time,
			release_time,
			value: [default_value; CHANNELS],
			sample_rate,
		}
	}

	/// Input a new value to the smoother
	/// 
	/// to get smoothed result, call [`Self::get_smoothed_result()`]
	pub fn input_value(&mut self, input_value: &[f32; CHANNELS]) {
		let sample_rate = self.sample_rate as f32;
		let attack_factor = 1.0 - (-1.0 / (self.attack_time * sample_rate / 1000.0)).exp();
		let release_factor = 1.0 - (-1.0 / (self.release_time * sample_rate / 1000.0)).exp();
		for (input_value, value) in input_value.iter().zip(self.value.iter_mut()) {
			if input_value < value {
				*value = attack_factor * input_value + (1.0 - attack_factor) * *value;
			}else {
				*value = (1.0 - release_factor) * *value + release_factor * input_value;
			}
		}
	}

	/// Get the smoothed result of the last input value.
	pub fn get_smoothed_result(&self) -> [f32; CHANNELS] {
		self.value
	}
}