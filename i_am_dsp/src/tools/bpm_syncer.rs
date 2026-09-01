//! A helper to translate BPM(maybe dynamic) from host to time signature

use crate::parameters::Parameters;

/// A helper to translate BPM(maybe dynamic) from host to time signature
pub struct BpmSyncer {
	current_signature: f32,
	sample_rate: usize
}

impl BpmSyncer {
	/// Create a new BpmSyncer with the given sample rate
	pub fn new(sample_rate: usize) -> Self {
		Self {
			current_signature: 0.0,
			sample_rate
		}
	}

	/// Update the current signature by one sample
	pub fn next(&mut self, current_bpm: f32) {
		self.next_k(current_bpm, 1);
	}

	/// Update the current signature by the given number of samples
	pub fn next_k(&mut self, current_bpm: f32, samples: usize) {
		let beat_per_second = current_bpm / 60.0;
		let beat_per_sample = beat_per_second / self.sample_rate as f32;
		self.current_signature += beat_per_sample * samples as f32;
	}

	/// Read the current signature
	pub fn read(&self) -> f32 {
		self.current_signature
	}
}

impl Parameters for BpmSyncer {
	fn get_parameters(&self) -> Vec<crate::prelude::Parameter> {
		vec![]
	}

	fn set_parameter(&mut self, _: &str, _: crate::prelude::SetValue) -> bool {
		false
	}

	fn set_parameter_by_index(&mut self, _: usize, _: crate::prelude::SetValue) -> bool {
		false
	}
}
