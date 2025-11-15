//! A helper to choose between two audio inputs or outputs.
//! 
//! Useful for effects like compressor

use crate::prelude::{Parameter, Parameters};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// A helper to choose between two audio inputs or outputs.
pub enum AudioIoChooser {
	/// Choose the current input or output.
	#[default] Current,
	/// Choose the other input or output.
	Other(usize)
}

impl AudioIoChooser {
	#[inline(always)]
	/// Choose the appropriate input or output.
	/// 
	/// Will not faithfully return the other input or output if the index is out of bounds.
	pub fn choose<'a, const CHANNELS: usize>(
		&self, 
		samples: &'a [f32; CHANNELS], 
		other: &'a [&[f32; CHANNELS]]
	) -> &'a [f32; CHANNELS] {
		match self {
			AudioIoChooser::Current => samples,
			AudioIoChooser::Other(i) => {
				other.get(*i).unwrap_or(&samples)
			}
		}
	}
}

impl Parameters for AudioIoChooser {
	fn get_parameters(&self) -> Vec<Parameter> {
		vec![Parameter {
			identifier: "AudioIoChooser".to_string(),
			value: crate::prelude::Value::Int { 
				value: match self {
					AudioIoChooser::Current => 0,
					AudioIoChooser::Other(i) => (*i + 1) as i32,
				}, 
				range: 0..=1023,
				logarithmic: false,
			}
		}]
	}

	fn set_parameter(&mut self, identifier: &str, value: crate::prelude::SetValue) -> bool {
		if identifier == "AudioIoChooser" && let crate::prelude::SetValue::Int(value)= value {
			*self = match value {
				0 => AudioIoChooser::Current,
				i => AudioIoChooser::Other(i as usize - 1),
			};
			return true;
		}

		false
	}
}