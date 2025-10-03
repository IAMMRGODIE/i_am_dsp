//! A helper to choose between two audio inputs or outputs.
//! 
//! Useful for effects like compressor

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