//! The piano, measured against The Gentleman at the same frequencies.
//!
//! `tools/build_calibration.py` fills this in. It is empty for now because the
//! first closed loop did not improve the match: the projection at a frequency
//! carries the body and the neighbouring modes, so scaling one mode's level
//! does not scale that projection one for one. The measurement itself is sound
//! and lives in `tools/calibrate_same.py`.

/// One measured partial.
#[derive(Clone, Copy, Debug)]
pub struct Mode {
	/// Where the partial sits, in hertz.
	pub frequency: f32,
	/// Multiplier on the level of the mode that lands on it.
	pub gain: f32,
	/// T60 of the prompt decay, in seconds.
	pub prompt: f32,
	/// T60 of the tail, in seconds.
	pub tail: f32,
}

/// The rows for one note, empty until the note is measured.
pub fn modes(_note: u8) -> &'static [Mode] {
	&[]
}