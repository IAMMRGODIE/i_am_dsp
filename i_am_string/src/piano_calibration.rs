//! The piano, measured against The Gentleman at the same frequencies.
//!
//! `tools/loop_calibration.py` fills this in. It is empty because neither an
//! open-loop correction nor a damped iteration brought the measured partial
//! levels closer: the level projected at a frequency is carried by the body and
//! the neighbouring modes as much as by the mode itself, so the map from a
//! mode's level to the measurement is too flat to invert. Matching the library
//! would mean building the mode bank from the measurement instead.

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