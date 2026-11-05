//! The piano, measured against The Gentleman at the same frequencies.
//!
//! `tools/table_from_sample.py` fills this in from the library. It is empty
//! here on purpose: the measurement path works and `ModalString::recalibrate`
//! builds a bank straight from these rows, but the table only covers the
//! twenty-two notes the library samples, and a piano that is calibrated in one
//! octave and physical in the next is worse than either.

/// One measured partial.
#[derive(Clone, Copy, Debug)]
pub struct Mode {
	/// Where the partial sits, in hertz.
	pub frequency: f32,
	/// The partial's level, relative to the note.
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