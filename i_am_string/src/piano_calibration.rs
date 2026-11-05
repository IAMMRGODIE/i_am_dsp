//! The piano's modes, as measured against The Gentleman.
//!
//! `tools/build_calibration.py` fills this in. It is deliberately empty while
//! the measurement is still being worked out: a table whose rows all carry the
//! same clamped decay is worse than no table at all. With no rows the mode
//! bank falls back to the fitted loss law and the bridge, exactly as before.

/// One calibrated mode.
#[derive(Clone, Copy, Debug)]
pub struct Mode {
	/// Harmonic order the mode belongs to.
	pub order: f32,
	/// Multiplier on the mode's level, relative to the note.
	pub gain: f32,
	/// T60 of the prompt decay, in seconds.
	pub prompt: f32,
	/// T60 of the tail, in seconds.
	pub tail: f32,
}

/// The calibrated modes for one note, empty until the note is measured.
pub fn modes(_note: u8) -> &'static [Mode] {
	&[]
}