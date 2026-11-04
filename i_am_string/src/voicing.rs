//! The colour a note keeps after the string has been set moving.
//!
//! A mode bank is one damped oscillator per partial, and that is a *single*
//! exponential per partial. A real string is not so tidy. A struck or plucked
//! partial is not one line but a close pair - the two polarisations of the
//! string, and the board's own reaction, split them by a few cents - so the
//! note blooms and beats instead of standing still. A piano string decays in
//! two stages, a prompt sound that loads the board and an aftersound the board
//! gives back. A wound string carries a metallic layer of its own on top of the
//! harmonic series, and the finger or plectrum has a finite width, which softens
//! the very top of the attack.
//!
//! These are the numbers that describe those four things. They are colour, not
//! physics: each is a small, documented departure from the ideal string, every
//! one defaults to off, and the waveguide core ignores the whole struct because
//! it has one loop gain where the mode bank has an oscillator per partial.

/// How the modal core colours a note after it has been excited.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Voicing {
	/// The ratio of the late decay rate to the early one.
	///
	/// One is a single exponential and the value the ideal string has. Below
	/// one the note starts at its own decay rate and, after
	/// [`Voicing::late_delay`], relaxes towards this fraction of it. Measured
	/// on a real piano, the aftersound of the lowest strings is about half as
	/// fast as the prompt sound, so a value near 0.7 is the useful range; the
	/// model caps the resulting T60 at thirty seconds, because a note that
	/// rings for a minute is a synthesiser pad, not a string.
	pub late_decay: f32,
	/// How long the early decay lasts before the tail takes over, in seconds.
	pub late_delay: f32,
	/// How long the change takes, in seconds.
	///
	/// Deliberately not instant. A crossfade shorter than a tenth of a second
	/// is heard as a kink in the envelope, and the physical hand-over between
	/// the prompt sound and the aftersound is a slow one.
	pub late_ramp: f32,
	/// The level of the extra, slightly detuned partial, relative to the
	/// harmonic it doubles.
	///
	/// A few percent is enough to hear. The pair beats at `f *
	/// doublet_detune` hertz, which for the fundamental of a guitar low E is a
	/// slow 0.3 Hz swell, and for its twentieth partial a 5 Hz roughness.
	pub doublet: f32,
	/// How far the extra partial is detuned, as a fraction of its frequency.
	///
	/// 0.003 is a little over five cents, which is the spread measured between
	/// the two polarisations of a real string. Much more and the pair is heard
	/// as two notes rather than one.
	pub doublet_detune: f32,
	/// The strength of the metallic layer a wound string adds, 0..1.
	///
	/// The layer is a handful of partials that sit off the harmonic grid and
	/// die within a second. It has no pitch of its own at these levels; what
	/// it adds is the short metallic edge a wound low string has and a plain
	/// one does not.
	pub winding: f32,
	/// How much the pluck's finite contact width darkens the attack.
	///
	/// The initial condition of the modal core is a triangle, whose spectrum
	/// is `1/n^2` and therefore has far more top than a finger has. This
	/// multiplies partial `n` by `exp(-tilt * (n - 1))`, which is the
	/// exponential the top of a plucked spectrum actually falls off with over
	/// the first octave or so. Zero is the ideal triangle.
	pub pluck_tilt: f32,
}

impl Default for Voicing {
	fn default() -> Self {
		Self {
			late_decay: 1.0,
			late_delay: 0.35,
			late_ramp: 1.2,
			doublet: 0.0,
			doublet_detune: 0.003,
			winding: 0.0,
			pluck_tilt: 0.0,
		}
	}
}
