//! Physical modelling of plucked, struck and bowed strings.
//!
//! The crate is built around one observation: a plucked, a struck and a bowed
//! string are *the same vibrating object*. They differ only in the constitutive
//! law of the contact at the excitation point:
//!
//! | excitation | contact law |
//! |---|---|
//! | plucked | unilateral displacement constraint (a penalty spring) |
//! | struck  | unilateral non-linear felt force, F = K * delta^p |
//! | bowed   | velocity dependent friction, F = N * mu(v_bow - v) |
//!
//! Everything else - the dispersive lossy string, the terminations, the body -
//! is shared. The contact is solved as a scalar implicit equation at a
//! scattering port, which is the only place where the three instruments diverge.
//!
//! # Two ways to be a string
//!
//! The string itself comes in two models, and [core::Core] chooses between them.
//!
//! A **waveguide** is two delay lines and a filter that has to realise the
//! string's dispersion and loss. It is cheap and its low partials are exact. A
//! **mode bank** is one damped oscillator per partial, so
//! `f_n = n f_0 sqrt(1 + B n^2)` is not something it approximates, it is
//! something it *is*.
//!
//! That distinction is the whole reason both exist. A stiff string's partials are
//! stretched, and the waveguide's stretch is a curve fitted through an allpass
//! cascade. Measured on rendered output at C7 of a grand piano, where the
//! stretch reaches a fifth of a tone, the waveguide's partials come out up to
//! **45 cents** from where the physics says they are, while the mode bank's come
//! out within **0.2 cents**. A string of mistuned partials does not sound stiff;
//! it sounds like a detuned saw, and no amount of filtering fixes it.
//!
//! The mode bank pays for that. It is about eight times the cost of the
//! waveguide per string, because it is a hundred and sixty real oscillators
//! where the waveguide is a filter standing in for them, and it models a bounded
//! number of partials where the delay lines model everything below Nyquist.
//! Switching between the two is not a quality setting - it is a different
//! instrument, and the two do not agree about which notes sound best.
//!
//! # Layout
//!
//! * [modal] - the mode bank string core: one damped oscillator per partial,
//!   exact in tuning at any stiffness.
//! * [string] - the waveguide string core: two delay lines, a dispersion
//!   allpass cascade and a frequency dependent loss filter.
//! * [core] - the [core::Core] switch between the two, and the [core::StringCore]
//!   wrapper that hides the choice from everything above it.
//! * [contact] - the [contact::ContactLaw] trait and its three implementations.
//! * [instrument] - string scaling, voice specs and instrument presets.
//! * [engine] - polyphony, note lifecycle and the render loop.
//!
//! # Example
//!
//!     use i_am_string::prelude::*;
//!
//!     let mut engine = StringEngine::new(48_000.0);
//!     engine.set_preset(Preset::Piano);
//!     engine.note_on(60, 0.8);
//!     let sample = engine.next_sample();

#![warn(missing_docs)]

mod body;
mod delay;
mod dispersion;

pub mod contact;
pub mod core;
pub mod engine;
pub mod modal;
pub mod sympathetic;
pub mod instrument;
pub mod string;

pub mod prelude {
    //! Everything that is needed to build and play an engine.
    pub use crate::contact::*;
    pub use crate::core::{Core, StringCore};
    pub use crate::engine::*;
    pub use crate::instrument::*;
    pub use crate::string::*;
    pub use crate::sympathetic::SympatheticBank;
}

/// A tiny deterministic noise source.
///
/// The engine must not pull in a random number crate, and reproducible noise
/// keeps the tests deterministic, so a xorshift generator is inlined here.
#[derive(Clone, Debug)]
pub(crate) struct Rng(u32);

impl Rng {
    pub(crate) const fn new(seed: u32) -> Self {
        Self(if seed == 0 { 0x9E37_79B9 } else { seed })
    }

    #[inline]
    pub(crate) fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }

    /// A uniform sample in [-1.0, 1.0].
    #[inline]
    pub(crate) fn bipolar(&mut self) -> f32 {
        (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}
