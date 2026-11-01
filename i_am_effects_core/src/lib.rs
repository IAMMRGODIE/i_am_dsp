//! Effect wrappers and the module catalog shared by the effects bundle and the rack.
//!
//! The catalog is the one place that knows a module's product faceplate metadata:
//! name, colour, hero, mode, header choices and per-mode visible controls. The rack
//! builds its cards from a descriptor and the effect's own parameter metadata, so
//! adding a module here is enough to expose it in both places.

pub mod catalog;
pub mod pitch;
pub mod reverb;

pub use pitch::{PITCH_VISIBLE, PitchEffect};
pub use reverb::ReverbEffect;
