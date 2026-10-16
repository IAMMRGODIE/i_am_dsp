//! Who is holding Shift, and who is holding Ctrl.
//!
//! iced sends keyboard events to the focused widget only, and mouse events carry
//! no modifiers at all, so a knob cannot learn about Shift from the events it
//! receives. The application subscribes once — see [crate::modifiers_subscription]
//! — which stores the modifiers here, and every control reads them when it needs
//! them.

use std::sync::atomic::{AtomicU32, Ordering};

use iced::keyboard::Modifiers;

/// The modifier bits last seen, as [Modifiers] stores them.
static PRESSED: AtomicU32 = AtomicU32::new(0);

/// Remembers which modifiers are down.
pub fn set_modifiers(modifiers: Modifiers) {
	PRESSED.store(modifiers.bits(), Ordering::Relaxed);
}

/// The modifiers that were down the last time they changed.
///
/// Empty until the application subscribes to [crate::modifiers_subscription],
/// which is the honest answer for a window that has never been told.
pub fn modifiers() -> Modifiers {
	Modifiers::from_bits_truncate(PRESSED.load(Ordering::Relaxed))
}
