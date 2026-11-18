//! A handover between the main thread and the audio thread.
//!
//! The audio thread renders with `&mut P` while the main thread may need the same
//! processor for a project save or load. Instead of a lock on the audio thread, the
//! two sides meet at a block boundary: the main thread asks, the audio thread sees
//! the request at the start of its next block, acknowledges it, and leaves the
//! processor alone until the main thread lets go.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// How long the main thread waits for the audio thread to yield.
///
/// One block is at most a few milliseconds; the wait is really for "is the audio
/// thread running at all". When the plugin is not processing, nothing will ever
/// acknowledge, and the main thread has the processor to itself anyway.
const WAIT: Duration = Duration::from_millis(100);

/// Hands the processor over from the audio thread to the main thread.
pub(crate) struct StateGuard {
	/// The main thread wants the processor to itself.
	requested: AtomicBool,
	/// The audio thread has stopped touching it.
	yielded: AtomicBool,
}

impl StateGuard {
	/// A guard with nobody holding the processor.
	pub(crate) fn new() -> Self {
		Self {
			requested: AtomicBool::new(false),
			yielded: AtomicBool::new(false),
		}
	}

	/// Takes the processor for the main thread.
	///
	/// The returned handle releases it when dropped, so a panic in the middle of a
	/// state load cannot leave the audio thread silent forever.
	pub(crate) fn take(&self) -> Handover<'_> {
		self.requested.store(true, Ordering::SeqCst);

		let deadline = Instant::now() + WAIT;

		while !self.yielded.load(Ordering::SeqCst) {
			if Instant::now() >= deadline {
				// Not rendering: there is no other side to wait for.
				break;
			}

			std::thread::yield_now();
		}

		Handover(self)
	}

	/// Whether the audio thread has to stay away from the processor this block.
	///
	/// Called once at the start of a block; a `true` answer means the caller must
	/// render silence and must not touch `P`.
	pub(crate) fn audio_should_yield(&self) -> bool {
		if self.requested.load(Ordering::SeqCst) {
			self.yielded.store(true, Ordering::SeqCst);

			true
		} else {
			false
		}
	}
}

/// The main thread's hold on the processor.
pub(crate) struct Handover<'a>(&'a StateGuard);

impl Drop for Handover<'_> {
	fn drop(&mut self) {
		self.0.yielded.store(false, Ordering::SeqCst);
		self.0.requested.store(false, Ordering::SeqCst);
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// The audio thread yields once and keeps yielding while the handover lasts;
	/// dropping it lets the audio thread render again.
	#[test]
	fn a_handover_holds_the_audio_thread_off() {
		let guard = StateGuard::new();
		assert!(!guard.audio_should_yield());

		let handover = guard.take();
		assert!(guard.audio_should_yield());
		assert!(guard.audio_should_yield());

		drop(handover);
		assert!(!guard.audio_should_yield());
	}

	/// When the audio thread is running, the handover completes at its next block
	/// instead of waiting out the timeout.
	#[test]
	fn a_handover_returns_as_soon_as_the_audio_thread_yields() {
		let guard = std::sync::Arc::new(StateGuard::new());
		let audio = guard.clone();
		let audio_thread = std::thread::spawn(move || {
			while !audio.audio_should_yield() {
				std::thread::yield_now();
			}
		});

		let started = Instant::now();
		let handover = guard.take();

		assert!(started.elapsed() < WAIT / 2);

		drop(handover);
		audio_thread
			.join()
			.expect("the audio thread saw the request");
	}
}
