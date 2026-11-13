//! The contract between a page's feed and the voices that play it.
//!
//! A feed is not a parameter: the page writes a whole drawing, table or script
//! into a handle the note voices also hold. A voice has two duties, and both
//! follow from the generation:
//!
//! - a new voice starts from the current state, never from the factory one;
//! - a running voice reloads when the generation moves.
//!
//! There is deliberately no one-shot queue here: a queue can only be taken by
//! one voice, which leaves the others on the old state.

/// A page-fed state that every note voice can snapshot and poll.
pub trait VoiceFeed {
	/// Moves whenever the page replaces the state.
	fn generation(&self) -> u64;

	/// Whether the generation moved since `seen`, updating `seen` when it did.
	fn changed(&self, seen: &mut u64) -> bool {
		let generation = self.generation();

		if generation == *seen {
			return false;
		}

		*seen = generation;
		true
	}
}

impl VoiceFeed for i_am_dsp::prelude::SpectralFeed {
	fn generation(&self) -> u64 {
		i_am_dsp::prelude::SpectralFeed::generation(self)
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use std::sync::atomic::{AtomicU64, Ordering};

	struct TestFeed(AtomicU64);

	impl VoiceFeed for TestFeed {
		fn generation(&self) -> u64 {
			self.0.load(Ordering::Acquire)
		}
	}

	#[test]
	fn changed_reports_each_generation_once() {
		let feed = TestFeed(AtomicU64::new(1));
		let mut seen = 1;

		assert!(!feed.changed(&mut seen));

		feed.0.store(2, Ordering::Release);

		assert!(feed.changed(&mut seen));
		assert_eq!(seen, 2);
		assert!(!feed.changed(&mut seen));
	}
}
