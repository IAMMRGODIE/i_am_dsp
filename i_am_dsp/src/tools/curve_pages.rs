//! Pages a curve editor owns and the audio thread reads.
//!
//! A page editor writes whole pages; a processor reads them while it renders.
//! Locking would put the editor in the audio thread's way, and copying the
//! points every block would allocate, so the writer swaps an Arc snapshot and
//! the reader takes one strong count. Nothing here knows what a page point is:
//! the filter's breakpoints, a spectral effect's threshold curve, and a
//! wavetable's frames are all just page lists.
//!
//! Starting the pages and making sure a page has enough points stay with the
//! caller, which is the only one that knows what an empty page should be.

use std::sync::Arc;
use std::sync::atomic::{AtomicPtr, Ordering};

/// One immutable set of pages, and the counter that moves when they change.
pub struct PageSnapshot<P> {
	generation: u64,
	pages: Vec<Vec<P>>,
}

impl<P> PageSnapshot<P> {
	/// Moves every time the pages are replaced.
	pub fn generation(&self) -> u64 {
		self.generation
	}

	/// Every page, in the order the morph walks them.
	pub fn pages(&self) -> &[Vec<P>] {
		&self.pages
	}
}

/// The pages a curve editor writes and a processor reads.
///
/// The audio thread loads a PageSnapshot with one atomic and never locks or
/// copies points. A reader that already holds a snapshot keeps seeing the pages
/// it started with, which is what lets it rebuild from one stable set while the
/// editor keeps writing.
pub struct CurvePages<P> {
	current: AtomicPtr<PageSnapshot<P>>,
}

impl<P: Send + Sync> CurvePages<P> {
	/// A page set that starts with a single page.
	pub fn new(first: Vec<P>) -> Self {
		let data = Arc::new(PageSnapshot {
			generation: 0,
			pages: vec![first],
		});

		Self {
			current: AtomicPtr::new(Arc::into_raw(data) as *mut PageSnapshot<P>),
		}
	}

	/// Every page now, for a card or a state save.
	pub fn pages(&self) -> Vec<Vec<P>>
	where
		P: Clone,
	{
		self.snapshot().pages.clone()
	}

	/// Moves whenever the pages change.
	pub fn generation(&self) -> u64 {
		self.snapshot().generation
	}

	/// A shared snapshot. A processor reads this once per rebuild.
	pub fn snapshot(&self) -> Arc<PageSnapshot<P>> {
		let pointer = self.current.load(Ordering::Acquire);
		// SAFETY: every pointer stored in the atomic came from Arc::into_raw,
		// and the writer only drops its own reference after the swap. Taking a
		// new strong count here keeps the data alive while we read it.
		unsafe {
			Arc::increment_strong_count(pointer);
			Arc::from_raw(pointer)
		}
	}

	/// Replaces every page; the swap is safe for the audio thread to read.
	///
	/// The list must not be empty: a page set always has at least the page a
	/// caller started with, because a processor indexes it without a check.
	pub fn replace(&self, pages: Vec<Vec<P>>) {
		assert!(!pages.is_empty(), "a page set keeps at least one page");

		let generation = self.generation() + 1;
		let data = Arc::new(PageSnapshot { generation, pages });
		let pointer = Arc::into_raw(data) as *mut PageSnapshot<P>;
		let old = self.current.swap(pointer, Ordering::AcqRel);
		// SAFETY: the pointer was created by Arc::into_raw in this struct.
		unsafe {
			drop(Arc::from_raw(old));
		}
	}
}

impl<P> Drop for CurvePages<P> {
	fn drop(&mut self) {
		let pointer = *self.current.get_mut();
		// SAFETY: the pointer was created by Arc::into_raw in this struct.
		unsafe {
			drop(Arc::from_raw(pointer));
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// A reader that took a snapshot keeps the pages it started with, even after
	/// the writer replaces them: this is what makes a lock-free read sound.
	#[test]
	fn a_snapshot_survives_a_replacement() {
		let pages = CurvePages::new(vec![1.0, 2.0]);
		let before = pages.snapshot();

		pages.replace(vec![vec![3.0]]);

		assert_eq!(before.pages(), &[vec![1.0, 2.0]]);
		assert_eq!(pages.pages(), vec![vec![3.0]]);
	}

	/// Every replacement moves the generation, which is the only thing a
	/// processor has to compare to know that it is stale.
	#[test]
	fn a_replacement_moves_the_generation() {
		let pages = CurvePages::new(vec![0.0]);
		let first = pages.generation();

		pages.replace(vec![vec![1.0]]);

		assert!(pages.generation() > first);
	}

	/// A page set always starts with the page its owner handed over.
	#[test]
	fn a_new_set_starts_with_its_first_page() {
		let pages = CurvePages::new(vec![7.0, 8.0]);

		assert_eq!(pages.pages(), vec![vec![7.0, 8.0]]);
		assert_eq!(pages.generation(), 0);
	}
}
