//! The last blocks of one node's output, for a page's live pictures.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::block::{AudioBlock, BLOCK};

/// A ring of frames written by the audio thread and read by the page.
pub struct Scope {
	frames: Mutex<Vec<[f32; 2]>>,
	write: AtomicUsize,
}

impl Scope {
	/// How many frames the ring holds.
	pub const LEN: usize = 4096;

	pub fn new() -> Self {
		Self {
			frames: Mutex::new(vec![[0.0; 2]; Self::LEN]),
			write: AtomicUsize::new(0),
		}
	}

	/// Copies one block in, dropping the oldest frames.
	pub fn write(&self, block: &AudioBlock) {
		let Ok(mut frames) = self.frames.lock() else {
			return;
		};
		let mut write = self.write.load(Ordering::Relaxed).min(Self::LEN - 1);

		for index in 0..block.frames.min(BLOCK) {
			frames[write] = block.frame(index);
			write = (write + 1) % Self::LEN;
		}

		self.write.store(write, Ordering::Relaxed);
	}

	/// The last `count` frames, oldest first.
	pub fn read(&self, count: usize) -> Vec<[f32; 2]> {
		let Ok(frames) = self.frames.lock() else {
			return Vec::new();
		};
		let count = count.min(Self::LEN);
		let write = self.write.load(Ordering::Relaxed).min(Self::LEN - 1);

		(0..count)
			.map(|index| frames[(write + Self::LEN - count + index) % Self::LEN])
			.collect()
	}
}

impl Default for Scope {
	fn default() -> Self {
		Self::new()
	}
}
