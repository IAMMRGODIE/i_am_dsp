//! The last blocks of one node's output, for a page's live pictures.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use crate::block::{AudioBlock, BLOCK};

/// The rings behind one scope: sample frames for an audio picture, and one
/// value per block for a control picture's history.
struct Inner {
	frames: Vec<[f32; 2]>,
	blocks: Vec<f32>,
}

/// A ring of frames written by the audio thread and read by the page.
pub struct Scope {
	inner: Mutex<Inner>,
	write: AtomicUsize,
	block_write: AtomicUsize,
	block_total: AtomicU64,
}

impl Scope {
	/// How many frames the ring holds.
	pub const LEN: usize = 4096;
	/// How many blocks the control history holds: about three seconds at 44.1 kHz.
	pub const BLOCKS: usize = 2048;

	pub fn new() -> Self {
		Self {
			inner: Mutex::new(Inner {
				frames: vec![[0.0; 2]; Self::LEN],
				blocks: vec![0.0; Self::BLOCKS],
			}),
			write: AtomicUsize::new(0),
			block_write: AtomicUsize::new(0),
			block_total: AtomicU64::new(0),
		}
	}

	/// Copies one block in, dropping the oldest frames and block value.
	pub fn write(&self, block: &AudioBlock) {
		let Ok(mut inner) = self.inner.lock() else {
			return;
		};
		let mut write = self.write.load(Ordering::Relaxed).min(Self::LEN - 1);

		for index in 0..block.frames.min(BLOCK) {
			inner.frames[write] = block.frame(index);
			write = (write + 1) % Self::LEN;
		}

		self.write.store(write, Ordering::Relaxed);

		// One value per block: a control buffer repeats its value across the
		// block, and for an audio buffer the last frame stands in. Only control
		// taps read this ring, but every scope keeps it so the tap does not have
		// to know which kind it is.
		let last = block.frames.saturating_sub(1).min(BLOCK - 1);
		let value = block.channels[0][last];
		let block_write = self
			.block_write
			.load(Ordering::Relaxed)
			.min(Self::BLOCKS - 1);

		inner.blocks[block_write] = value;
		self.block_write
			.store((block_write + 1) % Self::BLOCKS, Ordering::Relaxed);
		self.block_total.fetch_add(1, Ordering::Relaxed);
	}

	/// The last `count` frames, oldest first.
	pub fn read(&self, count: usize) -> Vec<[f32; 2]> {
		let Ok(inner) = self.inner.lock() else {
			return Vec::new();
		};
		let count = count.min(Self::LEN);
		let write = self.write.load(Ordering::Relaxed).min(Self::LEN - 1);

		(0..count)
			.map(|index| inner.frames[(write + Self::LEN - count + index) % Self::LEN])
			.collect()
	}

	/// The block counter and the values written after `since`, oldest first.
	///
	/// A counter at or before `since` means the scope was recreated; the call
	/// then returns the whole ring, so the page can start its history over.
	pub fn read_blocks_since(&self, since: u64, max: usize) -> (u64, Vec<f32>) {
		let Ok(inner) = self.inner.lock() else {
			return (0, Vec::new());
		};
		let total = self.block_total.load(Ordering::Relaxed);
		let count = if since == 0 || total < since {
			(total as usize).min(Self::BLOCKS)
		} else {
			((total - since) as usize).min(max).min(Self::BLOCKS)
		};
		let write = self
			.block_write
			.load(Ordering::Relaxed)
			.min(Self::BLOCKS - 1);

		let values = (0..count)
			.map(|index| inner.blocks[(write + Self::BLOCKS - count + index) % Self::BLOCKS])
			.collect();

		(total, values)
	}
}

impl Default for Scope {
	fn default() -> Self {
		Self::new()
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn block(value: f32) -> AudioBlock {
		let mut block = AudioBlock::silent(BLOCK);

		block.channels[0][BLOCK - 1] = value;
		block.channels[1][BLOCK - 1] = value;
		block
	}

	#[test]
	fn the_block_history_keeps_the_order_and_drops_the_oldest() {
		let scope = Scope::new();

		for value in 0..(Scope::BLOCKS + 8) {
			scope.write(&block(value as f32));
		}

		let (total, blocks) = scope.read_blocks_since(0, Scope::BLOCKS);

		assert_eq!(total, (Scope::BLOCKS + 8) as u64);
		assert_eq!(blocks.len(), Scope::BLOCKS);
		assert_eq!(blocks[0], 8.0);
		assert_eq!(blocks[Scope::BLOCKS - 1], (Scope::BLOCKS + 7) as f32);
	}

	#[test]
	fn a_since_returns_only_the_new_blocks() {
		let scope = Scope::new();

		for value in 0..10 {
			scope.write(&block(value as f32));
		}

		let (total, blocks) = scope.read_blocks_since(6, 64);

		assert_eq!(total, 10);
		assert_eq!(blocks, vec![6.0, 7.0, 8.0, 9.0]);
	}

	#[test]
	fn a_counter_that_went_backwards_reads_as_a_restart() {
		let scope = Scope::new();

		for value in 0..10 {
			scope.write(&block(value as f32));
		}

		let (total, blocks) = scope.read_blocks_since(1000, Scope::BLOCKS);

		assert_eq!(total, 10);
		assert_eq!(
			blocks,
			(0..10).map(|value| value as f32).collect::<Vec<_>>()
		);
	}

	#[test]
	fn nothing_new_reads_as_nothing() {
		let scope = Scope::new();

		for value in 0..10 {
			scope.write(&block(value as f32));
		}

		let (total, blocks) = scope.read_blocks_since(10, 64);

		assert_eq!(total, 10);
		assert!(blocks.is_empty());
	}
}
