//! A short ring of recent audio, for a live scope.
//!
//! The audio thread writes one mono frame at a time and the page reads the last
//! few hundred; there is no lock on either side. The write index is published with
//! Release and read with Acquire, so a snapshot that sees a newer index also sees
//! the samples that came before it.

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

/// How many frames the page draws.
pub const POINTS: usize = 256;

/// How many frames the ring keeps. A power of two, so the wrap is a mask.
const CAPACITY: usize = 4096;

/// The recent input and output of one plug-in, for a scope.
pub struct Scope {
	input: Box<[AtomicU32]>,
	output: Box<[AtomicU32]>,
	write: AtomicUsize,
}

impl Default for Scope {
	fn default() -> Self {
		Self::new()
	}
}

impl Scope {
	/// An empty scope.
	pub fn new() -> Self {
		let empty = || {
			(0..CAPACITY)
				.map(|_| AtomicU32::new(0))
				.collect::<Vec<_>>()
				.into_boxed_slice()
		};

		Self {
			input: empty(),
			output: empty(),
			write: AtomicUsize::new(0),
		}
	}

	/// Records one mono frame. Called from the audio thread.
	pub fn push(&self, input: f32, output: f32) {
		let index = self.write.load(Ordering::Relaxed);
		let slot = index % CAPACITY;

		self.input[slot].store(input.to_bits(), Ordering::Relaxed);
		self.output[slot].store(output.to_bits(), Ordering::Relaxed);
		self.write.store(index + 1, Ordering::Release);
	}

	/// The last [POINTS] frames as little-endian f32: input first, then output.
	pub fn snapshot(&self) -> Vec<u8> {
		let end = self.write.load(Ordering::Acquire);
		let start = end.saturating_sub(POINTS);
		let mut bytes = Vec::with_capacity(POINTS * 8);

		for channel in [&self.input, &self.output] {
			for offset in 0..POINTS {
				let frame = start + offset;
				let value = if frame < end {
					f32::from_bits(channel[frame % CAPACITY].load(Ordering::Relaxed))
				} else {
					0.0
				};

				bytes.extend_from_slice(&value.to_le_bytes());
			}
		}

		bytes
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// The page reads the newest frames, in order, and gets both channels.
	#[test]
	fn a_snapshot_holds_the_newest_frames() {
		let scope = Scope::new();

		for frame in 0..600 {
			scope.push(frame as f32, -(frame as f32));
		}

		let bytes = scope.snapshot();
		assert_eq!(bytes.len(), POINTS * 8);

		let read = |offset: usize| {
			f32::from_le_bytes(bytes[offset * 4..offset * 4 + 4].try_into().unwrap())
		};

		assert_eq!(read(0), 344.0, "the first of the last 256 frames");
		assert_eq!(read(POINTS - 1), 599.0);
		assert_eq!(read(POINTS), -344.0, "the output follows the input");
		assert_eq!(read(POINTS * 2 - 1), -599.0);
	}

	/// Before any audio arrives, a scope is silence rather than garbage.
	#[test]
	fn a_fresh_scope_is_silent() {
		let bytes = Scope::new().snapshot();

		assert_eq!(bytes.len(), POINTS * 8);
		assert!(bytes.iter().all(|byte| *byte == 0));
	}
}
