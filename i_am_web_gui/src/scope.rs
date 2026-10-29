//! A short ring of recent audio, for a live scope.
//!
//! The audio thread writes one mono frame at a time and the page reads the last
//! few hundred; there is no lock on either side. The write index is published with
//! Release and read with Acquire, so a snapshot that sees a newer index also sees
//! the samples that came before it.

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

/// How many frames the page reads. Full rate, so a spectrum can use them too.
pub const POINTS: usize = 4096;

/// How many frames the ring keeps. A power of two, so the wrap is a mask.
const CAPACITY: usize = 8192;

/// How many frames the meter averages. About 21 ms at 48 kHz: steady enough to
/// read, quick enough to follow a phrase.
const METER_POINTS: usize = 1024;

/// The recent input and output of one plug-in, for a scope.
pub struct Scope {
	input: Box<[AtomicU32]>,
	output: Box<[AtomicU32]>,
	sidechain: Box<[AtomicU32]>,
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
			sidechain: empty(),
			write: AtomicUsize::new(0),
		}
	}

	/// Records one mono frame. Called from the audio thread. Without a separate
	/// sidechain the input stands in for it, so the meter has a signal either way.
	pub fn push(&self, input: f32, output: f32) {
		self.push_with_sidechain(input, output, input);
	}

	/// Records one mono frame and the sidechain frame that fed the detector.
	pub fn push_with_sidechain(&self, input: f32, output: f32, sidechain: f32) {
		let index = self.write.load(Ordering::Relaxed);
		let slot = index % CAPACITY;

		self.input[slot].store(input.to_bits(), Ordering::Relaxed);
		self.output[slot].store(output.to_bits(), Ordering::Relaxed);
		self.sidechain[slot].store(sidechain.to_bits(), Ordering::Relaxed);
		self.write.store(index + 1, Ordering::Release);
	}

	/// The RMS level of the recent input and sidechain, in dBFS. The page uses
	/// the input for a normal detector and the sidechain when the compressor is
	/// listening to it.
	pub fn meter(&self) -> (f32, f32) {
		let end = self.write.load(Ordering::Acquire);
		let start = end.saturating_sub(METER_POINTS);
		let count = end - start;

		if count == 0 {
			return (-96.0, -96.0);
		}

		let mut input_energy = 0.0;
		let mut sidechain_energy = 0.0;

		for frame in start..end {
			let slot = frame % CAPACITY;
			let input = f32::from_bits(self.input[slot].load(Ordering::Relaxed));
			let sidechain = f32::from_bits(self.sidechain[slot].load(Ordering::Relaxed));

			input_energy += input * input;
			sidechain_energy += sidechain * sidechain;
		}

		let level = |energy: f32| 20.0 * (energy / count as f32).sqrt().max(1e-6).log10();

		(level(input_energy), level(sidechain_energy))
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
		let count = POINTS + 100;

		for frame in 0..count {
			scope.push(frame as f32, -(frame as f32));
		}

		let bytes = scope.snapshot();
		assert_eq!(bytes.len(), POINTS * 8);

		let read = |offset: usize| {
			f32::from_le_bytes(bytes[offset * 4..offset * 4 + 4].try_into().unwrap())
		};

		assert_eq!(read(0), 100.0, "the first of the newest frames");
		assert_eq!(read(POINTS - 1), (count - 1) as f32);
		assert_eq!(read(POINTS), -100.0, "the output follows the input");
		assert_eq!(read(POINTS * 2 - 1), -((count - 1) as f32));
	}

	/// Before any audio arrives, a scope is silence rather than garbage.
	#[test]
	fn a_fresh_scope_is_silent() {
		let bytes = Scope::new().snapshot();

		assert_eq!(bytes.len(), POINTS * 8);
		assert!(bytes.iter().all(|byte| *byte == 0));
	}
	/// The meter reads the input and the sidechain separately, in dBFS.
	#[test]
	fn a_meter_reads_the_input_and_the_sidechain() {
		let scope = Scope::new();

		for _ in 0..METER_POINTS {
			scope.push_with_sidechain(0.5, 0.0, 0.25);
		}

		let (input, sidechain) = scope.meter();

		assert!((input - -6.02).abs() < 0.05, "input {input}");
		assert!((sidechain - -12.04).abs() < 0.05, "sidechain {sidechain}");
	}

}