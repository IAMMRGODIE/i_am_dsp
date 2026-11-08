//! The block the engine runs on, and the buffers between nodes.

use i_am_dsp::prelude::NoteEvent;

/// How many frames one process call covers.
pub const BLOCK: usize = 64;

/// A stereo block of audio, planar per channel.
#[derive(Clone, Copy)]
pub struct AudioBlock {
	/// How many frames are actually valid.
	pub frames: usize,
	/// Left and right.
	pub channels: [[f32; BLOCK]; 2],
}

impl AudioBlock {
	/// A silent block of the given length.
	pub fn silent(frames: usize) -> Self {
		Self {
			frames: frames.min(BLOCK),
			channels: [[0.0; BLOCK]; 2],
		}
	}

	/// Fills every frame with zero.
	pub fn clear(&mut self) {
		for channel in &mut self.channels {
			channel.fill(0.0);
		}
	}

	/// Adds another block into this one.
	pub fn add(&mut self, other: &Self) {
		let frames = self.frames.min(other.frames);

		for channel in 0..2 {
			for frame in 0..frames {
				self.channels[channel][frame] += other.channels[channel][frame];
			}
		}
	}

	/// Copies another block into this one.
	pub fn copy_from(&mut self, other: &Self) {
		let frames = self.frames.min(other.frames);
		self.frames = frames;

		for channel in 0..2 {
			self.channels[channel][..frames]
				.copy_from_slice(&other.channels[channel][..frames]);
		}
	}

	/// One interleaved frame.
	#[inline]
	pub fn frame(&self, frame: usize) -> [f32; 2] {
		[self.channels[0][frame], self.channels[1][frame]]
	}
}

/// What travels along one edge.
///
/// The audio variant is much larger than the other two, but it is fixed-size
/// and stack-allocated: one buffer per port, created once when the engine is
/// built, so boxing it would only move the copy to the heap.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Copy)]
pub enum Buffer {
	/// A stereo audio block.
	Audio(AudioBlock),
	/// One control value for the whole block.
	Control(f32),
	/// Note events, which the context carries instead of the buffer.
	Midi,
}

impl Buffer {
	/// A silent buffer of the same kind.
	pub fn silent_of(self, frames: usize) -> Self {
		match self {
			Self::Audio(_) => Self::Audio(AudioBlock::silent(frames)),
			Self::Control(_) => Self::Control(0.0),
			Self::Midi => Self::Midi,
		}
	}

	/// Adds another buffer of the same kind.
	pub fn add(&mut self, other: &Self) {
		match (self, other) {
			(Self::Audio(into), Self::Audio(from)) => into.add(from),
			(Self::Control(into), Self::Control(from)) => *into += from,
			_ => {}
		}
	}

	/// The audio block inside, if this is one.
	pub fn audio(&self) -> Option<&AudioBlock> {
		match self {
			Self::Audio(block) => Some(block),
			_ => None,
		}
	}
}

/// What a node needs to know about one process call.
pub struct BlockContext {
	/// How many frames to write.
	pub frames: usize,
	/// The sample rate.
	pub rate: usize,
	/// A running block counter.
	pub block: u64,
	/// The note events of this block.
	pub events: Vec<NoteEvent>,
}
