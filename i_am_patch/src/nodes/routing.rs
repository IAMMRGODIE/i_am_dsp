//! The host's two inputs, the splitters, the mixers and the block delay.

use super::*;

/// The host input; the engine writes the block into this node's output.
pub(super) struct Source;

impl BlockNode for Source {
	fn process(&mut self, _: &mut BlockContext, _: &[Buffer], _: &mut [Buffer]) {}
}

/// Mid and side, each written to both channels of its mono port.
pub(super) struct SplitMs;

impl BlockNode for SplitMs {
	fn process(&mut self, context: &mut BlockContext, inputs: &[Buffer], outputs: &mut [Buffer]) {
		let input = audio(inputs, 0);

		if outputs.len() < 2 {
			return;
		}

		let (first, rest) = outputs.split_at_mut(1);

		if let (Buffer::Audio(mid), Buffer::Audio(side)) = (&mut first[0], &mut rest[0]) {
			mid.clear();
			side.clear();
			mid.frames = context.frames;
			side.frames = context.frames;

			for frame in 0..context.frames {
				let left = input.channels[0][frame];
				let right = input.channels[1][frame];

				mid.channels[0][frame] = (left + right) * 0.5;
				mid.channels[1][frame] = (left + right) * 0.5;
				side.channels[0][frame] = (left - right) * 0.5;
				side.channels[1][frame] = (left - right) * 0.5;
			}
		}
	}
}

/// Sums its inputs, with a gain each.
/// Left and right, each copied into both channels.
///
/// A split side is a normal stereo signal, so it can go through any node that
/// wants one; the copy is what makes the two sides interchangeable in a patch.
pub(super) struct SplitLr;

impl BlockNode for SplitLr {
	fn process(&mut self, context: &mut BlockContext, inputs: &[Buffer], outputs: &mut [Buffer]) {
		let input = audio(inputs, 0);

		for (port, channel) in [(0, 0), (1, 1)] {
			if let Some(Buffer::Audio(out)) = outputs.get_mut(port) {
				out.clear();
				out.frames = context.frames;

				for frame in 0..context.frames {
					out.channels[0][frame] = input.channels[channel][frame];
					out.channels[1][frame] = input.channels[channel][frame];
				}
			}
		}
	}
}

/// The rack's three-way split, as a node: low, mid and high.
pub(super) struct BandSplit {
	low: f32,
	high: f32,
	crossover: Crossover,
}

impl BandSplit {
	pub(super) fn new(rate: usize, params: &BTreeMap<&'static str, f32>) -> Self {
		let low = param(params, "low", 120.0);
		let high = param(params, "high", 2_000.0);

		Self {
			low,
			high,
			crossover: Crossover::new(rate, low, high),
		}
	}
}

impl BlockNode for BandSplit {
	fn set_param(&mut self, name: &str, value: f32) {
		match name {
			"low" if value != self.low => {
				self.low = value;
				self.crossover.set_splits(self.low, self.high);
			}
			"high" if value != self.high => {
				self.high = value;
				self.crossover.set_splits(self.low, self.high);
			}
			_ => {}
		}
	}

	fn process(&mut self, context: &mut BlockContext, inputs: &[Buffer], outputs: &mut [Buffer]) {
		let input = audio(inputs, 0);

		for out in outputs.iter_mut() {
			if let Buffer::Audio(out) = out {
				out.clear();
				out.frames = context.frames;
			}
		}

		for frame in 0..context.frames {
			let bands = self
				.crossover
				.split(input.frame(frame), &mut context.context);

			for (port, band) in bands.iter().enumerate() {
				if let Some(Buffer::Audio(out)) = outputs.get_mut(port) {
					out.channels[0][frame] = band[0];
					out.channels[1][frame] = band[1];
				}
			}
		}
	}
}

/// The three bands added back together: the inverse of [BandSplit].
///
/// Each band has a gain, so a split can be rebalanced without an extra node per
/// band; at 1.0 the sum is the input again.
pub(super) struct BandMix {
	pub(super) low: f32,
	pub(super) mid: f32,
	pub(super) high: f32,
}

impl BlockNode for BandMix {
	fn set_param(&mut self, name: &str, value: f32) {
		match name {
			"low" => self.low = value,
			"mid" => self.mid = value,
			"high" => self.high = value,
			_ => {}
		}
	}

	fn process(&mut self, context: &mut BlockContext, inputs: &[Buffer], outputs: &mut [Buffer]) {
		let low = audio(inputs, 0);
		let mid = audio(inputs, 1);
		let high = audio(inputs, 2);

		if let Some(Buffer::Audio(out)) = outputs.first_mut() {
			out.clear();
			out.frames = context.frames;

			for channel in 0..2 {
				for frame in 0..context.frames {
					out.channels[channel][frame] = low.channels[channel][frame] * self.low
						+ mid.channels[channel][frame] * self.mid
						+ high.channels[channel][frame] * self.high;
				}
			}
		}
	}
}

/// Left and right back into a stereo pair: the inverse of [SplitLr].
pub(super) struct MixLr;

impl BlockNode for MixLr {
	fn process(&mut self, context: &mut BlockContext, inputs: &[Buffer], outputs: &mut [Buffer]) {
		let left = audio(inputs, 0);
		let right = audio(inputs, 1);

		if let Some(Buffer::Audio(out)) = outputs.first_mut() {
			out.clear();
			out.frames = context.frames;

			for frame in 0..context.frames {
				out.channels[0][frame] = left.channels[0][frame];
				out.channels[1][frame] = right.channels[0][frame];
			}
		}
	}
}

/// Mid and side back into left and right: the inverse of [SplitMs].
pub(super) struct MixMs;

impl BlockNode for MixMs {
	fn process(&mut self, context: &mut BlockContext, inputs: &[Buffer], outputs: &mut [Buffer]) {
		let mid = audio(inputs, 0);
		let side = audio(inputs, 1);

		if let Some(Buffer::Audio(out)) = outputs.first_mut() {
			out.clear();
			out.frames = context.frames;

			for frame in 0..context.frames {
				let m = mid.channels[0][frame];
				let s = side.channels[0][frame];

				out.channels[0][frame] = m + s;
				out.channels[1][frame] = m - s;
			}
		}
	}
}

pub(super) struct Mixer {
	pub(super) a: f32,
	pub(super) b: f32,
}

impl BlockNode for Mixer {
	fn set_param(&mut self, name: &str, value: f32) {
		match name {
			"a" => self.a = value,
			"b" => self.b = value,
			_ => {}
		}
	}

	fn process(&mut self, context: &mut BlockContext, inputs: &[Buffer], outputs: &mut [Buffer]) {
		let a = audio(inputs, 0);
		let b = audio(inputs, 1);

		if let Some(Buffer::Audio(out)) = outputs.first_mut() {
			out.clear();
			out.frames = context.frames;

			for channel in 0..2 {
				for frame in 0..context.frames {
					out.channels[channel][frame] =
						a.channels[channel][frame] * self.a + b.channels[channel][frame] * self.b;
				}
			}
		}
	}
}

/// A fixed 64-frame delay, so the graph has something with latency.
pub(super) struct Delay {
	buffer: [[f32; 2]; 64],
	write: usize,
}

impl Delay {
	pub(super) fn new() -> Self {
		Self {
			buffer: [[0.0; 2]; 64],
			write: 0,
		}
	}
}

impl BlockNode for Delay {
	fn process(&mut self, context: &mut BlockContext, inputs: &[Buffer], outputs: &mut [Buffer]) {
		let input = audio(inputs, 0);

		if let Some(Buffer::Audio(out)) = outputs.first_mut() {
			out.clear();
			out.frames = context.frames;

			for frame in 0..context.frames {
				let read = self.write;

				for channel in 0..2 {
					out.channels[channel][frame] = self.buffer[read][channel];
					self.buffer[read][channel] = input.channels[channel][frame];
				}

				self.write = (self.write + 1) % 64;
			}
		}
	}
}
