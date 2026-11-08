//! The node implementations the catalog names.

use std::collections::BTreeMap;
use std::f32::consts::TAU;

use i_am_dsp::prelude::NoteEvent;

use crate::block::{AudioBlock, BLOCK, BlockContext, Buffer};

/// One node in the running graph.
pub trait BlockNode: Send {
	/// Processes one block: reads the inputs, writes the outputs.
	fn process(&mut self, context: &mut BlockContext, inputs: &[Buffer], outputs: &mut [Buffer]);
}

/// One parameter, or the fallback when the node does not have it.
fn param(params: &BTreeMap<&'static str, f32>, name: &str, fallback: f32) -> f32 {
	params.get(name).copied().unwrap_or(fallback)
}

/// Builds the node for a catalog kind at a sample rate.
pub fn make(
	kind: &str,
	rate: usize,
	params: &BTreeMap<&'static str, f32>,
) -> Option<Box<dyn BlockNode>> {
	let node: Box<dyn BlockNode> = match kind {
		"input:midi" => Box::new(Quiet),
		"input:audio" | "input:sidechain" => Box::new(Source),
		"out:audio" => Box::new(Quiet),
		"gen:additive" => Box::new(Additive::new(rate, param(params, "decay", 1.0))),
		"gain" => Box::new(Gain {
			gain: param(params, "gain", 1.0),
			pan: param(params, "pan", 0.0),
		}),
		"filter" => Box::new(Filter::new(rate, param(params, "cutoff", 2_000.0))),
		"split:ms" => Box::new(SplitMs),
		"mixer" => Box::new(Mixer {
			a: param(params, "a", 1.0),
			b: param(params, "b", 1.0),
		}),
		"lfo" => Box::new(Lfo::new(rate, param(params, "rate", 1.0))),
		"envelope" => Box::new(Envelope::new(
			rate,
			param(params, "attack", 10.0),
			param(params, "release", 100.0),
		)),
		"delay" => Box::new(Delay::new()),
		_ => return None,
	};

	Some(node)
}

fn audio(inputs: &[Buffer], port: usize) -> AudioBlock {
	inputs
		.get(port)
		.and_then(Buffer::audio)
		.copied()
		.unwrap_or_else(|| AudioBlock::silent(BLOCK))
}

/// A node with no work of its own.
struct Quiet;

impl BlockNode for Quiet {
	fn process(&mut self, _: &mut BlockContext, _: &[Buffer], _: &mut [Buffer]) {}
}

/// The host input; the engine writes the block into this node's output.
struct Source;

impl BlockNode for Source {
	fn process(&mut self, _: &mut BlockContext, _: &[Buffer], _: &mut [Buffer]) {}
}

/// A small additive voice, enough to make a note for the M1 tests.
struct Additive {
	rate: f32,
	frequency: f32,
	phases: [f32; 16],
	gains: [f32; 16],
	envelope: f32,
	decay: f32,
}

impl Additive {
	fn new(rate: usize, t60: f32) -> Self {
		let decay = 10f32.powf(-3.0 / (t60.clamp(0.05, 30.0) * rate as f32));
		let mut gains = [0.0; 16];

		for (index, gain) in gains.iter_mut().enumerate() {
			*gain = 1.0 / (index as f32 + 1.0);
		}

		Self {
			rate: rate as f32,
			frequency: 440.0,
			phases: [0.0; 16],
			gains,
			envelope: 0.0,
			decay,
		}
	}
}

impl BlockNode for Additive {
	fn process(&mut self, context: &mut BlockContext, _: &[Buffer], outputs: &mut [Buffer]) {
		for event in &context.events {
			if let NoteEvent::NoteOn { note, velocity, .. } = event {
				self.frequency = 440.0 * 2f32.powf((*note as f32 - 69.0) / 12.0);
				self.envelope = velocity.clamp(0.0, 1.0);
				self.phases.fill(0.0);
			}
		}

		let Some(Buffer::Audio(out)) = outputs.first_mut() else {
			return;
		};
		out.clear();
		out.frames = context.frames;

		for frame in 0..context.frames {
			let mut sample = 0.0;

			for partial in 0..16 {
				let ratio = partial as f32 + 1.0;
				let phase = (self.phases[partial] + self.frequency * ratio / self.rate).fract();
				self.phases[partial] = phase;
				sample += phase.sin() * self.gains[partial];
			}

			sample *= self.envelope * 0.1;
			out.channels[0][frame] = sample;
			out.channels[1][frame] = sample;
			self.envelope *= self.decay;
		}
	}
}

/// Gain and pan.
struct Gain {
	gain: f32,
	pan: f32,
}

impl BlockNode for Gain {
	fn process(&mut self, context: &mut BlockContext, inputs: &[Buffer], outputs: &mut [Buffer]) {
		let input = audio(inputs, 0);
		let angle = (self.pan.clamp(-1.0, 1.0) + 1.0) * std::f32::consts::FRAC_PI_4;
		let left = angle.cos() * self.gain;
		let right = angle.sin() * self.gain;

		if let Some(Buffer::Audio(out)) = outputs.first_mut() {
			out.clear();
			out.frames = context.frames;

			for frame in 0..context.frames {
				out.channels[0][frame] = input.channels[0][frame] * left;
				out.channels[1][frame] = input.channels[1][frame] * right;
			}
		}
	}
}

/// A one-pole lowpass.
struct Filter {
	coefficient: f32,
	state: [f32; 2],
}

impl Filter {
	fn new(rate: usize, cutoff: f32) -> Self {
		Self {
			coefficient: 1.0 - (-TAU * cutoff / rate as f32).exp(),
			state: [0.0; 2],
		}
	}
}

impl BlockNode for Filter {
	fn process(&mut self, context: &mut BlockContext, inputs: &[Buffer], outputs: &mut [Buffer]) {
		let input = audio(inputs, 0);
		let coefficient = self.coefficient;

		if let Some(Buffer::Audio(out)) = outputs.first_mut() {
			out.clear();
			out.frames = context.frames;

			for channel in 0..2 {
				let state = &mut self.state[channel];

				for frame in 0..context.frames {
					*state += coefficient * (input.channels[channel][frame] - *state);
					out.channels[channel][frame] = *state;
				}
			}
		}
	}
}

/// Mid and side, each written to both channels of its mono port.
struct SplitMs;

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
struct Mixer {
	a: f32,
	b: f32,
}

impl BlockNode for Mixer {
	fn process(&mut self, context: &mut BlockContext, inputs: &[Buffer], outputs: &mut [Buffer]) {
		let a = audio(inputs, 0);
		let b = audio(inputs, 1);

		if let Some(Buffer::Audio(out)) = outputs.first_mut() {
			out.clear();
			out.frames = context.frames;

			for channel in 0..2 {
				for frame in 0..context.frames {
					out.channels[channel][frame] = a.channels[channel][frame] * self.a
						+ b.channels[channel][frame] * self.b;
				}
			}
		}
	}
}

/// A control-rate sine.
struct Lfo {
	phase: f32,
	step: f32,
}

impl Lfo {
	fn new(rate: usize, frequency: f32) -> Self {
		Self {
			phase: 0.0,
			step: frequency / rate as f32,
		}
	}
}

impl BlockNode for Lfo {
	fn process(&mut self, context: &mut BlockContext, _: &[Buffer], outputs: &mut [Buffer]) {
		self.phase = (self.phase + self.step * context.frames as f32).fract();

		if let Some(Buffer::Control(value)) = outputs.first_mut() {
			*value = (self.phase * TAU).sin();
		}
	}
}

/// An envelope follower.
struct Envelope {
	attack: f32,
	release: f32,
	envelope: f32,
}

impl Envelope {
	fn new(rate: usize, attack_ms: f32, release_ms: f32) -> Self {
		let coefficient = |milliseconds: f32| {
			1.0 - (-1000.0 / (milliseconds.clamp(0.1, 2000.0) * rate as f32)).exp()
		};

		Self {
			attack: coefficient(attack_ms),
			release: coefficient(release_ms),
			envelope: 0.0,
		}
	}
}

impl BlockNode for Envelope {
	fn process(&mut self, context: &mut BlockContext, inputs: &[Buffer], outputs: &mut [Buffer]) {
		let input = audio(inputs, 0);
		let mut peak = 0.0f32;

		for frame in 0..context.frames {
			peak = peak
				.max(input.channels[0][frame].abs())
				.max(input.channels[1][frame].abs());
		}

		let coefficient = if peak > self.envelope {
			self.attack
		} else {
			self.release
		};
		self.envelope += (peak - self.envelope) * coefficient;

		if let Some(Buffer::Control(value)) = outputs.first_mut() {
			*value = self.envelope.clamp(0.0, 1.0);
		}
	}
}

/// A fixed 64-frame delay, so the graph has something with latency.
struct Delay {
	buffer: [[f32; 2]; 64],
	write: usize,
}

impl Delay {
	fn new() -> Self {
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
				let read = (self.write + 1) % 64;

				for channel in 0..2 {
					out.channels[channel][frame] = self.buffer[read][channel];
					self.buffer[self.write][channel] = input.channels[channel][frame];
				}

				self.write = read;
			}
		}
	}
}
