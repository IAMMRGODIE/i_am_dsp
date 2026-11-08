//! The node implementations the catalog names.

use std::collections::BTreeMap;
use std::f32::consts::TAU;

use i_am_dsp::prelude::{Effect, Gain, Multimode, NoteEvent, Parameters};

use crate::block::{AudioBlock, BLOCK, BlockContext, Buffer};

/// One node in the running graph.
pub trait BlockNode: Send + Sync {
	/// Processes one block: reads the inputs, writes the outputs.
	fn process(&mut self, context: &mut BlockContext, inputs: &[Buffer], outputs: &mut [Buffer]);

	/// Applies one parameter, before the block runs.
	///
	/// The engine calls this for every parameter every block, so a modulation
	/// only has to move the value it is assigned to.
	fn set_param(&mut self, _name: &str, _value: f32) {}

	/// Applies a parameter that moves over the block: from `start` to `end`.
	///
	/// A node that can ramp does so per sample or per sub-block; the default
	/// takes the end value and leaves the parameter where it was for the block.
	fn set_param_block(&mut self, name: &str, _start: f32, end: f32, _frames: usize) {
		self.set_param(name, end);
	}
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
		"voice:in" => Box::new(Quiet),
		"voice:out" => Box::new(Quiet),
		"gen:additive" => Box::new(Additive::new(rate, param(params, "decay", 1.0))),
		"gain" => Box::new(GainNode::new(params)),
		"filter" => Box::new(Filter::new(rate, params)),
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
	release: f32,
	releasing: bool,
	note: usize,
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
			// A release of 150 ms to -60 dB.
			release: 10f32.powf(-3.0 / (0.15 * rate as f32)),
			releasing: false,
			note: 0,
		}
	}
}

impl BlockNode for Additive {
	fn set_param(&mut self, name: &str, value: f32) {
		if name == "decay" {
			self.decay = 10f32.powf(-3.0 / (value.clamp(0.05, 30.0) * self.rate));
		}
	}

	fn process(&mut self, context: &mut BlockContext, _: &[Buffer], outputs: &mut [Buffer]) {
		for event in &context.events {
			match event {
				NoteEvent::NoteOn { note, velocity, .. } => {
					self.frequency = 440.0 * 2f32.powf((*note as f32 - 69.0) / 12.0);
					self.envelope = velocity.clamp(0.0, 1.0);
					self.phases.fill(0.0);
					self.note = *note;
					self.releasing = false;
				}
				NoteEvent::NoteOff { note, .. } if *note == self.note => {
					self.releasing = true;
				}
				_ => {}
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
				sample += (phase * TAU).sin() * self.gains[partial];
			}

			sample *= self.envelope * 0.1;
			out.channels[0][frame] = sample;
			out.channels[1][frame] = sample;

			if self.releasing {
				self.envelope *= self.release;
			} else {
				self.envelope *= self.decay;
			}
		}
	}
}

/// The product's gain: one knob, and the DSP the product ships.
struct GainNode {
	effect: Gain,
	ramp: Option<(f32, f32)>,
}

impl GainNode {
	fn new(params: &BTreeMap<&'static str, f32>) -> Self {
		let mut node = Self {
			effect: Gain::new(1.0),
			ramp: None,
		};

		for (name, value) in params {
			node.effect.set_parameter(name, (*value).into());
		}

		node
	}
}

impl BlockNode for GainNode {
	fn set_param(&mut self, name: &str, value: f32) {
		self.effect.set_parameter(name, value.into());
	}

	fn set_param_block(&mut self, name: &str, start: f32, end: f32, _frames: usize) {
		if name == "gain" {
			self.ramp = Some((start, end));
		} else {
			self.set_param(name, end);
		}
	}

	fn process(&mut self, context: &mut BlockContext, inputs: &[Buffer], outputs: &mut [Buffer]) {
		let input = audio(inputs, 0);

		if let Some(Buffer::Audio(out)) = outputs.first_mut() {
			out.clear();
			out.frames = context.frames;
			let frames = context.frames.max(1);

			for frame in 0..context.frames {
				if let Some((from, to)) = self.ramp {
					let t = (frame + 1) as f32 / frames as f32;
					self.effect.gain = from + (to - from) * t;
				}

				let mut samples = [input.channels[0][frame], input.channels[1][frame]];
				self.effect.process(&mut samples, &[], &mut context.context);
				out.channels[0][frame] = samples[0];
				out.channels[1][frame] = samples[1];
			}
		}
	}
}

/// The product's filter: the card's controls and the DSP are the same schema.
struct Filter {
	effect: Multimode,
	cutoff: Option<(f32, f32)>,
	q: Option<(f32, f32)>,
}

impl Filter {
	fn new(rate: usize, params: &BTreeMap<&'static str, f32>) -> Self {
		let mut filter = Self {
			effect: Multimode::new(rate, 1000.0, 0.707),
			cutoff: None,
			q: None,
		};

		for (name, value) in params {
			filter.effect.set_parameter(name, (*value).into());
		}

		filter
	}
}

impl BlockNode for Filter {
	fn set_param(&mut self, name: &str, value: f32) {
		self.effect.set_parameter(name, value.into());
	}

	fn set_param_block(&mut self, name: &str, start: f32, end: f32, _frames: usize) {
		match name {
			"cutoff" => self.cutoff = Some((start, end)),
			"q" => self.q = Some((start, end)),
			_ => self.set_param(name, end),
		}
	}

	fn process(&mut self, context: &mut BlockContext, inputs: &[Buffer], outputs: &mut [Buffer]) {
		const STEPS: usize = 8;
		let input = audio(inputs, 0);
		let frames = context.frames;

		if let Some(Buffer::Audio(out)) = outputs.first_mut() {
			out.clear();
			out.frames = frames;

			let step = frames.div_ceil(STEPS).max(1);

			for start in (0..frames).step_by(step) {
				let t = start as f32 / frames.max(1) as f32;

				if let Some((from, to)) = self.cutoff {
					self.effect
						.set_parameter("cutoff", (from + (to - from) * t).into());
				}

				if let Some((from, to)) = self.q {
					self.effect.set_parameter("q", (from + (to - from) * t).into());
				}

				for frame in start..(start + step).min(frames) {
					let mut samples = [input.channels[0][frame], input.channels[1][frame]];
					self.effect.process(&mut samples, &[], &mut context.context);
					out.channels[0][frame] = samples[0];
					out.channels[1][frame] = samples[1];
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
	rate: f32,
}

impl Lfo {
	fn new(rate: usize, frequency: f32) -> Self {
		Self {
			phase: 0.0,
			step: frequency / rate as f32,
			rate: rate as f32,
		}
	}
}

impl BlockNode for Lfo {
	fn set_param(&mut self, name: &str, value: f32) {
		if name == "rate" {
			self.step = value.clamp(0.01, 100.0) / self.rate;
		}
	}

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
	rate: f32,
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
			rate: rate as f32,
		}
	}
}

impl BlockNode for Envelope {
	fn set_param(&mut self, name: &str, value: f32) {
		let coefficient = |milliseconds: f32| {
			1.0 - (-1000.0 / (milliseconds.clamp(0.1, 2000.0) * self.rate)).exp()
		};

		match name {
			"attack" => self.attack = coefficient(value),
			"release" => self.release = coefficient(value),
			_ => {}
		}
	}

	fn process(&mut self, context: &mut BlockContext, inputs: &[Buffer], outputs: &mut [Buffer]) {
		let input = audio(inputs, 0);

		// The follower smooths per sample and reports the block's last value;
		// one step per block would make a ten millisecond attack take minutes.
		for frame in 0..context.frames {
			let peak = input.channels[0][frame]
				.abs()
				.max(input.channels[1][frame].abs());
			let coefficient = if peak > self.envelope {
				self.attack
			} else {
				self.release
			};

			self.envelope += (peak - self.envelope) * coefficient;
		}

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
