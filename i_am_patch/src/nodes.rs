//! The node implementations the catalog names.

use std::collections::BTreeMap;
use std::f32::consts::TAU;

use i_am_dsp::prelude::{Effect, Gain, Message, Multimode, NoteEvent, Parameters};
use i_am_plugin::Plugin;

use crate::block::{AudioBlock, BLOCK, BlockContext, Buffer};
use crate::modulation::SourceRuntime;
use crate::sources::{SourceKind, SourceSettings, lfo_preset};

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

	/// Where a source's cycle is, for the card's playhead.
	fn phase(&self) -> Option<f32> {
		None
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
	points: &[crate::graph::Point],
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
		"lfo" => Box::new(SourceNode::new(SourceKind::Lfo, rate, params, points)),
		"random" => Box::new(SourceNode::new(SourceKind::Random, rate, params, &[])),
		"table" => Box::new(SourceNode::new(SourceKind::Table, rate, params, &[])),
		"pitch" => Box::new(SourceNode::new(SourceKind::Pitch, rate, params, &[])),
		"midi" => Box::new(SourceNode::new(SourceKind::Midi, rate, params, &[])),
		"envelope" => Box::new(SourceNode::new(SourceKind::Envelope, rate, params, &[])),
		"delay" => Box::new(Delay::new()),
		kind if kind.starts_with("synth:") => {
			use i_am_synths::{
				AdditiveWeb, CanvasWeb, DrummerWeb, FittedWeb, SamplerWeb, ScriptWeb, ShapesWeb,
				SpectrumWeb, StringWeb,
			};

			match &kind[6..] {
				"additive" => Box::new(SynthNode::<AdditiveWeb>::new(rate)),
				"canvas" => Box::new(SynthNode::<CanvasWeb>::new(rate)),
				"drummer" => Box::new(SynthNode::<DrummerWeb>::new(rate)),
				"fitted" => Box::new(SynthNode::<FittedWeb>::new(rate)),
				"sampler" => Box::new(SynthNode::<SamplerWeb>::new(rate)),
				"script" => Box::new(SynthNode::<ScriptWeb>::new(rate)),
				"shapes" => Box::new(SynthNode::<ShapesWeb>::new(rate)),
				"spectrum" => Box::new(SynthNode::<SpectrumWeb>::new(rate)),
				"strings" => Box::new(SynthNode::<StringWeb>::new(rate)),
				_ => return None,
			}
		}
		kind if kind.starts_with("mod:") => {
			let module = i_am_effects::catalog::get(&kind[4..])?;
			Box::new(ProductNode::new((module.build)(rate)))
		}
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

/// A rack module, driven by its own product schema.
struct ProductNode {
	effect: Box<dyn Effect<2>>,
}

impl ProductNode {
	fn new(effect: Box<dyn Effect<2>>) -> Self {
		Self { effect }
	}
}

impl BlockNode for ProductNode {
	fn set_param(&mut self, name: &str, value: f32) {
		self.effect.set_parameter(name, value.into());
	}

	fn process(&mut self, context: &mut BlockContext, inputs: &[Buffer], outputs: &mut [Buffer]) {
		let input = audio(inputs, 0);

		if let Some(Buffer::Audio(out)) = outputs.first_mut() {
			out.clear();
			out.frames = context.frames;

			for frame in 0..context.frames {
				let mut samples = [input.channels[0][frame], input.channels[1][frame]];
				self.effect.process(&mut samples, &[], &mut context.context);
				out.channels[0][frame] = samples[0];
				out.channels[1][frame] = samples[1];
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
					self.effect
						.set_parameter("q", (from + (to - from) * t).into());
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
					out.channels[channel][frame] =
						a.channels[channel][frame] * self.a + b.channels[channel][frame] * self.b;
				}
			}
		}
	}
}

/// A control source played as a graph node.
///
/// The graph wants a node with a port; the rack wants a slot. Both run the same
/// [SourceRuntime] now, so an LFO, envelope, random walk or table set to the
/// same numbers behaves the same on either page. The node keeps the catalog's
/// parameter names and maps them onto [SourceSettings]; the runtime does the
/// rest, including the tempo sync, the random stream and the smoothing.
struct SourceNode {
	kind: SourceKind,
	settings: SourceSettings,
	runtime: SourceRuntime,
	dirty: bool,
}

impl SourceNode {
	fn new(
		kind: SourceKind,
		rate: usize,
		params: &BTreeMap<&'static str, f32>,
		points: &[crate::graph::Point],
	) -> Self {
		let settings = SourceSettings::default();
		let mut node = Self {
			kind,
			settings: settings.clone(),
			runtime: SourceRuntime::from_settings(kind, false, settings, rate, 0),
			dirty: true,
		};

		for (name, value) in params {
			node.set_param(name, *value);
		}

		// Points pasted on the node override the preset shape, exactly as the
		// rack's custom LFO does.
		if kind == SourceKind::Lfo && !points.is_empty() {
			node.settings.lfo_points = points
				.iter()
				.map(|point| (point.time, point.value, point.bend))
				.collect();
			node.settings.lfo_shape = 4;
			node.dirty = true;
		}

		node
	}
}

impl BlockNode for SourceNode {
	fn set_param(&mut self, name: &str, value: f32) {
		let mut changed = false;

		{
			let settings = &mut self.settings;

			match (self.kind, name) {
				(SourceKind::Lfo, "rate") => {
					let next = value.clamp(0.01, 20.0);
					if settings.lfo_rate != next {
						settings.lfo_rate = next;
						changed = true;
					}
				}
				(SourceKind::Lfo, "shape") => {
					let next = (value.round() as i32).clamp(0, 4);
					if settings.lfo_shape != next {
						settings.lfo_shape = next;

						if next <= 3 {
							settings.lfo_points = lfo_preset(next);
						}

						changed = true;
					}
				}
				(SourceKind::Lfo, "sync") => {
					let next = value != 0.0;
					if settings.lfo_sync != next {
						settings.lfo_sync = next;
						changed = true;
					}
				}
				(SourceKind::Lfo, "beats") => {
					let next = value.clamp(0.03125, 32.0);
					if settings.lfo_beats != next {
						settings.lfo_beats = next;
						changed = true;
					}
				}
				(SourceKind::Envelope, "attack") => {
					let next = value.clamp(1.0, 500.0);
					if settings.env_attack != next {
						settings.env_attack = next;
						changed = true;
					}
				}
				(SourceKind::Envelope, "release") => {
					let next = value.clamp(1.0, 500.0);
					if settings.env_release != next {
						settings.env_release = next;
						changed = true;
					}
				}
				(SourceKind::Random, "rate") => {
					let next = value.clamp(0.01, 20.0);
					if settings.random_rate != next {
						settings.random_rate = next;
						changed = true;
					}
				}
				(SourceKind::Random, "smooth") => {
					let next = value.clamp(0.0, 500.0);
					if settings.random_smooth != next {
						settings.random_smooth = next;
						changed = true;
					}
				}
				(SourceKind::Random, "sync") => {
					let next = value != 0.0;
					if settings.random_sync != next {
						settings.random_sync = next;
						changed = true;
					}
				}
				(SourceKind::Random, "beats") => {
					let next = value.clamp(0.03125, 32.0);
					if settings.random_beats != next {
						settings.random_beats = next;
						changed = true;
					}
				}
				(SourceKind::Table, "rate") => {
					let next = value.clamp(0.01, 20.0);
					if settings.table_rate != next {
						settings.table_rate = next;
						changed = true;
					}
				}
				(SourceKind::Table, "shape") => {
					let next = (value.round() as i32).clamp(0, 3);
					if settings.table_shape != next {
						settings.table_shape = next;
						changed = true;
					}
				}
				(SourceKind::Table, "morph") => {
					let next = value.clamp(0.0, 1.0);
					if settings.table_morph != next {
						settings.table_morph = next;
						changed = true;
					}
				}
				(SourceKind::Table, "sync") => {
					let next = value != 0.0;
					if settings.table_sync != next {
						settings.table_sync = next;
						changed = true;
					}
				}
				(SourceKind::Table, "beats") => {
					let next = value.clamp(0.03125, 32.0);
					if settings.table_beats != next {
						settings.table_beats = next;
						changed = true;
					}
				}
				(SourceKind::Pitch, "clarity") => {
					let next = value.clamp(0.0, 1.0);
					if settings.pitch_clarity != next {
						settings.pitch_clarity = next;
						changed = true;
					}
				}
				(SourceKind::Pitch, "glide") => {
					let next = value.clamp(0.0, 1000.0);
					if settings.pitch_glide != next {
						settings.pitch_glide = next;
						changed = true;
					}
				}
				(SourceKind::Midi, "mode") => {
					let next = (value.round() as i32).clamp(0, 1);
					if settings.midi_mode != next {
						settings.midi_mode = next;
						changed = true;
					}
				}
				(SourceKind::Midi, "transpose") => {
					let next = (value.round() as i32).clamp(-24, 24);
					if settings.midi_transpose != next {
						settings.midi_transpose = next;
						changed = true;
					}
				}
				(SourceKind::Midi, "fine") => {
					let next = value.clamp(-100.0, 100.0);
					if settings.midi_fine != next {
						settings.midi_fine = next;
						changed = true;
					}
				}
				_ => {}
			}
		}

		if changed {
			self.dirty = true;
		}
	}

	fn process(&mut self, context: &mut BlockContext, inputs: &[Buffer], outputs: &mut [Buffer]) {
		if self.dirty {
			self.runtime
				.update_settings(self.kind, false, self.settings.clone(), context.rate);
			self.dirty = false;
		}

		let frames = context.frames.min(BLOCK);
		let input = audio(inputs, 0);
		let infos = context.context.infos();

		for frame in 0..frames {
			let mono = input.channels[0][frame]
				.abs()
				.max(input.channels[1][frame].abs());
			// The note events belong to the block, not to a sample: hand them to
			// the runtime once.
			let events: &[NoteEvent] = if frame == 0 { &context.events } else { &[] };
			self.runtime.advance(mono, infos, context.rate, events);
		}

		if let Some(Buffer::Control(value)) = outputs.first_mut() {
			*value = self.runtime.value();
		}
	}

	fn phase(&self) -> Option<f32> {
		Some(self.runtime.phase())
	}
}

/// A synth product: its notes come from the block's events, its audio out the
/// node's first port.
struct SynthNode<P: Default + Plugin> {
	inner: P,
}

impl<P: Default + Plugin> SynthNode<P> {
	fn new(rate: usize) -> Self {
		let mut inner = P::default();
		inner.on_activate(rate);

		Self { inner }
	}
}

impl<P: Default + Plugin> BlockNode for SynthNode<P> {
	fn set_param(&mut self, name: &str, value: f32) {
		self.inner.set_parameter(name, value.into());
	}

	fn process(&mut self, context: &mut BlockContext, _: &[Buffer], outputs: &mut [Buffer]) {
		for event in &context.events {
			self.inner
				.on_message(<P::Message as Message>::from_note_event(event.clone()));
		}

		if let Some(Buffer::Audio(out)) = outputs.first_mut() {
			out.clear();
			out.frames = context.frames;

			for frame in 0..context.frames {
				context.context.set_sample_offset(frame);
				let mut samples = [0.0f32; 2];
				self.inner.process(&mut samples, &[], &mut context.context);
				out.channels[0][frame] = samples[0];
				out.channels[1][frame] = samples[1];
			}
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

#[cfg(test)]
mod source_tests {
	use super::*;

	fn infos() -> i_am_dsp::ProcessInfos {
		i_am_dsp::ProcessInfos::default()
	}

	fn block(rate: usize) -> BlockContext {
		BlockContext {
			frames: BLOCK,
			rate,
			block: 0,
			events: Vec::new(),
			context: Box::new(()),
		}
	}

	fn run(node: &mut SourceNode, context: &mut BlockContext, input: f32) -> f32 {
		let mut outputs = [Buffer::Control(0.0)];
		let audio = AudioBlock {
			frames: BLOCK,
			channels: [[input; BLOCK]; 2],
		};
		let inputs = [Buffer::Audio(audio)];
		node.process(context, &inputs, &mut outputs);

		match outputs[0] {
			Buffer::Control(value) => value,
			_ => panic!("a control source writes a control buffer"),
		}
	}

	#[test]
	fn an_lfo_node_runs_the_same_lfo_as_a_rack_slot() {
		let rate = 48_000;
		let mut params = BTreeMap::new();
		params.insert("rate", 1.0);
		params.insert("shape", 1.0);

		let mut node = SourceNode::new(SourceKind::Lfo, rate, &params, &[]);
		let settings = node.settings.clone();

		let mut runtime = SourceRuntime::from_settings(SourceKind::Lfo, false, settings, rate, 0);
		let from_node = run(&mut node, &mut block(rate), 0.0);

		for _ in 0..BLOCK {
			runtime.advance(0.0, &infos(), rate, &[]);
		}

		assert!(
			(from_node - runtime.value()).abs() < 1e-6,
			"{from_node} vs {}",
			runtime.value()
		);
	}

	#[test]
	fn a_random_node_runs_the_same_stream_as_a_rack_slot() {
		let rate = 48_000;
		let mut params = BTreeMap::new();
		params.insert("rate", 4.0);
		params.insert("smooth", 20.0);

		let mut node = SourceNode::new(SourceKind::Random, rate, &params, &[]);
		let settings = node.settings.clone();

		let mut runtime =
			SourceRuntime::from_settings(SourceKind::Random, false, settings, rate, 0);
		let from_node = run(&mut node, &mut block(rate), 0.0);

		for _ in 0..BLOCK {
			runtime.advance(0.0, &infos(), rate, &[]);
		}

		assert!(
			(from_node - runtime.value()).abs() < 1e-6,
			"{from_node} vs {}",
			runtime.value()
		);
	}

	#[test]
	fn an_envelope_node_follows_the_same_follower_as_a_rack_slot() {
		let rate = 48_000;
		let mut params = BTreeMap::new();
		params.insert("attack", 10.0);
		params.insert("release", 100.0);

		let mut node = SourceNode::new(SourceKind::Envelope, rate, &params, &[]);
		let settings = node.settings.clone();

		let mut runtime =
			SourceRuntime::from_settings(SourceKind::Envelope, false, settings, rate, 0);
		let from_node = run(&mut node, &mut block(rate), 1.0);

		for _ in 0..BLOCK {
			runtime.advance(1.0, &infos(), rate, &[]);
		}

		assert!(
			(from_node - runtime.value()).abs() < 1e-6,
			"{from_node} vs {}",
			runtime.value()
		);
	}
}
