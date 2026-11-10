//! The node implementations the catalog names.

use std::collections::BTreeMap;

use i_am_dsp::prelude::{Effect, Message, NoteEvent, Parameter, SetValue, Value};
use i_am_plugin::Plugin;
use std::sync::atomic::Ordering;

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
	fn set_param_block(&mut self, name: &'static str, _start: f32, end: f32, _frames: usize) {
		self.set_param(name, end);
	}

	/// Where a source's cycle is, for the card's playhead.
	fn phase(&self) -> Option<f32> {
		None
	}
}

/// What kind of value a parameter takes, from its declared value.
#[derive(Clone, Copy, PartialEq)]
enum ParamKind {
	Float,
	Int,
	Bool,
}

fn kind_of(value: &Value) -> ParamKind {
	match value {
		Value::Int { .. } => ParamKind::Int,
		Value::Bool(_) => ParamKind::Bool,
		_ => ParamKind::Float,
	}
}

/// The value a graph stores is always a number; the DSP may want an int or a
/// bool, and a map write of the wrong type is refused.
fn set_value(kind: ParamKind, value: f32) -> SetValue {
	match kind {
		ParamKind::Int => SetValue::Int(value.round() as i32),
		ParamKind::Bool => SetValue::Bool(value != 0.0),
		ParamKind::Float => SetValue::Float(value),
	}
}

fn parameter_kinds(parameters: &[Parameter]) -> BTreeMap<String, ParamKind> {
	parameters
		.iter()
		.map(|parameter| (parameter.identifier.clone(), kind_of(&parameter.value)))
		.collect()
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

/// A rack module, driven by its own product schema.
struct ProductNode {
	effect: Box<dyn Effect<2>>,
	/// Parameters that move across the current block: name, start, end.
	ramps: Vec<(&'static str, f32, f32)>,
	/// The value type each parameter declares.
	kinds: BTreeMap<String, ParamKind>,
}

impl ProductNode {
	fn new(effect: Box<dyn Effect<2>>) -> Self {
		let kinds = parameter_kinds(&effect.get_parameters());

		Self {
			effect,
			ramps: Vec::with_capacity(8),
			kinds,
		}
	}
}

impl BlockNode for ProductNode {
	fn set_param(&mut self, name: &str, value: f32) {
		let kind = self.kinds.get(name).copied().unwrap_or(ParamKind::Float);
		self.effect.set_parameter(name, set_value(kind, value));
	}

	fn set_param_block(&mut self, name: &'static str, start: f32, end: f32, _frames: usize) {
		// The buffer is allocated once, so a modulated block does not allocate.
		self.ramps.push((name, start, end));
	}

	fn process(&mut self, context: &mut BlockContext, inputs: &[Buffer], outputs: &mut [Buffer]) {
		let input = audio(inputs, 0);
		let sidechain = inputs.get(1).and_then(Buffer::audio).copied();
		let Self {
			effect,
			ramps,
			kinds,
		} = self;

		if let Some(Buffer::Audio(out)) = outputs.first_mut() {
			out.clear();
			out.frames = context.frames;
			let frames = context.frames.max(1) as f32;

			for frame in 0..context.frames {
				let unit = frame as f32 / frames;

				for &(name, start, end) in ramps.iter() {
					let kind = kinds.get(name).copied().unwrap_or(ParamKind::Float);
					let value = match kind {
						ParamKind::Float => start + (end - start) * unit,
						_ => end,
					};

					effect.set_parameter(name, set_value(kind, value));
				}

				let mut samples = [input.channels[0][frame], input.channels[1][frame]];
				// The product's chooser counts the main input as port 0, so the
				// sidechain frame sits at index 1; index 0 mirrors it.
				let side =
					sidechain.map(|block| [block.channels[0][frame], block.channels[1][frame]]);
				let other = side.map(|frame| [frame, frame]);
				let other: &[&[f32; 2]] = match &other {
					Some(frames) => &[&frames[0], &frames[1]],
					None => &[],
				};

				effect.process(&mut samples, other, &mut context.context);
				out.channels[0][frame] = samples[0];
				out.channels[1][frame] = samples[1];
			}

			ramps.clear();
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
	/// The value type each parameter declares.
	kinds: BTreeMap<String, ParamKind>,
	/// Whether the map moved since the last block.
	dirty: bool,
}

impl<P: Default + Plugin> SynthNode<P> {
	fn new(rate: usize) -> Self {
		let mut inner = P::default();
		inner.on_activate(rate);
		let kinds = parameter_kinds(&inner.get_parameters());

		Self {
			inner,
			kinds,
			dirty: false,
		}
	}
}

impl<P: Default + Plugin> BlockNode for SynthNode<P> {
	fn set_param(&mut self, name: &str, value: f32) {
		// A synth keeps its parameters in a shared map and reads them back on the
		// next block: write the map, then push it into the effect. The map refuses
		// a value of the wrong type, so the parameter's declared type decides it.
		let kind = self.kinds.get(name).copied().unwrap_or(ParamKind::Float);
		let map = self.inner.param_map();
		map.set(name, set_value(kind, value), Ordering::Relaxed);
		self.dirty = true;
	}

	fn process(&mut self, context: &mut BlockContext, _: &[Buffer], outputs: &mut [Buffer]) {
		// Every parameter was written into the map before this block; one sync
		// carries them all into the DSP.
		if self.dirty {
			self.inner.sync_params();
			self.dirty = false;
		}

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
	fn a_parameter_takes_the_value_type_its_dsp_declares() {
		assert_eq!(
			super::set_value(super::ParamKind::Int, 2.6),
			SetValue::Int(3)
		);
		assert_eq!(
			super::set_value(super::ParamKind::Bool, 0.0),
			SetValue::Bool(false)
		);
		assert_eq!(
			super::set_value(super::ParamKind::Float, 0.5),
			SetValue::Float(0.5)
		);
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
