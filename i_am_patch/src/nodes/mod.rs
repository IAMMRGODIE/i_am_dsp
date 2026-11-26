//! The node implementations the catalog names.

use std::collections::BTreeMap;

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64};

use i_am_dsp::prelude::{
	AdditiveOsc, Adsr, BendedSawGen, Effect, EqualTemperament, FmIntegral, Generator, Message,
	NoteEvent, OscTable, Oscillator, Parameter, Parameters, SetValue, TableBankOsc, TableOsc,
	Value, WaveFramesOsc, WaveTable, bend_phase, mirror_phase, quantize_phase, sync_phase,
};
use i_am_plugin::Plugin;
use i_am_synths::{CanvasOscillator, ScriptOscillator};
use std::sync::atomic::Ordering;

use crate::block::{AudioBlock, BLOCK, BlockContext, Buffer};
use crate::crossover::Crossover;
use crate::modulation::SourceRuntime;
use crate::sources::{SourceKind, SourceSettings, lfo_preset};

mod adsr;
mod oscillator;
mod product;
mod routing;
mod source;
mod synth;
mod table;

#[cfg(test)]
mod tests;

use adsr::AdsrNode;
use oscillator::{OscShapeNode, TableShapeNode};
use product::ProductNode;
use routing::{BandMix, BandSplit, Delay, MixLr, MixMs, Mixer, Source, SplitLr, SplitMs};
use source::{MacroNode, MapNode, ModNode, SourceNode, macro_index};
use synth::SynthNode;
use table::{ProductTableNode, WarpNode};

pub(crate) use oscillator::build_oscillator;
pub(crate) use table::build_table_node;

/// Whether a kind builds a table or an oscillator instead of running audio.
pub(crate) fn is_constructor(kind: &str) -> bool {
	crate::graph::is_constructor_kind(kind)
}

/// What a constructor node hands the node that consumes it.
pub enum ConstructorSpec {
	/// A single-cycle table.
	Table(Box<dyn WaveTable + Send + Sync>),
	/// A running oscillator.
	Osc(Box<dyn Oscillator<2> + Send + Sync>),
	/// A warp over a table from upstream.
	Warp {
		/// Which warp.
		mode: WarpMode,
		/// The strength, shared with the node's parameter.
		amount: Arc<AtomicU32>,
	},
	/// A modulation that needs its carrier and modulator from upstream.
	Mod {
		/// Which modulation.
		kind: ModKind,
		/// The depth, shared with the node's parameters.
		depth: Arc<AtomicU32>,
		/// The modulator ratio, shared with the node's parameters.
		ratio: Arc<AtomicU32>,
	},
}

/// The warps a patch can put between two tables.
#[derive(Clone, Copy)]
pub enum WarpMode {
	/// Pawn the phase towards the start or the end.
	Bend,
	/// Fold the second half of the cycle back over the first.
	Mirror,
	/// Repeat the table inside one cycle.
	Sync,
	/// Snap the read position to a grid.
	Quantize,
	/// Read a second table as the phase map.
	Remap,
}

/// The four modulations a patch can put between two tables.
#[derive(Clone, Copy)]
pub enum ModKind {
	/// Amplitude modulation: the carrier scaled by the modulator.
	Am,
	/// Ring modulation: the product of the two tables.
	Rm,
	/// Phase modulation: the modulator bends the carrier's phase.
	Pm,
	/// Frequency modulation: the integral of the modulator bends the phase.
	Fm,
}

/// One node in the running graph.
pub trait BlockNode: Send + Sync {
	/// The table, oscillator or modulation this node builds, when it is a constructor.
	fn constructor_spec(&self) -> Option<ConstructorSpec> {
		None
	}

	/// Takes the oscillator a downstream envelope asked for, when this node is one.
	fn install(&mut self, _oscillator: Box<dyn Oscillator<2> + Send + Sync>) -> bool {
		false
	}

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

	/// Where a MIDI source's input sits, 0..1, for its mapping picture.
	fn midi_position(&self) -> Option<f32> {
		None
	}

	/// The script this node runs, when its product has one. The page edits it
	/// through the patch product's /script route.
	fn script_source(&self) -> Option<std::sync::Arc<dyn i_am_effects::ScriptSource>> {
		None
	}

	/// Drops the tail a reverb or delay is holding, for the card's clear button.
	fn clear_tail(&mut self) {}

	/// Tells the node the note the next one should glide from.
	///
	/// A voice engine is built fresh for every note, so the envelope inside a synth
	/// node has never seen the note before it; the voice hands it over here just
	/// before the note-on reaches the node.
	fn glide_from(&mut self, _channel: u8, _note: usize) {}
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

/// Builds the node for a catalog kind at a sample rate, with empty slots.
pub fn make(
	kind: &str,
	rate: usize,
	params: &BTreeMap<&'static str, f32>,
	points: &[crate::graph::Point],
) -> Option<Box<dyn BlockNode>> {
	make_with(kind, 0, rate, params, points, &[], &crate::Slots::default())
}

/// Builds the node for a catalog kind over the slots the page already holds.
pub fn make_with(
	kind: &str,
	node: u32,
	rate: usize,
	params: &BTreeMap<&'static str, f32>,
	points: &[crate::graph::Point],
	velocity_points: &[crate::graph::Point],
	slots: &crate::Slots,
) -> Option<Box<dyn BlockNode>> {
	let node: Box<dyn BlockNode> = match kind {
		"input:midi" => Box::new(Quiet),
		"input:audio" | "input:sidechain" => Box::new(Source),
		"out:audio" => Box::new(Quiet),
		// A note is text on the canvas: it is built like any node so the graph
		// compiles, and it does nothing, because there is nothing for it to do.
		"text" => Box::new(Quiet),
		"voice:in" => Box::new(Quiet),
		"voice:out" => Box::new(Quiet),
		"split:ms" => Box::new(SplitMs),
		"split:lr" => Box::new(SplitLr),
		"split:bands" => Box::new(BandSplit::new(rate, params)),
		"mix:ms" => Box::new(MixMs),
		"mix:lr" => Box::new(MixLr),
		"mix:bands" => Box::new(BandMix {
			low: param(params, "low", 1.0),
			mid: param(params, "mid", 1.0),
			high: param(params, "high", 1.0),
		}),
		"mixer" => Box::new(Mixer {
			a: param(params, "a", 1.0),
			b: param(params, "b", 1.0),
		}),
		"lfo" => Box::new(SourceNode::new(SourceKind::Lfo, rate, params, points)),
		"macro" => Box::new(MacroNode {
			macros: slots.macros().clone(),
			index: macro_index(params),
		}),
		"random" => Box::new(SourceNode::new(SourceKind::Random, rate, params, &[])),
		"table" => Box::new(SourceNode::new(SourceKind::Table, rate, params, &[])),
		"pitch" => Box::new(SourceNode::new(SourceKind::Pitch, rate, params, &[])),
		"midi" => {
			let mut source = SourceNode::new(SourceKind::Midi, rate, params, points);
			source.set_velocity_points(velocity_points);
			Box::new(source)
		}
		"map" => Box::new(MapNode {
			points: points.to_vec(),
		}),
		"envelope" => Box::new(SourceNode::new(SourceKind::Envelope, rate, params, &[])),
		"delay" => Box::new(Delay::new()),
		"table:shape" => Box::new(TableShapeNode::new(params)),
		"osc:shape" => Box::new(OscShapeNode::new(params)),
		"adsr" => Box::new(AdsrNode::new(rate)),
		"osc:am" => Box::new(ModNode::new(ModKind::Am, params)),
		"osc:rm" => Box::new(ModNode::new(ModKind::Rm, params)),
		"osc:pm" => Box::new(ModNode::new(ModKind::Pm, params)),
		"osc:fm" => Box::new(ModNode::new(ModKind::Fm, params)),
		"table:additive" => Box::new(ProductTableNode::new(
			|| {
				Box::new(OscTable::<2, _>(AdditiveOsc::<BendedSawGen, 64, 2>::new(
					BendedSawGen::default(),
					32,
				)))
			},
			params,
		)),
		"table:canvas" => {
			let handle = slots.canvas(node);

			Box::new(ProductTableNode::new(
				move || Box::new(CanvasOscillator::new(handle.clone())),
				params,
			))
		}
		"table:spectrum" => {
			let handle = slots.spectrum(node);

			Box::new(ProductTableNode::new(
				move || Box::new(OscTable::<2, _>(TableBankOsc::new(handle.feed()))),
				params,
			))
		}
		"table:script" => {
			let handle = slots.script(node);

			Box::new(ProductTableNode::new(
				move || {
					Box::new(OscTable::<2, _>(ScriptOscillator::new(
						rate,
						handle.clone(),
					)))
				},
				params,
			))
		}
		"table:shapes" => {
			let slot = slots.waves(node);

			Box::new(ProductTableNode::new(
				move || Box::new(OscTable::<2, _>(WaveFramesOsc::with_slot(slot.clone()))),
				params,
			))
		}
		"warp:bend" => Box::new(WarpNode::new(WarpMode::Bend, "power", params)),
		"warp:mirror" => Box::new(WarpNode::new(WarpMode::Mirror, "amount", params)),
		"warp:sync" => Box::new(WarpNode::new(WarpMode::Sync, "ratio", params)),
		"warp:quantize" => Box::new(WarpNode::new(WarpMode::Quantize, "steps", params)),
		"warp:remap" => Box::new(WarpNode::new(WarpMode::Remap, "amount", params)),
		kind if kind.starts_with("synth:") => {
			use i_am_synths::{
				AdditiveWeb, CanvasWeb, DrummerWeb, FittedWeb, SamplerWeb, ScriptWeb, ShapesWeb,
				SpectrumWeb, StringWeb,
			};

			match &kind[6..] {
				"additive" => Box::new(SynthNode::<AdditiveWeb>::new(rate)),
				"canvas" => Box::new(SynthNode::with_plugin(
					CanvasWeb::with_handle(slots.canvas(node)),
					rate,
				)),

				"drummer" => Box::new(SynthNode::with_plugin(
					DrummerWeb::with_handle(slots.drummer(node)),
					rate,
				)),

				"fitted" => Box::new(SynthNode::with_plugin(
					FittedWeb::with_slot(slots.fitted(node)),
					rate,
				)),
				// The sampler and the shapes keep their sample and their table
				// in slots the page owns, so a drop on a patch card reaches the
				// voice without a rebuild.
				"sampler" => Box::new(SynthNode::with_plugin(
					SamplerWeb::with_slot(slots.sample(node)),
					rate,
				)),
				"script" => {
					let handle = slots.script(node);
					let inner = ScriptWeb::with_handle(handle.clone());
					let feed = handle.source();

					Box::new(SynthNode::with_script(inner, feed, rate))
				}
				"shapes" => Box::new(SynthNode::with_plugin(
					ShapesWeb::with_slot(slots.waves(node)),
					rate,
				)),
				"spectrum" => Box::new(SynthNode::with_plugin(
					SpectrumWeb::with_handle(slots.spectrum(node)),
					rate,
				)),

				"strings" => Box::new(SynthNode::<StringWeb>::new(rate)),
				_ => return None,
			}
		}
		kind if kind.starts_with("mod:") => {
			let module = i_am_effects::catalog::get(&kind[4..])?;

			match module.id {
				// The scripted effects run the worker the page edits: a fresh one
				// per build would lose every edit at the next structural change.
				"script_mapper" => {
					let Some(crate::ScriptSlot::Mapper(worker)) =
						slots.script_slot(node, "mod:script_mapper")
					else {
						return None;
					};
					let effect =
						Box::new(i_am_effects::ScriptMapperEffect::new(rate, worker.clone()));

					Box::new(ProductNode::with_script(effect, worker))
				}
				// The curves are the product's state, so the node is built over the
				// handle the card edits.
				"spectral_dynamics" => {
					let handle = slots.spectral(node);

					Box::new(ProductNode::new(i_am_effects::spectral_dynamics_with(
						&handle, rate,
					)))
				}
				"filter_table" => {
					let handle = slots.filter_table(node);
					Box::new(ProductNode::new(Box::new(
						i_am_dsp::prelude::FilterTableEffect::with_frame(rate, handle.frame()),
					)))
				}
				"history" => {
					let handle = slots.history(node);
					Box::new(ProductNode::new(Box::new(
						i_am_dsp::prelude::HistoryEffect::with_feed(rate, handle.feed()),
					)))
				}
				"script_distortion" => {
					let Some(crate::ScriptSlot::Distortion(worker)) =
						slots.script_slot(node, "mod:script_distortion")
					else {
						return None;
					};
					let effect =
						Box::new(i_am_effects::ScriptDistortionEffect::new(worker.clone()));

					Box::new(ProductNode::with_script(effect, worker))
				}
				// The OTT and the vocoder publish band meters, and the card's hero
				// draws them: the node is built with the handle the page reads, or
				// the picture is an empty grid nothing ever writes to.
				"ott" => Box::new(ProductNode::new(Box::new(i_am_effects::OttEffect::new(
					rate,
					slots.ott(node),
				)))),
				"vocoder" => Box::new(ProductNode::new(Box::new(
					i_am_dsp::prelude::VocoderEffect::<2>::new(rate, slots.vocoder(node)),
				))),
				// The convolver is the one module whose state is a slot the page
				// owns: a response dropped on its card lands in the same slot
				// the effect reads.
				_ => match module.build_with_ir {
					Some(build) => Box::new(ProductNode::new(build(rate, slots.ir(node)))),
					None => Box::new(ProductNode::new((module.build)(rate))),
				},
			}
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
