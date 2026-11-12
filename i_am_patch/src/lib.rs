//! The patch graph's reusable building blocks.
//!
//! The patcher and the rack share these: the LR4 crossover, the control-rate
//! modulation sources and their runtime. Nothing here knows about a graph, a
//! page or a host yet; those live in the products.

pub mod block;
pub mod catalog;
pub mod compile;
pub mod crossover;
pub mod edit;
pub mod engine;
pub mod graph;
pub mod modulation;
pub mod nodes;
pub mod poly;
pub mod scope;
mod source_dsp;
pub mod sources;
pub mod synths;
pub mod text;

/// How a node takes a file: the products' own decoders, re-exported so the crate
/// that hosts them does not need a dependency of its own.
pub use i_am_synths::{
	edit_slices, edit_waves, generate_slices, import_sample, import_waves, sample_preview,
	waves_preview,
};

#[cfg(test)]
/// A graph with one note voice, used by more than one module's tests.
pub(crate) fn voice_patch() -> crate::graph::Graph {
	use crate::graph::Graph;

	let mut graph = Graph::new();
	let midi = graph.add_node("input:midi", (0.0, 0.0)).expect("midi");
	let voice_in = graph.add_node("voice:in", (50.0, 0.0)).expect("voice in");
	let generator = graph.add_node("synth:additive", (100.0, 0.0)).expect("gen");
	let filter = graph.add_node("mod:filter", (200.0, 0.0)).expect("filter");
	let lfo = graph.add_node("lfo", (200.0, 100.0)).expect("lfo");
	let voice_out = graph
		.add_node("voice:out", (300.0, 0.0))
		.expect("voice out");
	let out = graph.add_node("out:audio", (400.0, 0.0)).expect("out");
	graph.connect((midi, 0), (voice_in, 0));
	graph.connect((voice_in, 0), (generator, 0));
	graph.connect((generator, 0), (filter, 0));
	graph.connect((filter, 0), (voice_out, 0));
	graph.connect((voice_out, 0), (out, 0));
	graph.modulate((lfo, 0), (filter, "cutoff"), 400.0);

	graph
}

/// Where a page reads the scripts the running nodes own, by node id.
pub type Scripts = std::sync::Arc<
	std::sync::Mutex<
		std::collections::HashMap<u32, std::sync::Arc<dyn i_am_effects::ScriptSource>>,
	>,
>;

/// What a page hands to the nodes: the things a graph parameter cannot carry.
///
/// A fitted table, a sample, a wavetable and an impulse response are all
/// serialized values, which a graph parameter cannot hold, so the page's upload
/// routes write them into slots here. The node is built with the slot the page
/// already holds, so an import lands on the next block and the graph does not
/// have to be rebuilt for it.
#[derive(Clone, Default)]
pub struct Slots {
	fitted: std::sync::Arc<
		std::sync::Mutex<std::collections::BTreeMap<u32, i_am_dsp::prelude::TableSlot>>,
	>,
	samples: std::sync::Arc<
		std::sync::Mutex<
			std::collections::BTreeMap<
				u32,
				std::sync::Arc<std::sync::Mutex<i_am_dsp::prelude::SampleSlot>>,
			>,
		>,
	>,
	waves: std::sync::Arc<
		std::sync::Mutex<
			std::collections::BTreeMap<
				u32,
				std::sync::Arc<std::sync::Mutex<i_am_dsp::prelude::WaveSlot>>,
			>,
		>,
	>,
	irs: std::sync::Arc<
		std::sync::Mutex<std::collections::BTreeMap<u32, std::sync::Arc<i_am_effects::IrSlot>>>,
	>,
}

impl Slots {
	/// The fitted table of one node, created the first time it is asked for.
	pub fn fitted(&self, node: u32) -> i_am_dsp::prelude::TableSlot {
		self.fitted
			.lock()
			.unwrap_or_else(|error| error.into_inner())
			.entry(node)
			.or_default()
			.clone()
	}

	/// The sample slot of one node, created the first time it is asked for.
	pub fn sample(
		&self,
		node: u32,
	) -> std::sync::Arc<std::sync::Mutex<i_am_dsp::prelude::SampleSlot>> {
		self.samples
			.lock()
			.unwrap_or_else(|error| error.into_inner())
			.entry(node)
			.or_default()
			.clone()
	}

	/// The wavetable slot of one node, created the first time it is asked for.
	pub fn waves(
		&self,
		node: u32,
	) -> std::sync::Arc<std::sync::Mutex<i_am_dsp::prelude::WaveSlot>> {
		self.waves
			.lock()
			.unwrap_or_else(|error| error.into_inner())
			.entry(node)
			.or_default()
			.clone()
	}

	/// The impulse response slot of one node, created the first time it is
	/// asked for.
	pub fn ir(&self, node: u32) -> std::sync::Arc<i_am_effects::IrSlot> {
		self.irs
			.lock()
			.unwrap_or_else(|error| error.into_inner())
			.entry(node)
			.or_default()
			.clone()
	}
}
