//! The patch graph's reusable building blocks.
//!
//! The patcher and the rack share these: the LR4 crossover, the control-rate
//! modulation sources and their runtime. Nothing here knows about a graph, a
//! page or a host yet; those live in the products.

pub mod block;
pub mod compile;
pub mod crossover;
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

#[cfg(test)]
/// A graph with one note voice, used by more than one module's tests.
pub(crate) fn voice_patch() -> crate::graph::Graph {
	use crate::graph::Graph;

	let mut graph = Graph::new();
	let midi = graph.add_node("input:midi", (0.0, 0.0)).expect("midi");
	let voice_in = graph.add_node("voice:in", (50.0, 0.0)).expect("voice in");
	let generator = graph.add_node("gen:additive", (100.0, 0.0)).expect("gen");
	let filter = graph.add_node("filter", (200.0, 0.0)).expect("filter");
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
