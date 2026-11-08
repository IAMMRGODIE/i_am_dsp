//! The block engine: runs a compiled plan one block at a time.

use i_am_dsp::prelude::NoteEvent;

use crate::block::{AudioBlock, BLOCK, BlockContext, Buffer};
use crate::compile::{CompileError, Plan, compile};
use crate::graph::{Graph, NodeId, SignalKind};
use crate::nodes::{BlockNode, make};

/// Why an engine could not be built.
#[derive(Debug)]
pub enum EngineError {
	/// The graph did not compile.
	Compile(CompileError),
	/// The catalog has no implementation for a kind.
	UnknownKind(String),
}

impl From<CompileError> for EngineError {
	fn from(error: CompileError) -> Self {
		Self::Compile(error)
	}
}

/// One running patch.
pub struct Engine {
	rate: usize,
	block: u64,
	order: Vec<NodeId>,
	kinds: Vec<&'static str>,
	nodes: Vec<Box<dyn BlockNode>>,
	outputs: Vec<Vec<Buffer>>,
	inputs: Vec<Vec<Buffer>>,
	edges: Vec<Vec<(usize, usize, usize)>>,
	output_step: Option<usize>,
}

impl Engine {
	/// Compiles a graph and builds its nodes.
	pub fn new(graph: &Graph, rate: usize) -> Result<Self, EngineError> {
		let plan: Plan = compile(graph)?;
		let steps: std::collections::BTreeMap<NodeId, usize> = plan
			.order
			.iter()
			.enumerate()
			.map(|(step, id)| (*id, step))
			.collect();

		let mut kinds = Vec::with_capacity(plan.order.len());
		let mut nodes = Vec::with_capacity(plan.order.len());
		let mut outputs = Vec::with_capacity(plan.order.len());
		let mut inputs = Vec::with_capacity(plan.order.len());
		let mut edges = vec![Vec::new(); plan.order.len()];
		let mut output_step = None;

		for (step, id) in plan.order.iter().enumerate() {
			let instance = graph
				.get(*id)
				.expect("the plan only names nodes in the graph");
			let kind = instance.kind;
			kinds.push(kind.id);
			nodes.push(make(kind.id, rate).ok_or_else(|| EngineError::UnknownKind(kind.id.to_string()))?);

			let mut node_outputs = Vec::with_capacity(kind.outputs.len());
			for port in kind.outputs {
				node_outputs.push(match port.kind {
					SignalKind::Audio(_) => Buffer::Audio(AudioBlock::silent(BLOCK)),
					SignalKind::Control => Buffer::Control(0.0),
					SignalKind::Midi => Buffer::Midi,
				});
			}
			outputs.push(node_outputs);

			let mut node_inputs = Vec::with_capacity(kind.inputs.len());
			for port in kind.inputs {
				node_inputs.push(match port.kind {
					SignalKind::Audio(_) => Buffer::Audio(AudioBlock::silent(BLOCK)),
					SignalKind::Control => Buffer::Control(0.0),
					SignalKind::Midi => Buffer::Midi,
				});
			}
			inputs.push(node_inputs);

			if kind.id == "out:audio" && output_step.is_none() {
				output_step = Some(step);
			}
		}

		for edge in &graph.edges {
			let from = steps[&edge.from.0];
			let to = steps[&edge.to.0];
			edges[to].push((edge.to.1, from, edge.from.1));
		}

		Ok(Self {
			rate,
			block: 0,
			order: plan.order,
			kinds,
			nodes,
			outputs,
			inputs,
			edges,
			output_step,
		})
	}

	/// Runs one block: the host input goes in, the graph's output comes out.
	pub fn process(&mut self, input: &[[f32; 2]], events: &[NoteEvent], out: &mut [[f32; 2]]) {
		let frames = input.len().min(BLOCK).min(out.len());
		let mut context = BlockContext {
			frames,
			rate: self.rate,
			block: self.block,
			events: events.to_vec(),
		};

		self.block = self.block.wrapping_add(1);

		// The audio input node is the host's buffer; the sidechain is silence
		// unless the wrapper fills it.
		for step in 0..self.order.len() {
			match self.kinds[step] {
				"input:audio" => {
					if let Some(Buffer::Audio(block)) = self.outputs[step].first_mut() {
						block.clear();
						block.frames = frames;

						for (frame, sample) in input.iter().take(frames).enumerate() {
							block.channels[0][frame] = sample[0];
							block.channels[1][frame] = sample[1];
						}
					}
				}
				"input:sidechain" => {
					if let Some(Buffer::Audio(block)) = self.outputs[step].first_mut() {
						block.clear();
						block.frames = frames;
					}
				}
				_ => {}
			}
		}

		for step in 0..self.order.len() {
			for (port, from, from_port) in &self.edges[step] {
				self.inputs[step][*port] = self.outputs[*from][*from_port];
			}

			let node = &mut self.nodes[step];
			let inputs = &self.inputs[step];
			let outputs = &mut self.outputs[step];

			node.process(&mut context, inputs, outputs);
		}

		for sample in out.iter_mut().take(frames) {
			*sample = [0.0; 2];
		}

		if let Some(step) = self.output_step
			&& let Some(Buffer::Audio(block)) = self.inputs[step].first()
		{
			for (frame, sample) in out.iter_mut().take(frames).enumerate() {
				*sample = block.frame(frame);
			}
		}
	}
}


#[cfg(test)]
mod tests {
	use super::*;
	use crate::graph::Graph;

	fn render(graph: &Graph) -> f32 {
		let mut engine = Engine::new(graph, 48_000).expect("the graph builds");
		let mut peak = 0.0f32;

		for block in 0..8 {
			let events = if block == 0 {
				vec![NoteEvent::NoteOn {
					channel: 0,
					note: 60,
					velocity: 1.0,
				}]
			} else {
				Vec::new()
			};
			let input = vec![[0.0f32; 2]; BLOCK];
			let mut out = vec![[0.0f32; 2]; BLOCK];

			engine.process(&input, &events, &mut out);

			for frame in out {
				peak = peak.max(frame[0].abs()).max(frame[1].abs());
			}
		}

		peak
	}

	#[test]
	fn a_chain_renders_a_note() {
		let mut graph = Graph::new();
		let midi = graph.add_node("input:midi", (0.0, 0.0)).expect("midi");
		let generator = graph.add_node("gen:additive", (100.0, 0.0)).expect("gen");
		let gain = graph.add_node("gain", (200.0, 0.0)).expect("gain");
		let out = graph.add_node("out:audio", (300.0, 0.0)).expect("out");
		graph.connect((midi, 0), (generator, 0));
		graph.connect((generator, 0), (gain, 0));
		graph.connect((gain, 0), (out, 0));

		let peak = render(&graph);

		assert!(peak > 1e-3, "the chain was silent: {peak}");
	}

	#[test]
	fn mid_and_side_can_be_summed_back() {
		let mut graph = Graph::new();
		let midi = graph.add_node("input:midi", (0.0, 0.0)).expect("midi");
		let generator = graph.add_node("gen:additive", (100.0, 0.0)).expect("gen");
		let split = graph.add_node("split:ms", (200.0, 0.0)).expect("split");
		let mixer = graph.add_node("mixer", (300.0, 0.0)).expect("mixer");
		let out = graph.add_node("out:audio", (400.0, 0.0)).expect("out");

		graph.connect((midi, 0), (generator, 0));
		graph.connect((generator, 0), (split, 0));
		graph.connect((split, 0), (mixer, 0));
		graph.connect((split, 1), (mixer, 1));
		graph.connect((mixer, 0), (out, 0));

		let peak = render(&graph);

		assert!(peak > 1e-3, "the split/mixer chain was silent: {peak}");
	}

	#[test]
	fn a_filtered_chain_still_sounds() {
		let mut graph = Graph::new();
		let midi = graph.add_node("input:midi", (0.0, 0.0)).expect("midi");
		let generator = graph.add_node("gen:additive", (100.0, 0.0)).expect("gen");
		let filter = graph.add_node("filter", (200.0, 0.0)).expect("filter");
		let out = graph.add_node("out:audio", (300.0, 0.0)).expect("out");
		graph.connect((midi, 0), (generator, 0));
		graph.connect((generator, 0), (filter, 0));
		graph.connect((filter, 0), (out, 0));

		assert!(render(&graph) > 1e-3);
	}
}
