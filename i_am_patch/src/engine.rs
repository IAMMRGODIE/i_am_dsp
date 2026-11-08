//! The block engine: runs a compiled plan one block at a time.

use std::collections::BTreeMap;

use i_am_dsp::prelude::NoteEvent;

use crate::block::{AudioBlock, BLOCK, BlockContext, Buffer, PatchContext};
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
	/// The graph has no voice group to make polyphonic.
	NoVoiceGroup,
}

impl From<CompileError> for EngineError {
	fn from(error: CompileError) -> Self {
		Self::Compile(error)
	}
}

/// One control assignment, resolved to steps.
struct Link {
	source: usize,
	port: usize,
	target: usize,
	param: &'static str,
	depth: f32,
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
	links: Vec<Link>,
	steps: std::collections::BTreeMap<NodeId, usize>,
	previous: Vec<Vec<f32>>,
	delta: Vec<Vec<f32>>,
	pure_control: Vec<bool>,
	retrigger: Vec<usize>,
	controls: Vec<Vec<f32>>,
	base: Vec<BTreeMap<&'static str, f32>>,
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
		let mut controls = Vec::with_capacity(plan.order.len());
		let mut base = Vec::with_capacity(plan.order.len());
		let mut output_step = None;

		for (step, id) in plan.order.iter().enumerate() {
			let instance = graph
				.get(*id)
				.expect("the plan only names nodes in the graph");
			let kind = instance.kind;
			kinds.push(kind.id);
			nodes.push(
				make(kind.id, rate, &instance.params)
					.ok_or_else(|| EngineError::UnknownKind(kind.id.to_string()))?,
			);

			let mut node_outputs = Vec::with_capacity(kind.outputs.len());
			for port in kind.outputs {
				node_outputs.push(match port.kind {
					SignalKind::Audio(_) => Buffer::Audio(AudioBlock::silent(BLOCK)),
					SignalKind::Control => Buffer::Control(0.0),
					SignalKind::Midi => Buffer::Midi,
				});
			}
			outputs.push(node_outputs);
			controls.push(vec![0.0; kind.outputs.len()]);
			base.push(instance.params.clone());

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

		let links: Vec<Link> = plan
			.modulations
			.iter()
			.map(|link| Link {
				source: steps[&link.source.0],
				port: link.source.1,
				target: steps[&link.target.0],
				param: link.target.1,
				depth: link.depth,
			})
			.collect();

		let retrigger: Vec<usize> = plan
			.retrigger
			.iter()
			.filter_map(|id| steps.get(id).copied())
			.collect();
		let step_map = steps.clone();

		let previous = controls.clone();
		let delta = controls.clone();
		let pure_control: Vec<bool> = plan
			.order
			.iter()
			.map(|id| {
				let kind = graph.get(*id).expect("a planned node").kind;
				kind.inputs.is_empty()
					&& kind
						.outputs
						.iter()
						.any(|port| port.kind == SignalKind::Control)
			})
			.collect();

		Ok(Self {
			rate,
			block: 0,
			order: plan.order,
			kinds,
			nodes,
			outputs,
			inputs,
			edges,
			links,
			steps: step_map,
			previous,
			delta,
			pure_control,
			retrigger,
			controls,
			base,
			output_step,
		})
	}

	/// Writes one node's output block, before the plan runs.
	///
	/// The poly wrapper uses this to place the summed voices on the voice
	/// boundary; the boundary node itself does no work.
	pub fn inject(&mut self, node: NodeId, block: AudioBlock) {
		if let Some(step) = self.steps.get(&node)
			&& let Some(Buffer::Audio(out)) = self.outputs[*step].first_mut()
		{
			*out = block;
		}
	}

	/// Renders a graph offline: the agent's loop, without a host.
	pub fn render_graph(
		graph: &Graph,
		rate: usize,
		seconds: f32,
		events: &[(u64, NoteEvent)],
	) -> Result<Vec<[f32; 2]>, EngineError> {
		let mut engine = Self::new(graph, rate)?;
		let blocks = (seconds.max(0.01) * rate as f32 / BLOCK as f32).ceil() as u64;
		let mut out = Vec::with_capacity(blocks as usize * BLOCK);

		for block in 0..blocks {
			let due: Vec<NoteEvent> = events
				.iter()
				.filter(|(at, _)| *at == block)
				.map(|(_, event)| event.clone())
				.collect();
			let input = vec![[0.0f32; 2]; BLOCK];
			let mut frame = vec![[0.0f32; 2]; BLOCK];

			engine.process(&input, &due, &mut frame);
			out.extend(frame);
		}

		Ok(out)
	}

	/// Runs one block: the host input goes in, the graph's output comes out.
	pub fn process(&mut self, input: &[[f32; 2]], events: &[NoteEvent], out: &mut [[f32; 2]]) {
		let frames = input.len().min(BLOCK).min(out.len());
		let mut context = BlockContext {
			frames,
			rate: self.rate,
			block: self.block,
			events: events.to_vec(),
			context: Box::new(PatchContext::new(self.rate)),
		};

		self.block = self.block.wrapping_add(1);

		// A note starts the voice group from scratch: the LFO, the filter and
		// the generator inside it all begin again, which is what makes a pluck
		// per note. Mono for now: the newest note takes the group over.
		if events
			.iter()
			.any(|event| matches!(event, NoteEvent::NoteOn { .. }))
		{
			for step in self.retrigger.clone() {
				if let Some(node) = make(self.kinds[step], self.rate, &self.base[step]) {
					self.nodes[step] = node;
				}

				self.controls[step].fill(0.0);
			}
		}

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

		// What the block started with: a ramp needs both ends.
		for (previous, controls) in self.previous.iter_mut().zip(&self.controls) {
			previous.copy_from_slice(controls);
		}

		// A control source with no audio input can run before the audio path, so
		// its value for this block is known while the parameters are set. The
		// modulation then ramps across the block instead of holding the last one.
		let mut done = vec![false; self.order.len()];

		for (step, pure) in self.pure_control.iter().copied().enumerate() {
			if !pure {
				continue;
			}

			let node = &mut self.nodes[step];
			let inputs = &self.inputs[step];
			let outputs = &mut self.outputs[step];

			node.process(&mut context, inputs, outputs);

			for (port, buffer) in self.outputs[step].iter().enumerate() {
				if let Buffer::Control(value) = buffer {
					self.controls[step][port] = *value;
				}
			}

			done[step] = true;
		}

		for step in 0..self.order.len() {
			let mut updates: Vec<(&'static str, f32, f32)> = Vec::new();

			for (name, base) in self.base[step].clone() {
				let mut start = base;
				let mut end = base;

				for link in &self.links {
					if link.target == step && link.param == name {
						let last = self.previous[link.source][link.port];
						let now = if done[link.source] {
							self.controls[link.source][link.port]
						} else {
							// An audio-driven source (the envelope) cannot run
							// before its input; the last step is the best estimate
							// of where it is going, clamped so a jump cannot
							// overshoot the block.
							last + self.delta[link.source][link.port].clamp(-0.25, 0.25)
						};

						start += link.depth * last;
						end += link.depth * now;
					}
				}

				updates.push((name, start, end));
			}

			for (name, start, end) in updates {
				self.nodes[step].set_param_block(name, start, end, frames);
			}
		}

		for (step, finished) in done.iter().copied().enumerate() {
			if finished {
				continue;
			}

			// An input sums every edge into it, so two wires into one port add.
			for buffer in &mut self.inputs[step] {
				*buffer = buffer.silent_of(frames);
			}

			for (port, from, from_port) in &self.edges[step] {
				let source = self.outputs[*from][*from_port];
				self.inputs[step][*port].add(&source);
			}

			let node = &mut self.nodes[step];
			let inputs = &self.inputs[step];
			let outputs = &mut self.outputs[step];

			node.process(&mut context, inputs, outputs);

			for (port, buffer) in self.outputs[step].iter().enumerate() {
				if let Buffer::Control(value) = buffer {
					self.controls[step][port] = *value;
				}
			}
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

	fn note_event(note: usize) -> NoteEvent {
		NoteEvent::NoteOn {
			channel: 0,
			note,
			velocity: 1.0,
		}
	}

	fn render_blocks(graph: &Graph, blocks: usize, notes: &[(u64, NoteEvent)]) -> Vec<[f32; 2]> {
		let mut engine = Engine::new(graph, 48_000).expect("the graph builds");
		let mut out = Vec::new();

		for block in 0..blocks as u64 {
			let events: Vec<NoteEvent> = notes
				.iter()
				.filter(|(at, _)| *at == block)
				.map(|(_, event)| event.clone())
				.collect();
			let input = vec![[0.0f32; 2]; BLOCK];
			let mut frame = vec![[0.0f32; 2]; BLOCK];

			engine.process(&input, &events, &mut frame);
			out.extend(frame);
		}

		out
	}

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

	fn voice_patch() -> Graph {
		let mut graph = Graph::new();
		let midi = graph.add_node("input:midi", (0.0, 0.0)).expect("midi");
		let voice_in = graph.add_node("voice:in", (50.0, 0.0)).expect("voice in");
		let generator = graph.add_node("gen:additive", (100.0, 0.0)).expect("gen");
		let filter = graph.add_node("filter", (200.0, 0.0)).expect("filter");
		let lfo = graph.add_node("lfo", (200.0, 100.0)).expect("lfo");
		let voice_out = graph.add_node("voice:out", (300.0, 0.0)).expect("voice out");
		let out = graph.add_node("out:audio", (400.0, 0.0)).expect("out");
		graph.connect((midi, 0), (voice_in, 0));
		graph.connect((voice_in, 0), (generator, 0));
		graph.connect((generator, 0), (filter, 0));
		graph.connect((filter, 0), (voice_out, 0));
		graph.connect((voice_out, 0), (out, 0));
		graph.modulate((lfo, 0), (filter, "cutoff"), 400.0);

		graph
	}

	#[test]
	fn a_voice_group_restarts_with_the_note() {
		let graph = voice_patch();
		let frames = render_blocks(
			&graph,
			8,
			&[(0, note_event(60)), (4, note_event(60))],
		);

		// The LFO, the filter and the generator all start again on the second
		// note, so the first block after each note is the same samples.
		assert_eq!(&frames[0..BLOCK], &frames[BLOCK * 4..BLOCK * 5]);
	}

	#[test]
	fn an_lfo_can_move_a_parameter() {
		let build = |modulated: bool| {
			let mut graph = Graph::new();
			let midi = graph.add_node("input:midi", (0.0, 0.0)).expect("midi");
			let generator = graph.add_node("gen:additive", (100.0, 0.0)).expect("gen");
			let gain_node = graph.add_node("gain", (200.0, 0.0)).expect("gain");
			let out = graph.add_node("out:audio", (300.0, 0.0)).expect("out");
			graph.connect((midi, 0), (generator, 0));
			graph.connect((generator, 0), (gain_node, 0));
			graph.connect((gain_node, 0), (out, 0));

			if modulated {
				let lfo = graph.add_node("lfo", (100.0, 100.0)).expect("lfo");
				graph.set_param(lfo, "rate", 120.0);
				graph.modulate((lfo, 0), (gain_node, "gain"), 0.6);
			}

			graph
		};

		let plain = render(&build(false));
		let moved = render(&build(true));

		assert!(plain != moved, "the modulation did nothing");
	}

	#[test]
	fn the_gain_parameter_is_heard() {
		let chain = |gain: f32| {
			let mut graph = Graph::new();
			let midi = graph.add_node("input:midi", (0.0, 0.0)).expect("midi");
			let generator = graph.add_node("gen:additive", (100.0, 0.0)).expect("gen");
			let gain_node = graph.add_node("gain", (200.0, 0.0)).expect("gain");
			let out = graph.add_node("out:audio", (300.0, 0.0)).expect("out");
			graph.connect((midi, 0), (generator, 0));
			graph.connect((generator, 0), (gain_node, 0));
			graph.connect((gain_node, 0), (out, 0));
			assert!(graph.set_param(gain_node, "gain", gain));

			render(&graph)
		};

		assert!(chain(0.0) < 1e-6, "a zero gain still sounded");
		assert!(chain(0.5) < chain(1.0) * 0.7, "the gain did nothing");
	}

	#[test]
	fn the_voice_has_no_dc() {
		let mut graph = Graph::new();
		let midi = graph.add_node("input:midi", (0.0, 0.0)).expect("midi");
		let generator = graph.add_node("gen:additive", (100.0, 0.0)).expect("gen");
		let out = graph.add_node("out:audio", (200.0, 0.0)).expect("out");
		graph.connect((midi, 0), (generator, 0));
		graph.connect((generator, 0), (out, 0));

		let frames = Engine::render_graph(
			&graph,
			48_000,
			1.0,
			&[(0, NoteEvent::NoteOn { channel: 0, note: 60, velocity: 1.0 })],
		)
		.expect("renders");
		let mean: f64 = frames.iter().map(|frame| frame[0] as f64).sum::<f64>() / frames.len() as f64;

		assert!(mean.abs() < 0.01, "the voice carries a DC offset: {mean}");
	}

	#[test]
	fn the_offline_render_produces_a_note() {
		let mut graph = Graph::new();
		let midi = graph.add_node("input:midi", (0.0, 0.0)).expect("midi");
		let generator = graph.add_node("gen:additive", (100.0, 0.0)).expect("gen");
		let out = graph.add_node("out:audio", (200.0, 0.0)).expect("out");
		graph.connect((midi, 0), (generator, 0));
		graph.connect((generator, 0), (out, 0));

		let frames = Engine::render_graph(
			&graph,
			48_000,
			0.1,
			&[(
				0,
				NoteEvent::NoteOn {
					channel: 0,
					note: 60,
					velocity: 1.0,
				},
			)],
		)
		.expect("the graph renders");

		let peak = frames
			.iter()
			.flat_map(|frame| frame.iter())
			.fold(0.0f32, |peak, sample| peak.max(sample.abs()));

		assert!(peak > 1e-3);
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
