//! The block engine: runs a compiled plan one block at a time.

use std::collections::BTreeMap;

use i_am_dsp::prelude::NoteEvent;

use i_am_dsp::ProcessContext;

use crate::block::{AudioBlock, BLOCK, BlockContext, Buffer, PatchContext};
use crate::compile::{CompileError, Plan, compile};
use crate::graph::{Graph, ModulationMode, NodeId, SignalKind};
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

/// A short ring that holds one edge back, so a diamond's paths meet.
struct DelayLine {
	buffer: Vec<[f32; 2]>,
	write: usize,
}

impl DelayLine {
	fn new(frames: usize) -> Self {
		Self {
			buffer: vec![[0.0; 2]; frames.max(1)],
			write: 0,
		}
	}

	fn next(&mut self, input: [f32; 2]) -> [f32; 2] {
		let output = self.buffer[self.write];
		self.buffer[self.write] = input;
		self.write = (self.write + 1) % self.buffer.len();
		output
	}
}

/// One control assignment, resolved to steps.
struct Link {
	source: usize,
	port: usize,
	target: usize,
	param: &'static str,
	depth: f32,
	mode: ModulationMode,
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
	edges: Vec<Vec<(usize, usize, usize, usize)>>,
	lines: Vec<Vec<Option<DelayLine>>>,
	links: Vec<Link>,
	steps: std::collections::BTreeMap<NodeId, usize>,
	previous: Vec<Vec<f32>>,
	delta: Vec<Vec<f32>>,
	pure_control: Vec<bool>,
	retrigger: Vec<usize>,
	controls: Vec<Vec<f32>>,
	base: Vec<BTreeMap<&'static str, f32>>,
	points: Vec<Vec<crate::graph::Point>>,
	output_step: Option<usize>,
	latency: usize,
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
		let mut lines: Vec<Vec<Option<DelayLine>>> =
			(0..plan.order.len()).map(|_| Vec::new()).collect();
		let mut controls = Vec::with_capacity(plan.order.len());
		let mut base = Vec::with_capacity(plan.order.len());
		let mut points = Vec::with_capacity(plan.order.len());
		let mut output_step = None;

		for (step, id) in plan.order.iter().enumerate() {
			let instance = graph
				.get(*id)
				.expect("the plan only names nodes in the graph");
			let kind = instance.kind;
			kinds.push(kind.id);
			nodes.push(
				make(kind.id, rate, &instance.params, &instance.points)
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
			points.push(instance.points.clone());

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

		for (index, edge) in graph.edges.iter().enumerate() {
			let from = steps[&edge.from.0];
			let to = steps[&edge.to.0];
			let delay = plan.delays.get(index).copied().unwrap_or(0);
			edges[to].push((edge.to.1, from, edge.from.1, delay));
			lines[to].push(if delay > 0 {
				Some(DelayLine::new(delay))
			} else {
				None
			});
		}

		let latency = output_step
			.and_then(|step| plan.latency.get(&plan.order[step]))
			.copied()
			.unwrap_or(0);

		let links: Vec<Link> = plan
			.modulations
			.iter()
			.map(|link| Link {
				source: steps[&link.source.0],
				port: link.source.1,
				target: steps[&link.target.0],
				param: link.target.1,
				depth: link.depth,
				mode: link.mode,
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
			lines,
			links,
			steps: step_map,
			previous,
			delta,
			pure_control,
			retrigger,
			controls,
			base,
			points,
			output_step,
			latency,
		})
	}

	/// The step a node ended up on.
	pub fn step_of(&self, id: NodeId) -> Option<usize> {
		self.steps.get(&id).copied()
	}

	/// One node's last audio output, for the page's pictures.
	pub fn output_block(&self, step: usize, port: usize) -> Option<&AudioBlock> {
		match self.outputs.get(step)?.get(port)? {
			Buffer::Audio(block) => Some(block),
			_ => None,
		}
	}

	/// One node's last output as a block: audio as it is, a control value
	/// repeated across the block.
	pub fn tap_block(&self, step: usize) -> Option<AudioBlock> {
		match self.outputs.get(step)?.first()? {
			Buffer::Audio(block) => Some(*block),
			Buffer::Control(value) => {
				let mut block = AudioBlock::silent(BLOCK);

				for frame in 0..BLOCK {
					block.channels[0][frame] = *value;
					block.channels[1][frame] = *value;
				}

				Some(block)
			}
			_ => None,
		}
	}

	/// How many frames the graph itself delays its output.
	pub fn latency(&self) -> usize {
		self.latency
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
		// A product instrument reads the block's notes from the context, the
		// same way a host hands them over.
		let mut patch_context = PatchContext::new(self.rate);

		for event in events {
			patch_context.send_event(event.clone());
		}

		let mut context = BlockContext {
			frames,
			rate: self.rate,
			block: self.block,
			events: events.to_vec(),
			context: Box::new(patch_context),
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
				if let Some(node) = make(self.kinds[step], self.rate, &self.base[step], &self.points[step]) {
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

						// Centered swings both ways around the base; from-base
						// only rises, so a bipolar source is folded into 0..1.
						let bipolar = self.kinds[link.source] != "envelope";
						let unit = |value: f32| match link.mode {
							ModulationMode::Centered => value,
							ModulationMode::FromBase if bipolar => (value + 1.0) * 0.5,
							ModulationMode::FromBase => value,
						};

						start += link.depth * unit(last);
						end += link.depth * unit(now);
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

			for index in 0..self.edges[step].len() {
				let (port, from, from_port, delay) = self.edges[step][index];
				let source = self.outputs[from][from_port];

				if delay == 0 {
					self.inputs[step][port].add(&source);
					continue;
				}

				if let Some(line) = &mut self.lines[step][index]
					&& let (Buffer::Audio(src), Buffer::Audio(input)) =
						(&source, &mut self.inputs[step][port])
				{
					for frame in 0..frames {
						let delayed = line.next(src.frame(frame));
						input.channels[0][frame] += delayed[0];
						input.channels[1][frame] += delayed[1];
					}

					input.frames = frames;
				}
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
	fn an_lfo_can_follow_points() {
		let mut params = BTreeMap::new();
		params.insert("rate", 375.0);
		let points = [
			crate::graph::Point { time: 0.0, value: 0.0, bend: 0.0 },
			crate::graph::Point { time: 0.5, value: 1.0, bend: 0.0 },
			crate::graph::Point { time: 1.0, value: 0.0, bend: 0.0 },
		];
		let mut lfo = make("lfo", 48_000, &params, &points).expect("lfo");
		let mut out = vec![Buffer::Control(0.0)];
		let mut context = BlockContext {
			frames: BLOCK,
			rate: 48_000,
			block: 0,
			events: Vec::new(),
			context: Box::new(PatchContext::new(48_000)),
		};

		lfo.process(&mut context, &[], &mut out);

		let Buffer::Control(value) = out[0] else {
			panic!("the lfo wrote audio");
		};

		assert!((value - 1.0).abs() < 1e-3, "the point table: {value}");
	}

	#[test]
	fn the_control_sources_move() {
		for kind in ["random", "table", "midi"] {
			let params = std::collections::BTreeMap::new();
			let mut node = make(kind, 48_000, &params, &[]).expect("a control source");
			let mut out = vec![Buffer::Control(0.0)];
			let mut context = BlockContext {
				frames: BLOCK,
				rate: 48_000,
				block: 0,
				events: vec![NoteEvent::NoteOn { channel: 0, note: 60, velocity: 0.8 }],
				context: Box::new(PatchContext::new(48_000)),
			};

			node.set_param("rate", 100.0);
			node.set_param("shape", 0.0);

			let mut moved = false;

			for block in 0..8 {
				context.block = block;
				context.events.clear();
				if block == 0 {
					context.events.push(NoteEvent::NoteOn { channel: 0, note: 60, velocity: 0.8 });
				}
				node.process(&mut context, &[], &mut out);

				if let Buffer::Control(value) = out[0] {
					moved |= value.abs() > 1e-4;
				}
			}

			assert!(moved, "{kind} never moved");
		}
	}

	#[test]
	fn a_negative_modulation_does_not_blow_up() {
		let mut graph = Graph::new();
		let input = graph.add_node("input:audio", (0.0, 0.0)).expect("input");
		let filter = graph.add_node("filter", (100.0, 0.0)).expect("filter");
		let out = graph.add_node("out:audio", (200.0, 0.0)).expect("out");
		graph.connect((input, 0), (filter, 0));
		graph.connect((filter, 0), (out, 0));

		let lfo = graph.add_node("lfo", (0.0, 100.0)).expect("lfo");
		graph.set_param(lfo, "rate", 3.0);
		graph.set_param(filter, "cutoff", 228.0);
		graph.modulate((lfo, 0), (filter, "cutoff"), -800.0);

		let mut engine = Engine::new(&graph, 48_000).expect("builds");
		let mut frames = Vec::new();

		for block in 0..200 {
			let mut host = vec![[0.0f32; 2]; BLOCK];

			for (index, frame) in host.iter_mut().enumerate() {
				let time = (block * BLOCK + index) as f32 / 48_000.0;
				let value = (time * 440.0 * std::f32::consts::TAU).sin() * 0.5;
				*frame = [value, value];
			}

			let mut block_out = vec![[0.0f32; 2]; BLOCK];
			engine.process(&host, &[], &mut block_out);
			frames.extend(block_out);
		}

		let peak = frames.iter().flat_map(|frame| frame.iter()).fold(0.0f32, |peak, sample| peak.max(sample.abs()));
		let mean: f32 = frames.iter().map(|frame| frame[0]).sum::<f32>() / frames.len() as f32;

		assert!(peak < 2.0, "the peak blew up: {peak}");
		assert!(mean.abs() < 0.01, "the output has DC: {mean}");
	}

	#[test]
	fn a_synth_node_plays() {
		let mut graph = Graph::new();
		let midi = graph.add_node("input:midi", (0.0, 0.0)).expect("midi");
		let synth = graph.add_node("synth:additive", (100.0, 0.0)).expect("synth");
		let out = graph.add_node("out:audio", (200.0, 0.0)).expect("out");
		graph.connect((midi, 0), (synth, 0));
		graph.connect((synth, 0), (out, 0));

		let mut engine = Engine::new(&graph, 48_000).expect("builds");
		let mut frames = Vec::new();

		for block in 0..8 {
			let events = if block == 0 {
				vec![NoteEvent::NoteOn {
					channel: 0,
					note: 60,
					velocity: 0.8,
				}]
			} else {
				Vec::new()
			};
			let host = vec![[0.0f32; 2]; BLOCK];
			let mut block_out = vec![[0.0f32; 2]; BLOCK];
			engine.process(&host, &events, &mut block_out);
			frames.extend(block_out);
		}

		let peak = frames
			.iter()
			.flat_map(|frame| frame.iter())
			.fold(0.0f32, |peak, sample| peak.max(sample.abs()));

		assert!(peak > 1e-3, "the synth was silent: {peak}");
	}

	#[test]
	fn a_module_node_passes_audio() {
		let mut graph = Graph::new();
		let input = graph.add_node("input:audio", (0.0, 0.0)).expect("input");
		let compressor = graph.add_node("mod:compressor", (100.0, 0.0)).expect("compressor");
		let out = graph.add_node("out:audio", (200.0, 0.0)).expect("out");
		graph.connect((input, 0), (compressor, 0));
		graph.connect((compressor, 0), (out, 0));

		let mut engine = Engine::new(&graph, 48_000).expect("builds");
		let mut frames = Vec::new();

		for _ in 0..4 {
			let host = vec![[0.5f32; 2]; BLOCK];
			let mut block = vec![[0.0f32; 2]; BLOCK];
			engine.process(&host, &[], &mut block);
			frames.extend(block);
		}

		let peak = frames.iter().flat_map(|frame| frame.iter()).fold(0.0f32, |peak, sample| peak.max(sample.abs()));
		assert!(peak > 1e-3, "the module node was silent");
	}

	#[test]
	fn a_short_path_is_delayed_to_align() {
		let mut graph = Graph::new();
		let input = graph.add_node("input:audio", (0.0, 0.0)).expect("input");
		let delay = graph.add_node("delay", (100.0, 0.0)).expect("delay");
		let mixer = graph.add_node("mixer", (200.0, 0.0)).expect("mixer");
		let out = graph.add_node("out:audio", (300.0, 0.0)).expect("out");
		graph.connect((input, 0), (mixer, 0));
		graph.connect((input, 0), (delay, 0));
		graph.connect((delay, 0), (mixer, 1));
		graph.connect((mixer, 0), (out, 0));

		let mut engine = Engine::new(&graph, 48_000).expect("builds");
		assert_eq!(engine.latency(), 64);

		let mut first = vec![[0.0f32; 2]; BLOCK];
		first[0] = [1.0, 1.0];
		let mut block = vec![[0.0f32; 2]; BLOCK];
		engine.process(&first, &[], &mut block);
		assert!(block.iter().all(|frame| frame[0].abs() < 1e-6));

		let silence = vec![[0.0f32; 2]; BLOCK];
		engine.process(&silence, &[], &mut block);
		// Both paths arrive together, and the short one was padded to meet the
		// delayed one.
		assert!((block[0][0] - 2.0).abs() < 1e-6, "got {}", block[0][0]);
	}

	#[test]
	fn a_delay_node_reports_its_latency() {
		let chain = |delays: usize| {
			let mut graph = Graph::new();
			let input = graph.add_node("input:audio", (0.0, 0.0)).expect("input");
			let mut last = input;
			let mut x = 100.0;

			for _ in 0..delays {
				let delay = graph.add_node("delay", (x, 0.0)).expect("delay");
				graph.connect((last, 0), (delay, 0));
				last = delay;
				x += 100.0;
			}

			let out = graph.add_node("out:audio", (x, 0.0)).expect("out");
			graph.connect((last, 0), (out, 0));

			Engine::new(&graph, 48_000).expect("builds").latency()
		};

		assert_eq!(chain(0), 0);
		assert_eq!(chain(1), 64);
		assert_eq!(chain(2), 128);
	}

	#[test]
	fn a_modulation_mode_moves_where_the_swing_starts() {
		fn render(mode: ModulationMode) -> (f32, f32) {
			let mut graph = Graph::new();
			let input = graph.add_node("input:audio", (0.0, 0.0)).expect("input");
			let gain = graph.add_node("gain", (100.0, 0.0)).expect("gain");
			let out = graph.add_node("out:audio", (200.0, 0.0)).expect("out");
			graph.connect((input, 0), (gain, 0));
			graph.connect((gain, 0), (out, 0));

			let lfo = graph.add_node("lfo", (0.0, 100.0)).expect("lfo");
			graph.set_param(lfo, "rate", 100.0);
			graph.modulate((lfo, 0), (gain, "gain"), 0.5);
			graph.set_modulation_mode(0, mode);

			let mut engine = Engine::new(&graph, 48_000).expect("builds");
			let mut frames = Vec::new();

			for _ in 0..16 {
				let host = vec![[0.5f32; 2]; BLOCK];
				let mut block = vec![[0.0f32; 2]; BLOCK];
				engine.process(&host, &[], &mut block);
				frames.extend(block);
			}

			let min = frames.iter().map(|frame| frame[0]).fold(f32::INFINITY, f32::min);
			let max = frames.iter().map(|frame| frame[0]).fold(0.0f32, f32::max);

			(min, max)
		}

		let (centered_min, centered_max) = render(ModulationMode::Centered);
		let (base_min, base_max) = render(ModulationMode::FromBase);

		assert!(centered_min < 0.45 && centered_max > 0.55, "centered: {centered_min}..{centered_max}");
		assert!(base_min > 0.49 && base_max > 0.55, "from base: {base_min}..{base_max}");
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
