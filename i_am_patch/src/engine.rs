//! The block engine: runs a compiled plan one block at a time.

use std::collections::BTreeMap;

use i_am_dsp::prelude::NoteEvent;

use i_am_dsp::ProcessContext;

use crate::block::{AudioBlock, BLOCK, BlockContext, Buffer, PatchContext};
use crate::compile::{CompileError, Plan, compile};
use crate::graph::{Graph, ModulationMode, NodeId, SignalKind};
use crate::nodes::{BlockNode, make_with};

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

impl std::fmt::Display for EngineError {
	fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			Self::Compile(error) => error.fmt(formatter),
			Self::UnknownKind(kind) => write!(formatter, "the catalog has no {kind} node"),
			Self::NoVoiceGroup => write!(formatter, "the graph has no voice group"),
		}
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
	/// The catalog entry of each step, for the range a parameter declares.
	descs: Vec<&'static crate::graph::NodeDesc>,
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
	velocity_points: Vec<Vec<crate::graph::Point>>,
	/// What the page handed the nodes: fitted tables, samples, responses.
	slots: crate::Slots,
	output_step: Option<usize>,
	latency: usize,
	tempo: Option<f32>,
	position: Option<f32>,
	/// The value each parameter ended a block at, so a parameter that did not
	/// move is not written into its node again.
	applied: Vec<Vec<f32>>,
	/// Reused across blocks so the audio path does not allocate.
	updates: Vec<(usize, &'static str, f32, f32)>,
	done: Vec<bool>,
	events: Vec<NoteEvent>,
}

impl Engine {
	/// Compiles a graph and builds its nodes.
	/// Hands the previous note to every node, for the voices that glide.
	///
	/// A patcher voice is a fresh engine per note, so a synth's envelope has no
	/// memory of the note before it: this is how the voice passes it on.
	pub fn glide_from(&mut self, channel: u8, note: usize) {
		for node in &mut self.nodes {
			node.glide_from(channel, note);
		}
	}

	/// Drops the tail the node with this id is holding.
	pub fn clear_tail(&mut self, node: u32) {
		if let Some(step) = self.steps.get(&NodeId(node)) {
			self.nodes[*step].clear_tail();
		}
	}

	/// The scripts the running nodes own, by node id, for the page's editor.
	pub fn script_sources(&self) -> Vec<(u32, std::sync::Arc<dyn i_am_effects::ScriptSource>)> {
		self.steps
			.iter()
			.filter_map(|(id, step)| {
				self.nodes[*step]
					.script_source()
					.map(|script| (id.0, script))
			})
			.collect()
	}

	pub fn new(graph: &Graph, rate: usize) -> Result<Self, EngineError> {
		Self::new_with(graph, rate, &crate::Slots::default())
	}

	/// Builds a graph over the slots the caller already owns, so a Modal node can
	/// play a sample the page fitted and a sampler the one it dropped.
	pub fn new_with(graph: &Graph, rate: usize, slots: &crate::Slots) -> Result<Self, EngineError> {
		let plan: Plan = compile(graph)?;
		let mut engine = Self::empty(rate);
		engine.build(&plan, graph, rate, slots)?;

		Ok(engine)
	}

	/// An engine with no graph, for a caller that builds it in place.
	pub fn empty(rate: usize) -> Self {
		Self {
			rate,
			block: 0,
			order: Vec::new(),
			kinds: Vec::new(),
			descs: Vec::new(),
			nodes: Vec::new(),
			outputs: Vec::new(),
			inputs: Vec::new(),
			edges: Vec::new(),
			lines: Vec::new(),
			links: Vec::new(),
			steps: std::collections::BTreeMap::new(),
			previous: Vec::new(),
			delta: Vec::new(),
			pure_control: Vec::new(),
			retrigger: Vec::new(),
			controls: Vec::new(),
			base: Vec::new(),
			points: Vec::new(),
			velocity_points: Vec::new(),
			slots: crate::Slots::default(),
			output_step: None,
			latency: 0,
			tempo: None,
			position: None,
			applied: Vec::new(),
			updates: Vec::with_capacity(16),
			done: Vec::new(),
			events: Vec::with_capacity(16),
		}
	}

	/// Builds a graph into this engine, reusing its buffers.
	///
	/// A voice is one engine per note, so the vectors and the buffer arrays are
	/// kept between notes: only the nodes, whose state has to start clean, are
	/// built again.
	pub fn rebuild(
		&mut self,
		plan: &Plan,
		graph: &Graph,
		rate: usize,
		slots: &crate::Slots,
	) -> Result<(), EngineError> {
		self.build(plan, graph, rate, slots)
	}

	fn build(
		&mut self,
		plan: &Plan,
		graph: &Graph,
		rate: usize,
		slots: &crate::Slots,
	) -> Result<(), EngineError> {
		self.rate = rate;
		self.block = 0;
		self.tempo = None;
		self.position = None;
		self.latency = 0;
		self.output_step = None;
		self.slots = slots.clone();

		let mut steps = std::mem::take(&mut self.steps);
		steps.clear();

		for (step, id) in plan.order.iter().enumerate() {
			steps.insert(*id, step);
		}

		let order_len = plan.order.len();
		let mut kinds = std::mem::take(&mut self.kinds);
		let mut descs = std::mem::take(&mut self.descs);
		let mut nodes = std::mem::take(&mut self.nodes);
		let mut outputs = std::mem::take(&mut self.outputs);
		let mut inputs = std::mem::take(&mut self.inputs);
		let mut edges = std::mem::take(&mut self.edges);
		let mut lines = std::mem::take(&mut self.lines);
		let mut controls = std::mem::take(&mut self.controls);
		let mut base = std::mem::take(&mut self.base);
		let mut points = std::mem::take(&mut self.points);
		let mut velocity_points = std::mem::take(&mut self.velocity_points);

		kinds.clear();
		descs.clear();
		nodes.clear();
		controls.clear();
		base.clear();
		points.clear();
		velocity_points.clear();
		edges.clear();
		edges.resize_with(order_len, Vec::new);
		lines.clear();
		lines.resize_with(order_len, Vec::new);
		outputs.resize_with(order_len, Vec::new);
		inputs.resize_with(order_len, Vec::new);
		self.applied.clear();
		self.applied.resize_with(order_len, Vec::new);

		let mut output_step = None;

		for (step, id) in plan.order.iter().enumerate() {
			let instance = graph
				.get(*id)
				.expect("the plan only names nodes in the graph");
			let kind = instance.kind;
			kinds.push(kind.id);
			descs.push(kind);
			nodes.push(
				make_with(
					kind.id,
					id.0,
					rate,
					&instance.params,
					&instance.points,
					&instance.velocity_points,
					slots,
				)
				.ok_or_else(|| EngineError::UnknownKind(kind.id.to_string()))?,
			);

			let mut node_outputs = std::mem::take(&mut outputs[step]);
			node_outputs.clear();
			for port in kind.outputs {
				node_outputs.push(match port.kind {
					SignalKind::Audio(_) => Buffer::Audio(AudioBlock::silent(BLOCK)),
					SignalKind::Control => Buffer::Control(0.0),
					SignalKind::Midi => Buffer::Midi,
					SignalKind::Wavetable | SignalKind::Osc => Buffer::Control(0.0),
				});
			}
			outputs[step] = node_outputs;
			controls.push(vec![0.0; kind.outputs.len()]);
			base.push(instance.params.clone());
			// NaN means "never written": it never equals a value, so the first
			// block writes every parameter once.
			self.applied[step] = vec![f32::NAN; instance.params.len()];
			points.push(instance.points.clone());
			velocity_points.push(instance.velocity_points.clone());

			let mut node_inputs = std::mem::take(&mut inputs[step]);
			node_inputs.clear();
			for port in kind.inputs {
				node_inputs.push(match port.kind {
					SignalKind::Audio(_) => Buffer::Audio(AudioBlock::silent(BLOCK)),
					SignalKind::Control => Buffer::Control(0.0),
					SignalKind::Midi => Buffer::Midi,
					SignalKind::Wavetable | SignalKind::Osc => Buffer::Control(0.0),
				});
			}
			inputs[step] = node_inputs;

			if kind.id == "out:audio" && output_step.is_none() {
				output_step = Some(step);
			}
		}

		// A constructor node hands its table or oscillator to the consumer. The
		// envelope builds the running oscillator once; the constructor nodes keep
		// only their parameters, which an atomic carries into every table copy.
		for edge in &graph.edges {
			let to_kind = graph.get(edge.to.0).map(|instance| instance.kind.id);

			if to_kind != Some("adsr") || edge.to.1 != 1 {
				continue;
			}

			let to = steps[&edge.to.0];

			if let Some(oscillator) =
				crate::nodes::build_oscillator(&nodes, &steps, graph, edge.from.0)
			{
				nodes[to].install(oscillator);
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

		let order_len = plan.order.len();

		self.order.clear();
		self.order.extend_from_slice(&plan.order);
		self.kinds = kinds;
		self.descs = descs;
		self.nodes = nodes;
		self.outputs = outputs;
		self.inputs = inputs;
		self.edges = edges;
		self.lines = lines;
		self.links = links;
		self.steps = step_map;
		self.previous = previous;
		self.delta = delta;
		self.pure_control = pure_control;
		self.retrigger = retrigger;
		self.controls = controls;
		self.base = base;
		self.points = points;
		self.velocity_points = velocity_points;
		self.output_step = output_step;
		self.latency = latency;
		self.done.clear();
		self.done.resize(order_len, false);

		Ok(())
	}

	/// Moves one parameter of a node the graph already holds.
	///
	/// The value is what the next block's ramp starts from, so a knob reaches
	/// the sound without rebuilding the graph: a rebuild re-creates every node
	/// and cuts the notes that were playing.
	pub fn set_param(&mut self, node: NodeId, key: &str, value: f32) -> bool {
		let Some(step) = self.order.iter().position(|id| *id == node) else {
			return false;
		};

		let Some((_, stored)) = self.base[step].iter_mut().find(|(name, _)| **name == key) else {
			return false;
		};

		*stored = value;

		true
	}

	/// Turns one modulation's depth, which is the same kind of edit as a knob.
	pub fn set_modulation_depth(&mut self, index: usize, depth: f32) -> bool {
		let Some(link) = self.links.get_mut(index) else {
			return false;
		};

		link.depth = depth;

		true
	}

	/// Swings one modulation around its base, or only up from it.
	pub fn set_modulation_mode(&mut self, index: usize, mode: ModulationMode) -> bool {
		let Some(link) = self.links.get_mut(index) else {
			return false;
		};

		link.mode = mode;

		true
	}

	/// The host's tempo and song position, for the sources that sync.
	pub fn set_transport(&mut self, tempo: Option<f32>, position: Option<f32>) {
		self.tempo = tempo;
		self.position = position;
	}

	/// Where a source's cycle is, for the page's playhead.
	pub fn control_phase(&self, step: usize) -> Option<f32> {
		self.nodes.get(step).and_then(|node| node.phase())
	}

	/// Where a MIDI source's input sits, for its card's mapping point.
	pub fn midi_position(&self, step: usize) -> Option<f32> {
		self.nodes.get(step).and_then(|node| node.midi_position())
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

	/// One node's last input as a block: what arrived at its first audio port.
	///
	/// A card that draws a before and an after needs both sides of the node,
	/// and the input is what the upstream nodes just handed it.
	pub fn input_block(&self, step: usize) -> Option<AudioBlock> {
		self.inputs
			.get(step)?
			.iter()
			.find_map(|buffer| match buffer {
				Buffer::Audio(block) => Some(*block),
				_ => None,
			})
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

	/// Whether the node's first output is a control value rather than audio.
	pub fn is_control(&self, step: usize) -> bool {
		matches!(
			self.outputs.get(step).and_then(|ports| ports.first()),
			Some(Buffer::Control(_))
		)
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
		Self::render_graph_with(graph, rate, seconds, events, &crate::Slots::default())
	}

	/// Renders a graph offline over tables the caller already owns, which is
	/// how a test hears a sample the page fitted.
	pub fn render_graph_with(
		graph: &Graph,
		rate: usize,
		seconds: f32,
		events: &[(u64, NoteEvent)],
		slots: &crate::Slots,
	) -> Result<Vec<[f32; 2]>, EngineError> {
		let mut engine = Self::new_with(graph, rate, slots)?;
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
		self.process_with_sidechain(input, &[], events, out);
	}

	/// Runs one block with the host's sidechain frames, which the graph reads
	/// through its `input:sidechain` node.
	pub fn process_with_sidechain(
		&mut self,
		input: &[[f32; 2]],
		sidechain: &[[f32; 2]],
		events: &[NoteEvent],
		out: &mut [[f32; 2]],
	) {
		let frames = input.len().min(BLOCK).min(out.len());
		// A product instrument reads the block's notes from the context, the
		// same way a host hands them over.
		let mut patch_context = PatchContext::new(self.rate);
		patch_context.set_transport(self.tempo, self.position);

		for event in events {
			patch_context.send_event(event.clone());
		}

		self.events.clear();
		self.events.extend_from_slice(events);
		let mut context = BlockContext {
			frames,
			rate: self.rate,
			block: self.block,
			events: std::mem::take(&mut self.events),
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
			let retrigger = std::mem::take(&mut self.retrigger);

			for &step in &retrigger {
				if let Some(node) = make_with(
					self.kinds[step],
					self.order[step].0,
					self.rate,
					&self.base[step],
					&self.points[step],
					&self.velocity_points[step],
					&self.slots,
				) {
					self.nodes[step] = node;
					// A fresh node starts at its defaults: every parameter has to
					// be written into it again.
					self.applied[step].fill(f32::NAN);
				}

				self.controls[step].fill(0.0);
			}

			self.retrigger = retrigger;
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

						for (frame, sample) in sidechain.iter().take(frames).enumerate() {
							block.channels[0][frame] = sample[0];
							block.channels[1][frame] = sample[1];
						}
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
		let mut done = std::mem::take(&mut self.done);
		done.clear();
		done.resize(self.order.len(), false);

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

		let mut updates = std::mem::take(&mut self.updates);

		for step in 0..self.order.len() {
			updates.clear();

			for (index, (name, base)) in self.base[step].iter().enumerate() {
				let name = *name;
				let mut offset_start = 0.0;
				let mut offset_end = 0.0;

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
						// The envelope, the pitch tracker and the MIDI source
						// already sit in 0..1; the LFO, random and table are the
						// ones that swing around zero.
						let unipolar =
							matches!(self.kinds[link.source], "envelope" | "pitch" | "midi");
						let unit = |value: f32| match link.mode {
							ModulationMode::Centered => value,
							ModulationMode::FromBase if unipolar => value,
							ModulationMode::FromBase => (value + 1.0) * 0.5,
						};

						offset_start += link.depth * unit(last);
						offset_end += link.depth * unit(now);
					}
				}

				// The offset is a share of the parameter's own range, so a
				// logarithmic control moves by a ratio instead of by a fixed
				// number of hertz, which is what its knob does. A parameter
				// nothing drives is passed on untouched: its base may sit
				// outside the range the catalog declares, and normalising would
				// quietly pull it in.
				let (start, end) = match self.descs[step]
					.params
					.iter()
					.find(|param| param.name == name)
				{
					Some(param) if offset_start != 0.0 || offset_end != 0.0 => {
						let unit = param.normalize(*base);

						(
							param.denormalize(unit + offset_start),
							param.denormalize(unit + offset_end),
						)
					}
					_ => (*base + offset_start, *base + offset_end),
				};

				// Nothing drives it and it has not moved: writing it again would
				// cost a parameter-map write per block for nothing.
				if start == end && self.applied[step][index] == end {
					continue;
				}

				updates.push((index, name, start, end));
			}

			for &(index, name, start, end) in &updates {
				self.nodes[step].set_param_block(name, start, end, frames);
				// The block ends at the value the ramp reached.
				self.applied[step][index] = end;
			}
		}

		self.updates = updates;

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
			// The output node's gain is the patch's master level: the page sets
			// it, the generation bump rebuilds the engine, and it lands here.
			let gain = self.base[step].get("gain").copied().unwrap_or(1.0);

			for (frame, sample) in out.iter_mut().take(frames).enumerate() {
				*sample = block.frame(frame);
				sample[0] *= gain;
				sample[1] *= gain;
			}
		}

		self.done = done;
		self.events = std::mem::take(&mut context.events);
	}
}

#[cfg(test)]
mod tests;
