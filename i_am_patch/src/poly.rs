//! Polyphony: one voice group, one engine per note.
//!
//! The group is cut out of the graph; its nodes become the voice graph, which
//! is instantiated once per note. The outer graph keeps everything else and
//! receives the summed voices through the `voice:out` boundary.

use std::collections::BTreeMap;

use i_am_dsp::prelude::NoteEvent;

use crate::block::{AudioBlock, BLOCK};
use crate::compile::voice_group;
use crate::engine::{Engine, EngineError};
use crate::graph::{Edge, Graph, NodeId, NodeInstance, node};

/// How many notes can sound at once, unless the graph asks for fewer.
pub const POLYPHONY: usize = 8;

/// How many voices a graph allows: the `voices` control on its `voice:in`.
///
/// The page writes it as a graph parameter, so it is read when the patch is
/// built; moving the control bumps the generation and the engine is rebuilt.
pub fn voices_of(graph: &Graph) -> usize {
	graph
		.nodes
		.iter()
		.find(|node| node.kind.id == "voice:in")
		.and_then(|node| node.params.get("voices"))
		.map(|voices| (*voices as usize).clamp(1, POLYPHONY))
		.unwrap_or(POLYPHONY)
}

/// How many quiet blocks retire a voice whose graph has already faded out.
const SILENT_BLOCKS: u32 = 150;

/// How long a released voice may keep ringing before its slot is free again,
/// in blocks: two seconds, so a long release still finishes.
fn release_blocks(rate: usize) -> u32 {
	(2 * rate / BLOCK).max(1) as u32
}

/// Whether a voice's slot can be reused.
///
/// A note-off starts the release; the voice is retired when its graph goes
/// quiet after that, when the release window runs out (a graph that ignores the note-off
/// must not hold the slot forever), or when it has simply lived too long.
fn expired(age: u64, silent: u32, released: bool, release_age: u32, release_window: u32) -> bool {
	age >= 10_000
		|| (released && silent >= SILENT_BLOCKS)
		|| (released && release_age >= release_window)
}

struct Voice {
	engine: Engine,
	pending: Vec<NoteEvent>,
	note: usize,
	age: u64,
	silent: u32,
	released: bool,
	release_age: u32,
}

/// A polyphonic patch: one voice group and the graph around it.
pub struct Poly {
	outer: Engine,
	voice_graph: Graph,
	/// Where a page reads the scripts the voices own, by node id.
	scripts: Option<crate::Scripts>,
	/// What the page handed the nodes: fitted tables, samples, responses.
	slots: crate::Slots,
	rate: usize,
	out: NodeId,
	voice_latency: usize,
	tempo: Option<f32>,
	position: Option<f32>,
	/// How many voices may sound at once.
	limit: usize,
	voices: Vec<Voice>,
}

impl Poly {
	/// Drops the tail the node with this id is holding, in the outer graph and
	/// in every voice.
	pub fn clear_tail(&mut self, node: u32) {
		self.outer.clear_tail(node);

		for voice in &mut self.voices {
			voice.engine.clear_tail(node);
		}
	}

	/// Where a page reads the scripts the voices own. The outer engine's own
	/// scripts are registered here; a voice adds its own as it is built.
	pub fn set_scripts(&mut self, scripts: crate::Scripts) {
		{
			let mut map = scripts.lock().expect("the script lock");
			for (node, script) in self.outer.script_sources() {
				map.insert(node, script);
			}
		}

		self.scripts = Some(scripts);
	}

	/// Splits the graph and builds the outer engine.
	pub fn new(graph: &Graph, rate: usize) -> Result<Self, EngineError> {
		Self::new_with(graph, rate, &crate::Slots::default())
	}

	/// A polyphonic graph over the slots the caller already owns.
	pub fn new_with(graph: &Graph, rate: usize, slots: &crate::Slots) -> Result<Self, EngineError> {
		let group = voice_group(graph);

		if group.is_empty() {
			return Err(EngineError::NoVoiceGroup);
		}

		// Splitting the graph drops the links that cross the boundary, so a
		// mistake there would otherwise vanish into silence. Compile the whole
		// graph first: the same checks, and a build error the page can show.
		crate::compile::compile(graph)?;

		let out = graph
			.nodes
			.iter()
			.find(|node| node.kind.id == "voice:out")
			.map(|node| node.id)
			.ok_or(EngineError::NoVoiceGroup)?;

		// The voice graph: the group's nodes, the edges inside it, and a
		// synthetic output fed by whatever reached the voice boundary.
		let mut voice_graph = Graph::new();
		let ids: BTreeMap<NodeId, NodeId> = group.iter().map(|id| (*id, *id)).collect();

		for instance in &graph.nodes {
			if group.contains(&instance.id) {
				voice_graph.nodes.push(instance.clone());
			}
		}

		for edge in &graph.edges {
			if ids.contains_key(&edge.from.0) && ids.contains_key(&edge.to.0) {
				voice_graph.edges.push(*edge);
			}
		}

		for link in &graph.modulations {
			if ids.contains_key(&link.source.0) && ids.contains_key(&link.target.0) {
				voice_graph.modulations.push(*link);
			}
		}

		let out_id = NodeId(
			voice_graph
				.nodes
				.iter()
				.map(|instance| instance.id.0)
				.max()
				.unwrap_or(0)
				+ 1,
		);
		let sink = node("out:audio").ok_or(EngineError::UnknownKind("out:audio".into()))?;

		voice_graph.nodes.push(NodeInstance {
			id: out_id,
			kind: sink,
			position: (0.0, 0.0),
			params: BTreeMap::new(),
			points: Vec::new(),
		});

		for edge in &graph.edges {
			if edge.to.0 == out && group.contains(&edge.from.0) {
				voice_graph.edges.push(Edge {
					from: edge.from,
					to: (out_id, 0),
				});
			}
		}

		// The outer graph: everything not in the group.
		let mut outer = Graph::new();

		for instance in &graph.nodes {
			if !group.contains(&instance.id) {
				outer.nodes.push(instance.clone());
			}
		}

		for edge in &graph.edges {
			if !group.contains(&edge.from.0) && !group.contains(&edge.to.0) {
				outer.edges.push(*edge);
			}
		}

		for link in &graph.modulations {
			if !group.contains(&link.source.0) && !group.contains(&link.target.0) {
				outer.modulations.push(*link);
			}
		}

		let voice_latency = crate::compile::compile(&voice_graph)
			.map(|plan| plan.latency.get(&out_id).copied().unwrap_or(0))
			.unwrap_or(0);

		Ok(Self {
			scripts: None,
			slots: slots.clone(),
			outer: Engine::new_with(&outer, rate, slots)?,
			voice_graph,
			rate,
			out,
			voice_latency,
			tempo: None,
			position: None,
			limit: voices_of(graph),
			voices: Vec::new(),
		})
	}

	/// The same poly, over at most `limit` voices.
	pub fn with_limit(mut self, limit: usize) -> Self {
		self.limit = limit.clamp(1, POLYPHONY);
		self
	}

	/// Where a source's cycle is: the outer graph's, or the newest voice's.
	///
	/// The newest voice is the one whose value the tap reports, so the playhead
	/// and the dot belong to the same cycle; the first voice would be a different
	/// note's, and its phase moves whenever that voice retires.
	pub fn control_phase(&self, id: NodeId) -> Option<f32> {
		self.outer
			.step_of(id)
			.and_then(|step| self.outer.control_phase(step))
			.or_else(|| {
				self.voices.iter().rev().find_map(|voice| {
					voice
						.engine
						.step_of(id)
						.and_then(|step| voice.engine.control_phase(step))
				})
			})
	}

	/// The step a node ended up on, in the outer graph.
	pub fn step_of(&self, id: NodeId) -> Option<usize> {
		self.outer.step_of(id)
	}

	/// One outer node's last audio output, for the page's pictures.
	pub fn output_block(&self, step: usize, port: usize) -> Option<&AudioBlock> {
		self.outer.output_block(step, port)
	}

	/// One outer node's last output as a block.
	pub fn tap_block(&self, step: usize) -> Option<AudioBlock> {
		self.outer.tap_block(step)
	}

	/// A node inside the voice group, as the page watches it: audio summed
	/// across the voices playing it, a control value taken from the newest one.
	/// The page draws one picture, not one per voice.
	pub fn tap_voice_block(&self, id: NodeId) -> Option<AudioBlock> {
		let control = self
			.voices
			.iter()
			.find_map(|voice| {
				voice
					.engine
					.step_of(id)
					.map(|step| voice.engine.is_control(step))
			})
			.unwrap_or(false);

		if control {
			// A control source exists once per voice and holds the same kind of
			// number in each. Adding them would read N times the value with N
			// notes held, so the card's dot leaves its curve; the newest voice is
			// the note that was just played, and it is the one the playhead
			// reports too.
			return self.voices.iter().rev().find_map(|voice| {
				voice
					.engine
					.step_of(id)
					.and_then(|step| voice.engine.tap_block(step))
			});
		}

		let mut sum: Option<AudioBlock> = None;

		for voice in &self.voices {
			if let Some(step) = voice.engine.step_of(id)
				&& let Some(block) = voice.engine.tap_block(step)
			{
				match &mut sum {
					Some(total) => total.add(&block),
					None => sum = Some(block),
				}
			}
		}

		sum
	}

	/// The host's tempo and song position, for the voices and the outer graph.
	pub fn set_transport(&mut self, tempo: Option<f32>, position: Option<f32>) {
		self.tempo = tempo;
		self.position = position;
		self.outer.set_transport(tempo, position);

		for voice in &mut self.voices {
			voice.engine.set_transport(tempo, position);
		}
	}

	/// How many frames the outer graph and the voice graph delay the output.
	pub fn latency(&self) -> usize {
		self.outer.latency() + self.voice_latency
	}

	/// Runs one block: notes start voices, the voices are summed and the outer
	/// graph processes the rest.
	pub fn process(&mut self, input: &[[f32; 2]], events: &[NoteEvent], out: &mut [[f32; 2]]) {
		self.process_with_sidechain(input, &[], events, out);
	}

	/// Runs one block with the host's sidechain frames.
	pub fn process_with_sidechain(
		&mut self,
		input: &[[f32; 2]],
		sidechain: &[[f32; 2]],
		events: &[NoteEvent],
		out: &mut [[f32; 2]],
	) {
		let frames = input.len().min(BLOCK).min(out.len());

		for event in events {
			match event {
				// A note-on with no velocity is the MIDI convention for a
				// note-off; treating it as a new note would hold a voice forever.
				NoteEvent::NoteOn { note, velocity, .. } if *velocity <= 0.0 => {
					if let Some(voice) = self
						.voices
						.iter_mut()
						.rev()
						.find(|voice| voice.note == *note)
					{
						voice.pending.push(event.clone());
						voice.released = true;
						voice.release_age = 0;
					}
				}
				NoteEvent::NoteOn { note, .. } => {
					// A repeated note-on retriggers its own voice: some hosts send
					// duplicates, and every extra voice runs the whole group.
					if let Some(position) = self.voices.iter().position(|voice| voice.note == *note)
					{
						self.voices.remove(position);
					}

					if self.voices.len() >= self.limit {
						// A released voice is the one to lose; otherwise the oldest.
						let position = self
							.voices
							.iter()
							.position(|voice| voice.released)
							.unwrap_or(0);
						self.voices.remove(position);
					}

					if let Ok(engine) = Engine::new_with(&self.voice_graph, self.rate, &self.slots)
					{
						if let Some(scripts) = &self.scripts {
							let mut scripts = scripts.lock().expect("the script lock");
							for (node, script) in engine.script_sources() {
								scripts.insert(node, script);
							}
						}

						self.voices.push(Voice {
							engine,
							pending: vec![event.clone()],
							note: *note,
							age: 0,
							silent: 0,
							released: false,
							release_age: 0,
						});
					}
				}
				NoteEvent::Stop { note, .. } => {
					// A host's immediate stop releases every voice playing that note.
					for voice in self.voices.iter_mut().filter(|voice| voice.note == *note) {
						voice.pending.push(event.clone());
						voice.released = true;
						voice.release_age = 0;
					}
				}
				NoteEvent::ImmediateStop => {
					for voice in &mut self.voices {
						voice.pending.push(event.clone());
						voice.released = true;
						voice.release_age = 0;
					}
				}
				NoteEvent::NoteOff { note, .. } => {
					// The newest voice of that note is the one this off belongs to;
					// an older one with the same note was released by its own off.
					if let Some(voice) = self
						.voices
						.iter_mut()
						.rev()
						.find(|voice| voice.note == *note)
					{
						voice.pending.push(event.clone());
						voice.released = true;
						voice.release_age = 0;
					}
				}
				_ => {}
			}
		}

		let mut mix = AudioBlock::silent(frames);
		// Both are one block on the stack: the audio path must not allocate.
		let silence = [[0.0f32; 2]; BLOCK];

		for voice in &mut self.voices {
			let mut buffer = [[0.0f32; 2]; BLOCK];

			voice
				.engine
				.process_with_sidechain(&silence, sidechain, &voice.pending, &mut buffer);
			voice.pending.clear();
			voice.age += 1;

			for (frame, sample) in buffer.iter().take(frames).enumerate() {
				mix.channels[0][frame] += sample[0];
				mix.channels[1][frame] += sample[1];
			}

			// A released voice is retired once it has been quiet for a while, or
			// when its release window runs out.
			if voice.released {
				voice.release_age += 1;
			}

			let peak = buffer.iter().take(frames).fold(0.0f32, |peak, frame| {
				peak.max(frame[0].abs()).max(frame[1].abs())
			});

			if peak < 1e-4 {
				voice.silent += 1;
			} else {
				voice.silent = 0;
			}
		}

		let release_window = release_blocks(self.rate);
		self.voices.retain(|voice| {
			!expired(
				voice.age,
				voice.silent,
				voice.released,
				voice.release_age,
				release_window,
			)
		});

		self.outer.inject(self.out, mix);
		self.outer
			.process_with_sidechain(input, sidechain, events, out);
	}
}

#[cfg(test)]
mod tests {
	#[test]
	fn a_released_voice_expires_even_when_it_keeps_sounding() {
		// Released, still audible, past the two-second window.
		assert!(super::expired(200, 0, true, 1_500, 1_500));
		// Released but the window has not run out.
		assert!(!super::expired(200, 0, true, 1_499, 1_500));
		// A held note stays even when it is quiet: a long attack is not an end.
		assert!(!super::expired(200, 150, false, 0, 1_500));
		assert!(!super::expired(200, 0, false, 9_999, 1_500));
		// A released note goes quiet and is retired.
		assert!(super::expired(200, 150, true, 0, 1_500));
		assert!(super::expired(10_000, 0, false, 0, 1_500));
	}

	use crate::voice_patch;

	use super::*;
	use i_am_dsp::prelude::NoteEvent;

	fn note(note: usize) -> NoteEvent {
		NoteEvent::NoteOn {
			channel: 0,
			note,
			velocity: 1.0,
		}
	}
	fn render(poly: &mut Poly, blocks: usize, notes: &[(u64, NoteEvent)]) -> Vec<[f32; 2]> {
		let mut out = Vec::new();

		for block in 0..blocks as u64 {
			let due: Vec<NoteEvent> = notes
				.iter()
				.filter(|(at, _)| *at == block)
				.map(|(_, event)| event.clone())
				.collect();
			let input = vec![[0.0f32; 2]; BLOCK];
			let mut frame = vec![[0.0f32; 2]; BLOCK];

			poly.process(&input, &due, &mut frame);
			out.extend(frame);
		}

		out
	}

	#[test]
	fn an_immediate_stop_releases_every_voice() {
		let graph = voice_patch();
		let mut poly = Poly::new(&graph, 48_000).expect("poly builds");

		let _ = render(
			&mut poly,
			8,
			&[(0, note(60)), (2, note(67)), (4, NoteEvent::ImmediateStop)],
		);

		assert!(
			poly.voices.iter().all(|voice| voice.released),
			"a voice kept sounding after the stop"
		);
	}

	#[test]
	fn a_repeated_note_on_reuses_its_voice() {
		let graph = voice_patch();
		let mut poly = Poly::new(&graph, 48_000).expect("poly builds");

		let _ = render(
			&mut poly,
			8,
			&[(0, note(60)), (1, note(60)), (2, note(60)), (3, note(60))],
		);

		assert_eq!(
			poly.voices.len(),
			1,
			"a repeated note-on stacked {} voices",
			poly.voices.len()
		);
	}

	#[test]
	fn a_voice_limit_in_the_graph_caps_the_poly() {
		let mut graph = voice_patch();
		let voice_in = graph
			.nodes
			.iter()
			.find(|node| node.kind.id == "voice:in")
			.expect("voice in")
			.id;
		graph.set_param(voice_in, "voices", 1.0);

		assert_eq!(voices_of(&graph), 1);

		let mut poly = Poly::new(&graph, 48_000).expect("poly builds");
		let _ = render(&mut poly, 8, &[(0, note(60)), (2, note(67))]);

		assert_eq!(poly.voices.len(), 1, "a second note found a second voice");
		assert_eq!(poly.voices[0].note, 67, "the newest note did not take over");
	}

	#[test]
	fn a_graph_without_a_voice_control_keeps_the_default() {
		let graph = voice_patch();

		assert_eq!(voices_of(&graph), POLYPHONY);
	}

	#[test]
	fn a_note_on_with_no_velocity_is_a_note_off() {
		let graph = voice_patch();
		let mut poly = Poly::new(&graph, 48_000).expect("poly builds");
		let silent_off = |note: usize| NoteEvent::NoteOn {
			channel: 0,
			note,
			velocity: 0.0,
		};

		let _ = render(&mut poly, 8, &[(0, note(60)), (2, silent_off(60))]);

		assert_eq!(poly.voices.len(), 1, "the silent note-on stacked a voice");
		assert!(
			poly.voices.iter().all(|voice| voice.released),
			"the silent note-on did not release the voice"
		);
	}

	#[test]
	fn a_note_off_releases_the_note_that_started_last() {
		let graph = voice_patch();
		let mut poly = Poly::new(&graph, 48_000).expect("poly builds");
		let off = |note: usize| NoteEvent::NoteOff {
			channel: 0,
			note,
			velocity: 0.0,
		};

		// The same note twice, the second before the first has finished releasing.
		let _ = render(
			&mut poly,
			8,
			&[(0, note(60)), (2, off(60)), (3, note(60)), (5, off(60))],
		);

		assert!(
			poly.voices.iter().all(|voice| voice.released),
			"a retriggered note was left unreleased"
		);
	}

	#[test]
	fn two_notes_sound_together() {
		let graph = voice_patch();
		let mut poly = Poly::new(&graph, 48_000).expect("poly builds");
		let both = render(&mut poly, 8, &[(0, note(60)), (2, note(72))]);

		let mut poly = Poly::new(&graph, 48_000).expect("poly builds");
		let first_only = render(&mut poly, 8, &[(0, note(60))]);

		let mut poly = Poly::new(&graph, 48_000).expect("poly builds");
		let second_only = render(&mut poly, 8, &[(2, note(72))]);

		assert!(both != first_only, "the second note was lost");
		assert!(both != second_only, "the first note was lost");
	}

	#[test]
	fn a_note_off_lets_the_voice_go_silent() {
		let graph = voice_patch();
		let mut poly = Poly::new(&graph, 48_000).expect("poly builds");
		let frames = render(
			&mut poly,
			140,
			&[
				(0, note(60)),
				(
					8,
					NoteEvent::NoteOff {
						channel: 0,
						note: 60,
						velocity: 0.0,
					},
				),
			],
		);

		let tail = &frames[(140 - 4) * BLOCK..];
		let peak = tail
			.iter()
			.flat_map(|frame| frame.iter())
			.fold(0.0f32, |peak, sample| peak.max(sample.abs()));

		assert!(peak < 1e-4, "the voice did not release: {peak}");
	}

	#[test]
	fn a_graph_without_a_voice_group_is_refused() {
		let mut graph = Graph::new();
		let midi = graph.add_node("input:midi", (0.0, 0.0)).expect("midi");
		let generator = graph.add_node("synth:additive", (100.0, 0.0)).expect("gen");
		let out = graph.add_node("out:audio", (200.0, 0.0)).expect("out");
		graph.connect((midi, 0), (generator, 0));
		graph.connect((generator, 0), (out, 0));

		assert!(matches!(
			Poly::new(&graph, 48_000),
			Err(EngineError::NoVoiceGroup)
		));
	}
	#[test]
	fn a_control_tap_reads_one_voice_not_their_sum() {
		let graph = voice_patch();
		let lfo = graph
			.nodes
			.iter()
			.find(|node| node.kind.id == "lfo")
			.expect("lfo")
			.id;
		let mut poly = Poly::new(&graph, 48_000).expect("poly builds");
		let input = vec![[0.0f32; 2]; BLOCK];
		let mut out = vec![[0.0f32; 2]; BLOCK];

		poly.process(&input, &[note(60)], &mut out);

		for _ in 0..8 {
			poly.process(&input, &[], &mut out);
		}

		poly.process(&input, &[note(67)], &mut out);

		for _ in 0..8 {
			poly.process(&input, &[], &mut out);
		}

		assert_eq!(
			poly.voices.len(),
			2,
			"the second note did not start a voice"
		);

		let newest = poly.voices.last().expect("a voice");
		let step = newest
			.engine
			.step_of(lfo)
			.expect("the lfo is in the voice graph");
		let expected = newest.engine.tap_block(step).expect("a block");
		let tapped = poly.tap_voice_block(lfo).expect("a tap");

		assert_eq!(
			tapped.frame(0),
			expected.frame(0),
			"the tap added the voices up"
		);
		assert!(
			tapped.frame(0)[0].abs() <= 1.0,
			"the value left the lfo's range: {}",
			tapped.frame(0)[0]
		);
		assert_eq!(
			poly.control_phase(lfo),
			newest.engine.control_phase(step),
			"the playhead came from another voice"
		);
	}

	#[test]
	fn a_voice_node_can_be_watched() {
		let mut graph = crate::graph::Graph::new();
		let midi = graph.add_node("input:midi", (0.0, 0.0)).expect("midi");
		let voice_in = graph.add_node("voice:in", (100.0, 0.0)).expect("voice in");
		let generator = graph
			.add_node("synth:additive", (200.0, 0.0))
			.expect("generator");
		let gain = graph.add_node("mod:gain", (300.0, 0.0)).expect("gain");
		let voice_out = graph
			.add_node("voice:out", (400.0, 0.0))
			.expect("voice out");
		let out = graph.add_node("out:audio", (500.0, 0.0)).expect("out");
		graph.connect((midi, 0), (voice_in, 0));
		graph.connect((voice_in, 0), (generator, 0));
		graph.connect((generator, 0), (gain, 0));
		graph.connect((gain, 0), (voice_out, 0));
		graph.connect((voice_out, 0), (out, 0));

		let mut poly = Poly::new(&graph, 48_000).expect("a poly");
		let input = [[0.0f32; 2]; crate::block::BLOCK];
		let mut output = [[0.0f32; 2]; crate::block::BLOCK];
		let event = NoteEvent::NoteOn {
			channel: 0,
			note: 60,
			velocity: 0.8,
		};

		poly.process(&input, &[event], &mut output);

		let block = poly
			.tap_voice_block(gain)
			.expect("the gain inside the voice");
		let peak = block.channels[0]
			.iter()
			.fold(0.0f32, |peak, sample| peak.max(sample.abs()));

		assert!(peak > 1e-4, "the voice node was silent: {peak}");
	}
}
