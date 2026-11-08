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

/// How many notes can sound at once.
pub const POLYPHONY: usize = 8;

struct Voice {
	engine: Engine,
	pending: Vec<NoteEvent>,
	note: usize,
	age: u64,
	silent: u32,
}

/// A polyphonic patch: one voice group and the graph around it.
pub struct Poly {
	outer: Engine,
	voice_graph: Graph,
	rate: usize,
	out: NodeId,
	voice_latency: usize,
	voices: Vec<Voice>,
}

impl Poly {
	/// Splits the graph and builds the outer engine.
	pub fn new(graph: &Graph, rate: usize) -> Result<Self, EngineError> {
		let group = voice_group(graph);

		if group.is_empty() {
			return Err(EngineError::NoVoiceGroup);
		}

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
			outer: Engine::new(&outer, rate)?,
			voice_graph,
			rate,
			out,
			voice_latency,
			voices: Vec::new(),
		})
	}

	/// How many frames the outer graph and the voice graph delay the output.
	pub fn latency(&self) -> usize {
		self.outer.latency() + self.voice_latency
	}

	/// Runs one block: notes start voices, the voices are summed and the outer
	/// graph processes the rest.
	pub fn process(&mut self, input: &[[f32; 2]], events: &[NoteEvent], out: &mut [[f32; 2]]) {
		let frames = input.len().min(BLOCK).min(out.len());

		for event in events {
			match event {
				NoteEvent::NoteOn { note, .. } => {
					if self.voices.len() >= POLYPHONY {
						self.voices.remove(0);
					}

					if let Ok(engine) = Engine::new(&self.voice_graph, self.rate) {
						self.voices.push(Voice {
							engine,
							pending: vec![event.clone()],
							note: *note,
							age: 0,
							silent: 0,
						});
					}
				}
				NoteEvent::NoteOff { note, .. } => {
					if let Some(voice) = self.voices.iter_mut().find(|voice| voice.note == *note) {
						voice.pending.push(event.clone());
					}
				}
				_ => {}
			}
		}

		let mut mix = AudioBlock::silent(frames);
		let silence = vec![[0.0f32; 2]; BLOCK];

		for voice in &mut self.voices {
			let mut buffer = vec![[0.0f32; 2]; BLOCK];

			voice.engine.process(&silence, &voice.pending, &mut buffer);
			voice.pending.clear();
			voice.age += 1;

			for (frame, sample) in buffer.iter().take(frames).enumerate() {
				mix.channels[0][frame] += sample[0];
				mix.channels[1][frame] += sample[1];
			}

			// A released voice is retired once it has been quiet for a while.
			let peak = buffer
				.iter()
				.take(frames)
				.fold(0.0f32, |peak, frame| {
					peak.max(frame[0].abs()).max(frame[1].abs())
				});

			if peak < 1e-4 {
				voice.silent += 1;
			} else {
				voice.silent = 0;
			}
		}

		// A voice lives until it has been silent for a while or the pool needs
		// the slot; a note-off is not needed for the additive voice yet.
		self.voices
			.retain(|voice| voice.age < 10_000 && voice.silent < 150);

		self.outer.inject(self.out, mix);
		self.outer.process(input, events, out);
	}
}


#[cfg(test)]
mod tests {
	use super::*;
	use i_am_dsp::prelude::NoteEvent;

	fn note(note: usize) -> NoteEvent {
		NoteEvent::NoteOn {
			channel: 0,
			note,
			velocity: 1.0,
		}
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
		let generator = graph.add_node("gen:additive", (100.0, 0.0)).expect("gen");
		let out = graph.add_node("out:audio", (200.0, 0.0)).expect("out");
		graph.connect((midi, 0), (generator, 0));
		graph.connect((generator, 0), (out, 0));

		assert!(matches!(
			Poly::new(&graph, 48_000),
			Err(EngineError::NoVoiceGroup)
		));
	}
}
