//! Offline checks that do not need a page or a host.

use std::collections::BTreeMap;

use i_am_dsp::prelude::NoteEvent;
use i_am_patch::block::{AudioBlock, BLOCK, BlockContext, Buffer};
use i_am_patch::engine::Engine;
use i_am_patch::graph::Graph;
use i_am_patch::nodes::make;
use i_am_patch::text::{from_text, to_text};

fn note(note: usize) -> NoteEvent {
	NoteEvent::NoteOn {
		channel: 0,
		note,
		velocity: 1.0,
	}
}

fn render(graph: &Graph, blocks: usize, notes: &[(u64, NoteEvent)]) -> Vec<[f32; 2]> {
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

fn peak(frames: &[[f32; 2]]) -> f32 {
	frames
		.iter()
		.flat_map(|frame| frame.iter())
		.fold(0.0f32, |peak, sample| peak.max(sample.abs()))
}

fn chain(kind: &str, params: &[(&str, f32)]) -> Graph {
	let mut graph = Graph::new();
	let midi = graph.add_node("input:midi", (0.0, 0.0)).expect("midi");
	let generator = graph.add_node("gen:additive", (100.0, 0.0)).expect("gen");
	let node = graph.add_node(kind, (200.0, 0.0)).expect("node");
	let out = graph.add_node("out:audio", (300.0, 0.0)).expect("out");
	graph.connect((midi, 0), (generator, 0));
	graph.connect((generator, 0), (node, 0));
	graph.connect((node, 0), (out, 0));

	for (key, value) in params {
		assert!(graph.set_param(node, key, *value), "{key}");
	}

	graph
}

#[test]
fn the_input_passes_through_untouched() {
	let mut graph = Graph::new();
	let input = graph.add_node("input:audio", (0.0, 0.0)).expect("input");
	let out = graph.add_node("out:audio", (100.0, 0.0)).expect("out");
	graph.connect((input, 0), (out, 0));

	let mut engine = Engine::new(&graph, 48_000).expect("builds");
	let mut expected = Vec::new();
	let mut actual = Vec::new();

	for block in 0..2 {
		let mut host = vec![[0.0f32; 2]; BLOCK];
		for (frame, sample) in host.iter_mut().enumerate() {
			sample[0] = (block * BLOCK + frame) as f32 / 1_000.0;
			sample[1] = -sample[0];
		}
		let mut out = vec![[0.0f32; 2]; BLOCK];

		engine.process(&host, &[], &mut out);
		expected.extend(host);
		actual.extend(out);
	}

	for (want, got) in expected.iter().zip(&actual) {
		assert!((want[0] - got[0]).abs() < 1e-6, "{want:?} vs {got:?}");
		assert!((want[1] - got[1]).abs() < 1e-6);
	}
}

#[test]
fn two_edges_into_one_input_sum() {
	let mut graph = Graph::new();
	let midi = graph.add_node("input:midi", (0.0, 0.0)).expect("midi");
	let generator = graph.add_node("gen:additive", (100.0, 0.0)).expect("gen");
	let out = graph.add_node("out:audio", (200.0, 0.0)).expect("out");
	graph.connect((midi, 0), (generator, 0));
	graph.connect((generator, 0), (out, 0));

	let single = peak(&render(&graph, 8, &[(0, note(60))]));

	graph.connect((generator, 0), (out, 0));

	let doubled = peak(&render(&graph, 8, &[(0, note(60))]));

	assert!(doubled > single * 1.8, "{single} then {doubled}");
}

#[test]
fn a_lower_gain_is_quieter() {
	let loud = peak(&render(&chain("gain", &[("gain", 1.0)]), 8, &[(0, note(60))]));
	let quiet = peak(&render(&chain("gain", &[("gain", 0.25)]), 8, &[(0, note(60))]));

	assert!(quiet < loud * 0.6, "{quiet} vs {loud}");
}

#[test]
fn a_low_cutoff_darkens_the_voice() {
	let brightness = |cutoff: f32| {
		let graph = chain("filter", &[("cutoff", cutoff)]);
		let frames = render(&graph, 8, &[(0, note(60))]);
		let high: f32 = frames
			.windows(2)
			.map(|pair| (pair[1][0] - pair[0][0]).powi(2))
			.sum();
		let all: f32 = frames.iter().map(|frame| frame[0] * frame[0]).sum();

		high / all.max(1e-12)
	};

	let dark = brightness(300.0);
	let bright = brightness(8_000.0);

	assert!(dark < bright * 0.5, "{dark} vs {bright}");
}

#[test]
fn the_delay_lands_one_block_late() {
	let mut graph = Graph::new();
	let input = graph.add_node("input:audio", (0.0, 0.0)).expect("input");
	let delay = graph.add_node("delay", (100.0, 0.0)).expect("delay");
	let out = graph.add_node("out:audio", (200.0, 0.0)).expect("out");
	graph.connect((input, 0), (delay, 0));
	graph.connect((delay, 0), (out, 0));

	let mut engine = Engine::new(&graph, 48_000).expect("builds");
	let mut first = vec![[0.0f32; 2]; BLOCK];
	first[5] = [1.0, 1.0];
	let mut out = vec![[0.0f32; 2]; BLOCK];
	engine.process(&first, &[], &mut out);
	assert!(out.iter().all(|frame| frame[0].abs() < 1e-6));

	let silence = vec![[0.0f32; 2]; BLOCK];
	engine.process(&silence, &[], &mut out);
	assert!((out[5][0] - 1.0).abs() < 1e-6, "{:?}", out[5]);
}

#[test]
fn a_modulation_ramps_across_the_block() {
	let mut graph = Graph::new();
	let input = graph.add_node("input:audio", (0.0, 0.0)).expect("input");
	let gain = graph.add_node("gain", (100.0, 0.0)).expect("gain");
	let out = graph.add_node("out:audio", (200.0, 0.0)).expect("out");
	graph.connect((input, 0), (gain, 0));
	graph.connect((gain, 0), (out, 0));

	let lfo = graph.add_node("lfo", (0.0, 100.0)).expect("lfo");
	graph.set_param(lfo, "rate", 100.0);
	graph.modulate((lfo, 0), (gain, "gain"), 0.5);

	let mut engine = Engine::new(&graph, 48_000).expect("builds");
	let mut frames = Vec::new();

	for _ in 0..8 {
		let host = vec![[0.5f32; 2]; BLOCK];
		let mut block = vec![[0.0f32; 2]; BLOCK];
		engine.process(&host, &[], &mut block);
		frames.extend(block);
	}

	// A staircase leaves a constant input constant inside a block; a ramp moves
	// it, because the parameter is between the two ends of the block.
	let start = 4 * BLOCK;
	let first = frames[start][0];
	let moved = frames[start..start + BLOCK]
		.iter()
		.any(|frame| (frame[0] - first).abs() > 1e-4);

	assert!(moved, "the modulation was still a per-block staircase");
}

#[test]
fn the_render_length_follows_the_seconds() {
	let mut graph = Graph::new();
	let midi = graph.add_node("input:midi", (0.0, 0.0)).expect("midi");
	let generator = graph.add_node("gen:additive", (100.0, 0.0)).expect("gen");
	let out = graph.add_node("out:audio", (200.0, 0.0)).expect("out");
	graph.connect((midi, 0), (generator, 0));
	graph.connect((generator, 0), (out, 0));

	let frames = Engine::render_graph(&graph, 48_000, 0.5, &[(0, note(60))]).expect("renders");
	let expected = (0.5f32 * 48_000.0 / BLOCK as f32).ceil() as usize * BLOCK;

	assert_eq!(frames.len(), expected);
}

#[test]
fn the_same_graph_renders_the_same_frames() {
	let graph = chain("gain", &[("gain", 0.7)]);
	let first = render(&graph, 8, &[(0, note(60))]);
	let second = render(&graph, 8, &[(0, note(60))]);

	assert_eq!(first, second);
}

#[test]
fn the_text_round_trip_keeps_the_params() {
	let graph = chain("filter", &[("cutoff", 500.0)]);
	let text = to_text(&graph);
	let back = from_text(&text).expect("parses");

	assert_eq!(back.nodes.len(), graph.nodes.len());
	assert_eq!(back.edges.len(), graph.edges.len());

	let original = graph.get(graph.nodes[2].id).expect("filter");
	let parsed = back.get(back.nodes[2].id).expect("filter");
	assert!((original.param("cutoff") - parsed.param("cutoff")).abs() < 1e-6);
}

#[test]
fn a_control_node_moves_and_an_envelope_follows() {
	let params = BTreeMap::new();
	let mut lfo = make("lfo", 48_000, &params).expect("lfo");
	let mut out = vec![Buffer::Control(0.0)];
	let mut context = BlockContext {
		frames: BLOCK,
		rate: 48_000,
		block: 0,
		events: Vec::new(),
		context: Box::new(i_am_patch::block::PatchContext::new(48_000)),
	};

	lfo.process(&mut context, &[], &mut out);
	let first = match out[0] {
		Buffer::Control(value) => value,
		_ => panic!("lfo wrote an audio buffer"),
	};
	lfo.process(&mut context, &[], &mut out);
	let second = match out[0] {
		Buffer::Control(value) => value,
		_ => panic!("lfo wrote an audio buffer"),
	};
	assert!(first != second, "{first} then {second}");

	let mut envelope = make("envelope", 48_000, &params).expect("envelope");
	let mut input = vec![Buffer::Audio(AudioBlock::silent(BLOCK))];
	if let Buffer::Audio(block) = &mut input[0] {
		block.frames = BLOCK;
		block.channels[0].fill(0.5);
		block.channels[1].fill(0.5);
	}
	let mut out = vec![Buffer::Control(0.0)];

	for _ in 0..8 {
		envelope.process(&mut context, &input, &mut out);
	}

	let value = match out[0] {
		Buffer::Control(value) => value,
		_ => panic!("envelope wrote an audio buffer"),
	};
	assert!(value > 0.1, "the envelope did not follow: {value}");
}
