//! Rough timing for the patcher, for a release build.
//!
//! ```text
//! cargo run --release -p i_am_patch --example bench
//! ```
//!
//! Each line renders five seconds of audio through one patch and prints the CPU
//! time it took, as a share of the audio's own length. A host needs the whole
//! block to fit in the buffer it hands over, so lower is better.

use std::time::Instant;

use i_am_dsp::NoteEvent;
use i_am_patch::block::BLOCK;
use i_am_patch::graph::{Graph, NodeId};
use i_am_patch::poly::Poly;

const RATE: usize = 48_000;
const BLOCKS: usize = 3_750; // five seconds

fn note(note: usize) -> NoteEvent {
	NoteEvent::NoteOn {
		channel: 0,
		note,
		velocity: 0.8,
	}
}

fn off(note: usize) -> NoteEvent {
	NoteEvent::NoteOff {
		channel: 0,
		note,
		velocity: 0.0,
	}
}

fn run(name: &str, graph: &Graph, blocks: usize, events: &[(usize, NoteEvent)]) {
	let mut poly = Poly::new(graph, RATE).expect("the patch plays");
	let input = vec![[0.0f32; 2]; BLOCK];
	let mut out = vec![[0.0f32; 2]; BLOCK];
	let mut due = Vec::with_capacity(4);

	let started = Instant::now();

	for block in 0..blocks {
		due.clear();
		due.extend(events.iter().filter(|(at, _)| *at == block).map(|(_, event)| event.clone()));
		poly.process(&input, &due, &mut out);
	}

	let elapsed = started.elapsed().as_secs_f64();
	let audio = blocks as f64 * BLOCK as f64 / RATE as f64;

	println!(
		"{name:28} {:8.2} ms  {:6.1}% of real time  {:6.2} us/block",
		elapsed * 1000.0,
		elapsed / audio * 100.0,
		elapsed * 1e6 / blocks as f64
	);
}

/// A graph with no voice group: the plain engine's block loop.
fn run_engine(name: &str, graph: &Graph, blocks: usize) {
	let engine_error = i_am_patch::engine::Engine::new(graph, RATE);
	let mut engine = engine_error.expect("the patch plays");
	let input = vec![[0.0f32; 2]; BLOCK];
	let mut out = vec![[0.0f32; 2]; BLOCK];

	let started = Instant::now();

	for _ in 0..blocks {
		engine.process(&input, &[], &mut out);
	}

	let elapsed = started.elapsed().as_secs_f64();
	let audio = blocks as f64 * BLOCK as f64 / RATE as f64;

	println!(
		"{name:28} {:8.2} ms  {:6.1}% of real time  {:6.2} us/block",
		elapsed * 1000.0,
		elapsed / audio * 100.0,
		elapsed * 1e6 / blocks as f64
	);
}

/// A one-voice patch: a table into an envelope, inside the voice group.
fn envelope_graph(table: &str) -> (Graph, NodeId) {
	let mut graph = Graph::new();
	let midi = graph.add_node("input:midi", (0.0, 0.0)).expect("midi");
	let voice_in = graph.add_node("voice:in", (50.0, 0.0)).expect("voice in");
	let table = graph.add_node(table, (0.0, 100.0)).expect("table");
	let envelope = graph.add_node("adsr", (150.0, 0.0)).expect("adsr");
	let voice_out = graph.add_node("voice:out", (300.0, 0.0)).expect("voice out");
	let out = graph.add_node("out:audio", (400.0, 0.0)).expect("out");

	graph.connect((midi, 0), (voice_in, 0));
	graph.connect((voice_in, 0), (envelope, 0));
	graph.connect((table, 0), (envelope, 1));
	graph.connect((envelope, 0), (voice_out, 0));
	graph.connect((voice_out, 0), (out, 0));

	(graph, table)
}

/// A bare pass-through: the block loop and the buffer copies, no DSP.
fn pass_through() -> Graph {
	let mut graph = Graph::new();
	let input = graph.add_node("input:audio", (0.0, 0.0)).expect("audio");
	let out = graph.add_node("out:audio", (100.0, 0.0)).expect("out");
	graph.connect((input, 0), (out, 0));

	graph
}

/// The same with a run of gain nodes, to price one node's plumbing.
fn gains(count: usize) -> Graph {
	let mut graph = Graph::new();
	let input = graph.add_node("input:audio", (0.0, 0.0)).expect("audio");
	let mut previous = (input, 0);

	for index in 0..count {
		let gain = graph
			.add_node("mod:gain", (100.0 * (index + 1) as f32, 0.0))
			.expect("gain");
		graph.connect(previous, (gain, 0));
		previous = (gain, 0);
	}

	let out = graph.add_node("out:audio", (100.0 * (count + 1) as f32, 0.0)).expect("out");
	graph.connect(previous, (out, 0));

	graph
}

/// A modulation chain: table into a warp into an FM into the envelope.
fn chain_graph() -> Graph {
	let mut graph = Graph::new();
	let midi = graph.add_node("input:midi", (0.0, 0.0)).expect("midi");
	let voice_in = graph.add_node("voice:in", (50.0, 0.0)).expect("voice in");
	let carrier = graph.add_node("table:shape", (0.0, 100.0)).expect("carrier");
	let modulator = graph.add_node("table:shape", (0.0, 220.0)).expect("modulator");
	let warp = graph.add_node("warp:sync", (120.0, 220.0)).expect("warp");
	let fm = graph.add_node("osc:fm", (220.0, 100.0)).expect("fm");
	let envelope = graph.add_node("adsr", (320.0, 0.0)).expect("adsr");
	let voice_out = graph.add_node("voice:out", (420.0, 0.0)).expect("voice out");
	let out = graph.add_node("out:audio", (520.0, 0.0)).expect("out");

	graph.connect((midi, 0), (voice_in, 0));
	graph.connect((voice_in, 0), (envelope, 0));
	graph.connect((carrier, 0), (fm, 0));
	graph.connect((modulator, 0), (warp, 0));
	graph.connect((warp, 0), (fm, 1));
	graph.connect((fm, 0), (envelope, 1));
	graph.connect((envelope, 0), (voice_out, 0));
	graph.connect((voice_out, 0), (out, 0));

	graph
}

/// A product chain: a synth product through a filter.
fn product_graph(kind: &str) -> Graph {
	let mut graph = Graph::new();
	let midi = graph.add_node("input:midi", (0.0, 0.0)).expect("midi");
	let voice_in = graph.add_node("voice:in", (50.0, 0.0)).expect("voice in");
	let synth = graph.add_node(kind, (150.0, 0.0)).expect("synth");
	let filter = graph.add_node("mod:filter", (300.0, 0.0)).expect("filter");
	let voice_out = graph.add_node("voice:out", (450.0, 0.0)).expect("voice out");
	let out = graph.add_node("out:audio", (550.0, 0.0)).expect("out");

	graph.connect((midi, 0), (voice_in, 0));
	graph.connect((voice_in, 0), (synth, 0));
	graph.connect((synth, 0), (filter, 0));
	graph.connect((filter, 0), (voice_out, 0));
	graph.connect((voice_out, 0), (out, 0));

	graph
}

fn main() {
	println!("samplerate {RATE} Hz, block {BLOCK} frames, five seconds each");

	run_engine("pass through", &pass_through(), BLOCKS);
	run_engine("1 gain", &gains(1), BLOCKS);
	run_engine("8 gains", &gains(8), BLOCKS);
	run_engine("32 gains", &gains(32), BLOCKS);

	let (table_graph, _) = envelope_graph("table:shape");
	run("envelope, shape table", &table_graph, BLOCKS, &[(0, note(60))]);

	run("envelope, chain", &chain_graph(), BLOCKS, &[(0, note(60))]);

	let (shapes, _) = envelope_graph("table:shapes");
	run("envelope, shapes table", &shapes, BLOCKS, &[(0, note(60))]);

	let (additive_table, _) = envelope_graph("table:additive");
	run("envelope, additive table", &additive_table, BLOCKS, &[(0, note(60))]);

	let (spectrum, _) = envelope_graph("table:spectrum");
	run("envelope, spectrum table", &spectrum, BLOCKS, &[(0, note(60))]);

	run("synth:additive", &product_graph("synth:additive"), BLOCKS, &[(0, note(60))]);
	run("synth:shapes", &product_graph("synth:shapes"), BLOCKS, &[(0, note(60))]);
	run("synth:spectrum", &product_graph("synth:spectrum"), BLOCKS, &[(0, note(60))]);

	// Eight voices at once: the group is built once per note.
	let eight: Vec<(usize, NoteEvent)> = (0..8).map(|index| (index, note(48 + index * 3))).collect();
	run("envelope, eight voices", &table_graph, BLOCKS, &eight);

	// A note every eighth of a second: the per-note cost, not the steady state.
	let churn: Vec<(usize, NoteEvent)> = (0..40)
		.flat_map(|index| {
			let at = index * 47;
			[(at, note(48 + index % 24)), (at + 23, off(48 + index % 24))]
		})
		.collect();
	run("envelope, note churn", &table_graph, BLOCKS, &churn);

	// What the page asks for when it draws a hero: the engine's own cycle.
	let slots = i_am_patch::Slots::default();
	let started = Instant::now();

	for _ in 0..100 {
		let _ = i_am_patch::sample_constructor(&table_graph, &slots, RATE, NodeId(3), 192);
	}

	println!(
		"{:28} {:8.2} ms  100 renders of 192 points",
		"sample_constructor",
		started.elapsed().as_secs_f64() * 1000.0
	);
}
