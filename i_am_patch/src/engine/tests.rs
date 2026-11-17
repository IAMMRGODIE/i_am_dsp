use crate::voice_patch;

use super::*;
use crate::graph::Graph;
use crate::nodes::make;

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
	let generator = graph.add_node("synth:additive", (100.0, 0.0)).expect("gen");
	let gain = graph.add_node("mod:gain", (200.0, 0.0)).expect("gain");
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
	let generator = graph.add_node("synth:additive", (100.0, 0.0)).expect("gen");
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
fn an_lfo_can_follow_points() {
	let mut params = BTreeMap::new();
	params.insert("rate", 1.0);
	// The peak sits just past the first block, so the test reads the table
	// inside one process call instead of racing a 375 Hz LFO past the clamp.
	let points = [
		crate::graph::Point {
			time: 0.0,
			value: 0.0,
			bend: 0.0,
		},
		crate::graph::Point {
			time: 0.001,
			value: 1.0,
			bend: 0.0,
		},
		crate::graph::Point {
			time: 1.0,
			value: 0.0,
			bend: 0.0,
		},
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
			events: vec![NoteEvent::NoteOn {
				channel: 0,
				note: 60,
				velocity: 0.8,
			}],
			context: Box::new(PatchContext::new(48_000)),
		};

		node.set_param("rate", 100.0);
		node.set_param("shape", 0.0);

		let mut moved = false;

		// A random source at the shared 20 Hz ceiling still needs a few hundred
		// milliseconds to step: give the test a full cycle.
		for block in 0..64 {
			context.block = block;
			context.events.clear();
			if block == 0 {
				context.events.push(NoteEvent::NoteOn {
					channel: 0,
					note: 60,
					velocity: 0.8,
				});
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
	let filter = graph.add_node("mod:filter", (100.0, 0.0)).expect("filter");
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

	let peak = frames
		.iter()
		.flat_map(|frame| frame.iter())
		.fold(0.0f32, |peak, sample| peak.max(sample.abs()));
	let mean: f32 = frames.iter().map(|frame| frame[0]).sum::<f32>() / frames.len() as f32;

	assert!(peak < 2.0, "the peak blew up: {peak}");
	assert!(mean.abs() < 0.01, "the output has DC: {mean}");
}

fn modulation_graph(kind: &str) -> (Graph, crate::graph::NodeId, crate::graph::NodeId) {
	let mut graph = Graph::new();
	let midi = graph.add_node("input:midi", (0.0, 0.0)).expect("midi");
	let carrier = graph
		.add_node("table:shape", (0.0, 100.0))
		.expect("carrier");
	let modulator = graph
		.add_node("table:shape", (0.0, 200.0))
		.expect("modulator");
	let oscillator = graph.add_node(kind, (100.0, 150.0)).expect("oscillator");
	let envelope = graph.add_node("adsr", (200.0, 0.0)).expect("adsr");
	let out = graph.add_node("out:audio", (300.0, 0.0)).expect("out");

	graph.connect((midi, 0), (envelope, 0));
	graph.connect((carrier, 0), (oscillator, 0));
	graph.connect((modulator, 0), (oscillator, 1));
	graph.connect((oscillator, 0), (envelope, 1));
	graph.connect((envelope, 0), (out, 0));

	(graph, oscillator, envelope)
}

fn render_blocks_of(graph: &Graph, blocks: usize) -> Vec<[f32; 2]> {
	let mut engine = Engine::new(graph, 48_000).expect("the graph builds");
	let input = vec![[0.0f32; 2]; BLOCK];
	let mut out = vec![[0.0f32; 2]; BLOCK];
	let mut collected = Vec::new();

	for block in 0..blocks {
		let events = if block == 0 {
			vec![note_event(60)]
		} else {
			vec![]
		};

		engine.process(&input, &events, &mut out);
		collected.extend_from_slice(&out);
	}

	collected
}

#[test]
fn a_ring_modulation_follows_its_depth() {
	let rms = |frames: &[[f32; 2]]| {
		(frames.iter().map(|frame| frame[0] * frame[0]).sum::<f32>() / frames.len() as f32).sqrt()
	};
	let (graph, oscillator, envelope) = modulation_graph("osc:rm");
	let mut engine = Engine::new(&graph, 48_000).expect("the graph builds");
	let input = vec![[0.0f32; 2]; BLOCK];
	let mut frames = vec![[0.0f32; 2]; BLOCK];

	// A steady note, so the two measurements differ by the ring alone.
	assert!(engine.set_param(envelope, "random_phase", 0.0));
	assert!(engine.set_param(envelope, "attack_time", 0.001));
	assert!(engine.set_param(envelope, "decay_time", 0.001));
	assert!(engine.set_param(envelope, "sustain_level", 1.0));
	assert!(engine.set_param(oscillator, "depth", 0.0));

	let mut carrier = Vec::new();

	for block in 0..8 {
		let events = if block == 0 {
			vec![note_event(60)]
		} else {
			vec![]
		};

		engine.process(&input, &events, &mut frames);

		if block >= 6 {
			carrier.extend_from_slice(&frames);
		}
	}

	// A full ring multiplies the carrier by the modulator, which halves its
	// level for two sines; depth zero leaves the carrier alone.
	assert!(engine.set_param(oscillator, "depth", 1.0));

	let mut ring = Vec::new();

	for block in 8..12 {
		engine.process(&input, &[], &mut frames);

		if block >= 10 {
			ring.extend_from_slice(&frames);
		}
	}

	let carrier = rms(&carrier);
	let ring = rms(&ring);

	assert!(carrier > 1e-3, "depth zero was silent: {carrier}");
	assert!(
		ring < carrier * 0.95,
		"the depth did not reach the sound: {carrier} against {ring}"
	);
}

#[test]
fn the_modulation_nodes_play() {
	for kind in ["osc:am", "osc:pm", "osc:fm"] {
		let (graph, _, _) = modulation_graph(kind);

		assert!(
			render_blocks_of(&graph, 4)
				.iter()
				.any(|frame| frame[0].abs() > 1e-3),
			"{kind} was silent"
		);
	}
}

#[test]
fn a_product_table_plays() {
	for kind in [
		"table:additive",
		"table:spectrum",
		"table:script",
		"table:canvas",
	] {
		let mut graph = Graph::new();
		let midi = graph.add_node("input:midi", (0.0, 0.0)).expect("midi");
		let oscillator = graph.add_node(kind, (0.0, 100.0)).expect("osc");
		let envelope = graph.add_node("adsr", (200.0, 0.0)).expect("adsr");
		let out = graph.add_node("out:audio", (300.0, 0.0)).expect("out");

		graph.connect((midi, 0), (envelope, 0));
		graph.connect((oscillator, 0), (envelope, 1));
		graph.connect((envelope, 0), (out, 0));

		assert!(render(&graph) > 1e-3, "{kind} was silent");
	}
}

#[test]
fn a_canvas_table_plays_through_the_envelope() {
	let mut graph = Graph::new();
	let midi = graph.add_node("input:midi", (0.0, 0.0)).expect("midi");
	let table = graph.add_node("table:canvas", (0.0, 100.0)).expect("table");
	let envelope = graph.add_node("adsr", (200.0, 0.0)).expect("adsr");
	let out = graph.add_node("out:audio", (300.0, 0.0)).expect("out");

	graph.connect((midi, 0), (envelope, 0));
	graph.connect((table, 0), (envelope, 1));
	graph.connect((envelope, 0), (out, 0));

	assert!(render(&graph) > 1e-3);
}

/// The shapes oscillator reads the page's wavetable slot; an empty one is
/// silent, but the graph still has to build.
#[test]
fn a_shapes_oscillator_builds_without_a_table() {
	let mut graph = Graph::new();
	let midi = graph.add_node("input:midi", (0.0, 0.0)).expect("midi");
	let oscillator = graph.add_node("table:shapes", (0.0, 100.0)).expect("osc");
	let envelope = graph.add_node("adsr", (200.0, 0.0)).expect("adsr");
	let out = graph.add_node("out:audio", (300.0, 0.0)).expect("out");

	graph.connect((midi, 0), (envelope, 0));
	graph.connect((oscillator, 0), (envelope, 1));
	graph.connect((envelope, 0), (out, 0));

	Engine::new(&graph, 48_000).expect("the graph builds");
}

#[test]
fn a_table_node_takes_its_product_parameters() {
	let mut graph = Graph::new();
	let midi = graph.add_node("input:midi", (0.0, 0.0)).expect("midi");
	let table = graph.add_node("table:shapes", (0.0, 100.0)).expect("table");
	let envelope = graph.add_node("adsr", (200.0, 0.0)).expect("adsr");
	let out = graph.add_node("out:audio", (300.0, 0.0)).expect("out");

	graph.connect((midi, 0), (envelope, 0));
	graph.connect((table, 0), (envelope, 1));
	graph.connect((envelope, 0), (out, 0));

	let mut engine = Engine::new(&graph, 48_000).expect("the graph builds");

	assert!(
		engine.set_param(table, "oscillator.smooth_factor", 0.5),
		"the table dropped the product parameter"
	);
	assert!(
		!engine.set_param(table, "gain", 0.5),
		"the envelope's output stage belongs to the envelope"
	);
}

#[test]
fn a_warp_between_tables_plays() {
	let mut graph = Graph::new();
	let midi = graph.add_node("input:midi", (0.0, 0.0)).expect("midi");
	let table = graph.add_node("table:shape", (0.0, 100.0)).expect("table");
	let warp = graph.add_node("warp:sync", (100.0, 100.0)).expect("warp");
	let envelope = graph.add_node("adsr", (200.0, 0.0)).expect("adsr");
	let out = graph.add_node("out:audio", (300.0, 0.0)).expect("out");

	graph.connect((midi, 0), (envelope, 0));
	graph.connect((table, 0), (warp, 0));
	graph.connect((warp, 0), (envelope, 1));
	graph.connect((envelope, 0), (out, 0));

	assert!(render(&graph) > 1e-3);
}

#[test]
fn an_envelope_plays_a_wavetable() {
	let mut graph = Graph::new();
	let midi = graph.add_node("input:midi", (0.0, 0.0)).expect("midi");
	let table = graph.add_node("table:shape", (0.0, 100.0)).expect("table");
	let envelope = graph.add_node("adsr", (100.0, 0.0)).expect("adsr");
	let out = graph.add_node("out:audio", (200.0, 0.0)).expect("out");

	graph.connect((midi, 0), (envelope, 0));
	graph.connect((table, 0), (envelope, 1));
	graph.connect((envelope, 0), (out, 0));

	let peak = render(&graph);

	assert!(peak > 1e-3, "the envelope was silent: {peak}");
}

#[test]
fn an_oscillator_plays_without_a_table_node() {
	let mut graph = Graph::new();
	let midi = graph.add_node("input:midi", (0.0, 0.0)).expect("midi");
	let oscillator = graph.add_node("osc:shape", (0.0, 100.0)).expect("osc");
	let envelope = graph.add_node("adsr", (100.0, 0.0)).expect("adsr");
	let out = graph.add_node("out:audio", (200.0, 0.0)).expect("out");

	graph.connect((midi, 0), (envelope, 0));
	graph.connect((oscillator, 0), (envelope, 1));
	graph.connect((envelope, 0), (out, 0));

	assert!(render(&graph) > 1e-3);
}

#[test]
fn the_table_node_shares_its_shape_with_the_oscillator() {
	let mut graph = Graph::new();
	let midi = graph.add_node("input:midi", (0.0, 0.0)).expect("midi");
	let table = graph.add_node("table:shape", (0.0, 100.0)).expect("table");
	let envelope = graph.add_node("adsr", (100.0, 0.0)).expect("adsr");
	let out = graph.add_node("out:audio", (200.0, 0.0)).expect("out");

	graph.connect((midi, 0), (envelope, 0));
	graph.connect((table, 0), (envelope, 1));
	graph.connect((envelope, 0), (out, 0));

	let mut engine = Engine::new(&graph, 48_000).expect("the graph builds");
	let input = vec![[0.0f32; 2]; BLOCK];
	let mut frames = vec![[0.0f32; 2]; BLOCK];

	for block in 0..8 {
		let events = if block == 0 {
			vec![note_event(60)]
		} else {
			vec![]
		};

		engine.process(&input, &events, &mut frames);
	}

	// A sine spends time away from its peak, a square sits on it.
	let fill = |frames: &[[f32; 2]]| {
		let peak = frames
			.iter()
			.map(|frame| frame[0].abs())
			.fold(0.0f32, f32::max);
		let rms = (frames.iter().map(|frame| frame[0] * frame[0]).sum::<f32>()
			/ frames.len() as f32)
			.sqrt();

		rms / peak.max(1e-6)
	};
	let sine = fill(&frames);

	assert!(sine < 0.8, "a sine should not fill its peak: {sine}");

	assert!(engine.set_param(table, "shape", 3.0));

	// One block carries the change into the node, the next one plays it.
	engine.process(&input, &[], &mut frames);
	engine.process(&input, &[], &mut frames);

	let square = fill(&frames);

	assert!(square > 0.9, "the square did not reach the table: {square}");
}

#[test]
fn an_audio_edge_cannot_feed_a_constructor_port() {
	use crate::compile::{CompileError, compile};

	let mut graph = Graph::new();
	let audio = graph.add_node("input:audio", (0.0, 0.0)).expect("audio");
	let envelope = graph.add_node("adsr", (100.0, 0.0)).expect("adsr");

	graph.connect((audio, 0), (envelope, 1));

	assert!(matches!(
		compile(&graph),
		Err(CompileError::KindMismatch { .. })
	));
}

#[test]
fn a_synth_node_plays() {
	let mut graph = Graph::new();
	let midi = graph.add_node("input:midi", (0.0, 0.0)).expect("midi");
	let synth = graph
		.add_node("synth:additive", (100.0, 0.0))
		.expect("synth");
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
	let compressor = graph
		.add_node("mod:compressor", (100.0, 0.0))
		.expect("compressor");
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

	let peak = frames
		.iter()
		.flat_map(|frame| frame.iter())
		.fold(0.0f32, |peak, sample| peak.max(sample.abs()));
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
		let gain = graph.add_node("mod:gain", (100.0, 0.0)).expect("gain");
		let out = graph.add_node("out:audio", (200.0, 0.0)).expect("out");
		graph.connect((input, 0), (gain, 0));
		graph.connect((gain, 0), (out, 0));

		let lfo = graph.add_node("lfo", (0.0, 100.0)).expect("lfo");
		graph.set_param(lfo, "rate", 100.0);
		graph.modulate((lfo, 0), (gain, "gain"), 0.5);
		graph.set_modulation_mode(0, mode);

		let mut engine = Engine::new(&graph, 48_000).expect("builds");
		let mut frames = Vec::new();

		for _ in 0..64 {
			let host = vec![[0.5f32; 2]; BLOCK];
			let mut block = vec![[0.0f32; 2]; BLOCK];
			engine.process(&host, &[], &mut block);
			frames.extend(block);
		}

		let min = frames
			.iter()
			.map(|frame| frame[0])
			.fold(f32::INFINITY, f32::min);
		let max = frames.iter().map(|frame| frame[0]).fold(0.0f32, f32::max);

		(min, max)
	}

	let (centered_min, centered_max) = render(ModulationMode::Centered);
	let (base_min, base_max) = render(ModulationMode::FromBase);

	assert!(
		centered_min < 0.45 && centered_max > 0.55,
		"centered: {centered_min}..{centered_max}"
	);
	assert!(
		base_min > 0.49 && base_max > 0.55,
		"from base: {base_min}..{base_max}"
	);
}

#[test]
fn a_voice_group_restarts_with_the_note() {
	let graph = voice_patch();
	let frames = render_blocks(&graph, 8, &[(0, note_event(60)), (4, note_event(60))]);

	// The LFO, the filter and the generator all start again on the second
	// note, so the first block after each note is the same samples.
	assert_eq!(&frames[0..BLOCK], &frames[BLOCK * 4..BLOCK * 5]);
}

#[test]
fn an_lfo_can_move_a_parameter() {
	let build = |modulated: bool| {
		let mut graph = Graph::new();
		let midi = graph.add_node("input:midi", (0.0, 0.0)).expect("midi");
		let generator = graph.add_node("synth:additive", (100.0, 0.0)).expect("gen");
		for key in ["random_phase", "random_phase_by_channel", "random_pan"] {
			assert!(graph.set_param(generator, key, 0.0), "{key}");
		}
		let gain_node = graph.add_node("mod:gain", (200.0, 0.0)).expect("gain");
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
fn the_sidechain_node_carries_the_host_signal() {
	let mut graph = Graph::new();
	let sidechain = graph
		.add_node("input:sidechain", (0.0, 0.0))
		.expect("sidechain");
	let out = graph.add_node("out:audio", (100.0, 0.0)).expect("out");
	graph.connect((sidechain, 0), (out, 0));

	let mut engine = Engine::new(&graph, 48_000).expect("builds");
	let host = vec![[0.0f32; 2]; BLOCK];
	let side = vec![[0.5f32, -0.25]; BLOCK];
	let mut block = vec![[0.0f32; 2]; BLOCK];
	engine.process_with_sidechain(&host, &side, &[], &mut block);

	assert!((block[3][0] - 0.5).abs() < 1e-6, "{}", block[3][0]);
	assert!((block[3][1] + 0.25).abs() < 1e-6, "{}", block[3][1]);
}

#[test]
fn the_gain_parameter_is_heard() {
	// A constant input, so the peak is the gain itself and not a synth's
	// envelope or phase.
	let chain = |gain: f32| {
		let mut graph = Graph::new();
		let input = graph.add_node("input:audio", (0.0, 0.0)).expect("input");
		let gain_node = graph.add_node("mod:gain", (200.0, 0.0)).expect("gain");
		let out = graph.add_node("out:audio", (300.0, 0.0)).expect("out");
		graph.connect((input, 0), (gain_node, 0));
		graph.connect((gain_node, 0), (out, 0));
		assert!(graph.set_param(gain_node, "gain", gain));

		let mut engine = Engine::new(&graph, 48_000).expect("builds");
		let host = vec![[0.5f32; 2]; BLOCK];
		let mut block = vec![[0.0f32; 2]; BLOCK];
		engine.process(&host, &[], &mut block);
		block
			.iter()
			.fold(0.0f32, |peak, frame| peak.max(frame[0].abs()))
	};

	assert!(chain(0.0) < 1e-6, "a zero gain still sounded");
	assert!(chain(0.5) < chain(1.0) * 0.7, "the gain did nothing");
}

#[test]
fn the_voice_has_no_dc() {
	let mut graph = Graph::new();
	let midi = graph.add_node("input:midi", (0.0, 0.0)).expect("midi");
	let generator = graph.add_node("synth:additive", (100.0, 0.0)).expect("gen");
	let out = graph.add_node("out:audio", (200.0, 0.0)).expect("out");
	graph.connect((midi, 0), (generator, 0));
	graph.connect((generator, 0), (out, 0));

	let frames = Engine::render_graph(
		&graph,
		48_000,
		1.0,
		&[(
			0,
			NoteEvent::NoteOn {
				channel: 0,
				note: 60,
				velocity: 1.0,
			},
		)],
	)
	.expect("renders");
	let mean: f64 = frames.iter().map(|frame| frame[0] as f64).sum::<f64>() / frames.len() as f64;

	assert!(mean.abs() < 0.01, "the voice carries a DC offset: {mean}");
}

#[test]
fn the_offline_render_produces_a_note() {
	let mut graph = Graph::new();
	let midi = graph.add_node("input:midi", (0.0, 0.0)).expect("midi");
	let generator = graph.add_node("synth:additive", (100.0, 0.0)).expect("gen");
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
	let generator = graph.add_node("synth:additive", (100.0, 0.0)).expect("gen");
	let filter = graph.add_node("mod:filter", (200.0, 0.0)).expect("filter");
	let out = graph.add_node("out:audio", (300.0, 0.0)).expect("out");
	graph.connect((midi, 0), (generator, 0));
	graph.connect((generator, 0), (filter, 0));
	graph.connect((filter, 0), (out, 0));

	assert!(render(&graph) > 1e-3);
}

#[test]
fn an_lfo_plays_its_points() {
	use crate::graph::Point;

	let mut graph = Graph::new();
	let input = graph.add_node("input:audio", (0.0, 0.0)).expect("input");
	let lfo = graph.add_node("lfo", (0.0, 100.0)).expect("lfo");
	// The mixer's a is a plain 0..2 level, so the depth is the offset it adds.
	let mixer = graph.add_node("mixer", (100.0, 0.0)).expect("mixer");
	let out = graph.add_node("out:audio", (200.0, 0.0)).expect("out");
	graph.connect((input, 0), (mixer, 0));
	graph.connect((mixer, 0), (out, 0));
	// A flat curve at the top: the level doubles across the whole block.
	graph.set_points(
		lfo,
		vec![
			Point {
				time: 0.0,
				value: 1.0,
				bend: 0.0,
			},
			Point {
				time: 1.0,
				value: 1.0,
				bend: 0.0,
			},
		],
	);
	graph.modulate((lfo, 0), (mixer, "a"), 1.0);

	let mut engine = Engine::new(&graph, 48_000).expect("builds");
	let host = vec![[0.5f32; 2]; BLOCK];
	let mut block = vec![[0.0f32; 2]; BLOCK];

	// The first block ramps from the source's previous value; the second one
	// already knows the curve sits at the top.
	engine.process(&host, &[], &mut block);
	engine.process(&host, &[], &mut block);

	assert!(
		(block[BLOCK - 1][0] - 1.0).abs() < 1e-3,
		"{}",
		block[BLOCK - 1][0]
	);
}

#[test]
fn a_unipolar_source_rises_from_zero() {
	use crate::graph::ModulationMode;

	// The MIDI source is 0..1: from-base has to add that value, not the
	// bipolar fold (value + 1) / 2, which would start halfway up.
	let mut graph = Graph::new();
	let input = graph.add_node("input:audio", (0.0, 0.0)).expect("input");
	let midi_in = graph.add_node("input:midi", (0.0, 100.0)).expect("midi in");
	let midi = graph.add_node("midi", (100.0, 100.0)).expect("midi");
	let mixer = graph.add_node("mixer", (100.0, 0.0)).expect("mixer");
	let out = graph.add_node("out:audio", (200.0, 0.0)).expect("out");
	graph.connect((input, 0), (mixer, 0));
	graph.connect((mixer, 0), (out, 0));
	graph.connect((midi_in, 0), (midi, 0));
	// The mixer's a spans 0..2, so a depth of half a range is the whole unit
	// the source adds; from-base adds the source value itself.
	graph.modulate((midi, 0), (mixer, "a"), 0.5);
	assert!(graph.set_modulation_mode(0, ModulationMode::FromBase));

	let mut engine = Engine::new(&graph, 48_000).expect("builds");
	let host = vec![[0.5f32; 2]; BLOCK];
	let mut block = vec![[0.0f32; 2]; BLOCK];
	let note = NoteEvent::NoteOn {
		channel: 0,
		note: 60,
		velocity: 1.0,
	};
	engine.process(&host, &[note], &mut block);
	engine.process(&host, &[], &mut block);

	// 1 + 60 / 127 = 1.472; the bipolar fold would have given 1.736.
	let expected = 0.5 * (1.0 + 60.0 / 127.0);
	assert!(
		(block[BLOCK - 1][0] - expected).abs() < 1e-3,
		"{} vs {expected}",
		block[BLOCK - 1][0]
	);
}

#[test]
fn one_note_reaches_every_synth_node() {
	// The graph broadcasts notes: two synth nodes both play the one key. That
	// is the modular design, not a duplicated note.
	let mut graph = Graph::new();
	let midi = graph.add_node("input:midi", (0.0, 0.0)).expect("midi");
	let first = graph
		.add_node("synth:additive", (100.0, 0.0))
		.expect("first");
	let second = graph
		.add_node("synth:shapes", (100.0, 100.0))
		.expect("second");
	let out = graph.add_node("out:audio", (300.0, 0.0)).expect("out");
	graph.connect((midi, 0), (first, 0));
	graph.connect((midi, 0), (second, 0));
	graph.connect((first, 0), (out, 0));
	graph.connect((second, 0), (out, 0));

	let mut engine = Engine::new(&graph, 48_000).expect("builds");
	let host = vec![[0.0f32; 2]; BLOCK];
	let mut block = vec![[0.0f32; 2]; BLOCK];
	engine.process(
		&host,
		&[NoteEvent::NoteOn {
			channel: 0,
			note: 60,
			velocity: 0.8,
		}],
		&mut block,
	);
	engine.process(&host, &[], &mut block);

	let peak = |node| {
		let step = engine.step_of(node).expect("a step");
		let block = engine.tap_block(step).expect("a tap");

		block.channels[0]
			.iter()
			.fold(0.0f32, |peak, sample| peak.max(sample.abs()))
	};

	assert!(peak(first) > 1e-4, "the first synth did not play");
	assert!(peak(second) > 1e-4, "the second synth did not play");
}

#[test]
fn the_output_gain_scales_the_mix() {
	let render = |gain: f32| {
		let mut graph = Graph::new();
		let input = graph.add_node("input:audio", (0.0, 0.0)).expect("input");
		let out = graph.add_node("out:audio", (100.0, 0.0)).expect("out");
		graph.connect((input, 0), (out, 0));
		assert!(graph.set_param(out, "gain", gain));

		let mut engine = Engine::new(&graph, 48_000).expect("builds");
		let host = vec![[0.5f32; 2]; BLOCK];
		let mut block = vec![[0.0f32; 2]; BLOCK];
		engine.process(&host, &[], &mut block);
		block[BLOCK - 1][0]
	};

	assert!((render(1.0) - 0.5).abs() < 1e-6, "{}", render(1.0));
	assert!((render(0.5) - 0.25).abs() < 1e-6, "{}", render(0.5));
	assert!(render(0.0).abs() < 1e-6, "{}", render(0.0));
}

#[test]
fn the_voice_group_does_not_double_the_level() {
	let peak = |frames: &[[f32; 2]]| {
		frames
			.iter()
			.fold(0.0f32, |peak, frame| peak.max(frame[0].abs()))
	};

	let note = NoteEvent::NoteOn {
		channel: 0,
		note: 60,
		velocity: 0.8,
	};

	// The default graph, through Poly (the product's path).
	let mut graph = Graph::new();
	let midi = graph.add_node("input:midi", (0.0, 120.0)).expect("midi");
	let voice_in = graph
		.add_node("voice:in", (340.0, 120.0))
		.expect("voice in");
	let generator = graph
		.add_node("synth:additive", (680.0, 120.0))
		.expect("gen");
	let filter = graph
		.add_node("mod:filter", (1020.0, 120.0))
		.expect("filter");
	let lfo = graph.add_node("lfo", (1020.0, 580.0)).expect("lfo");
	let voice_out = graph
		.add_node("voice:out", (1620.0, 120.0))
		.expect("voice out");
	let out = graph.add_node("out:audio", (1960.0, 120.0)).expect("out");
	graph.connect((midi, 0), (voice_in, 0));
	graph.connect((voice_in, 0), (generator, 0));
	graph.connect((generator, 0), (filter, 0));
	graph.connect((filter, 0), (voice_out, 0));
	graph.connect((voice_out, 0), (out, 0));
	graph.modulate((lfo, 0), (filter, "cutoff"), 800.0);
	graph.set_param(filter, "cutoff", 1_200.0);
	graph.set_param(generator, "random_phase", 0.0);
	graph.set_param(generator, "random_pan", 0.0);
	let mut poly = crate::poly::Poly::new(&graph, 48_000).expect("poly builds");
	let mut frames = Vec::new();
	let events = [(0u64, note.clone())];
	for block in 0..128u64 {
		let due: Vec<NoteEvent> = events
			.iter()
			.filter(|(at, _)| *at == block)
			.map(|(_, event)| event.clone())
			.collect();
		let input = vec![[0.0f32; 2]; BLOCK];
		let mut out = vec![[0.0f32; 2]; BLOCK];
		poly.process(&input, &due, &mut out);
		frames.extend(out);
	}
	let poly_peak = peak(&frames);

	// The same chain without the voice boundary.
	let mut plain = Graph::new();
	let midi = plain.add_node("input:midi", (0.0, 0.0)).expect("midi");
	let synth = plain
		.add_node("synth:additive", (100.0, 0.0))
		.expect("synth");
	let filter = plain.add_node("mod:filter", (200.0, 0.0)).expect("filter");
	let out = plain.add_node("out:audio", (300.0, 0.0)).expect("out");
	let lfo = plain.add_node("lfo", (100.0, 200.0)).expect("lfo");
	plain.connect((midi, 0), (synth, 0));
	plain.connect((synth, 0), (filter, 0));
	plain.connect((filter, 0), (out, 0));
	plain.modulate((lfo, 0), (filter, "cutoff"), 800.0);
	plain.set_param(filter, "cutoff", 1_200.0);
	plain.set_param(synth, "random_phase", 0.0);
	plain.set_param(synth, "random_pan", 0.0);
	let frames =
		Engine::render_graph(&plain, 48_000, 8192.0 / 48_000.0, &[(0, note)]).expect("renders");
	let plain_peak = peak(&frames);

	assert!(
		(poly_peak - plain_peak).abs() < 0.01 * plain_peak,
		"the voice group doubled the level: {poly_peak} vs {plain_peak}"
	);
}

/// Runs a graph on a host signal and returns what came out.
fn render_host(graph: &Graph, blocks: usize, host: [f32; 2]) -> Vec<[f32; 2]> {
	let mut engine = Engine::new(graph, 48_000).expect("the graph builds");
	let mut out = Vec::new();

	for _ in 0..blocks {
		let input = vec![host; BLOCK];
		let mut block = vec![[0.0f32; 2]; BLOCK];

		engine.process(&input, &[], &mut block);
		out.extend(block);
	}

	out
}

#[test]
fn an_lr_split_and_mix_is_the_identity() {
	let mut graph = Graph::new();
	let input = graph.add_node("input:audio", (0.0, 0.0)).expect("input");
	let split = graph.add_node("split:lr", (100.0, 0.0)).expect("split");
	let mix = graph.add_node("mix:lr", (200.0, 0.0)).expect("mix");
	let out = graph.add_node("out:audio", (300.0, 0.0)).expect("out");
	graph.connect((input, 0), (split, 0));
	graph.connect((split, 0), (mix, 0));
	graph.connect((split, 1), (mix, 1));
	graph.connect((mix, 0), (out, 0));

	let frames = render_host(&graph, 2, [0.5, -0.25]);

	for frame in &frames {
		assert!((frame[0] - 0.5).abs() < 1e-6, "left {}", frame[0]);
		assert!((frame[1] + 0.25).abs() < 1e-6, "right {}", frame[1]);
	}
}

#[test]
fn a_ms_split_and_mix_is_the_identity() {
	let mut graph = Graph::new();
	let input = graph.add_node("input:audio", (0.0, 0.0)).expect("input");
	let split = graph.add_node("split:ms", (100.0, 0.0)).expect("split");
	let mix = graph.add_node("mix:ms", (200.0, 0.0)).expect("mix");
	let out = graph.add_node("out:audio", (300.0, 0.0)).expect("out");
	graph.connect((input, 0), (split, 0));
	graph.connect((split, 0), (mix, 0));
	graph.connect((split, 1), (mix, 1));
	graph.connect((mix, 0), (out, 0));

	let frames = render_host(&graph, 2, [0.5, -0.25]);

	for frame in &frames {
		assert!((frame[0] - 0.5).abs() < 1e-6, "left {}", frame[0]);
		assert!((frame[1] + 0.25).abs() < 1e-6, "right {}", frame[1]);
	}
}

/// Renders a sine through a graph and returns how loud each of the split
/// node's three outputs ended up.
fn band_levels(frequency: f32) -> [f32; 3] {
	let mut graph = Graph::new();
	let input = graph.add_node("input:audio", (0.0, 0.0)).expect("input");
	let split = graph.add_node("split:bands", (100.0, 0.0)).expect("split");
	graph.connect((input, 0), (split, 0));

	let mut engine = Engine::new(&graph, 48_000).expect("the graph builds");
	let step = engine.step_of(split).expect("the split is in the plan");
	let mut levels = [0.0f32; 3];

	for block in 0..32 {
		let mut input = vec![[0.0f32; 2]; BLOCK];

		for (frame, slot) in input.iter_mut().enumerate() {
			let index = block * BLOCK + frame;
			let value = (std::f32::consts::TAU * frequency * index as f32 / 48_000.0).sin() * 0.5;
			*slot = [value, value];
		}

		let mut out = vec![[0.0f32; 2]; BLOCK];
		engine.process(&input, &[], &mut out);

		// The filters settle in the first blocks; the rest is the steady state.
		if block >= 16 {
			for (port, level) in levels.iter_mut().enumerate() {
				let band = engine.output_block(step, port).expect("a band").frame(0);
				*level = (*level).max(band[0].abs());
			}
		}
	}

	levels
}

#[test]
fn the_bands_carry_what_they_name() {
	let low = band_levels(30.0);
	let high = band_levels(10_000.0);

	assert!(
		low[0] > 10.0 * low[2] && low[0] > 2.0 * low[1],
		"30 Hz did not land in the low band: {low:?}"
	);
	assert!(
		high[2] > 10.0 * high[0] && high[2] > 2.0 * high[1],
		"10 kHz did not land in the high band: {high:?}"
	);
}

#[test]
fn a_band_split_sums_back_to_the_input() {
	let mut graph = Graph::new();
	let input = graph.add_node("input:audio", (0.0, 0.0)).expect("input");
	let split = graph.add_node("split:bands", (100.0, 0.0)).expect("split");
	let mix = graph.add_node("mix:bands", (200.0, 0.0)).expect("mix");
	let out = graph.add_node("out:audio", (300.0, 0.0)).expect("out");
	graph.connect((input, 0), (split, 0));
	graph.connect((split, 0), (mix, 0));
	graph.connect((split, 1), (mix, 1));
	graph.connect((split, 2), (mix, 2));
	graph.connect((mix, 0), (out, 0));

	let mut engine = Engine::new(&graph, 48_000).expect("the graph builds");
	let mut peak = 0.0f32;

	for block in 0..32 {
		let mut input = vec![[0.0f32; 2]; BLOCK];

		for (frame, slot) in input.iter_mut().enumerate() {
			let index = block * BLOCK + frame;
			let value = (std::f32::consts::TAU * 1_000.0 * index as f32 / 48_000.0).sin() * 0.5;
			*slot = [value, value];
		}

		let mut out = vec![[0.0f32; 2]; BLOCK];
		engine.process(&input, &[], &mut out);

		if block >= 16 {
			for frame in &out {
				peak = peak.max(frame[0].abs());
			}
		}
	}

	// A 1 kHz sine sampled at 48 kHz peaks at 0.5 within a thousandth.
	let decibels = 20.0 * (peak / 0.5).log10();

	assert!(
		decibels.abs() < 0.2,
		"the bands did not sum back: {decibels:.3} dB"
	);
}

#[test]
fn a_one_shot_lfo_stops_at_the_end_of_its_cycle() {
	// A control source needs no wiring: the engine runs it before the audio path.
	let mut graph = Graph::new();
	let lfo = graph.add_node("lfo", (0.0, 0.0)).expect("lfo");
	graph.set_param(lfo, "rate", 20.0);
	graph.set_param(lfo, "one_shot", 1.0);

	let mut engine = Engine::new(&graph, 48_000).expect("builds");
	let step = engine.step_of(lfo).expect("the lfo is in the plan");
	let input = vec![[0.0f32; 2]; BLOCK];
	let mut block = vec![[0.0f32; 2]; BLOCK];

	// 20 Hz is a 2400 frame cycle, so forty blocks is past the end of it.
	for _ in 0..40 {
		engine.process(&input, &[], &mut block);
	}

	let phase = engine.control_phase(step).expect("a phase");
	let held = engine.tap_block(step).expect("a value").frame(0)[0];

	assert!(
		(phase - 1.0).abs() < 1e-3,
		"the one-shot did not stop: {phase}"
	);

	for _ in 0..8 {
		engine.process(&input, &[], &mut block);
	}

	let phase = engine.control_phase(step).expect("a phase");
	let after = engine.tap_block(step).expect("a value").frame(0)[0];

	assert!((phase - 1.0).abs() < 1e-3, "the phase wrapped: {phase}");
	assert_eq!(after, held, "the held value moved");
}

#[test]
fn a_looping_lfo_keeps_wrapping() {
	let mut graph = Graph::new();
	let lfo = graph.add_node("lfo", (0.0, 0.0)).expect("lfo");
	graph.set_param(lfo, "rate", 20.0);

	let mut engine = Engine::new(&graph, 48_000).expect("builds");
	let step = engine.step_of(lfo).expect("the lfo is in the plan");
	let input = vec![[0.0f32; 2]; BLOCK];
	let mut block = vec![[0.0f32; 2]; BLOCK];

	for _ in 0..40 {
		engine.process(&input, &[], &mut block);
	}

	let phase = engine.control_phase(step).expect("a phase");
	assert!(phase < 1.0, "the loop parked at the end: {phase}");
}

/// The steady-state peak a sine leaves after a graph's filters settle.
fn engine_peak(graph: &Graph, frequency: f32) -> f32 {
	let mut engine = Engine::new(graph, 48_000).expect("builds");
	let mut peak = 0.0f32;

	for block in 0..32 {
		let mut input = vec![[0.0f32; 2]; BLOCK];

		for (frame, slot) in input.iter_mut().enumerate() {
			let index = block * BLOCK + frame;
			let value = (std::f32::consts::TAU * frequency * index as f32 / 48_000.0).sin() * 0.5;
			*slot = [value, value];
		}

		let mut out = vec![[0.0f32; 2]; BLOCK];
		engine.process(&input, &[], &mut out);

		if block >= 16 {
			for frame in &out {
				peak = peak.max(frame[0].abs());
			}
		}
	}

	peak
}

#[test]
fn a_band_mix_gain_balances_its_band() {
	let mut graph = Graph::new();
	let input = graph.add_node("input:audio", (0.0, 0.0)).expect("input");
	let split = graph.add_node("split:bands", (100.0, 0.0)).expect("split");
	let mix = graph.add_node("mix:bands", (200.0, 0.0)).expect("mix");
	let out = graph.add_node("out:audio", (300.0, 0.0)).expect("out");
	graph.connect((input, 0), (split, 0));
	graph.connect((split, 0), (mix, 0));
	graph.connect((split, 1), (mix, 1));
	graph.connect((split, 2), (mix, 2));
	graph.connect((mix, 0), (out, 0));
	let full = engine_peak(&graph, 1_000.0);
	graph.set_param(mix, "mid", 0.0);
	let muted = engine_peak(&graph, 1_000.0);

	assert!(full > 0.4, "the mid band was not carrying the tone: {full}");
	assert!(
		muted < 0.1 * full,
		"muting mid left {muted} of {full}: the gain did not reach the band"
	);
}

#[test]
fn a_one_shot_lfo_holds_the_end_of_its_curve() {
	use crate::graph::Point;

	let mut graph = Graph::new();
	let lfo = graph.add_node("lfo", (0.0, 0.0)).expect("lfo");
	graph.set_param(lfo, "rate", 20.0);
	graph.set_param(lfo, "one_shot", 1.0);
	// A ramp from the bottom of the curve to the top: one shot has to end high.
	graph.set_points(
		lfo,
		vec![
			Point {
				time: 0.0,
				value: 0.0,
				bend: 0.0,
			},
			Point {
				time: 1.0,
				value: 1.0,
				bend: 0.0,
			},
		],
	);

	let mut engine = Engine::new(&graph, 48_000).expect("builds");
	let step = engine.step_of(lfo).expect("the lfo is in the plan");
	let input = vec![[0.0f32; 2]; BLOCK];
	let mut block = vec![[0.0f32; 2]; BLOCK];

	for _ in 0..48 {
		engine.process(&input, &[], &mut block);
	}

	let held = engine.tap_block(step).expect("a value").frame(0)[0];

	assert!(
		(held - 1.0).abs() < 0.01,
		"the one-shot held {held}, not the end of its curve"
	);
}

#[test]
fn a_parameter_moves_through_the_running_engine() {
	// A knob is not a graph change: the engine that is playing takes the value
	// and keeps running, which is what leaves the notes in flight alone.
	let mut graph = Graph::new();
	let input = graph.add_node("input:audio", (0.0, 0.0)).expect("input");
	let gain = graph.add_node("mod:gain", (100.0, 0.0)).expect("gain");
	let out = graph.add_node("out:audio", (200.0, 0.0)).expect("out");
	graph.connect((input, 0), (gain, 0));
	graph.connect((gain, 0), (out, 0));

	let mut engine = Engine::new(&graph, 48_000).expect("builds");
	let host = vec![[0.5f32; 2]; BLOCK];
	let mut block = vec![[0.0f32; 2]; BLOCK];

	engine.process(&host, &[], &mut block);
	let before = block[BLOCK - 1][0];

	assert!(engine.set_param(gain, "gain", 2.0));
	engine.process(&host, &[], &mut block);

	assert!(
		block[BLOCK - 1][0] > before * 1.5,
		"{} did not follow {before}",
		block[BLOCK - 1][0]
	);

	// A parameter the node does not have is refused rather than invented.
	assert!(!engine.set_param(gain, "nope", 1.0));
	assert!(!engine.set_param(crate::graph::NodeId(99), "gain", 1.0));
}

#[test]
fn a_logarithmic_parameter_moves_by_a_ratio() {
	use crate::graph::Point;

	// The gain's level is logarithmic, so a depth is a share of its curve: the
	// offset that doubles it once is the same offset anywhere on the knob.
	let level = crate::graph::node("mod:gain")
		.expect("gain")
		.params
		.iter()
		.find(|param| param.name == "gain")
		.copied()
		.expect("gain");
	let depth = level.normalize(2.0) - level.normalize(1.0);

	let mut graph = Graph::new();
	let input = graph.add_node("input:audio", (0.0, 0.0)).expect("input");
	let gain = graph.add_node("mod:gain", (100.0, 0.0)).expect("gain");
	let out = graph.add_node("out:audio", (200.0, 0.0)).expect("out");
	let lfo = graph.add_node("lfo", (0.0, 100.0)).expect("lfo");
	graph.connect((input, 0), (gain, 0));
	graph.connect((gain, 0), (out, 0));
	graph.set_param(gain, "gain", 1.0);
	// A flat curve at the top, so the offset is the whole depth.
	graph.set_points(
		lfo,
		vec![
			Point {
				time: 0.0,
				value: 1.0,
				bend: 0.0,
			},
			Point {
				time: 1.0,
				value: 1.0,
				bend: 0.0,
			},
		],
	);
	graph.modulate((lfo, 0), (gain, "gain"), depth);

	let mut engine = Engine::new(&graph, 48_000).expect("builds");
	let host = vec![[0.5f32; 2]; BLOCK];
	let mut block = vec![[0.0f32; 2]; BLOCK];

	engine.process(&host, &[], &mut block);
	engine.process(&host, &[], &mut block);

	// A gain of 2 on a 0.5 input is 1.0; a linear reading of the same depth
	// would have left it near 1.0 instead of doubling it.
	assert!(
		(block[BLOCK - 1][0] - 1.0).abs() < 1e-2,
		"{}",
		block[BLOCK - 1][0]
	);
}
