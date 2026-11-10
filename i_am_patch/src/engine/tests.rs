use crate::voice_patch;

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
fn the_gain_parameter_is_heard() {
	let chain = |gain: f32| {
		let mut graph = Graph::new();
		let midi = graph.add_node("input:midi", (0.0, 0.0)).expect("midi");
		let generator = graph.add_node("synth:additive", (100.0, 0.0)).expect("gen");
		let gain_node = graph.add_node("mod:gain", (200.0, 0.0)).expect("gain");
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
