//! Renders a small patch to a WAV, for an agent or a person to listen.
//!
//! cargo run -p i_am_patch --example render -- target/wav/patch.wav

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use i_am_dsp::prelude::NoteEvent;
use i_am_patch::engine::Engine;
use i_am_patch::graph::Graph;

fn main() {
	let output = std::env::args()
		.nth(1)
		.unwrap_or_else(|| "target/wav/patch.wav".to_string());
	const RATE: usize = 48_000;

	let mut graph = Graph::new();
	let midi = graph.add_node("input:midi", (0.0, 0.0)).expect("midi");
	let generator = graph.add_node("synth:additive", (100.0, 0.0)).expect("gen");
	let filter = graph.add_node("mod:filter", (200.0, 0.0)).expect("filter");
	let gain = graph.add_node("mod:gain", (300.0, 0.0)).expect("gain");
	let out = graph.add_node("out:audio", (400.0, 0.0)).expect("out");

	graph.connect((midi, 0), (generator, 0));
	graph.connect((generator, 0), (filter, 0));
	graph.connect((filter, 0), (gain, 0));
	graph.connect((gain, 0), (out, 0));

	graph.set_param(filter, "cutoff", 1_200.0);
	graph.set_param(gain, "gain", 0.8);

	let events: Vec<(u64, NoteEvent)> = (0..4)
		.map(|index| {
			(
				(index as u64) * 48_000 / 64 / 4,
				NoteEvent::NoteOn {
					channel: 0,
					note: 60 + index,
					velocity: 0.9,
				},
			)
		})
		.collect();

	let frames = Engine::render_graph(&graph, RATE, 1.2, &events).expect("the patch renders");
	let peak = frames
		.iter()
		.flat_map(|frame| frame.iter())
		.fold(0.0f32, |peak, sample| peak.max(sample.abs()));
	let scale = if peak > 0.0 { 0.7 / peak } else { 1.0 };

	if let Some(parent) = Path::new(&output).parent() {
		std::fs::create_dir_all(parent).expect("create the output directory");
	}
	write_wav(Path::new(&output), &frames, RATE as u32, scale).expect("write the wav");
	println!("{output}: {} frames, peak {peak:.3}", frames.len());
}

fn write_wav(path: &Path, frames: &[[f32; 2]], rate: u32, scale: f32) -> std::io::Result<()> {
	let mut writer = BufWriter::new(File::create(path)?);
	let data_len = (frames.len() * 8) as u32;

	writer.write_all(b"RIFF")?;
	writer.write_all(&(36 + data_len).to_le_bytes())?;
	writer.write_all(b"WAVEfmt ")?;
	writer.write_all(&16u32.to_le_bytes())?;
	writer.write_all(&1u16.to_le_bytes())?;
	writer.write_all(&2u16.to_le_bytes())?;
	writer.write_all(&rate.to_le_bytes())?;
	writer.write_all(&(rate * 8).to_le_bytes())?;
	writer.write_all(&8u16.to_le_bytes())?;
	writer.write_all(&16u16.to_le_bytes())?;
	writer.write_all(b"data")?;
	writer.write_all(&data_len.to_le_bytes())?;

	for frame in frames {
		for sample in frame {
			let value = (sample * scale).clamp(-1.0, 1.0);
			let bits = (value * 32_767.0) as i16;
			writer.write_all(&bits.to_le_bytes())?;
		}
	}

	writer.flush()
}
