//! Renders a line patch to a WAV, the agent's loop without a host.
//!
//! cargo run -p i_am_patch --example play -- i_am_patch/patches/demo.patch target/wav/demo.wav [seconds]

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use i_am_dsp::prelude::NoteEvent;
use i_am_patch::engine::Engine;
use i_am_patch::text::from_text;

fn main() {
	let mut arguments = std::env::args().skip(1);
	let input = arguments
		.next()
		.unwrap_or_else(|| "i_am_patch/patches/demo.patch".to_string());
	let output = arguments
		.next()
		.unwrap_or_else(|| "target/wav/patch.wav".to_string());
	let seconds: f32 = arguments
		.next()
		.and_then(|value| value.parse().ok())
		.unwrap_or(1.2);
	const RATE: usize = 48_000;

	let text = std::fs::read_to_string(&input).expect("read the patch");
	let graph = from_text(&text).expect("parse the patch");

	// A note at the start and one every quarter second, so a slow envelope is
	// audible without the patch having to carry its own sequencer yet.
	let events: Vec<(u64, NoteEvent)> = (0..4)
		.map(|index: u64| {
			(
				index * 48_000 / 64 / 4,
				NoteEvent::NoteOn {
					channel: 0,
					note: 60 + index as usize,
					velocity: 0.9,
				},
			)
		})
		.collect();

	let frames = Engine::render_graph(&graph, RATE, seconds, &events).expect("render the patch");
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
