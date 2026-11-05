//! Renders the piano model one note at a time, for measuring partial decays.
//!
//! ```text
//! cargo run --release -p i_am_string --example probe_piano -- target/wav/piano-t60 30 28 40 52 64 76
//! ```

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use i_am_string::prelude::*;

fn main() {
	let mut args = std::env::args().skip(1);
	let directory = args.next().unwrap_or_else(|| "target/wav/piano-t60".to_string());
	let seconds: usize = args
		.next()
		.and_then(|value| value.parse().ok())
		.unwrap_or(30);
	let notes: Vec<u8> = args.filter_map(|value| value.parse().ok()).collect();
	let notes = if notes.is_empty() {
		vec![28, 40, 52, 64, 76]
	} else {
		notes
	};
	std::fs::create_dir_all(&directory).expect("create the output directory");

	for note in notes {
		let rate = 48_000.0;
		let mut engine = StringEngine::new(rate);
		engine.set_preset(Preset::Piano);
		engine.note_on(note, 0.8);

		let mut samples = Vec::with_capacity((rate as usize) * seconds);
		for _ in 0..(rate as usize) * seconds {
			samples.push(engine.next_sample());
		}

		let path = Path::new(&directory).join(format!("piano-{note:03}.wav"));
		write_wav(&path, &samples, 48_000).expect("write the wav");
		println!("{note}: {}", path.display());
	}
}

fn write_wav(path: &Path, samples: &[f32], sample_rate: u32) -> std::io::Result<()> {
	let mut writer = BufWriter::new(File::create(path)?);
	let data_len = (samples.len() * 2) as u32;

	writer.write_all(b"RIFF")?;
	writer.write_all(&(36 + data_len).to_le_bytes())?;
	writer.write_all(b"WAVEfmt ")?;
	writer.write_all(&16u32.to_le_bytes())?;
	writer.write_all(&1u16.to_le_bytes())?;
	writer.write_all(&1u16.to_le_bytes())?;
	writer.write_all(&sample_rate.to_le_bytes())?;
	writer.write_all(&(sample_rate * 2).to_le_bytes())?;
	writer.write_all(&2u16.to_le_bytes())?;
	writer.write_all(&16u16.to_le_bytes())?;
	writer.write_all(b"data")?;
	writer.write_all(&data_len.to_le_bytes())?;

	for sample in samples {
		let value = (sample.clamp(-1.0, 1.0) * 32_767.0) as i16;
		writer.write_all(&value.to_le_bytes())?;
	}

	writer.flush()
}