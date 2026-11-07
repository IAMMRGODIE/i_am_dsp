//! Renders a bowed note, to listen to its start.
//!
//! cargo run -p i_am_string --example render_bow -- target/wav/strings-bow.wav

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use i_am_string::prelude::*;

fn main() {
	let path = std::env::args()
		.nth(1)
		.unwrap_or_else(|| "target/wav/strings-bow.wav".to_string());
	let rate = 48_000.0f32;
	let mut engine = StringEngine::new(rate);
	engine.set_preset(Preset::Violin);

	let mut params = Preset::Violin.defaults();
	params.excitation = Excitation::Bow;
	engine.apply_params(params);

	let mut samples = Vec::new();
	for note in [52u8, 64] {
		engine.note_on(note, 0.8);
		for _ in 0..(rate * 1.6) as usize {
			samples.push(engine.next_sample());
		}
		engine.note_off(note);
		for _ in 0..(rate * 0.5) as usize {
			samples.push(engine.next_sample());
		}
	}

	let peak = samples.iter().fold(0.0f32, |peak, s| peak.max(s.abs()));
	let scale = if peak > 0.0 { 0.7 / peak } else { 1.0 };

	if let Some(parent) = Path::new(&path).parent() {
		std::fs::create_dir_all(parent).expect("create the output directory");
	}
	write_wav(Path::new(&path), &samples, 48_000, scale).expect("write the wav");
	println!("{path}: {} samples, peak {peak:.3}", samples.len());
}

fn write_wav(path: &Path, samples: &[f32], rate: u32, scale: f32) -> std::io::Result<()> {
	let mut writer = BufWriter::new(File::create(path)?);
	let data_len = (samples.len() * 4) as u32;

	writer.write_all(b"RIFF")?;
	writer.write_all(&(36 + data_len).to_le_bytes())?;
	writer.write_all(b"WAVEfmt ")?;
	writer.write_all(&16u32.to_le_bytes())?;
	writer.write_all(&1u16.to_le_bytes())?;
	writer.write_all(&2u16.to_le_bytes())?;
	writer.write_all(&rate.to_le_bytes())?;
	writer.write_all(&(rate * 4).to_le_bytes())?;
	writer.write_all(&4u16.to_le_bytes())?;
	writer.write_all(&16u16.to_le_bytes())?;
	writer.write_all(b"data")?;
	writer.write_all(&data_len.to_le_bytes())?;

	for sample in samples {
		let value = (sample * scale).clamp(-1.0, 1.0);
		let bits = (value * 32_767.0) as i16;
		writer.write_all(&bits.to_le_bytes())?;
		writer.write_all(&bits.to_le_bytes())?;
	}

	writer.flush()
}
