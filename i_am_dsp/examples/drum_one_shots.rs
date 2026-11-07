//! Renders one-shot WAVs for every drum in the kit, at two velocities.
//!
//! cargo run -p i_am_dsp --example drum_one_shots -- target/wav/oneshots

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use i_am_dsp::prelude::DrumInstrument;

fn main() {
	let directory = std::env::args()
		.nth(1)
		.unwrap_or_else(|| "target/wav/oneshots".to_string());
	std::fs::create_dir_all(&directory).expect("create the output directory");

	let drums: [(&str, u8, f32); 9] = [
		("kick", 36, 2.0),
		("snare", 38, 1.2),
		("clap", 39, 1.0),
		("hat-closed", 42, 0.4),
		("hat-open", 46, 1.6),
		("tom-low", 43, 1.2),
		("tom-high", 48, 1.2),
		("crash", 49, 4.0),
		("ride", 51, 3.0),
	];

	for (name, note, seconds) in drums {
		for velocity in [0.5f32, 1.0] {
			let mut kit = DrumInstrument::new(48_000);
			let samples = kit.render(note, velocity, seconds);
			let path = Path::new(&directory).join(format!("{name}-{}0.wav", (velocity * 10.0) as u8));
			write_wav(&path, &samples, 48_000).expect("write the wav");
			println!("{}", path.display());
		}
	}
}

fn write_wav(path: &Path, samples: &[f32], rate: u32) -> std::io::Result<()> {
	let peak = samples.iter().fold(0.0f32, |peak, sample| peak.max(sample.abs()));
	let scale = if peak > 0.0 { 0.7 / peak } else { 1.0 };
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
