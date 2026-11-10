//! Renders the fitted instrument, to check it plays what the table says.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use i_am_string::fitted;

fn main() {
	let directory = std::env::args()
		.nth(1)
		.unwrap_or_else(|| "target/wav/fitted".to_string());
	std::fs::create_dir_all(&directory).expect("create the output directory");
	let rate = 48_000.0;
	for note in [38u8, 50, 62, 74, 86] {
		for (name, velocity) in [("soft", 0.2f32), ("hard", 1.0f32)] {
			let mut voice = fitted::Voice::start(note, velocity, 0.2, rate);
			let mut samples = Vec::with_capacity(48_000 * 2);
			for _ in 0..48_000 * 2 {
				samples.push(voice.next_sample());
			}
			let path = Path::new(&directory).join(format!("{note:03}-{name}.wav"));
			write_wav(&path, &samples, 48_000).expect("write the wav");
			println!("{note} {name}: {}", path.display());
		}
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
