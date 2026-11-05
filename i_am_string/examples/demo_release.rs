//! Renders the same guitar note with a stab and with a held release.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use i_am_string::prelude::*;

const FS: f32 = 44_100.0;

fn main() {
	let directory = std::env::args()
		.nth(1)
		.unwrap_or_else(|| "target/wav/voicing-demo".to_string());
	std::fs::create_dir_all(&directory).expect("create the output directory");
	for (name, held) in [("stab", 0.10f32), ("held", 0.90)] {
		let mut engine = StringEngine::new(FS);
		engine.set_preset(Preset::Cello);
		engine.params_mut().strings = 1;
		engine.params_mut().sympathetic = 0.0;
		engine.note_on(40, 0.85);
		let mut audio = Vec::new();
		for _ in 0..(FS * held) as usize {
			audio.push(engine.next_sample());
		}
		engine.note_off(40);
		for _ in 0..(FS * 2.0) as usize {
			audio.push(engine.next_sample());
		}
		let path = Path::new(&directory).join(format!("{name}.wav"));
		write_wav(&path, &audio, FS as u32).expect("write the wav");
		println!("{} frames into {}", audio.len(), path.display());
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
