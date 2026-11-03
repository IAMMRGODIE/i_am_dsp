//! Renders a few reference notes to WAV files, for comparing the string models.
//!
//! ```text
//! cargo run -p i_am_string --example render -- target/wav
//! ```

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use i_am_string::prelude::*;

fn main() {
	let directory = std::env::args()
		.nth(1)
		.unwrap_or_else(|| "target/wav".to_string());
	std::fs::create_dir_all(&directory).expect("create the output directory");

	let cases = [
		(
			"violin-bow-waveguide",
			Preset::Violin,
			Excitation::Bow,
			Core::Waveguide,
		),
		(
			"violin-bow-modal",
			Preset::Violin,
			Excitation::Bow,
			Core::Modal,
		),
		(
			"violin-pluck-waveguide",
			Preset::Violin,
			Excitation::Pluck,
			Core::Waveguide,
		),
		(
			"violin-hammer-waveguide",
			Preset::Violin,
			Excitation::Hammer,
			Core::Waveguide,
		),
	];

	for (name, preset, excitation, core) in cases {
		let mut engine = StringEngine::new(48_000.0);
		engine.set_preset(preset);
		{
			let params = engine.params_mut();
			params.excitation = excitation;
			params.core = core;
			params.strings = 1;
		}
		engine.note_on(69, 0.8);

		let mut samples = Vec::with_capacity(48_000 * 2);
		for _ in 0..48_000 * 2 {
			samples.push(engine.next_sample());
		}

		let path = Path::new(&directory).join(format!("{name}.wav"));
		write_wav(&path, &samples, 48_000).expect("write the wav");
		println!("{name}: {}", path.display());
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
