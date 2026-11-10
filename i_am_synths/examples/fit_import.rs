//! Renders a fitted drum bank, for listening to an analysis.
//!
//! cargo run -p i_am_synths --example fit_import -- preset.json out.wav

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use i_am_dsp::prelude::DrumInstrument;
use i_am_dsp::prelude::DrumPreset;

fn main() {
	let mut arguments = std::env::args().skip(1);
	let input = arguments
		.next()
		.unwrap_or_else(|| "preset.json".to_string());
	let output = arguments
		.next()
		.unwrap_or_else(|| "target/wav/fitted.wav".to_string());

	let text = std::fs::read_to_string(&input).expect("read the preset");
	let preset: DrumPreset = serde_json::from_str(&text).expect("parse the preset");

	println!(
		"{}: {} partials, {} noise bands, {} s",
		if preset.name.is_empty() {
			"preset"
		} else {
			&preset.name
		},
		preset.partials.len(),
		preset.noise.len(),
		preset.decay
	);

	let mut kit = DrumInstrument::new(48_000);
	kit.set_fitted(Some(preset));
	// The crash is the drum the fitted bank replaces.
	let samples = kit.render(49, 1.0, 3.0);

	let peak = samples
		.iter()
		.fold(0.0f32, |peak, sample| peak.max(sample.abs()));
	let scale = if peak > 0.0 { 0.7 / peak } else { 1.0 };

	if let Some(parent) = Path::new(&output).parent() {
		std::fs::create_dir_all(parent).expect("create the output directory");
	}
	write_wav(Path::new(&output), &samples, 48_000, scale).expect("write the wav");
	println!("{output}: {} samples, peak {peak:.3}", samples.len());
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
