//! Renders one cymbal hit, for A/B between the synth tail and an IR.
//!
//! cargo run -p i_am_dsp --example cymbal_render -- target/wav/cymbal.wav [tail] [fm] [ride]

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use i_am_dsp::prelude::DrumInstrument;

fn main() {
	let mut arguments = std::env::args().skip(1);
	let output = arguments
		.next()
		.unwrap_or_else(|| "target/wav/cymbal.wav".to_string());
	let tail: f32 = arguments.next().and_then(|value| value.parse().ok()).unwrap_or(0.8);
	let fm: f32 = arguments.next().and_then(|value| value.parse().ok()).unwrap_or(0.7);
	let ride = arguments.next().is_some();

	let mut kit = DrumInstrument::new(48_000);
	kit.cymbal_tail = tail;
	kit.cymbal_fm = fm;
	let note = if ride { 51 } else { 49 };
	let samples = kit.render(note, 1.0, 4.0);

	let peak = samples.iter().fold(0.0f32, |peak, sample| peak.max(sample.abs()));
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
