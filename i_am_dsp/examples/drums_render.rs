//! Renders a short drum pattern to a WAV, for listening without a host.
//!
//! cargo run -p i_am_dsp --example drums_render -- target/wav/drums.wav

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use i_am_dsp::prelude::DrumInstrument;

fn main() {
	let path = std::env::args()
		.nth(1)
		.unwrap_or_else(|| "target/wav/drums.wav".to_string());
	const RATE: usize = 48_000;

	let mut kit = DrumInstrument::new(RATE);
	// (time, note, velocity)
	let mut hits: Vec<(f32, u8, f32)> = vec![
		(0.00, 36, 1.0),
		(0.00, 49, 0.8),
		(0.45, 38, 0.9),
		(0.90, 36, 0.85),
		(1.10, 38, 0.8),
		(1.55, 38, 0.45),
		(1.68, 38, 0.35),
		(2.05, 36, 0.95),
		(2.50, 38, 1.0),
		(2.55, 39, 0.7),
		(2.95, 36, 0.8),
		(3.10, 45, 0.8),
		(3.25, 48, 0.85),
		(3.60, 51, 0.6),
	];

	// A hat on every eighth, and an open one where the pattern needs air.
	for step in 0..14 {
		let time = 0.225 * step as f32;
		let velocity = if step % 2 == 0 { 0.5 } else { 0.32 };
		hits.push((time, 42, velocity));
	}
	hits.push((1.80, 46, 0.7));

	hits.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

	let total = (RATE as f32 * 4.6) as usize;
	let mut samples = Vec::with_capacity(total);
	let mut next = 0usize;

	for index in 0..total {
		let now = index as f32 / RATE as f32;

		while next < hits.len() && hits[next].0 <= now {
			let (_, note, velocity) = hits[next];
			kit.start(note, velocity);
			next += 1;
		}

		samples.push(kit.next_sample());
	}

	let peak = samples.iter().fold(0.0f32, |peak, sample| peak.max(sample.abs()));
	let scale = if peak > 0.0 { 0.7 / peak } else { 1.0 };

	if let Some(parent) = Path::new(&path).parent() {
		std::fs::create_dir_all(parent).expect("create the output directory");
	}

	write_wav(Path::new(&path), &samples, RATE as u32, scale).expect("write the wav");
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
