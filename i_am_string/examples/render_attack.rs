//! Renders a short phrase so the attack can be listened to.
//!
//! cargo run -p i_am_string --example render_attack -- target/wav/strings-attack.wav

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use i_am_string::prelude::*;

fn main() {
	let path = std::env::args()
		.nth(1)
		.unwrap_or_else(|| "target/wav/strings-attack.wav".to_string());
	let rate = 48_000.0f32;
	let mut engine = StringEngine::new(rate);
	engine.set_preset(Preset::Violin);

	let mut params = Preset::Violin.defaults();
	params.excitation = Excitation::Pluck;
	engine.apply_params(params.clone());

	let mut samples: Vec<f32> = Vec::new();
	let seconds = |engine: &mut StringEngine, samples: &mut Vec<f32>, length: f32| {
		for _ in 0..(rate * length) as usize {
			samples.push(engine.next_sample());
		}
	};

	// Four plucks on A3, the tails overlapping, which is where the thump used
	// to come back.
	for _ in 0..4 {
		engine.note_on(57, 0.85);
		seconds(&mut engine, &mut samples, 0.28);
		engine.note_off(57);
		seconds(&mut engine, &mut samples, 0.22);
	}

	// A low C2 twice, where the fixed fourteen milliseconds sat right on the
	// fundamental's first cycle.
	for _ in 0..2 {
		engine.note_on(36, 0.9);
		seconds(&mut engine, &mut samples, 0.5);
		engine.note_off(36);
		seconds(&mut engine, &mut samples, 0.5);
	}

	// A hammer note for comparison.
	params.excitation = Excitation::Hammer;
	engine.apply_params(params.clone());
	engine.note_on(60, 0.9);
	seconds(&mut engine, &mut samples, 1.2);

	// And a bowed note held for a moment.
	params.excitation = Excitation::Bow;
	engine.apply_params(params);
	engine.note_on(64, 0.8);
	seconds(&mut engine, &mut samples, 1.6);
	engine.note_off(64);
	seconds(&mut engine, &mut samples, 0.4);

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
		// 16 bit PCM, not the float the engine works in: a float payload behind a
		// 16 bit header reads as noise.
		let value = (sample * scale).clamp(-1.0, 1.0);
		let bits = (value * 32_767.0) as i16;
		writer.write_all(&bits.to_le_bytes())?;
		writer.write_all(&bits.to_le_bytes())?;
	}

	writer.flush()
}
