//! Renders a sweep of notes for one preset, so a render can be compared with a
//! real recording note by note.
//!
//!     cargo run -r -p i_am_string --example sweep -- target/sweep cello 43:67
//!     cargo run -r -p i_am_string --example sweep -- target/sweep violin 60:84 3.0
//!
//! The files are named preset_note_hertz.wav and are 44.1 kHz PCM16, so the same
//! analysis script can read them next to the samples they are compared with.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use i_am_string::prelude::*;

fn main() {
	let arguments: Vec<String> = std::env::args().collect();
	let directory = arguments
		.get(1)
		.cloned()
		.unwrap_or_else(|| "target/sweep".to_string());
	let preset_name = arguments
		.get(2)
		.cloned()
		.unwrap_or_else(|| "violin".to_string());
	let notes = arguments
		.get(3)
		.cloned()
		.unwrap_or_else(|| "60:84".to_string());
	let seconds: f32 = arguments
		.get(4)
		.and_then(|value| value.parse().ok())
		.unwrap_or(4.5);

	let excitation = match arguments.get(5).map(String::as_str) {
		Some("pluck") => Excitation::Pluck,
		Some("hammer") => Excitation::Hammer,
		_ => Excitation::Bow,
	};
	let core = match arguments.get(6).map(String::as_str) {
		Some("modal") => Core::Modal,
		_ => Core::Waveguide,
	};
	// The number of unison strings, so a piano can be rendered the way it is
	// played (three) and compared with a recording.
	let strings: usize = arguments
		.get(7)
		.and_then(|value| value.parse().ok())
		.unwrap_or(1);

	let preset = match preset_name.as_str() {
		"piano" => Preset::Piano,
		"guitar" => Preset::Guitar,
		"cello" => Preset::Cello,
		_ => Preset::Violin,
	};

	// A range ("48:60"), one note ("69" or "A4"), or a list ("32,36,40,44").
	let list: Option<Vec<u8>> = if notes.contains(',') {
		Some(
			notes
				.split(',')
				.map(|note| note_argument(note.trim()))
				.collect(),
		)
	} else {
		None
	};
	let notes: Vec<u8> = match list {
		Some(list) => list,
		None => match notes.split_once(':') {
			Some((a, b)) => {
				let first: u8 = a.parse().expect("first note");
				let last: u8 = b.parse().expect("last note");
				(first..=last).collect()
			}
			None => vec![note_argument(&notes)],
		},
	};
	let (first, last) = (
		*notes.first().expect("a note"),
		*notes.last().expect("a note"),
	);

	let sample_rate = 44_100.0;
	std::fs::create_dir_all(&directory).expect("create the output directory");

	for note in notes {
		let mut engine = StringEngine::new(sample_rate);
		engine.set_preset(preset);
		{
			let params = engine.params_mut();
			params.strings = strings;
			params.core = core;
			params.excitation = excitation;
		}
		engine.note_on(note, 0.8);

		let frames = (seconds * sample_rate) as usize;
		let mut samples = Vec::with_capacity(frames);
		for _ in 0..frames {
			samples.push(engine.next_sample());
		}

		let hertz = note_to_frequency(note as f32);
		let path = Path::new(&directory).join(format!("{preset_name}_{note:03}_{hertz:.1}.wav"));
		write_wav(&path, &samples, sample_rate as u32).expect("write the wav");
	}

	println!("{}: notes {first}..{last} into {}", preset_name, directory);
}

/// Accepts a note as a number, or as a scientific pitch name such as C3.
fn note_argument(text: &str) -> u8 {
	let names = [
		"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
	];
	for octave in 0..9 {
		for (index, name) in names.iter().enumerate() {
			if text.eq_ignore_ascii_case(&format!("{name}{octave}")) {
				return (octave as u8 + 1) * 12 + index as u8;
			}
		}
	}
	text.parse().expect("a note number")
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
