//! Fit one recorded note into the table a Modal instrument plays.
//!
//!     cargo run -p i_am_dsp --release --example fit_sample -- sample.wav out.bin [note]
//!
//! The note is optional: without it the fit reads the fundamental from the
//! spectrum and snaps it to the nearest key.

use std::path::Path;
use std::time::Instant;

fn main() {
	let arguments: Vec<String> = std::env::args().collect();
	let input = arguments.get(1).map(String::as_str).unwrap_or("sample.wav");
	let output = arguments
		.get(2)
		.map(String::as_str)
		.unwrap_or("target/wav/fitted.bin");
	let note = arguments.get(3).and_then(|value| value.parse::<u8>().ok());

	let wav = std::fs::read(input).unwrap_or_else(|error| panic!("read {input}: {error}"));
	let started = Instant::now();
	let fitted = i_am_dsp::prelude::fit_sample(&wav, note)
		.unwrap_or_else(|error| panic!("fit {input}: {error}"));
	println!(
		"note {} at {:.2} Hz, {} partials, {:.2} s",
		fitted.note,
		fitted.frequency,
		fitted.modes,
		started.elapsed().as_secs_f32()
	);

	if let Some(parent) = Path::new(output).parent() {
		std::fs::create_dir_all(parent).ok();
	}
	std::fs::write(output, &fitted.bytes).unwrap_or_else(|error| panic!("write {output}: {error}"));
	println!("wrote {output} ({} bytes)", fitted.bytes.len());
}
