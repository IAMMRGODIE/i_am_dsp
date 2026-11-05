//! Render one kick to raw f32, so it can be listened to without a host.
//!
//! `cargo run -p i_am_dsp --example kick_render` writes `kick.raw`.

use std::io::Write;

use i_am_dsp::prelude::DrumInstrument;

fn main() -> std::io::Result<()> {
	const RATE: usize = 48_000;

	let mut kit = DrumInstrument::new(RATE);
	assert!(kit.start(36, 1.0), "the kick answers note 36");

	let mut bytes = Vec::with_capacity(RATE * 4 * 2);
	for _ in 0..RATE * 2 {
		bytes.extend_from_slice(&kit.next_sample().to_le_bytes());
	}

	let mut file = std::fs::File::create("kick.raw")?;
	file.write_all(&bytes)?;
	println!("wrote kick.raw ({} samples)", RATE * 2);

	Ok(())
}
