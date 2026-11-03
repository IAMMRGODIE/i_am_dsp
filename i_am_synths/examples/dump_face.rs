//! Writes the faceplate and state JSON for the string instrument, for a browser
//! preview.
//!
//! ```text
//! cargo run -p i_am_synths --example dump_face -- target/synth-preview
//! ```
//!
//! The page fetches /face and /state, so a static server whose root also holds
//! the kit's index.html, kit.css, card.css and their scripts is enough.

use std::path::Path;

use i_am_plugin::Plugin;
use i_am_synths::{StringWeb, faceplate};

fn main() {
	let directory = std::env::args()
		.nth(1)
		.unwrap_or_else(|| "target/synth-preview".to_string());

	let plugin = StringWeb::new();
	let face = i_am_web_gui::face_json(&plugin, &faceplate());
	let state = i_am_web_gui::state_json(&plugin);

	if let Err(error) = std::fs::create_dir_all(&directory) {
		eprintln!("cannot create {directory}: {error}");
		std::process::exit(1);
	}

	for (name, body) in [("face", face), ("state", state)] {
		let path = Path::new(&directory).join(name);

		if let Err(error) = std::fs::write(&path, body) {
			eprintln!("cannot write {}: {error}", path.display());
			std::process::exit(1);
		}
	}

	println!("strings: {directory}/face, {directory}/state");
}
