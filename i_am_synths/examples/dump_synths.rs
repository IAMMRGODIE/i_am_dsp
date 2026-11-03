//! Writes the faceplate and state JSON of one instrument, for a browser preview.
//!
//! ```text
//! cargo run -p i_am_synths --example dump_face -- target/synth-preview strings
//! cargo run -p i_am_synths --example dump_face -- target/shapes-preview shapes
//! ```
//!
//! The page fetches /face and /state, so a static server whose root also holds
//! the kit's index.html, kit.css, card.css and their scripts is enough.

use std::path::Path;

use i_am_plugin::Plugin;
use i_am_synths::{ShapesWeb, StringWeb, faceplate, shapes_faceplate};

fn main() {
	let mut arguments = std::env::args().skip(1);
	let directory = arguments
		.next()
		.unwrap_or_else(|| "target/synth-preview".to_string());
	let product = arguments.next().unwrap_or_else(|| "strings".to_string());

	let (face, state) = match product.as_str() {
		"shapes" => {
			let plugin = ShapesWeb::new();
			(
				i_am_web_gui::face_json(&plugin, &shapes_faceplate()),
				i_am_web_gui::state_json(&plugin),
			)
		}
		_ => {
			let plugin = StringWeb::new();
			(
				i_am_web_gui::face_json(&plugin, &faceplate()),
				i_am_web_gui::state_json(&plugin),
			)
		}
	};

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

	println!("{product}: {directory}/face, {directory}/state");
}
