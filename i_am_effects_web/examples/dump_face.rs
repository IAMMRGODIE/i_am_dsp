//! Writes the faceplate and state JSON for one product, for a browser preview.
//!
//! ```text
//! cargo run -p i_am_effects_web --example dump_face -- target/preview gain
//! cargo run -p i_am_effects_web --example dump_face -- target/preview drive
//! cargo run -p i_am_effects_web --example dump_face -- target/preview filter
//! ```
//!
//! The page fetches /face and /state, so a static server whose root also holds the
//! kit's index.html, kit.css and kit.js is enough to look at a real faceplate
//! without opening a DAW.

use std::path::Path;

use i_am_effects_web::{CrusherWeb, DriveWeb, FilterWeb, GainWeb, PitchWeb, WidthWeb};
use i_am_plugin::Plugin;
use i_am_web_gui::Faceplate;

fn main() {
	let arguments: Vec<String> = std::env::args().collect();
	let directory = arguments
		.get(1)
		.cloned()
		.unwrap_or_else(|| "target/web-preview".to_string());
	let product = arguments
		.get(2)
		.cloned()
		.unwrap_or_else(|| "gain".to_string());

	let (face, state) = match product.as_str() {
		"gain" => {
			let plugin = GainWeb::new();
			(
				i_am_web_gui::face_json(
					&plugin,
					&Faceplate::new("I Am Gain", "#35c77a", (360.0, 230.0)),
				),
				i_am_web_gui::state_json(&plugin),
			)
		},
		"drive" => {
			let plugin = DriveWeb::new();
			(
				i_am_web_gui::face_json(
					&plugin,
					&Faceplate::new("I Am Drive", "#ff9a3d", (420.0, 310.0)).with_mode("mode"),
				),
				i_am_web_gui::state_json(&plugin),
			)
		},
		"filter" => {
			let plugin = FilterWeb::new();
			(
				i_am_web_gui::face_json(
					&plugin,
					&Faceplate::new("I Am Filter", "#4c8dff", (520.0, 330.0)).with_mode("mode"),
				),
				i_am_web_gui::state_json(&plugin),
			)
		},
		"width" => {
			let plugin = WidthWeb::new();
			(
				i_am_web_gui::face_json(
					&plugin,
					&Faceplate::new("I Am Width", "#ff6fae", (360.0, 230.0)),
				),
				i_am_web_gui::state_json(&plugin),
			)
		},
		"pitch" => {
			let plugin = PitchWeb::new();
			(
				i_am_web_gui::face_json(
					&plugin,
					&Faceplate::new("I Am Pitch", "#c8e04a", (360.0, 230.0)),
				),
				i_am_web_gui::state_json(&plugin),
			)
		},
		"crusher" => {
			let plugin = CrusherWeb::new();
			(
				i_am_web_gui::face_json(
					&plugin,
					&Faceplate::new("I Am Crusher Web", "#8a6aff", (400.0, 640.0)),
				),
				i_am_web_gui::state_json(&plugin),
			)
		},
		other => {
			eprintln!(
				"unknown product {other}; expected gain, drive, filter, width, pitch or crusher"
			);
			std::process::exit(2);
		},
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
