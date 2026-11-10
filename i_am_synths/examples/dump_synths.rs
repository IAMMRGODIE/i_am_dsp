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
use i_am_synths::{
	AdditiveWeb, CanvasWeb, DrummerWeb, FittedWeb, SamplerWeb, ScriptWeb, ShapesWeb, SpectrumWeb,
	StringWeb, additive_faceplate, canvas_faceplate, canvas_shape, drummer_faceplate, faceplate,
	fitted_faceplate,
	sampler_faceplate, script_faceplate, shapes_faceplate, spectrum_faceplate,
};

fn main() {
	let mut arguments = std::env::args().skip(1);
	let directory = arguments
		.next()
		.unwrap_or_else(|| "target/synth-preview".to_string());
	let product = arguments.next().unwrap_or_else(|| "strings".to_string());

	let (face, state) = match product.as_str() {
		"sampler" => {
			let plugin = SamplerWeb::new();
			(
				i_am_web_gui::face_json(&plugin, &sampler_faceplate()),
				i_am_web_gui::state_json(&plugin),
			)
		}
		"shapes" => {
			let plugin = ShapesWeb::new();
			(
				i_am_web_gui::face_json(&plugin, &shapes_faceplate()),
				i_am_web_gui::state_json(&plugin),
			)
		}
		"additive" => {
			let plugin = AdditiveWeb::new();
			(
				i_am_web_gui::face_json(&plugin, &additive_faceplate()),
				i_am_web_gui::state_json(&plugin),
			)
		}
		"spectrum" => {
			let plugin = SpectrumWeb::new();
			(
				i_am_web_gui::face_json(&plugin, &spectrum_faceplate()),
				i_am_web_gui::state_json(&plugin),
			)
		}
		"script" => {
			let plugin = ScriptWeb::new();
			(
				i_am_web_gui::face_json(&plugin, &script_faceplate()),
				i_am_web_gui::state_json(&plugin),
			)
		}
		"drummer" => {
			let plugin = DrummerWeb::new();
			(
				i_am_web_gui::face_json(&plugin, &drummer_faceplate()),
				i_am_web_gui::state_json(&plugin),
			)
		}
		"fitted" => {
			let plugin = FittedWeb::new();
			(
				i_am_web_gui::face_json(&plugin, &fitted_faceplate()),
				i_am_web_gui::state_json(&plugin),
			)
		}
		"canvas" => {
			let plugin = CanvasWeb::new();
			(
				i_am_web_gui::face_json(&plugin, &canvas_faceplate()),
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

	if product == "canvas" {
		let path = Path::new(&directory).join("canvas");
		let body = format!("{{\"shape\":\"{}\"}}", canvas_shape());

		if let Err(error) = std::fs::write(&path, body) {
			eprintln!("cannot write {}: {error}", path.display());
			std::process::exit(1);
		}
	}

	println!("{product}: {directory}/face, {directory}/state");
}
