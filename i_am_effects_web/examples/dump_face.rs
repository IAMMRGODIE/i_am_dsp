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

use i_am_effects_web::{
	AllpassWeb, BalanceWeb, ChorusWeb, CombWeb, CompressorWeb, CrusherWeb, DelayWeb,
	ConvolverWeb, DisperserWeb, DistortionWeb, FilterWeb, FlangerWeb, GainWeb, IrReverbWeb, PhaserWeb,
	ScriptDistortionWeb, ScriptMapperWeb,
	ParticleWiderWeb, PitchWeb, ReverbWeb, SchroederWeb, ShifterWeb, UnstablizerWeb, WidthWeb,
	CHORUS_HEADER, CHORUS_SWITCHES, COMPRESSOR_VISIBLE, DELAY_HEADER, DELAY_SWITCHES,
	DISTORTION_VISIBLE, FLANGER_HEADER, FLANGER_SWITCHES, PHASER_HEADER, PHASER_SWITCHES,
	PITCH_VISIBLE, SHIFTER_CURVES, SHIFTER_VISIBLE,
};
use i_am_effects_web::{FILTER_HEADER, FILTER_VISIBLE};
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
		"distortion" => {
			let plugin = DistortionWeb::new();
			(
				i_am_web_gui::face_json(
					&plugin,
					&Faceplate::new("I Am Distortion", "#ff9a3d", (420.0, 310.0))
						.with_mode("mode")
						.with_visible(DISTORTION_VISIBLE),
				),
				i_am_web_gui::state_json(&plugin),
			)
		},
		"filter" => {
			let plugin = FilterWeb::new();
			(
				i_am_web_gui::face_json(
					&plugin,
					&Faceplate::new("I Am Filter", "#4c8dff", (520.0, 330.0))
						.with_mode("mode")
						.with_visible(FILTER_VISIBLE)
						.with_header(FILTER_HEADER),
				),
				i_am_web_gui::state_json(&plugin),
			)
		},
		"comb" => {
			let plugin = CombWeb::new();
			(
				i_am_web_gui::face_json(
					&plugin,
					&Faceplate::new("I Am Comb", "#f0b35e", (360.0, 230.0)),
				),
				i_am_web_gui::state_json(&plugin),
			)
		},
		"allpass" => {
			let plugin = AllpassWeb::new();
			(
				i_am_web_gui::face_json(
					&plugin,
					&Faceplate::new("I Am Allpass", "#b18cff", (360.0, 230.0)),
				),
				i_am_web_gui::state_json(&plugin),
			)
		},
		"unstablizer" => {
			let plugin = UnstablizerWeb::new();
			(
				i_am_web_gui::face_json(
					&plugin,
					&Faceplate::new("I Am Unstablizer", "#ff8f6b", (700.0, 340.0))
						.with_mode("window_size"),
				),
				i_am_web_gui::state_json(&plugin),
			)
		},
		"particle_wider" => {
			let plugin = ParticleWiderWeb::new();
			(
				i_am_web_gui::face_json(
					&plugin,
					&Faceplate::new("I Am Particle Wider", "#67d7a8", (700.0, 340.0))
						.with_mode("fft_size"),
				),
				i_am_web_gui::state_json(&plugin),
			)
		},
		"schroeder" => {
			let plugin = SchroederWeb::new();
			(
				i_am_web_gui::face_json(
					&plugin,
					&Faceplate::new("I Am Schroeder", "#e0b36a", (700.0, 340.0)),
				),
				i_am_web_gui::state_json(&plugin),
			)
		},
		"width" => {
			let plugin = WidthWeb::new();
			(
				i_am_web_gui::face_json(
					&plugin,
					&Faceplate::new("I Am Width", "#ff6fae", (420.0, 340.0)),
				),
				i_am_web_gui::state_json(&plugin),
			)
		},
		"pitch" => {
			let plugin = PitchWeb::new();
			(
				i_am_web_gui::face_json(
					&plugin,
					&Faceplate::new("I Am Pitch", "#c8e04a", (360.0, 230.0))
						.with_mode("algorithm")
						.with_visible(PITCH_VISIBLE)
						.with_header(&["window_size"]),
				),
				i_am_web_gui::state_json(&plugin),
			)
		},
		"shifter" => {
			let plugin = ShifterWeb::new();
			(
				i_am_web_gui::face_json(
					&plugin,
					&Faceplate::new("I Am Shifter", "#7aa2ff", (360.0, 230.0))
						.with_mode("algorithm")
						.with_visible(SHIFTER_VISIBLE)
						.with_curves(SHIFTER_CURVES)
						.with_header(&["window_size"]),
				),
				i_am_web_gui::state_json(&plugin),
			)
		},
		"disperser" => {
			let plugin = DisperserWeb::new();
			(
				i_am_web_gui::face_json(
					&plugin,
					&Faceplate::new("I Am Disperser", "#2fc6c0", (380.0, 250.0)),
				),
				i_am_web_gui::state_json(&plugin),
			)
		},
		"balance" => {
			let plugin = BalanceWeb::new();
			(
				i_am_web_gui::face_json(
					&plugin,
					&Faceplate::new("I Am Balance", "#d96aff", (360.0, 230.0)),
				),
				i_am_web_gui::state_json(&plugin),
			)
		},
		"ir_reverb" => {
			let plugin = IrReverbWeb::new();
			(
				i_am_web_gui::face_json(
					&plugin,
					&Faceplate::new("I Am IR Reverb", "#7bd3ff", (700.0, 340.0)),
				),
				i_am_web_gui::state_json(&plugin),
			)
		},
		"reverb" => {
			let plugin = ReverbWeb::new();
			(
				i_am_web_gui::face_json(
					&plugin,
					&Faceplate::new("I Am Reverb", "#6c8cff", (700.0, 340.0)),
				),
				i_am_web_gui::state_json(&plugin),
			)
		},
		"compressor" => {
			let plugin = CompressorWeb::new();
			(
				i_am_web_gui::face_json(
					&plugin,
					&Faceplate::new("I Am Compressor", "#ff7a6b", (700.0, 380.0))
						.with_mode("enveloper")
						.with_visible(COMPRESSOR_VISIBLE),
				),
				i_am_web_gui::state_json(&plugin),
			)
		},
		"chorus" => {
			let plugin = ChorusWeb::new();
			(
				i_am_web_gui::face_json(
					&plugin,
					&Faceplate::new("I Am Chorus", "#5fe0b0", (560.0, 300.0))
						.with_header(CHORUS_HEADER)
						.with_switches(CHORUS_SWITCHES),
				),
				i_am_web_gui::state_json(&plugin),
			)
		},
		"flanger" => {
			let plugin = FlangerWeb::new();
			(
				i_am_web_gui::face_json(
					&plugin,
					&Faceplate::new("I Am Flanger", "#6bb8ff", (560.0, 300.0))
						.with_header(FLANGER_HEADER)
						.with_switches(FLANGER_SWITCHES),
				),
				i_am_web_gui::state_json(&plugin),
			)
		},
		"phaser" => {
			let plugin = PhaserWeb::new();
			(
				i_am_web_gui::face_json(
					&plugin,
					&Faceplate::new("I Am Phaser", "#b48cff", (620.0, 300.0))
						.with_header(PHASER_HEADER)
						.with_switches(PHASER_SWITCHES),
				),
				i_am_web_gui::state_json(&plugin),
			)
		},
		"delay" => {
			let plugin = DelayWeb::new();
			(
				i_am_web_gui::face_json(
					&plugin,
					&Faceplate::new("I Am Delay", "#e8c35a", (560.0, 300.0))
						.with_header(DELAY_HEADER)
						.with_switches(DELAY_SWITCHES),
				),
				i_am_web_gui::state_json(&plugin),
			)
		},
		"script_mapper" => {
			let plugin = ScriptMapperWeb::new();
			(
				i_am_web_gui::face_json(
					&plugin,
					&Faceplate::new("I Am Script Mapper", "#8ad3ff", (700.0, 560.0))
						.with_script(),
				),
				i_am_web_gui::state_json(&plugin),
			)
		},
		"script_distortion" => {
			let plugin = ScriptDistortionWeb::new();
			(
				i_am_web_gui::face_json(
					&plugin,
					&Faceplate::new("I Am Script Distortion", "#ff7a6b", (700.0, 560.0))
						.with_script(),
				),
				i_am_web_gui::state_json(&plugin),
			)
		},
		"convolver" => {
			let plugin = ConvolverWeb::new();
			(
				i_am_web_gui::face_json(
					&plugin,
					&Faceplate::new("I Am Convolver", "#b08cff", (520.0, 330.0)),
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
				"unknown product {other}; expected gain, distortion, filter, comb, allpass, width, pitch, shifter, disperser, balance, compressor, reverb, schroeder, particle_wider, unstablizer, ir_reverb, chorus, flanger, phaser, delay or crusher"
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
