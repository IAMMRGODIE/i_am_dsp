//! The first products with a web faceplate.
//!
//! Each one is the DSP, the metadata its parameters already carry, and one line
//! that names the window. Gain, Drive and Filter are the simple cases on purpose:
//! one or two controls and one picture, so the faceplate itself is what is being
//! tried out, not the effect.
//!
//! The ids end in .web, so a host can load these next to the released iced
//! plug-ins without confusing the two.
//!
//! ```text
//! cargo build -p i_am_effects_web
//! cargo bundle -p i_am_effects_web --no-vst3
//! cargo run -p i_am_effects_web --example dump_face -- target/web-preview gain
//! ```

mod allpass;
mod comb;
mod compressor;
mod disperser;
mod filter;
mod ir_reverb;
mod particle_wider;
mod pitch;
mod reverb;
mod schroeder;
mod shifter;
mod unstablizer;

use allpass::AllpassEffect;
use comb::CombEffect;
use compressor::CompressorEffect;
use disperser::Dispersion;
use ir_reverb::IrReverbEffect;
use particle_wider::ParticleWiderEffect;
use pitch::PitchEffect;
use reverb::ReverbEffect;
use schroeder::SchroederEffect;
use shifter::ShifterEffect;
use unstablizer::UnstablizerEffect;
pub use shifter::{SHIFTER_CURVES, SHIFTER_VISIBLE};
pub use pitch::PITCH_VISIBLE;
use filter::Multimode;
pub use filter::{FILTER_HEADER, FILTER_VISIBLE};
use i_am_dsp::prelude::{
	Chorus, Crusher, Delay, Distortion, Flanger, Gain as GainDsp, LrControl, Phaser,
	SineWave, StereoController,
};
use i_am_web_gui::Switch;
use i_am_plugin::export_clap;
use i_am_web_gui::web_plugin;

web_plugin!(
	GainWeb,
	GainDsp,
	|_| GainDsp::new(1.0),
	"iamdsp.effects.gain.web",
	"I Am Gain",
	(360.0, 230.0),
	"#35c77a"
);

/// The controls each distortion mode reads; the page hides the rest.
pub const DISTORTION_VISIBLE: &[(i32, &[&str])] = &[
	(0, &["threshold", "gain"]),
	(1, &["alpha", "gain"]),
	(2, &["k", "gain"]),
	(3, &["a", "p", "gain"]),
	(4, &["bits", "gain"]),
	(5, &["target_sample_rate", "gain"]),
];

web_plugin!(
	DistortionWeb,
	Distortion<2>,
	Distortion::<2>::new,
	"iamdsp.effects.distortion.web",
	"I Am Distortion",
	(420.0, 310.0),
	"#ff9a3d",
	mode = "mode",
	visible = DISTORTION_VISIBLE
);

web_plugin!(
	FilterWeb,
	Multimode,
	|rate| Multimode::new(rate, 1_000.0, 0.707),
	"iamdsp.effects.filter.web",
	"I Am Filter",
	(520.0, 330.0),
	"#4c8dff",
	mode = "mode",
	visible = FILTER_VISIBLE,
	header = FILTER_HEADER
);

web_plugin!(
	CombWeb,
	CombEffect,
	CombEffect::new,
	"iamdsp.effects.comb.web",
	"I Am Comb",
	(360.0, 230.0),
	"#f0b35e"
);

web_plugin!(
	AllpassWeb,
	AllpassEffect,
	AllpassEffect::new,
	"iamdsp.effects.allpass.web",
	"I Am Allpass",
	(360.0, 230.0),
	"#b18cff"
);

web_plugin!(
	WidthWeb,
	StereoController,
	|_| StereoController::new(),
	"iamdsp.effects.width.web",
	"I Am Width",
	(420.0, 340.0),
	"#ff6fae"
);

web_plugin!(
	PitchWeb,
	PitchEffect,
	PitchEffect::new,
	"iamdsp.effects.pitch.web",
	"I Am Pitch",
	(360.0, 230.0),
	"#c8e04a",
	mode = "algorithm",
	visible = PITCH_VISIBLE,
	header = &["window_size"]
);

web_plugin!(
	ShifterWeb,
	ShifterEffect,
	ShifterEffect::new,
	"iamdsp.effects.shifter.web",
	"I Am Shifter",
	(360.0, 230.0),
	"#7aa2ff",
	mode = "algorithm",
	visible = SHIFTER_VISIBLE,
	curves = SHIFTER_CURVES,
	header = &["window_size"]
);

web_plugin!(
	DisperserWeb,
	Dispersion,
	Dispersion::new,
	"iamdsp.effects.disperser.web",
	"I Am Disperser",
	(380.0, 250.0),
	"#2fc6c0"
);

web_plugin!(
	BalanceWeb,
	LrControl,
	|_| LrControl::default(),
	"iamdsp.effects.balance.web",
	"I Am Balance",
	(360.0, 230.0),
	"#d96aff"
);

/// The delay's sync switch lives in the header as a dot; its time control trades
/// between milliseconds and beats in the same slot.
pub const DELAY_HEADER: &[&str] = &["tempo.sync"];
/// Free time when sync is off, musical time when it is on.
pub const DELAY_SWITCHES: &[Switch] = &[
	Switch::new("tempo.sync", &["tempo.beats"], &["delay_time"]),
	// The pan only means something while ping-pong is on.
	Switch::new("pingpong", &["pingpong_pan"], &[]),
];

/// The chorus sync switch: free hertz or beats.
pub const CHORUS_HEADER: &[&str] = &["tempo.sync"];
pub const CHORUS_SWITCHES: &[Switch] = &[Switch::new(
	"tempo.sync",
	&["tempo.beats"],
	&["lfo_frequency"],
)];
/// The flanger uses the same pair.
pub const FLANGER_HEADER: &[&str] = &["tempo.sync"];
pub const FLANGER_SWITCHES: &[Switch] = &[Switch::new(
	"tempo.sync",
	&["tempo.beats"],
	&["lfo_frequency"],
)];
/// The phaser names its free rate lfo_freq.
pub const PHASER_HEADER: &[&str] = &["tempo.sync"];
pub const PHASER_SWITCHES: &[Switch] = &[Switch::new(
	"tempo.sync",
	&["tempo.beats"],
	&["lfo_freq"],
)];

/// Main input plus a sidechain input. The compressor reads the sidechain as
/// other[1]; port 0 is always the main input.
pub const COMPRESSOR_PORTS: &[i_am_plugin::AudioPort] = &[
	i_am_plugin::AudioPort { name: "Main" },
	i_am_plugin::AudioPort { name: "Sidechain" },
];

/// The compressor's controls by envelope. The gain controls repeat in every
/// mode; the detection controls that belong to the selected detector ride with
/// it.
pub const COMPRESSOR_VISIBLE: &[(i32, &[&str])] = &[
	(
		0,
		&[
			"threshold", "ratio", "attack", "release", "lookahead", "makeup_db", "dry_gain",
			"wet_gain", "sidechain", "env_attack", "env_release", "env_gain",
		],
	),
	(
		1,
		&[
			"threshold", "ratio", "attack", "release", "lookahead", "makeup_db", "dry_gain",
			"wet_gain", "sidechain", "env_cutoff", "env_gain",
		],
	),
	(
		2,
		&[
			"threshold", "ratio", "attack", "release", "lookahead", "makeup_db", "dry_gain",
			"wet_gain", "sidechain", "env_gain",
		],
	),
	(
		3,
		&[
			"threshold", "ratio", "attack", "release", "lookahead", "makeup_db", "dry_gain",
			"wet_gain", "sidechain", "env_gain",
		],
	),
];

web_plugin!(
	CompressorWeb,
	CompressorEffect,
	CompressorEffect::new,
	"iamdsp.effects.compressor.web",
	"I Am Compressor",
	(700.0, 380.0),
	"#ff7a6b",
	mode = "enveloper",
	visible = COMPRESSOR_VISIBLE,
	ports = COMPRESSOR_PORTS
);

web_plugin!(
	ReverbWeb,
	ReverbEffect,
	ReverbEffect::new,
	"iamdsp.effects.reverb.web",
	"I Am Reverb",
	(700.0, 340.0),
	"#6c8cff"
);

web_plugin!(
	UnstablizerWeb,
	UnstablizerEffect,
	UnstablizerEffect::new,
	"iamdsp.effects.unstablizer.web",
	"I Am Unstablizer",
	(700.0, 340.0),
	"#ff8f6b",
	mode = "window_size",
	midi = &[i_am_plugin::MidiPort { name: "Notes" }]
);

web_plugin!(
	ParticleWiderWeb,
	ParticleWiderEffect,
	ParticleWiderEffect::new,
	"iamdsp.effects.particle_wider.web",
	"I Am Particle Wider",
	(700.0, 340.0),
	"#67d7a8",
	mode = "fft_size"
);

web_plugin!(
	SchroederWeb,
	SchroederEffect,
	SchroederEffect::new,
	"iamdsp.effects.schroeder.web",
	"I Am Schroeder",
	(700.0, 340.0),
	"#e0b36a"
);

web_plugin!(
	IrReverbWeb,
	IrReverbEffect,
	IrReverbEffect::new,
	"iamdsp.effects.ir_reverb.web",
	"I Am IR Reverb",
	(700.0, 340.0),
	"#7bd3ff"
);

web_plugin!(
	ChorusWeb,
	Chorus<SineWave, 2>,
	|rate| Chorus::<SineWave, 2>::new(1 << 14, 4, rate, SineWave),
	"iamdsp.effects.chorus.web",
	"I Am Chorus",
	(560.0, 300.0),
	"#5fe0b0",
	header = CHORUS_HEADER,
	switches = CHORUS_SWITCHES
);

web_plugin!(
	FlangerWeb,
	Flanger<SineWave, 2>,
	|rate| Flanger::<SineWave, 2>::new(1 << 18, rate, SineWave),
	"iamdsp.effects.flanger.web",
	"I Am Flanger",
	(560.0, 300.0),
	"#6bb8ff",
	header = FLANGER_HEADER,
	switches = FLANGER_SWITCHES
);

web_plugin!(
	PhaserWeb,
	Phaser<SineWave, 2>,
	|rate| Phaser::<SineWave, 2>::new(SineWave, 6, rate),
	"iamdsp.effects.phaser.web",
	"I Am Phaser",
	(620.0, 300.0),
	"#b48cff",
	header = PHASER_HEADER,
	switches = PHASER_SWITCHES
);

web_plugin!(
	DelayWeb,
	Delay<(), 2>,
	|rate| Delay::<(), 2>::new((), 1 << 16, 250.0, rate),
	"iamdsp.effects.delay.web",
	"I Am Delay",
	(560.0, 300.0),
	"#e8c35a",
	header = DELAY_HEADER,
	switches = DELAY_SWITCHES
);

web_plugin!(
	CrusherWeb,
	Crusher<2>,
	Crusher::<2>::new,
	"iamdsp.effects.crusher.web",
	"I Am Crusher Web",
	(400.0, 640.0),
	"#8a6aff"
);

export_clap!(
	GainWeb,
	DistortionWeb,
	FilterWeb,
	CombWeb,
	AllpassWeb,
	WidthWeb,
	PitchWeb,
	ShifterWeb,
	DisperserWeb,
	BalanceWeb,
	DelayWeb,
	CompressorWeb,
	ReverbWeb,
	SchroederWeb,
	ParticleWiderWeb,
	UnstablizerWeb,
	IrReverbWeb,
	ChorusWeb,
	FlangerWeb,
	PhaserWeb,
	CrusherWeb
);

#[cfg(test)]
mod tests {
	use super::*;
	use std::sync::atomic::Ordering;

	use i_am_dsp::prelude::SetValue;
	use i_am_plugin::Plugin;
	use i_am_web_gui::Faceplate;

	fn gain_face() -> String {
		i_am_web_gui::face_json(
			&GainWeb::new(),
			&Faceplate::new("I Am Gain", "#35c77a", (360.0, 230.0)),
		)
	}

	/// The page builds its controls from this, so every control has to be here with
	/// the layout the DSP declared.
	#[test]
	fn the_gain_faceplate_is_generated_from_metadata() {
		let face = gain_face();

		assert!(face.contains("\"id\":\"gain\""), "{face}");
		assert!(face.contains("\"label\":\"Gain\""), "{face}");
		assert!(face.contains("\"unit\":\"percent\""), "{face}");
		assert!(face.contains("\"log\":true"), "{face}");
		assert!(face.contains("\"size\":\"large\""), "{face}");
		assert!(face.contains("\"hero\":\"waveform\""), "{face}");
		assert!(face.contains("\"size\":[360,230]"), "{face}");
	}

	/// Distortion's picture is its transfer curve, and Filter's is its response.
	#[test]
	fn each_effect_names_its_own_picture() {
		let distortion = i_am_web_gui::face_json(
			&DistortionWeb::new(),
			&Faceplate::new("I Am Distortion", "#ff9a3d", (420.0, 310.0))
				.with_mode("mode")
				.with_visible(DISTORTION_VISIBLE),
		);
		let filter = i_am_web_gui::face_json(
			&FilterWeb::new(),
			&Faceplate::new("I Am Filter", "#4c8dff", (520.0, 330.0)),
		);

		assert!(distortion.contains("\"hero\":\"transfer\""), "{distortion}");
		assert!(distortion.contains("\"id\":\"threshold\""), "{distortion}");
		assert!(filter.contains("\"hero\":\"spectrum\""), "{filter}");
		assert!(filter.contains("\"unit\":\"hz\""), "{filter}");
	}

	/// A drag posts a number and it has to reach the map the audio thread reads.
	#[test]
	fn a_web_command_writes_the_parameter() {
		let plugin = GainWeb::new();

		assert!(i_am_web_gui::apply_command(&plugin, "gain=0.5"));

		let map = plugin.param_map();
		assert_eq!(
			map.get("gain").unwrap().load(Ordering::Relaxed),
			SetValue::Float(0.5)
		);
	}

	/// Nothing a page posts may panic or write past a bad parse.
	#[test]
	fn a_malformed_command_is_ignored() {
		let plugin = GainWeb::new();

		assert!(!i_am_web_gui::apply_command(&plugin, "nonsense"));
		assert!(!i_am_web_gui::apply_command(&plugin, "gain=not-a-number"));
		assert!(!i_am_web_gui::apply_command(&plugin, "missing.parameter=1.0"));

		// And the good value is still the one it started with.
		assert_eq!(
			plugin.param_map().get("gain").unwrap().load(Ordering::Relaxed),
			SetValue::Float(1.0)
		);
	}

	/// The page polls this; a control missing from it freezes at its default.
	#[test]
	fn the_state_has_every_control() {
		let plugin = FilterWeb::new();
		let state = i_am_web_gui::state_json(&plugin);

		assert!(state.contains("\"cutoff\""), "{state}");
		assert!(state.contains("\"q\""), "{state}");
	}

	/// A product with controls has an editor, and it is the web one.
	#[test]
	fn each_product_has_a_web_editor() {
		assert!(GainWeb::new().editor().is_some());
		assert!(DistortionWeb::new().editor().is_some());
		assert!(FilterWeb::new().editor().is_some());
		assert!(CombWeb::new().editor().is_some());
		assert!(AllpassWeb::new().editor().is_some());
		assert!(WidthWeb::new().editor().is_some());
		assert!(PitchWeb::new().editor().is_some());
		assert!(ShifterWeb::new().editor().is_some());
		assert!(DisperserWeb::new().editor().is_some());
		assert!(BalanceWeb::new().editor().is_some());
		assert!(DelayWeb::new().editor().is_some());
		assert!(CompressorWeb::new().editor().is_some());
		assert!(ReverbWeb::new().editor().is_some());
		assert!(SchroederWeb::new().editor().is_some());
		assert!(ParticleWiderWeb::new().editor().is_some());
		assert!(UnstablizerWeb::new().editor().is_some());
		assert!(IrReverbWeb::new().editor().is_some());
		assert!(ChorusWeb::new().editor().is_some());
		assert!(FlangerWeb::new().editor().is_some());
		assert!(PhaserWeb::new().editor().is_some());
		assert!(CrusherWeb::new().editor().is_some());
	}

	/// The crusher has three named sections, and the Drive one shows the scope.
	#[test]
	fn the_crusher_has_its_sections_and_a_scope() {
		let face = i_am_web_gui::face_json(
			&CrusherWeb::new(),
			&Faceplate::new("I Am Crusher Web", "#8a6aff", (400.0, 640.0)),
		);

		assert!(face.contains("\"hero\":\"waveform\""), "{face}");
		assert!(face.contains("\"id\":\"amount\""), "{face}");
		assert!(face.contains("\"name\":\"Drive\""), "{face}");
		assert!(face.contains("\"name\":\"Amplitude\""), "{face}");
		assert!(face.contains("\"name\":\"Gain\""), "{face}");
	}

	/// Two more modules, each with the picture its section asked for.
	#[test]
	fn width_and_pitch_have_their_own_pictures() {
		let width = i_am_web_gui::face_json(
			&WidthWeb::new(),
			&Faceplate::new("I Am Width", "#ff6fae", (360.0, 230.0)),
		);
		let pitch = i_am_web_gui::face_json(
			&PitchWeb::new(),
			&Faceplate::new("I Am Pitch", "#c8e04a", (360.0, 230.0)),
		);

		assert!(width.contains("\"hero\":\"stereo\""), "{width}");
		assert!(width.contains("\"id\":\"mid_gain\""), "{width}");
		assert!(width.contains("\"label\":\"Mid\""), "{width}");
		assert!(pitch.contains("\"hero\":\"pitch\""), "{pitch}");
		assert!(pitch.contains("\"id\":\"pitch_shift_factor\""), "{pitch}");
		assert!(pitch.contains("\"label\":\"Shift\""), "{pitch}");
	}

	/// The filter has three algorithms behind one set of controls, and the page
	/// is told to draw the switch in its header.
	#[test]
	fn the_filter_has_a_mode_switch() {
		let plugin = FilterWeb::new();
		let face = i_am_web_gui::face_json(
			&plugin,
			&Faceplate::new("I Am Filter", "#4c8dff", (520.0, 330.0))
				.with_mode("mode")
				.with_visible(FILTER_VISIBLE)
				.with_header(FILTER_HEADER),
		);

		assert!(face.contains("\"mode\":\"mode\""), "{face}");
		assert!(face.contains("\"header\":[\"stages\"]"), "{face}");
		assert!(face.contains("\"id\":\"mode\""), "{face}");
		assert!(
			face.contains("\"choices\":[\"Lowpass\",\"Highpass\",\"Bandpass\",\"Notch\",\"Peak\",\"Low Shelf\",\"High Shelf\"]"),
			"{face}"
		);
		assert!(face.contains("\"id\":\"gain_db\""), "{face}");
		assert!(face.contains("\"id\":\"slope\""), "{face}");
		assert!(face.contains("\"id\":\"stages\""), "{face}");
		assert!(
			face.contains("\"choices\":[\"12 dB/oct\",\"24 dB/oct\",\"36 dB/oct\",\"48 dB/oct\"]"),
			"{face}"
		);
		assert!(face.contains("[0,[\"cutoff\",\"q\",\"stages\"]]"), "{face}");
		assert!(face.contains("[5,[\"cutoff\",\"gain_db\",\"slope\"]]"), "{face}");
	}

	/// The mode is a normal parameter: it is in the map the host saves and in the
	/// state the page reads back after a reload.
	#[test]
	fn the_mode_is_saved_with_the_other_parameters() {
		let plugin = FilterWeb::new();
		assert!(i_am_web_gui::apply_command(&plugin, "mode=5"));

		assert_eq!(
			plugin.param_map().get("mode").unwrap().load(Ordering::Relaxed),
			SetValue::Int(5)
		);
		assert!(i_am_web_gui::state_json(&plugin).contains("\"mode\":5"));
	}

	/// The delay's sync dot is a parameter in the header, and the time knob is the
	/// free milliseconds or the same length in beats.
	#[test]
	fn the_delay_swaps_free_and_synced_time() {
		let face = i_am_web_gui::face_json(
			&DelayWeb::new(),
			&Faceplate::new("I Am Delay", "#e8c35a", (560.0, 450.0))
				.with_header(DELAY_HEADER)
				.with_switches(DELAY_SWITCHES),
		);

		assert!(face.contains("\"hero\":\"taps\""), "{face}");
		assert!(face.contains("\"id\":\"delay_time\""), "{face}");
		assert!(face.contains("\"id\":\"tempo.sync\""), "{face}");
		assert!(face.contains("\"id\":\"tempo.beats\""), "{face}");
		assert!(face.contains("\"header\":[\"tempo.sync\"]"), "{face}");
		assert!(
			face.contains("{\"control\":\"tempo.sync\",\"on\":[\"tempo.beats\"],\"off\":[\"delay_time\"]}"),
			"{face}"
		);
		assert!(
			face.contains("{\"control\":\"pingpong\",\"on\":[\"pingpong_pan\"],\"off\":[]}"),
			"{face}"
		);
		assert!(face.contains("\"id\":\"pingpong_pan\""), "{face}");
		assert!(face.contains("\"id\":\"dry_gain\""), "{face}");
		assert!(face.contains("\"id\":\"wet_gain\""), "{face}");
		assert!(face.contains("\"name\":\"Delay\""), "{face}");
	}

	/// The comb/allpass filters get their own small faces.
	#[test]
	fn the_comb_and_allpass_have_simple_faces() {
		let comb = i_am_web_gui::face_json(
			&CombWeb::new(),
			&Faceplate::new("I Am Comb", "#f0b35e", (360.0, 230.0)),
		);
		let allpass = i_am_web_gui::face_json(
			&AllpassWeb::new(),
			&Faceplate::new("I Am Allpass", "#b18cff", (360.0, 230.0)),
		);

		for face in [comb, allpass] {
			assert!(face.contains("\"hero\":\"waveform\""), "{face}");
			assert!(face.contains("\"id\":\"delay_ms\""), "{face}");
			assert!(face.contains("\"id\":\"feedback\""), "{face}");
		}
	}

	/// The Schroeder reverb is the classic comb/allpass construction.
	#[test]
	fn the_schroeder_has_a_tail_and_the_classic_controls() {
		let face = i_am_web_gui::face_json(
			&SchroederWeb::new(),
			&Faceplate::new("I Am Schroeder", "#e0b36a", (700.0, 340.0)),
		);

		assert!(face.contains("\"hero\":\"tail\""), "{face}");
		assert!(face.contains("\"id\":\"size\""), "{face}");
		assert!(face.contains("\"id\":\"decay_time\""), "{face}");
		assert!(face.contains("\"id\":\"damping\""), "{face}");
		assert!(face.contains("\"id\":\"dry_gain\""), "{face}");
		assert!(face.contains("\"id\":\"wet_gain\""), "{face}");
	}

	/// The particle wider has a small surface over the spectral mapper.
	#[test]
	fn the_particle_wider_has_a_particle_hero() {
		let face = i_am_web_gui::face_json(
			&ParticleWiderWeb::new(),
			&Faceplate::new("I Am Particle Wider", "#67d7a8", (700.0, 340.0))
				.with_mode("fft_size"),
		);

		assert!(face.contains("\"hero\":\"particles\""), "{face}");
		assert!(face.contains("\"id\":\"depth\""), "{face}");
		assert!(face.contains("\"id\":\"low_frequency\""), "{face}");
		assert!(face.contains("\"id\":\"high_frequency\""), "{face}");
		assert!(face.contains("\"id\":\"mid_gain\""), "{face}");
		assert!(face.contains("\"id\":\"side_gain\""), "{face}");
		assert!(face.contains("\"id\":\"allow_negative\""), "{face}");
		assert!(face.contains("\"id\":\"window_factor\""), "{face}");
		assert!(face.contains("\"id\":\"fft_size\""), "{face}");
		assert!(face.contains("\"mode\":\"fft_size\""), "{face}");
		assert!(face.contains("\"choices\":[\"512\",\"1024\",\"2048\",\"4096\"]"), "{face}");
	}

	/// Pitch and Shifter both offer the time-domain engine and the phase vocoder.
	#[test]
	fn the_pitch_and_shifter_have_algorithm_modes() {
		let pitch = i_am_web_gui::face_json(
			&PitchWeb::new(),
			&Faceplate::new("I Am Pitch", "#c8e04a", (360.0, 230.0))
				.with_mode("algorithm")
				.with_visible(PITCH_VISIBLE)
				.with_header(&["window_size"]),
		);
		let shifter = i_am_web_gui::face_json(
			&ShifterWeb::new(),
			&Faceplate::new("I Am Shifter", "#7aa2ff", (360.0, 230.0))
				.with_mode("algorithm")
				.with_visible(SHIFTER_VISIBLE)
				.with_header(&["window_size"]),
		);

		for face in [pitch, shifter] {
			assert!(face.contains("\"mode\":\"algorithm\""), "{face}");
			assert!(face.contains("\"header\":[\"window_size\"]"), "{face}");
			assert!(face.contains("\"id\":\"window_size\""), "{face}");
			assert!(face.contains("[1,[\"pitch_shift_factor\",\"window_size\"]]") || face.contains("[1,[\"shift_freq\",\"window_size\"]]"), "{face}");
		}
	}

	/// The Unstablizer uses the spectra hero and a header window size.
	#[test]
	fn the_unstablizer_has_before_and_after_spectra() {
		let face = i_am_web_gui::face_json(
			&UnstablizerWeb::new(),
			&Faceplate::new("I Am Unstablizer", "#ff8f6b", (700.0, 340.0))
				.with_mode("window_size"),
		);

		assert!(face.contains("\"hero\":\"spectra\""), "{face}");
		assert!(face.contains("\"id\":\"reference_pitch\""), "{face}");
		assert!(face.contains("\"id\":\"transpose\""), "{face}");
		assert!(face.contains("\"id\":\"shift\""), "{face}");
		assert!(face.contains("\"id\":\"anchor\""), "{face}");
		assert!(face.contains("\"id\":\"window_size\""), "{face}");
		assert!(face.contains("\"id\":\"follow_notes\""), "{face}");
		assert!(face.contains("\"mode\":\"window_size\""), "{face}");
		assert!(face.contains("\"choices\":[\"512\",\"1024\",\"2048\",\"4096\"]"), "{face}");
		assert_eq!(
			<UnstablizerWeb as i_am_plugin::Plugin>::INPUT_MIDI_PORTS.len(),
			1,
			"the note follower needs a MIDI input"
		);
	}

	/// The IR reverb exposes the room, the T60, and a wet-only convolver.
	#[test]
	fn the_ir_reverb_has_a_tail_and_a_room() {
		let face = i_am_web_gui::face_json(
			&IrReverbWeb::new(),
			&Faceplate::new("I Am IR Reverb", "#7bd3ff", (700.0, 340.0)),
		);

		assert!(face.contains("\"hero\":\"tail\""), "{face}");
		assert!(face.contains("\"id\":\"size\""), "{face}");
		assert!(face.contains("\"id\":\"sound_velocity\""), "{face}");
		assert!(face.contains("\"id\":\"decay_time\""), "{face}");
		assert!(face.contains("\"id\":\"dry_gain\""), "{face}");
		assert!(face.contains("\"id\":\"wet_gain\""), "{face}");
		assert!(face.contains("\"name\":\"IR Reverb\""), "{face}");
	}

	/// The compressor's envelope choice is a header drop-down, and its sidechain
	/// is a second input port.
	#[test]
	fn the_compressor_has_an_enveloper_menu_and_a_sidechain() {
		let face = i_am_web_gui::face_json(
			&CompressorWeb::new(),
			&Faceplate::new("I Am Compressor", "#ff7a6b", (700.0, 380.0))
				.with_mode("enveloper")
				.with_visible(COMPRESSOR_VISIBLE),
		);

		assert!(face.contains("\"hero\":\"compressor\""), "{face}");
		assert!(face.contains("\"id\":\"threshold\""), "{face}");
		assert!(face.contains("\"id\":\"sidechain\""), "{face}");
		assert!(face.contains("\"mode\":\"enveloper\""), "{face}");
		assert!(face.contains("\"id\":\"env_attack\""), "{face}");
		assert!(face.contains("[0,[\"threshold\",\"ratio\""), "{face}");
		assert!(
			face.contains("\"choices\":[\"Peak\",\"RMS\",\"TKeo\",\"Hilbert\"]"),
			"{face}"
		);
		assert!(face.contains("\"name\":\"Compressor\""), "{face}");
		assert_eq!(
			<CompressorWeb as i_am_plugin::Plugin>::INPUT_PORTS.len(),
			2,
			"main plus sidechain"
		);
	}

	/// The reverb's tail hero reads Decay, Damping and Pre-delay.
	#[test]
	fn the_reverb_has_a_tail_hero() {
		let face = i_am_web_gui::face_json(
			&ReverbWeb::new(),
			&Faceplate::new("I Am Reverb", "#6c8cff", (700.0, 340.0)),
		);

		assert!(face.contains("\"hero\":\"tail\""), "{face}");
		assert!(face.contains("\"id\":\"decay_time\""), "{face}");
		assert!(face.contains("\"id\":\"pre_delay\""), "{face}");
		assert!(face.contains("\"id\":\"wet_gain\""), "{face}");
		assert!(face.contains("\"name\":\"Reverb\""), "{face}");
	}

	/// Chorus, flanger and phaser share the sync dot; only the free parameter's id
	/// changes, and phaser calls it lfo_freq.
	#[test]
	fn the_b_group_uses_the_same_sync_pair() {
		let chorus = i_am_web_gui::face_json(
			&ChorusWeb::new(),
			&Faceplate::new("I Am Chorus", "#5fe0b0", (560.0, 300.0))
				.with_header(CHORUS_HEADER)
				.with_switches(CHORUS_SWITCHES),
		);
		let flanger = i_am_web_gui::face_json(
			&FlangerWeb::new(),
			&Faceplate::new("I Am Flanger", "#6bb8ff", (560.0, 300.0))
				.with_header(FLANGER_HEADER)
				.with_switches(FLANGER_SWITCHES),
		);
		let phaser = i_am_web_gui::face_json(
			&PhaserWeb::new(),
			&Faceplate::new("I Am Phaser", "#b48cff", (560.0, 300.0))
				.with_header(PHASER_HEADER)
				.with_switches(PHASER_SWITCHES),
		);

		assert!(chorus.contains("\"hero\":\"lfo\""), "{chorus}");
		assert!(chorus.contains("\"id\":\"lfo_frequency\""), "{chorus}");
		assert!(chorus.contains("\"name\":\"Chorus\""), "{chorus}");
		assert!(
			chorus.contains("\"switches\":[{\"control\":\"tempo.sync\",\"on\":[\"tempo.beats\"],\"off\":[\"lfo_frequency\"]}]"),
			"{chorus}"
		);

		assert!(flanger.contains("\"name\":\"Flanger\""), "{flanger}");
		assert!(flanger.contains("\"id\":\"feed_back_gain\""), "{flanger}");

		assert!(phaser.contains("\"name\":\"Phaser\""), "{phaser}");
		assert!(phaser.contains("\"id\":\"lfo_freq\""), "{phaser}");
		assert!(phaser.contains("\"id\":\"dry_gain\""), "{phaser}");
		assert!(phaser.contains("\"id\":\"wet_gain\""), "{phaser}");
		assert!(phaser.contains("\"off\":[\"lfo_freq\"]"), "{phaser}");
	}

	/// Distortion has six algorithms, and each mode lists the controls it reads.
	#[test]
	fn the_distortion_has_modes_and_visible_controls() {
		let face = i_am_web_gui::face_json(
			&DistortionWeb::new(),
			&Faceplate::new("I Am Distortion", "#ff9a3d", (420.0, 310.0))
				.with_mode("mode")
				.with_visible(DISTORTION_VISIBLE),
		);

		assert!(face.contains("\"mode\":\"mode\""), "{face}");
		assert!(
			face.contains("\"choices\":[\"Hard\",\"Soft\",\"Overdrive\",\"Saturator\",\"Bitcrush\",\"Downsample\"]"),
			"{face}"
		);
		assert!(face.contains("\"id\":\"threshold\""), "{face}");
		assert!(face.contains("\"id\":\"bits\""), "{face}");
		assert!(face.contains("\"visible\":[[0,[\"threshold\",\"gain\"]]"), "{face}");
		assert!(face.contains("[4,[\"bits\",\"gain\"]]"), "{face}");
	}

	/// The A group: shifter, disperser and balance, each with its picture.
	#[test]
	fn the_a_group_has_its_pictures() {
		let shifter = i_am_web_gui::face_json(
			&ShifterWeb::new(),
			&Faceplate::new("I Am Shifter", "#7aa2ff", (360.0, 230.0))
				.with_curves(SHIFTER_CURVES),
		);
		let disperser = i_am_web_gui::face_json(
			&DisperserWeb::new(),
			&Faceplate::new("I Am Disperser", "#2fc6c0", (380.0, 250.0)),
		);
		let balance = i_am_web_gui::face_json(
			&BalanceWeb::new(),
			&Faceplate::new("I Am Balance", "#d96aff", (360.0, 230.0)),
		);

		assert!(shifter.contains("\"hero\":\"spectra\""), "{shifter}");
		assert!(shifter.contains("\"id\":\"shift_freq\""), "{shifter}");
		assert!(shifter.contains("\"curves\":{\"shift_freq\":\"atan\"}"), "{shifter}");
		assert!(disperser.contains("\"hero\":\"phase\""), "{disperser}");
		assert!(disperser.contains("\"id\":\"amount\""), "{disperser}");
		assert!(disperser.contains("\"id\":\"cutoff\""), "{disperser}");
		assert!(balance.contains("\"hero\":\"balance\""), "{balance}");
		assert!(balance.contains("\"label\":\"Left\""), "{balance}");
	}

	/// Crusher Web asked for resizing with an aspect ratio; these lay themselves
	/// out, so they ask for resizing without one.
	#[test]
	fn the_windows_are_resizable() {
		for options in [
			GainWeb::window_options(),
			DistortionWeb::window_options(),
			FilterWeb::window_options(),
			CombWeb::window_options(),
			AllpassWeb::window_options(),
			WidthWeb::window_options(),
			PitchWeb::window_options(),
			ShifterWeb::window_options(),
			DisperserWeb::window_options(),
			BalanceWeb::window_options(),
			DelayWeb::window_options(),
			CompressorWeb::window_options(),
			ReverbWeb::window_options(),
			SchroederWeb::window_options(),
			ParticleWiderWeb::window_options(),
			UnstablizerWeb::window_options(),
			IrReverbWeb::window_options(),
			ChorusWeb::window_options(),
			FlangerWeb::window_options(),
			PhaserWeb::window_options(),
			CrusherWeb::window_options(),
		] {
			let hints = options.resize_hints.expect("the web windows are resizable");

			assert!(hints.can_resize_horizontally);
			assert!(hints.can_resize_vertically);
			assert!(matches!(
				hints.strategy,
				i_am_plugin::AspectRatioStrategy::Disregard
			));
		}
	}

	/// Bypass is not a parameter: the processor has to skip its effect and leave
	/// the frames it was handed alone.
	#[test]
	fn bypass_passes_the_input_untouched() {
		use i_am_dsp::Processor;

		let mut plugin = GainWeb::new();
		// A host, and the editor, write the map; set_parameter on the product is
		// for a caller that owns it and would be overwritten by the next sync.
		plugin
			.param_map()
			.set("gain", SetValue::Float(0.5), Ordering::SeqCst);

		let mut context: Box<dyn i_am_dsp::ProcessContext> = Box::new(());
		let mut bypassed = [0.25_f32, -0.5];
		plugin.set_bypassed(true);
		plugin.process(&mut bypassed, &[], &mut context);
		assert_eq!(bypassed, [0.25, -0.5]);

		let mut active = [0.25_f32, -0.5];
		plugin.set_bypassed(false);
		plugin.process(&mut active, &[], &mut context);
		assert_eq!(active, [0.125, -0.25]);
	}
}