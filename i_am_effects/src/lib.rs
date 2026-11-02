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
//! The DSP and the registration of every product live here; the CLAP/VST3 entry
//! points live in `i_am_effects_bundle`, so that a plug-in can link these effects
//! without colliding on `clap_entry`.
//!
//! ```text
//! cargo build -p i_am_effects_bundle
//! cargo bundle -p i_am_effects_bundle --no-vst3
//! cargo run -p i_am_effects --example dump_face -- target/web-preview gain
//! ```

mod allpass;
pub mod catalog;
mod comb;
mod compressor;
mod convolver;
mod disperser;
mod ir_reverb;
mod ott;
mod particle_wider;
pub mod pitch;
mod schroeder;
mod script;
mod script_distortion;
mod script_mapper;
pub mod reverb;
mod shifter;
mod stft_vocoder;
mod unstablizer;
mod vocoder;

use allpass::AllpassEffect;
use comb::CombEffect;
pub use compressor::CompressorEffect;
pub use convolver::{ConvolverWeb, IrData, IrSlot, decode_ir};
pub use script_distortion::ScriptDistortionWeb;
pub use script_mapper::ScriptMapperWeb;
pub use ott::OttWeb;
pub use vocoder::VocoderWeb;
pub use stft_vocoder::StftVocoderWeb;
use disperser::Dispersion;
use ir_reverb::IrReverbEffect;
use particle_wider::ParticleWiderEffect;
use pitch::PitchEffect;
use reverb::ReverbEffect;
use schroeder::SchroederEffect;
use shifter::ShifterEffect;
use unstablizer::UnstablizerEffect;
pub use shifter::SHIFTER_VISIBLE;
pub use pitch::PITCH_VISIBLE;

/// The controls each autotune engine reads; the note mask lives in its own
/// section and is always visible.
pub const AUTOTUNE_VISIBLE: &[(i32, &[&str])] = &[
	(0, &["strength", "speed", "notes"]),
	(1, &["strength", "speed", "window_size", "notes"]),
	(2, &["strength", "speed", "notes"]),
];
/// The parameters the page draws in the header, beside the mode.
pub const FILTER_HEADER: &[&str] = &["stages"];

/// The controls each mode reads; the page hides the rest.
pub const FILTER_VISIBLE: &[(i32, &[&str])] = &[
	(0, &["cutoff", "q", "stages"]),
	(1, &["cutoff", "q", "stages"]),
	(2, &["cutoff", "q", "stages"]),
	(3, &["cutoff", "q", "stages"]),
	(4, &["cutoff", "gain_db"]),
	(5, &["cutoff", "gain_db", "slope"]),
	(6, &["cutoff", "gain_db", "slope"]),
];

use i_am_dsp::prelude::{
	AutoTune, Chorus, Crusher, Delay, Distortion, Expander, Flanger, FormantFilter,
	Gain as GainDsp, Gate, Limiter, LrControl, Multimode, ParametricEq, Phaser, Resonator,
	SpectralRetuneEffect, SineWave, StereoController,
	StereoRouter, TapeStopEffect, TransientShaper,
};
use i_am_web_gui::Switch;
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
	AutoTuneWeb,
	AutoTune<2>,
	AutoTune::<2>::new,
	"iamdsp.effects.autotune.web",
	"I Am Autotune",
	(700.0, 420.0),
	"#ff6fae",
	mode = "algorithm",
	visible = AUTOTUNE_VISIBLE,
	header = &["window_size"],
	midi = &[i_am_plugin::MidiPort { name: "Notes" }],
	telemetry = |tuner: &AutoTune<2>, scope: &i_am_web_gui::Scope| {
		scope.set_allowed_notes(tuner.allowed_mask() as i32)
	}
);

web_plugin!(
	SpectralRetuneWeb,
	SpectralRetuneEffect,
	SpectralRetuneEffect::new,
	"iamdsp.effects.spectral_retune.web",
	"I Am Spectral Retune",
	(700.0, 400.0),
	"#5fd0c8",
	header = &["window_size"],
	midi = &[i_am_plugin::MidiPort { name: "Notes" }],
	telemetry = |retune: &SpectralRetuneEffect, scope: &i_am_web_gui::Scope| {
		scope.set_allowed_notes(retune.allowed_mask() as i32)
	}
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

/// Main input plus a sidechain input, for the gate and the expander.
pub const SIDECHAIN_PORTS: &[i_am_plugin::AudioPort] = &[
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
	"#6c8cff",
	clear = true
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
	"#e0b36a",
	clear = true
);

web_plugin!(
	IrReverbWeb,
	IrReverbEffect,
	IrReverbEffect::new,
	"iamdsp.effects.ir_reverb.web",
	"I Am IR Reverb",
	(700.0, 340.0),
	"#7bd3ff",
	clear = true
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
	switches = DELAY_SWITCHES,
	clear = true
);



/// The EQ's six bands: the page shows the selected band's controls and the
/// output, and the hero draws all six.
pub const EQ_VISIBLE: &[(i32, &[&str])] = &[
	(0, &["b1_on", "b1_type", "b1_freq", "b1_gain", "b1_q", "b1_slope", "output"]),
	(1, &["b2_on", "b2_type", "b2_freq", "b2_gain", "b2_q", "b2_slope", "output"]),
	(2, &["b3_on", "b3_type", "b3_freq", "b3_gain", "b3_q", "b3_slope", "output"]),
	(3, &["b4_on", "b4_type", "b4_freq", "b4_gain", "b4_q", "b4_slope", "output"]),
	(4, &["b5_on", "b5_type", "b5_freq", "b5_gain", "b5_q", "b5_slope", "output"]),
	(5, &["b6_on", "b6_type", "b6_freq", "b6_gain", "b6_q", "b6_slope", "output"]),
];

web_plugin!(
	EqWeb,
	ParametricEq<2>,
	ParametricEq::<2>::new,
	"iamdsp.effects.eq.web",
	"I Am EQ",
	(760.0, 420.0),
	"#8a7bff",
	mode = "band",
	visible = EQ_VISIBLE
);

web_plugin!(
	TapeStopWeb,
	TapeStopEffect<2>,
	TapeStopEffect::<2>::new,
	"iamdsp.effects.tape_stop.web",
	"I Am Tape Stop",
	(420.0, 260.0),
	"#7ad1e0"
);

web_plugin!(
	GateWeb,
	Gate<2>,
	Gate::<2>::new,
	"iamdsp.effects.gate.web",
	"I Am Gate",
	(700.0, 340.0),
	"#6fd3a3",
	ports = SIDECHAIN_PORTS
);

web_plugin!(
	ExpanderWeb,
	Expander<2>,
	Expander::<2>::new,
	"iamdsp.effects.expander.web",
	"I Am Expander",
	(700.0, 340.0),
	"#79b8ff",
	ports = SIDECHAIN_PORTS
);

web_plugin!(
	LimiterWeb,
	Limiter<2>,
	Limiter::<2>::new,
	"iamdsp.effects.limiter.web",
	"I Am Limiter",
	(700.0, 320.0),
	"#ffc45a"
);

web_plugin!(
	ResonatorWeb,
	Resonator<2>,
	Resonator::<2>::new,
	"iamdsp.effects.resonator.web",
	"I Am Resonator",
	(460.0, 300.0),
	"#b98cff",
	midi = &[i_am_plugin::MidiPort { name: "Notes" }]
);

web_plugin!(
	FormantFilterWeb,
	FormantFilter<2>,
	FormantFilter::<2>::new,
	"iamdsp.effects.formant_filter.web",
	"I Am Formant Filter",
	(520.0, 340.0),
	"#ff9f6e"
);

web_plugin!(
	TransientWeb,
	TransientShaper<2>,
	TransientShaper::<2>::new,
	"iamdsp.effects.transient_shaper.web",
	"I Am Transient Shaper",
	(520.0, 320.0),
	"#ff7ad9"
);

web_plugin!(
	RouterWeb,
	StereoRouter,
	StereoRouter::new,
	"iamdsp.effects.router.web",
	"I Am Router",
	(520.0, 320.0),
	"#8ad3ff"
);

web_plugin!(
	CrusherWeb,
	Crusher<2>,
	Crusher::<2>::new,
	"iamdsp.effects.crusher.web",
	"I Am Crusher",
	(400.0, 640.0),
	"#8a6aff"
);

#[cfg(test)]
mod tests {
	use super::*;
	use std::sync::atomic::Ordering;

	use i_am_dsp::prelude::SetValue;
	use i_am_plugin::Plugin;
	use i_am_web_gui::Faceplate;

	struct NoteContext {
		events: Vec<i_am_dsp::NoteEvent>,
		infos: i_am_dsp::ProcessInfos,
	}

	impl i_am_dsp::ProcessContext for NoteContext {
		fn infos(&self) -> &i_am_dsp::ProcessInfos {
			&self.infos
		}

		fn next_event(&mut self) -> Option<i_am_dsp::NoteEvent> {
			if self.events.is_empty() { None } else { Some(self.events.remove(0)) }
		}

		fn send_event(&mut self, event: i_am_dsp::NoteEvent) {
			self.events.push(event);
		}

		fn events(&self) -> &[i_am_dsp::NoteEvent] {
			&self.events
		}

		fn clear_events(&mut self) {
			self.events.clear();
		}
	}

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
		assert!(AutoTuneWeb::new().editor().is_some());
		assert!(SpectralRetuneWeb::new().editor().is_some());
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
		assert!(EqWeb::new().editor().is_some());
		assert!(TapeStopWeb::new().editor().is_some());
		assert!(GateWeb::new().editor().is_some());
		assert!(ExpanderWeb::new().editor().is_some());
		assert!(LimiterWeb::new().editor().is_some());
		assert!(ResonatorWeb::new().editor().is_some());
		assert!(FormantFilterWeb::new().editor().is_some());
		assert!(TransientWeb::new().editor().is_some());
		assert!(RouterWeb::new().editor().is_some());
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

		assert!(
			pitch.contains("\"choices\":[\"WSOLA\",\"Phase Vocoder\",\"PSOLA\"]"),
			"{pitch}"
		);

		for face in [pitch, shifter] {
			assert!(face.contains("\"mode\":\"algorithm\""), "{face}");
			assert!(face.contains("\"header\":[\"window_size\"]"), "{face}");
			assert!(face.contains("\"id\":\"window_size\""), "{face}");
			assert!(face.contains("[1,[\"pitch_shift_factor\",\"window_size\"]]") || face.contains("[1,[\"shift_freq\",\"window_size\"]]"), "{face}");
		}
	}

	/// Autotune has three engines, a note mask and a MIDI note port.
	#[test]
	fn the_autotune_has_a_note_picker_and_three_engines() {
		let face = i_am_web_gui::face_json(
			&AutoTuneWeb::new(),
			&Faceplate::new("I Am Autotune", "#ff6fae", (700.0, 420.0))
				.with_mode("algorithm")
				.with_visible(AUTOTUNE_VISIBLE)
				.with_header(&["window_size"]),
		);

		assert!(face.contains("\"hero\":\"autotune\""), "{face}");
		assert!(face.contains("\"mode\":\"algorithm\""), "{face}");
		assert!(face.contains("\"header\":[\"window_size\"]"), "{face}");
		for id in ["strength", "speed", "window_size", "notes"] {
			assert!(face.contains(&format!("\"id\":\"{id}\"")), "{face}");
		}
		assert!(face.contains("\"widget\":\"notes\""), "{face}");
		assert!(
			face.contains("\"choices\":[\"WSOLA\",\"Phase Vocoder\",\"PSOLA\"]"),
			"{face}"
		);
		assert!(
			face.contains("[1,[\"strength\",\"speed\",\"window_size\",\"notes\"]]"),
			"{face}"
		);
		assert_eq!(
			<AutoTuneWeb as i_am_plugin::Plugin>::INPUT_MIDI_PORTS.len(),
			1,
			"autotune needs a note port"
		);
	}

	/// The autotune publishes the live mask for its hero; MIDI is the override.
	#[test]
	fn the_autotune_publishes_its_live_mask() {
		let mut plugin = AutoTuneWeb::new();
		let mut context: Box<dyn i_am_dsp::ProcessContext> = Box::new(NoteContext {
			events: vec![
				i_am_dsp::NoteEvent::NoteOn { channel: 0, note: 60, velocity: 1.0 },
				i_am_dsp::NoteEvent::NoteOn { channel: 0, note: 64, velocity: 1.0 },
			],
			infos: i_am_dsp::ProcessInfos::default(),
		});
		let mut frame = [0.0_f32, 0.0];

		i_am_dsp::Processor::process(&mut plugin, &mut frame, &[], &mut context);

		assert_eq!(plugin.scope.allowed_notes(), (1 << 0) | (1 << 4));
	}

	/// Spectral Retune exposes the band, the strength/speed and the same note mask.
	#[test]
	fn the_spectral_retune_has_its_band_and_notes() {
		let face = i_am_web_gui::face_json(
			&SpectralRetuneWeb::new(),
			&Faceplate::new("I Am Spectral Retune", "#5fd0c8", (700.0, 400.0))
				.with_header(&["window_size"]),
		);

		assert!(face.contains("\"hero\":\"autotune\""), "{face}");
		for id in ["strength", "speed", "low_frequency", "high_frequency", "mix", "notes", "window_size"] {
			assert!(face.contains(&format!("\"id\":\"{id}\"")), "{face}");
		}
		assert!(face.contains("\"widget\":\"notes\""), "{face}");
		assert!(face.contains("\"header\":[\"window_size\"]"), "{face}");
		assert_eq!(
			<SpectralRetuneWeb as i_am_plugin::Plugin>::INPUT_MIDI_PORTS.len(),
			1,
			"spectral retune needs a note port"
		);
	}

	/// Spectral Retune publishes its live MIDI/UI mask through the same scope
	/// telemetry as Autotune.
	#[test]
	fn the_spectral_retune_publishes_its_live_mask() {
		let mut plugin = SpectralRetuneWeb::new();
		let mut context: Box<dyn i_am_dsp::ProcessContext> = Box::new(NoteContext {
			events: vec![
				i_am_dsp::NoteEvent::NoteOn { channel: 0, note: 60, velocity: 1.0 },
				i_am_dsp::NoteEvent::NoteOn { channel: 0, note: 64, velocity: 1.0 },
			],
			infos: i_am_dsp::ProcessInfos::default(),
		});
		let mut frame = [0.0_f32, 0.0];

		i_am_dsp::Processor::process(&mut plugin, &mut frame, &[], &mut context);

		assert_eq!(plugin.scope.allowed_notes(), (1 << 0) | (1 << 4));
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

	/// The convolver's picture is the impulse response it has loaded.
	#[test]
	fn the_convolver_has_an_impulse_hero_and_dry_wet_gains() {
		let face = i_am_web_gui::face_json(
			&ConvolverWeb::new(),
			&Faceplate::new("I Am Convolver", "#b08cff", (520.0, 330.0)),
		);

		assert!(face.contains("\"hero\":\"impulse\""), "{face}");
		assert!(face.contains("\"id\":\"dry_gain\""), "{face}");
		assert!(face.contains("\"id\":\"wet_gain\""), "{face}");
		assert!(!face.contains("\"id\":\"ir\""), "the response is state, not a control: {face}");
	}

	/// The scripted distortion is a curve with a script panel and four controls.
	#[test]
	fn the_script_distortion_has_a_curve_and_a_script_panel() {
		let face = i_am_web_gui::face_json(
			&ScriptDistortionWeb::new(),
			&Faceplate::new("I Am Script Distortion", "#ff7a6b", (700.0, 560.0)).with_script(),
		);

		assert!(face.contains("\"script\":true"), "{face}");
		assert!(face.contains("\"hero\":\"curve\""), "{face}");
		assert!(face.contains("\"id\":\"in_gain\""), "{face}");
		assert!(face.contains("\"id\":\"out_gain\""), "{face}");
		assert!(face.contains("\"id\":\"mix\""), "{face}");

		for id in ["p1", "p2", "p3", "p4"] {
			assert!(face.contains(&format!("\"id\":\"{id}\"")), "{face}");
		}

		assert!(face.contains("\"id\":\"realtime\""), "{face}");
	}

	/// The scripted mapper has a mapping hero and the same script panel.
	#[test]
	fn the_script_mapper_has_a_mapping_hero_and_a_script_panel() {
		let face = i_am_web_gui::face_json(
			&ScriptMapperWeb::new(),
			&Faceplate::new("I Am Script Mapper", "#8ad3ff", (700.0, 560.0))
				.with_script()
				.with_header(&["window_size", "realtime"]),
		);

		assert!(face.contains("\"script\":true"), "{face}");
		assert!(face.contains("\"hero\":\"mapping\""), "{face}");
		assert!(face.contains("\"id\":\"mix\""), "{face}");
		assert!(
			face.contains("\"header\":[\"window_size\",\"realtime\"]"),
			"{face}"
		);
		assert!(
			face.contains("\"choices\":[\"512\",\"1024\",\"2048\",\"4096\"]"),
			"{face}"
		);

		for id in ["p1", "p2", "p3", "p4"] {
			assert!(face.contains(&format!("\"id\":\"{id}\"")), "{face}");
		}
	}

	/// The vocoder draws a band meter pair and declares a sidechain input.
	#[test]
	fn the_vocoder_has_band_meters_and_a_sidechain() {
		let face = i_am_web_gui::face_json(
			&VocoderWeb::new(),
			&Faceplate::new("I Am Vocoder", "#e0a45a", (700.0, 380.0)),
		);

		assert!(face.contains("\"hero\":\"vocodex\""), "{face}");

		for id in [
			"bands",
			"bandwidth",
			"attack",
			"release",
			"mix",
			"output",
			"sidechain",
		] {
			assert!(face.contains(&format!("\"id\":\"{id}\"")), "{face}");
		}

		assert!(face.contains("\"sidechain\""), "{face}");
		assert!(
			<VocoderWeb as i_am_plugin::Plugin>::INPUT_PORTS.len() == 2,
			"main plus sidechain"
		);
	}

	/// The STFT vocoder has a window choice, a sidechain input and a spectrum view.
	#[test]
	fn the_stft_vocoder_has_a_window_choice_and_a_spectrum_hero() {
		let face = i_am_web_gui::face_json(
			&StftVocoderWeb::new(),
			&Faceplate::new("I Am STFT Vocoder", "#7bb8ff", (700.0, 340.0))
				.with_header(&["window_size"]),
		);

		assert!(face.contains("\"hero\":\"spectra\""), "{face}");
		assert!(face.contains("\"header\":[\"window_size\"]"), "{face}");

		for id in ["window_size", "mix", "output", "sidechain"] {
			assert!(face.contains(&format!("\"id\":\"{id}\"")), "{face}");
		}

		assert!(
			face.contains("\"choices\":[\"512\",\"1024\",\"2048\",\"4096\"]"),
			"{face}"
		);
		assert!(
			<StftVocoderWeb as i_am_plugin::Plugin>::INPUT_PORTS.len() == 2,
			"main plus sidechain"
		);
	}



	/// The EQ edits one band at a time and draws all six in its hero.
	#[test]
	fn the_eq_has_six_bands_and_a_graphical_hero() {
		let face = i_am_web_gui::face_json(
			&EqWeb::new(),
			&Faceplate::new("I Am EQ", "#8a7bff", (760.0, 420.0))
				.with_mode("band")
				.with_visible(EQ_VISIBLE),
		);

		assert!(face.contains("\"hero\":\"eq\""), "{face}");
		assert!(face.contains("\"mode\":\"band\""), "{face}");
		assert!(
			face.contains("\"choices\":[\"Bell\",\"Low Shelf\",\"High Shelf\",\"Low Cut\",\"High Cut\",\"Notch\"]"),
			"{face}"
		);

		for id in [
			"b1_on", "b1_type", "b1_freq", "b1_gain", "b1_q", "b1_slope", "b6_on", "b6_q",
			"b6_slope", "output",
		] {
			assert!(face.contains(&format!("\"id\":\"{id}\"")), "{face}");
		}

		assert!(face.contains("\"visible\":[[0,[\"b1_on\""), "{face}");
		assert!(face.contains("[5,[\"b6_on\""), "{face}");
	}

	/// The new utility products: a tape stop, the three downward dynamics, a
	/// transient shaper and the stereo router.
	#[test]
	fn the_utility_products_have_their_controls() {
		let tape_stop = i_am_web_gui::face_json(
			&TapeStopWeb::new(),
			&Faceplate::new("I Am Tape Stop", "#7ad1e0", (420.0, 260.0)),
		);
		let gate = i_am_web_gui::face_json(
			&GateWeb::new(),
			&Faceplate::new("I Am Gate", "#6fd3a3", (700.0, 340.0)),
		);
		let expander = i_am_web_gui::face_json(
			&ExpanderWeb::new(),
			&Faceplate::new("I Am Expander", "#79b8ff", (700.0, 340.0)),
		);
		let limiter = i_am_web_gui::face_json(
			&LimiterWeb::new(),
			&Faceplate::new("I Am Limiter", "#ffc45a", (700.0, 320.0)),
		);
		let transient = i_am_web_gui::face_json(
			&TransientWeb::new(),
			&Faceplate::new("I Am Transient Shaper", "#ff7ad9", (520.0, 320.0)),
		);
		let router = i_am_web_gui::face_json(
			&RouterWeb::new(),
			&Faceplate::new("I Am Router", "#8ad3ff", (520.0, 320.0)),
		);

		assert!(tape_stop.contains("\"hero\":\"waveform\""), "{tape_stop}");
		assert!(tape_stop.contains("\"id\":\"stop\""), "{tape_stop}");
		assert!(tape_stop.contains("\"id\":\"time\""), "{tape_stop}");

		assert!(gate.contains("\"hero\":\"gate\""), "{gate}");
		for id in ["threshold", "range", "attack", "hold", "release", "hysteresis", "sidechain"] {
			assert!(gate.contains(&format!("\"id\":\"{id}\"")), "{gate}");
		}
		assert_eq!(<GateWeb as i_am_plugin::Plugin>::INPUT_PORTS.len(), 2);

		assert!(expander.contains("\"hero\":\"expander\""), "{expander}");
		assert!(expander.contains("\"id\":\"ratio\""), "{expander}");
		assert_eq!(<ExpanderWeb as i_am_plugin::Plugin>::INPUT_PORTS.len(), 2);

		assert!(limiter.contains("\"hero\":\"limiter\""), "{limiter}");
		for id in ["ceiling", "release", "lookahead", "output", "true_peak"] {
			assert!(limiter.contains(&format!("\"id\":\"{id}\"")), "{limiter}");
		}
		assert!(
			limiter.contains("\"widget\":\"toggle\""),
			"true peak is a switch"
		);

		assert!(transient.contains("\"hero\":\"waveform\""), "{transient}");
		assert!(transient.contains("\"id\":\"attack\""), "{transient}");
		assert!(transient.contains("\"id\":\"sustain\""), "{transient}");

		assert!(router.contains("\"hero\":\"stereo\""), "{router}");
		for id in ["l_to_l", "l_to_r", "r_to_l", "r_to_r"] {
			assert!(router.contains(&format!("\"id\":\"{id}\"")), "{router}");
		}
	}

	/// The resonator follows MIDI notes and the formant shifter exposes a vowel
	/// and a formant shift.
	#[test]
	fn the_resonator_and_formant_filter_have_their_controls() {
		let resonator = i_am_web_gui::face_json(
			&ResonatorWeb::new(),
			&Faceplate::new("I Am Resonator", "#b98cff", (460.0, 300.0)),
		);
		let formant = i_am_web_gui::face_json(
			&FormantFilterWeb::new(),
			&Faceplate::new("I Am Formant Filter", "#ff9f6e", (520.0, 340.0)),
		);

		assert!(resonator.contains("\"hero\":\"resonator\""), "{resonator}");
		for id in ["note", "decay", "damping", "saw", "mix"] {
			assert!(resonator.contains(&format!("\"id\":\"{id}\"")), "{resonator}");
		}
		assert_eq!(
			<ResonatorWeb as i_am_plugin::Plugin>::INPUT_MIDI_PORTS.len(),
			1,
			"the resonator needs a note port"
		);

		assert!(formant.contains("\"hero\":\"formant\""), "{formant}");
		for id in ["vowel", "shift", "resonance", "mix"] {
			assert!(formant.contains(&format!("\"id\":\"{id}\"")), "{formant}");
		}
		assert!(formant.contains("\"choices\":[\"A\",\"E\",\"I\",\"O\",\"U\"]"), "{formant}");
	}

	/// The OTT is a three-band meter and a second section for the bands.
	#[test]
	fn the_ott_has_band_meters_and_a_bands_section() {
		let face = i_am_web_gui::face_json(
			&OttWeb::new(),
			&Faceplate::new("I Am OTT", "#ffb454", (700.0, 580.0)),
		);

		assert!(face.contains("\"hero\":\"ott\""), "{face}");

		for id in [
			"input_gain",
			"output_gain",
			"depth",
			"time",
			"upward",
			"downward",
			"band1_threshold",
			"band1_gain",
			"band2_threshold",
			"band2_gain",
			"band3_threshold",
			"band3_gain",
		] {
			assert!(face.contains(&format!("\"id\":\"{id}\"")), "{face}");
		}

		assert!(face.contains("\"name\":\"Bands\""), "{face}");
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
			&Faceplate::new("I Am Shifter", "#7aa2ff", (360.0, 230.0)),
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
		assert!(shifter.contains("\"log\":true"), "{shifter}");
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
			EqWeb::window_options(),
			TapeStopWeb::window_options(),
			GateWeb::window_options(),
			ExpanderWeb::window_options(),
			LimiterWeb::window_options(),
			TransientWeb::window_options(),
			RouterWeb::window_options(),
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