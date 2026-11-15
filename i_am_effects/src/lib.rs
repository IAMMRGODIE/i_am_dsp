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
mod filter_table;
mod history;
mod ir_reverb;
mod ott;
mod particle_wider;
pub mod pitch;
pub mod reverb;
mod schroeder;
mod script;

pub use script::ScriptSource;
mod script_distortion;
mod script_mapper;
mod shifter;
mod spectral_dynamics;
mod stft_vocoder;
mod unstablizer;
mod vocoder;

use allpass::AllpassEffect;
use comb::CombEffect;
pub use compressor::CompressorEffect;
pub use convolver::{ConvolverWeb, IrData, IrFades, IrSlot, decode_ir};
use disperser::Dispersion;
pub use filter_table::{FilterTableHandle, FilterTableWeb, faceplate as filter_table_faceplate};
pub use history::{HistoryHandle, HistoryWeb, faceplate as history_faceplate};
use i_am_dsp::prelude::OuterWider;
use ir_reverb::IrReverbEffect;
pub use ott::{OTT_PAGES, OttWeb};
use particle_wider::ParticleWiderEffect;
pub use pitch::PITCH_VISIBLE;
use pitch::PitchEffect;
use reverb::ReverbEffect;
use schroeder::SchroederEffect;
pub use script::ScriptWorker;
pub use script_distortion::{
	DistortionTable, ScriptDistortionEffect, ScriptDistortionWeb, distortion_worker,
};
pub use script_mapper::{MapperTable, ScriptMapperEffect, ScriptMapperWeb, mapper_worker};
pub use shifter::SHIFTER_VISIBLE;
use shifter::ShifterEffect;
pub use spectral_dynamics::{
	SPECTRAL_DYNAMICS_PAGES, SpectralDynamicsHandle, SpectralDynamicsWeb,
	faceplate as spectral_dynamics_faceplate,
};
pub use stft_vocoder::StftVocoderWeb;
use unstablizer::UnstablizerEffect;
pub use vocoder::VocoderWeb;

/// The controls each autotune engine reads; the note mask lives in its own
/// section and is always visible, and so is the header's MIDI switch.
pub const AUTOTUNE_VISIBLE: &[(i32, &[&str])] = &[
	(0, &["strength", "speed", "notes", "midi"]),
	(1, &["strength", "speed", "window_size", "notes", "midi"]),
	(2, &["strength", "speed", "notes", "midi"]),
];
/// The parameters the page draws in the header, beside the mode.
pub const FILTER_HEADER: &[&str] = &["stages"];

/// The compressor's two pages: the gain stage and the detector that drives it.
pub const COMPRESSOR_PAGES: &[i_am_web_gui::Page] = &[
	i_am_web_gui::Page::new("Compressor", &["Compressor"]),
	i_am_web_gui::Page::new("Detector", &["Detector"]),
];

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
	SineWave, SpectralRetuneEffect, StereoController, StereoRouter, TapeStopEffect,
	TransientShaper,
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
	OuterWiderWeb,
	OuterWider,
	OuterWider::new,
	"iamdsp.effects.outer_wider.web",
	"I Am Wider",
	(640.0, 360.0),
	"#ff6fae"
);

web_plugin!(
	PitchWeb,
	PitchEffect,
	PitchEffect::new,
	"iamdsp.effects.pitch.web",
	"I Am Pitch Shifter",
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
	header = &["window_size", "midi"],
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
	header = &["window_size", "midi"],
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
pub const PHASER_SWITCHES: &[Switch] =
	&[Switch::new("tempo.sync", &["tempo.beats"], &["lfo_freq"])];

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
			"threshold",
			"ratio",
			"attack",
			"release",
			"lookahead",
			"makeup_db",
			"dry_gain",
			"wet_gain",
			"sidechain",
			"env_attack",
			"env_release",
			"env_gain",
		],
	),
	(
		1,
		&[
			"threshold",
			"ratio",
			"attack",
			"release",
			"lookahead",
			"makeup_db",
			"dry_gain",
			"wet_gain",
			"sidechain",
			"env_cutoff",
			"env_gain",
		],
	),
	(
		2,
		&[
			"threshold",
			"ratio",
			"attack",
			"release",
			"lookahead",
			"makeup_db",
			"dry_gain",
			"wet_gain",
			"sidechain",
			"env_gain",
		],
	),
	(
		3,
		&[
			"threshold",
			"ratio",
			"attack",
			"release",
			"lookahead",
			"makeup_db",
			"dry_gain",
			"wet_gain",
			"sidechain",
			"env_gain",
		],
	),
];

web_plugin!(
	CompressorWeb,
	CompressorEffect,
	CompressorEffect::new,
	"iamdsp.effects.compressor.web",
	"I Am Compressor",
	(640.0, 320.0),
	"#ff7a6b",
	mode = "enveloper",
	visible = COMPRESSOR_VISIBLE,
	pages = COMPRESSOR_PAGES,
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
	header = &["midi"],
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
	(
		0,
		&[
			"b1_on", "b1_type", "b1_freq", "b1_gain", "b1_q", "b1_slope", "output",
		],
	),
	(
		1,
		&[
			"b2_on", "b2_type", "b2_freq", "b2_gain", "b2_q", "b2_slope", "output",
		],
	),
	(
		2,
		&[
			"b3_on", "b3_type", "b3_freq", "b3_gain", "b3_q", "b3_slope", "output",
		],
	),
	(
		3,
		&[
			"b4_on", "b4_type", "b4_freq", "b4_gain", "b4_q", "b4_slope", "output",
		],
	),
	(
		4,
		&[
			"b5_on", "b5_type", "b5_freq", "b5_gain", "b5_q", "b5_slope", "output",
		],
	),
	(
		5,
		&[
			"b6_on", "b6_type", "b6_freq", "b6_gain", "b6_q", "b6_slope", "output",
		],
	),
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
	header = &["midi"],
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
#[cfg(test)]
mod tests;
