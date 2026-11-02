//! The module catalog: every product the rack can host, described once.
//!
//! A descriptor carries the whole product — the DSP construction the standalone
//! plug-in uses, and the card metadata its faceplate draws (name, colour, hero,
//! mode, header, visible controls). The rack builds its modules and cards from
//! these, so a module cannot drift from the product it mirrors.

use i_am_dsp::prelude::{
	Chorus, Crusher, Delay, Distortion, Effect, Expander, Flanger, FormantFilter, Gain, Gate,
	Limiter, LrControl, Multimode, Parameter, ParametricEq, Phaser, SineWave, StereoController,
	StereoRouter, TapeStopEffect, TransientShaper,
};

use i_am_web_gui::Switch;

use crate::allpass::AllpassEffect;
use crate::comb::CombEffect;
use crate::compressor::CompressorEffect;
use crate::disperser::Dispersion;
use crate::particle_wider::ParticleWiderEffect;
use crate::schroeder::SchroederEffect;
use crate::shifter::ShifterEffect;
use crate::{PITCH_VISIBLE, PitchEffect, ReverbEffect};

/// One rack module: its product metadata and a fresh effect builder.
pub struct ModuleDescriptor {
	/// Stable identifier used by the rack state.
	pub id: &'static str,
	/// Name shown on the card and in the add menu.
	pub name: &'static str,
	/// Product accent colour.
	pub accent: &'static str,
	/// Hero kind drawn above the controls.
	pub hero: &'static str,
	/// Parameter drawn as the header mode switch.
	pub mode: Option<&'static str>,
	/// Parameters drawn in the header instead of the body.
	pub header: &'static [&'static str],
	/// Parameters visible for each mode, when the product hides the rest.
	pub visible: Option<&'static [(i32, &'static [&'static str])]>,
	/// Pairs of controls that trade places when a switch moves.
	pub switches: &'static [Switch],
	/// Builds a fresh effect at the host sample rate.
	pub build: fn(usize) -> Box<dyn Effect<2>>,
}

impl ModuleDescriptor {
	/// The parameter schema a fresh module starts with.
	pub fn parameters(&self) -> Vec<Parameter> {
		(self.build)(44_100).get_parameters()
	}
}

fn filter(sample_rate: usize) -> Box<dyn Effect<2>> {
	Box::new(Multimode::new(sample_rate, 1_000.0, 0.707))
}

fn distortion(sample_rate: usize) -> Box<dyn Effect<2>> {
	Box::new(Distortion::<2>::new(sample_rate))
}

fn compressor(sample_rate: usize) -> Box<dyn Effect<2>> {
	Box::new(CompressorEffect::new(sample_rate))
}

fn gain(_: usize) -> Box<dyn Effect<2>> {
	Box::new(Gain::new(1.0))
}

fn delay(sample_rate: usize) -> Box<dyn Effect<2>> {
	Box::new(Delay::<(), 2>::new((), 1 << 16, 250.0, sample_rate))
}

fn crusher(sample_rate: usize) -> Box<dyn Effect<2>> {
	Box::new(Crusher::<2>::new(sample_rate))
}

fn chorus(sample_rate: usize) -> Box<dyn Effect<2>> {
	Box::new(Chorus::<SineWave, 2>::new(1 << 14, 4, sample_rate, SineWave))
}

fn phaser(sample_rate: usize) -> Box<dyn Effect<2>> {
	Box::new(Phaser::<SineWave, 2>::new(SineWave, 6, sample_rate))
}

fn reverb(sample_rate: usize) -> Box<dyn Effect<2>> {
	Box::new(ReverbEffect::new(sample_rate))
}

fn pitch(sample_rate: usize) -> Box<dyn Effect<2>> {
	Box::new(PitchEffect::new(sample_rate))
}


fn comb(sample_rate: usize) -> Box<dyn Effect<2>> {
	Box::new(CombEffect::new(sample_rate))
}

fn allpass(sample_rate: usize) -> Box<dyn Effect<2>> {
	Box::new(AllpassEffect::new(sample_rate))
}

fn width(_: usize) -> Box<dyn Effect<2>> {
	Box::new(StereoController::new())
}

fn shifter(sample_rate: usize) -> Box<dyn Effect<2>> {
	Box::new(ShifterEffect::new(sample_rate))
}

fn disperser(sample_rate: usize) -> Box<dyn Effect<2>> {
	Box::new(Dispersion::new(sample_rate))
}

fn balance(_: usize) -> Box<dyn Effect<2>> {
	Box::new(LrControl::default())
}

fn particle_wider(sample_rate: usize) -> Box<dyn Effect<2>> {
	Box::new(ParticleWiderEffect::new(sample_rate))
}

fn schroeder(sample_rate: usize) -> Box<dyn Effect<2>> {
	Box::new(SchroederEffect::new(sample_rate))
}

fn flanger(sample_rate: usize) -> Box<dyn Effect<2>> {
	Box::new(Flanger::<SineWave, 2>::new(1 << 18, sample_rate, SineWave))
}

fn eq(sample_rate: usize) -> Box<dyn Effect<2>> {
	Box::new(ParametricEq::<2>::new(sample_rate))
}

fn tape_stop(sample_rate: usize) -> Box<dyn Effect<2>> {
	Box::new(TapeStopEffect::<2>::new(sample_rate))
}

fn gate(sample_rate: usize) -> Box<dyn Effect<2>> {
	Box::new(Gate::<2>::new(sample_rate))
}

fn expander(sample_rate: usize) -> Box<dyn Effect<2>> {
	Box::new(Expander::<2>::new(sample_rate))
}

fn limiter(sample_rate: usize) -> Box<dyn Effect<2>> {
	Box::new(Limiter::<2>::new(sample_rate))
}

fn formant(sample_rate: usize) -> Box<dyn Effect<2>> {
	Box::new(FormantFilter::<2>::new(sample_rate))
}

fn transient(sample_rate: usize) -> Box<dyn Effect<2>> {
	Box::new(TransientShaper::<2>::new(sample_rate))
}

fn router(sample_rate: usize) -> Box<dyn Effect<2>> {
	Box::new(StereoRouter::new(sample_rate))
}

/// The multimode filter product.
pub const FILTER: ModuleDescriptor = ModuleDescriptor {
	id: "filter",
	name: "Filter",
	accent: "#4c8dff",
	hero: "spectrum",
	mode: Some("mode"),
	header: crate::FILTER_HEADER,
	visible: Some(crate::FILTER_VISIBLE),
	switches: &[],
	build: filter,
};

/// The six-shaper distortion product.
pub const DISTORTION: ModuleDescriptor = ModuleDescriptor {
	id: "distortion",
	name: "Distortion",
	accent: "#ff9a3d",
	hero: "transfer",
	mode: Some("mode"),
	header: &[],
	visible: Some(crate::DISTORTION_VISIBLE),
	switches: &[],
	build: distortion,
};

/// The compressor product, with its selectable detectors.
pub const COMPRESSOR: ModuleDescriptor = ModuleDescriptor {
	id: "compressor",
	name: "Compressor",
	accent: "#ff7a6b",
	hero: "compressor",
	mode: Some("enveloper"),
	header: &[],
	visible: Some(crate::COMPRESSOR_VISIBLE),
	switches: &[],
	build: compressor,
};

/// The gain product.
pub const GAIN: ModuleDescriptor = ModuleDescriptor {
	id: "gain",
	name: "Gain",
	accent: "#35c77a",
	hero: "waveform",
	mode: None,
	header: &[],
	visible: None,
	switches: &[],
	build: gain,
};

/// The delay product, with its tempo switch.
pub const DELAY: ModuleDescriptor = ModuleDescriptor {
	id: "delay",
	name: "Delay",
	accent: "#e8c35a",
	hero: "waveform",
	mode: None,
	header: crate::DELAY_HEADER,
	visible: None,
	switches: crate::DELAY_SWITCHES,
	build: delay,
};

/// The crusher product.
pub const CRUSHER: ModuleDescriptor = ModuleDescriptor {
	id: "crusher",
	name: "Crusher",
	accent: "#8a6aff",
	hero: "transfer",
	mode: None,
	header: &[],
	visible: None,
	switches: &[],
	build: crusher,
};

/// The chorus product, with its tempo switch.
pub const CHORUS: ModuleDescriptor = ModuleDescriptor {
	id: "chorus",
	name: "Chorus",
	accent: "#5fe0b0",
	hero: "waveform",
	mode: None,
	header: crate::CHORUS_HEADER,
	visible: None,
	switches: crate::CHORUS_SWITCHES,
	build: chorus,
};

/// The phaser product, with its tempo switch.
pub const PHASER: ModuleDescriptor = ModuleDescriptor {
	id: "phaser",
	name: "Phaser",
	accent: "#b48cff",
	hero: "spectrum",
	mode: None,
	header: crate::PHASER_HEADER,
	visible: None,
	switches: crate::PHASER_SWITCHES,
	build: phaser,
};

/// The reverb product.
pub const REVERB: ModuleDescriptor = ModuleDescriptor {
	id: "reverb",
	name: "Reverb",
	accent: "#6c8cff",
	hero: "tail",
	mode: None,
	header: &[],
	visible: None,
	switches: &[],
	build: reverb,
};

/// The pitch product.
pub const PITCH: ModuleDescriptor = ModuleDescriptor {
	id: "pitch",
	name: "Pitch",
	accent: "#c8e04a",
	hero: "pitch",
	mode: Some("algorithm"),
	header: &["window_size"],
	visible: Some(PITCH_VISIBLE),
	switches: &[],
	build: pitch,
};

/// The comb product.
pub const COMB: ModuleDescriptor = ModuleDescriptor {
	id: "comb",
	name: "Comb",
	accent: "#f0b35e",
	hero: "tail",
	mode: None,
	header: &[],
	visible: None,
	switches: &[],
	build: comb,
};

/// The allpass product.
pub const ALLPASS: ModuleDescriptor = ModuleDescriptor {
	id: "allpass",
	name: "Allpass",
	accent: "#b18cff",
	hero: "phase",
	mode: None,
	header: &[],
	visible: None,
	switches: &[],
	build: allpass,
};

/// The width product.
pub const WIDTH: ModuleDescriptor = ModuleDescriptor {
	id: "width",
	name: "Width",
	accent: "#ff6fae",
	hero: "stereo",
	mode: None,
	header: &[],
	visible: None,
	switches: &[],
	build: width,
};

/// The shifter product.
pub const SHIFTER: ModuleDescriptor = ModuleDescriptor {
	id: "shifter",
	name: "Shifter",
	accent: "#7aa2ff",
	hero: "pitch",
	mode: Some("algorithm"),
	header: &["window_size"],
	visible: Some(crate::SHIFTER_VISIBLE),
	switches: &[],
	build: shifter,
};

/// The disperser product.
pub const DISPERSER: ModuleDescriptor = ModuleDescriptor {
	id: "disperser",
	name: "Disperser",
	accent: "#2fc6c0",
	hero: "spectrum",
	mode: None,
	header: &[],
	visible: None,
	switches: &[],
	build: disperser,
};

/// The balance product.
pub const BALANCE: ModuleDescriptor = ModuleDescriptor {
	id: "balance",
	name: "Balance",
	accent: "#d96aff",
	hero: "balance",
	mode: None,
	header: &[],
	visible: None,
	switches: &[],
	build: balance,
};

/// The particle wider product.
pub const PARTICLE_WIDER: ModuleDescriptor = ModuleDescriptor {
	id: "particle_wider",
	name: "Particle Wider",
	accent: "#67d7a8",
	hero: "particles",
	mode: Some("fft_size"),
	header: &[],
	visible: None,
	switches: &[],
	build: particle_wider,
};

/// The schroeder product.
pub const SCHROEDER: ModuleDescriptor = ModuleDescriptor {
	id: "schroeder",
	name: "Schroeder",
	accent: "#e0b36a",
	hero: "impulse",
	mode: None,
	header: &[],
	visible: None,
	switches: &[],
	build: schroeder,
};

/// The flanger product.
pub const FLANGER: ModuleDescriptor = ModuleDescriptor {
	id: "flanger",
	name: "Flanger",
	accent: "#6bb8ff",
	hero: "phase",
	mode: None,
	header: crate::FLANGER_HEADER,
	visible: None,
	switches: crate::FLANGER_SWITCHES,
	build: flanger,
};

/// The eq product.
pub const EQ: ModuleDescriptor = ModuleDescriptor {
	id: "eq",
	name: "EQ",
	accent: "#8a7bff",
	hero: "eq",
	mode: Some("band"),
	header: &[],
	visible: Some(crate::EQ_VISIBLE),
	switches: &[],
	build: eq,
};

/// The tape stop product.
pub const TAPE_STOP: ModuleDescriptor = ModuleDescriptor {
	id: "tape_stop",
	name: "Tape Stop",
	accent: "#7ad1e0",
	hero: "waveform",
	mode: None,
	header: &[],
	visible: None,
	switches: &[],
	build: tape_stop,
};

/// The gate product.
pub const GATE: ModuleDescriptor = ModuleDescriptor {
	id: "gate",
	name: "Gate",
	accent: "#6fd3a3",
	hero: "gate",
	mode: None,
	header: &[],
	visible: None,
	switches: &[],
	build: gate,
};

/// The expander product.
pub const EXPANDER: ModuleDescriptor = ModuleDescriptor {
	id: "expander",
	name: "Expander",
	accent: "#79b8ff",
	hero: "expander",
	mode: None,
	header: &[],
	visible: None,
	switches: &[],
	build: expander,
};

/// The limiter product.
pub const LIMITER: ModuleDescriptor = ModuleDescriptor {
	id: "limiter",
	name: "Limiter",
	accent: "#ffc45a",
	hero: "limiter",
	mode: None,
	header: &[],
	visible: None,
	switches: &[],
	build: limiter,
};

/// The formant filter product.
pub const FORMANT: ModuleDescriptor = ModuleDescriptor {
	id: "formant_filter",
	name: "Formant Filter",
	accent: "#ff9f6e",
	hero: "formant",
	mode: None,
	header: &[],
	visible: None,
	switches: &[],
	build: formant,
};

/// The transient product.
pub const TRANSIENT: ModuleDescriptor = ModuleDescriptor {
	id: "transient",
	name: "Transient",
	accent: "#ff7ad9",
	hero: "waveform",
	mode: None,
	header: &[],
	visible: None,
	switches: &[],
	build: transient,
};

/// The router product.
pub const ROUTER: ModuleDescriptor = ModuleDescriptor {
	id: "router",
	name: "Router",
	accent: "#8ad3ff",
	hero: "stereo",
	mode: None,
	header: &[],
	visible: None,
	switches: &[],
	build: router,
};

/// Every module the catalog currently exposes.
pub const MODULES: &[ModuleDescriptor] = &[
	FILTER,
	DISTORTION,
	COMPRESSOR,
	GATE,
	EXPANDER,
	LIMITER,
	GAIN,
	DELAY,
	COMB,
	ALLPASS,
	CHORUS,
	FLANGER,
	PHASER,
	CRUSHER,
	DISPERSER,
	WIDTH,
	BALANCE,
	ROUTER,
	TAPE_STOP,
	TRANSIENT,
	FORMANT,
	EQ,
	REVERB,
	SCHROEDER,
	PARTICLE_WIDER,
	PITCH,
	SHIFTER,
];

/// Finds a descriptor by its stable id.
pub fn get(id: &str) -> Option<&'static ModuleDescriptor> {
	MODULES.iter().find(|module| module.id == id)
}
