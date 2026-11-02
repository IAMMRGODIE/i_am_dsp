//! The module catalog: every product the rack can host, described once.
//!
//! A descriptor carries the whole product — the DSP construction the standalone
//! plug-in uses, and the card metadata its faceplate draws (name, colour, hero,
//! mode, header, visible controls). The rack builds its modules and cards from
//! these, so a module cannot drift from the product it mirrors.

use i_am_dsp::prelude::{
	Chorus, Crusher, Delay, Distortion, Effect, Gain, Multimode, Parameter, Phaser, SineWave,
};

use i_am_web_gui::Switch;

use crate::compressor::CompressorEffect;
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

/// Every module the catalog currently exposes.
pub const MODULES: &[ModuleDescriptor] = &[
	FILTER, DISTORTION, COMPRESSOR, GAIN, DELAY, CRUSHER, CHORUS, PHASER, REVERB, PITCH,
];

/// Finds a descriptor by its stable id.
pub fn get(id: &str) -> Option<&'static ModuleDescriptor> {
	MODULES.iter().find(|module| module.id == id)
}
