//! The module catalog shared with the rack.
//!
//! A module descriptor carries the same product-level information the standalone
//! faceplate uses: name, colour, hero, mode, header choices and per-mode visible
//! controls. The rack builds its cards from the descriptor and the effect's
//! parameter metadata, so adding a module here is enough to make it available in
//! both places.

use i_am_dsp::prelude::{Effect, Parameter};

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
	/// Builds a fresh effect at the host sample rate.
	pub build: fn(usize) -> Box<dyn Effect<2>>,
}

impl ModuleDescriptor {
	/// The parameter schema a fresh module starts with.
	pub fn parameters(&self) -> Vec<Parameter> {
		(self.build)(44_100).get_parameters()
	}
}

fn reverb(sample_rate: usize) -> Box<dyn Effect<2>> {
	Box::new(crate::reverb::ReverbEffect::new(sample_rate))
}

fn pitch(sample_rate: usize) -> Box<dyn Effect<2>> {
	Box::new(crate::pitch::PitchEffect::new(sample_rate))
}

/// The reverb product, as a rack module.
pub const REVERB: ModuleDescriptor = ModuleDescriptor {
	id: "reverb",
	name: "Reverb",
	accent: "#6c8cff",
	hero: "tail",
	mode: None,
	header: &[],
	visible: None,
	build: reverb,
};

/// The pitch product, as a rack module.
pub const PITCH: ModuleDescriptor = ModuleDescriptor {
	id: "pitch",
	name: "Pitch",
	accent: "#c8e04a",
	hero: "pitch",
	mode: Some("algorithm"),
	header: &["window_size"],
	visible: Some(crate::PITCH_VISIBLE),
	build: pitch,
};

/// Every module the catalog currently exposes.
pub const MODULES: &[ModuleDescriptor] = &[REVERB, PITCH];

/// Finds a descriptor by its stable id.
pub fn get(id: &str) -> Option<&'static ModuleDescriptor> {
	MODULES.iter().find(|module| module.id == id)
}
