//! The module types the rack can hold.
//!
//! Every module is the product itself — the same construction, the same
//! parameters and the same card metadata — read from `i_am_effects::catalog`.
//! Nothing here describes an effect a second time, so the rack cannot drift from
//! what the standalone plug-in does.

use std::collections::BTreeMap;

use i_am_dsp::prelude::{Effect, Parameter, Value};
use i_am_effects::catalog;

/// A module type a rack slot can hold.
#[derive(
	Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash,
	serde::Serialize, serde::Deserialize,
)]
pub enum ModuleKind {
	/// A seven-mode resonant filter.
	Filter,
	/// The six-shaper distortion product.
	Distortion,
	/// A peak/RMS compressor with selectable detectors.
	Compressor,
	/// A plain gain.
	Gain,
	/// The delay product, with its tempo switch.
	Delay,
	/// A disperser, saturator and amplitude modulation in one.
	Crusher,
	/// A modulated short delay.
	Chorus,
	/// A swept allpass stack.
	Phaser,
	/// The product reverb.
	Reverb,
	/// The product pitch shifter.
	Pitch,
}

impl ModuleKind {
	/// Every kind the rack knows, in menu order.
	pub const ALL: [Self; 10] = [
		Self::Filter,
		Self::Distortion,
		Self::Compressor,
		Self::Gain,
		Self::Delay,
		Self::Crusher,
		Self::Chorus,
		Self::Phaser,
		Self::Reverb,
		Self::Pitch,
	];

	/// The catalog id this kind stores in its state.
	pub const fn id(self) -> &'static str {
		match self {
			Self::Filter => "filter",
			Self::Distortion => "distortion",
			Self::Compressor => "compressor",
			Self::Gain => "gain",
			Self::Delay => "delay",
			Self::Crusher => "crusher",
			Self::Chorus => "chorus",
			Self::Phaser => "phaser",
			Self::Reverb => "reverb",
			Self::Pitch => "pitch",
		}
	}

	/// The product descriptor behind this kind.
	pub fn descriptor(self) -> &'static catalog::ModuleDescriptor {
		catalog::get(self.id()).expect("every rack kind has a catalog descriptor")
	}

	/// The name a card shows.
	pub fn name(self) -> &'static str {
		self.descriptor().name
	}

	/// The identity colour the product uses.
	pub fn accent(self) -> &'static str {
		self.descriptor().accent
	}

	/// The picture a card draws above its controls.
	pub fn hero(self) -> &'static str {
		self.descriptor().hero
	}

	/// The parameter the page draws as a mode switch in the header.
	pub fn mode(self) -> Option<&'static str> {
		self.descriptor().mode
	}

	/// The parameters the page draws in the header instead of the body.
	pub fn header(self) -> &'static [&'static str] {
		self.descriptor().header
	}

	/// The controls each mode reads, when the card hides the rest.
	pub fn visible(self) -> Option<&'static [(i32, &'static [&'static str])]> {
		self.descriptor().visible
	}

	/// The pairs that trade places when a switch in the header moves.
	pub fn switches(self) -> &'static [i_am_web_gui::Switch] {
		self.descriptor().switches
	}
}

/// Builds a fresh effect for a kind at a sample rate.
pub(crate) fn build(kind: ModuleKind, sample_rate: usize) -> Box<dyn Effect<2>> {
	(kind.descriptor().build)(sample_rate)
}

/// The parameter schema of a kind, at a rate that does not affect the metadata.
pub(crate) fn schema(kind: ModuleKind) -> Vec<Parameter> {
	kind.descriptor().parameters()
}

/// The parameters a fresh module starts with.
///
/// Everything has a numeric form here: a float is itself, an integer is widened,
/// and a boolean is 0 or 1. The engine converts back to the type the parameter
/// map holds.
pub(crate) fn default_params(kind: ModuleKind) -> BTreeMap<String, f64> {
	schema(kind)
		.into_iter()
		.filter_map(|parameter| match parameter.value {
			Value::Float { value, .. } => Some((parameter.identifier, value as f64)),
			Value::Int { value, .. } => Some((parameter.identifier, value as f64)),
			Value::Bool(value) => Some((parameter.identifier, if value { 1.0 } else { 0.0 })),
			_ => None,
		})
		.collect()
}
