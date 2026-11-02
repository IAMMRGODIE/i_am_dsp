//! The module types the rack can hold.
//!
//! Every module is the product itself — the same construction, the same
//! parameters and the same card metadata — read from `i_am_effects::catalog`.
//! Nothing here describes an effect a second time, so the rack cannot drift from
//! what the standalone plug-in does.

use std::collections::{BTreeMap, HashMap};
use std::sync::OnceLock;

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
	/// A downward gate.
	Gate,
	/// An upward expander.
	Expander,
	/// A brickwall limiter.
	Limiter,
	/// A plain gain.
	Gain,
	/// The delay product, with its tempo switch.
	Delay,
	/// A short feed-forward comb.
	Comb,
	/// A swept allpass stack.
	Allpass,
	/// A modulated short delay.
	Chorus,
	/// A short modulated delay with feedback.
	Flanger,
	/// A swept allpass stack, phaser flavour.
	Phaser,
	/// A disperser, saturator and amplitude modulation in one.
	Crusher,
	/// An allpass dispersion chain.
	Disperser,
	/// Mid/side width control.
	Width,
	/// A left/right balance control.
	Balance,
	/// A stereo router.
	Router,
	/// A tape-stop effect.
	TapeStop,
	/// A transient shaper.
	Transient,
	/// A vowel formant filter.
	Formant,
	/// A six-band parametric EQ.
	Eq,
	/// The product reverb.
	Reverb,
	/// A Schroeder reverb.
	Schroeder,
	/// A particle-based widener.
	ParticleWider,
	/// The product pitch shifter.
	Pitch,
	/// A shifter product with its own window controls.
	Shifter,
}

impl ModuleKind {
	/// Every kind the rack knows, in menu order.
	pub const ALL: [Self; 27] = [
		Self::Filter,
		Self::Distortion,
		Self::Compressor,
		Self::Gate,
		Self::Expander,
		Self::Limiter,
		Self::Gain,
		Self::Delay,
		Self::Comb,
		Self::Allpass,
		Self::Chorus,
		Self::Flanger,
		Self::Phaser,
		Self::Crusher,
		Self::Disperser,
		Self::Width,
		Self::Balance,
		Self::Router,
		Self::TapeStop,
		Self::Transient,
		Self::Formant,
		Self::Eq,
		Self::Reverb,
		Self::Schroeder,
		Self::ParticleWider,
		Self::Pitch,
		Self::Shifter,
	];

	/// The catalog id this kind stores in its state.
	pub const fn id(self) -> &'static str {
		match self {
			Self::Filter => "filter",
			Self::Distortion => "distortion",
			Self::Compressor => "compressor",
			Self::Gate => "gate",
			Self::Expander => "expander",
			Self::Limiter => "limiter",
			Self::Gain => "gain",
			Self::Delay => "delay",
			Self::Comb => "comb",
			Self::Allpass => "allpass",
			Self::Chorus => "chorus",
			Self::Flanger => "flanger",
			Self::Phaser => "phaser",
			Self::Crusher => "crusher",
			Self::Disperser => "disperser",
			Self::Width => "width",
			Self::Balance => "balance",
			Self::Router => "router",
			Self::TapeStop => "tape_stop",
			Self::Transient => "transient",
			Self::Formant => "formant_filter",
			Self::Eq => "eq",
			Self::Reverb => "reverb",
			Self::Schroeder => "schroeder",
			Self::ParticleWider => "particle_wider",
			Self::Pitch => "pitch",
			Self::Shifter => "shifter",
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

/// The picture a card draws above its controls.
///
/// The parameter metadata is the only place that names a hero — the product page
/// reads the same field — so the rack asks the effect instead of keeping a list
/// of its own that can go stale.
pub fn hero(kind: ModuleKind) -> &'static str {
	heroes()
		.get(&kind)
		.map(String::as_str)
		.unwrap_or("waveform")
}

fn heroes() -> &'static HashMap<ModuleKind, String> {
	static HEROES: OnceLock<HashMap<ModuleKind, String>> = OnceLock::new();

	HEROES.get_or_init(|| {
		ModuleKind::ALL
			.into_iter()
			.map(|kind| {
				let hero = schema(kind)
					.into_iter()
					.find_map(|parameter| {
						parameter.meta.layout.hero.map(|hero| hero.to_string())
					})
					.unwrap_or_else(|| "waveform".to_string());

				(kind, hero)
			})
			.collect()
	})
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
