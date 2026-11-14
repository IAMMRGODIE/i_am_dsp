//! The module types the rack can hold.
//!
//! Every module is the product itself — the same construction, the same
//! parameters and the same card metadata — read from `i_am_effects::catalog`.
//! Nothing here describes an effect a second time, so the rack cannot drift from
//! what the standalone plug-in does.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::sync::OnceLock;

use i_am_dsp::prelude::{Effect, Parameter, Value};
use i_am_effects::catalog;
use i_am_web_gui::Channels;

/// A module type a rack slot can hold.
#[derive(
	Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
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
	/// The autotune product, which follows the host's notes.
	Autotune,
	/// The spectral retune product.
	SpectralRetune,
	/// A resonator that rings at the host's notes.
	Resonator,
	/// The unstablizer product.
	Unstablizer,
	/// A reverb that convolves with a generated response.
	IrReverb,
	/// The STFT vocoder product.
	StftVocoder,
	/// The OTT multiband compressor.
	Ott,
	/// The vocoder product.
	Vocoder,
	/// A convolver whose impulse response the card loads from a file.
	Convolver,
	/// A distortion whose transfer curve is a script.
	ScriptDistortion,
	/// A frequency mapper whose table is a script.
	ScriptMapper,
	/// The History time-remap product.
	History,
}

impl ModuleKind {
	/// Every kind the rack knows, in menu order.
	pub const ALL: [Self; 39] = [
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
		Self::Autotune,
		Self::SpectralRetune,
		Self::Resonator,
		Self::Unstablizer,
		Self::IrReverb,
		Self::StftVocoder,
		Self::Ott,
		Self::Vocoder,
		Self::Convolver,
		Self::ScriptDistortion,
		Self::ScriptMapper,
		Self::History,
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
			Self::Autotune => "autotune",
			Self::SpectralRetune => "spectral_retune",
			Self::Resonator => "resonator",
			Self::Unstablizer => "unstablizer",
			Self::IrReverb => "ir_reverb",
			Self::StftVocoder => "stft_vocoder",
			Self::Ott => "ott",
			Self::Vocoder => "vocoder",
			Self::Convolver => "convolver",
			Self::ScriptDistortion => "script_distortion",
			Self::ScriptMapper => "script_mapper",
			Self::History => "history",
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
					.find_map(|parameter| parameter.meta.layout.hero.map(|hero| hero.to_string()))
					.unwrap_or_else(|| "waveform".to_string());

				(kind, hero)
			})
			.collect()
	})
}

/// The audio a card reads, decided by the hero it draws: a waveform wants the
/// three mono rings, a goniometer the L/R pair, and a hero that draws parameters
/// wants nothing. This is also the list the page polls by, so a module with no
/// audio on screen never has its rings allocated.
pub fn audio_scope(kind: ModuleKind) -> Channels {
	match hero(kind) {
		"stereo" => Channels::Stereo,
		"waveform" | "transfer" | "spectrum" | "spectra" | "compressor" | "limiter" | "gate"
		| "expander" | "eq" => Channels::Mono,
		_ => Channels::None,
	}
}

/// Kinds whose card shows the notes they follow: the pitch-tracking products.
pub fn shows_notes(kind: ModuleKind) -> bool {
	matches!(
		kind,
		ModuleKind::Autotune
			| ModuleKind::SpectralRetune
			| ModuleKind::Resonator
			| ModuleKind::Unstablizer
	)
}

/// The meter a kind publishes for its card: the OTT's three gain reductions, or
/// a vocoder's band levels. Everything else draws parameters only.
pub fn meter(kind: ModuleKind) -> Option<&'static str> {
	match kind {
		ModuleKind::Ott => Some("gain"),
		ModuleKind::Vocoder => Some("bands"),
		_ => None,
	}
}

/// The group a kind is listed under, so a menu of thirty-eight modules stays
/// readable.
pub fn group(kind: ModuleKind) -> &'static str {
	match kind {
		ModuleKind::Filter
		| ModuleKind::Allpass
		| ModuleKind::Comb
		| ModuleKind::Formant
		| ModuleKind::Eq => "Filter",
		ModuleKind::Distortion
		| ModuleKind::Crusher
		| ModuleKind::Disperser
		| ModuleKind::TapeStop
		| ModuleKind::Transient => "Colour",
		ModuleKind::Compressor
		| ModuleKind::Gate
		| ModuleKind::Expander
		| ModuleKind::Limiter
		| ModuleKind::Ott => "Dynamics",
		ModuleKind::Delay
		| ModuleKind::Chorus
		| ModuleKind::Flanger
		| ModuleKind::Phaser
		| ModuleKind::Reverb
		| ModuleKind::Schroeder
		| ModuleKind::IrReverb
		| ModuleKind::Convolver
		| ModuleKind::ParticleWider => "Space",
		ModuleKind::Pitch
		| ModuleKind::Shifter
		| ModuleKind::Autotune
		| ModuleKind::SpectralRetune
		| ModuleKind::Resonator
		| ModuleKind::Unstablizer
		| ModuleKind::Vocoder
		| ModuleKind::StftVocoder => "Pitch",
		ModuleKind::Gain
		| ModuleKind::Balance
		| ModuleKind::Width
		| ModuleKind::Router
		| ModuleKind::ScriptDistortion
		| ModuleKind::ScriptMapper
		| ModuleKind::History => "Utility",
	}
}

/// Whether a kind runs a script the page can edit.
pub fn needs_script(kind: ModuleKind) -> bool {
	matches!(
		kind,
		ModuleKind::ScriptDistortion | ModuleKind::ScriptMapper
	)
}

/// Whether a kind keeps an impulse response the page can load into it.
pub fn needs_ir(kind: ModuleKind) -> bool {
	kind == ModuleKind::Convolver
}

/// Builds a fresh effect for a kind at a sample rate.
///
/// The kinds whose page talks to them (the convolver's impulse slot) get their
/// shared object built into them, so the route and the audio thread see one.
pub(crate) fn build(
	kind: ModuleKind,
	sample_rate: usize,
	ir: Option<Arc<i_am_effects::IrSlot>>,
	script: Option<Arc<crate::script::ScriptSlot>>,
	history: Option<i_am_effects::HistoryHandle>,
) -> Box<dyn Effect<2>> {
	if kind == ModuleKind::History {
		return Box::new(i_am_dsp::prelude::HistoryEffect::with_feed(
			sample_rate,
			history.expect("the history kind needs its pages").feed(),
		));
	}

	let descriptor = kind.descriptor();

	if let Some(build) = descriptor.build_with_ir {
		return build(sample_rate, ir.expect("the kind needs its slot"));
	}

	// The scripted products are built around the worker the page edits.
	if let Some(script) = script {
		match &*script {
			crate::script::ScriptSlot::Distortion(worker) => {
				return Box::new(i_am_effects::ScriptDistortionEffect::new(worker.clone()));
			}
			crate::script::ScriptSlot::Mapper(worker) => {
				return Box::new(i_am_effects::ScriptMapperEffect::new(
					sample_rate,
					worker.clone(),
				));
			}
		}
	}

	(descriptor.build)(sample_rate)
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
