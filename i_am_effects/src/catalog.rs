//! The module catalog: every product the rack can host, described once.
//!
//! A descriptor carries the whole product — the DSP construction the standalone
//! plug-in uses, and the card metadata its faceplate draws (name, colour, mode,
//! header, visible controls, switches). The hero is not here: it lives on the
//! parameters, where the product page reads it too.

use std::sync::Arc;

use i_am_dsp::prelude::{
	AutoTune, Chorus, Crusher, Delay, Distortion, Effect, Expander, Flanger, FormantFilter, Gain,
	Gate, HistoryEffect, HistoryFeed, Limiter, LrControl, Multimode, OttMeters, OuterWider,
	Parameter, ParametricEq, Phaser, Resonator, RingMod, SineWave, SpectralRetuneEffect,
	StereoController, StereoRouter, StftVocoderEffect, TapeStopEffect, TransientShaper,
	VocoderEffect, VocoderMeters,
};

use i_am_dsp::prelude::{FilterFrame, FilterTableEffect};

use i_am_web_gui::Switch;

use crate::allpass::AllpassEffect;
use crate::comb::CombEffect;
use crate::compressor::CompressorEffect;
use crate::convolver::{ConvolverEffect, IrSlot};
use crate::disperser::Dispersion;
use crate::ir_reverb::IrReverbEffect;
use crate::ott::OttEffect;
use crate::particle_wider::ParticleWiderEffect;
use crate::schroeder::SchroederEffect;
use crate::script_distortion::{ScriptDistortionEffect, distortion_worker};
use crate::script_mapper::{ScriptMapperEffect, mapper_worker};
use crate::shifter::ShifterEffect;
use crate::spectral_dynamics::spectral_dynamics;
use crate::unstablizer::UnstablizerEffect;
use crate::{PITCH_VISIBLE, PitchEffect, ReverbEffect};

/// One rack module: its product metadata and a fresh effect builder.
pub struct ModuleDescriptor {
	/// Stable identifier used by the rack state.
	pub id: &'static str,
	/// Name shown on the card and in the add menu.
	pub name: &'static str,
	/// Product accent colour.
	pub accent: &'static str,
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
	/// A build that hands the effect the shared object its page talks to — the
	/// convolver's impulse response slot. Modules with nothing to hand over
	/// leave this as None and are built by ["build"].
	pub build_with_ir: Option<IrBuilder>,
}

/// Builds an impulse-driven module around the slot its page writes to.
pub type IrBuilder = fn(usize, Arc<IrSlot>) -> Box<dyn Effect<2>>;

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
	Box::new(Chorus::<SineWave, 2>::new(
		1 << 14,
		4,
		sample_rate,
		SineWave,
	))
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

fn outer_wider(sample_rate: usize) -> Box<dyn Effect<2>> {
	Box::new(OuterWider::new(sample_rate))
}

fn shifter(sample_rate: usize) -> Box<dyn Effect<2>> {
	Box::new(ShifterEffect::new(sample_rate))
}

fn ring_mod(sample_rate: usize) -> Box<dyn Effect<2>> {
	Box::new(RingMod::new(sample_rate))
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

fn autotune(sample_rate: usize) -> Box<dyn Effect<2>> {
	Box::new(AutoTune::<2>::new(sample_rate))
}

fn spectral_retune(sample_rate: usize) -> Box<dyn Effect<2>> {
	Box::new(SpectralRetuneEffect::new(sample_rate))
}

fn resonator(sample_rate: usize) -> Box<dyn Effect<2>> {
	Box::new(Resonator::<2>::new(sample_rate))
}

fn unstablizer(sample_rate: usize) -> Box<dyn Effect<2>> {
	Box::new(UnstablizerEffect::new(sample_rate))
}

fn ir_reverb(sample_rate: usize) -> Box<dyn Effect<2>> {
	Box::new(IrReverbEffect::new(sample_rate))
}

fn stft_vocoder(sample_rate: usize) -> Box<dyn Effect<2>> {
	Box::new(StftVocoderEffect::new(sample_rate))
}

fn ott(sample_rate: usize) -> Box<dyn Effect<2>> {
	// The rack's card reads the meter through the effect, so this Arc stays
	// private to it.
	Box::new(OttEffect::new(sample_rate, Arc::new(OttMeters::default())))
}

fn vocoder(sample_rate: usize) -> Box<dyn Effect<2>> {
	Box::new(VocoderEffect::<2>::new(
		sample_rate,
		Arc::new(VocoderMeters::default()),
	))
}

/// What a module's product declares on its input side.
///
/// The product type owns these lists (`INPUT_PORTS`, `INPUT_MIDI_PORTS`), and a
/// graph cannot read a type: the facts are restated once here so the patch can
/// draw the ports the host sees. A test in the effects' own suite holds this
/// against the products, so a product that gains a port fails it until added.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModuleIo {
	/// How many audio inputs the product declares; the main one comes first.
	pub audio_inputs: usize,
	/// Whether the product declares a note input.
	pub notes: bool,
}

/// The inputs of one module's product.
///
/// Two products read a key signal from a second audio input without naming a
/// `sidechain` parameter — the ring modulator picks it with its mode, the filter
/// table with its frame source — so this is a port list, not a parameter guess.
pub fn module_io(id: &str) -> ModuleIo {
	match id {
		"compressor" | "gate" | "expander" | "ring_mod" | "filter_table" | "vocoder"
		| "stft_vocoder" => ModuleIo {
			audio_inputs: 2,
			notes: false,
		},
		// The effects that follow the note that is played.
		"autotune" | "spectral_retune" | "resonator" | "unstablizer" => ModuleIo {
			audio_inputs: 1,
			notes: true,
		},
		_ => ModuleIo {
			audio_inputs: 1,
			notes: false,
		},
	}
}

/// The pages a product splits its controls into, for a card that draws the
/// product's own tabs.
///
/// The compressor, the OTT and the spectral dynamics are the products that split;
/// everything else is one body. A card that lists its own pages falls behind every
/// time a product grows one, which is how the patcher ended up drawing the OTT as a
/// single column of twelve knobs.
pub fn pages(id: &str) -> &'static [i_am_web_gui::Page] {
	match id {
		"compressor" => crate::COMPRESSOR_PAGES,
		"ott" => crate::OTT_PAGES,
		"spectral_dynamics" => crate::SPECTRAL_DYNAMICS_PAGES,
		_ => &[],
	}
}

/// The faceplate size a product declares, for a card that draws the product's
/// own shape rather than guessing one.
pub fn face(id: &str) -> Option<(f32, f32)> {
	match id {
		"filter" => Some(crate::FilterWeb::size()),
		"distortion" => Some(crate::DistortionWeb::size()),
		"compressor" => Some(crate::CompressorWeb::size()),
		"gain" => Some(crate::GainWeb::size()),
		"delay" => Some(crate::DelayWeb::size()),
		"crusher" => Some(crate::CrusherWeb::size()),
		"chorus" => Some(crate::ChorusWeb::size()),
		"phaser" => Some(crate::PhaserWeb::size()),
		"reverb" => Some(crate::ReverbWeb::size()),
		"pitch" => Some(crate::PitchWeb::size()),
		"comb" => Some(crate::CombWeb::size()),
		"allpass" => Some(crate::AllpassWeb::size()),
		"width" => Some(crate::WidthWeb::size()),
		"ring_mod" => Some(crate::RingModWeb::size()),
		"outer_wider" => Some(crate::OuterWiderWeb::size()),
		"shifter" => Some(crate::ShifterWeb::size()),
		"disperser" => Some(crate::DisperserWeb::size()),
		"balance" => Some(crate::BalanceWeb::size()),
		"particle_wider" => Some(crate::ParticleWiderWeb::size()),
		"schroeder" => Some(crate::SchroederWeb::size()),
		"flanger" => Some(crate::FlangerWeb::size()),
		"eq" => Some(crate::EqWeb::size()),
		"history" => Some(crate::HistoryWeb::size()),
		"filter_table" => Some(crate::FilterTableWeb::size()),
		"tape_stop" => Some(crate::TapeStopWeb::size()),
		"gate" => Some(crate::GateWeb::size()),
		"expander" => Some(crate::ExpanderWeb::size()),
		"limiter" => Some(crate::LimiterWeb::size()),
		"formant_filter" => Some(crate::FormantFilterWeb::size()),
		"transient" => Some(crate::TransientWeb::size()),
		"router" => Some(crate::RouterWeb::size()),
		"autotune" => Some(crate::AutoTuneWeb::size()),
		"spectral_retune" => Some(crate::SpectralRetuneWeb::size()),
		"spectral_dynamics" => Some(crate::SpectralDynamicsWeb::size()),
		"resonator" => Some(crate::ResonatorWeb::size()),
		"unstablizer" => Some(crate::UnstablizerWeb::size()),
		"ir_reverb" => Some(crate::IrReverbWeb::size()),
		"stft_vocoder" => Some(crate::StftVocoderWeb::size()),
		"ott" => Some(crate::OttWeb::size()),
		"vocoder" => Some(crate::VocoderWeb::size()),
		"convolver" => Some(crate::ConvolverWeb::size()),
		"script_distortion" => Some(crate::ScriptDistortionWeb::size()),
		"script_mapper" => Some(crate::ScriptMapperWeb::size()),
		_ => None,
	}
}

/// The multimode filter product.
pub const FILTER: ModuleDescriptor = ModuleDescriptor {
	id: "filter",
	name: "Filter",
	accent: "#4c8dff",
	mode: Some("mode"),
	header: crate::FILTER_HEADER,
	visible: Some(crate::FILTER_VISIBLE),
	switches: &[],
	build_with_ir: None,
	build: filter,
};

/// The six-shaper distortion product.
pub const DISTORTION: ModuleDescriptor = ModuleDescriptor {
	id: "distortion",
	name: "Distortion",
	accent: "#ff9a3d",
	mode: Some("mode"),
	header: &[],
	visible: Some(crate::DISTORTION_VISIBLE),
	switches: &[],
	build_with_ir: None,
	build: distortion,
};

/// The compressor product, with its selectable detectors.
pub const COMPRESSOR: ModuleDescriptor = ModuleDescriptor {
	id: "compressor",
	name: "Compressor",
	accent: "#ff7a6b",
	mode: Some("enveloper"),
	header: &[],
	visible: Some(crate::COMPRESSOR_VISIBLE),
	switches: &[],
	build_with_ir: None,
	build: compressor,
};

/// The gain product.
pub const GAIN: ModuleDescriptor = ModuleDescriptor {
	id: "gain",
	name: "Gain",
	accent: "#35c77a",
	mode: None,
	header: &[],
	visible: None,
	switches: &[],
	build_with_ir: None,
	build: gain,
};

/// The delay product, with its tempo switch.
pub const DELAY: ModuleDescriptor = ModuleDescriptor {
	id: "delay",
	name: "Delay",
	accent: "#e8c35a",
	mode: None,
	header: crate::DELAY_HEADER,
	visible: None,
	switches: crate::DELAY_SWITCHES,
	build_with_ir: None,
	build: delay,
};

/// The crusher product.
pub const CRUSHER: ModuleDescriptor = ModuleDescriptor {
	id: "crusher",
	name: "Crusher",
	accent: "#8a6aff",
	mode: None,
	header: &[],
	visible: None,
	switches: &[],
	build_with_ir: None,
	build: crusher,
};

/// The chorus product, with its tempo switch.
pub const CHORUS: ModuleDescriptor = ModuleDescriptor {
	id: "chorus",
	name: "Chorus",
	accent: "#5fe0b0",
	mode: None,
	header: crate::CHORUS_HEADER,
	visible: None,
	switches: crate::CHORUS_SWITCHES,
	build_with_ir: None,
	build: chorus,
};

/// The phaser product, with its tempo switch.
pub const PHASER: ModuleDescriptor = ModuleDescriptor {
	id: "phaser",
	name: "Phaser",
	accent: "#b48cff",
	mode: None,
	header: crate::PHASER_HEADER,
	visible: None,
	switches: crate::PHASER_SWITCHES,
	build_with_ir: None,
	build: phaser,
};

/// The reverb product.
pub const REVERB: ModuleDescriptor = ModuleDescriptor {
	id: "reverb",
	name: "Reverb",
	accent: "#6c8cff",
	mode: None,
	header: &[],
	visible: None,
	switches: &[],
	build_with_ir: None,
	build: reverb,
};

/// The pitch product.
pub const PITCH: ModuleDescriptor = ModuleDescriptor {
	id: "pitch",
	name: "Pitch Shifter",
	accent: "#c8e04a",
	mode: Some("algorithm"),
	header: &["window_size"],
	visible: Some(PITCH_VISIBLE),
	switches: &[],
	build_with_ir: None,
	build: pitch,
};

/// The comb product.
pub const COMB: ModuleDescriptor = ModuleDescriptor {
	id: "comb",
	name: "Comb",
	accent: "#f0b35e",
	mode: None,
	header: &[],
	visible: None,
	switches: &[],
	build_with_ir: None,
	build: comb,
};

/// The allpass product.
pub const ALLPASS: ModuleDescriptor = ModuleDescriptor {
	id: "allpass",
	name: "Allpass",
	accent: "#b18cff",
	mode: None,
	header: &[],
	visible: None,
	switches: &[],
	build_with_ir: None,
	build: allpass,
};

/// The width product.
pub const WIDTH: ModuleDescriptor = ModuleDescriptor {
	id: "width",
	name: "Width",
	accent: "#ff6fae",
	mode: None,
	header: &[],
	visible: None,
	switches: &[],
	build_with_ir: None,
	build: width,
};

/// The wider product: a side synthesized from the mid.
pub const OUTER_WIDER: ModuleDescriptor = ModuleDescriptor {
	id: "outer_wider",
	name: "Wider",
	accent: "#ff6fae",
	mode: None,
	header: &[],
	visible: None,
	switches: &[],
	build_with_ir: None,
	build: outer_wider,
};

/// The shifter product.
pub const SHIFTER: ModuleDescriptor = ModuleDescriptor {
	id: "shifter",
	name: "Shifter",
	accent: "#7aa2ff",
	mode: Some("algorithm"),
	header: &["window_size"],
	visible: Some(crate::SHIFTER_VISIBLE),
	switches: &[],
	build_with_ir: None,
	build: shifter,
};

/// The ring modulator product.
pub const RING_MOD: ModuleDescriptor = ModuleDescriptor {
	id: "ring_mod",
	name: "Ring",
	accent: "#5ec8e0",
	mode: Some("mode"),
	header: &[],
	visible: Some(crate::RING_VISIBLE),
	switches: &[],
	build_with_ir: None,
	build: ring_mod,
};

/// The disperser product.
pub const DISPERSER: ModuleDescriptor = ModuleDescriptor {
	id: "disperser",
	name: "Disperser",
	accent: "#2fc6c0",
	mode: None,
	header: &[],
	visible: None,
	switches: &[],
	build_with_ir: None,
	build: disperser,
};

/// The balance product.
pub const BALANCE: ModuleDescriptor = ModuleDescriptor {
	id: "balance",
	name: "Balance",
	accent: "#d96aff",
	mode: None,
	header: &[],
	visible: None,
	switches: &[],
	build_with_ir: None,
	build: balance,
};

/// The particle wider product.
pub const PARTICLE_WIDER: ModuleDescriptor = ModuleDescriptor {
	id: "particle_wider",
	name: "Particle Wider",
	accent: "#67d7a8",
	mode: Some("fft_size"),
	header: &[],
	visible: None,
	switches: &[],
	build_with_ir: None,
	build: particle_wider,
};

/// The schroeder product.
pub const SCHROEDER: ModuleDescriptor = ModuleDescriptor {
	id: "schroeder",
	name: "Schroeder",
	accent: "#e0b36a",
	mode: None,
	header: &[],
	visible: None,
	switches: &[],
	build_with_ir: None,
	build: schroeder,
};

/// The flanger product.
pub const FLANGER: ModuleDescriptor = ModuleDescriptor {
	id: "flanger",
	name: "Flanger",
	accent: "#6bb8ff",
	mode: None,
	header: crate::FLANGER_HEADER,
	visible: None,
	switches: crate::FLANGER_SWITCHES,
	build_with_ir: None,
	build: flanger,
};

/// The eq product.
pub const EQ: ModuleDescriptor = ModuleDescriptor {
	id: "eq",
	name: "EQ",
	accent: "#8a7bff",
	mode: Some("band"),
	header: &[],
	visible: Some(crate::EQ_VISIBLE),
	switches: &[],
	build_with_ir: None,
	build: eq,
};

/// The tape stop product.
pub const TAPE_STOP: ModuleDescriptor = ModuleDescriptor {
	id: "tape_stop",
	name: "Tape Stop",
	accent: "#7ad1e0",
	mode: None,
	header: &[],
	visible: None,
	switches: &[],
	build_with_ir: None,
	build: tape_stop,
};

/// The gate product.
pub const GATE: ModuleDescriptor = ModuleDescriptor {
	id: "gate",
	name: "Gate",
	accent: "#6fd3a3",
	mode: None,
	header: &[],
	visible: None,
	switches: &[],
	build_with_ir: None,
	build: gate,
};

/// The expander product.
pub const EXPANDER: ModuleDescriptor = ModuleDescriptor {
	id: "expander",
	name: "Expander",
	accent: "#79b8ff",
	mode: None,
	header: &[],
	visible: None,
	switches: &[],
	build_with_ir: None,
	build: expander,
};

/// The limiter product.
pub const LIMITER: ModuleDescriptor = ModuleDescriptor {
	id: "limiter",
	name: "Limiter",
	accent: "#ffc45a",
	mode: None,
	header: &[],
	visible: None,
	switches: &[],
	build_with_ir: None,
	build: limiter,
};

/// The formant filter product.
pub const FORMANT: ModuleDescriptor = ModuleDescriptor {
	id: "formant_filter",
	name: "Formant Filter",
	accent: "#ff9f6e",
	mode: None,
	header: &[],
	visible: None,
	switches: &[],
	build_with_ir: None,
	build: formant,
};

/// The transient product.
pub const TRANSIENT: ModuleDescriptor = ModuleDescriptor {
	id: "transient",
	name: "Transient",
	accent: "#ff7ad9",
	mode: None,
	header: &[],
	visible: None,
	switches: &[],
	build_with_ir: None,
	build: transient,
};

/// The router product.
pub const ROUTER: ModuleDescriptor = ModuleDescriptor {
	id: "router",
	name: "Router",
	accent: "#8ad3ff",
	mode: None,
	header: &[],
	visible: None,
	switches: &[],
	build_with_ir: None,
	build: router,
};

/// The autotune product, which follows the notes the host plays while its MIDI
/// switch is on, and its faceplate mask while it is off.
pub const AUTOTUNE: ModuleDescriptor = ModuleDescriptor {
	id: "autotune",
	name: "Autotune",
	accent: "#ff6fae",
	mode: Some("algorithm"),
	header: &["window_size", "midi"],
	visible: Some(crate::AUTOTUNE_VISIBLE),
	switches: &[],
	build_with_ir: None,
	build: autotune,
};

/// The spectral retune product.
pub const SPECTRAL_RETUNE: ModuleDescriptor = ModuleDescriptor {
	id: "spectral_retune",
	name: "Spectral Retune",
	accent: "#5fd0c8",
	mode: None,
	header: &["window_size", "midi"],
	visible: None,
	switches: &[],
	build_with_ir: None,
	build: spectral_retune,
};

/// The resonator product, which rings at the notes the host plays, or at the
/// Note knob when the MIDI switch in its header is off.
pub const RESONATOR: ModuleDescriptor = ModuleDescriptor {
	id: "resonator",
	name: "Resonator",
	accent: "#b98cff",
	mode: None,
	header: &["midi"],
	visible: None,
	switches: &[],
	build_with_ir: None,
	build: resonator,
};

/// The unstablizer product.
pub const UNSTABLIZER: ModuleDescriptor = ModuleDescriptor {
	id: "unstablizer",
	name: "Unstablizer",
	accent: "#ff8f6b",
	mode: Some("window_size"),
	header: &["midi"],
	visible: None,
	switches: &[],
	build_with_ir: None,
	build: unstablizer,
};
/// The IR reverb product, which convolves with a generated response.
pub const IR_REVERB: ModuleDescriptor = ModuleDescriptor {
	id: "ir_reverb",
	name: "IR Reverb",
	accent: "#7bd3ff",
	mode: None,
	header: &[],
	visible: None,
	switches: &[],
	build_with_ir: None,
	build: ir_reverb,
};

/// The spectral dynamics product.
pub const SPECTRAL_DYNAMICS: ModuleDescriptor = ModuleDescriptor {
	id: "spectral_dynamics",
	name: "Spectral Dynamics",
	accent: "#8fd44a",
	mode: None,
	header: &[],
	visible: None,
	switches: &[],
	build_with_ir: None,
	build: spectral_dynamics,
};

/// The STFT vocoder product.
pub const STFT_VOCODER: ModuleDescriptor = ModuleDescriptor {
	id: "stft_vocoder",
	name: "STFT Vocoder",
	accent: "#7bb8ff",
	mode: None,
	header: &["window_size"],
	visible: None,
	switches: &[],
	build_with_ir: None,
	build: stft_vocoder,
};

/// The OTT product.
pub const OTT: ModuleDescriptor = ModuleDescriptor {
	id: "ott",
	name: "OTT",
	accent: "#ffb454",
	mode: None,
	header: &[],
	visible: None,
	switches: &[],
	build_with_ir: None,
	build: ott,
};

/// The vocoder product.
pub const VOCODER: ModuleDescriptor = ModuleDescriptor {
	id: "vocoder",
	name: "Vocoder",
	accent: "#e0a45a",
	mode: None,
	header: &[],
	visible: None,
	switches: &[],
	build_with_ir: None,
	build: vocoder,
};

/// The convolver product, whose impulse response the page loads from a file.
pub const CONVOLVER: ModuleDescriptor = ModuleDescriptor {
	id: "convolver",
	name: "Convolver",
	accent: "#b08cff",
	mode: None,
	header: &[],
	visible: None,
	switches: &[],
	build: |rate| Box::new(ConvolverEffect::new(rate, Arc::new(IrSlot::new()))),
	build_with_ir: Some(|rate, slot| Box::new(ConvolverEffect::new(rate, slot))),
};

/// The Filter Table product.
pub const FILTER_TABLE: ModuleDescriptor = ModuleDescriptor {
	id: "filter_table",
	name: "Filter Table",
	accent: "#c8e04a",
	mode: None,
	header: &[],
	visible: None,
	switches: &[],
	build: |rate| Box::new(FilterTableEffect::with_frame(rate, FilterFrame::new())),
	build_with_ir: None,
};

/// The history time-remap product.
pub const HISTORY: ModuleDescriptor = ModuleDescriptor {
	id: "history",
	name: "History",
	accent: "#6ee7ff",
	mode: None,
	header: &[],
	visible: None,
	switches: &[],
	build: |rate| Box::new(HistoryEffect::with_feed(rate, HistoryFeed::new())),
	build_with_ir: None,
};

/// The scripted distortion product.
pub const SCRIPT_DISTORTION: ModuleDescriptor = ModuleDescriptor {
	id: "script_distortion",
	name: "Script Distortion",
	accent: "#ff7a6b",
	mode: None,
	header: &[],
	visible: None,
	switches: &[],
	build: |_| Box::new(ScriptDistortionEffect::new(distortion_worker())),
	build_with_ir: None,
};

/// The scripted frequency mapper product.
pub const SCRIPT_MAPPER: ModuleDescriptor = ModuleDescriptor {
	id: "script_mapper",
	name: "Script Mapper",
	accent: "#8ad3ff",
	mode: None,
	header: &[],
	visible: None,
	switches: &[],
	build: |rate| Box::new(ScriptMapperEffect::new(rate, mapper_worker())),
	build_with_ir: None,
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
	RING_MOD,
	WIDTH,
	OUTER_WIDER,
	BALANCE,
	ROUTER,
	TAPE_STOP,
	TRANSIENT,
	FORMANT,
	EQ,
	HISTORY,
	FILTER_TABLE,
	REVERB,
	SCHROEDER,
	PARTICLE_WIDER,
	PITCH,
	SHIFTER,
	AUTOTUNE,
	SPECTRAL_RETUNE,
	SPECTRAL_DYNAMICS,
	RESONATOR,
	UNSTABLIZER,
	IR_REVERB,
	STFT_VOCODER,
	OTT,
	VOCODER,
	CONVOLVER,
	SCRIPT_DISTORTION,
	SCRIPT_MAPPER,
];

/// Finds a descriptor by its stable id.
pub fn get(id: &str) -> Option<&'static ModuleDescriptor> {
	MODULES.iter().find(|module| module.id == id)
}
