//! The CLAP and VST3 entry points of the web-faceplate products.
//!
//! Everything a product *is* lives in `i_am_effects`: the DSP, the web faceplate,
//! and the `Plugin` implementation. This crate adds only what makes the bundle
//! loadable — `clap_entry` and, with the `vst3` feature, the VST3 entry points.
//!
//! It is a crate of its own because `clap_entry` is a global symbol: a library that
//! exported one could not also be linked into another plug-in, which is exactly
//! what the rack needs to do to reuse these effects.
//!
//! ```text
//! cargo build -p i_am_effects_bundle --features vst3
//! cargo bundle -p i_am_effects_bundle --features vst3
//! ```

pub use i_am_effects::*;

use i_am_plugin::export_plugin;

export_plugin!(
	GainWeb,
	DistortionWeb,
	FilterWeb,
	CombWeb,
	AllpassWeb,
	WidthWeb,
	PitchWeb,
	AutoTuneWeb,
	SpectralRetuneWeb,
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
	CrusherWeb,
	ConvolverWeb,
	ScriptDistortionWeb,
	ScriptMapperWeb,
	OttWeb,
	VocoderWeb,
	StftVocoderWeb,
	TapeStopWeb,
	GateWeb,
	ExpanderWeb,
	LimiterWeb,
	ResonatorWeb,
	FormantFilterWeb,
	TransientWeb,
	RouterWeb,
	EqWeb,
	HistoryWeb,
	FilterTableWeb,
	SpectralDynamicsWeb
);
