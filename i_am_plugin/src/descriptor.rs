//! descriptor

use crate::shared::*;

/// A struct to hold a plugin's audio port.
pub struct AudioPort {
	/// The name of the audio port.
	pub name: &'static str,
}

/// A struct to hold a plugin's MIDI port.
pub struct MidiPort {
	/// The name of the Midi port.
	pub name: &'static str,
}

/// The plugin's descriptor.
pub struct Descriptor {
	pub id: &'static str,
	pub name: &'static str,
	pub vendor: Option<&'static str>,
	pub url: Option<&'static str>,
	pub manual_url: Option<&'static str>,
	pub support_url: Option<&'static str>,
	pub version: Option<&'static str>,
	pub description: Option<&'static str>,
	pub tags: &'static [Tag],
}

/// A tag for the plugin.
pub enum Tag {
	Instrument,
	AudioEffect,
	NoteEffect,
	Analyzer,
	Synthesizer,
	Sampler,
	Drum,
	DrumMachine,
	Filter,
	Phaser,
	Equalizer,
	Deesser,
	PhaseVocoder,
	Granular,
	FrequencyShifter,
	PitchShifter,
	Distortion,
	TransientShaper,
	Compressor,
	Limiter,
}

impl Tag {
	pub(crate) fn as_cstr(&self) -> &CStr {
		match self {
			Self::Instrument => clack_plugin::plugin::features::INSTRUMENT,
			Self::AudioEffect => clack_plugin::plugin::features::AUDIO_EFFECT,
			Self::NoteEffect => clack_plugin::plugin::features::NOTE_EFFECT,
			Self::Analyzer => clack_plugin::plugin::features::ANALYZER,
			Self::Synthesizer => clack_plugin::plugin::features::SYNTHESIZER,
			Self::Sampler => clack_plugin::plugin::features::SAMPLER,
			Self::Drum => clack_plugin::plugin::features::DRUM,
			Self::DrumMachine => clack_plugin::plugin::features::DRUM_MACHINE,
			Self::Filter => clack_plugin::plugin::features::FILTER,
			Self::Phaser => clack_plugin::plugin::features::PHASER,
			Self::Equalizer => clack_plugin::plugin::features::EQUALIZER,
			Self::Deesser => clack_plugin::plugin::features::DEESSER,
			Self::PhaseVocoder => clack_plugin::plugin::features::PHASE_VOCODER,
			Self::Granular => clack_plugin::plugin::features::GRANULAR,
			Self::FrequencyShifter => clack_plugin::plugin::features::FREQUENCY_SHIFTER,
			Self::PitchShifter => clack_plugin::plugin::features::PITCH_SHIFTER,
			Self::Distortion => clack_plugin::plugin::features::DISTORTION,
			Self::TransientShaper => clack_plugin::plugin::features::TRANSIENT_SHAPER,
			Self::Compressor => clack_plugin::plugin::features::COMPRESSOR,
			Self::Limiter => clack_plugin::plugin::features::LIMITER,
		}
	}
}

impl Descriptor {
	/// Create a new descriptor with the given values.
	pub const fn new(
		id: &'static str,
		name: &'static str,
	) -> Self {
		Self {
			id,
			name,
			vendor: None,
			url: None,
			manual_url: None,
			support_url: None,
			version: None,
			description: None,
			tags: &[],
		}
	}

	/// Set the vendor of the plugin.
	pub const fn with_vendor(
		self,
		vendor: &'static str,
	) -> Self {
		Self {
			vendor: Some(vendor),
			..self
		}
	}

	/// Set the URL of the plugin's website.
	pub const fn with_url(
		self,
		url: &'static str,
	) -> Self {
		Self {
			url: Some(url),
			..self
		}
	}

	/// Set the URL of the plugin's manual.
	pub const fn with_manual_url(
		self,
		manual_url: &'static str,
	) -> Self {
		Self {
			manual_url: Some(manual_url),
			..self
		}
	}

	/// Set the URL of the plugin's support.
	pub const fn with_support_url(
		self,
		support_url: &'static str,
	) -> Self {
		Self {
			support_url: Some(support_url),
			..self
		}
	}

	/// Set the version of the plugin.
	pub const fn with_version(
		self,
		version: &'static str,
	) -> Self {
		Self {
			version: Some(version),
			..self
		}
	}

	/// Set the description of the plugin.
	pub const fn with_description(
		self,
		description: &'static str,
	) -> Self {
		Self {
			description: Some(description),
			..self
		}
	}

	/// Set the tags of the plugin.
	pub const fn with_tags(
		self,
		tags: &'static [Tag],
	) -> Self {
		Self {
			tags,
			..self
		}
	}
}
