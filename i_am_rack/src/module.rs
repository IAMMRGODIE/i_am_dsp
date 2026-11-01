//! The module types the rack can hold, and the card metadata each one brings.
//!
//! Every module is the same effect the standalone product ships: Multimode for
//! Filter, Distortion for Drive, Compressor for Compressor, Gain for Gain. The
//! card a module shows is the product card, so the mode/header/visible rules
//! travel with the kind.

use std::collections::BTreeMap;

use i_am_dsp::prelude::{
	Compressor, Crusher, Distortion, Effect, Gain, Multimode, Parameter, PureDelay,
	RectifyEnvelope, Unit, Value,
};

/// The controls each Multimode algorithm reads; everything else is hidden.
pub const FILTER_VISIBLE: &[(i32, &[&str])] = &[
	(0, &["cutoff", "q", "stages"]),
	(1, &["cutoff", "q", "stages"]),
	(2, &["cutoff", "q", "stages"]),
	(3, &["cutoff", "q", "stages"]),
	(4, &["cutoff", "gain_db"]),
	(5, &["cutoff", "gain_db", "slope"]),
	(6, &["cutoff", "gain_db", "slope"]),
];

/// The controls each shaper reads.
pub const DISTORTION_VISIBLE: &[(i32, &[&str])] = &[
	(0, &["threshold", "gain"]),
	(1, &["alpha", "gain"]),
	(2, &["k", "gain"]),
	(3, &["a", "p", "gain"]),
	(4, &["bits", "gain"]),
	(5, &["target_sample_rate", "gain"]),
];

/// The controls a compressor card shows; the library's sub-parameters are
/// trimmed down to the ones a person turns.
pub const COMPRESSOR_VISIBLE: &[&str] = &[
	"threshold",
	"ratio",
	"smoother.attack_time",
	"smoother.release_time",
	"lookahead",
	"gain_linear",
];

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
	/// A peak/RMS compressor.
	Compressor,
	/// A plain gain.
	Gain,
	/// A pure delay.
	Delay,
	/// A disperser, saturator and amplitude modulation in one.
	Crusher,
}

impl ModuleKind {
	/// Every kind the rack knows, in menu order.
	pub const ALL: [Self; 6] = [
		Self::Filter,
		Self::Distortion,
		Self::Compressor,
		Self::Gain,
		Self::Delay,
		Self::Crusher,
	];

	/// The name a card shows.
	pub const fn name(self) -> &'static str {
		match self {
			Self::Filter => "Filter",
			Self::Distortion => "Distortion",
			Self::Compressor => "Compressor",
			Self::Gain => "Gain",
			Self::Delay => "Delay",
			Self::Crusher => "Crusher",
		}
	}

	/// The identity colour the product uses.
	pub const fn accent(self) -> &'static str {
		match self {
			Self::Filter => "#4c8dff",
			Self::Distortion => "#ff9a3d",
			Self::Compressor => "#ef5b5b",
			Self::Gain => "#35c77a",
			Self::Delay => "#c77dff",
			Self::Crusher => "#ff5f9e",
		}
	}

	/// The picture a card draws above its controls.
	pub const fn hero(self) -> &'static str {
		match self {
			Self::Filter => "spectrum",
			Self::Distortion => "transfer",
			Self::Compressor => "compressor",
			Self::Gain => "waveform",
			Self::Delay => "waveform",
			Self::Crusher => "transfer",
		}
	}

	/// The parameter the page draws as a mode switch in the header.
	pub const fn mode(self) -> Option<&'static str> {
		match self {
			Self::Filter | Self::Distortion => Some("mode"),
			Self::Compressor | Self::Gain | Self::Delay | Self::Crusher => None,
		}
	}

	/// The parameters the page draws in the header instead of the body.
	pub const fn header(self) -> &'static [&'static str] {
		match self {
			Self::Filter => &["stages"],
			_ => &[],
		}
	}

	/// The controls each mode reads, when the card hides the rest.
	pub const fn visible(self) -> Option<&'static [(i32, &'static [&'static str])]> {
		match self {
			Self::Filter => Some(FILTER_VISIBLE),
			Self::Distortion => Some(DISTORTION_VISIBLE),
			Self::Compressor | Self::Gain | Self::Delay | Self::Crusher => None,
		}
	}
}

/// Builds a fresh effect for a kind at a sample rate.
pub(crate) fn build(kind: ModuleKind, sample_rate: usize) -> Box<dyn Effect<2>> {
	match kind {
		ModuleKind::Filter => Box::new(Multimode::new(sample_rate, 1_000.0, 0.707)),
		ModuleKind::Distortion => Box::new(Distortion::<2>::new(sample_rate)),
		ModuleKind::Compressor => Box::new(Compressor::<RectifyEnvelope<2>, 2>::new(
			RectifyEnvelope::<2>::new(sample_rate, 30.0),
			sample_rate,
			10.0,
			100.0,
			-24.0,
			4.0,
		)),
		ModuleKind::Gain => Box::new(Gain::new(1.0)),
		ModuleKind::Delay => Box::new(PureDelay::<2>::new(
			sample_rate * 2 + 8,
			250.0,
			sample_rate,
		)),
		ModuleKind::Crusher => Box::new(Crusher::<2>::new(sample_rate)),
	}
}

/// The parameter schema of a kind, at a rate that does not affect the metadata.
pub(crate) fn schema(kind: ModuleKind) -> Vec<Parameter> {
	let mut parameters = build(kind, 44_100).get_parameters();

	if kind == ModuleKind::Delay {
		for parameter in &mut parameters {
			match parameter.identifier.as_str() {
				"delay_time" => {
					parameter.meta.display_name = Some("Delay".into());

					if let Value::Float { range, .. } = &mut parameter.value {
						*range = 0.0..=2_000.0;
					}
				},
				"tempo.sync" => parameter.meta.display_name = Some("Sync".into()),
				"tempo.beats" => parameter.meta.display_name = Some("Time".into()),
				_ => {},
			}
		}

		parameters
	} else if kind == ModuleKind::Compressor {
		for parameter in &mut parameters {
			match parameter.identifier.as_str() {
				"threshold" => parameter.meta.display_name = Some("Threshold".into()),
				"ratio" => parameter.meta.display_name = Some("Ratio".into()),
				"smoother.attack_time" => parameter.meta.display_name = Some("Attack".into()),
				"smoother.release_time" => parameter.meta.display_name = Some("Release".into()),
				"lookahead" => {
					parameter.meta.display_name = Some("Lookahead".into());
					parameter.meta.unit = Unit::Samples;
				},
				"gain_linear" => parameter.meta.display_name = Some("Makeup".into()),
				_ => {},
			}
		}

		parameters
			.into_iter()
			.filter(|parameter| COMPRESSOR_VISIBLE.contains(&parameter.identifier.as_str()))
			.collect()
	} else {
		parameters
	}
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
