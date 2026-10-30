//! The module types the rack can hold, and the card metadata each one brings.
//!
//! Every module is the same effect the standalone product ships: Multimode for
//! Filter, Distortion for Drive, Gain for Gain. The card a module shows is the
//! product card, so the mode/header/visible rules travel with the kind.

use std::collections::BTreeMap;

use i_am_dsp::prelude::{Distortion, Effect, Gain, Multimode, Parameter, Value};

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
	/// A plain gain.
	Gain,
}

impl ModuleKind {
	/// Every kind the rack knows, in menu order.
	pub const ALL: [Self; 3] = [Self::Filter, Self::Distortion, Self::Gain];

	/// The name a card shows.
	pub const fn name(self) -> &'static str {
		match self {
			Self::Filter => "Filter",
			Self::Distortion => "Distortion",
			Self::Gain => "Gain",
		}
	}

	/// The identity colour the product uses.
	pub const fn accent(self) -> &'static str {
		match self {
			Self::Filter => "#4c8dff",
			Self::Distortion => "#ff9a3d",
			Self::Gain => "#35c77a",
		}
	}

	/// The picture a card draws above its controls.
	pub const fn hero(self) -> &'static str {
		match self {
			Self::Filter => "spectrum",
			Self::Distortion => "transfer",
			Self::Gain => "waveform",
		}
	}

	/// The parameter the page draws as a mode switch in the header.
	pub const fn mode(self) -> Option<&'static str> {
		match self {
			Self::Filter | Self::Distortion => Some("mode"),
			Self::Gain => None,
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
			Self::Gain => None,
		}
	}
}

/// Builds a fresh effect for a kind at a sample rate.
pub(crate) fn build(kind: ModuleKind, sample_rate: usize) -> Box<dyn Effect<2>> {
	match kind {
		ModuleKind::Filter => Box::new(Multimode::new(sample_rate, 1_000.0, 0.707)),
		ModuleKind::Distortion => Box::new(Distortion::<2>::new(sample_rate)),
		ModuleKind::Gain => Box::new(Gain::new(1.0)),
	}
}

/// The parameter schema of a kind, at a rate that does not affect the metadata.
pub(crate) fn schema(kind: ModuleKind) -> Vec<Parameter> {
	build(kind, 44_100).get_parameters()
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
