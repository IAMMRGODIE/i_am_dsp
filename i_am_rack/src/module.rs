//! The module types M0 can put in a rack.
//!
//! Every module is an Effect<2> from i_am_dsp, erased behind a Box, so the rack
//! engine can hold a different type in each slot without knowing which one it is.
//! The parameter schema comes from the effect itself: the page gets the same
//! control JSON a faceplate would.

use std::collections::BTreeMap;

use i_am_dsp::prelude::{
	Effect, Gain, Lowpass, Parameter, ParameterMeta, Parameters, ProcessContext,
	SetValue, SoftClipper, Unit, Value,
};

/// A module type a rack slot can hold.
#[derive(
	Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash,
	serde::Serialize, serde::Deserialize,
)]
pub enum ModuleKind {
	/// A resonant lowpass.
	Filter,
	/// A soft clipper.
	Drive,
	/// A plain gain.
	Gain,
}

impl ModuleKind {
	/// Every kind M0 knows, in menu order.
	pub const ALL: [Self; 3] = [Self::Filter, Self::Drive, Self::Gain];

	/// The name a card shows.
	pub const fn name(self) -> &'static str {
		match self {
			Self::Filter => "Filter",
			Self::Drive => "Drive",
			Self::Gain => "Gain",
		}
	}
}

/// A soft clipper with the labels a person reads, instead of the field names.
///
/// The DSP crate's SoftClipper calls its controls alpha and gain, which is right
/// for a struct and wrong for a card. The module keeps the two floats and hands
/// the clipper the values on every block; the extra copy is two floats.
struct DriveModule {
	amount: f32,
	output: f32,
	inner: SoftClipper,
}

impl DriveModule {
	fn new() -> Self {
		Self {
			amount: 0.5,
			output: 1.0,
			inner: SoftClipper::new(0.5, 1.0),
		}
	}

	fn apply(&mut self) {
		self.inner.alpha = self.amount.clamp(0.01, 1.0);
		self.inner.gain = self.output.clamp(0.01, 4.0);
	}

	fn parameter(
		identifier: &'static str,
		label: &'static str,
		value: f32,
		range: std::ops::RangeInclusive<f32>,
		default: f32,
		logarithmic: bool,
	) -> Parameter {
		Parameter::new(
			identifier,
			Value::Float {
				value,
				range,
				logarithmic,
			},
		)
		.with_meta(ParameterMeta {
			default: Some(default as f64),
			unit: Unit::Percent,
			display_name: Some(label.into()),
			..Default::default()
		})
	}
}

impl Parameters for DriveModule {
	fn get_parameters(&self) -> Vec<Parameter> {
		vec![
			Self::parameter("amount", "Amount", self.amount, 0.01..=1.0, 0.5, false),
			Self::parameter("output", "Output", self.output, 0.01..=4.0, 1.0, true),
		]
	}

	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		match (identifier, value) {
			("amount", SetValue::Float(value)) => {
				self.amount = value.clamp(0.01, 1.0);
				self.apply();
				true
			},
			("output", SetValue::Float(value)) => {
				self.output = value.clamp(0.01, 4.0);
				self.apply();
				true
			},
			_ => false,
		}
	}
}

impl Effect<2> for DriveModule {
	fn process(
		&mut self,
		samples: &mut [f32; 2],
		other: &[&[f32; 2]],
		process_context: &mut Box<dyn ProcessContext>,
	) {
		self.apply();
		self.inner.process(samples, other, process_context);
	}

	fn delay(&self) -> usize {
		0
	}
}

/// Builds a fresh effect for a kind at a sample rate.
pub(crate) fn build(kind: ModuleKind, sample_rate: usize) -> Box<dyn Effect<2>> {
	match kind {
		ModuleKind::Filter => Box::new(Lowpass::<2>::new(sample_rate, 1_000.0, 0.707)),
		ModuleKind::Drive => Box::new(DriveModule::new()),
		ModuleKind::Gain => Box::new(Gain::new(1.0)),
	}
}

/// The parameter schema of a kind, at a rate that does not affect the metadata.
pub(crate) fn schema(kind: ModuleKind) -> Vec<Parameter> {
	build(kind, 44_100).get_parameters()
}

/// The float parameters a fresh module starts with.
///
/// Integer choices and booleans do not belong in the M0 state: every module in
/// this set is controlled by floats only. A kernel that has a mode would store it
/// under the same map with its own value type.
pub(crate) fn default_params(kind: ModuleKind) -> BTreeMap<String, f32> {
	schema(kind)
		.into_iter()
		.filter_map(|parameter| match parameter.value {
			Value::Float { value, .. } => Some((parameter.identifier, value)),
			_ => None,
		})
		.collect()
}
