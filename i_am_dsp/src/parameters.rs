//! A helper trait for report parameters change to hosts.

use std::{ops::{Index, RangeInclusive}, sync::{Arc, atomic::Ordering}};

use bimap::BiMap;
use ciborium::{from_reader, into_writer};
// use crossbeam_queue::SegQueue;
use portable_atomic::{AtomicBool, AtomicF32, AtomicI32};
use rustfft::num_complex::Complex;

use crate::{Effect, Generator};

/// How a parameter wants its value shown to a person.
///
/// A host that only knows a number can do little better than printing it, which is how a
/// synced time ends up reading `0.00` where a musician expects `1/8`. A parameter declares its
/// unit and the host can ask for text; see [`ParameterMeta::default`] for the other half, the
/// value a parameter starts at.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum Unit {
	/// A plain number.
	#[default]
	None,
	/// A length in milliseconds.
	Milliseconds,
	/// A frequency in hertz.
	Hertz,
	/// A length in quarter-note beats: `0.5` is an eighth note, `0.75` a dotted eighth.
	Beats,
	/// A length in seconds.
	Seconds,
	/// A count of audio samples.
	Samples,
	/// A level in decibels.
	Decibels,
	/// A fraction of one, shown as a percentage: `0.5` reads as `50.0 %`.
	Percent,
}

impl Unit {
	/// Renders a value the way this unit wants to be read.
	pub fn format(self, value: f64) -> String {
		match self {
			Unit::None => format!("{value:.2}"),
			Unit::Milliseconds => format!("{value:.2} ms"),
			Unit::Hertz => format!("{value:.2} Hz"),
			Unit::Beats => format_beats(value),
			Unit::Seconds => format!("{value:.3} s"),
			Unit::Samples => format!("{value:.0} samples"),
			Unit::Decibels => format!("{value:.2} dB"),
			// The value is a fraction, so half of it reads as fifty percent.
			Unit::Percent => format!("{:.1} %", value * 100.0),
		}
	}

	/// Reads back what [`Self::format`] produced, for a host that lets the text be edited.
	///
	/// Returns [`None`] when the text is not this unit's business, so a caller can fall back
	/// to reading a plain number.
	pub fn parse(self, text: &str) -> Option<f64> {
		let text = text.trim();

		match self {
			Unit::None => text.parse().ok(),
			Unit::Milliseconds => Some(text.trim_end_matches("ms").trim().parse().ok()?),
			Unit::Hertz => Some(text.trim_end_matches("Hz").trim().parse().ok()?),
			Unit::Beats => parse_beats(text),
			Unit::Seconds => Some(text.trim_end_matches('s').trim().parse().ok()?),
			Unit::Samples => Some(text.trim_end_matches("samples").trim().parse().ok()?),
			Unit::Decibels => Some(text.trim_end_matches("dB").trim().parse().ok()?),
			Unit::Percent => Some(text.trim_end_matches('%').trim().parse::<f64>().ok()? / 100.0),
		}
	}
}

/// Renders a length in quarter-note beats as note values, which is how a musician reads one.
///
/// Anything that is not a note value (or a dotted or triplet one) falls back to a beat count,
/// because `3.7 beats` is more use than a wrong fraction.
fn format_beats(beats: f64) -> String {
	// A whole note is four quarter-note beats, so a note value is `4 / beats`.
	for (multiplier, suffix) in [(1.0, ""), (1.5, " dotted"), (2.0 / 3.0, " triplet")] {
		let note = beats / multiplier;
		if note > 0.0 {
			let denominator = (4.0 / note).round();
			// A power-of-two denominator is a note value; 3, 5, 7 and the rest are not.
			if (1.0..=128.0).contains(&denominator) {
				let denominator = denominator as u32;
				if denominator.is_power_of_two()
					&& (4.0 / denominator as f64 - note).abs() < 1e-4
				{
					return format!("1/{}{}", denominator, suffix);
				}
			}
		}
	}

	format!("{beats:.2} beats")
}

/// Reads the note values [`format_beats`] writes, or a plain beat count.
fn parse_beats(text: &str) -> Option<f64> {
	let (text, multiplier) = if let Some(text) = text.strip_suffix("dotted") {
		(text, 1.5)
	} else if let Some(text) = text.strip_suffix("triplet") {
		(text, 2.0 / 3.0)
	} else {
		(text, 1.0)
	};
	let text = text.trim().trim_end_matches("beats").trim();

	if let Some((numerator, denominator)) = text.split_once('/') {
		let numerator: f64 = numerator.trim().parse().ok()?;
		let denominator: f64 = denominator.trim().parse().ok()?;
		if denominator == 0.0 {
			return None;
		}
		// `1/8` is an eighth note, which is half a quarter-note beat.
		return Some(4.0 * numerator / denominator * multiplier);
	}

	text.parse::<f64>().ok().map(|beats| beats * multiplier)
}

/// What a host needs to present a parameter beyond its value and its range.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ParameterMeta {
	/// The value the parameter starts at, for a host that resets parameters to a default.
	///
	/// **Declare one wherever there is an answer to give.** A host that is not told falls
	/// back to the bottom of the range, which is rarely where an effect starts: a synced time
	/// would come back as a thirty-second note rather than a beat, and a gain at its quietest
	/// rather than at unity. It is only right to leave it as [`None`] when there is no single
	/// answer — a parameter whose starting value is an argument to the constructor, like a
	/// delay time or a sample rate. See the `#[default(..)]` attribute for declaring one.
	pub default: Option<f64>,
	/// How to render the value as text.
	pub unit: Unit,
	/// What a host should call the parameter, when the identifier is not what a person should
	/// read: `#[id(display = "BPM Sync")]`.
	///
	/// A nested parameter is named by joining the names of the levels that contain it with a
	/// space, so a `Beats` field under a `Tempo` sub-parameter shows to a host as
	/// `Tempo Beats`. A level that declares nothing contributes its field name.
	pub display_name: Option<std::borrow::Cow<'static, str>>,
}

/// A helper struct for report parameters.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Parameter {
	/// An unique identifier for the parameter.
	pub identifier: String,
	/// The parameter value.
	pub value: Value,
	/// How to present it. Skipped when saving state: it comes from the code, not from the
	/// project, so a project saved by an older build simply takes the current one.
	#[serde(skip)]
	pub meta: ParameterMeta,
}

impl Parameter {
	/// A parameter with no metadata, for a hand-written [`Parameters`] implementation.
	pub fn new(identifier: impl Into<String>, value: Value) -> Self {
		Self {
			identifier: identifier.into(),
			value,
			meta: ParameterMeta::default(),
		}
	}

	/// The same parameter, with the metadata a host presents it by.
	pub fn with_meta(mut self, meta: ParameterMeta) -> Self {
		self.meta = meta;
		self
	}
}

#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
/// A helper enum for parameter values, but its atomic.
/// 
/// Note: we don't support something like [`Value::Serialized`] here, 
/// since we need a lock to update the value, and may be expensive to update.
pub enum AtomicValue {
	/// A floating-point value with a range.
	Float {
		/// The actual value.
		value: AtomicF32,
		/// The range of valid values.
		range: RangeInclusive<f32>,
		/// Whether the value is logarithmic.
		logarithmic: bool,
		/// Whether the value has been changed.
		changed: AtomicBool,
	},
	/// An integer value with a range.
	Int {
		/// The actual value.
		value: AtomicI32,
		/// The range of valid values.
		range: RangeInclusive<i32>,
		/// Whether the value is logarithmic.
		logarithmic: bool,
		/// Whether the value has been changed.
		changed: AtomicBool,
	},
	/// A boolean value.
	Bool{
		/// The actual value.
		value: AtomicBool,
		/// Whether the value has been changed.
		changed: AtomicBool,
	},
	/// A placeholder value, used for parameters that are not yet implemented or is None.
	#[default] Nothing,
}

impl AtomicValue {
	/// Loads the value with the given ordering.
	/// 
	/// # Panics
	/// 
	/// Panics if order is [`Ordering::Release`] or [`Ordering::AcqRel`].
	pub fn load(&self, order: Ordering) -> SetValue {
		match self {
			Self::Bool{ value, ..} => SetValue::Bool(value.load(order)),
			Self::Float { value, .. } => SetValue::Float(value.load(order)),
			Self::Int { value, .. } => SetValue::Int(value.load(order)),
			Self::Nothing => SetValue::Nothing,
		}
	}

	/// Returns true if the value is a float.
	pub fn is_float(&self) -> bool {
		matches!(self, Self::Float { .. })
	}

	/// Returns true if the value is an int.
	pub fn is_int(&self) -> bool {
		matches!(self, Self::Int { .. })
	}

	/// Returns true if the value is a bool.
	pub fn is_bool(&self) -> bool {
		matches!(self, Self::Bool { .. })
	}

	/// Returns true if the value is nothing.
	pub fn is_nothing(&self) -> bool {
		matches!(self, Self::Nothing)
	}

	/// Stores the value with the given ordering
	/// 
	/// Returns true if the value is updated, false for value type mismatch.
	/// 
	/// Will clamp the value to the range if it is out of range.
	/// 
	/// # Panics
	/// 
	/// Panics if order is [`Ordering::Release`] or [`Ordering::AcqRel`].
	pub fn store(&self, value: impl Into<SetValue>, order: Ordering) -> bool {
		let value = value.into();
		match (self, value) {
			(Self::Bool{ value: v, changed }, SetValue::Bool(value)) => {
				v.store(value, order);
				changed.store(true, order);
				true
			},
			(Self::Float { value: v, range, changed, .. }, SetValue::Float(value)) => {
				let value = value.clamp(*range.start(), *range.end());
				changed.store(true, order);
				v.store(value, order);
				true
			},
			(Self::Int { value: v, range, changed, .. }, SetValue::Int(value)) => {
				let value = value.clamp(*range.start(), *range.end());
				v.store(value, order);
				changed.store(true, order);
				true
			},
			(Self::Nothing, SetValue::Nothing) => true,
			_ => false,
		}
	}

	/// Atomically updates the value with the given ordering and a closure.
	/// 
	/// Returns the old value if the closure returns `None`, otherwise returns the new value.
	/// 
	/// # Panics
	/// 
	/// Panics if order is [`Ordering::Release`] or [`Ordering::AcqRel`].
	pub fn fetch_update<T>(
		&self,
		set_order: Ordering,
		fetch_order: Ordering,
		f: impl Fn(SetValue) -> Option<T>,
	) -> Result<SetValue, SetValue> 
	where 
		T: Into<SetValue>,
	{
		let value = self.load(fetch_order);
		if let Some(new_value) = f(value) {
			self.store(new_value, set_order);
		}else {
			return Err(self.load(fetch_order));
		};

		Ok(self.load(fetch_order))
	}

	/// Sets the changed flag with the given ordering.
	pub fn set_changed(&self, changed: bool, order: Ordering) {
		match self {
			Self::Bool{ changed: c, .. } => c.store(changed, order),
			Self::Float { changed: c, .. } => c.store(changed, order),
			Self::Int { changed: c, .. } => c.store(changed, order),
			Self::Nothing => {},
		}
	}

	/// Returns true if the value has been changed.
	pub fn is_chanegd(&self) -> bool {
		match self {
			Self::Bool{ changed, .. } => changed.load(Ordering::Relaxed),
			Self::Float { changed, .. } => changed.load(Ordering::Relaxed),
			Self::Int { changed, .. } => changed.load(Ordering::Relaxed),
			Self::Nothing => false,
		}
	}
}

/// A helper enum for parameter values.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum Value {
	/// A floating-point value with a range.
	Float {
		/// The actual value.
		value: f32,
		/// The range of valid values.
		range: RangeInclusive<f32>,
		/// Whether the value is logarithmic.
		logarithmic: bool,
	},
	/// An integer value with a range.
	Int {
		/// The actual value.
		value: i32,
		/// The range of valid values.
		range: RangeInclusive<i32>,
		/// Whether the value is logarithmic.
		logarithmic: bool,
	},
	/// A boolean value.
	Bool(bool),
	/// A serialized value, used for data that needs to persist across sessions,
	/// While it can be any type, it is recommended to use a binary format for this value.
	/// Also the crate will mark this value as non-editable in the host UI.
	Serialized(Vec<u8>),
	/// A placeholder value, used for parameters that are not yet implemented or is None.
	#[default] Nothing,
}

impl Value {
	/// Returns the value as a set value.
	pub fn to_set_value(self) -> SetValue {
		match self {
			Value::Float { value,.. } => SetValue::Float(value),
			Value::Int { value,.. } => SetValue::Int(value),
			Value::Bool(value) => SetValue::Bool(value),
			Value::Serialized(data) => SetValue::Serialized(data),
			Value::Nothing => SetValue::Nothing,
		}
	}

	/// Returns true if the value is a float.
	pub fn is_float(&self) -> bool {
		matches!(self, Value::Float { .. })
	}

	/// Returns true if the value is an int.
	pub fn is_int(&self) -> bool {
		matches!(self, Value::Int { .. })
	}

	/// Returns true if the value is a bool.
	pub fn is_bool(&self) -> bool {
		matches!(self, Value::Bool(_))
	}

	/// Returns true if the value is nothing.
	pub fn is_nothing(&self) -> bool {
		matches!(self, Value::Nothing)
	}

	/// Returns true if the value is serialized.
	pub fn is_serialized(&self) -> bool {
		matches!(self, Value::Serialized(_))
	}

	/// Returns the value as an atomic value.
	pub fn to_atomic_value(self) -> AtomicValue {
		match self {
			Value::Float { value, range, logarithmic } => AtomicValue::Float {
				value: AtomicF32::new(value),
				range,
				logarithmic,
				changed: AtomicBool::new(false),
			},
			Value::Int { value, range, logarithmic } => AtomicValue::Int {
				value: AtomicI32::new(value),
				range,
				logarithmic,
				changed: AtomicBool::new(false),
			},
			Value::Bool(value) => AtomicValue::Bool{
				value: AtomicBool::new(value),
				changed: AtomicBool::new(false),
			},
			Value::Serialized(_) => AtomicValue::Nothing,
			Value::Nothing => AtomicValue::Nothing,
		}
	}
}

/// A helper enum for parameter set values.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum SetValue {
	/// A floating-point value.
	Float(f32),
	/// An integer value.
	Int(i32),
	/// A serialized value, used for data that needs to persist across sessions,
	Serialized(Vec<u8>),
	/// A boolean value.
	Bool(bool),
	/// A placeholder value, used for parameters that are not yet implemented or is None.
	#[default] Nothing
}

impl From<f32> for SetValue {
	fn from(v: f32) -> Self {
		Self::Float(v)
	}
}

impl From<&f32> for SetValue {
	fn from(v: &f32) -> Self {
		Self::Float(*v)
	}
}

impl From<&mut f32> for SetValue {
	fn from(v: &mut f32) -> Self {
		Self::Float(*v)
	}
}

impl From<i32> for SetValue {
	fn from(v: i32) -> Self {
		Self::Int(v)
	}
}

impl From<&i32> for SetValue {
	fn from(v: &i32) -> Self {
		Self::Int(*v)
	}
}

impl From<&mut i32> for SetValue {
	fn from(v: &mut i32) -> Self {
		Self::Int(*v)
	}
}

impl From<bool> for SetValue {
	fn from(v: bool) -> Self {
		Self::Bool(v)
	}
}

impl From<&bool> for SetValue {
	fn from(v: &bool) -> Self {
		Self::Bool(*v)
	}
}

impl From<&mut bool> for SetValue {
	fn from(v: &mut bool) -> Self {
		Self::Bool(*v)
	}
}

impl From<Vec<u8>> for SetValue {
	fn from(v: Vec<u8>) -> Self {
		Self::Serialized(v)
	}
}

impl From<()> for SetValue {
	fn from(_: ()) -> Self {
		Self::Nothing
	}
}

impl From<Option<f32>> for SetValue {
	fn from(v: Option<f32>) -> Self {
		if let Some(v) = v {
			Self::Float(v)
		} else {
			Self::Nothing
		}
	}
}

impl From<Option<i32>> for SetValue {
	fn from(v: Option<i32>) -> Self {
		if let Some(v) = v {
			Self::Int(v)
		} else {
			Self::Nothing
		}
	}
}

impl From<Option<bool>> for SetValue {
	fn from(v: Option<bool>) -> Self {
		if let Some(v) = v {
			Self::Bool(v)
		} else {
			Self::Nothing
		}
	}
}

impl From<Option<Vec<u8>>> for SetValue {
	fn from(v: Option<Vec<u8>>) -> Self {
		if let Some(v) = v {
			Self::Serialized(v)
		} else {
			Self::Nothing
		}
	}
}

impl SetValue {
	/// Returns the float value if it is a float value, otherwise returns None.
	pub fn float(&self) -> Option<f32> {
		if let Self::Float(v) = self {
			Some(*v)
		} else {
			None
		}
	}

	/// Returns the int value if it is an int value, otherwise returns None.
	pub fn int(&self) -> Option<i32> {
		if let Self::Int(v) = self {
			Some(*v)
		} else {
			None
		}
	}

	/// Returns the bool value if it is a bool value, otherwise returns None.
	pub fn bool(&self) -> Option<bool> {
		if let Self::Bool(v) = self {
			Some(*v)
		} else {
			None
		}
	}

	/// Returns the serialized value if it is a serialized value, otherwise returns None.
	pub fn serialized(self) -> Option<Vec<u8>> {
		if let Self::Serialized(v) = self {
			Some(v)
		} else {
			None
		}
	}

	/// Returns the nothing value if it is a nothing value, otherwise returns None.
	pub fn nothing(&self) -> Option<()> {
		if let Self::Nothing = self {
			Some(())
		} else {
			None
		}
	}
}

/// A helper trait for report parameters change to hosts.
pub trait Parameters {
	/// Returns a list of parameters with their current values.
	fn get_parameters(&self) -> Vec<Parameter>;
	/// Sets the value of a parameter, will return false if identifier is not found.
	/// 
	/// Host should call this method to update the value and make sure the value is within the valid range.
	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool;

	/// Sets the value of a parameter by index, will return false if index is out of range.
	///
	/// The `index` is the position in the flattened parameter list returned by
	/// [`Self::get_parameters`]. In derive-based implementations `#[sub_param]` fields
	/// contribute one entry per sub-parameter, so indices are *not* the same as the
	/// struct's field ordinals.
	fn set_parameter_by_index(&mut self, index: usize, value: SetValue) -> bool {
		let params = self.get_parameters();
		let Some(param) = params.get(index) else {
			return false;
		};

		self.set_parameter(&param.identifier, value)
	}

	/// The identifiers whose host-facing id is pinned, as `(identifier, id)` pairs.
	///
	/// A host keys the automation and modulation it stores on a parameter's id, and CLAP
	/// requires an id to never move between releases. [`ParamMap`] derives ids from the
	/// identifiers by default, which covers most cases; this is the escape hatch for the ones
	/// it does not:
	///
	/// - a plugin that already shipped ids before they were derived from identifiers pins them
	///   here, or reproduces them wholesale with [`StableIdScheme::Position`];
	/// - a parameter whose id must survive its field being renamed pins the id so the name can
	///   change freely.
	///
	/// The derive generates this from `#[id(stable = N)]`; a hand-written implementation
	/// returns [`Vec::new`] (the default) and lets the scheme decide.
	fn stable_id_overrides(&self) -> Vec<(String, u32)> {
		Vec::new()
	}
}

impl Parameters for Complex<f32> {
	fn get_parameters(&self) -> Vec<Parameter> {
		vec![
			Parameter::new(
				"re",
				Value::Float {
					value: self.re,
					range: -10000.0..=10000.0,
					logarithmic: false,
				},
			),
			Parameter::new(
				"im",
				Value::Float {
					value: self.im,
					range: -10000.0..=10000.0,
					logarithmic: false,
				},
			),
		]
	}

	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		match identifier {
			"re" => if let SetValue::Float(v) = value {
				self.re = v;
				true
			} else {
				false
			},
			"im" => if let SetValue::Float(v) = value {
				self.im = v;
				true
			} else {
				false
			},
			_ => false,
		}
	}
}

impl<T: Parameters> Parameters for Vec<T> {
	fn get_parameters(&self) -> Vec<Parameter> {
		let mut result = Vec::new();
		for (i, p) in self.iter().enumerate() {
			for mut param in p.get_parameters() {
				param.identifier = format!("{i}.{}", param.identifier);
				result.push(param);
			}
		}
		result
	}

	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		let mut parts = identifier.split(".").collect::<Vec<&str>>();
		let index = parts.remove(0).parse::<usize>().expect("Invalid index");
		let rest_identifier = parts.join(".");
		if index >= self.len() {
			return false;
		}
		self[index].set_parameter(&rest_identifier, value)
	}
}

impl Parameters for Vec<Box<dyn Parameters>> {
	fn get_parameters(&self) -> Vec<Parameter> {
		let mut result = Vec::new();
		for (i, p) in self.iter().enumerate() {
			for mut param in p.get_parameters() {
				param.identifier = format!("{i}.{}", param.identifier);
				result.push(param);
			}
		}
		result
	}

	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		let mut parts = identifier.split(".").collect::<Vec<&str>>();
		let index = parts.remove(0).parse::<usize>().expect("Invalid index");
		let rest_identifier = parts.join(".");
		if index >= self.len() {
			return false;
		}
		self[index].set_parameter(&rest_identifier, value)
	}

	fn stable_id_overrides(&self) -> Vec<(String, u32)> {
		// The identifiers of the elements are prefixed with their position, so pinned ids
		// have to be prefixed the same way to keep matching them.
		let mut result = Vec::new();
		for (i, p) in self.iter().enumerate() {
			for (identifier, id) in p.stable_id_overrides() {
				result.push((format!("{i}.{identifier}"), id));
			}
		}
		result
	}
}

impl Parameters for () {
	fn get_parameters(&self) -> Vec<Parameter> {
		Vec::new()
	}

	fn set_parameter(&mut self, _identifier: &str, _value: SetValue) -> bool {
		false
	}
}

impl<T: Parameters> Parameters for &mut [T] {
	fn get_parameters(&self) -> Vec<Parameter> {
		let mut result = Vec::new();
		for (i, p) in self.iter().enumerate() {
			for mut param in p.get_parameters() {
				param.identifier = format!("{i}.{}", param.identifier);
				result.push(param);
			}
		}
		result
	}

	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		let mut parts = identifier.split(".").collect::<Vec<&str>>();
		let index = parts.remove(0).parse::<usize>().expect("Invalid index");
		let rest_identifier = parts.join(".");
		if index >= self.len() {
			return false;
		}
		self[index].set_parameter(&rest_identifier, value)
	}
}

impl<T: Parameters> Parameters for [T] {
	fn get_parameters(&self) -> Vec<Parameter> {
		let mut result = Vec::new();
		for (i, p) in self.iter().enumerate() {
			for mut param in p.get_parameters() {
				param.identifier = format!("{i}.{}", param.identifier);
				result.push(param);
			}
		}
		result
	}

	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		let mut parts = identifier.split(".").collect::<Vec<&str>>();
		let index = parts.remove(0).parse::<usize>().expect("Invalid index");
		let rest_identifier = parts.join(".");
		if index >= self.len() {
			return false;
		}
		self[index].set_parameter(&rest_identifier, value)
	}
}

impl<T: Parameters> Parameters for &[T] {
	fn get_parameters(&self) -> Vec<Parameter> {
		let mut result = Vec::new();
		for (i, p) in self.iter().enumerate() {
			for mut param in p.get_parameters() {
				param.identifier = format!("{i}.{}", param.identifier);
				result.push(param);
			}
		}
		result
	}

	fn set_parameter(&mut self, _: &str, _: SetValue) -> bool {
		false
	}
}

impl Parameters for f32 {
	fn get_parameters(&self) -> Vec<Parameter> {
		vec![Parameter::new(
			"data",
			Value::Float {
				value: *self,
				range: -10000.0..=10000.0,
				logarithmic: false,
			},
		)]
	}

	fn set_parameter(&mut self, _: &str, value: SetValue) -> bool {
		if let SetValue::Float(v) = value {
			*self = v;
			return true;
		}
		false
	}
}

impl Parameters for String {
	fn get_parameters(&self) -> Vec<Parameter> {
		vec![Parameter::new(
			"data",
			Value::Serialized(self.as_bytes().to_vec()),
		)]
	}

	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		if identifier == "data" && let SetValue::Serialized(data) = value {
			*self = String::from_utf8_lossy(&data).to_string();
			return true;
		}
		false
	}
}

impl<T: Parameters, const N: usize> Parameters for [T; N] {
	fn get_parameters(&self) -> Vec<Parameter> {
		let mut result = Vec::new();
		for (i, p) in self.iter().enumerate() {
			for mut param in p.get_parameters() {
				param.identifier = format!("{i}.{}", param.identifier);
				result.push(param);
			}
		}
		result
	}

	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		let mut parts = identifier.split(".").collect::<Vec<&str>>();
		let index = parts.remove(0).parse::<usize>().expect("Invalid index");
		let rest_identifier = parts.join(".");
		if index >= N {
			return false;
		}
		self[index].set_parameter(&rest_identifier, value)
	}
}

impl<T: Parameters> Parameters for Option<T> {
	fn get_parameters(&self) -> Vec<Parameter> {
		if let Some(p) = self {
			p.get_parameters()
		} else {
			Vec::new()
		}
	}

	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		if let SetValue::Nothing = value {
			*self = None;
		}else if let Some(p) = self {
			return p.set_parameter(identifier, value);
		}

		false
	}
}

/// A helper function
/// 
/// It can be used to convert any type implementing the `Serialize` trait to a binary format.
/// We will use cbor as the binary format.
/// 
/// # Panics
/// 
/// This function will panic if the value cannot be serialized.
pub fn to_binary<T: serde::Serialize>(value: &T) -> Result<Vec<u8>, ciborium::ser::Error<std::io::Error>> {
	let mut output = vec![];
	into_writer(value, &mut output)?;
	Ok(output)
}

/// A helper function
/// 
/// It can be used to convert a binary format to any type implementing the `Deserialize` trait.
/// We will use cbor as the binary format.
/// 
/// # Panics
/// 
/// This function will panic if the data cannot be deserialized.
pub fn from_binary<T: serde::de::DeserializeOwned>(data: Vec<u8>) -> Result<T, ciborium::de::Error<std::io::Error>> {
	let slice = data.as_slice();
	from_reader(slice)
}

/// A simple wrapper around a sturct makes it possible to have a sturct with multiple parameter setters.
/// 
/// Though you need set params manually, it can be useful when you want to parameterize a complex struct.
pub struct Paramed<T: Parameters> {
	/// The struct that we want to parameterize.
	pub value: T,
	params: ParamMap,
	/// How the host-facing ids of the parameters are assigned.
	scheme: StableIdScheme,
	// sync_set_cached: HashSet<usize>,
}

impl<T: Parameters> Parameters for Paramed<T> {
	fn get_parameters(&self) -> Vec<Parameter> {
		self.value.get_parameters()
	}

	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		self.value.set_parameter(identifier, value)
	}

	fn stable_id_overrides(&self) -> Vec<(String, u32)> {
		self.value.stable_id_overrides()
	}
}

impl<const CHANNELS: usize, T: Generator<CHANNELS>> Generator<CHANNELS> for Paramed<T> {
	fn generate(&mut self, process_context: &mut Box<dyn crate::ProcessContext>) -> [f32; CHANNELS] {
		self.sync_params();
		self.value.generate(process_context)
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		self.value.name()
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		self.value.demo_ui(ui, id_prefix)
	}
}

impl<const CHANNELS: usize, T: Effect<CHANNELS>> Effect<CHANNELS> for Paramed<T> {
	fn delay(&self) -> usize {
		self.value.delay()
	}

	fn process(
		&mut self, 
		samples: &mut [f32; CHANNELS], 
		other: &[&[f32; CHANNELS]],
		process_context: &mut Box<dyn crate::ProcessContext>,
	) {
		self.sync_params();
		self.value.process(samples, other, process_context)
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		self.value.name()
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		self.value.demo_ui(ui, id_prefix)
	}
}

impl<T: Parameters> Paramed<T> {
	/// Create a new parameterized object.
	///
	/// Host-facing ids are derived from the parameter identifiers; see
	/// [`Self::new_with_scheme`] to reproduce the ids an older build handed out.
	pub fn new(to_parameterize: T) -> Self {
		Self::new_with_scheme(to_parameterize, StableIdScheme::default())
	}

	/// Create a new parameterized object with an explicit [`StableIdScheme`].
	///
	/// A plugin that shipped index-based ids before ids were derived from identifiers picks
	/// [`StableIdScheme::Position`] here, so the projects its users already saved keep
	/// addressing the same parameters.
	pub fn new_with_scheme(to_parameterize: T, scheme: StableIdScheme) -> Self {
		let mut empty_self = Self {
			value: to_parameterize,
			params: ParamMap::new(),
			scheme,
		};
		empty_self.update_map();
		empty_self
	}

	/// The scheme used to assign host-facing ids.
	pub fn stable_id_scheme(&self) -> StableIdScheme {
		self.scheme
	}

	/// Create a new parameterized object with additional parameters.
	pub fn new_with_additional(to_parameterize: T, additional_params: impl IntoIterator<Item = (String, Arc<AtomicValue>)>) -> Self {
		let mut empty_self = Self {
			value: to_parameterize,
			params: ParamMap::new(),
			scheme: StableIdScheme::default(),
		};
		empty_self.update_map_with_additional(additional_params);
		empty_self
	}

	/// Synchronize the parameters with the host.
	pub fn sync_params(&mut self) {
		for (id, param) in self.params.values.iter().enumerate() {
			if let Some(identifier) = self.params.query_param_id(id) {
				self.value.set_parameter(identifier, param.load(Ordering::SeqCst));
			}
		}
	}

	/// Get the parameter map.
	/// 
	/// This function will return a [`ParamMap`] object, which can be used to query and set parameter values.
	pub fn param_map(&self) -> ParamMap {
		self.params.clone()
	}

	/// Update the parameter map.
	/// 
	/// Useful for those structs that have unfixed number of parameters.
	/// 
	/// Note: this function will **not** affect the previous [`ParamMap`] got from [`Self::param_map`].
	pub fn update_map(&mut self) {
		let entries = self.value.get_parameters().into_iter().map(|param| {
			(param.identifier, param.meta, Arc::new(param.value.to_atomic_value()))
		});

		let overrides = self.value.stable_id_overrides();
		self.params = ParamMap::build(entries, self.scheme, &overrides);
	}

	/// Update the parameter map and add additional parameters.
	/// 
	/// Useful for those structs that have unfixed number of parameters.
	/// 
	/// Note: this function will **not** affect the previous [`ParamMap`] got from [`Self::param_map`].
	pub fn update_map_with_additional(&mut self, additional_params: impl IntoIterator<Item = (String, Arc<AtomicValue>)>) {
		let mut entries = self.value.get_parameters().into_iter().map(|param| {
			(param.identifier, param.meta, Arc::new(param.value.to_atomic_value()))
		}).collect::<Vec<_>>();

		// Parameters the caller adds by hand carry no metadata; a host falls back to its own
		// defaults for them.
		entries.extend(
			additional_params
				.into_iter()
				.map(|(identifier, value)| (identifier, ParameterMeta::default(), value)),
		);

		let overrides = self.value.stable_id_overrides();
		self.params = ParamMap::build(entries, self.scheme, &overrides);
	}
}

/// The offset basis of the 32-bit FNV-1a hash.
const FNV_OFFSET_BASIS: u32 = 0x811c_9dc5;
/// The prime of the 32-bit FNV-1a hash.
const FNV_PRIME: u32 = 0x0100_0193;

/// Avalanches a hash, so that two identifiers differing in one character do not end up
/// with ids differing in a few bits.
fn finalize_hash(mut hash: u32) -> u32 {
	hash ^= hash >> 16;
	hash = hash.wrapping_mul(0x85eb_ca6b);
	hash ^= hash >> 13;
	hash = hash.wrapping_mul(0xc2b2_ae35);
	hash ^= hash >> 16;
	hash
}

/// Hashes a parameter identifier.
///
/// `salt` is zero for the canonical id and is only raised to break a collision, which keeps
/// [`stable_parameter_id`] a pure function of the identifier.
fn hash_identifier(identifier: &str, salt: u32) -> u32 {
	let mut hash = FNV_OFFSET_BASIS;

	// The salt goes into the byte stream rather than being folded into the state afterwards.
	// Two identifiers that collide in the state stay collided under any later XOR, so a
	// post-hoc salt could never separate them and the search for a free id would never end.
	for byte in salt.to_le_bytes() {
		hash ^= byte as u32;
		hash = hash.wrapping_mul(FNV_PRIME);
	}

	for byte in identifier.as_bytes() {
		hash ^= *byte as u32;
		hash = hash.wrapping_mul(FNV_PRIME);
	}

	let hash = finalize_hash(hash);

	// Steinberg's SDK reserves the top bit of a VST3 parameter id, and clap-wrapper strips it
	// (`v.id = info->id & 0x7FFFFFFF`) before handing the id to the host — and hands the stripped
	// id back. An id with that bit set would therefore address a different parameter in a VST3
	// host than in a CLAP one, so ids stay inside 31 bits. This also rules out `u32::MAX`, which
	// hosts reserve for "no id".
	hash & 0x7FFF_FFFF
}

/// Returns the stable numeric id of the parameter with the given identifier.
///
/// A host keys everything it persists about a parameter (automation lanes, modulation
/// routings, …) on this number, and CLAP requires a plugin to never change the id of an
/// existing parameter. Deriving that id from the identifier rather than from the
/// parameter's position is what lets an effect gain a field without renumbering every
/// parameter declared after it.
///
/// The id is a pure function of `identifier`: the same identifier always maps to the same
/// id, in any process and on any run. It always fits in 31 bits, so it survives the trip
/// through a VST3 host (whose parameter ids have their top bit reserved) unchanged, and it
/// is never [`u32::MAX`].
///
/// Identifiers come from [`Parameters::get_parameters`]. A named struct field contributes
/// its field name (optionally renamed or prefixed with `#[id(...)]`), a tuple struct field
/// contributes its position, and a nested struct contributes a dotted path — so a
/// parameter whose identifier is positional still has a positional id.
///
/// # Examples
///
/// ```
/// use i_am_dsp::prelude::stable_parameter_id;
///
/// assert_eq!(stable_parameter_id("delay_time"), stable_parameter_id("delay_time"));
/// assert_ne!(stable_parameter_id("delay_time"), stable_parameter_id("decay_factor"));
/// ```
pub fn stable_parameter_id(identifier: &str) -> u32 {
	hash_identifier(identifier, 0)
}

/// How a [`ParamMap`] assigns the host-facing id of a parameter that pins none.
///
/// A pinned id (see [`Parameters::stable_id_overrides`]) always wins; the scheme only
/// decides the rest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StableIdScheme {
	/// Derive the id from the parameter's identifier.
	///
	/// Inserting, removing, reordering or renaming a parameter leaves every *other*
	/// parameter's id alone, which is what a plugin that has not shipped yet wants.
	#[default]
	Identifier,
	/// Use the parameter's position in the flattened parameter list.
	///
	/// This reproduces the ids of a plugin that shipped before ids were derived from
	/// identifiers, so the projects its users already saved keep addressing the same
	/// parameters. It only holds while parameters are appended at the end: inserting or
	/// reordering one moves the id of every parameter after it, so pin the ids of anything
	/// new with [`Parameters::stable_id_overrides`].
	Position,
}

#[derive(Debug, Clone)]
/// A helper struct for managing parameter values for using [`Paramed`] struct.
/// 
/// This struct should be cheap to clone and will assosiated with one specific parameterized object.
pub struct ParamMap {
	values: Arc<Vec<Arc<AtomicValue>>>,
	id: Arc<BiMap<String, usize>>,
	stable_ids: Arc<Vec<u32>>,
	by_stable_id: Arc<BiMap<u32, usize>>,
	/// What a host needs to present each parameter, by index.
	metas: Arc<Vec<ParameterMeta>>,
}

impl Index<usize> for ParamMap {
	type Output = AtomicValue;

	fn index(&self, index: usize) -> &Self::Output {
		&self.values[index]
	}
}

impl Index<&str> for ParamMap {
	type Output = AtomicValue;

	fn index(&self, index: &str) -> &Self::Output {
		let index = self.id.get_by_left(index).unwrap_or_else(|| panic!("index `{}` not found", index));
		self.values.get(*index).unwrap_or_else(|| panic!("index `{}` not found", index))
	}
}

impl Default for ParamMap {
	fn default() -> Self {
		Self::new()
	}
}

impl ParamMap {
	/// Builds a parameter map from identifiers and values.
	///
	/// The position in the iterator is the index that [`Self::get_by_index`] and
	/// [`Self::query_param_index`] address, while the stable id is what a host addresses.
	///
	/// `overrides` pins the id of the parameters it names (see
	/// [`Parameters::stable_id_overrides`]); `scheme` decides the rest.
	fn build(
		entries: impl IntoIterator<Item = (String, ParameterMeta, Arc<AtomicValue>)>,
		scheme: StableIdScheme,
		overrides: &[(String, u32)],
	) -> Self {
		let entries = entries.into_iter().collect::<Vec<_>>();
		let mut values = Vec::with_capacity(entries.len());
		let mut metas = Vec::with_capacity(entries.len());
		let mut id = BiMap::new();
		let mut stable_ids = vec![0u32; entries.len()];
		let mut by_stable_id = BiMap::new();
		let mut assigned = vec![false; entries.len()];

		for (index, (identifier, meta, value)) in entries.iter().enumerate() {
			id.insert(identifier.clone(), index);
			metas.push(meta.clone());
			values.push(value.clone());
		}

		// Pinned ids are handed out first, in declaration order, so the same table always
		// resolves the same way. A pin that collides with an earlier one is left to the scheme
		// rather than aliasing two parameters onto one id.
		for (index, (identifier, _, _)) in entries.iter().enumerate() {
			let Some(id) = overrides.iter().find(|(pinned, _)| pinned == identifier).map(|(_, id)| *id) else {
				continue;
			};

			if by_stable_id.contains_left(&id) {
				debug_assert!(false, "two parameters are pinned to the same host id ({id})");
				continue;
			}

			by_stable_id.insert(id, index);
			stable_ids[index] = id;
			assigned[index] = true;
		}

		// Everyone else follows the scheme. Under `Identifier` they are served in identifier
		// order rather than declaration order, so that when two identifiers collide, which of
		// them keeps the canonical id does not depend on how the fields happen to be laid out:
		// the assignment is a pure function of the set of identifiers.
		let mut order = (0..entries.len()).filter(|index| !assigned[*index]).collect::<Vec<_>>();
		if scheme == StableIdScheme::Identifier {
			order.sort_by(|left, right| entries[*left].0.cmp(&entries[*right].0));
		}

		for index in order {
			let identifier = &entries[index].0;
			let preferred = match scheme {
				StableIdScheme::Identifier => hash_identifier(identifier, 0),
				StableIdScheme::Position => index as u32,
			};

			// A taken id means either two identifiers hashing alike (~1 in 4 billion) or a
			// position that a pinned id already claimed. Either way the parameter draws a fresh
			// id instead of aliasing, and the salt goes into the hash so the draw really varies.
			let mut stable_id = preferred;
			let mut salt = 0u32;
			while by_stable_id.contains_left(&stable_id) {
				salt = salt.wrapping_add(1);
				stable_id = hash_identifier(identifier, salt);
			}

			by_stable_id.insert(stable_id, index);
			stable_ids[index] = stable_id;
		}

		Self {
			values: Arc::new(values),
			id: Arc::new(id),
			stable_ids: Arc::new(stable_ids),
			by_stable_id: Arc::new(by_stable_id),
			metas: Arc::new(metas),
		}
	}

	/// Create a new empty parameter map.
	pub fn new() -> Self {
		Self::build(std::iter::empty(), StableIdScheme::default(), &[])
	}

	/// Create a new parameter map with a pre-defined map.
	///
	/// Host-facing ids are derived from the identifiers; use [`ParamMap::new_with_scheme`] to
	/// reproduce the position-based ids an older build handed out.
	pub fn new_with(map: impl IntoIterator<Item = (String, Arc<AtomicValue>)>) -> Self {
		Self::build(
			map.into_iter().map(|(id, value)| (id, ParameterMeta::default(), value)),
			StableIdScheme::default(),
			&[],
		)
	}

	/// Create a new parameter map with a pre-defined map and an explicit [`StableIdScheme`].
	pub fn new_with_scheme(
		map: impl IntoIterator<Item = (String, Arc<AtomicValue>)>,
		scheme: StableIdScheme,
	) -> Self {
		Self::build(
			map.into_iter().map(|(id, value)| (id, ParameterMeta::default(), value)),
			scheme,
			&[],
		)
	}

	/// Check if the parameter map is empty.
	pub fn is_empty(&self) -> bool {
		self.values.is_empty()
	}

	/// Get the number of parameters.
	pub fn len(&self) -> usize {
		self.values.len()
	}

	/// Get the parameter value by its identifier.
	pub fn get(&self, id: &str) -> Option<Arc<AtomicValue>> {
		let index = self.id.get_by_left(id)?;
		let value = self.values.get(*index)?;
		// self.update_queue(*index);
		
		Some(value.clone())
	}

	/// Set the parameter value by its identifier.
	/// 
	/// Returns `true` if the parameter is set successfully, `false` otherwise.
	pub fn set(&self, id: &str, to_set: impl Into<SetValue>, ordering: Ordering) -> bool {
		let Some(index) = self.id.get_by_left(id) else {
			return false;
		};
		let Some(value) = self.values.get(*index) else {
			return false;
		};

		value.store(to_set, ordering);
		
		true
	} 

	/// Query the parameter value by its index.
	pub fn query_param_index(&self, id: &str) -> Option<usize> {
		self.id.get_by_left(id).copied()
	}

	/// Query the parameter identifier by its index.
	pub fn query_param_id(&self, index: usize) -> Option<&str> {
		self.id.get_by_right(&index).map(|s| s.as_str())
	}

	/// Get the parameter value by its index.
	pub fn get_by_index(&self, index: usize) -> Option<Arc<AtomicValue>> {
		let value = self.values.get(index)?;
		// self.update_queue(index);

		Some(value.clone())
	}

	/// Query the stable id of the parameter at the given index.
	///
	/// This is the id a host knows the parameter by, see [`stable_parameter_id`].
	pub fn query_stable_id(&self, index: usize) -> Option<u32> {
		self.stable_ids.get(index).copied()
	}

	/// Query what a host needs to present the parameter at the given index: its starting value
	/// and how to show it.
	pub fn query_meta(&self, index: usize) -> Option<ParameterMeta> {
		self.metas.get(index).cloned()
	}

	/// Query the index of the parameter with the given stable id.
	pub fn query_param_index_by_stable_id(&self, stable_id: u32) -> Option<usize> {
		self.by_stable_id.get_by_left(&stable_id).copied()
	}

	/// Get the parameter value by its stable id.
	pub fn get_by_stable_id(&self, stable_id: u32) -> Option<Arc<AtomicValue>> {
		let index = self.query_param_index_by_stable_id(stable_id)?;
		self.get_by_index(index)
	}

	/// Iterate over the parameter values.
	pub fn iter(&self) -> impl Iterator<Item = &Arc<AtomicValue>> {
		self.values.iter()
	}

	/// Iterate over the parameter values and their indices.
	/// 
	/// Will automatically set the `changed` flag to `false` for each value.
	pub fn iter_changed(&self) -> impl Iterator<Item = (usize, &Arc<AtomicValue>)> {
		self.values.iter().enumerate().filter(|(_, val)| {
			let out = val.is_chanegd();
			if out {
				val.set_changed(false, Ordering::SeqCst);
			}
			out
		})
	}
}

#[cfg(test)]
mod tests {
	use i_am_dsp_derive::Parameters;
	use super::*;

	#[derive(Parameters, Default)]
	struct MyParameters {
		#[range(min = 0.0, max = 1.0)]
		a: f32,
		#[range(min = 1, max = 10)]
		b: i32,
		#[id(name = "Foo")]
		#[persist(serialize = "string_to_binary", deserialize = "binary_to_string")]
		c: String,

		#[sub_param]
		d: SubParameters,

		boo: bool,

		#[skip]
		_e: Vec<f32>
	}

	#[derive(Parameters, Default)]
	struct SubParameters {
		#[range(min = 0.0, max = 1.0)]
		f: f32,
		#[range(min = 0, max = 1)]
		g: i32,
		#[serde]
		h: (f32, f32),
	}


	fn string_to_binary(s: &String) -> Vec<u8> {
		s.as_bytes().to_vec()
	}

	fn binary_to_string(b: Vec<u8>) -> String {
		String::from_utf8_lossy(&b).to_string()
	}

	#[test]
	fn test_parameter() {
		let mut params = MyParameters::default();

		let params_output = params.get_parameters();
		// println!("{:#?}", params_output);
		assert_eq!(params_output.len(), 7);
		params.set_parameter("a", SetValue::Float(0.5));
		assert_eq!(params.a, 0.5);
		params.set_parameter("b", SetValue::Int(5));
		assert_eq!(params.b, 5);
		params.set_parameter("Foo", SetValue::Serialized("World".as_bytes().to_vec()));
		assert_eq!(params.c, "World");
		params.set_parameter("d.f", SetValue::Float(0.75));
		assert_eq!(params.d.f, 0.75);
		params.set_parameter("d.g", SetValue::Int(0));
		assert_eq!(params.d.g, 0);
		params.set_parameter("boo", SetValue::Bool(true));
		assert!(params.boo);
		params.set_parameter("_e", SetValue::Nothing);
		assert_eq!(params._e, Vec::new());
		params.set_parameter("d.h", SetValue::Serialized(to_binary(&(2.0, 3.0)).unwrap()));
		assert_eq!(params.d.h, (2.0, 3.0));
	}

	/// The id is a pure function of the identifier, and never the value hosts reserve.
	#[test]
	fn stable_ids_are_a_pure_function_of_the_identifier() {
		assert_eq!(stable_parameter_id("delay_time"), stable_parameter_id("delay_time"));
		assert_ne!(stable_parameter_id("delay_time"), stable_parameter_id("decay_factor"));

		// A VST3 host only keeps the low 31 bits of a parameter id, so anything above that
		// would address a different parameter there; `u32::MAX` is reserved for "no id".
		for identifier in ["", "a", "b", "d.f", "oscillator.carrier.smooth_factor"] {
			assert!(stable_parameter_id(identifier) <= 0x7FFF_FFFF);
		}
	}

	#[test]
	fn a_plain_unit_renders_a_number() {
		assert_eq!(Unit::None.format(0.5), "0.50");
		assert_eq!(Unit::Milliseconds.format(250.0), "250.00 ms");
		assert_eq!(Unit::Hertz.format(440.0), "440.00 Hz");
	}

	/// A synced time is why the unit exists: `0.5` is an eighth note to a musician, and
	/// `0.00` is not.
	#[test]
	fn beats_render_as_note_values() {
		assert_eq!(Unit::Beats.format(4.0), "1/1");
		assert_eq!(Unit::Beats.format(2.0), "1/2");
		assert_eq!(Unit::Beats.format(1.0), "1/4");
		assert_eq!(Unit::Beats.format(0.5), "1/8");
		assert_eq!(Unit::Beats.format(0.25), "1/16");

		assert_eq!(Unit::Beats.format(1.5), "1/4 dotted");
		assert_eq!(Unit::Beats.format(0.75), "1/8 dotted");
		// A dotted half note is three beats.
		assert_eq!(Unit::Beats.format(3.0), "1/2 dotted");
		assert_eq!(Unit::Beats.format(1.0 / 3.0), "1/8 triplet");
		assert_eq!(Unit::Beats.format(2.0 / 3.0), "1/4 triplet");

		// Anything that is not a note value stays a count rather than a wrong fraction.
		assert_eq!(Unit::Beats.format(3.7), "3.70 beats");
		assert_eq!(Unit::Beats.format(7.0), "7.00 beats");
	}

	#[test]
	fn a_unit_reads_back_what_it_wrote() {
		for beats in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 1.0 / 3.0, 2.0 / 3.0] {
			let text = Unit::Beats.format(beats);
			let parsed = Unit::Beats.parse(&text).expect("what it wrote has to read back");
			assert!(
				(parsed - beats).abs() < 1e-6,
				"`{text}` read back as {parsed}, not {beats}"
			);
		}

		assert_eq!(Unit::Milliseconds.parse("250 ms"), Some(250.0));
		assert_eq!(Unit::Hertz.parse("440 Hz"), Some(440.0));
		// A bare number is always readable, whatever the unit.
		assert_eq!(Unit::Beats.parse("0.5"), Some(0.5));
		assert_eq!(Unit::Beats.parse("1/8"), Some(0.5));
		assert_eq!(Unit::Beats.parse("1/4 dotted"), Some(1.5));
		assert_eq!(Unit::None.parse("loud"), None);
	}

	/// The units that are not about music still have to render and read back.
	#[test]
	fn the_other_units_render_and_read_back() {
		assert_eq!(Unit::Seconds.format(0.5), "0.500 s");
		assert_eq!(Unit::Samples.format(480.0), "480 samples");
		assert_eq!(Unit::Decibels.format(-6.0), "-6.00 dB");
		assert_eq!(Unit::Percent.format(0.5), "50.0 %");

		assert_eq!(Unit::Seconds.parse("0.5 s"), Some(0.5));
		assert_eq!(Unit::Samples.parse("480 samples"), Some(480.0));
		assert_eq!(Unit::Decibels.parse("-6 dB"), Some(-6.0));
		// A percentage is a fraction, so reading it back has to undo the hundred.
		assert_eq!(Unit::Percent.parse("50 %"), Some(0.5));
		assert_eq!(Unit::Percent.parse("12.5"), Some(0.125));
	}

	/// Every parameter gets its own id, and both directions of the mapping agree.
	#[test]
	fn parameter_map_hands_out_distinct_stable_ids() {
		let param_map = Paramed::new(MyParameters::default()).param_map();
		let ids = (0..param_map.len())
			.map(|index| param_map.query_stable_id(index).expect("every index has a stable id"))
			.collect::<Vec<_>>();

		let mut distinct = ids.clone();
		distinct.sort_unstable();
		distinct.dedup();
		assert_eq!(distinct.len(), ids.len(), "two parameters were given the same stable id");

		for (index, id) in ids.iter().enumerate() {
			assert_eq!(param_map.query_stable_id(index), Some(*id));
			assert_eq!(param_map.query_param_index_by_stable_id(*id), Some(index));
			assert!(param_map.get_by_stable_id(*id).is_some());
		}

		// The derived identifier is what the id is derived from, so the two have to agree.
		assert_eq!(param_map.query_stable_id(0), Some(stable_parameter_id("a")));
		assert_eq!(param_map.query_stable_id(2), Some(stable_parameter_id("Foo")));
		assert_eq!(param_map.query_stable_id(3), Some(stable_parameter_id("d.f")));
		assert_eq!(param_map.query_stable_id(6), Some(stable_parameter_id("boo")));

		// An id no parameter was given resolves to nothing.
		assert!(param_map.get_by_stable_id(u32::MAX).is_none());
	}

	/// A 32-bit id space cannot be collision free, and two parameters sharing an id would
	/// route a host's automation to the wrong one. This pins a pair that really does collide.
	#[test]
	fn a_hash_collision_does_not_alias_two_parameters() {
		// Found by scanning identifiers; if the hash changes, the premise assertion below
		// fails and a new pair has to be found.
		const FIRST: &str = "probe_21567";
		const SECOND: &str = "probe_54170";

		assert_eq!(
			stable_parameter_id(FIRST),
			stable_parameter_id(SECOND),
			"the pinned pair no longer collides, so this test proves nothing"
		);
		assert_ne!(
			hash_identifier(FIRST, 1),
			hash_identifier(SECOND, 1),
			"raising the salt has to separate the pair, or the search for a free id never ends"
		);

		let entry = |identifier: &str| {
			(identifier.to_string(), Arc::new(Value::Bool(true).to_atomic_value()))
		};

		let param_map = ParamMap::new_with([entry(FIRST), entry(SECOND)]);
		let first = param_map.query_stable_id(0).unwrap();
		let second = param_map.query_stable_id(1).unwrap();

		assert_ne!(first, second, "colliding identifiers were aliased onto one id");
		assert_eq!(param_map.query_param_index_by_stable_id(first), Some(0));
		assert_eq!(param_map.query_param_index_by_stable_id(second), Some(1));

		// Which of the two keeps the canonical id is decided by identifier order, not by the
		// order the map was built in, so either order has to give the same parameter the same id.
		let reversed = ParamMap::new_with([entry(SECOND), entry(FIRST)]);
		assert_eq!(reversed.query_stable_id(1), Some(first));
		assert_eq!(reversed.query_stable_id(0), Some(second));
	}

	/// The scheme a plugin that already shipped positional ids has to select: the id is the
	/// declaration order, exactly as the host that saw the old build knows it.
	#[test]
	fn the_position_scheme_reproduces_positional_ids() {
		let entry = |identifier: &str| {
			(identifier.to_string(), Arc::new(Value::Bool(true).to_atomic_value()))
		};

		let positional = ParamMap::new_with_scheme(
			[entry("zeta"), entry("alpha"), entry("mid")],
			StableIdScheme::Position,
		);
		assert_eq!(positional.query_stable_id(0), Some(0));
		assert_eq!(positional.query_stable_id(1), Some(1));
		assert_eq!(positional.query_stable_id(2), Some(2));

		// The default scheme is the opposite: the id follows the identifier, not the position.
		let hashed = ParamMap::new_with([entry("zeta"), entry("alpha"), entry("mid")]);
		assert_eq!(hashed.query_stable_id(0), Some(stable_parameter_id("zeta")));
	}

	/// A pinned id beats the scheme, which is what lets a renamed field keep its identity.
	#[test]
	fn a_pinned_id_wins_over_the_scheme() {
		let entry = |identifier: &str| {
			(identifier.to_string(), Arc::new(Value::Bool(true).to_atomic_value()))
		};
		let overrides = [("something_else".to_string(), 4021u32)];

		let entries = [entry("something_else"), entry("other")]
			.map(|(identifier, value)| (identifier, ParameterMeta::default(), value));
		let param_map = ParamMap::build(
			entries,
			StableIdScheme::Identifier,
			&overrides,
		);
		assert_eq!(param_map.query_stable_id(0), Some(4021));
		assert_eq!(param_map.query_param_index_by_stable_id(4021), Some(0));
		assert_eq!(param_map.query_stable_id(1), Some(stable_parameter_id("other")));
	}

	/// Pinned ids are handed out before the scheme does, so a pin that lands on a position
	/// takes it and the position moves to a derived id. Nothing may alias.
	#[test]
	fn a_pin_that_lands_on_a_position_moves_that_position() {
		let entry = |identifier: &str| {
			(identifier.to_string(), Arc::new(Value::Bool(true).to_atomic_value()))
		};
		let overrides = [("one".to_string(), 0u32)];

		let entries = [entry("zero"), entry("one")]
			.map(|(identifier, value)| (identifier, ParameterMeta::default(), value));
		let param_map = ParamMap::build(
			entries,
			StableIdScheme::Position,
			&overrides,
		);

		assert_eq!(param_map.query_stable_id(1), Some(0), "the pinned id wins");
		assert_ne!(param_map.query_stable_id(0), Some(0), "the position has to move, not alias");
		assert_eq!(param_map.query_param_index_by_stable_id(0), Some(1));
	}
}
