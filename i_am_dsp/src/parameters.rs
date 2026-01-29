//! A helper trait for report parameters change to hosts.

use std::{ops::RangeInclusive, sync::{Arc, atomic::Ordering}};

use bimap::BiMap;
use ciborium::{from_reader, into_writer};
// use crossbeam_queue::SegQueue;
use portable_atomic::{AtomicBool, AtomicF32, AtomicI32};
use rustfft::num_complex::Complex;

use crate::{Effect, Generator};

/// A helper struct for report parameters.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Parameter {
	/// An unique identifier for the parameter.
	pub identifier: String,
	/// The parameter value.
	pub value: Value,
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
	},
	/// An integer value with a range.
	Int {
		/// The actual value.
		value: AtomicI32,
		/// The range of valid values.
		range: RangeInclusive<i32>,
		/// Whether the value is logarithmic.
		logarithmic: bool,
	},
	/// A boolean value.
	Bool(AtomicBool),
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
			Self::Bool(v) => SetValue::Bool(v.load(order)),
			Self::Float { value, .. } => SetValue::Float(value.load(order)),
			Self::Int { value, .. } => SetValue::Int(value.load(order)),
			Self::Nothing => SetValue::Nothing,
		}
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
	pub fn store(&self, value: SetValue, order: Ordering) -> bool {
		match (self, value) {
			(Self::Bool(v), SetValue::Bool(value)) => {
				v.store(value, order);
				true
			},
			(Self::Float { value: v, range, .. }, SetValue::Float(value)) => {
				let value = value.clamp(*range.start(), *range.end());
				v.store(value, order);
				true
			},
			(Self::Int { value: v, range, .. }, SetValue::Int(value)) => {
				let value = value.clamp(*range.start(), *range.end());
				v.store(value, order);
				true
			},
			(Self::Nothing, SetValue::Nothing) => true,
			_ => false,
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

	/// Returns the value as an atomic value.
	pub fn to_atomic_value(self) -> AtomicValue {
		match self {
			Value::Float { value, range, logarithmic } => {
				AtomicValue::Float {
					value: AtomicF32::new(value),
					range,
					logarithmic,
				}
			},
			Value::Int { value, range, logarithmic } => {
				AtomicValue::Int {
					value: AtomicI32::new(value),
					range,
					logarithmic,
				}
			},
			Value::Bool(value) => AtomicValue::Bool(AtomicBool::new(value)),
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

/// A helper trait for report parameters change to hosts.
pub trait Parameters {
	/// Returns a list of parameters with their current values.
	fn get_parameters(&self) -> Vec<Parameter>;
	/// Sets the value of a parameter, will return false if identifier is not found.
	/// 
	/// Host should call this method to update the value and make sure the value is within the valid range.
	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool;
}

impl Parameters for Complex<f32> {
	fn get_parameters(&self) -> Vec<Parameter> {
		vec![
			Parameter {
				identifier: "re".to_string(),
				value: Value::Float {
					value: self.re,
					range: -10000.0..=10000.0,
					logarithmic: false,
				},
			},
			Parameter {
				identifier: "im".to_string(),
				value: Value::Float {
					value: self.im,
					range: -10000.0..=10000.0,
					logarithmic: false,
				},
			},
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
		vec![Parameter {
			identifier: "data".to_string(),
			value: Value::Float {
				value: *self,
				range: -10000.0..=10000.0,
				logarithmic: false,
			},
		}]
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
		vec![Parameter {
			identifier: "data".to_string(),
			value: Value::Serialized(self.as_bytes().to_vec()),
		}]
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
pub fn to_binary<T: serde::Serialize>(value: &T) -> Vec<u8> {
	let mut output = vec![];
	into_writer(value, &mut output).expect("Can not serialized input value");
	output
}

/// A helper function
/// 
/// It can be used to convert a binary format to any type implementing the `Deserialize` trait.
/// We will use cbor as the binary format.
/// 
/// # Panics
/// 
/// This function will panic if the data cannot be deserialized.
pub fn from_binary<T: serde::de::DeserializeOwned>(data: Vec<u8>) -> T {
	let slice = data.as_slice();
	from_reader(slice).expect("Can note deserialized input value")
}

/// A simple wrapper around a sturct makes it possible to have a sturct with multiple parameter setters.
/// 
/// Though you need set params manually, it can be useful when you want to parameterize a complex struct.
pub struct Paramed<T: Parameters> {
	/// The struct that we want to parameterize.
	pub value: T,
	params: ParamMap,
	// sync_set_cached: HashSet<usize>,
}

impl<T: Parameters> Parameters for Paramed<T> {
	fn get_parameters(&self) -> Vec<Parameter> {
		self.value.get_parameters()
	}

	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		self.value.set_parameter(identifier, value)
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
	pub fn new(to_parameterize: T) -> Self {
		let mut empty_self = Self {
			value: to_parameterize,
			params: ParamMap {
				values: Arc::new(Vec::new()),
				id: Arc::new(BiMap::new()),
				// update_queue: Arc::new(SegQueue::new()),
			},
			// sync_set_cached: HashSet::new(),
		};
		empty_self.update_map();
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
	/// Ussful for those structs that have unfixed number of parameters.
	/// 
	/// Note: this function will **not** affect the previous [`ParamMap`] got from [`param_map`].
	pub fn update_map(&mut self) {
		let params = self.value.get_parameters();
		let mut new_map = BiMap::with_capacity(params.len());
		let mut new_values = Vec::with_capacity(params.len());

		for (i, param) in params.into_iter().enumerate() {
			new_map.insert(param.identifier, i);
			new_values.push(Arc::new(param.value.to_atomic_value()));
		}

		self.params = ParamMap {
			values: Arc::new(new_values),
			id: Arc::new(new_map),
			// update_queue: Arc::new(SegQueue::new()),
		}
	}
}

#[derive(Debug, Clone)]
/// A helper struct for managing parameter values for using [`Paramed`] struct.
/// 
/// This struct should be cheap to clone and will assosiated with one specific parameterized object.
pub struct ParamMap {
	values: Arc<Vec<Arc<AtomicValue>>>,
	id: Arc<BiMap<String, usize>>,
}

impl ParamMap {
	/// Get the parameter value by its identifier.
	pub fn get(&self, id: &str) -> Option<Arc<AtomicValue>> {
		let index = self.id.get_by_left(id)?;
		let value = self.values.get(*index)?;
		// self.update_queue(*index);
		
		Some(value.clone())
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

	// fn update_queue(&self, index: usize) {
	// 	let mut updated_set = HashSet::new();

	// 	while let Some(id) = self.update_queue.pop() {
	// 		updated_set.insert(id);
	// 	}

	// 	updated_set.insert(index);

	// 	for id in updated_set {
	// 		self.update_queue.push(id);
	// 	}
	// }
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
		println!("{:#?}", params_output);
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
		params.set_parameter("d.h", SetValue::Serialized(to_binary(&(2.0, 3.0))));
		assert_eq!(params.d.h, (2.0, 3.0));
	}
}