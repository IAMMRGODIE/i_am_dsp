//! A helper trait for report parameters change to hosts.

use std::ops::RangeInclusive;

use ciborium::{from_reader, into_writer};
use rustfft::num_complex::Complex;

/// A helper struct for report parameters.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Parameter {
	/// An unique identifier for the parameter.
	pub identifier: String,
	/// The parameter value.
	pub value: Value,
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