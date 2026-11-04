//! The values a parameter can hold, plain and atomic.

use std::{ops::RangeInclusive, sync::atomic::Ordering};

use portable_atomic::{AtomicBool, AtomicF32, AtomicI32};

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
	Bool {
		/// The actual value.
		value: AtomicBool,
		/// Whether the value has been changed.
		changed: AtomicBool,
	},
	/// A placeholder value, used for parameters that are not yet implemented or is None.
	#[default]
	Nothing,
}

impl AtomicValue {
	/// Loads the value with the given ordering.
	///
	/// # Panics
	///
	/// Panics if order is [`Ordering::Release`] or [`Ordering::AcqRel`].
	pub fn load(&self, order: Ordering) -> SetValue {
		match self {
			Self::Bool { value, .. } => SetValue::Bool(value.load(order)),
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
			(Self::Bool { value: v, changed }, SetValue::Bool(value)) => {
				v.store(value, order);
				changed.store(true, order);
				true
			}
			(
				Self::Float {
					value: v,
					range,
					changed,
					..
				},
				SetValue::Float(value),
			) => {
				let value = value.clamp(*range.start(), *range.end());
				changed.store(true, order);
				v.store(value, order);
				true
			}
			(
				Self::Int {
					value: v,
					range,
					changed,
					..
				},
				SetValue::Int(value),
			) => {
				let value = value.clamp(*range.start(), *range.end());
				v.store(value, order);
				changed.store(true, order);
				true
			}
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
		} else {
			return Err(self.load(fetch_order));
		};

		Ok(self.load(fetch_order))
	}

	/// Sets the changed flag with the given ordering.
	pub fn set_changed(&self, changed: bool, order: Ordering) {
		match self {
			Self::Bool { changed: c, .. } => c.store(changed, order),
			Self::Float { changed: c, .. } => c.store(changed, order),
			Self::Int { changed: c, .. } => c.store(changed, order),
			Self::Nothing => {}
		}
	}

	/// Returns true if the value has been changed.
	pub fn is_chanegd(&self) -> bool {
		match self {
			Self::Bool { changed, .. } => changed.load(Ordering::Relaxed),
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
	#[default]
	Nothing,
}

impl Value {
	/// Returns the value as a set value.
	pub fn to_set_value(self) -> SetValue {
		match self {
			Value::Float { value, .. } => SetValue::Float(value),
			Value::Int { value, .. } => SetValue::Int(value),
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
			Value::Float {
				value,
				range,
				logarithmic,
			} => AtomicValue::Float {
				value: AtomicF32::new(value),
				range,
				logarithmic,
				changed: AtomicBool::new(false),
			},
			Value::Int {
				value,
				range,
				logarithmic,
			} => AtomicValue::Int {
				value: AtomicI32::new(value),
				range,
				logarithmic,
				changed: AtomicBool::new(false),
			},
			Value::Bool(value) => AtomicValue::Bool {
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
	#[default]
	Nothing,
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
