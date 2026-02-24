//! useful tools for gui

pub mod waveform;
pub mod adsr_editor;
pub mod unison;
pub mod knob;
pub mod selector;
pub mod slider;
pub mod equalizer;
pub(crate) mod utils;

/// A number trait that provides a conversion from f32 to the number type and vice versa.
pub trait Number {
	/// Converts a f32 to the number type.
	fn from_f32(input: f32) -> Self;
	/// Converts the number type to a f32.
	fn into_f32(self) -> f32;
}

macro_rules! impl_number {
	($num: ty) => {
		impl Number for $num {
			fn from_f32(input: f32) -> Self {
				input as $num
			}

			fn into_f32(self) -> f32 {
				self as f32
			}
		}
	};
}

impl_number!(f32);
impl_number!(f64);
impl_number!(i8);
impl_number!(i16);
impl_number!(i32);
impl_number!(i64);
impl_number!(i128);
impl_number!(u8);
impl_number!(u16);
impl_number!(u32);
impl_number!(u64);
impl_number!(u128);
impl_number!(usize);
impl_number!(isize);