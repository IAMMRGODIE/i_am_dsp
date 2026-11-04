//! A helper trait for report parameters change to hosts.

mod meta;
mod traits;
mod value;

#[cfg(feature = "layout")]
pub use meta::{ParamLayout, ParamSize, ParamWidget};
pub use meta::{Parameter, ParameterMeta, Unit};
pub use traits::{
	ParamMap, Paramed, Parameters, StableIdScheme, from_binary, stable_parameter_id, to_binary,
};
pub use value::{AtomicValue, SetValue, Value};

#[cfg(test)]
mod tests;
