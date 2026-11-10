//! A helper proc macro that derives the `Parameters` trait for structs.
//!
//! Supports named structs, tuple (positional) structs and empty / unit structs.

#![warn(missing_docs)]

use proc_macro::TokenStream;
use syn::{DeriveInput, parse_macro_input};

#[proc_macro_derive(
	Parameters,
	attributes(
		range,
		id,
		persist,
		serde,
		skip,
		sub_param,
		logarithmic,
		default,
		unit,
		choices,
		layout,
		default_float_range,
		default_int_range,
		default_uint_range
	)
)]
/// A helper macro to derive the `Parameters` trait for a struct (named fields, tuple/positional fields, or empty).
///
/// # Syntax
///
/// ```ignore
/// #[derive(Parameters)]
/// #[default_float_range(min = 0.0, max = 1.0)]
/// #[default_int_range(min = 0, max = 10)]
/// #[default_uint_range(min = 0, max = 10)]
/// struct MyParameters {
///     #[range(min = 0.0, max = 1.0)]
///     a: f32,
///     #[range(min = 1, max = 10)]
///     b: i32,
///     #[id(name = "Foo")]
///     #[persist(serialize = "string_to_binary", deserialize = "binary_to_string")]
///     c: String,
///
///     #[sub_param]
///     d: SubParameters,
///
///     boo: bool,
///
///     #[skip]
///     _e: Vec<f32>,
/// }
///
/// #[derive(Parameters)]
/// struct SubParameters {
///     #[range(min = 0.0, max = 1.0)]
///     #[id(prefix = "sub_")]
///     f: f32,
///     #[range(min = 0, max = 1)]
///     g: i32,
///     #[logarithmic]
///     h: f32,
///     #[serde]
///     i: (f32, f32),
/// }
///
/// fn string_to_binary(s: &String) -> Vec<u8> {
///     s.as_bytes().to_vec()
/// }
///
/// fn binary_to_string(b: Vec<u8>) -> String {
///     String::from_utf8_lossy(&b).to_string()
/// }
/// ```
///
/// ## Tuple structs
///
/// Tuple structs (positional / "anonymous" fields) are supported too; fields are
/// addressed by their parameter position (`"0"`, `"1"`, ...) unless renamed with `#[id(name = "...")]`:
///
/// ```ignore
/// #[derive(Parameters)]
/// #[default_float_range(min = 0.0, max = 1.0)]
/// struct MyTuple(
///     #[range(min = -1.0, max = 1.0)] f32,
///     #[id(name = "mix")] f32,
///     #[skip] Vec<f32>,
///     #[sub_param] SubParameters,
///     bool,
/// );
/// ```
///
/// Unit structs (`struct Foo;`), empty tuple structs (`struct Foo();`) and empty
/// named structs (`struct Foo {}`) are also supported and simply expose no parameters.
///
/// ## Identifiers and indices
///
/// Every parameter has a string `identifier`. For plain fields the identifier is the
/// field name (or the parameter position for tuple fields), optionally prefixed by a
/// struct-level `#[id(prefix = "...")]` or renamed by a field-level `#[id(name = "...")]`.
/// `#[sub_param]` fields contribute multiple parameters with dotted identifiers like
/// `"sub.a"`, `"sub.b"`, ....
///
/// `Parameters::set_parameter_by_index` addresses parameters by their position in the
/// flattened list returned by `get_parameters()` (sub-parameters count as individual
/// entries). The derive never relies on those positions for its own dispatch; identifier
/// lookup is the single source of truth, so the two are always consistent.
///
/// ## Attributes
///
/// - `range`: A range attribute that specifies the minimum and maximum values of the parameter.
/// - `id`: An id attribute that specifies the name of the parameter. Can also be used on the struct itself as `#[id(prefix = "...")]` to prefix every parameter identifier. A field-level `#[id(stable = 4021)]` additionally pins the id the host knows the parameter by, so it survives the field being renamed.
/// - `persist`: A persist attribute that specifies how to serialize and deserialize the parameter.
/// - `serde`: Use serde to serialize and deserialize the parameter.
/// - `skip`: A skip attribute that specifies that the field should be skipped when deriving the `Parameters` trait.
/// - `sub_param`: A sub_param attribute that specifies that the field is a sub-parameter struct.
/// - `logarithmic`: A logarithmic attribute that specifies that the parameter should be displayed in logarithmic scale. A range that reaches zero or crosses it uses a smooth symmetric-log curve instead of `ln`.
/// - `default`: The value the parameter starts at, so that a host can offer "reset to default" instead of guessing the minimum: `#[default(0.8)]`, `#[default(false)]`.
///
///   **Declare it wherever there is an answer to give.** A host that is not told falls back to the bottom of the range, which is rarely where an effect starts: a synced time comes back as a thirty-second note instead of a beat, a gain at its quietest instead of at unity. The one case that should be left alone is a parameter whose starting value is an argument to the constructor — a delay time, a sample rate — because then there is no single answer to write down.
/// - `unit`: How a host should show the value as text: `#[unit(beats)]` renders `0.5` as `1/8`, and `#[unit(milliseconds)]` appends `ms`. One of `none`, `milliseconds` (or `ms`), `hertz` (or `hz`), `beats`, `seconds` (or `s`), `samples`, `decibels` (or `db`) and `percent`. A percentage is a *fraction of one*: `0.5` reads back and forth as `50.0 %`.
///
///   Declaring a unit is what keeps a host from having to guess: a value with one is shown with it, and a value that declares none stays a plain number. Not every parameter has one to declare — a gain, a ratio, a mix — and those are fine as they are.
/// - layout: Where the parameter wants to sit in a faceplate, and how it wants to be
///   drawn: which group it belongs to, where it sits inside it, whether it wants a knob or
///   a slider, how much room it wants, whether the middle of its range is its neutral
///   point, and the interval its values snap to. Every one of them is a hint, and a
///   faceplate is free to ignore all of them:
///   `#[layout(group = "Time", order = 0, widget = knob, size = large, bipolar, step = 0.25)]`.
///   The options are group, order, widget (auto, knob, slider, toggle, choice), size
///   (small, medium, large), bipolar and step.
///
/// - `id(display = "...")`: What a host should *call* the parameter, when the identifier is not what a person should read: `#[id(display = "BPM Sync")]`. It never touches the identifier, so it can sit on a `sub_param`, where it names the level those parameters sit under: a `Beats` field under a `Tempo` sub-parameter is shown as `Tempo Beats`. A level that declares nothing contributes its field name, so the worst a host sees is `tempo beats` rather than `tempo.beats`.
/// - `default_float_range`: A default range attribute that specifies the default minimum and maximum values of the parameter for float type.
/// - `default_int_range`: A default range attribute that specifies the default minimum and maximum values of the parameter for signed integer type.
/// - `default_uint_range`: A default range attribute that specifies the default minimum and maximum values of the parameter for unsigned integer type.
///
/// by default, default_float_range is (0.0, 1.0), default_int_range is (0, 256), default_uint_range is (0, 256)
pub fn derive_parameters(input: TokenStream) -> TokenStream {
	let input = parse_macro_input!(input as DeriveInput);

	match impl_config(&input) {
		Ok(tokens) => tokens,
		Err(err) => err.to_compile_error().into(),
	}
}

mod expand;
mod parse;

use expand::impl_config;
