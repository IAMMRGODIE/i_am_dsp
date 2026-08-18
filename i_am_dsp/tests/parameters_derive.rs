//! Integration tests for the `#[derive(Parameters)]` proc macro from `i_am_dsp_derive`.
//!
//! Covers: named structs, tuple structs, empty structs, `#[sub_param]` nesting,
//! `#[skip]`, struct-level `#[id(prefix = "...")]`, `#[serde]` / `#[persist]`
//! fields and index-based setting.

use i_am_dsp::prelude::{Parameter, Parameters, SetValue, Value, to_binary};
use i_am_dsp_derive::Parameters; // derive macro (distinct from the trait of the same name)

fn ids(params: &[Parameter]) -> Vec<&str> {
	params.iter().map(|p| p.identifier.as_str()).collect()
}

// ---------- named struct ----------

#[derive(Parameters)]
#[default_float_range(min = 0.0, max = 1.0)]
struct Named {
	volume: f32,
	#[range(min = 2, max = 8)]
	voices: i32,
	#[range(min = 0.01, max = 100.0)]
	#[logarithmic]
	freq: f32,
	on: bool,
	#[skip]
	_cache: Vec<u8>,
}

#[test]
fn named_struct() {
	let mut p = Named { volume: 0.5, voices: 4, freq: 1.0, on: true, _cache: vec![] };
	let params = p.get_parameters();
	assert_eq!(ids(&params), vec!["volume", "voices", "freq", "on"]);
	assert_eq!(
		params[2].value,
		Value::Float { value: 1.0, range: 0.01..=100.0, logarithmic: true }
	);

	assert!(p.set_parameter("volume", SetValue::Float(0.8)));
	assert_eq!(p.volume, 0.8);
	assert!(p.set_parameter("voices", SetValue::Int(6)));
	assert_eq!(p.voices, 6);
	assert!(p.set_parameter("on", SetValue::Bool(false)));
	assert!(!p.on);

	// wrong type / unknown identifier / out-of-range index all fail
	assert!(!p.set_parameter("volume", SetValue::Bool(true)));
	assert!(!p.set_parameter("nope", SetValue::Float(0.0)));
	assert!(!p.set_parameter_by_index(99, SetValue::Float(0.0)));

	assert!(p.set_parameter_by_index(0, SetValue::Float(0.25)));
	assert_eq!(p.volume, 0.25);
}

// ---------- tuple struct ----------

#[derive(Parameters)]
#[allow(dead_code)]
struct Tuple(
	#[range(min = -1.0, max = 1.0)] f32,
	#[id(name = "mix")] f32,
	#[skip] Vec<f32>,
	f32,
	i32,
	bool,
);

#[test]
fn tuple_struct() {
	let mut p = Tuple(0.1, 0.2, vec![], 0.3, 4, false);
	let params = p.get_parameters();
	assert_eq!(ids(&params), vec!["0", "mix", "2", "3", "4"]);

	// address by positional identifier
	assert!(p.set_parameter("0", SetValue::Float(0.9)));
	assert_eq!(p.0, 0.9);
	assert!(p.set_parameter("mix", SetValue::Float(1.0)));
	assert_eq!(p.1, 1.0);
	assert!(p.set_parameter("2", SetValue::Float(0.7)));
	assert_eq!(p.3, 0.7);
	assert!(p.set_parameter("3", SetValue::Int(7)));
	assert_eq!(p.4, 7);
	assert!(p.set_parameter("4", SetValue::Bool(true)));
	assert!(p.5);
	// type mismatch rejected
	assert!(!p.set_parameter("0", SetValue::Int(1)));

	// address by positional index
	assert!(p.set_parameter_by_index(1, SetValue::Float(0.5)));
	assert_eq!(p.1, 0.5);
	assert!(p.set_parameter_by_index(4, SetValue::Bool(false)));
	assert!(!p.5);
}

// ---------- sub parameters ----------

#[derive(Parameters)]
struct Sub {
	#[range(min = 0.0, max = 1.0)]
	a: f32,
	#[range(min = 0.0, max = 1.0)]
	b: f32,
}

#[derive(Parameters)]
struct WithNamedSub {
	#[range(min = 0.0, max = 1.0)] volume: f32,
	#[sub_param] sub: Sub,
	#[range(min = 0.0, max = 1.0)] pan: f32,
}

#[derive(Parameters)]
struct WithTupleSub(
	#[range(min = 0.0, max = 1.0)] f32,
	#[sub_param] Sub,
	#[range(min = 0.0, max = 1.0)] f32,
);

#[test]
fn sub_parameters() {
	let mut a = WithNamedSub { volume: 0.1, sub: Sub { a: 0.2, b: 0.3 }, pan: 0.4 };
	let params = a.get_parameters();
	assert_eq!(ids(&params), vec!["volume", "sub.a", "sub.b", "pan"]);
	assert!(a.set_parameter("sub.b", SetValue::Float(0.9)));
	assert_eq!(a.sub.b, 0.9);
	assert!(a.set_parameter("pan", SetValue::Float(0.6)));
	assert_eq!(a.pan, 0.6);

	let mut b = WithTupleSub(0.1, Sub { a: 0.2, b: 0.3 }, 0.4);
	let params = b.get_parameters();
	assert_eq!(ids(&params), vec!["0", "1.a", "1.b", "2"]);
	assert!(b.set_parameter("1.a", SetValue::Float(0.5)));
	assert_eq!(b.1.a, 0.5);

	// by index walks the flattened parameter list, including sub params
	assert!(b.set_parameter_by_index(2, SetValue::Float(0.7)));
	assert_eq!(b.1.b, 0.7);
}

// ---------- empty structs ----------

#[derive(Parameters)]
struct EmptyNamed {}

#[derive(Parameters)]
struct EmptyTuple();

#[derive(Parameters)]
struct UnitStruct;

#[test]
fn empty_structs() {
	assert!(UnitStruct.get_parameters().is_empty());
	assert!(EmptyNamed {}.get_parameters().is_empty());
	assert!(EmptyTuple().get_parameters().is_empty());
	assert!(!UnitStruct.set_parameter("x", SetValue::Float(0.0)));
	assert!(!EmptyTuple().set_parameter_by_index(0, SetValue::Float(0.0)));
}

// ---------- struct-level id prefix ----------

#[derive(Parameters)]
#[id(prefix = "synth_")]
struct Prefixed {
	#[range(min = 0.0, max = 1.0)] cutoff: f32,
	#[id(name = "reset")] on: bool,
}

#[test]
fn struct_level_id_prefix() {
	let mut p = Prefixed { cutoff: 0.3, on: false };
	let params = p.get_parameters();
	assert_eq!(ids(&params), vec!["synth_cutoff", "reset"]);
	assert!(p.set_parameter("synth_cutoff", SetValue::Float(0.9)));
	assert_eq!(p.cutoff, 0.9);
	assert!(p.set_parameter("reset", SetValue::Bool(true)));
	assert!(p.on);
}

// ---------- serde / persist ----------

#[derive(Parameters)]
struct SerdeField {
	#[serde]
	pos: (f32, f32),
}

#[test]
fn serde_field() {
	let mut s = SerdeField { pos: (1.0, 2.0) };
	let params = s.get_parameters();
	assert_eq!(params[0].identifier, "pos");
	if let Value::Serialized(data) = &params[0].value {
		assert!(!data.is_empty());
	} else {
		panic!("expected a serialized value");
	}

	// round-trip through the serialized form
	let bytes = to_binary(&(3.0, 4.0)).expect("serialize should work");
	assert!(s.set_parameter("pos", SetValue::Serialized(bytes)));
	assert_eq!(s.pos, (3.0, 4.0));
	assert!(!s.set_parameter("pos", SetValue::Float(0.0)));
}

fn str_to_bytes(s: &String) -> Vec<u8> {
	s.as_bytes().to_vec()
}

fn bytes_to_str(b: Vec<u8>) -> String {
	String::from_utf8_lossy(&b).to_string()
}

#[derive(Parameters)]
struct PersistField {
	#[persist(serialize = "str_to_bytes", deserialize = "bytes_to_str")]
	name: String,
}

#[test]
fn persist_field() {
	let mut p = PersistField { name: "hello".to_string() };
	let params = p.get_parameters();
	assert_eq!(params[0].identifier, "name");
	if let Value::Serialized(data) = &params[0].value {
		assert_eq!(data, b"hello");
	} else {
		panic!("expected a serialized value");
	}

	assert!(p.set_parameter("name", SetValue::Serialized(b"world".to_vec())));
	assert_eq!(p.name, "world");
	assert!(!p.set_parameter("name", SetValue::Float(0.0)));
}

// ---------- small integer types ----------

#[derive(Parameters)]
struct SmallInts {
	#[range(min = -128, max = 127)] a: i8,
	#[range(min = 0, max = 255)] b: u8,
	#[range(min = -32768, max = 32767)] c: i16,
	#[range(min = 0, max = 65535)] d: u16,
	x: u64,
}

#[test]
fn small_int_types() {
	let mut p = SmallInts { a: 1, b: 2, c: 3, d: 4, x: 5 };
	let params = p.get_parameters();
	assert_eq!(ids(&params), vec!["a", "b", "c", "d", "x"]);

	assert!(p.set_parameter("a", SetValue::Int(-42)));
	assert_eq!(p.a, -42);
	assert!(p.set_parameter("b", SetValue::Int(200)));
	assert_eq!(p.b, 200);
	assert!(p.set_parameter("c", SetValue::Int(-3000)));
	assert_eq!(p.c, -3000);
	assert!(p.set_parameter("d", SetValue::Int(60000)));
	assert_eq!(p.d, 60000);
	assert!(p.set_parameter("x", SetValue::Int(123)));
	assert_eq!(p.x, 123);
}

// ---------- path-qualified primitive type ----------

#[derive(Parameters)]
struct PathTyped {
	#[range(min = 0.0, max = 1.0)] a: std::primitive::f32,
}

#[test]
fn path_qualified_primitive() {
	let mut p = PathTyped { a: 0.5 };
	assert_eq!(ids(&p.get_parameters()), vec!["a"]);
	assert!(p.set_parameter("a", SetValue::Float(0.75)));
	assert_eq!(p.a, 0.75);
}

// ---------- same-named structs in different modules (regression: generated code
// must not create module-level items, otherwise these two would collide) ----------

mod teacher {
	use i_am_dsp_derive::Parameters;

	#[derive(Parameters)]
	pub struct Same {
		#[range(min = 0.0, max = 1.0)] pub a: f32,
	}
}

mod student {
	use i_am_dsp_derive::Parameters;

	#[derive(Parameters)]
	pub struct Same {
		#[range(min = 0.0, max = 1.0)] pub b: f32,
	}
}

#[test]
fn same_named_structs_in_different_modules() {
	let mut t = teacher::Same { a: 0.1 };
	let mut s = student::Same { b: 0.2 };
	assert_eq!(ids(&t.get_parameters()), vec!["a"]);
	assert_eq!(ids(&s.get_parameters()), vec!["b"]);
	assert!(t.set_parameter("a", SetValue::Float(0.3)));
	assert!(s.set_parameter("b", SetValue::Float(0.4)));
	assert_eq!(t.a, 0.3);
	assert_eq!(s.b, 0.4);
}

// ---------- large int range error case is a compile-time error and can't be
// tested here; instead verify a legitimately large-but-fitting range works ----------

#[derive(Parameters)]
#[default_int_range(min = 1, max = 2147483647)]
struct WideInt {
	count: i32,
}

#[test]
fn wide_int_range() {
	let mut p = WideInt { count: 5 };
	let params = p.get_parameters();
	if let Value::Int { range, .. } = &params[0].value {
		assert_eq!(*range.start(), 1);
		assert_eq!(*range.end(), 2_147_483_647);
	} else {
		panic!("expected int value");
	}
	assert!(p.set_parameter("count", SetValue::Int(123456)));
	assert_eq!(p.count, 123456);
}
