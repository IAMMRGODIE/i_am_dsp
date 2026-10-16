//! Integration tests for the `#[derive(Parameters)]` proc macro from `i_am_dsp_derive`.
//!
//! Covers: named structs, tuple structs, empty structs, `#[sub_param]` nesting,
//! `#[skip]`, struct-level `#[id(prefix = "...")]`, `#[serde]` / `#[persist]`
//! fields and index-based setting.

use i_am_dsp::prelude::{
	Parameter, ParameterMeta, Parameters, Paramed, SetValue, SyncedTime, Unit, Value, from_binary, to_binary,
};
#[cfg(feature = "layout")]
use i_am_dsp::prelude::ParamWidget;
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

// ---------- stable parameter ids ----------

#[derive(Parameters)]
struct IdsBefore {
	#[range(min = 0.0, max = 1.0)] alpha: f32,
	#[range(min = 0.0, max = 1.0)] beta: f32,
	#[range(min = 0.0, max = 1.0)] gamma: f32,
}

#[derive(Parameters)]
struct IdsAfter {
	#[range(min = 0.0, max = 1.0)] alpha: f32,
	#[range(min = 0.0, max = 1.0)] inserted: f32,
	#[range(min = 0.0, max = 1.0)] beta: f32,
	#[range(min = 0.0, max = 1.0)] gamma: f32,
}

/// A host keys automation and modulation on the id it is given, and CLAP forbids changing
/// the id of an existing parameter, so inserting a parameter must not move the others.
#[test]
fn stable_ids_survive_inserting_a_parameter() {
	let before = Paramed::new(IdsBefore { alpha: 0.0, beta: 0.0, gamma: 0.0 }).param_map();
	let after = Paramed::new(IdsAfter { alpha: 0.0, inserted: 0.0, beta: 0.0, gamma: 0.0 }).param_map();

	let id_of = |param_map: &i_am_dsp::prelude::ParamMap, identifier: &str| {
		let index = param_map.query_param_index(identifier).expect("parameter should exist");
		param_map.query_stable_id(index).expect("parameter should have a stable id")
	};

	// The insertion does move the position...
	assert_eq!(before.query_param_index("beta"), Some(1));
	assert_eq!(after.query_param_index("beta"), Some(2));

	// ...but not the id the earlier parameters hand out.
	for identifier in ["alpha", "beta", "gamma"] {
		assert_eq!(
			id_of(&before, identifier),
			id_of(&after, identifier),
			"`{identifier}` changed id when a parameter was inserted before it"
		);
	}

	// The parameter that was inserted is the only one whose id is new.
	let inserted = id_of(&after, "inserted");
	for identifier in ["alpha", "beta", "gamma"] {
		assert_ne!(inserted, id_of(&before, identifier));
	}
}

// ---------- pinned (explicit) stable ids ----------

#[derive(Parameters)]
struct Pinned {
	#[range(min = 0.0, max = 1.0)]
	#[id(stable = 4021)]
	value: f32,
	#[range(min = 0.0, max = 1.0)] other: f32,
}

/// The same parameter after its field was renamed: the identifier moved, the pin did not.
#[derive(Parameters)]
struct PinnedRenamed {
	#[range(min = 0.0, max = 1.0)]
	#[id(stable = 4021)]
	renamed: f32,
	#[range(min = 0.0, max = 1.0)] other: f32,
}

#[derive(Parameters)]
struct PinnedOuter {
	#[sub_param] inner: Pinned,
	#[range(min = 0.0, max = 1.0)] top: f32,
}

#[test]
fn pinned_ids_are_handed_to_the_host() {
	let param_map = Paramed::new(Pinned { value: 0.0, other: 0.0 }).param_map();

	assert_eq!(param_map.query_stable_id(0), Some(4021));
	assert_eq!(param_map.query_param_index_by_stable_id(4021), Some(0));
	// A parameter that pins nothing still follows the identifier scheme.
	assert_eq!(
		param_map.query_stable_id(1),
		Some(i_am_dsp::prelude::stable_parameter_id("other"))
	);
}

/// The point of pinning: the id is what the host stores, so the name is free to change.
#[test]
fn a_pinned_id_survives_a_rename() {
	let before = Paramed::new(Pinned { value: 0.0, other: 0.0 }).param_map();
	let after = Paramed::new(PinnedRenamed { renamed: 0.0, other: 0.0 }).param_map();

	assert_ne!(
		i_am_dsp::prelude::stable_parameter_id("value"),
		i_am_dsp::prelude::stable_parameter_id("renamed"),
		"the rename has to actually move the derived id for this test to mean anything"
	);
	assert_eq!(before.query_stable_id(0), after.query_stable_id(0));
	assert_eq!(after.query_stable_id(0), Some(4021));
}

#[test]
fn pinned_ids_nest_with_their_prefix() {
	let param_map = Paramed::new(PinnedOuter {
		inner: Pinned { value: 0.0, other: 0.0 },
		top: 0.0,
	}).param_map();

	let index = param_map.query_param_index("inner.value").expect("the nested parameter exists");
	assert_eq!(param_map.query_stable_id(index), Some(4021));
}
// ---------- parameter metadata ----------

#[derive(Parameters)]
struct WithMeta {
	#[range(min = 0.03125, max = 32.0)]
	#[logarithmic]
	#[default(1.0)]
	#[unit(beats)]
	beats: f32,
	#[default(true)]
	on: bool,
	#[range(min = 0, max = 8)]
	#[default(4)]
	voices: i32,
	#[range(min = 0.0, max = 4000.0)]
	#[unit(milliseconds)]
	time: f32,
}

#[test]
fn a_parameter_carries_its_default_and_its_unit() {
	let parameters = WithMeta { beats: 0.0, on: false, voices: 0, time: 0.0 }.get_parameters();
	assert_eq!(ids(&parameters), vec!["beats", "on", "voices", "time"]);

	assert_eq!(parameters[0].meta.default, Some(1.0));
	assert_eq!(parameters[0].meta.unit, Unit::Beats);

	// A bool default is stated as one, and read back as the number a host wants.
	assert_eq!(parameters[1].meta.default, Some(1.0));
	assert_eq!(parameters[1].meta.unit, Unit::None);
	assert_eq!(parameters[2].meta.default, Some(4.0));

	// Nothing declared: the host falls back to its own idea of a default.
	assert_eq!(parameters[3].meta.default, None);
	assert_eq!(parameters[3].meta.unit, Unit::Milliseconds);
}

/// The metadata describes the code, not the project, so it is not what a saved state carries.
#[test]
fn metadata_is_not_part_of_the_saved_state() {
	let parameters = WithMeta { beats: 2.0, on: true, voices: 3, time: 250.0 }.get_parameters();
	let loaded: Vec<Parameter> = from_binary(to_binary(&parameters).expect("serialize")).expect("deserialize");

	assert_eq!(loaded.len(), parameters.len());
	assert_eq!(loaded[0].meta, ParameterMeta::default());
	assert_eq!(loaded[0].identifier, "beats");
	// ...but the values are, which is the point of saving state at all.
	assert_eq!(loaded[0].value, parameters[0].value);
}
// ---------- display names ----------

#[derive(Parameters)]
struct PlainInner {
	x: f32,
}

#[derive(Parameters)]
struct PlainOuter {
	#[sub_param]
	inner: PlainInner,
}

#[derive(Parameters)]
struct NamedWithDisplay {
	/// The real thing, so the composition is exercised through it rather than a stand-in.
	#[sub_param]
	#[id(display = "Tempo")]
	tempo: SyncedTime,
	#[id(display = "BPM Sync")]
	#[range(min = 0.0, max = 1.0)]
	own: f32,
}

/// A display name is what a host shows; the identifier is what it addresses, and the two
/// are independent.
#[test]
fn a_display_name_reaches_the_host_without_touching_the_identifier() {
	let parameters = NamedWithDisplay { tempo: SyncedTime::default(), own: 0.5 }.get_parameters();

	// The identifiers are exactly what they would be without any display name.
	assert_eq!(ids(&parameters), vec!["tempo.sync", "tempo.beats", "own"]);

	// A nested parameter is named by the level that contains it.
	assert_eq!(parameters[0].meta.display_name.as_deref(), Some("Tempo Sync"));
	assert_eq!(parameters[1].meta.display_name.as_deref(), Some("Tempo Beats"));
	assert_eq!(parameters[2].meta.display_name.as_deref(), Some("BPM Sync"));

	// A display name on its own leaves everything else alone, including the default
	// identifier a field would otherwise get.
	assert_eq!(parameters[2].identifier, "own");
	assert_eq!(parameters[2].meta.unit, Unit::None);
	assert_eq!(parameters[2].meta.default, None);
}

/// A level that declares no name contributes its field name, so a host still gets words
/// rather than nothing.
#[test]
fn a_nested_parameter_without_names_is_labelled_by_its_fields() {
	let parameters = PlainOuter { inner: PlainInner { x: 0.0 } }.get_parameters();
	assert_eq!(ids(&parameters), vec!["inner.x"]);
	assert_eq!(parameters[0].meta.display_name.as_deref(), Some("inner x"));
}

// ---------- layout ----------

#[cfg(feature = "layout")]
#[derive(Parameters)]
struct LaidOutChild {
	#[range(min = 0.0, max = 4.0)]
	beats: f32,
	#[range(min = 0.0, max = 1.0)]
	#[layout(group = "Elsewhere", order = 0, widget = slider)]
	own: f32,
}

#[cfg(feature = "layout")]
#[derive(Parameters)]
struct LaidOutParent {
	#[sub_param]
	#[layout(group = "Time", order = 3, widget = knob)]
	tempo: LaidOutChild,
	#[range(min = 0.0, max = 1.0)]
	#[layout(group = "Time", order = 4)]
	amount: f32,
}

/// A nested level sits where its parent sits, unless the child asked for somewhere
/// of its own: five controls that all have to be placed by hand is how a faceplate
/// stops being automatic.
#[cfg(feature = "layout")]
#[test]
fn a_sub_parameter_inherits_the_place_its_parent_asked_for() {
	let parent = LaidOutParent {
		tempo: LaidOutChild { beats: 1.0, own: 0.5 },
		amount: 0.5,
	};
	let parameters = parent.get_parameters();
	let layout = |id: &str| {
		parameters
			.iter()
			.find(|parameter| parameter.identifier == id)
			.unwrap_or_else(|| panic!("{id} is a parameter"))
			.meta
			.layout
			.clone()
	};

	let beats = layout("tempo.beats");
	assert_eq!(beats.group.as_deref(), Some("Time"), "a child with no place of its own inherits its parent's");
	assert_eq!(beats.order, Some(3));

	let own = layout("tempo.own");
	assert_eq!(own.group.as_deref(), Some("Elsewhere"), "a child that asked for its own place keeps it");
	assert_eq!(own.widget, ParamWidget::Slider, "and keeps everything else it asked for");

	assert_eq!(layout("amount").order, Some(4), "a sibling still sorts after the whole nested level");
}
