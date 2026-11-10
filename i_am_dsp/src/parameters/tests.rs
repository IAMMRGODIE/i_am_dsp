use super::*;
use crate::parameters::traits::hash_identifier;

use std::sync::Arc;
use std::sync::atomic::Ordering;

use i_am_dsp_derive::Parameters;

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
	_e: Vec<f32>,
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

#[cfg(feature = "layout")]
#[derive(Parameters, Default)]
struct LaidOut {
	#[range(min = 0.0, max = 1.0)]
	#[layout(group = "Time", order = 0, widget = knob, size = large)]
	time: f32,
	#[range(min = -1.0, max = 1.0)]
	#[layout(group = "Time", order = 1, widget = slider, bipolar)]
	pan: f32,
	#[range(min = 0.0, max = 16.0)]
	#[layout(group = "Voices", widget = slider, size = small, step = 1.0)]
	voices: f32,
	#[range(min = 0.0, max = 1.0)]
	plain: f32,
}

/// A parameter can say where it wants to sit and how it wants to be drawn, and
/// one that says nothing keeps an empty layout.
#[cfg(feature = "layout")]
#[test]
fn a_layout_reaches_the_metadata() {
	let parameters = LaidOut::default().get_parameters();
	let layout = |id: &str| {
		parameters
			.iter()
			.find(|parameter| parameter.identifier == id)
			.expect("a declared parameter")
			.meta
			.layout
			.clone()
	};

	let time = layout("time");
	assert_eq!(time.group.as_deref(), Some("Time"));
	assert_eq!(time.order, Some(0));
	assert_eq!(time.widget, ParamWidget::Knob);
	assert_eq!(time.size, ParamSize::Large);
	assert!(!time.bipolar);
	assert_eq!(time.step, None);

	let pan = layout("pan");
	assert_eq!(pan.widget, ParamWidget::Slider);
	assert!(pan.bipolar);
	assert_eq!(pan.order, Some(1));

	let voices = layout("voices");
	assert_eq!(voices.step, Some(1.0));
	assert_eq!(voices.size, ParamSize::Small);

	assert!(
		layout("plain").is_empty(),
		"a parameter that declares nothing keeps an empty layout"
	);
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
	assert_eq!(
		stable_parameter_id("delay_time"),
		stable_parameter_id("delay_time")
	);
	assert_ne!(
		stable_parameter_id("delay_time"),
		stable_parameter_id("decay_factor")
	);

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
	for beats in [
		0.25,
		0.5,
		0.75,
		1.0,
		1.5,
		2.0,
		3.0,
		4.0,
		1.0 / 3.0,
		2.0 / 3.0,
	] {
		let text = Unit::Beats.format(beats);
		let parsed = Unit::Beats
			.parse(&text)
			.expect("what it wrote has to read back");
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
		.map(|index| {
			param_map
				.query_stable_id(index)
				.expect("every index has a stable id")
		})
		.collect::<Vec<_>>();

	let mut distinct = ids.clone();
	distinct.sort_unstable();
	distinct.dedup();
	assert_eq!(
		distinct.len(),
		ids.len(),
		"two parameters were given the same stable id"
	);

	for (index, id) in ids.iter().enumerate() {
		assert_eq!(param_map.query_stable_id(index), Some(*id));
		assert_eq!(param_map.query_param_index_by_stable_id(*id), Some(index));
		assert!(param_map.get_by_stable_id(*id).is_some());
	}

	// The derived identifier is what the id is derived from, so the two have to agree.
	assert_eq!(param_map.query_stable_id(0), Some(stable_parameter_id("a")));
	assert_eq!(
		param_map.query_stable_id(2),
		Some(stable_parameter_id("Foo"))
	);
	assert_eq!(
		param_map.query_stable_id(3),
		Some(stable_parameter_id("d.f"))
	);
	assert_eq!(
		param_map.query_stable_id(6),
		Some(stable_parameter_id("boo"))
	);

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
		(
			identifier.to_string(),
			Arc::new(Value::Bool(true).to_atomic_value()),
		)
	};

	let param_map = ParamMap::new_with([entry(FIRST), entry(SECOND)]);
	let first = param_map.query_stable_id(0).unwrap();
	let second = param_map.query_stable_id(1).unwrap();

	assert_ne!(
		first, second,
		"colliding identifiers were aliased onto one id"
	);
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
		(
			identifier.to_string(),
			Arc::new(Value::Bool(true).to_atomic_value()),
		)
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
		(
			identifier.to_string(),
			Arc::new(Value::Bool(true).to_atomic_value()),
		)
	};
	let overrides = [("something_else".to_string(), 4021u32)];

	let entries = [entry("something_else"), entry("other")]
		.map(|(identifier, value)| (identifier, ParameterMeta::default(), value));
	let param_map = ParamMap::build(entries, StableIdScheme::Identifier, &overrides);
	assert_eq!(param_map.query_stable_id(0), Some(4021));
	assert_eq!(param_map.query_param_index_by_stable_id(4021), Some(0));
	assert_eq!(
		param_map.query_stable_id(1),
		Some(stable_parameter_id("other"))
	);
}

/// Pinned ids are handed out before the scheme does, so a pin that lands on a position
/// takes it and the position moves to a derived id. Nothing may alias.
#[test]
fn a_pin_that_lands_on_a_position_moves_that_position() {
	let entry = |identifier: &str| {
		(
			identifier.to_string(),
			Arc::new(Value::Bool(true).to_atomic_value()),
		)
	};
	let overrides = [("one".to_string(), 0u32)];

	let entries = [entry("zero"), entry("one")]
		.map(|(identifier, value)| (identifier, ParameterMeta::default(), value));
	let param_map = ParamMap::build(entries, StableIdScheme::Position, &overrides);

	assert_eq!(param_map.query_stable_id(1), Some(0), "the pinned id wins");
	assert_ne!(
		param_map.query_stable_id(0),
		Some(0),
		"the position has to move, not alias"
	);
	assert_eq!(param_map.query_param_index_by_stable_id(0), Some(1));
}

/// A host writes the map and clears the changed flag: the flag reports the
/// interface to the host, so it must not be what tells the effect to read.
#[test]
fn a_value_the_host_wrote_reaches_the_effect() {
	let mut paramed = Paramed::new(MyParameters::default());
	let map = paramed.param_map();

	map["a"].store(0.75_f32, Ordering::SeqCst);
	map["a"].set_changed(false, Ordering::SeqCst);

	paramed.sync_params();

	assert_eq!(paramed.value.a, 0.75);
}
