//! A faceplate built from nothing but the parameters.
//!
//! The point of the parameter metadata is that an effect can say where its
//! controls belong and how they want to be drawn, and get a window without anyone
//! laying one out. These two tests hold both halves of that down: the arrangement
//! is the order the parameters asked for, and the drawing is a frame that can be
//! looked at.

use std::path::Path;

use i_am_dsp::prelude::{Parameters, SetValue};
use i_am_dsp_derive::Parameters;
use i_am_dsp_iced::{faceplate, styles};

/// A stand-in for a real effect: four parameters that declare where they want to
/// go, written in an order that is deliberately not the order they asked for.
// A manual Default, not a derived one: the parameter attribute for a starting
// value shares its name with the built-in one that derive(Default) uses.
#[derive(Parameters)]
struct Demo {
	#[range(min = 0.0, max = 1.0)]
	#[default(0.5)]
	#[layout(group = "Time", order = 1, widget = knob)]
	second: f32,
	#[range(min = 0.0, max = 1.0)]
	#[default(0.5)]
	#[layout(group = "Amplitude", order = 0, widget = slider)]
	mix: f32,
	#[range(min = 0.0, max = 1.0)]
	#[default(0.5)]
	#[layout(group = "Time", order = 0, widget = knob)]
	first: f32,
	#[range(min = 0.0, max = 1.0)]
	#[default(0.5)]
	loose: f32,
	#[default(false)]
	#[layout(group = "Amplitude", order = 1, widget = toggle)]
	polarity: bool,
	#[range(min = -12.0, max = 12.0)]
	#[default(0.0)]
	#[layout(group = "Amplitude", order = 2, widget = slider, step = 1.0)]
	semitones: f32,
}

impl Default for Demo {
	fn default() -> Self {
		Self {
			second: 0.5,
			mix: 0.5,
			first: 0.5,
			loose: 0.5,
			polarity: false,
			semitones: 0.0,
		}
	}
}

#[derive(Debug, Clone, PartialEq)]
struct Change(SetValue);

#[test]
fn a_faceplate_arranges_parameters_the_way_they_asked() {
	let parameters = Demo::default().get_parameters();
	let sections = faceplate::arrange(&parameters);

	let names: Vec<_> = sections.iter().map(|(name, _)| *name).collect();
	assert_eq!(
		names,
		vec![Some("Time"), Some("Amplitude"), None],
		"sections appear in the order they were first declared in, and a parameter with no group still gets one"
	);

	let members = |group: Option<&str>| {
		sections
			.iter()
			.find(|(name, _)| *name == group)
			.expect("a section")
			.1
			.iter()
			.map(|parameter| parameter.identifier.as_str())
			.collect::<Vec<_>>()
	};

	assert_eq!(
		members(Some("Time")),
		["first", "second"],
		"inside a section the declared order wins over the order they were written in"
	);
	assert_eq!(members(Some("Amplitude")), ["mix", "polarity", "semitones"]);
	assert_eq!(members(None), ["loose"], "a parameter that declares no order keeps its place");
}

#[test]
fn a_faceplate_is_drawn_from_the_metadata() {
	let parameters = Demo::default().get_parameters();
	let mut ui = iced_test::simulator(faceplate::faceplate(&parameters, |_, value| Change(value)));

	let snapshot = match ui.snapshot(&styles::theme()) {
		Ok(snapshot) => snapshot,
		Err(error) => panic!("the faceplate renders: {error:?}"),
	};

	let path = Path::new(env!("CARGO_MANIFEST_DIR"))
		.join("tests")
		.join("snapshots")
		.join("faceplate.png");

	assert!(
		snapshot.matches_image(&path).expect("the frame is written"),
		"the faceplate no longer matches its reference image; delete {} and run again to accept the change",
		path.display()
	);
}
