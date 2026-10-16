#![cfg(feature = "faceplate")]
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

/// The same faceplate, four real effects, and not one line of interface code
/// between them: each effect declares where its controls belong and how they want
/// to be drawn, and that is the whole of what it knows about being on screen.
#[test]
fn four_effects_share_one_faceplate() {
	use i_am_dsp::prelude::{Crusher, HardClipper, Lowpass, Phaser, SineWave};
	use i_am_dsp_iced::iced::{
		Element,
		widget::{column, row, text},
	};

	let effects = [
		("I Am Crusher", Crusher::<2>::new(44_100).get_parameters()),
		("I Am Drive", HardClipper::new(0.8, 1.0).get_parameters()),
		("I Am Filter", Lowpass::<2>::new(44_100, 1_000.0, 0.7).get_parameters()),
		("I Am Phaser", Phaser::<SineWave, 2>::new(SineWave, 4, 44_100).get_parameters()),
	];

	// A function rather than a closure: a closure cannot say that what it returns
	// borrows from what it was handed.
	fn card<'a>(effect: &'a (&'static str, Vec<i_am_dsp::prelude::Parameter>)) -> Element<'a, Change> {
		column![
			text(effect.0).size(styles::px(styles::font::TITLE)),
			faceplate::faceplate(&effect.1, |_, value| Change(value)),
		]
		.spacing(styles::u(styles::space::S))
		.into()
	}

	let sheet: Element<'_, Change> = row![
		column![card(&effects[0]), card(&effects[1])].spacing(styles::u(styles::space::XL)),
		column![card(&effects[2]), card(&effects[3])].spacing(styles::u(styles::space::XL)),
	]
	.spacing(styles::u(styles::space::XL))
	.padding(styles::u(styles::space::L))
	.into();

	let mut ui = iced_test::simulator(sheet);
	let snapshot = match ui.snapshot(&styles::theme()) {
		Ok(snapshot) => snapshot,
		Err(error) => panic!("the family sheet renders: {error:?}"),
	};

	let path = Path::new(env!("CARGO_MANIFEST_DIR"))
		.join("tests")
		.join("snapshots")
		.join("family.png");

	assert!(
		snapshot.matches_image(&path).expect("the frame is written"),
		"the family sheet changed; delete {} and run again to accept it",
		path.display()
	);
}
