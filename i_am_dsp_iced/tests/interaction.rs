//! Interaction tests: the controls are driven the way a person drives them.
//!
//! iced_test dispatches events straight into the widget tree, with no window and
//! no pointer involved, so a drag can be aimed at an exact pixel and its result
//! asserted exactly. The controls read the cursor the runtime hands them rather
//! than the position inside a mouse event, so every event here is preceded by a
//! point_at: that is also what a real run does, and a test that skipped it would
//! be testing something the widget never does.
//!
//! The tests are written to survive the process-global bits they do not own (the
//! UI scale is set by another test, the modifier keys are set inside these): every
//! control is given an explicit size, so where it lands does not depend on the
//! scale, and the modifier keys are set and cleared inside one test rather than
//! between two.

use i_am_dsp_iced::{
	iced::{
		self, Event, Length, Point,
		keyboard::Modifiers,
		mouse::{self, Button},
	},
	tools::{button::push_button, input, knob::knob, slider::slider},
};

#[derive(Debug, Clone, PartialEq)]
enum Message {
	Value(f32),
	Pressed,
}

/// Points the cursor somewhere, then sends one event from there.
macro_rules! mouse_at {
	($ui:expr, $point:expr, $event:expr) => {{
		$ui.point_at($point);
		$ui.simulate([Event::Mouse($event)]);
	}};
}

/// The value carried by the last message, which is what a control publishes once
/// the interaction is over.
fn last_value(ui: iced_test::Simulator<'_, Message>) -> f32 {
	match ui.into_messages().last() {
		Some(Message::Value(value)) => value,
		other => panic!("expected the interaction to publish a value, got {other:?}"),
	}
}

#[test]
fn a_knob_follows_the_cursor_fine_tunes_and_resets() {
	// An explicit size keeps the ring inside the box whatever scale another test
	// has left behind, so the pixel below is always on the ring.
	let view = || -> iced::Element<'static, Message> {
		knob(0.0..=100.0_f32, 50.0, Message::Value)
			.width(56.0)
			.height(56.0)
			.default_value(20.0)
			.into()
	};

	let center = Point::new(28.0, 28.0);
	let up = Point::new(28.0, 8.0);

	// Twenty pixels of drag is ten units: the range is a hundred wide and the
	// whole of it takes two hundred pixels.
	{
		let mut ui = iced_test::simulator(view());
		mouse_at!(ui, center, mouse::Event::ButtonPressed(Button::Left));
		mouse_at!(ui, up, mouse::Event::CursorMoved { position: up });
		mouse_at!(ui, up, mouse::Event::ButtonReleased(Button::Left));

		assert!((last_value(ui) - 60.0).abs() < 0.01, "a drag has to follow the cursor");
	}

	// Shift makes the same drag a tenth of the size.
	{
		let mut ui = iced_test::simulator(view());
		input::set_modifiers(Modifiers::SHIFT);

		mouse_at!(ui, center, mouse::Event::ButtonPressed(Button::Left));
		mouse_at!(ui, up, mouse::Event::CursorMoved { position: up });
		mouse_at!(ui, up, mouse::Event::ButtonReleased(Button::Left));

		input::set_modifiers(Modifiers::NONE);

		assert!((last_value(ui) - 51.0).abs() < 0.01, "shift has to fine-tune the drag");
	}

	// Two clicks in quick succession restore the declared default.
	{
		let mut ui = iced_test::simulator(view());
		mouse_at!(ui, center, mouse::Event::ButtonPressed(Button::Left));
		mouse_at!(ui, center, mouse::Event::ButtonReleased(Button::Left));
		mouse_at!(ui, center, mouse::Event::ButtonPressed(Button::Left));
		mouse_at!(ui, center, mouse::Event::ButtonReleased(Button::Left));

		assert!((last_value(ui) - 20.0).abs() < 0.01, "a double click has to restore the default");
	}
}

#[test]
fn a_slider_drags_across_its_track() {
	let view = || -> iced::Element<'static, Message> {
		slider(0.0..=1.0_f32, 0.5, Message::Value)
			.width(Length::Fixed(200.0))
			.into()
	};

	let mut ui = iced_test::simulator(view());
	let start = Point::new(40.0, 16.0);
	let end = Point::new(140.0, 16.0);

	mouse_at!(ui, start, mouse::Event::ButtonPressed(Button::Left));
	mouse_at!(ui, end, mouse::Event::CursorMoved { position: end });
	mouse_at!(ui, end, mouse::Event::ButtonReleased(Button::Left));

	// Half the track is half the range.
	assert!((last_value(ui) - 1.0).abs() < 0.01, "half a track of drag is half the range");
}

#[test]
fn a_button_fires_on_release_only() {
	let view = || -> iced::Element<'static, Message> {
		push_button("Test")
			.width(80.0)
			.on_press(Message::Pressed)
			.into()
	};

	let center = Point::new(40.0, 16.0);
	let away = Point::new(400.0, 300.0);

	// A press on its own is not a click.
	{
		let mut ui = iced_test::simulator(view());
		mouse_at!(ui, center, mouse::Event::ButtonPressed(Button::Left));

		assert_eq!(ui.into_messages().count(), 0, "a press alone must not fire");
	}

	// A press and a release inside fire exactly once.
	{
		let mut ui = iced_test::simulator(view());
		mouse_at!(ui, center, mouse::Event::ButtonPressed(Button::Left));
		mouse_at!(ui, center, mouse::Event::ButtonReleased(Button::Left));

		assert_eq!(ui.into_messages().count(), 1, "a click fires once");
	}

	// Releasing after leaving the button is not a click.
	{
		let mut ui = iced_test::simulator(view());
		mouse_at!(ui, center, mouse::Event::ButtonPressed(Button::Left));
		mouse_at!(ui, away, mouse::Event::CursorMoved { position: away });
		mouse_at!(ui, away, mouse::Event::ButtonReleased(Button::Left));

		assert_eq!(ui.into_messages().count(), 0, "a press that left the button is not a click");
	}
}

#[test]
fn a_button_that_is_born_selected_is_drawn_selected() {
	// The frame a widget draws the first time it exists. A fade that starts on the
	// first sight of a button makes an interface open with the wrong colours, and
	// that is invisible in any test that only looks at messages.
	fn frame(dir: &std::path::Path, element: iced::Element<'static, Message>) -> Vec<u8> {
		let _ = std::fs::remove_dir_all(dir);
		std::fs::create_dir_all(dir).expect("a scratch directory");

		let mut ui = iced_test::simulator(element);
		let snapshot = ui.snapshot(&i_am_dsp_iced::styles::theme()).expect("the button renders");

		// iced_test appends the name of the renderer to the file, so the frame is
		// found by looking for it rather than by guessing its name.
		snapshot
			.matches_image(dir.join("frame.png"))
			.expect("the frame is written");

		let frame = std::fs::read_dir(dir)
			.expect("the scratch directory")
			.filter_map(|entry| entry.ok())
			.map(|entry| entry.path())
			.find(|path| path.extension().is_some_and(|extension| extension == "png"))
			.expect("a frame was written");

		std::fs::read(frame).expect("the frame is readable")
	}

	let root = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
	let plain = frame(&root.join("button-unselected"), push_button("Test").width(80.0).into());
	let selected = frame(
		&root.join("button-selected"),
		push_button("Test").width(80.0).accent(true).into(),
	);

	assert_ne!(
		plain, selected,
		"a button born selected must be drawn in the accent colour on its first frame"
	);
}
