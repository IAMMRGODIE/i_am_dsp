//! Every control in the kit, at every size and every scale.
//!
//! Run it with:
//!
//! cargo run --release -p i_am_dsp_iced --example widget_kit
//!
//! Drag a knob, hold Shift while dragging it, spin the wheel over one, double
//! click one, and right click one: that is the interaction contract the whole
//! bundle is meant to share. The scale buttons at the top re-render everything at
//! another UI scale, which is how a faceplate is meant to handle DPI.

use i_am_dsp::prelude::Unit;
use i_am_dsp_iced::{
	iced::{
		self, Element, Length, Subscription, alignment,
		widget::{column, container, row, scrollable, text},
	},
	styles::{self, KnobSize},
	tools::{button::push_button, knob::knob, selector::selector, slider::slider},
};

/// The state of the gallery.
pub struct WidgetKit {
	cutoff: f32,
	gain: f32,
	mix: f32,
	drive: f32,
	sync: f32,
	modulated: f32,
	modulation: f32,
	slot: usize,
	pan: f32,
	scale: f32,
}

impl Default for WidgetKit {
	fn default() -> Self {
		Self {
			cutoff: 1200.0,
			gain: -3.0,
			mix: 0.65,
			drive: 0.4,
			sync: 0.5,
			modulated: 0.35,
			modulation: 0.72,
			slot: 0,
			pan: 0.0,
			scale: styles::ui_scale(),
		}
	}
}

#[derive(Debug, Clone)]
pub enum Message {
	Cutoff(f32),
	Gain(f32),
	Mix(f32),
	Drive(f32),
	Sync(f32),
	Modulated(f32),
	Slot(usize),
	Pan(f32),
	Scale(f32),
	Toggle,
	/// Keeps the frame redrawing, which is what makes the hover and click
	/// animations advance; without it a control animates only while the mouse
	/// is actually moving.
	Tick,
}

impl WidgetKit {
	fn update(&mut self, message: Message) {
		match message {
			Message::Cutoff(value) => self.cutoff = value,
			Message::Gain(value) => self.gain = value,
			Message::Mix(value) => self.mix = value,
			Message::Drive(value) => self.drive = value,
			Message::Sync(value) => self.sync = value,
			Message::Modulated(value) => self.modulated = value,
			Message::Slot(slot) => self.slot = slot,
			Message::Pan(value) => self.pan = value,
			Message::Scale(scale) => {
				styles::set_ui_scale(scale);
				self.scale = scale;
			},
			Message::Toggle | Message::Tick => {},
		}
	}

	/// Keeps the modifier state the controls read, and nothing else.
	///
	/// A plug-in gets this from [i_am_dsp_iced::modifiers_subscription]; the
	/// gallery has no processor, so it subscribes by hand.
	fn subscription(&self) -> Subscription<Message> {
		Subscription::batch([
			// A plug-in gets this from i_am_dsp_iced::timer; the gallery has no
			// processor, so it asks for the same 16 ms heartbeat directly.
			iced::time::every(std::time::Duration::from_millis(16)).map(|_| Message::Tick),
			iced::event::listen_with::<Message>(|event, _status, _window| {
				if let iced::Event::Keyboard(iced::keyboard::Event::ModifiersChanged(modifiers)) = event {
					i_am_dsp_iced::tools::input::set_modifiers(modifiers);
				}

				None
			}),
		])
	}

	/// The whole gallery.
	pub fn view(&self) -> Element<'_, Message> {
		let header = row![
			text("Widget kit").size(styles::px(styles::font::TITLE)),
			text("Shift: fine, double click or Ctrl+click: reset, wheel: step, right click: menu")
				.size(styles::px(styles::font::CAPTION))
				.color(styles::SECONDARY_TEXT_COLOR),
		]
		.spacing(styles::u(styles::space::L))
		.align_y(alignment::Vertical::Center);

		let scales = row![
			text("Scale").size(styles::px(styles::font::LABEL)),
			self.scale_button(0.75),
			self.scale_button(1.0),
			self.scale_button(1.25),
			self.scale_button(1.5),
			self.scale_button(2.0),
		]
		.spacing(styles::u(styles::space::S))
		.align_y(alignment::Vertical::Center);

		// The knobs themselves: the same widget, told apart only by the metadata
		// a faceplate would carry anyway.
		// One common height for the row, so the labels line up on a baseline even
		// though the rings are three different sizes: the label sticks to the
		// bottom of the box and the ring to the top.
		let cell = Length::Fixed(KnobSize::Large.labeled_height());
		let sizes = self.section(
			"Sizes",
			row![
				knob(0.0..=1.0, self.mix, Message::Mix)
					.size(KnobSize::Small)
					.height(cell)
					.label("Mix")
					.unit(Unit::Percent)
					.default_value(1.0),
				knob(0.0..=1.0, self.mix, Message::Mix)
					.size(KnobSize::Medium)
					.height(cell)
					.label("Mix")
					.unit(Unit::Percent)
					.default_value(1.0),
				knob(0.0..=1.0, self.mix, Message::Mix)
					.size(KnobSize::Large)
					.height(cell)
					.label("Mix")
					.unit(Unit::Percent)
					.default_value(1.0),
			]
			.spacing(styles::u(styles::space::L))
			.align_y(alignment::Vertical::Top),
		);

		let readouts = self.section(
			"Units",
			row![
				knob(20.0..=20000.0, self.cutoff, Message::Cutoff)
					.logarithmic(true)
					.label("Cutoff")
					.unit(Unit::Hertz)
					.default_value(1000.0),
				knob(-24.0..=24.0, self.gain, Message::Gain)
					.bipolar(true)
					.label("Gain")
					.unit(Unit::Decibels)
					.step(0.1)
					.default_value(0.0),
				knob(0.03125..=4.0, self.sync, Message::Sync)
					.label("Time")
					.unit(Unit::Beats)
					.default_value(0.5),
				knob(0.0..=1.0, self.drive, Message::Drive)
					.label("Drive")
					.value_formatter(|value| format!("{:.0} %", value * 100.0)),
			]
			.spacing(styles::u(styles::space::L)),
		);

		let modulation = self.section(
			"Modulation",
			row![
				knob(0.0..=1.0, self.modulated, Message::Modulated)
					.label("Modulated")
					.unit(Unit::Percent)
					.modulation(self.modulation),
				knob(0.0..=1.0, self.modulated, Message::Modulated)
					.label("Unmodulated")
					.unit(Unit::Percent),
				knob(-24.0..=24.0, self.gain, Message::Gain)
					.bipolar(true)
					.label("Modulated gain")
					.unit(Unit::Decibels)
					.modulation(9.0)
					.default_value(0.0),
			]
			.spacing(styles::u(styles::space::L)),
		);

		let width = Length::Fixed(styles::u(72.0));
		let rows = self.section(
			"Row controls",
			column![
				selector(vec!["Low", "Band", "High"], self.slot)
					.on_change(|_slots, slot| Message::Slot(slot))
					.width(width),
				slider(0.0..=1.0_f32, self.mix, Message::Mix)
					.text("Mix")
					.formatter(|value| format!("{:.0} %", value * 100.0))
					.default_value(1.0)
					.width(width),
				slider(-1.0..=1.0_f32, self.pan, Message::Pan)
					.text("Pan")
					.bipolar(true)
					.default_value(0.0)
					.modulation(self.pan + 0.45)
					.width(width),
				push_button("A button").on_press(Message::Toggle),
			]
			.spacing(styles::u(styles::space::S)),
		);

		container(
			scrollable(
				column![header, scales, sizes, rows, readouts, modulation]
					.spacing(styles::u(styles::space::XL))
					.padding(styles::u(styles::space::XL)),
			)
			.width(Length::Fill),
		)
		.width(Length::Fill)
		.height(Length::Fill)
		.style(|_| container::Style {
			background: Some(styles::BACKGROUND_COLOR.into()),
			..Default::default()
		})
		.into()
	}

	fn scale_button(&self, scale: f32) -> Element<'_, Message> {
		let label = format!("{:.0} %", scale * 100.0);
		let active = (self.scale - scale).abs() < f32::EPSILON;
		push_button(label).accent(active).on_press(Message::Scale(scale)).into()
	}

	/// One titled card of controls.
	fn section<'a>(&'a self, title: &'a str, content: impl Into<Element<'a, Message>>) -> Element<'a, Message> {
		let content: Element<'a, Message> = content.into();
		container(
			column![
				text(title).size(styles::px(styles::font::SECTION)).color(styles::SECONDARY_TEXT_COLOR),
				content,
			]
			.spacing(styles::u(styles::space::M)),
		)
		.padding(styles::u(styles::space::L))
		.width(Length::Fill)
		.style(|_| container::Style {
			background: Some(styles::CARD_COLOR.into()),
			border: iced::Border {
				color: styles::CARD_BORDER_COLOR,
				width: 1.0,
				radius: styles::u(styles::size::RADIUS).into(),
			},
			..Default::default()
		})
		.into()
	}
}

#[cfg(feature = "standalone")]
fn main() -> iced::Result {
	// A plug-in reads its scale from the host; the gallery takes it from the
	// environment, so the same build can be compared at different scales.
	if let Ok(scale) = std::env::var("IAM_UI_SCALE")
		&& let Ok(scale) = scale.parse::<f32>()
	{
		styles::set_ui_scale(scale);
	}

	iced::application(WidgetKit::default, WidgetKit::update, WidgetKit::view)
		.subscription(WidgetKit::subscription)
		.theme(styles::theme())
		.window_size((940.0, 900.0))
		.run()
}

#[cfg(not(feature = "standalone"))]
fn main() {
	println!("the standalone feature is not enabled, nothing to run");
}
