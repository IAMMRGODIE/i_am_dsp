//! A knob, and the interaction contract every control in the family shares.
//!
//! # Interaction
//!
//! | Gesture | Effect |
//! | --- | --- |
//! | drag up/down (or left/right) | change the value |
//! | Shift + drag, Shift + wheel | fine adjust, a tenth of the speed |
//! | mouse wheel | step by one notch |
//! | double click, or Ctrl/Cmd + click | reset to the declared default |
//! | right click | [Knob::on_right_click], for a context menu |
//!
//! # What it draws
//!
//! A knob is a ring, not a picture of a knob: the value is an arc, the body is
//! the hole in the middle, the label and the readout sit underneath, and a
//! modulator gets a ring of its own *outside* the value ring. The band the
//! modulation ring occupies is reserved even when nothing is modulating the
//! parameter, because adding a modulator later must not move a faceplate.
//!
//! The readout is drawn from [Unit], so a synced time reads 1/8 and a gain reads
//! -6.00 dB without the faceplate formatting anything itself.
//!
//! # Coordinates
//!
//! Everything is measured with the [mouse::Cursor] the widget is handed, never
//! with the raw position in [iced::Event::Mouse]. A scrollable translates the
//! cursor into the coordinate space of its content before passing it down, and a
//! widget that reads the raw position instead is off by exactly the scroll
//! offset: clicks land on the wrong control as soon as anything scrolls.

use std::{ops::RangeInclusive, time::{Duration, Instant}};

use iced::{
	Color, Element, Font, Length, Pixels, Point, Radians, Rectangle, Renderer, Size, Theme,
	advanced::{
		Widget,
		graphics::geometry::{Renderer as _, frame::Backend},
		layout, renderer,
	},
	mouse::{self, Event, ScrollDelta},
	theme::palette::lighten,
	widget::canvas::{self, Path, Stroke, Text},
};

use i_am_dsp::prelude::Unit;

use crate::{
	styles::{self, BORDER_WIDTH, BRIGHT_FACTOR, KnobSize},
	tools::{Number, input, utils::Animator},
};

/// How long apart two clicks may be and still count as a double click.
const DOUBLE_CLICK: Duration = Duration::from_millis(300);
/// Where the ring starts: the lower left, 135 degrees clockwise from east.
const START_ANGLE: f32 = 0.75 * std::f32::consts::PI;
/// How far the ring sweeps: 270 degrees, leaving the usual gap at the bottom.
const SWEEP: f32 = 1.5 * std::f32::consts::PI;
/// How wide a character is assumed to be, when guessing how much room a label
/// needs. Only used to pick a sensible default width; the caller can override.
const CHARACTER_WIDTH: f32 = 0.62;

/// A knob controller.
pub struct Knob<'a, Message> {
	value: f32,
	step: f32,
	speed: f32,
	speed_set: bool,
	from: f32,
	to: f32,
	is_logarithmic: bool,
	bipolar: bool,
	default_value: Option<f32>,
	label: Option<String>,
	unit: Unit,
	value_formatter: Option<Box<dyn Fn(f32) -> String + 'a>>,
	modulation: Option<f32>,
	diameter: f32,
	on_change: Box<dyn Fn(f32) -> Message + 'a>,
	on_release: Option<Box<dyn Fn(f32) -> Message + 'a>>,
	on_right_click: Option<Box<dyn Fn(f32) -> Message + 'a>>,
	width: Option<Length>,
	height: Option<Length>,
}

#[derive(Default)]
struct KnobState {
	current_mouse_pos: Point,
	drag_origin: Option<Point>,
	last_press: Option<Instant>,
	hover_animator: Animator,
	click_animator: Animator,
}

/// Create a new knob with the given range and initial value.
pub fn knob<'a, Message, T: Number>(range: RangeInclusive<T>, value: T, on_change: impl Fn(T) -> Message + 'a) -> Knob<'a, Message> {
	Knob::<Message>::new(range, value, on_change)
}

impl<'a, Message> Knob<'a, Message> {
	/// Create a new knob with the given range and initial value.
	pub fn new<T: Number>(range: RangeInclusive<T>, value: T, on_change: impl Fn(T) -> Message + 'a) -> Self {
		let (from, to) = range.into_inner();
		let from = from.into_f32();
		let to = to.into_f32();
		let speed = (to - from) / 200.0;
		Knob {
			value: value.into_f32(),
			step: 0.0,
			speed,
			speed_set: false,
			from,
			to,
			is_logarithmic: false,
			bipolar: false,
			default_value: None,
			label: None,
			unit: Unit::None,
			value_formatter: None,
			modulation: None,
			diameter: KnobSize::default().diameter(),
			on_change: Box::new(move |value| on_change(T::from_f32(value))),
			on_release: None,
			on_right_click: None,
			width: None,
			height: None,
		}
	}

	/// Set the on_release callback of the knob.
	pub fn on_release(self, on_release: impl Fn(f32) -> Message + 'a) -> Self {
		Self { on_release: Some(Box::new(on_release)), ..self }
	}

	/// Set the callback a right click publishes, for a context menu.
	pub fn on_right_click(self, on_right_click: impl Fn(f32) -> Message + 'a) -> Self {
		Self { on_right_click: Some(Box::new(on_right_click)), ..self }
	}

	/// Set the width of the knob.
	pub fn width(self, width: impl Into<Length>) -> Self {
		Self { width: Some(width.into()), ..self }
	}

	/// Set the height of the knob.
	pub fn height(self, height: impl Into<Length>) -> Self {
		Self { height: Some(height.into()), ..self }
	}

	/// Set the speed of the knob, overriding the sensible default.
	pub fn speed(self, speed: f32) -> Self {
		Self { speed, speed_set: true, ..self }
	}

	/// Set the value of the knob.
	pub fn value(self, value: f32) -> Self {
		Self { value, ..self }
	}

	/// Set the step of the knob. A step of zero, the default, does not snap.
	pub fn step(self, step: f32) -> Self {
		Self { step, ..self }
	}

	/// Set the logarithmic mode of the knob.
	///
	/// A logarithmic knob that is not given an explicit speed spreads its whole
	/// range over the same drag distance a linear one does, which a linear
	/// pixels-to-value factor cannot do: the range spans decades.
	pub fn logarithmic(self, log: bool) -> Self {
		Self { is_logarithmic: log, ..self }
	}

	/// Draw the value outward from the middle of the range instead of from the
	/// bottom. For a pan, a gain in dB, a pitch offset.
	pub fn bipolar(self, bipolar: bool) -> Self {
		Self { bipolar, ..self }
	}

	/// The size of the ring. Defaults to [KnobSize::Medium].
	pub fn size(self, size: KnobSize) -> Self {
		Self { diameter: size.diameter(), ..self }
	}

	/// The name drawn under the knob.
	pub fn label(self, label: impl Into<String>) -> Self {
		Self { label: Some(label.into()), ..self }
	}

	/// How the value is rendered. Setting a unit other than [Unit::None] also
	/// turns the readout on.
	pub fn unit(self, unit: Unit) -> Self {
		Self { unit, ..self }
	}

	/// Renders the value as text, when a [Unit] cannot say it.
	pub fn value_formatter(self, formatter: impl Fn(f32) -> String + 'a) -> Self {
		Self { value_formatter: Some(Box::new(formatter)), ..self }
	}

	/// The value a double click, or Ctrl/Cmd + click, restores.
	pub fn default_value(self, default: f32) -> Self {
		Self { default_value: Some(default), ..self }
	}

	/// Where a modulator is currently holding the parameter.
	///
	/// The knob draws a ring from the value to this, and a marker at this end.
	/// The caller does the modulation; the knob only shows it.
	pub fn modulation(self, modulated_value: f32) -> Self {
		Self { modulation: Some(modulated_value), ..self }
	}

	/// Stop drawing a modulation ring.
	pub fn no_modulation(self) -> Self {
		Self { modulation: None, ..self }
	}

	/// How far one pixel of drag moves the value.
	fn drag_speed(&self) -> f32 {
		if self.speed_set {
			return self.speed;
		}

		let span = if self.is_logarithmic {
			let from = self.from.max(f32::MIN_POSITIVE);
			let to = self.to.max(f32::MIN_POSITIVE);
			(to.log10() - from.log10()).abs()
		} else {
			(self.to - self.from).abs()
		};

		span / 200.0
	}

	/// Whether a readout is drawn under the knob.
	fn shows_value(&self) -> bool {
		self.label.is_some() || self.value_formatter.is_some() || self.unit != Unit::None
	}

	/// The text drawn under the knob.
	fn value_text(&self) -> String {
		if let Some(formatter) = &self.value_formatter {
			formatter(self.value)
		} else {
			self.unit.format(self.value as f64)
		}
	}

	/// The height the label and the readout take, gap included.
	fn text_height(&self) -> f32 {
		let mut height = 0.0;

		if self.label.is_some() {
			height += styles::px(styles::font::LABEL) * 1.3;
		}
		if self.shows_value() {
			height += styles::px(styles::font::VALUE) * 1.3;
		}

		if height > 0.0 {
			height + styles::u(styles::space::XS)
		} else {
			0.0
		}
	}

	/// The width the knob asks for when the caller does not set one.
	///
	/// A label is usually wider than the ring, and a widget's own bounds clip it,
	/// so the label gets to widen the widget instead of being cut in half.
	fn intrinsic_width(&self) -> f32 {
		let text_width = |text: &str, size: f32| text.chars().count() as f32 * size * CHARACTER_WIDTH;
		let mut width = self.diameter;

		if let Some(label) = &self.label {
			width = width.max(text_width(label, styles::px(styles::font::LABEL)));
		}
		if self.shows_value() {
			width = width.max(text_width(&self.value_text(), styles::px(styles::font::VALUE)));
		}

		width + styles::u(styles::space::S)
	}

	/// The height the knob asks for when the caller does not set one.
	fn intrinsic_height(&self) -> f32 {
		self.diameter + self.text_height()
	}

	/// The square the ring is drawn in, and the only part that reacts to the
	/// mouse. The label and the readout underneath are not a handle: a drag has
	/// to start on the ring itself.
	fn ring_rect(&self, bounds: Rectangle) -> Rectangle {
		let diameter = self.diameter.min(bounds.width).min(bounds.height).max(1.0);

		Rectangle {
			x: bounds.x + (bounds.width - diameter) / 2.0,
			y: bounds.y,
			width: diameter,
			height: diameter,
		}
	}

	/// Where a value sits on the ring, from 0 at the bottom left to 1 at the
	/// bottom right.
	fn percent_of(&self, value: f32) -> f32 {
		let percent = if self.is_logarithmic {
			let from = self.from.max(f32::MIN_POSITIVE);
			let to = self.to.max(f32::MIN_POSITIVE);
			(value.max(f32::MIN_POSITIVE).log10() - from.log10()) / (to.log10() - from.log10())
		} else {
			(value - self.from) / (self.to - self.from)
		};

		if percent.is_finite() { percent.clamp(0.0, 1.0) } else { 0.0 }
	}

	/// The value one drag of some pixels produces, which is where the
	/// exponential taper of a logarithmic knob lives.
	fn dragged(&self, pixels: f32, fine: f32) -> f32 {
		let change = pixels * fine * self.drag_speed();

		if self.is_logarithmic {
			self.value.max(f32::MIN_POSITIVE) * 10.0_f32.powf(change)
		} else {
			self.value + change
		}
	}

	/// Restores the declared default, if there is one.
	fn reset(&mut self, shell: &mut iced::advanced::Shell<'_, Message>) {
		let Some(default) = self.default_value else {
			return;
		};

		let default = default.clamp(self.from.min(self.to), self.from.max(self.to));

		if default != self.value {
			self.value = default;
			shell.publish((self.on_change)(default));
		}
	}

	/// Publishes a new value, if it is actually a new value.
	fn set(&mut self, value: f32, shell: &mut iced::advanced::Shell<'_, Message>) {
		let value = value.clamp(self.from.min(self.to), self.from.max(self.to));

		if value != self.value {
			self.value = value;
			shell.publish((self.on_change)(value));
		}
	}
}

impl<'a, Message> Widget<Message, Theme, Renderer> for Knob<'a, Message> {
	fn size(&self) -> iced::Size<Length> {
		Size::new(
			self.width.unwrap_or(Length::Fixed(self.intrinsic_width())),
			self.height.unwrap_or(Length::Fixed(self.intrinsic_height())),
		)
	}

	fn layout(
		&mut self,
		_: &mut iced::advanced::widget::Tree,
		_: &Renderer,
		limits: &iced::advanced::layout::Limits,
	) -> iced::advanced::layout::Node {
		let width = self.width.unwrap_or(Length::Fixed(self.intrinsic_width()));
		let height = self.height.unwrap_or(Length::Fixed(self.intrinsic_height()));
		let size = limits.resolve(width, height, Size::new(self.intrinsic_width(), self.intrinsic_height()));
		layout::Node::new(size)
	}

	fn draw(
		&self,
		tree: &iced::advanced::widget::Tree,
		renderer: &mut Renderer,
		theme: &Theme,
		_: &renderer::Style,
		layout: layout::Layout<'_>,
		_: iced::advanced::mouse::Cursor,
		_: &iced::Rectangle,
	) {
		let state = tree.state.downcast_ref::<KnobState>();
		let bounds = layout.bounds();
		let palette = theme.extended_palette();
		let brighten = state.hover_animator.calc() * BRIGHT_FACTOR;
		let click = state.click_animator.calc();

		// The knob is a square at the top of the widget; anything under it is the
		// label and the readout.
		let knob = self.ring_rect(bounds);
		let diameter = knob.width;
		let center = knob.center();
		let radius = diameter / 2.0;
		let ring_width = (radius / 64.0 * 10.0).max(styles::u(0.75));

		// The modulation ring lives outside the value ring, and its band is
		// reserved whether or not there is a modulation to draw.
		let mod_width = styles::u(styles::size::MOD_RING);
		let reserved = mod_width + styles::u(styles::size::MOD_GAP);
		let arc_radius = (radius - reserved - ring_width / 2.0).max(1.0);
		let body_radius = (arc_radius - ring_width / 2.0).max(0.5);

		let value_color = lighten(theme.palette().primary, brighten);
		let text_color = lighten(theme.palette().text, brighten);
		let track_color = palette.background.weakest.color;
		let body_color = lighten(palette.background.strongest.color, brighten);

		let percent = self.percent_of(self.value);
		let zero = if self.bipolar { 0.5 } else { 0.0 };

		let mut frame = renderer.new_frame(bounds);

		frame.stroke(
			&arc_path(center, arc_radius, 0.0, 1.0),
			Stroke::default().with_color(track_color).with_width(ring_width),
		);

		if (percent - zero).abs() > f32::EPSILON {
			let color = if percent < zero { styles::NEGATIVE_COLOR } else { value_color };
			frame.stroke(
				&arc_path(center, arc_radius, zero.min(percent), zero.max(percent)),
				Stroke::default().with_color(color).with_width(ring_width),
			);
		}

		if let Some(modulated) = self.modulation {
			let modulated_percent = self.percent_of(modulated);
			let ring_radius = radius - mod_width / 2.0;

			frame.stroke(
				&arc_path(center, ring_radius, percent.min(modulated_percent), percent.max(modulated_percent)),
				Stroke::default().with_color(styles::MOD_COLOR).with_width(mod_width),
			);

			let angle = angle_of(modulated_percent);
			let marker = Point::new(center.x + angle.cos() * ring_radius, center.y + angle.sin() * ring_radius);
			// The marker sits on the ring, so half the ring width is all the room
			// there is before it pokes past the edge of the widget.
			frame.fill(&Path::circle(marker, mod_width / 2.0), styles::MOD_COLOR);
		}

		frame.stroke(
			&Path::new(|builder| {
				let angle = angle_of(percent);
				let start = body_radius * 0.35;
				let end = arc_radius + ring_width * 0.5;

				builder.move_to(Point::new(center.x + angle.cos() * start, center.y + angle.sin() * start));
				builder.line_to(Point::new(center.x + angle.cos() * end, center.y + angle.sin() * end));
			}),
			Stroke::default().with_color(text_color).with_width(BORDER_WIDTH),
		);

		frame.fill(&Path::circle(center, body_radius * (1.0 - 0.1 * click)), body_color);

		// The label, then the readout: anchored on the middle of the ring, and on
		// the bottom of the widget, so knobs of different sizes in one row still
		// line their labels up as long as the caller gives them a common height.
		let mut y = bounds.y + bounds.height - self.text_height() + styles::u(styles::space::XS);

		if let Some(label) = &self.label {
			let height = styles::px(styles::font::LABEL) * 1.3;
			draw_text(
				&mut frame,
				label,
				Point::new(center.x, y),
				bounds.width,
				styles::px(styles::font::LABEL),
				styles::font::LABEL_FONT,
				palette.background.weakest.text,
			);
			y += height;
		}

		if self.shows_value() {
			draw_text(
				&mut frame,
				&self.value_text(),
				Point::new(center.x, y),
				bounds.width,
				styles::px(styles::font::VALUE),
				styles::font::VALUE_FONT,
				text_color,
			);
		}

		renderer.draw_geometry(frame.into_geometry());
	}

	fn tag(&self) -> iced::advanced::widget::tree::Tag {
		// Without a tag of its own this widget inherits the stateless one, which
		// is the tag of the empty tree a root element is diffed against: the two
		// match, the state is never built, and the first event panics on the
		// downcast. A widget with state has to say so.
		iced::advanced::widget::tree::Tag::of::<KnobState>()
	}

	fn state(&self) -> iced::advanced::widget::tree::State {
		iced::advanced::widget::tree::State::new(KnobState::default())
	}

	fn update(
		&mut self,
		tree: &mut iced::advanced::widget::Tree,
		event: &iced::Event,
		layout: layout::Layout<'_>,
		cursor: iced::advanced::mouse::Cursor,
		_: &Renderer,
		_: &mut dyn iced::advanced::Clipboard,
		shell: &mut iced::advanced::Shell<'_, Message>,
		_: &iced::Rectangle,
	) {
		let state = tree.state.downcast_mut::<KnobState>();
		let bounds = layout.bounds();

		// The cursor is already in this widget's own coordinate space.
		if let Some(position) = cursor.position() {
			state.current_mouse_pos = position;
		}

		let over = cursor.is_over(self.ring_rect(bounds));
		let modifiers = input::modifiers();
		let fine = if modifiers.shift() { 0.1 } else { 1.0 };

		// While dragging, the value follows the cursor, using the cursor's own
		// translation so a drag does not jump when the view scrolls under it.
		if let Some(origin) = state.drag_origin {
			let position = cursor.position().unwrap_or(origin);
			let drag = (position.x - origin.x) - (position.y - origin.y);

			state.drag_origin = Some(position);

			if drag != 0.0 {
				let value = self.dragged(drag, fine);
				self.set(value, shell);
			}
		}

		self.set(self.value, shell);

		if over {
			state.hover_animator.in_if_out();
		} else {
			state.hover_animator.out_if_in();
		}

		let iced::Event::Mouse(mouse_event) = event else {
			return;
		};

		match mouse_event {
			Event::ButtonPressed(mouse::Button::Left) if over => {
				let now = Instant::now();
				let double = state.last_press.is_some_and(|last| now.duration_since(last) < DOUBLE_CLICK);
				state.last_press = if double { None } else { Some(now) };

				if double || modifiers.command() {
					self.reset(shell);
				} else {
					state.drag_origin = cursor.position();
					state.click_animator.in_if_out();
				}
			},
			Event::ButtonPressed(mouse::Button::Right) if over => {
				if let Some(on_right_click) = &self.on_right_click {
					shell.publish(on_right_click(self.value));
				}
			},
			Event::ButtonReleased(mouse::Button::Left) => {
				let was_dragging = state.drag_origin.take().is_some();
				state.click_animator.out_if_in();

				if !was_dragging {
					return;
				}

				if self.step > 0.0 {
					let snapped = (self.value / self.step).round() * self.step;
					self.set(snapped, shell);
				}

				if let Some(on_release) = &self.on_release {
					shell.publish(on_release(self.value));
				}
			},
			Event::WheelScrolled { delta } if over => {
				// A wheel notch is a step when the parameter has one, and one
				// percent of the range when it does not.
				let notches = match delta {
					ScrollDelta::Lines { y, .. } => *y,
					ScrollDelta::Pixels { y, .. } => *y / 50.0,
				};

				if notches == 0.0 {
					return;
				}

				let step = if self.step > 0.0 {
					self.step
				} else if self.is_logarithmic {
					// About a decade for every ten notches.
					self.drag_speed() * 2.0
				} else {
					(self.from - self.to).abs() * 0.01
				};

				let value = if self.is_logarithmic {
					self.value.max(f32::MIN_POSITIVE) * 10.0_f32.powf(notches * step * 20.0 * fine)
				} else {
					self.value + notches * step * fine
				};

				self.set(value, shell);
			},
			_ => {},
		}
	}
}

impl<'a, Messsage: 'a> From<Knob<'a, Messsage>> for Element<'a, Messsage> {
	fn from(value: Knob<'a, Messsage>) -> Self {
		Element::new(value)
	}
}

/// Where a position on the ring is, in radians.
fn angle_of(percent: f32) -> f32 {
	START_ANGLE + percent * SWEEP
}

/// The arc between two positions on the ring.
fn arc_path(center: Point, radius: f32, from: f32, to: f32) -> Path {
	Path::new(|builder| {
		if (to - from).abs() <= f32::EPSILON {
			return;
		}

		builder.arc(canvas::path::Arc {
			radius,
			center,
			start_angle: Radians(angle_of(from)),
			end_angle: Radians(angle_of(to)),
		});
	})
}

/// Draws one line of text, centred on the anchor.
///
/// Text goes through the canvas, which is the only text path this crate has ever
/// rendered correctly; the low-level text call on the renderer lays a paragraph
/// out by its own rules and clips it to the widget, which is how labels end up
/// half drawn.
#[allow(clippy::too_many_arguments)]
fn draw_text<F: Backend>(
	frame: &mut F,
	content: &str,
	anchor: Point,
	max_width: f32,
	size: f32,
	font: Font,
	color: Color,
) {
	Text {
		size: Pixels(size),
		font,
		color,
		position: anchor,
		align_x: iced::widget::text::Alignment::Center,
		align_y: iced::alignment::Vertical::Top,
		max_width,
		..Text::from(content)
	}
	.draw_with(|path, color| frame.fill(&path, color));
}
