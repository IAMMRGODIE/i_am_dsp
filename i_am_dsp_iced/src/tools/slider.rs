//! A horizontal slider: a track, a value, and the same contract as the knob.
//!
//! The value is drawn as a filled track rather than a moving block, the label
//! and the readout sit on a line above it, and a modulator gets the same
//! treatment a knob gives it — a marker above the track, drawn with the reserved
//! modulation colour, so the value underneath stays readable.
//!
//! A bipolar slider fills outward from the middle, which is what a pan or a
//! gain in dB wants; a logarithmic one spreads its range over decades rather
//! than raw units, or dragging it would cross the whole range in a few pixels.

use std::ops::RangeInclusive;

use iced::{
	Element, Length, Point, Rectangle, Renderer, Size, Theme, alignment,
	advanced::{Widget, graphics::geometry::{Renderer as _, frame::Backend}, layout, renderer},
	mouse::Event,
	theme::palette::lighten,
	widget::canvas::Path,
};

use crate::{
	styles::{self, BRIGHT_FACTOR},
	tools::{Number, input, utils, utils::Animator},
};

/// A horizontal slider.
pub struct Slider<'a, Message> {
	value: f32,
	step: f32,
	speed: f32,
	from: f32,
	to: f32,
	is_logarithmic: bool,
	bipolar: bool,
	default_value: Option<f32>,
	modulation: Option<f32>,
	on_change: Box<dyn Fn(f32) -> Message + 'a>,
	on_release: Option<Box<dyn Fn(f32) -> Message + 'a>>,
	width: Length,
	text: String,
	value_formatter: Option<Box<dyn Fn(f32) -> String + 'a>>,
}

#[derive(Default)]
struct SliderState {
	current_mouse_pos: Point,
	last_mouse_pos: Option<Point>,
	last_press: Option<std::time::Instant>,
	hover_animator: Animator,
	click_animator: Animator,
}

/// Create a new slider with the given range and initial value.
pub fn slider<'a, Message, T: Number>(range: RangeInclusive<T>, value: T, on_change: impl Fn(T) -> Message + 'a) -> Slider<'a, Message> {
	Slider::new(range, value, on_change)
}

impl<'a, Message> Slider<'a, Message> {
	/// Create a new slider with the given range and initial value.
	pub fn new<T: Number>(range: RangeInclusive<T>, value: T, on_change: impl Fn(T) -> Message + 'a) -> Self {
		let (from, to) = range.into_inner();
		let from = from.into_f32();
		let to = to.into_f32();
		Self {
			value: value.into_f32(),
			step: 0.0,
			speed: 1.0,
			from,
			to,
			is_logarithmic: false,
			bipolar: false,
			default_value: None,
			modulation: None,
			on_change: Box::new(move |value| on_change(T::from_f32(value))),
			on_release: None,
			width: Length::Fill,
			text: String::new(),
			value_formatter: None,
		}
	}

	/// Set the on_release callback.
	pub fn on_release(self, on_release: impl Fn(f32) -> Message + 'a) -> Self {
		Self { on_release: Some(Box::new(on_release)), ..self }
	}

	/// Set the width of the slider.
	pub fn width(self, width: impl Into<Length>) -> Self {
		Self { width: width.into(), ..self }
	}

	/// Set the text of the slider.
	pub fn text(self, text: impl Into<String>) -> Self {
		Self { text: text.into(), ..self }
	}

	/// Set the slider to be logarithmic.
	pub fn logarithmic(self) -> Self {
		Self { is_logarithmic: true, ..self }
	}

	/// Fill outward from the middle of the range instead of from the left.
	pub fn bipolar(self, bipolar: bool) -> Self {
		Self { bipolar, ..self }
	}

	/// Set the step of the slider.
	pub fn step(self, step: f32) -> Self {
		Self { step, ..self }
	}

	/// Set the speed of the slider.
	pub fn speed(self, speed: f32) -> Self {
		Self { speed, ..self }
	}

	/// The value a double click restores.
	pub fn default_value(self, default: f32) -> Self {
		Self { default_value: Some(default), ..self }
	}

	/// Where a modulator is currently holding the parameter.
	pub fn modulation(self, modulated_value: f32) -> Self {
		Self { modulation: Some(modulated_value), ..self }
	}

	/// Stop drawing a modulation marker.
	pub fn no_modulation(self) -> Self {
		Self { modulation: None, ..self }
	}

	/// Set the value formatter.
	pub fn formatter(self, formatter: impl Fn(f32) -> String + 'a) -> Self {
		Self { value_formatter: Some(Box::new(formatter)), ..self }
	}

	/// Whether a label row is drawn above the track.
	fn has_label(&self) -> bool {
		!self.text.is_empty() || self.value_formatter.is_some()
	}

	/// The text drawn on the label row.
	fn value_text(&self) -> String {
		if let Some(formatter) = &self.value_formatter {
			formatter(self.value)
		} else {
			format!("{:.2}", self.value)
		}
	}

	/// Where a value sits on the track, from 0 at the left to 1 at the right.
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

	/// The value a drag of so many pixels across a track of this width produces.
	fn dragged(&self, delta: f32, width: f32, fine: f32) -> f32 {
		if width <= 0.0 {
			return self.value;
		}

		// A logarithmic slider counts decades, not units: the range is the
		// exponent, and dragging the whole width must cover all of it.
		let span = if self.is_logarithmic {
			let from = self.from.max(f32::MIN_POSITIVE);
			let to = self.to.max(f32::MIN_POSITIVE);
			(to.log10() - from.log10()).abs()
		} else {
			(self.to - self.from).abs()
		};

		let change = delta / width * span * self.speed * fine;

		if self.is_logarithmic {
			self.value.max(f32::MIN_POSITIVE) * 10.0_f32.powf(change)
		} else {
			self.value + change
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

impl<'a, Message> Widget<Message, Theme, Renderer> for Slider<'a, Message> {
	fn size(&self) -> iced::Size<Length> {
		Size::new(self.width, Length::Fixed(styles::u(styles::size::ROW)))
	}

	fn layout(
		&mut self,
		_: &mut iced::advanced::widget::Tree,
		_: &Renderer,
		limits: &iced::advanced::layout::Limits,
	) -> iced::advanced::layout::Node {
		let height = Length::Fixed(styles::u(styles::size::ROW));
		let size = limits.resolve(self.width, height, Size::new(styles::u(16.0), styles::u(styles::size::ROW)));
		layout::Node::new(size)
	}

	fn state(&self) -> iced::advanced::widget::tree::State {
		iced::advanced::widget::tree::State::new(SliderState::default())
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
		let state = tree.state.downcast_ref::<SliderState>();
		let bounds = layout.bounds();
		let palette = theme.extended_palette();
		let brighten = state.hover_animator.calc() * BRIGHT_FACTOR;

		let accent = lighten(theme.palette().primary, brighten);
		let text_color = lighten(theme.palette().text, brighten);
		let muted = palette.background.weakest.text;
		let track_color = palette.background.weakest.color;

		let mut frame = renderer.new_frame(bounds);

		let label_size = styles::px(styles::font::LABEL);
		let value_size = styles::px(styles::font::VALUE);
		let track_height = styles::u(styles::size::TRACK);
		let gap = styles::u(styles::space::XS);
		let label_height = if self.has_label() { label_size * 1.3 } else { 0.0 };
		let radius = track_height / 2.0;

		if self.has_label() {
			if !self.text.is_empty() {
				utils::draw_text(
					&mut frame,
					&self.text,
					Point::new(bounds.x, bounds.y),
					iced::widget::text::Alignment::Left,
					alignment::Vertical::Top,
					label_size,
					styles::font::LABEL_FONT,
					muted,
					bounds.width,
				);
			}

			utils::draw_text(
				&mut frame,
				&self.value_text(),
				Point::new(bounds.x + bounds.width, bounds.y),
				iced::widget::text::Alignment::Right,
				alignment::Vertical::Top,
				value_size,
				styles::font::VALUE_FONT,
				accent,
				bounds.width,
			);
		}

		// The track sits under the label row, or in the middle when there is none.
		let track_y = if self.has_label() {
			bounds.y + label_height + gap
		} else {
			bounds.y + (bounds.height - track_height) / 2.0
		};
		let track = Rectangle {
			x: bounds.x,
			y: track_y,
			width: bounds.width,
			height: track_height,
		};

		frame.fill(&rounded(track, radius), track_color);

		let percent = self.percent_of(self.value);
		let zero = if self.bipolar { 0.5 } else { 0.0 };

		if (percent - zero).abs() > f32::EPSILON {
			let from = zero.min(percent);
			let to = zero.max(percent);
			let fill = Rectangle {
				x: track.x + track.width * from,
				y: track.y,
				width: track.width * (to - from),
				height: track.height,
			};

			let color = if percent < zero { styles::NEGATIVE_COLOR } else { accent };
			frame.fill(&rounded(fill, radius), color);
		}

		// Modulation lives above the track, mirroring the ring around a knob, so
		// that the value underneath is never covered up.
		if let Some(modulated) = self.modulation {
			let modulated_percent = self.percent_of(modulated);
			let from = percent.min(modulated_percent);
			let to = percent.max(modulated_percent);
			let thickness = styles::u(0.5);
			let band = Rectangle {
				x: track.x + track.width * from,
				y: track.y - styles::u(0.5) - thickness,
				width: (track.width * (to - from)).max(thickness),
				height: thickness,
			};

			frame.fill(&rounded(band, thickness / 2.0), styles::MOD_COLOR);

			let marker = Point::new(track.x + track.width * modulated_percent, band.y + thickness / 2.0);
			frame.fill(&Path::circle(marker, thickness), styles::MOD_COLOR);
		}

		// A thumb, so the track reads as something that can be grabbed.
		let click = state.click_animator.calc();
		let thumb = Point::new(track.x + track.width * percent, track.y + track.height / 2.0);
		let thumb_radius = track_height * (0.8 - 0.2 * click);

		frame.fill(&Path::circle(thumb, thumb_radius), text_color);
		frame.fill(
			&Path::circle(thumb, (thumb_radius - styles::u(0.5)).max(0.5)),
			lighten(accent, brighten),
		);

		renderer.draw_geometry(frame.into_geometry());
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
		let state = tree.state.downcast_mut::<SliderState>();
		let bounds = layout.bounds();

		// The cursor the widget is handed was already translated by any scrollable
		// above it. The position inside a mouse event was not, and is off by the
		// scroll offset, which is how a control starts reacting in the wrong place.
		if let Some(position) = cursor.position() {
			state.current_mouse_pos = position;
		}

		let min = self.from.min(self.to);
		let max = self.from.max(self.to);
		let over = cursor.is_over(bounds);
		let fine = if input::modifiers().shift() { 0.1 } else { 1.0 };

		if let Some(origin) = state.last_mouse_pos {
			let position = cursor.position().unwrap_or(origin);
			let dragged = self.dragged(position.x - origin.x, bounds.width, fine);

			state.last_mouse_pos = Some(position);
			self.set(dragged, shell);
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
			Event::ButtonPressed(iced::mouse::Button::Left) if over => {
				let now = std::time::Instant::now();
				let double = state.last_press.is_some_and(|last| now.duration_since(last) < std::time::Duration::from_millis(300));
				state.last_press = if double { None } else { Some(now) };

				if double {
					if let Some(default) = self.default_value {
						self.set(default, shell);
					}
				} else {
					state.last_mouse_pos = cursor.position();
					state.click_animator.in_if_out();
				}
			},
			Event::ButtonReleased(iced::mouse::Button::Left) => {
				let was_dragging = state.last_mouse_pos.take().is_some();
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
				let notches = match delta {
					iced::mouse::ScrollDelta::Lines { y, .. } => *y,
					iced::mouse::ScrollDelta::Pixels { y, .. } => *y / 50.0,
				};

				if notches == 0.0 {
					return;
				}

				let step = if self.step > 0.0 {
					self.step
				} else if self.is_logarithmic {
					let from = self.from.max(f32::MIN_POSITIVE);
					let to = self.to.max(f32::MIN_POSITIVE);
					(to.log10() - from.log10()).abs() / 10.0
				} else {
					(max - min) * 0.01
				};

				let value = if self.is_logarithmic {
					self.value.max(f32::MIN_POSITIVE) * 10.0_f32.powf(notches * step * fine)
				} else {
					self.value + notches * step * fine
				};

				self.set(value, shell);
			},
			_ => {},
		}
	}
}

impl<'a, Messsage: 'a> From<Slider<'a, Messsage>> for Element<'a, Messsage> {
	fn from(value: Slider<'a, Messsage>) -> Self {
		Element::new(value)
	}
}

/// A rounded rectangle path.
fn rounded(rect: Rectangle, radius: f32) -> Path {
	Path::new(|builder| {
		builder.rounded_rectangle(rect.position(), rect.size(), radius.into());
	})
}
