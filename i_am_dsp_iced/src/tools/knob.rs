use std::ops::RangeInclusive;

use iced::{Element, Length, Point, Radians, Renderer, Size, Theme, advanced::{Widget, graphics::geometry::{Renderer as _, frame::Backend}, layout, renderer}, mouse::Event, theme::palette::lighten, widget::canvas::{self, Path, Stroke}};

use crate::{styles::{BORDER_WIDTH, BRIGHT_FACTOR}, tools::{Number, utils::Animator}};

/// A Simple knob controller.
pub struct Knob<'a, Message> {
	value: f32,
	step: f32,
	speed: f32,
	from: f32,
	to: f32,
	is_logarithmic: bool,
	on_change: Box<dyn Fn(f32) -> Message + 'a>,
	on_release: Option<Box<dyn Fn(f32) -> Message + 'a>>,
	width: Length,
	height: Length,
}

#[derive(Default)]
struct KnobState {
	current_mouse_pos: Point,
	last_mouse_pos: Option<(Point, f32)>,
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
			from,
			to,
			is_logarithmic: false,
			on_change: Box::new(move |value| on_change(T::from_f32(value))),
			on_release: None,
			width: Length::Fixed(16.0),
			height: Length::Fixed(16.0),
			// current_mouse_pos: Point::new(0.0, 0.0),
			// last_mouse_pos: None,
		}
	}

	pub fn on_release(self, on_release: impl Fn(f32) -> Message + 'a) -> Self {
		Self {
			on_release: Some(Box::new(on_release)),
			..self
		}
	}

	/// Set the width of the knob.
	pub fn width(self, width: impl Into<Length>) -> Self {
		Self {
			width: width.into(),
			..self
		}
	}

	/// Set the height of the knob.
	pub fn height(self, height: impl Into<Length>) -> Self {
		Self {
			height: height.into(),
			..self
		}
	}

	/// Set the speed of the knob.
	pub fn speed(self, speed: impl Into<f32>) -> Self {
		Self {
			speed: speed.into(),
			..self
		}
	}

	/// Set the value of the knob.
	pub fn value(self, value: impl Into<f32>) -> Self {
		Self {
			value: value.into(),
			..self
		}
	}

	/// Set the step of the knob.
	pub fn step(self, step: impl Into<f32>) -> Self {
		Self {
			step: step.into(),
			..self
		}
	}

	/// Set the logarithmic mode of the knob.
	pub fn logarithmic(self, log: bool) -> Self {
		Self {
			is_logarithmic: log,
			..self
		}
	}
}

impl<'a, Message> Widget<Message, Theme, Renderer> for Knob<'a, Message> {
	fn size(&self) -> iced::Size<Length> {
		Size::new(self.width, self.height)
	}

	fn layout(
		&mut self,
		_: &mut iced::advanced::widget::Tree,
		_: &Renderer,
		limits: &iced::advanced::layout::Limits,
	) -> iced::advanced::layout::Node {
		let size = limits.resolve(self.width, self.height, Size::new(16.0, 16.0));
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
		let mut frame = renderer.new_frame(bounds);
		let center = bounds.center();
		let radius = bounds.width.min(bounds.height) / 2.0;
		let knob_width = radius / 64.0 * 10.0;
		let current_value = if let Some((_, value)) = &state.last_mouse_pos  {
			*value
		}else {
			self.value
		};

		let percent = if self.is_logarithmic {
			(current_value.log10() - self.from.log10()) / (self.to.log10() - self.from.log10())
		}else {
			(current_value - self.from) / (self.to - self.from)
		}.max(0.0);

		let click_animator = state.click_animator.calc();
		let brighten_factor = state.hover_animator.calc() * BRIGHT_FACTOR;

		let path = Path::new(|builder| {
			builder.arc(canvas::path::Arc {
				radius: radius - knob_width / 2.0,
				center,
				start_angle: Radians(0.75 * std::f32::consts::PI),
				end_angle: Radians(2.25 * std::f32::consts::PI),
			});
		});

		frame.stroke(&path, Stroke::default().with_color(theme.extended_palette().background.weakest.color).with_width(knob_width));

		let path = Path::new(|builder| {
			builder.arc(canvas::path::Arc {
				radius: radius - knob_width / 2.0,
				center,
				start_angle: Radians(0.75 * std::f32::consts::PI),
				end_angle: Radians(0.75 * std::f32::consts::PI + percent * 1.5 * std::f32::consts::PI),
			});
		});

		let fill_color = lighten(theme.palette().primary, brighten_factor);

		frame.stroke(&path, Stroke::default().with_color(fill_color).with_width(knob_width));

		let path = Path::new(|builder| {
			let x = (percent * 1.5 * std::f32::consts::PI + 0.75 * std::f32::consts::PI).cos() * radius;
			let y = (percent * 1.5 * std::f32::consts::PI + 0.75 * std::f32::consts::PI).sin() * radius;

			builder.move_to(center);
			builder.line_to(Point::new(center.x + x, center.y + y));
		});

		let fill_color = lighten(theme.palette().text, brighten_factor);

		frame.stroke(&path, Stroke::default().with_color(fill_color).with_width(BORDER_WIDTH));

		let path = Path::new(|builder| {
			builder.circle(center, (radius - knob_width) * (1.0 - 0.1 * click_animator));
		});

		let fill_color = lighten(theme.extended_palette().background.strongest.color, brighten_factor);

		frame.fill(&path, fill_color);

		renderer.draw_geometry(frame.into_geometry());
	}

	fn state(&self) -> iced::advanced::widget::tree::State {
		iced::advanced::widget::tree::State::new(KnobState::default())
	}

	fn update(
		&mut self,
		tree: &mut iced::advanced::widget::Tree,
		event: &iced::Event,
		layout: layout::Layout<'_>,
		_: iced::advanced::mouse::Cursor,
		_: &Renderer,
		_: &mut dyn iced::advanced::Clipboard,
		shell: &mut iced::advanced::Shell<'_, Message>,
		_viewport: &iced::Rectangle,
	) {
		let state = tree.state.downcast_mut::<KnobState>();

		if let Some((last_mouse_pos, current_value)) = &mut state.last_mouse_pos {
			let min = self.from.min(self.to);
			let max = self.from.max(self.to);

			let delta_x = state.current_mouse_pos.x - last_mouse_pos.x;
			let delta_y = state.current_mouse_pos.y - last_mouse_pos.y;
			let change = delta_x - delta_y;
			let new_value = if self.is_logarithmic {
				*current_value * 10.0_f32.powf(change * self.speed)
			}else {
				*current_value + change * self.speed
			}.clamp(min, max);

			if new_value != self.value {
				self.value = new_value;
				shell.publish((self.on_change)(new_value));
			}

			*last_mouse_pos = state.current_mouse_pos;
			*current_value = new_value;
		}

		let clamped_value = self.value.clamp(self.from, self.to);

		if clamped_value != self.value {
			self.value = clamped_value;
			shell.publish((self.on_change)(clamped_value));
		}

		if layout.bounds().contains(state.current_mouse_pos) {
			state.hover_animator.in_if_out();
		}else {
			state.hover_animator.out_if_in();
		}

		let iced::Event::Mouse(mouse_event) = event else {
			return;
		};
		
		// dbg!(self.last_mouse_pos.is_none());

		match mouse_event {
			Event::CursorMoved { position } => {
				state.current_mouse_pos = *position;
			},
			Event::ButtonPressed(_) if state.last_mouse_pos.is_none() && layout.bounds().contains(state.current_mouse_pos) => {
				state.last_mouse_pos = Some((state.current_mouse_pos, self.value));
				state.click_animator.in_if_out();
			},
			Event::ButtonReleased(_) => {
				state.last_mouse_pos = None;
				state.click_animator.out_if_in();

				let min = self.from.min(self.to);
				let max = self.from.max(self.to);
				
				let new_value = if self.step > 0.0 {
					(self.value / self.step).round() * self.step
				}else {
					self.value
				}.clamp(min, max);

				if new_value != self.value {
					self.value = new_value;
					shell.publish((self.on_change)(new_value));
				}

				if let Some(on_release) = &self.on_release {
					shell.publish(on_release(self.value));
				}
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