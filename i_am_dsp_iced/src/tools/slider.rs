use std::ops::RangeInclusive;

use iced::{Element, Length, Pixels, Point, Renderer, Size, Theme, advanced::{Widget, graphics::geometry::{Renderer as _, frame::Backend}, layout, renderer}, mouse::Event, theme::palette::lighten, widget::canvas::{Path, Stroke, Text}};

use crate::{styles::{BORDER_WIDTH, BRIGHT_FACTOR, PADDING}, tools::{Number, utils::Animator}};

pub struct Slider<'a, Message> {
	value: f32,
	step: f32,
	speed: f32,
	from: f32,
	to: f32,
	is_logarithmic: bool,
	on_change: Box<dyn Fn(f32) -> Message + 'a>,
	on_release: Option<Box<dyn Fn(f32) -> Message + 'a>>,
	width: Length,
	text: String,
	value_formatter: Option<Box<dyn Fn(f32) -> String + 'a>>,
}

#[derive(Default)]
struct SliderState {
	current_mouse_pos: Point,
	last_mouse_pos: Option<(Point, f32)>,
	hover_animator: Animator,
	click_animator: Animator,
}

pub fn slider<'a, Message, T: Number>(range: RangeInclusive<T>, value: T, on_change: impl Fn(T) -> Message + 'a) -> Slider<'a, Message> {
	Slider::new(range, value, on_change)
}

impl<'a, Message> Slider<'a, Message> {
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
			on_change: Box::new(move |value| on_change(T::from_f32(value))),
			width: Length::Fill,
			text: String::new(),
			value_formatter: None,
			on_release: None
		}
	}

	pub fn on_release(self, on_release: impl Fn(f32) -> Message + 'a) -> Self {
		Self {
			on_release: Some(Box::new(on_release)),
			..self
		}
	}

	pub fn width(self, width: impl Into<Length>) -> Self {
		Self {
			width: width.into(),
			..self
		}
	}

	pub fn text(self, text: impl Into<String>) -> Self {
		Self {
			text: text.into(),
			..self
		}
	}

	pub fn logarithmic(self) -> Self {
		Self {
			is_logarithmic: true,
			..self
		}
	}

	pub fn step(self, step: f32) -> Self {
		Self {
			step,
			..self
		}
	}

	pub fn speed(self, speed: f32) -> Self {
		Self {
			speed,
			..self
		}
	}

	pub fn formatter(self, formatter: impl Fn(f32) -> String + 'a) -> Self {
		Self {
			value_formatter: Some(Box::new(formatter)),
			..self
		}
	}
}

impl<'a, Message> Widget<Message, Theme, Renderer> for Slider<'a, Message> {
	fn size(&self) -> iced::Size<Length> {
		Size::new(self.width, Length::from(32.0))
	}

	fn layout(
		&mut self,
		_: &mut iced::advanced::widget::Tree,
		_: &Renderer,
		limits: &iced::advanced::layout::Limits,
	) -> iced::advanced::layout::Node {
		let size = limits.resolve(self.width, Length::from(32.0), Size::new(64.0, 32.0));
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
		// dbg!(bounds.height);

		let mut frame = renderer.new_frame(bounds);

		let click_animator = state.click_animator.calc();
		let brighten_factor = state.hover_animator.calc() * BRIGHT_FACTOR;
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

		let title_color = lighten(theme.extended_palette().background.weakest.text, brighten_factor);
		let value_color = lighten(theme.extended_palette().primary.base.color, brighten_factor);
		
		let title = Text {
			size: Pixels(16.0),
			align_x: iced::widget::text::Alignment::Left,
			color: title_color,
			position: bounds.position(),
			..Text::from(self.text.as_str())
		};

		let value_text = if let Some(formatter) = &self.value_formatter {
			formatter(self.value)
		}else {
			format!("{:.2}", self.value)
		};

		let value_text = Text {
			size: Pixels(16.0),
			align_x: iced::widget::text::Alignment::Right,
			color: value_color,
			position: Point::new(bounds.width + bounds.x, bounds.y),
			..Text::from(value_text)
		};

		title.draw_with(|path, color| {
			frame.fill(&path, color);
		});

		value_text.draw_with(|path, color| {
			frame.fill(&path, color);
		});

		let background = Path::new(|builer| {
			let size: Size = Size::new(
				bounds.width - BORDER_WIDTH * 2.0, 
				bounds.height - BORDER_WIDTH * 2.0 - 16.0 - PADDING
			);
			let radius = size.height.min(size.width) / 2.0;

			builer.rounded_rectangle(
				Point::new(
					bounds.x + BORDER_WIDTH, 
					bounds.y + 16.0 + BORDER_WIDTH + PADDING,
				), 
				size, 
				radius.into()
			);
		});

		let background_color = theme.extended_palette().background.weakest.color;
		let stroke_color = lighten(theme.extended_palette().background.strongest.color, brighten_factor);

		frame.fill(&background, background_color);
		frame.stroke(&background, Stroke::default().with_color(stroke_color).with_width(BORDER_WIDTH));

		let foreground = Path::new(|builer| {
			const SHRINK_RATIO: f32 = 0.5;

			let size: Size = Size::new(
				(bounds.width - BORDER_WIDTH * 2.0) * percent, 
				bounds.height - BORDER_WIDTH * 2.0 - 16.0 - PADDING
			);
			let position = Point::new(
				bounds.x + BORDER_WIDTH + size.width.min(PADDING) * click_animator * SHRINK_RATIO / 2.0, 
				bounds.y + 16.0 + BORDER_WIDTH + size.height * click_animator * SHRINK_RATIO / 2.0 + PADDING,
			);

			let size: Size = Size::new(
				(size.width - size.width.min(PADDING) * click_animator * SHRINK_RATIO).max(0.0),
				size.height - size.height * click_animator * SHRINK_RATIO
			);
			let radius = size.height.min(size.width) / 2.0;

			builer.rounded_rectangle(position, size, radius.into());
		});

		frame.fill(&foreground, value_color);

		renderer.draw_geometry(frame.into_geometry());
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
		_: &iced::Rectangle,
	) {
		let state = tree.state.downcast_mut::<SliderState>();

		if let Some((last_mouse_pos, current_value)) = &mut state.last_mouse_pos {
			let min = self.from.min(self.to);
			let max = self.from.max(self.to);
			
			let delta_x = state.current_mouse_pos.x - last_mouse_pos.x;
			let change = delta_x / layout.bounds().width * (max - min);
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

impl<'a, Messsage: 'a> From<Slider<'a, Messsage>> for Element<'a, Messsage> {
	fn from(value: Slider<'a, Messsage>) -> Self {
		Element::new(value)
	}
}