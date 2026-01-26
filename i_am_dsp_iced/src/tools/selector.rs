use iced::{Element, Length, Point, Rectangle, Renderer, Size, Theme, advanced::{Widget, graphics::geometry::{Renderer as _, frame::Backend}, layout}, border::Radius, mouse::Event, theme::palette::lighten, widget::canvas::{Path, Stroke, Text}};

use crate::{styles::{BORDER_WIDTH, BRIGHT_FACTOR, PADDING}, tools::utils::Animator};

type ChangeFunc<'a, Message, Slot> = Box<dyn Fn(&Vec<Slot>, usize) -> Message + 'a>;

pub struct Selector<'a, Message, Slot: ToString> {
	slots: Vec<Slot>,
	current_id: usize,
	on_change: Option<ChangeFunc<'a, Message, Slot>>,
	width: Length,
	height: Length,
}

pub fn selector<'a, Message, Slot: ToString>(
	slots: Vec<Slot>,
	current_id: usize,
) -> Selector<'a, Message, Slot> {
	Selector::new(slots, current_id)
}

impl<'a, Message, Slot: ToString> Selector<'a, Message, Slot> {
	pub fn new(
		slots: Vec<Slot>,
		current_id: usize,
	) -> Self {
		Self {
			slots,
			on_change: None,
			current_id,
			width: Length::Fill,
			height: Length::Fixed(32.0),
		}
	}

	/// Change the current id and return the message if there is a change
	pub fn change_id(&mut self, id: usize) -> Option<Message> {
		let id = id % self.slots.len();
		if id != self.current_id {
			self.current_id = id;
			if let Some(f) = &self.on_change {
				Some(f(&self.slots, id))
			}else {
				None
			}
		}else {
			None
		}
	}

	pub fn current_slot(&self) -> &Slot {
		&self.slots[self.current_id % self.slots.len()]
	}

	pub fn on_change(self, f: impl  Fn(&Vec<Slot>, usize) -> Message + 'a) -> Self {
		Self {
			on_change: Some(Box::new(f)),
			..self
		}
	}

	pub fn remove_on_change(self) -> Self {
		Self {
			on_change: None,
			..self
		}
	}

	pub fn width(self, width: impl Into<Length>) -> Self {
		Self {
			width: width.into(),
			..self
		}
	}

	pub fn height(self, height: impl Into<Length>) -> Self {
		Self {
			height: height.into(),
			..self
		}
	}
}

#[derive(Default)]
struct SelectorState {
	is_pressed_on: bool,
	current_mouse_pos: Point,
	last_slot_id: usize,
	hover_animator: Animator,
	click_animator: Animator,
}

impl<'a, Message, Slot: ToString> Widget<Message, Theme, Renderer> for Selector<'a, Message, Slot> {
	fn size(&self) -> Size<Length> {
		Size::new(self.width, self.height)
	}

	fn state(&self) -> iced::advanced::widget::tree::State {
		iced::advanced::widget::tree::State::new(SelectorState {
			last_slot_id: self.current_id,
			..Default::default()
		})
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
		_: &iced::advanced::renderer::Style,
		layout: layout::Layout<'_>,
		_: iced::advanced::mouse::Cursor,
		_: &iced::Rectangle,
	) {
		let state = tree.state.downcast_ref::<SelectorState>();
		let bounds = layout.bounds();
		let slots_count = self.slots.len();

		let usable_width = bounds.width - 2.0 * BORDER_WIDTH;
		let usable_height = bounds.height - 2.0 * BORDER_WIDTH;

		let width = usable_width / slots_count as f32;
		let mut frame = renderer.new_frame(bounds);
		let radius = usable_width.min(usable_height) / 2.0;

		let bright_factor = BRIGHT_FACTOR * state.hover_animator.calc();

		let background = Path::new(|builder| {
			builder.rounded_rectangle(
				Point::new(bounds.x + BORDER_WIDTH, bounds.y + BORDER_WIDTH), 
				Size::new(usable_width, usable_height), 
				Radius::new(radius)
			);
		});

		let current_pos = width * self.current_id as f32;
		let last_pos = width * state.last_slot_id as f32;
		let click_animator = state.click_animator.calc();

		let x = last_pos + (current_pos - last_pos) * click_animator;

		let foreground = Path::new(|builder| {
			builder.rounded_rectangle(
				Point::new(x + bounds.x + BORDER_WIDTH + PADDING / 2.0, bounds.y + BORDER_WIDTH + PADDING / 2.0), 
				Size::new(width - PADDING, usable_height - PADDING), 
				Radius::new(radius)
			);
		});

		frame.fill(&background, theme.extended_palette().background.weakest.color);
		frame.stroke(&background, Stroke::default()
			.with_color(theme.extended_palette().background.strongest.color)
			.with_width(BORDER_WIDTH)
		);

		// let factor = if slots_count <= 1 {
		// 	0.0
		// }else {
		// 	state.last_slot_id as f32 / (slots_count - 1) as f32 * (1.0 - click_animator) +
		// 	self.current_id as f32 / (slots_count - 1) as f32 * click_animator
		// };
		let forground_color = theme.palette().primary;
		let foreground_color = lighten(forground_color, bright_factor);
		frame.fill(&foreground, foreground_color);

		for (i, slot) in self.slots.iter().enumerate() {
			let slot = slot.to_string();
			let mut slot = Text::from(slot);
			slot.align_x = iced::widget::text::Alignment::Center;
			slot.align_y = iced::alignment::Vertical::Center;
			slot.max_width = width;
			slot.position = Point::new(width * i as f32 + width / 2.0 + bounds.x, bounds.height / 2.0 + bounds.y);
			slot.color = if i == self.current_id {
				iced::theme::palette::mix(
					theme.extended_palette().background.weakest.text,
					theme.palette().text,
					click_animator,
				)
			}else {
				theme.extended_palette().background.weakest.text
			};

			slot.draw_with(|path, color| {
				frame.fill(&path, color);
			});
		}

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
		let state = tree.state.downcast_mut::<SelectorState>();
		let bounds = layout.bounds();
		let slots_count = self.slots.len();

		let width = bounds.width / slots_count as f32;
		let rect = Rectangle::new(
			Point::new(width * self.current_id as f32 + bounds.x, bounds.y), 
			Size::new(width, bounds.height)
		);

		if rect.contains(state.current_mouse_pos) {
			state.hover_animator.in_if_out();
		}else {
			state.hover_animator.out_if_in();
		}

		if !state.click_animator.is_animating() {
			state.last_slot_id = self.current_id;
		}

		let iced::Event::Mouse(mouse_event) = event else { return; };

		match mouse_event {
			Event::CursorMoved { position } => {
				state.current_mouse_pos = *position;
			},
			Event::ButtonPressed(_) if bounds.contains(state.current_mouse_pos) => {
				state.is_pressed_on = true;
			},
			Event::ButtonReleased(_) if bounds.contains(state.current_mouse_pos) && !state.click_animator.is_animating() && state.is_pressed_on => {
				for i in 0..self.slots.len() {
					let rect = Rectangle::new(
						Point::new(width * i as f32 + bounds.x, bounds.y), 
						Size::new(width, bounds.height)
					);
					if rect.contains(state.current_mouse_pos) {
						if let Some(message) = self.change_id(i) {
							shell.publish(message);
						}
						state.click_animator.in_now();
						state.click_animator.clear_start();
						break;
					}
				}
				state.is_pressed_on = false;
			},
			_ => {},
		}
	}
}

impl<'a, Messsage: 'a, Slot: ToString + 'a> From<Selector<'a, Messsage, Slot>> for Element<'a, Messsage> {
	fn from(value: Selector<'a, Messsage, Slot>) -> Self {
		Element::new(value)
	}
}