//! A push button that brightens instead of blinking.
//!
//! iced styles its own button from the theme, and a style closure is a pure
//! function of the theme and the widget status, so it cannot animate anything.
//! This one carries an [Animator] for the hover and the press, which is how the
//! knobs already behave: the whole button gets lighter, the colour never
//! changes, and a press pushes the face in a little.

use std::marker::PhantomData;

use iced::{
	Element, Length, Point, Renderer, Size, Theme, alignment,
	advanced::{Widget, graphics::geometry::{Renderer as _, frame::Backend}, layout, renderer},
	mouse::{self, Event},
	theme::palette::{lighten, mix},
	widget::canvas::{Path, Stroke},
};

use crate::{
	styles::{self, BRIGHT_FACTOR},
	tools::utils::{self, Animator},
};

/// How much wider a character is assumed to be when guessing a label width.
const CHARACTER_WIDTH: f32 = 0.62;

/// A push button.
pub struct PushButton<'a, Message> {
	label: String,
	on_press: Option<Message>,
	accent: bool,
	width: Option<Length>,
	marker: PhantomData<&'a ()>,
}

#[derive(Default)]
struct ButtonState {
	hover_animator: Animator,
	click_animator: Animator,
	/// 0 while the button is not selected, 1 while it is.
	accent_animator: Animator,
	/// The selection this button had the last time it was reconciled, so that a
	/// fade only runs when the state actually changes.
	accent: Option<bool>,
	armed: bool,
}

/// Create a new push button with the given label.
pub fn push_button<'a, Message>(label: impl Into<String>) -> PushButton<'a, Message> {
	PushButton {
		label: label.into(),
		on_press: None,
		accent: false,
		width: None,
		marker: PhantomData,
	}
}

impl<'a, Message> PushButton<'a, Message> {
	/// Set the message published when the button is clicked.
	pub fn on_press(self, message: Message) -> Self {
		Self { on_press: Some(message), ..self }
	}

	/// Draw the button in the accent colour, for a selected state.
	pub fn accent(self, accent: bool) -> Self {
		Self { accent, ..self }
	}

	/// Set the width of the button.
	pub fn width(self, width: impl Into<Length>) -> Self {
		Self { width: Some(width.into()), ..self }
	}

	/// The width the button asks for when the caller does not set one.
	fn intrinsic_width(&self) -> f32 {
		let text = self.label.chars().count() as f32 * styles::px(styles::font::VALUE) * CHARACTER_WIDTH;

		text + 2.0 * styles::u(styles::space::M)
	}

	/// The height of a row control.
	fn intrinsic_height(&self) -> f32 {
		styles::u(styles::size::ROW)
	}
}

impl<'a, Message: Clone> Widget<Message, Theme, Renderer> for PushButton<'a, Message> {
	fn size(&self) -> iced::Size<Length> {
		Size::new(
			self.width.unwrap_or(Length::Fixed(self.intrinsic_width())),
			Length::Fixed(self.intrinsic_height()),
		)
	}

	fn layout(
		&mut self,
		_: &mut iced::advanced::widget::Tree,
		_: &Renderer,
		limits: &iced::advanced::layout::Limits,
	) -> iced::advanced::layout::Node {
		let width = self.width.unwrap_or(Length::Fixed(self.intrinsic_width()));
		let height = Length::Fixed(self.intrinsic_height());
		let size = limits.resolve(width, height, Size::new(self.intrinsic_width(), self.intrinsic_height()));
		layout::Node::new(size)
	}

	fn tag(&self) -> iced::advanced::widget::tree::Tag {
		// Without a tag of its own this widget inherits the stateless one, which
		// is the tag of the empty tree a root element is diffed against: the two
		// match, the state is never built, and the first event panics on the
		// downcast. A widget with state has to say so.
		iced::advanced::widget::tree::Tag::of::<ButtonState>()
	}

	fn state(&self) -> iced::advanced::widget::tree::State {
		// A button that is born selected is born at its final colour: the fade is
		// for a change of state, not for the first frame of the interface.
		let mut accent_animator = Animator::new();
		accent_animator.set(if self.accent { 1.0 } else { 0.0 });

		let state = ButtonState {
			accent: Some(self.accent),
			accent_animator,
			..ButtonState::default()
		};

		iced::advanced::widget::tree::State::new(state)
	}

	fn diff(&self, tree: &mut iced::advanced::widget::Tree) {
		// Becoming selected is a change the application made, and no event follows
		// it: update() would never run, so the fade has to be started here, where
		// the new widget meets the state the old one left behind. Only a real
		// change animates; the first sight of a button does not.
		let state = tree.state.downcast_mut::<ButtonState>();

		if should_animate(state.accent, self.accent) {
			if self.accent {
				state.accent_animator.in_now();
			} else {
				state.accent_animator.out_now();
			}
		}

		state.accent = Some(self.accent);
		tree.children.clear();
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
		let state = tree.state.downcast_ref::<ButtonState>();
		let bounds = layout.bounds();
		let palette = theme.extended_palette();
		let hover = state.hover_animator.calc();
		let click = state.click_animator.calc();

		// Selecting fades in rather than switching: a hard cut to the accent
		// colour reads as a glitch when it happens under the pointer.
		let accent = state.accent_animator.calc();
		let base = mix(palette.background.strong.color, theme.palette().primary, accent);
		let text = mix(palette.background.strong.text, palette.primary.base.text, accent);

		// Brighten the whole face, never recolour it.
		let background = lighten(base, hover * BRIGHT_FACTOR * 2.0);
		let border = lighten(palette.background.strongest.color, hover * BRIGHT_FACTOR);
		let text_color = lighten(text, hover * BRIGHT_FACTOR * 2.0);

		// A stroke is centred on its path, so half of it lands outside the shape.
		// The face is inset by half the stroke width to give both halves room,
		// and a press pushes it in from there; without this the border is drawn
		// half outside the widget and clipped, which reads as a faded edge.
		let border_width = styles::px(1.0);
		let inset = border_width / 2.0 + styles::px(1.0) * click;
		let face = iced::Rectangle {
			x: bounds.x + inset,
			y: bounds.y + inset,
			width: (bounds.width - inset * 2.0).max(1.0),
			height: (bounds.height - inset * 2.0).max(1.0),
		};
		let radius = styles::u(styles::size::CONTROL_RADIUS);

		let mut frame = renderer.new_frame(bounds);
		let path = Path::new(|builder| {
			builder.rounded_rectangle(face.position(), face.size(), radius.into());
		});

		frame.fill(&path, background);
		frame.stroke(&path, Stroke::default().with_color(border).with_width(border_width));

		utils::draw_text(
			&mut frame,
			&self.label,
			Point::new(face.center_x(), face.center_y()),
			iced::widget::text::Alignment::Center,
			alignment::Vertical::Center,
			styles::px(styles::font::VALUE),
			styles::font::VALUE_FONT,
			text_color,
			face.width,
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
		let state = tree.state.downcast_mut::<ButtonState>();
		let bounds = layout.bounds();
		let over = cursor.is_over(bounds);

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
				state.armed = true;
				state.click_animator.in_if_out();
			},
			Event::ButtonReleased(mouse::Button::Left) => {
				let armed = state.armed;
				state.armed = false;
				state.click_animator.out_if_in();

				// A button fires on release, and only if the press started here.
				if armed && over && let Some(message) = &self.on_press {
					shell.publish(message.clone());
				}
			},
			_ => {},
		}
	}
}

/// Whether a change of selection should be played rather than snapped.
///
/// A button that is born selected is born at its final colour: the first sight of
/// a widget is not a change of state, and animating it would make the interface
/// fade in from nothing every time it is opened.
fn should_animate(previous: Option<bool>, current: bool) -> bool {
	previous.is_some_and(|previous| previous != current)
}

impl<'a, Message: Clone + 'a> From<PushButton<'a, Message>> for Element<'a, Message> {
	fn from(value: PushButton<'a, Message>) -> Self {
		Element::new(value)
	}
}

#[cfg(test)]
mod tests {
	use super::should_animate;

	#[test]
	fn only_a_real_change_of_selection_animates() {
		assert!(!should_animate(None, true), "a button born selected must not fade in");
		assert!(!should_animate(None, false));
		assert!(!should_animate(Some(true), true));
		assert!(!should_animate(Some(false), false));
		assert!(should_animate(Some(false), true));
		assert!(should_animate(Some(true), false));
	}
}
