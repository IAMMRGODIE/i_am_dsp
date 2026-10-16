//! What a shaper does to a signal, drawn as the curve it is.
//!
//! A drive effect's whole character is its transfer function, and a row of knobs
//! cannot show it: the curve can. The area between the curve and the straight
//! line is what the shaper adds -- where it bulges, it is making harmonics; where
//! it flattens, it is clipping them off.

use iced::{
	Point, Rectangle, Renderer, Theme,
	mouse,
	theme::palette::lighten,
	widget::canvas::{Cache, Path, Program, Stroke},
};

use crate::{
	styles,
	tools::utils,
};

/// How many points the curve is sampled at.
const POINTS: usize = 128;

/// The transfer function of a shaper.
pub struct Transfer {
	curve: Box<dyn Fn(f32) -> f32 + Send + Sync>,
	cache: Cache,
}

/// Creates a display of the given transfer function, over an input of -1 to 1.
pub fn transfer(curve: impl Fn(f32) -> f32 + Send + Sync + 'static) -> Transfer {
	Transfer {
		curve: Box::new(curve),
		cache: Cache::default(),
	}
}

impl Transfer {
	/// Replaces the curve, for a shaper whose shape follows its controls.
	pub fn set_curve(&mut self, curve: impl Fn(f32) -> f32 + Send + Sync + 'static) {
		self.curve = Box::new(curve);
		self.cache.clear();
	}
}

impl<Message> Program<Message> for Transfer {
	type State = ();

	fn draw(
		&self,
		_state: &Self::State,
		renderer: &Renderer,
		theme: &Theme,
		bounds: Rectangle,
		_cursor: mouse::Cursor,
	) -> Vec<iced::widget::canvas::Geometry<Renderer>> {
		let curve_color = theme.palette().primary;
		let geometry = self.cache.draw(renderer, bounds.size(), |frame| {
			utils::card(theme, frame);

			let size = frame.size();
			let padding = utils::card_padding();
			let width = (size.width - padding * 2.0).max(1.0);
			let height = (size.height - padding * 2.0).max(1.0);
			let identity_color = theme.extended_palette().background.strongest.color;

			// The axes, through the middle: what goes in is the horizontal, what
			// comes out is the vertical.
			let to_screen = |input: f32, output: f32| {
				Point::new(
					padding + width * (input + 1.0) / 2.0,
					padding + height * (1.0 - (output + 1.0) / 2.0),
				)
			};

			for step in 1..4 {
				let x = padding + width * step as f32 / 4.0;
				let y = padding + height * step as f32 / 4.0;

				frame.stroke(
					&Path::new(|builder| {
						builder.move_to(Point::new(x, padding));
						builder.line_to(Point::new(x, padding + height));
						builder.move_to(Point::new(padding, y));
						builder.line_to(Point::new(padding + width, y));
					}),
					Stroke::default().with_color(identity_color.scale_alpha(0.4)).with_width(1.0),
				);
			}

			// The straight line: what the shaper would do if it did nothing.
			frame.stroke(
				&Path::new(|builder| {
					builder.move_to(to_screen(-1.0, -1.0));
					builder.line_to(to_screen(1.0, 1.0));
				}),
				Stroke::default().with_color(identity_color).with_width(1.0),
			);

			let sample = |index: usize| {
				let input = -1.0 + 2.0 * index as f32 / (POINTS - 1) as f32;
				let output = (self.curve)(input).clamp(-1.5, 1.5);

				(input, output)
			};

			// What the shaper adds, shaded: the area between the two curves.
			let added = Path::new(|builder| {
				let (input, output) = sample(0);
				builder.move_to(to_screen(input, input));
				builder.line_to(to_screen(input, output));

				for index in 1..POINTS {
					let (input, output) = sample(index);
					builder.line_to(to_screen(input, output));
				}

				for index in (0..POINTS).rev() {
					let (input, _) = sample(index);
					builder.line_to(to_screen(input, input));
				}

				builder.close();
			});

			let line = Path::new(|builder| {
				let (input, output) = sample(0);
				builder.move_to(to_screen(input, output));

				for index in 1..POINTS {
					let (input, output) = sample(index);
					builder.line_to(to_screen(input, output));
				}
			});

			frame.fill(&added, curve_color.scale_alpha(styles::ALPHA_FACTOR));
			frame.stroke(&line, Stroke::default().with_color(lighten(curve_color, 0.1)).with_width(1.5));
		});

		vec![geometry]
	}
}
