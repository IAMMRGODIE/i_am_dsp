//! A stereo meter: where the two channels sit against each other.
//!
//! The picture is a vectorscope. Every frame is a point at
//!
//! (side, mid) = ((right - left) / sqrt 2, (right + left) / sqrt 2)
//!
//! so a mono signal draws a vertical line, a wide one draws a horizontal line, and
//! something out of phase points downwards. The number a stereo meter is actually
//! read for is the correlation, which is drawn underneath: +1 is mono, 0 is wide
//! and uncorrelated, and a negative value is the thing that disappears when a mix
//! is summed.
//!
//! Frames are pushed in blocks and kept behind a lock, because a vectorscope wants
//! many points per frame and the audio thread has no business building a vector:
//! call this from the interface side, on a tick.

use std::sync::Mutex;

use iced::{
	Point, Rectangle, Renderer, Theme,
	mouse,
	theme::palette::lighten,
	widget::canvas::{Cache, Path, Program, Stroke},
};
use portable_atomic::AtomicF32;

use crate::{
	styles,
	tools::utils,
};

/// The reciprocal of the square root of two, which is how a frame is rotated onto
/// the mid and side axes.
const DIAGONAL: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// A stereo field display, with the correlation underneath.
pub struct StereoMeter {
	frames: Mutex<Vec<(f32, f32)>>,
	correlation: AtomicF32,
	capacity: usize,
	cache: Cache,
}

impl StereoMeter {
	/// Creates a display that keeps the given number of the newest frames.
	pub fn new(capacity: usize) -> Self {
		Self {
			frames: Mutex::new(Vec::with_capacity(capacity)),
			correlation: AtomicF32::new(0.0),
			capacity: capacity.max(1),
			cache: Cache::default(),
		}
	}

	/// Pushes a block of frames, one (left, right) pair per sample.
	pub fn update(&self, frames: &[(f32, f32)]) {
		let mut sum_lr = 0.0f64;
		let mut sum_ll = 0.0f64;
		let mut sum_rr = 0.0f64;

		for (left, right) in frames {
			sum_lr += (*left as f64) * (*right as f64);
			sum_ll += (*left as f64) * (*left as f64);
			sum_rr += (*right as f64) * (*right as f64);
		}

		let denominator = (sum_ll * sum_rr).sqrt();
		let correlation = if denominator > 1e-12 {
			(sum_lr / denominator) as f32
		} else {
			0.0
		};

		self.correlation
			.store(correlation.clamp(-1.0, 1.0), std::sync::atomic::Ordering::Relaxed);

		{
			let mut held = self.frames.lock().expect("the frames are never held across a panic");

			held.clear();
			held.extend(frames.iter().copied());

			// Keep the newest, if a caller hands over more than there is room for.
			if held.len() > self.capacity {
				let excess = held.len() - self.capacity;
				held.drain(0..excess);
			}
		}

		self.cache.clear();
	}

	/// The correlation of the last block, from -1 to 1.
	pub fn correlation(&self) -> f32 {
		self.correlation.load(std::sync::atomic::Ordering::Relaxed)
	}
}

impl<Message> Program<Message> for StereoMeter {
	type State = ();

	fn draw(
		&self,
		_state: &Self::State,
		renderer: &Renderer,
		theme: &Theme,
		bounds: Rectangle,
		_cursor: mouse::Cursor,
	) -> Vec<iced::widget::canvas::Geometry<Renderer>> {
		let frames = self.frames.lock().expect("the frames are never held across a panic");
		let color = theme.palette().primary;
		let correlation = self.correlation();

		let geometry = self.cache.draw(renderer, bounds.size(), |frame| {
			utils::card(theme, frame);

			let palette = theme.extended_palette();
			let size = frame.size();
			let padding = utils::card_padding();
			let width = (size.width - padding * 2.0).max(1.0);
			let height = (size.height - padding * 2.0).max(1.0);

			// The correlation groove along the bottom, with a gap above it so it does
			// not read as part of the field.
			let groove_height = styles::u(1.5);
			let gap = styles::u(2.5);
			let field_height = (height - groove_height - gap).max(1.0);
			let radius = field_height.min(width) / 2.0;
			let centre = Point::new(padding + width / 2.0, padding + field_height / 2.0);

			// The circle a full-scale signal reaches, and the two diagonals: a point
			// on one of them is all left or all right.
			frame.stroke(
				&Path::circle(centre, radius),
				Stroke::default()
					.with_color(palette.background.strongest.color)
					.with_width(1.0),
			);

			for diagonal in [1.0f32, -1.0] {
				frame.stroke(
					&Path::new(|builder| {
						let reach = radius * DIAGONAL;
						builder.move_to(Point::new(centre.x - reach, centre.y + reach * diagonal));
						builder.line_to(Point::new(centre.x + reach, centre.y - reach * diagonal));
					}),
					Stroke::default()
						.with_color(palette.background.strong.color)
						.with_width(1.0),
				);
			}

			if frames.len() > 1 {
				let position = |(left, right): (f32, f32)| {
					let side = (right - left) * DIAGONAL;
					let mid = (right + left) * DIAGONAL;

					Point::new(
						centre.x + side.clamp(-1.0, 1.0) * radius,
						centre.y - mid.clamp(-1.0, 1.0) * radius,
					)
				};

				let trace = Path::new(|builder| {
					builder.move_to(position(frames[0]));

					for frame in frames.iter().skip(1) {
						builder.line_to(position(*frame));
					}
				});

				frame.stroke(
					&trace,
					Stroke::default()
						.with_color(lighten(color, 0.1).scale_alpha(0.55))
						.with_width(1.0),
				);
			}

			// The correlation: a groove from -1 to 1, with the value riding in it the
			// way a slider's handle rides in its track, so the range it can travel is
			// the range that is drawn.
			let groove = Rectangle {
				x: padding,
				y: padding + field_height + gap,
				width,
				height: groove_height,
			};
			let rounding = groove.height / 2.0;

			frame.fill(
				&Path::rounded_rectangle(groove.position(), groove.size(), rounding.into()),
				palette.background.weakest.color,
			);
			frame.stroke(
				&Path::rounded_rectangle(groove.position(), groove.size(), rounding.into()),
				Stroke::default()
					.with_color(palette.background.strong.color)
					.with_width(1.0),
			);

			// The slots it is read against: -1, -0.5, 0, +0.5, +1.
			for slot in [-1.0f32, -0.5, 0.0, 0.5, 1.0] {
				let x = padding + width * (slot + 1.0) / 2.0;
				let tall = slot == 0.0 || slot.abs() == 1.0;
				let height = if tall { groove.height * 0.55 } else { groove.height * 0.3 };

				frame.fill(
					&Path::new(|builder| {
						builder.rectangle(
							Point::new(x - styles::px(0.5), groove.y + (groove.height - height) / 2.0),
							iced::Size::new(styles::px(1.0), height),
						);
					}),
					palette.background.strongest.color,
				);
			}

			let marker = padding + width * (correlation + 1.0) / 2.0;
			frame.fill(
				&Path::circle(Point::new(marker, groove.y + groove.height / 2.0), groove.height * 0.35),
				if correlation < 0.0 {
					styles::ERROR_COLOR
				} else {
					lighten(color, 0.1)
				},
			);
		});

		vec![geometry]
	}
}
