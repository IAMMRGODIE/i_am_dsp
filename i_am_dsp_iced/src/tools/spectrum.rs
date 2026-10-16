//! A spectrum display: the shape of a signal, one frame at a time.
//!
//! The widget draws bins and does nothing else. Whoever owns the audio decides
//! how many bins there are, how they are spaced, and how much they are smoothed
//! -- a transform on the UI thread, or a bank of filters in the audio thread --
//! which keeps the widget useful for a spectrum analyser, an EQ's curve, and
//! anything else shaped like a row of magnitudes.

use std::sync::Mutex;

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
///
/// More than there are bins, because the smoothing below interpolates: the curve
/// is a shape, not a bar chart of the bins that happen to be there.
const POINTS: usize = 160;

/// A row of magnitudes, drawn left to right.
pub struct Spectrum {
	bins: Mutex<Vec<f32>>,
	cache: Cache,
	/// How wide the smoothing window is, as a fraction of the axis.
	smoothing: f32,
}

impl Default for Spectrum {
	fn default() -> Self {
		Self::new()
	}
}

impl Spectrum {
	/// Creates an empty spectrum display.
	pub fn new() -> Self {
		Self {
			bins: Mutex::new(Vec::new()),
			cache: Cache::default(),
			// About a twenty-fourth of the axis: a little over half an octave on a
			// twelve-octave display.
			smoothing: 1.0 / 24.0,
		}
	}

	/// Sets how wide the smoothing window is, as a fraction of the axis.
	///
	/// Zero draws the bins as they are.
	pub fn smoothing(mut self, width: f32) -> Self {
		self.smoothing = width.max(0.0);
		self.cache.clear();
		self
	}

	/// Replaces the displayed bins with magnitudes from 0 to 1, left to right.
	///
	/// Call this from the interface side, on a frame: the audio thread has no
	/// business allocating a vector.
	pub fn update(&self, bins: &[f32]) {
		{
			let mut held = self.bins.lock().expect("the bins are never held across a panic");
			held.clear();
			held.extend_from_slice(bins);
		}

		self.cache.clear();
	}
}

impl<Message> Program<Message> for Spectrum {
	type State = ();

	fn draw(
		&self,
		_state: &Self::State,
		renderer: &Renderer,
		theme: &Theme,
		bounds: Rectangle,
		_cursor: mouse::Cursor,
	) -> Vec<iced::widget::canvas::Geometry<Renderer>> {
		let bins = self.bins.lock().expect("the bins are never held across a panic");
		let palette = theme.extended_palette();
		let color = theme.palette().primary;
		let count = bins.len();

		let geometry = self.cache.draw(renderer, bounds.size(), |frame| {
			utils::card(theme, frame);

			let size = frame.size();
			let padding = utils::card_padding();
			let width = (size.width - padding * 2.0).max(1.0);
			let height = (size.height - padding * 2.0).max(1.0);

			if count == 0 {
				return;
			}

			// A few quiet rules, so a peak has somewhere to sit.
			for step in 1..4 {
				let y = padding + height * step as f32 / 4.0;
				frame.stroke(
					&Path::new(|builder| {
						builder.move_to(Point::new(padding, y));
						builder.line_to(Point::new(padding + width, y));
					}),
					Stroke::default()
						.with_color(palette.background.strong.color)
						.with_width(1.0),
				);
			}

			// Real FFT bins are evenly spaced in frequency, so on a logarithmic axis
			// the low end has a handful of them spread over a wide range: drawn as
			// they are, a spectrum is a row of spikes down there and a smooth line up
			// top. Averaging over a window that is a fixed fraction of the axis is
			// fractional-octave smoothing, which is exactly the resolution the low
			// end does not have on its own.
			//
			// This assumes what the display is for: the axis is logarithmic, so low
			// frequencies are on the left. Handing it linear bins would smooth the
			// wrong end.
			let window = (self.smoothing * (count - 1) as f32).max(0.5);
			let sample = |t: f32| -> f32 {
				let centre = t * (count - 1) as f32;
				let first = (centre - window).max(0.0) as usize;
				let last = (((centre + window).min((count - 1) as f32)) as usize + 1).min(count);
				let mut sum = 0.0;
				let mut weight = 0.0;

				for index in first..last {
					let distance = (index as f32 - centre).abs();
					let w = (1.0 - distance / (window + 1.0)).max(0.0);

					sum += bins[index] * w;
					weight += w;
				}

				if weight > 0.0 {
					(sum / weight).clamp(0.0, 1.0)
				} else {
					0.0
				}
			};

			let point = |index: usize| {
				let t = index as f32 / (POINTS - 1) as f32;

				Point::new(padding + width * t, padding + height * (1.0 - sample(t)))
			};

			let curve = Path::new(|builder| {
				builder.move_to(point(0));

				for index in 1..POINTS {
					builder.line_to(point(index));
				}
			});

			let area = Path::new(|builder| {
				builder.move_to(Point::new(padding, padding + height));

				for index in 0..POINTS {
					builder.line_to(point(index));
				}

				builder.line_to(Point::new(padding + width, padding + height));
				builder.close();
			});

			frame.fill(&area, color.scale_alpha(styles::ALPHA_FACTOR));
			frame.stroke(&curve, Stroke::default().with_color(lighten(color, 0.1)).with_width(1.5));
		});

		vec![geometry]
	}
}
