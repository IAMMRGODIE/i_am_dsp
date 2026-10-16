//! A spectrogram: the last few moments of a spectrum, newest on the right.
//!
//! Where a spectrum shows one frame, this shows time: a column per update, with the
//! older ones sliding off to the left. It reduces whatever it is handed into the
//! number of bands it can draw, so a caller can push raw FFT bins and let the
//! display sort out the shape.
//!
//! The colour runs from the background through the accent and the warning colour to
//! white, which is the same order a meter uses for level: quiet, loud, louder than
//! that.

use std::sync::Mutex;

use iced::{
	Color, Point, Rectangle, Renderer, Size, Theme,
	mouse,
	theme::palette::mix,
	widget::canvas::{Cache, Path, Program},
};

use crate::{
	styles,
	tools::utils,
};

/// How many bands a column is reduced to.
const BANDS: usize = 20;

/// A scrolling spectrogram.
pub struct Spectrogram {
	/// The bands of every column, oldest first, BANDS per column.
	columns: Mutex<Vec<f32>>,
	cache: Cache,
	capacity: usize,
}

impl Spectrogram {
	/// Creates a spectrogram that remembers the given number of columns.
	pub fn new(capacity: usize) -> Self {
		let capacity = capacity.max(1);

		Self {
			columns: Mutex::new(Vec::with_capacity(capacity * BANDS)),
			cache: Cache::default(),
			capacity,
		}
	}

	/// Pushes one column: the magnitudes of a frame, from low to high.
	pub fn update(&self, bins: &[f32]) {
		if bins.is_empty() {
			return;
		}

		let mut column = [0.0f32; BANDS];

		for (band, slot) in column.iter_mut().enumerate() {
			let first = band * bins.len() / BANDS;
			let last = ((band + 1) * bins.len() / BANDS).max(first + 1).min(bins.len());
			let slice = &bins[first..last];

			*slot = (slice.iter().sum::<f32>() / slice.len() as f32).clamp(0.0, 1.0);
		}

		{
			let mut held = self.columns.lock().expect("the columns are never held across a panic");

			held.extend_from_slice(&column);

			let limit = self.capacity * BANDS;

			if held.len() > limit {
				let excess = held.len() - limit;
				held.drain(0..excess);
			}
		}

		self.cache.clear();
	}
}

/// The colour of a magnitude: the background, the accent, the warning colour, and
/// then white.
fn heat(magnitude: f32) -> Color {
	let stops = [
		styles::BACKGROUND_COLOR,
		styles::PRIMARY_COLOR,
		styles::WARNING_COLOR,
		Color::WHITE,
	];

	let scaled = magnitude.clamp(0.0, 1.0) * (stops.len() - 1) as f32;
	let index = (scaled.floor() as usize).min(stops.len() - 2);

	mix(stops[index], stops[index + 1], scaled - index as f32)
}

impl<Message> Program<Message> for Spectrogram {
	type State = ();

	fn draw(
		&self,
		_state: &Self::State,
		renderer: &Renderer,
		theme: &Theme,
		bounds: Rectangle,
		_cursor: mouse::Cursor,
	) -> Vec<iced::widget::canvas::Geometry<Renderer>> {
		let columns = self.columns.lock().expect("the columns are never held across a panic");
		let count = columns.len() / BANDS;

		let geometry = self.cache.draw(renderer, bounds.size(), |frame| {
			utils::card(theme, frame);

			let size = frame.size();
			let padding = utils::card_padding();
			let width = (size.width - padding * 2.0).max(1.0);
			let height = (size.height - padding * 2.0).max(1.0);

			if count == 0 {
				return;
			}

			// The newest column sits against the right edge, so a display that has
			// only just started fills in from the right rather than jumping about.
			let step = width / self.capacity as f32;
			let column_width = step.max(1.0);
			let band_height = height / BANDS as f32;

			for column in 0..count {
				let x = padding + width - (count - column) as f32 * step;

				for band in 0..BANDS {
					let magnitude = columns[column * BANDS + band];

					// Bands run from low to high upwards on the display.
					let y = padding + height - (band + 1) as f32 * band_height;

					frame.fill(
						&Path::new(|builder| {
							builder.rectangle(
								Point::new(x, y),
								Size::new(column_width + 0.5, band_height + 0.5),
							);
						}),
						heat(magnitude),
					);
				}
			}
		});

		vec![geometry]
	}
}
