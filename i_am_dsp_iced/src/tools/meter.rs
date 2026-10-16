//! A level meter: how loud a signal is, and how loud it was a moment ago.
//!
//! Two bars, one per channel: the whole bar is the peak and the brighter core is
//! the RMS, which is the pair worth seeing at once -- a signal whose peak is far
//! above its RMS is a transient, and one whose two bars match is a drone. A line
//! holds the loudest peak and falls back slowly, so a hit that lasted one block
//! is still visible after it is gone.
//!
//! The levels are fed in as linear amplitudes through atomics, so the audio
//! thread can write them without a lock and without allocating.

use std::sync::atomic::Ordering;

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

/// How quiet a signal can be before the meter stops showing it.
const FLOOR_DB: f32 = -60.0;
/// The loudest the meter shows: a little above full scale, so clipping is visible.
const CEILING_DB: f32 = 6.0;
/// Where a signal stops being comfortable, and where it stops being safe.
const HOT_DB: f32 = -12.0;
const CLIP_DB: f32 = -3.0;
/// How far the peak hold falls back on every update.
const HOLD_FALL_DB: f32 = 0.4;
/// How much of a channel's width the bar takes: the rest is the gap beside it.
const BAR_RATIO: f32 = 0.62;
/// The RMS core inside the bar.
const CORE_RATIO: f32 = 0.34;

/// A stereo level meter.
pub struct Meter {
	peak: [AtomicF32; 2],
	rms: [AtomicF32; 2],
	hold: [AtomicF32; 2],
	cache: Cache,
}

impl Default for Meter {
	fn default() -> Self {
		Self::new()
	}
}

impl Meter {
	/// Creates a meter with nothing on it.
	pub fn new() -> Self {
		Self {
			peak: [AtomicF32::new(0.0), AtomicF32::new(0.0)],
			rms: [AtomicF32::new(0.0), AtomicF32::new(0.0)],
			hold: [AtomicF32::new(0.0), AtomicF32::new(0.0)],
			cache: Cache::default(),
		}
	}

	/// Feeds one block: the peak and the RMS of each channel, as linear
	/// amplitudes. Callable from the audio thread.
	pub fn update(&self, peak: [f32; 2], rms: [f32; 2]) {
		for channel in 0..2 {
			let held = self.hold[channel].load(Ordering::Relaxed);
			let falling = to_db(held) - HOLD_FALL_DB;
			let level = to_db(peak[channel]).max(falling).max(FLOOR_DB - 1.0);

			self.peak[channel].store(peak[channel], Ordering::Relaxed);
			self.rms[channel].store(rms[channel], Ordering::Relaxed);
			self.hold[channel].store(db_to_amplitude(level), Ordering::Relaxed);
		}
	}
}

/// Turns an amplitude into decibels, with silence landing at the floor.
fn to_db(amplitude: f32) -> f32 {
	if amplitude <= 1e-6 {
		FLOOR_DB
	} else {
		20.0 * amplitude.log10()
	}
}

/// Turns decibels back into the amplitude that produced them.
fn db_to_amplitude(db: f32) -> f32 {
	10.0_f32.powf(db / 20.0)
}

/// Where a level sits on the bar, from the bottom to the top.
fn position(db: f32) -> f32 {
	((db - FLOOR_DB) / (CEILING_DB - FLOOR_DB)).clamp(0.0, 1.0)
}

/// The colour of the band a level falls in.
///
/// The bar is drawn band by band rather than in one colour, so it turns at the
/// points a meter is read at instead of switching colour wholesale.
fn band_color(db: f32, accent: iced::Color) -> iced::Color {
	if db > CLIP_DB {
		styles::ERROR_COLOR
	} else if db > HOT_DB {
		styles::WARNING_COLOR
	} else {
		accent
	}
}

/// Draws the part of a bar between two levels, split at the band boundaries.
fn fill_bands(
	frame: &mut iced::widget::canvas::Frame,
	track: Rectangle,
	top_db: f32,
	width: f32,
	accent: iced::Color,
	alpha: f32,
) {
	let height = track.height;
	let bar = Rectangle {
		x: track.x + (track.width - width) / 2.0,
		y: track.y,
		width,
		height,
	};

	for (low, high) in [
		(FLOOR_DB, HOT_DB),
		(HOT_DB, CLIP_DB),
		(CLIP_DB, CEILING_DB),
	] {
		let high = high.min(top_db);

		if high <= low {
			continue;
		}

		let from = height * (1.0 - position(low));
		let to = height * (1.0 - position(high));
		let segment = Rectangle {
			x: bar.x,
			y: bar.y + to,
			width: bar.width,
			height: (from - to).max(0.5),
		};
		let radius = (bar.width / 2.0).min(segment.height / 2.0);

		frame.fill(
			&Path::rounded_rectangle(segment.position(), segment.size(), radius.into()),
			band_color(high, accent).scale_alpha(alpha),
		);
	}
}

impl<Message> Program<Message> for Meter {
	type State = ();

	fn draw(
		&self,
		_state: &Self::State,
		renderer: &Renderer,
		theme: &Theme,
		bounds: Rectangle,
		_cursor: mouse::Cursor,
	) -> Vec<iced::widget::canvas::Geometry<Renderer>> {
		let palette = theme.extended_palette();
		let accent = theme.palette().primary;
		let geometry = self.cache.draw(renderer, bounds.size(), |frame| {
			utils::card(theme, frame);

			let size = frame.size();
			let padding = utils::card_padding();
			let gap = styles::u(styles::space::XS);
			let width = ((size.width - padding * 2.0 - gap) / 2.0).max(1.0);
			let top = padding;
			let height = (size.height - padding * 2.0).max(1.0);

			// The scale, behind the bars.
			for db in [-48.0, -36.0, -24.0, -12.0, 0.0] {
				let y = top + height * (1.0 - position(db));
				frame.stroke(
					&Path::new(|builder| {
						builder.move_to(Point::new(padding, y));
						builder.line_to(Point::new(size.width - padding, y));
					}),
					Stroke::default()
						.with_color(palette.background.strong.color)
						.with_width(1.0),
				);
			}

			for channel in 0..2 {
				let x = padding + channel as f32 * (width + gap);
				let track = Rectangle {
					x,
					y: top,
					width,
					height,
				};
				let radius = width / 2.0;

				frame.fill(
					&Path::rounded_rectangle(track.position(), track.size(), radius.into()),
					palette.background.weakest.color,
				);
				frame.stroke(
					&Path::rounded_rectangle(track.position(), track.size(), radius.into()),
					Stroke::default()
						.with_color(palette.background.strong.color)
						.with_width(1.0),
				);

				// The peak, in bands: the bar changes colour where a meter is read,
				// not all at once.
				let peak_db = to_db(self.peak[channel].load(Ordering::Relaxed));

				if peak_db > FLOOR_DB {
					fill_bands(frame, track, peak_db, width * BAR_RATIO, accent, 0.7);
				}

				// The RMS, as a brighter core inside it.
				let rms_db = to_db(self.rms[channel].load(Ordering::Relaxed));

				if rms_db > FLOOR_DB {
					fill_bands(frame, track, rms_db, width * CORE_RATIO, accent, 1.0);
				}

				// The peak hold: a thin line, with a cap so it reads as a mark rather
				// than the top of the bar.
				let hold_db = to_db(self.hold[channel].load(Ordering::Relaxed));

				if hold_db > FLOOR_DB {
					let y = top + height * (1.0 - position(hold_db));
					let line = Rectangle {
						x: x + width * 0.1,
						y: y - 1.0,
						width: width * 0.8,
						height: 2.0,
					};

					frame.fill(
						&Path::rounded_rectangle(line.position(), line.size(), 1.0.into()),
						lighten(theme.palette().text, 0.15),
					);
				}
			}
		});

		vec![geometry]
	}
}
