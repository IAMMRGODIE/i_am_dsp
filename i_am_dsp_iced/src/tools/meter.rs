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
	Point, Rectangle, Renderer, Size, Theme,
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
/// How far the peak hold falls back on every update.
const HOLD_FALL_DB: f32 = 0.4;

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

/// The colour a level is drawn in: the accent until it is hot, then the meter
/// says so.
fn level_color(db: f32, accent: iced::Color) -> iced::Color {
	if db > -3.0 {
		styles::ERROR_COLOR
	} else if db > -12.0 {
		styles::WARNING_COLOR
	} else {
		accent
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

				let peak_db = to_db(self.peak[channel].load(Ordering::Relaxed));

				// The peak, as the whole bar.
				if peak_db > FLOOR_DB {
					let filled = height * position(peak_db);
					let bar = Rectangle {
						x,
						y: top + height - filled,
						width,
						height: filled,
					};
					let bar_radius = (width / 2.0).min(filled / 2.0);

					frame.fill(
						&Path::rounded_rectangle(bar.position(), bar.size(), bar_radius.into()),
						level_color(peak_db, accent).scale_alpha(0.55),
					);
				}

				// The RMS, as the core of it.
				let rms_db = to_db(self.rms[channel].load(Ordering::Relaxed));

				if rms_db > FLOOR_DB {
					let filled = height * position(rms_db);
					let bar = Rectangle {
						x: x + width * 0.25,
						y: top + height - filled,
						width: width * 0.5,
						height: filled,
					};
					let bar_radius = (width * 0.25).min(filled / 2.0);

					frame.fill(
						&Path::rounded_rectangle(bar.position(), bar.size(), bar_radius.into()),
						lighten(level_color(rms_db, accent), 0.05),
					);
				}

				// The peak hold.
				let hold_db = to_db(self.hold[channel].load(Ordering::Relaxed));

				if hold_db > FLOOR_DB {
					let y = top + height * (1.0 - position(hold_db));
					frame.fill(
						&Path::new(|builder| {
							builder.rectangle(Point::new(x, y - 1.0), Size::new(width, 2.0));
						}),
						lighten(theme.palette().text, 0.1),
					);
				}
			}
		});

		vec![geometry]
	}
}
