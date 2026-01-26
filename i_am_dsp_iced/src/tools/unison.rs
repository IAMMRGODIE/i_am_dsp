use std::sync::atomic::Ordering;

use i_am_dsp::prelude::{Adsr, Oscillator, Tuning};
use iced::{Point, Renderer, Theme, mouse::Cursor, widget::canvas::{Frame, Path, Program, Stroke}};
use portable_atomic::{AtomicBool, AtomicF32, AtomicUsize};

use crate::{styles::PADDING, tools::utils::{bend, card}};

pub struct UnisonEditor {
	pub random_pan: AtomicF32,
	pub random_phase: AtomicF32,
	pub random_phase_by_channel: AtomicBool,
	pub unison_detune: AtomicF32,
	pub unison_bend: AtomicF32,
	pub unison_blend: AtomicF32,
	pub unisons: AtomicUsize,
}

impl UnisonEditor {
	pub fn new<Osc: Oscillator<CHANNELS>, TuningSys: Tuning, const CHANNELS: usize>(adsr: &Adsr<Osc, TuningSys, CHANNELS>) -> Self {
		Self {
			random_pan: AtomicF32::new(adsr.random_pan),
			random_phase: AtomicF32::new(adsr.random_phase),
			random_phase_by_channel: AtomicBool::new(adsr.random_phase_by_channel),
			unison_detune: AtomicF32::new(adsr.unison_detune),
			unison_bend: AtomicF32::new(adsr.unison_bend),
			unison_blend: AtomicF32::new(adsr.unison_blend),
			unisons: AtomicUsize::new(adsr.unisons),
		}
	}

	pub fn adjust<Osc: Oscillator<CHANNELS>, TuningSys: Tuning, const CHANNELS: usize>(&self, adsr: &mut Adsr<Osc, TuningSys, CHANNELS>) {
		adsr.random_pan = self.random_pan.load(Ordering::Relaxed);
		adsr.random_phase = self.random_phase.load(Ordering::Relaxed);
		adsr.random_phase_by_channel = self.random_phase_by_channel.load(Ordering::Relaxed);
		adsr.unison_detune = self.unison_detune.load(Ordering::Relaxed);
		adsr.unison_bend = self.unison_bend.load(Ordering::Relaxed);
		adsr.unison_blend = self.unison_blend.load(Ordering::Relaxed);
		adsr.unisons = self.unisons.load(Ordering::Relaxed);
	}
}

impl<Message> Program<Message> for UnisonEditor {
	type State = ();

	fn draw(
		&self,
		_state: &Self::State,
		renderer: &Renderer,
		theme: &Theme,
		bounds: iced::Rectangle,
		_cursor: Cursor,
	) -> Vec<iced::widget::canvas::Geometry<Renderer>> {
		let mut frame = Frame::new(renderer, bounds.size());
		card(theme, &mut frame);

		let usable_width = bounds.width - 2.0 * PADDING;
		let usable_height = bounds.height - 2.0 * PADDING;

		let path = Path::new(|builder| {
			let unisons = self.unisons.load(Ordering::Relaxed);
			let unison_bend = self.unison_bend.load(Ordering::Relaxed);
			let unison_blend = self.unison_blend.load(Ordering::Relaxed);
			let unison_detune = self.unison_detune.load(Ordering::Relaxed);

			let mid_point = (unisons - 1) as f32 / 2.0;
			for i in 0..unisons {
				let index = if mid_point == 0.0 {
					0.0
				}else {
					(i as f32 - mid_point) / mid_point
				};

				let blend = index.abs() * unison_blend;
				let index = if index >= 0.0 {
					bend(index, unison_bend)
				}else {
					- bend(index.abs(), unison_bend)
				};
				let detune_factor = (unison_detune * index / 2.0 + 1.0) / 2.0;
				builder.move_to(Point::new(detune_factor * usable_width + PADDING, blend * usable_height + PADDING));
				builder.line_to(Point::new(detune_factor * usable_width + PADDING, usable_height + PADDING));
			}
		});

		frame.stroke(&path, Stroke::default().with_color(theme.palette().text).with_width(3.0));

		vec![frame.into_geometry()]
	}
}