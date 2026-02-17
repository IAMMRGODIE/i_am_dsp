use std::sync::{Arc, atomic::Ordering};

use i_am_dsp::prelude::{Adsr, AtomicValue, Oscillator, Paramed, Tuning};
use iced::{Point, Renderer, Theme, mouse::Cursor, widget::canvas::{Frame, Path, Program, Stroke}};
use crate::{styles::PADDING, tools::utils::{bend, card}};

pub struct UnisonEditor {
	pub random_pan: Arc<AtomicValue>,
	pub random_phase: Arc<AtomicValue>,
	pub unison_detune: Arc<AtomicValue>,
	pub unison_bend: Arc<AtomicValue>,
	pub unison_blend: Arc<AtomicValue>,
	pub random_phase_by_channel: Arc<AtomicValue>,
	pub unisons: Arc<AtomicValue>,
}

impl UnisonEditor {
	pub fn new<Osc: Oscillator<CHANNELS>, TuningSys: Tuning, const CHANNELS: usize>(adsr: &Paramed<Adsr<Osc, TuningSys, CHANNELS>>) -> Self {
		let param_map = adsr.param_map();

		Self {
			random_pan: param_map.get("random_pan").unwrap(),
			random_phase: param_map.get("random_phase").unwrap(),
			random_phase_by_channel: param_map.get("random_phase_by_channel").unwrap(),
			unison_detune: param_map.get("unison_detune").unwrap(),
			unison_bend: param_map.get("unison_bend").unwrap(),
			unison_blend: param_map.get("unison_blend").unwrap(),
			unisons: param_map.get("unisons").unwrap(),
		}
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
			let unisons = self.unisons.load(Ordering::Relaxed).int().unwrap() as usize;
			let unison_bend = self.unison_bend.load(Ordering::Relaxed).float().unwrap();
			let unison_blend = self.unison_blend.load(Ordering::Relaxed).float().unwrap();
			let unison_detune = self.unison_detune.load(Ordering::Relaxed).float().unwrap();

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