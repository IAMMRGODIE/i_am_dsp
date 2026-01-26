use std::{collections::HashMap, sync::atomic::Ordering};

use i_am_dsp::prelude::{Adsr, Oscillator, Tuning};
use iced::{Point, Renderer, Theme, mouse::{Cursor, Event}, widget::{Action, canvas::{Frame, Path, Program, Stroke}}};
use portable_atomic::AtomicF32;

use crate::{styles::{ALPHA_FACTOR, PADDING}, tools::utils::{Animator, bend, card}};

#[derive(Debug)]
/// Adsr editor for real-time audio processing.
pub struct AdsrEditor {
	pub delay_time: AtomicF32,
	pub attack_time: AtomicF32,
	pub hold_time: AtomicF32,
	pub decay_time: AtomicF32,
	pub sustain_level: AtomicF32,
	pub release_time: AtomicF32,

	pub attack_bend: AtomicF32,
	pub decay_bend: AtomicF32,
	pub release_bend: AtomicF32,
}

impl AdsrEditor {
	/// Create a new AdsrEditor from an Adsr object.
	pub fn new<Osc: Oscillator<CHANNELS>, TuningSys: Tuning, const CHANNELS: usize>(adsr: &Adsr<Osc, TuningSys, CHANNELS>) -> Self {
		Self {
			delay_time: AtomicF32::new(adsr.delay_time),
			attack_time: AtomicF32::new(adsr.attack_time),
			hold_time: AtomicF32::new(adsr.hold_time),
			decay_time: AtomicF32::new(adsr.decay_time),
			sustain_level: AtomicF32::new(adsr.sustain_level),
			release_time: AtomicF32::new(adsr.release_time),

			attack_bend: AtomicF32::new(adsr.attack_bend),
			decay_bend: AtomicF32::new(adsr.decay_bend),
			release_bend: AtomicF32::new(adsr.release_bend),
		}
	}

	/// Adjust the Adsr object with the values of the editor.
	pub fn adjust<Osc: Oscillator<CHANNELS>, TuningSys: Tuning, const CHANNELS: usize>(&self, adsr: &mut Adsr<Osc, TuningSys, CHANNELS>) {
		adsr.delay_time = self.delay_time.load(Ordering::Relaxed);
		adsr.attack_time = self.attack_time.load(Ordering::Relaxed);
		adsr.hold_time = self.hold_time.load(Ordering::Relaxed);
		adsr.decay_time = self.decay_time.load(Ordering::Relaxed);
		adsr.sustain_level = self.sustain_level.load(Ordering::Relaxed);
		adsr.release_time = self.release_time.load(Ordering::Relaxed);

		adsr.attack_bend = self.attack_bend.load(Ordering::Relaxed);
		adsr.decay_bend = self.decay_bend.load(Ordering::Relaxed);
		adsr.release_bend = self.release_bend.load(Ordering::Relaxed);
	}

	fn get_circle_positions(&self, usable_width: f32, usable_height: f32, with_path: bool) -> (Vec<(Point, NodeId)>, f32, Option<Path>) {
		let delay_time = self.delay_time.load(Ordering::Relaxed);
		let attack_time = self.attack_time.load(Ordering::Relaxed);
		let hold_time = self.hold_time.load(Ordering::Relaxed);
		let decay_time = self.decay_time.load(Ordering::Relaxed);
		let sustain_level = self.sustain_level.load(Ordering::Relaxed);
		let release_time = self.release_time.load(Ordering::Relaxed);

		let total_time = delay_time + attack_time + hold_time + decay_time + release_time;

		let attack_bend = self.attack_bend.load(Ordering::Relaxed);
		let decay_bend = self.decay_bend.load(Ordering::Relaxed);
		let release_bend = self.release_bend.load(Ordering::Relaxed);

		let sample_time = |t: f32| -> f32 {
			let mut time = t * total_time;

			if time < delay_time {
				return 0.0;
			}
			time -= delay_time;

			if time < attack_time {
				let t = time / attack_time;
				return bend(t, attack_bend);
			}
			time -= attack_time;

			if time < hold_time {
				return 1.0;
			}
			time -= hold_time;

			if time < decay_time {
				let t = time / decay_time;
				let a = bend(t, decay_bend);
				let v = 1.0 + (sustain_level - 1.0) * a;
				return v;
			}

			time -= decay_time;

			if time < release_time {
				let t = time / release_time;
				let a = bend(t, release_bend);
				let v = (1.0 - a) * sustain_level;
				return v;
			}

			0.0
		};

		let delay_x = delay_time / total_time * usable_width + PADDING;
		let attack_x = (delay_time + attack_time) / total_time * usable_width + PADDING;
		let hold_x = (delay_time + attack_time + hold_time) / total_time * usable_width + PADDING;
		let decay_x = (delay_time + attack_time + hold_time + decay_time) / total_time * usable_width + PADDING;

		let points = vec![
			(
				Point::new(usable_width + PADDING, usable_height + PADDING),
				NodeId::ReleaseEnd,
			),
			(
				Point::new(decay_x, usable_height * (1.0 - sustain_level) + PADDING),
				NodeId::DecayEnd,
			),
			(
				Point::new(hold_x, PADDING),
				NodeId::HoldEnd,
			),
			(
				Point::new(attack_x, PADDING),
				NodeId::AttackEnd,
			),
			(
				Point::new(delay_x, usable_height + PADDING),
				NodeId::DelayEnd,
			),

			(
				Point::new(
					(delay_x + attack_x) / 2.0,
					usable_height * (1.0 - sample_time((delay_time + attack_time / 2.0) / total_time)) + PADDING
				),
				NodeId::AttackBend,
			),
			(
				Point::new(
					(hold_x + decay_x) / 2.0,
					usable_height * (1.0 - sample_time((delay_time + attack_time + hold_time + decay_time / 2.0) / total_time)) + PADDING
				),
				NodeId::DecayBend,
			),
			(
				Point::new(
					(usable_width + PADDING + decay_x) / 2.0,
					usable_height * (1.0 - sample_time((delay_time + attack_time + hold_time + decay_time + release_time / 2.0) / total_time)) + PADDING
				),
				NodeId::ReleaseBend,
			),
		];

		if with_path {
			const SAMPLE_PER_PX: f32 = 0.5;

			let sample_points = ((usable_width * SAMPLE_PER_PX) as usize).max(2);

			let path = Path::new(|builder| {
				builder.move_to(Point::new(PADDING, usable_height + PADDING));
				for i in 0..sample_points {
					let t = i as f32 / (sample_points - 1) as f32;
					let x = PADDING + usable_width * t;
					let y = usable_height * (1.0 - sample_time(t)) + PADDING;

					builder.line_to(Point::new(x, y));
				}

				builder.line_to(Point::new(PADDING + usable_width, PADDING + usable_height));
			});
			(points, total_time, Some(path))
		}else {
			(points, total_time, None)
		}
	}
}

#[derive(Default)]
pub struct AdsrState {
	pub current_mouse_pos: Point,
	hovering: HashMap<NodeId, Animator>,
	dragging: Option<DraggingInfo>,
	dragging_animator: Animator,
	last_drag_id: Option<NodeId>,
}

struct DraggingInfo {
	start_node_id: NodeId,
	total_time: f32,
	last_pos: Point,
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
#[derive(Debug, Clone, Copy)]
#[derive(Hash)]
enum NodeId {
	DelayEnd,
	AttackEnd,
	HoldEnd,
	DecayEnd,
	ReleaseEnd,

	AttackBend,
	DecayBend,
	ReleaseBend,
}

impl NodeId {
	fn is_bend(&self) -> bool {
		matches!(self, NodeId::AttackBend | NodeId::DecayBend | NodeId::ReleaseBend)
	}
}

impl<Message> Program<Message> for AdsrEditor {
	type State = AdsrState;

	fn update(
		&self,
		state: &mut Self::State,
		event: &iced::Event,
		bounds: iced::Rectangle,
		_cursor: Cursor,
	) -> Option<iced::widget::Action<Message>> {
		let usable_width = bounds.width - 2.0 * PADDING;
		let usable_height = bounds.height - 2.0 * PADDING;

		if let Some(info) = &mut state.dragging {
			if info.start_node_id.is_bend() {
				let delta_y = state.current_mouse_pos.y - info.last_pos.y;
				let mut bend_delta = - delta_y / usable_height * 10.0;

				let _ = match &info.start_node_id {
					NodeId::AttackBend => {
						bend_delta = - bend_delta;
						&self.attack_bend
					},
					NodeId::DecayBend => {
						&self.decay_bend
					},
					NodeId::ReleaseBend => {
						&self.release_bend
					},
					_ => unreachable!(),
				}.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |inner| {
					Some((inner + bend_delta).clamp(-10.0, 10.0))
				});
			}else {
				let delta_x = state.current_mouse_pos.x - info.last_pos.x;
				let time_delta = delta_x / usable_width * info.total_time;

				let _ = match &info.start_node_id {
					NodeId::DelayEnd => {
						&self.delay_time
					},
					NodeId::AttackEnd => {
						&self.attack_time
					},
					NodeId::HoldEnd => {
						&self.hold_time
					},
					NodeId::DecayEnd => {
						let delta_y = state.current_mouse_pos.y - info.last_pos.y;
						let y_delta = - delta_y / usable_height;
						
						let _ = self.sustain_level.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |inner| {
							Some((inner + y_delta).clamp(0.0, 1.0))
						});
						&self.decay_time
					},
					NodeId::ReleaseEnd => {
						&self.release_time
					},
					_ => unreachable!(),
				}.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |inner| {
					Some((inner + time_delta).max(0.0))
				});
			}

			info.last_pos = state.current_mouse_pos;
		}

		let (points, _, _) = self.get_circle_positions(
			usable_width, 
			usable_height, 
			false
		);
		for (point, start_node_id) in points {
			if state.current_mouse_pos.distance(Point::new(point.x + bounds.x, point.y + bounds.y)) < 10.0 && state.dragging.is_none() {
				state.hovering.entry(start_node_id).or_default().in_if_out();
			}else {
				state.hovering.entry(start_node_id).or_default().out_if_in();
			}
		}

		let action = if state.hovering.values().any(|inner| inner.is_animating()) || state.dragging_animator.is_animating() {
			Some(Action::request_redraw())
		}else {
			None
		};

		let iced::Event::Mouse(mouse_event) = event else {
			return action;
		};

		match mouse_event {
			Event::CursorMoved { position } => {
				state.current_mouse_pos = *position;
			},
			Event::ButtonPressed(_) if state.dragging.is_none() => {
				let (points, total_time, _) = self.get_circle_positions(
					usable_width, 
					usable_height, 
					false
				);
				for (point, start_node_id) in points {
					if state.current_mouse_pos.distance(Point::new(point.x + bounds.x, point.y + bounds.y)) < 10.0 {
						state.dragging = Some(DraggingInfo { 
							start_node_id, 
							total_time, 
							last_pos: state.current_mouse_pos, 
						});
						state.dragging_animator.in_if_out();
						state.last_drag_id = Some(start_node_id);
						break;
					}
				}
			},
			Event::ButtonReleased(_) => {
				state.dragging = None;
				state.dragging_animator.out_if_in();
			},
			_ => {}
		}

		action
	}

	fn draw(
		&self,
		state: &Self::State,
		renderer: &Renderer,
		theme: &Theme,
		bounds: iced::Rectangle,
		_cursor: Cursor,
	) -> Vec<iced::widget::canvas::Geometry<Renderer>> {
		let mut frame = Frame::new(renderer, bounds.size());
		let color = theme.extended_palette().primary.base.color;

		card(theme, &mut frame);

		let usable_width = bounds.width - 2.0 * PADDING;
		let usable_height = bounds.height - 2.0 * PADDING;

		let (points, total_time, path) = self.get_circle_positions(usable_width, usable_height, true);
		let path = path.unwrap();

		let step = usable_width / total_time * 200.0;
		let lines = (usable_width / step).ceil() as usize;

		let lines = Path::new(|builder| {
			for i in 1..lines {
				let x = PADDING + i as f32 * step;
				builder.move_to(Point::new(x, PADDING));
				builder.line_to(Point::new(x, usable_height + PADDING));
			}
		});

		frame.stroke(&lines, Stroke::default().with_color(theme.extended_palette().background.strongest.color).with_width(2.0));

		frame.stroke(&path, Stroke::default().with_color(color).with_width(1.0));
		frame.fill(&path, color.scale_alpha(ALPHA_FACTOR));

		let node_circles = Path::new(|builder| {
			const RADIUS: f32 = 10.0;

			for (point, node_id) in points {
				let factor = if node_id.is_bend() { 0.5 } else { 1.0 };
				builder.circle(point, RADIUS * factor);
				let factor = if node_id.is_bend() { 0.7 } else { 1.0 };

				let animator = state.hovering.get(&node_id).map(|inner| inner.calc()).unwrap_or_default();

				if let Some(info) = &state.last_drag_id && *info == node_id {
					let inter = 1.75 * animator + (1.5 - 1.75 * animator) * state.dragging_animator.calc();  

					builder.circle(point, RADIUS * inter * factor);
				}else {
					builder.circle(point, RADIUS * factor * 1.75 * animator);
				}
			}
		});

		frame.stroke(&node_circles, Stroke::default().with_color(color).with_width(1.0));
		frame.fill(&node_circles, color.scale_alpha(ALPHA_FACTOR * ALPHA_FACTOR));

		vec![frame.into_geometry()]
	}
}