//! The view effects such as spectrum analyzer, waveform display, etc.

use std::sync::{Arc, atomic::Ordering};

use i_am_dsp::{prelude::WaveTable, tools::ring_buffer::RingBuffer};
use iced::{Point, Rectangle, Renderer, Size, Theme, mouse::{self, Event}, theme::palette::lighten, widget::canvas::{Path, Program, Stroke, path::Builder}};
use portable_atomic::{AtomicBool, AtomicF32};

use crate::{styles::{ALPHA_FACTOR, BRIGHT_FACTOR, PADDING}, tools::utils::{Animator, card, card_border_sized}};

/// A waveform displayer based on buffer, will automatically draw waveform based on the bound size
pub struct WaveformBuf {
	cache: RingBuffer<f32>,
	width: AtomicF32,
	is_changed: AtomicBool,

	update_count: usize,
	current_max: f32,
	current_min: f32,

	max_cache: RingBuffer<f32>,
	min_cache: RingBuffer<f32>,

	canvas_cache: iced::widget::canvas::Cache,

	// _phantom_data: std::marker::PhantomData<Message>,
}

impl<Message> Program<Message> for WaveformBuf {
	type State = ();

	fn draw(
		&self,
		_state: &Self::State,
		renderer: &Renderer,
		theme: &Theme,
		bounds: Rectangle,
		_cursor: mouse::Cursor,
	) -> Vec<iced::widget::canvas::Geometry<Renderer>> {
		let mem_width = self.width.load(Ordering::Relaxed);
		let color = theme.palette().primary;

		if mem_width != bounds.width {
			self.width.store(bounds.width, Ordering::Relaxed);
			self.is_changed.store(true, Ordering::Relaxed);
		}

		let geometry = self.canvas_cache.draw(renderer, bounds.size(), |frame| {
			let Size { width, height } = frame.size();

			card(theme, frame);

			let usable_width = width - PADDING * 2.0;
			let usable_height = height - PADDING * 2.0;

			let cache_buffer_len = self.cache.capacity();
			if cache_buffer_len == 0 {
				return;
			}

			let sample_per_px = cache_buffer_len as f32 / width;

			fn draw_point(i: usize, sample: f32, step: f32, height: f32, builder: &mut Builder) {
				let x = i as f32 * step;
				let y = (- sample.clamp(-1.0, 1.0) + 1.0) / 2.0 * height;

				let p = Point::new(x + PADDING, y + PADDING);

				builder.line_to(p);
			}

			let path = if sample_per_px <= 2.5 && cache_buffer_len <= 2048 {
				let step = usable_width / cache_buffer_len as f32;

				Path::new(|builder| {
					builder.move_to(Point::new(PADDING, usable_height / 2.0 + PADDING));
					for (i, sample) in self.cache.iter().enumerate() {
						draw_point(i, *sample, step, usable_height, builder);
					}
					builder.line_to(Point::new(width - PADDING, usable_height / 2.0 + PADDING));
				})
			}else {
				let step = usable_width / self.min_cache.capacity() as f32;
				
				Path::new(|builder| {
					builder.move_to(Point::new(PADDING, usable_height / 2.0 + PADDING));
					for (i, sample) in self.max_cache.iter().enumerate() {
						draw_point(i, *sample, step, usable_height, builder);
					}

					for (i, sample) in self.min_cache.iter().enumerate().rev() {
						draw_point(i, *sample, step, usable_height, builder);
					}

					builder.close();
				})
			};
			frame.stroke(&path, Stroke::default().with_color(color).with_width(1.0));
			frame.fill(&path, color.scale_alpha(ALPHA_FACTOR));
		});

		vec![geometry]
	}
}

impl WaveformBuf {
	pub fn cache_capacity(&self) -> usize {
		self.cache.capacity()
	}

	pub fn new(cache_size: usize, target_width: f32) -> Self {
		let cache = RingBuffer::new(cache_size);
		let max_cache = RingBuffer::new(target_width.ceil() as usize);
		let min_cache = RingBuffer::new(target_width.ceil() as usize);

		Self {
			cache,
			width: AtomicF32::new(target_width),
			is_changed: AtomicBool::new(false),

			update_count: 0,
			current_max: f32::NEG_INFINITY,
			current_min: f32::INFINITY,

			max_cache,
			min_cache,

			canvas_cache: iced::widget::canvas::Cache::default(),
		}
	}

	pub fn resize_cache(&mut self, new_size: usize) {
		self.cache.resize(new_size);
		self.cache.clear();
		self.max_cache.clear();
		self.min_cache.clear();
	}

	pub fn update(&mut self, samples: &[f32]) {
		let sample = samples.iter().sum::<f32>() / samples.len() as f32;

		self.cache.push(sample);
		let width = self.width.load(Ordering::Relaxed);

		if self.is_changed.load(Ordering::Relaxed) {
			self.max_cache.resize(width.ceil() as usize);
			self.min_cache.resize(width.ceil() as usize);
			self.is_changed.store(false, Ordering::Relaxed);
		}

		self.current_min = self.current_min.min(sample);
		self.current_max = self.current_max.max(sample);

		self.update_count += 1;

		let sample_per_px = self.cache.capacity() as f32 / width;

		if self.update_count >= sample_per_px.round() as usize {
			self.update_count = 0;

			self.max_cache.push(self.current_max);
			self.min_cache.push(self.current_min);

			self.current_max = f32::NEG_INFINITY;
			self.current_min = f32::INFINITY;
			self.canvas_cache.clear();
		}

		if sample_per_px <= 2.5 && self.cache.capacity() <= 2048 {
			self.canvas_cache.clear();
		}
	}
}

// #[derive(Clone)]
pub struct Waveform<Table: WaveTable> {
	pub table: Table,
	selected: AtomicBool,
	sample_points: usize,
	canvas_cache: Arc<iced::widget::canvas::Cache>,
	on_change: Option<Box<dyn Fn(bool) -> bool + Send + Sync + 'static>>,
	disable_hover: bool,
	color: Option<iced::Color>,
}

#[derive(Default)]
pub struct WaveformState {
	mouse_pos: Point,
	last_selected: bool,
	hover_animator: Animator,
	click_animator: Animator,
}

impl<Table: WaveTable, Message> Program<Message> for Waveform<Table> {
	type State = WaveformState;

	fn draw(
		&self,
		state: &Self::State,
		renderer: &Renderer,
		theme: &Theme,
		bounds: Rectangle,
		_cursor: mouse::Cursor,
	) -> Vec<iced::widget::canvas::Geometry<Renderer>> {
		if state.click_animator.is_animating() || state.hover_animator.is_animating() {
			self.canvas_cache.clear();
		}

		let geometry = self.canvas_cache.draw(renderer, bounds.size(), |frame| {
			let Size { width, height } = frame.size();
			let bright_factor = state.hover_animator.calc() * BRIGHT_FACTOR * if self.disable_hover { 0.0 } else { 1.0 };
			let color = self.color.unwrap_or(theme.palette().primary);

			let click_animator = state.click_animator.calc();

			let border_color = iced::theme::palette::mix(
				theme.extended_palette().background.strongest.color,
				theme.extended_palette().primary.base.color,
				click_animator
			);

			card_border_sized(theme, frame, lighten(border_color, bright_factor), frame.size());

			let usable_width = width - PADDING * 2.0;
			let usable_height = height - PADDING * 2.0;

			if self.sample_points <= 1 {
				return;
			}

			let step = usable_width / self.sample_points as f32;

			let path = Path::new(|builder| {
				builder.move_to(Point::new(PADDING, usable_height / 2.0 + PADDING));
				for i in 0..self.sample_points {
					let t = i as f32 / self.sample_points as f32;
					let sample = self.table.sample(t, 0).clamp(-1.0, 1.0);

					let x = i as f32 * step + PADDING;
					let y = (- sample + 1.0) / 2.0 * usable_height + PADDING;

					let p = Point::new(x, y);

					builder.line_to(p);
				}
				builder.line_to(Point::new(width - PADDING, usable_height / 2.0 + PADDING));
			});

			frame.stroke(&path, Stroke::default().with_color(color).with_width(1.0));
			frame.fill(&path, color.scale_alpha(ALPHA_FACTOR));
		});

		vec![geometry]
	}

	fn update(
		&self,
		state: &mut Self::State,
		event: &iced::Event,
		bounds: Rectangle,
		_: iced::advanced::mouse::Cursor,
	) -> Option<iced::widget::Action<Message>> {
		if bounds.contains(state.mouse_pos) {
			state.hover_animator.in_if_out();
		}else {
			state.hover_animator.out_if_in();
		}

		let selected = self.selected.load(Ordering::Relaxed);

		if !state.click_animator.is_animating() && state.last_selected != selected {
			state.last_selected = selected;
			if selected {
				state.click_animator.in_if_out();
			}else {
				state.click_animator.out_if_in();
			}
		}

		let iced::Event::Mouse(mouse_event) = event else {
			return None;
		};

		match mouse_event {
			Event::CursorMoved { position } => {
				state.mouse_pos = *position;
			},
			Event::ButtonPressed(_) if bounds.contains(state.mouse_pos) => {
				if if let Some(f) = &self.on_change {
					f(!selected)
				}else {
					true
				} {
					self.selected.store(!selected, Ordering::Relaxed);
				}
			}
			_ => {}
		}
		
		None
	}
}

impl<Table: WaveTable> Waveform<Table> {
	pub fn new(table: Table, sample_points: usize, selected: bool) -> Self {
		Self {
			table,
			sample_points,
			canvas_cache: Default::default(),
			selected: AtomicBool::new(selected),
			on_change: None,
			disable_hover: false,
			color: None,
		}
	}

	pub fn set_selected(&self, selected: bool) {
		self.selected.store(selected, Ordering::Relaxed);
		self.canvas_cache.clear();
	}

	pub fn on_change<F: 'static + Fn(bool) -> bool + Send + Sync>(self, f: F) -> Self {
		Self {
			on_change: Some(Box::new(f)),
			..self
		}
	}

	pub fn toggle_update(&mut self) {
		self.canvas_cache.clear();
	}

	pub fn change_table(&mut self, table: Table) {
		self.table = table;
		self.canvas_cache.clear();
	}

	pub fn get_table(&self) -> &Table {
		&self.table
	}

	pub fn set_sample_points(&mut self, sample_points: usize) {
		self.sample_points = sample_points;
		self.canvas_cache.clear();
	}

	pub fn get_sample_points(&self) -> usize {
		self.sample_points
	}

	pub fn disable_hover(self) -> Self {
		Self {
			disable_hover: true,
			..self
		}
	}

	pub fn color(self, color: impl Into<iced::Color>) -> Self {
		Self {
			color: Some(color.into()),
			..self
		}
	}
}