use std::time::{Duration, Instant};

use iced::{Point, Size, Theme, widget::canvas::{Frame, Path, Stroke}};

use crate::styles::{BORDER_WIDTH, PADDING};

const ANIMATION_DURATION: Duration = Duration::from_millis(150);

pub(crate) fn card(theme: &Theme, frame: &mut Frame) {
	card_border_sized(theme, frame, theme.extended_palette().background.strongest.color, frame.size())
}

pub(crate) fn card_border_sized(theme: &Theme, frame: &mut Frame, border_color: iced::Color, size: Size) {
	let rect = Path::rounded_rectangle(
		Point::new(BORDER_WIDTH, BORDER_WIDTH), 
		size.expand([- BORDER_WIDTH * 2.0, - BORDER_WIDTH * 2.0]), 
		PADDING.into()
	);

	frame.fill(&rect, theme.extended_palette().background.weakest.color);
	frame.stroke(&rect, Stroke::default().with_color(border_color).with_width(BORDER_WIDTH));
}

#[inline(always)]
pub(crate) fn bend(t: f32, bend: f32) -> f32 {
	if bend == 0.0 {
		t
	}else {
		((t * bend).exp() - 1.0) / (bend.exp() - 1.0)
	}
}

#[inline(always)]
pub(crate) fn animation(t: f32) -> f32 {
	let total_time = ANIMATION_DURATION.as_secs_f32();
	fn calc(t: f32) -> f32 {
		const BEND: f32 = 3.5;

		if t <= 0.0 {
			0.0
		}else if t >= 1.0 {
			1.0
		}else {
			bend(t, -BEND) 
		}
	}
	
	calc(t / total_time)
}

#[derive(Debug, Clone)]
pub struct Animator {
	last_in_time: Instant,
	last_out_time: Instant,
	start: f32,
}

impl Default for Animator {
	fn default() -> Self {
		Self::new()
	}
}

impl Animator {
	#[inline(always)]
	pub fn new() -> Self {
		Self {
			last_in_time: Instant::now(),
			last_out_time: Instant::now(),
			start: 0.0,
		}
	}

	#[inline(always)]
	pub fn clear_start(&mut self) {
		self.start = 0.0;
	}

	#[inline(always)]
	pub fn in_now(&mut self) {
		let in_time = self.last_in_time.elapsed().as_secs_f32();
		let out_time = self.last_out_time.elapsed().as_secs_f32();
		let total_time = ANIMATION_DURATION.as_secs_f32();
		self.start = (out_time - in_time).clamp(0.0, total_time);
		self.last_in_time = Instant::now();
	}

	#[inline(always)]
	pub fn out_now(&mut self) {
		let in_time = self.last_out_time.elapsed().as_secs_f32();
		let total_time = ANIMATION_DURATION.as_secs_f32();
		self.start = (total_time - in_time).clamp(0.0, total_time);
		self.last_out_time = Instant::now();
	}

	#[inline(always)]
	pub fn in_if_out(&mut self) {
		if !self.is_hovering() {
			self.in_now();
		}
	}

	#[inline(always)]
	pub fn out_if_in(&mut self) {
		if self.is_hovering() {
			self.out_now();
		}
	}
	
	#[inline(always)]
	pub fn is_hovering(&self) -> bool {
		self.last_in_time > self.last_out_time
	}

	#[inline(always)]
	pub fn calc(&self) -> f32 {
		if self.is_hovering() {
			let t = self.last_in_time.elapsed().as_secs_f32();
			animation(t + self.start)
		}else {
			let total_time = ANIMATION_DURATION.as_secs_f32();
			let last_time = (self.last_out_time - self.last_in_time).as_secs_f32().min(total_time);
			let t = self.last_out_time.elapsed().as_secs_f32() + self.start;
			1.0 - animation(total_time - (last_time - t))
		}
	}

	#[inline(always)]
	pub fn is_animating(&self) -> bool {
		if self.is_hovering() {
			let t = self.last_in_time.elapsed().as_secs_f32();
			t + self.start < ANIMATION_DURATION.as_secs_f32()
		}else {
			let total_time = ANIMATION_DURATION.as_secs_f32();
			let last_time = (self.last_out_time - self.last_in_time).as_secs_f32().min(total_time);
			let t = self.last_out_time.elapsed().as_secs_f32() + self.start;
			total_time - (last_time - t) < ANIMATION_DURATION.as_secs_f32()
		}
	}
}