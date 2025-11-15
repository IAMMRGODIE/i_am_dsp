//! A simple stereo generator that can be used in oscillator musics.
//! 
//! Only supports two channels for now.

const EPSILON_LENGTH: f32 = 0.01;
const DEFAULT_CACHE_LENGTH: usize = 1024;

use i_am_parameters_derive::Parameters;

use crate::prelude::{Oscillator, WaveTable};

/// A simple stereo generator that can be used in oscillator musics.
#[derive(Parameters, Clone)]
pub struct StereoGenerator {
	#[serde]
	/// start value of the generator.
	pub start: (f32, f32),
	#[serde]
	/// graph of the generator.
	pub graph: Vec<Node>,

	#[serde]
	cache_l: Vec<f32>,
	#[serde]
	cache_r: Vec<f32>,

	#[cfg(feature = "real_time_demo")]
	#[skip]
	dragging: Option<(usize, usize)>,
}

impl Default for StereoGenerator {
	fn default() -> Self {
		Self::new()
	}
}

impl StereoGenerator {
	/// Creates a new `StereoGenerator` with default values.
	pub fn new() -> Self {
		Self {
			start: (0.0, 0.0),
			graph: vec![],
			cache_l: vec![],
			cache_r: vec![],
			dragging: None,
		}
	}

	/// Updates the cache of the generator with the cache length.
	/// 
	/// Every time the data inside the generator changes, the cache needs to be updated.
	/// 
	/// Otherwise the generator will not produce the correct output.
	pub fn update_cache(&mut self, cache_length: usize) {
		if self.graph.is_empty() {
			self.cache_l = vec![self.start.0; cache_length];
			self.cache_r = vec![self.start.1; cache_length];
			return;
		}

		if cache_length == 1 {
			self.cache_l = vec![self.graph[0].to.0];
			self.cache_r = vec![self.graph[0].to.1];
			return;
		}

		self.cache_l = vec![0.0; cache_length];
		self.cache_r = vec![0.0; cache_length];

		let mut lengths = vec![];
		let mut last_node = self.start;

		for node in &self.graph {
			let len = calc_bezier_len(1.0, last_node, node.ctrl1, node.ctrl2, node.to);
			lengths.push(len);
			last_node = node.to;
		}

		let total_len: f32 = lengths.iter().sum();
		let mut current_len = 0.0;
		let step = total_len / (cache_length as f32 - 1.0);
		let mut current_node = 0;
		let mut last_node = self.start;
		for i in 0..cache_length {
			let node = &self.graph[current_node];
			let t = current_len / lengths[current_node];
			let value = arc_length_bezier(t, last_node, node.ctrl1, node.ctrl2, node.to);
			self.cache_l[i] = value.0;
			self.cache_r[i] = value.1;
			
			current_len += step;
			if current_len > lengths[current_node] {
				current_len -= lengths[current_node];
				current_node += 1;
				last_node = node.to;
				if current_node >= self.graph.len() {
					break;
				}
			}
		}
	}
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
/// A Bezier curve node.
pub struct Node {
	/// the first control point of the node.
	pub ctrl1: (f32, f32),
	/// the second control point of the node.
	pub ctrl2: (f32, f32),
	/// the end point of the node.
	pub to: (f32, f32),
}

fn bezier(t: f32, p0: (f32, f32), p1: (f32, f32), p2: (f32, f32), p3: (f32, f32)) -> (f32, f32) {
	let u = 1.0 - t;
	let u2 = u * u;
	let u3 = u2 * u;
	let t2 = t * t;
	let t3 = t2 * t;
	
	let x = u3 * p0.0 + 3.0 * u2 * t * p1.0 + 3.0 * u * t2 * p2.0 + t3 * p3.0;
	let y = u3 * p0.1 + 3.0 * u2 * t * p1.1 + 3.0 * u * t2 * p2.1 + t3 * p3.1;
	
	(x, y)
}

fn bezier_derivative(t: f32, p0: (f32, f32), p1: (f32, f32), p2: (f32, f32), p3: (f32, f32)) -> (f32, f32) {
	let u = 1.0 - t;
	let u2 = u * u;
	let t2 = t * t;
	
	let dx = 3.0 * u2 * (p1.0 - p0.0) + 6.0 * u * t * (p2.0 - p1.0) + 3.0 * t2 * (p3.0 - p2.0);
	let dy = 3.0 * u2 * (p1.1 - p0.1) + 6.0 * u * t * (p2.1 - p1.1) + 3.0 * t2 * (p3.1 - p2.1);
	
	(dx, dy)
}

fn speed(t: f32, p0: (f32, f32), p1: (f32, f32), p2: (f32, f32), p3: (f32, f32)) -> f32 {
	let (dx, dy) = bezier_derivative(t, p0, p1, p2, p3);
	(dx * dx + dy * dy).sqrt()
}

fn simpson_3_8_rule<F: Fn(f32) -> f32>(f: F, a: f32, b: f32) -> f32 {
	let delta = (b - a) / 8.0;
	let param_1 = f(a);
	let param_2 = 3.0 * f(a * 2.0 / 3.0 + b / 3.0);
	let param_3 = 3.0 * f(a + b * 2.0 / 3.0);
	let param_4 = f(b);
	delta * (param_1 + param_2 + param_3 + param_4)
}

fn calc_bezier_len(
	t: f32,
	p0: (f32, f32),
	p1: (f32, f32),
	p2: (f32, f32),
	p3: (f32, f32),
) -> f32 {
	// let lerped_p1 = (
	// 	p1.0 * (p3.0 - p0.0) + p0.0,
	// 	p1.1 * (p3.1 - p0.1) + p0.1
	// );
	// let lerped_p2 = (
	// 	p2.0 * (p3.0 - p0.0) + p0.0,
	// 	p2.1 * (p3.1 - p0.1) + p0.1
	// );

	simpson_3_8_rule(
		|t| {
			speed(t, p0, p1, p2, p3)
		},
		0.0,
		t
	)
}

fn arc_length_bezier(
	uniformed_s: f32,
	p0: (f32, f32),
	p1: (f32, f32),
	p2: (f32, f32),
	p3: (f32, f32)
) -> (f32, f32) {
	if uniformed_s <= 0.0 {
		return p0;
	}else if uniformed_s >= 1.0 {
		return p3;
	}

	let total_length = calc_bezier_len(1.0, p0, p1, p2, p3);
	let target_length = uniformed_s * total_length;

	let mut low = 0.0;
	let mut high = 1.0;
	let mut t = 0.5;

	while high - low > EPSILON_LENGTH {
		let mid = (low + high) / 2.0;
		let current_length = calc_bezier_len(mid, p0, p1, p2, p3);
		
		if current_length < target_length {
			low = mid;
		}else {
			high = mid;
		}
		t = mid;
	}

	// let lerped_p1 = (
	// 	p1.0 * (p3.0 - p0.0) + p0.0,
	// 	p1.1 * (p3.1 - p0.1) + p0.1
	// );
	// let lerped_p2 = (
	// 	p2.0 * (p3.0 - p0.0) + p0.0,
	// 	p2.1 * (p3.1 - p0.1) + p0.1
	// );

	bezier(t, p0, p1, p2, p3)
}

impl Oscillator<2> for StereoGenerator {
	fn play_at(&mut self, frequency: f32, time: f32, phase: [f32; 2]) -> [f32; 2] {
		let l_time = (frequency * time + phase[0]) % 1.0;
		let r_time = (frequency * time + phase[1]) % 1.0;

		let l_value = self.cache_l.sample(l_time, 0);
		let r_value = self.cache_r.sample(r_time, 0);

		[l_value, r_value]
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		use egui::epaint::CubicBezierShape;
		use egui::*;

		Resize::default().id_salt(format!("{}_stereo_generator_resize", id_prefix)).show(ui, |ui| {
			Frame::canvas(ui.style()).show(ui, |ui| {
				const RADIUS: f32 = 10.0;
				const STROKE: f32 = 3.0;

				let size = ui.available_size();
				let response = ui.allocate_response(size, Sense::click_and_drag());
				let rect = response.rect;
				let to_screen = emath::RectTransform::from_to(
					Rect::from_x_y_ranges(-1.0..=1.0, 1.0..=-1.0), 
					rect
				);

				let dragging = response.is_pointer_button_down_on();
				let pointer_position = ui.input(|input| {
					input.pointer.latest_pos()
				});

				let mut paint = vec![];
				let mut last_node = Pos2::new(self.start.0, self.start.1);
				paint.push(Shape::circle_stroke(to_screen * last_node, RADIUS, (STROKE, Color32::WHITE)));
				if dragging && let Some(pos) = pointer_position
					&& pos.distance(to_screen * last_node) <= RADIUS * 2.0 && self.dragging.is_none() {
						self.dragging = Some((0, 2));
					}
				
				for (i, node) in self.graph.iter().enumerate() {
					let to = Pos2::new(node.to.0, node.to.1);
					let ctrl1 = to_screen * Pos2::new(
						node.ctrl1.0, 
						node.ctrl1.1
					);
					let ctrl2 = to_screen * Pos2::new(
						node.ctrl2.0, 
						node.ctrl2.1
					);

					paint.push(Shape::line_segment([
						to_screen * last_node,
						ctrl1
					], (STROKE, Color32::WHITE)));
					paint.push(Shape::line_segment([
						ctrl2,
						to_screen * to
					], (STROKE, Color32::WHITE)));
					paint.push(Shape::circle_stroke(ctrl1, RADIUS, (STROKE, Color32::WHITE)));
					paint.push(Shape::circle_stroke(ctrl2, RADIUS, (STROKE, Color32::WHITE)));
					paint.push(Shape::circle_stroke(to_screen * to, RADIUS, (STROKE, Color32::WHITE)));
					paint.push(Shape::CubicBezier(CubicBezierShape {
						points: [to_screen * last_node, ctrl1, ctrl2, to_screen * to],
						closed: false,
						fill: Color32::TRANSPARENT,
						stroke: (3.0, Color32::WHITE).into(),
					}));

					if dragging && let Some(pos) = pointer_position {
						if pos.distance(ctrl1) <= RADIUS * 2.0 && self.dragging.is_none() {
							self.dragging = Some((i + 1, 0));
						}else if pos.distance(ctrl2) <= RADIUS * 2.0 && self.dragging.is_none() {
							self.dragging = Some((i + 1, 1));
						}else if pos.distance(to_screen * to) <= RADIUS * 2.0 && self.dragging.is_none() {
							self.dragging = Some((i + 1, 2));
						}
					}
					last_node = to;
				}


				ui.painter_at(rect).extend(paint);

				let Some(pointer_position) = pointer_position else { return };
		
				if let Some((i, j)) = self.dragging {
					let x = (pointer_position.x - rect.center().x) / rect.width() * 2.0;
					let y = (pointer_position.y - rect.center().y) / rect.height() * 2.0;
					let x = x.clamp(-1.0, 1.0);
					let y = - y.clamp(-1.0, 1.0);
		
					if i == 0 {
						self.start.0 = x;
						self.start.1 = y;
					}else {
						let i = i - 1;
						// let last_node = if i == 0 { self.start } else { self.graph[i - 1].to };
						let node = &mut self.graph[i];
						match j {
							0 => {
								// let t_x = (x - last_node.0) / (node.to.0 - last_node.0);
								// let t_y = (y - last_node.1) / (node.to.1 - last_node.1);
								// let t_x = t_x.clamp(0.0, 1.0);
								// let t_y = t_y.clamp(0.0, 1.0);
								node.ctrl1 = (x, y);
							},
							1 => {
								// let t_x = (x - last_node.0) / (node.to.0 - last_node.0);
								// let t_y = (y - last_node.1) / (node.to.1 - last_node.1);
								// let t_x = t_x.clamp(0.0, 1.0);
								// let t_y = t_y.clamp(0.0, 1.0);
								node.ctrl2 = (x, y);
							},
							2 => node.to = (x, y),
							_ => {}
						}
					}
					self.update_cache(if self.cache_l.is_empty() { DEFAULT_CACHE_LENGTH } else { self.cache_l.len() });
				}

				if response.double_clicked() {
					let x = (pointer_position.x - rect.center().x) / rect.width() * 2.0;
					let y = (pointer_position.y - rect.center().y) / rect.height() * 2.0;
					let x = x.clamp(-1.0, 1.0);
					let y = - y.clamp(-1.0, 1.0);
					let last_node = if self.graph.is_empty() { self.start } else { self.graph[self.graph.len() - 1].to };
					self.graph.push(Node {
						ctrl1: (0.5 * (x + last_node.0), last_node.1),
						ctrl2: (0.5 * (x + last_node.0), y),
						to: (x, y),
					});
					self.update_cache(if self.cache_l.is_empty() { DEFAULT_CACHE_LENGTH } else { self.cache_l.len() });
				}

				if response.clicked_by(PointerButton::Secondary) && let Some((i, _)) = &self.dragging && *i != 0 {
					self.graph.remove(i - 1);
					self.update_cache(if self.cache_l.is_empty() { DEFAULT_CACHE_LENGTH } else { self.cache_l.len() });
				}

				if !dragging {
					self.dragging = None;
				}
			});
		});

	}
}

/// A collection of multiple stereo generators.
#[derive(Parameters)]
pub struct MultiStereoGenerators {
	/// The collection of stereo generators.
	#[sub_param]
	pub generators: Vec<StereoGenerator>,
	/// The smooth factor for the interpolation.
	#[range(min = 0.0, max = 1.0)]
	pub smooth_factor: f32,

	#[cfg(feature = "real_time_demo")]
	current_generator: usize,
}

impl Default for MultiStereoGenerators {
	fn default() -> Self {
		Self::new()
	}
}

impl MultiStereoGenerators {
	/// Creates a new multi stereo generator.
	pub fn new() -> Self {
		Self {
			generators: vec![],
			smooth_factor: 0.0,
			#[cfg(feature = "real_time_demo")]
			current_generator: 0,
		}
	}
}

impl Oscillator<2> for MultiStereoGenerators {
	fn play_at(&mut self, frequency: f32, time: f32, phase: [f32; 2]) -> [f32; 2] {
		if self.generators.is_empty() {
			return [0.0; 2];
		}

		let current_page = self.smooth_factor * (self.generators.len() - 1) as f32;
		let t = current_page.fract();
		let current_page = current_page.floor() as usize;
		
		if current_page == self.generators.len() - 1 || t == 0.0 {
			return self.generators[current_page].play_at(frequency, time, phase);
		}

		let next_page = current_page + 1;
		let prev = self.generators[current_page].play_at(frequency, time, phase);
		let next = self.generators[next_page].play_at(frequency, time, phase);

		[
			prev[0] * (1.0 - t) + next[0] * t,
			prev[1] * (1.0 - t) + next[1] * t,
		]
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		use egui::*;

		if self.generators.is_empty() {
			if ui.button("Add Page").clicked() {
				self.generators.push(StereoGenerator::default());
			}
			return;
		}

		if let Some(generator) = self.generators.get_mut(self.current_generator) {
			generator.demo_ui(ui, format!("{}_{}", id_prefix, self.current_generator));
		}else {
			self.current_generator = 0;
		}

		ui.horizontal(|ui| {
			ui.add(Slider::new(&mut self.smooth_factor, 0.0..=1.0).text("Smooth Factor"));
			ui.menu_button(format!("Current Page {}", self.current_generator), |ui| {
				for i in 0..self.generators.len() {
					if ui.button(format!("Page {}", i)).clicked() {
						self.current_generator = i;
						if !self.generators.is_empty() {
							self.smooth_factor = i as f32 / (self.generators.len() - 1) as f32;
						}
						ui.close_menu();
					}
				}

				if ui.button("Add Page").clicked() {
					self.generators.push(StereoGenerator::default());
					ui.close_menu();
				}
			});
			if ui.button("Remove Page").clicked() {
				self.generators.remove(self.current_generator);
				self.current_generator = self.current_generator.saturating_sub(1);
			}
			if ui.button("Reset").clicked() {
				self.generators[self.current_generator] = StereoGenerator::default();
			}
			if ui.button("Clone").clicked() {
				self.generators.push(self.generators[self.current_generator].clone());
			}
		});
	}
}