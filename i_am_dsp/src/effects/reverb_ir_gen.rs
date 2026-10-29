//! Implmentation of the reverb impulse response generator.

use std::f32::{self, consts::PI};
use i_am_dsp_derive::Parameters;

use crate::{prelude::{FftConvolver, Parameters}, Effect};

/// A trait for generating reverb impulse response.
pub trait GenIr: Parameters {
	/// Generate the reverb impulse response.
	/// 
	/// Maximum length of the impulse response, in samples, none for unlimited length.
	fn gen_ir(&self, maxium_ir_len: Option<usize>) -> Vec<f32>;

	/// Generate a stereo impulse response. The default keeps the mono response
	/// in both channels; the ray tracer overrides it with positional panning.
	fn gen_stereo_ir(&self, maxium_ir_len: Option<usize>) -> [Vec<f32>; 2] {
		let mono = self.gen_ir(maxium_ir_len);
		[mono.clone(), mono]
	}

	/// Simple UI for the reverb impulse response generator.
	/// 
	/// Returns `true` if we need regenerate the impulse response.
	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) -> bool;
}

/// A Face shape for generating reverb impulse response.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Face {
	/// Shape of the room.
	pub tri: Triangle,
	/// The decay factor of the room.
	/// 
	/// 0.0 means no decay, 1.0 means complete decay.
	pub decay_factor: f32,
	/// Roughness factor of the room.
	/// 
	/// 0.0 means smooth surface, which use accurate reflections, 
	/// 1.0 means rough surface, which use completely random reflections.
	pub roughness_factor: f32,
}

/// Ray tracing algorithm for generating reverb impulse response.
#[derive(Parameters)]
pub struct RayTracing {
	/// Speed of sound, in meters per second.
	/// 
	/// Can be used to simulate reflections from different materials.
	#[range(min = 1.0, max = 20000.0)]
	#[logarithmic]
	pub sound_velocity: f32,
	/// Sample rate of the audio signal, in Hz.
	#[serde]
	pub sample_rate: usize,
	#[serde]
	/// Shape of the room.
	pub shape: Vec<Face>,
	/// Position of the sound source.
	#[serde]
	pub sound_source: (f32, f32, f32),
	/// Position of the listener.
	#[serde]
	pub listener: (f32, f32, f32),
	#[serde]
	/// Distance at which the sound is hearable.
	pub hearable_distance: f32,
	/// Decay factor of the medium.
	#[range(min = 0.001, max = 10.0)]
	#[logarithmic]
	pub decay_factor: f32,
}

/// One traced reflection: when it reaches the listener, how loud it is, and
/// which side of the listener it arrives from.
struct Echo {
	time: f32,
	amplitude: f32,
	/// Unit direction from the listener towards the reflection point. With the
	/// listener facing +x and +y up, +z is the right-hand side.
	direction: (f32, f32, f32),
}

/// Add a sample at a fractional sample index.
fn add_sample(output: &mut Vec<f32>, index_f32: f32, sample: f32) {
	let index = index_f32.max(0.0).floor() as usize;
	let fraction = index_f32 - index as f32;

	if output.len() < index + 2 {
		output.resize(index + 2, 0.0);
	}

	output[index] += sample * (1.0 - fraction);
	output[index + 1] += sample * fraction;
}

/// Triangle shape for generating reverb impulse response.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Triangle {
	/// Vertex 1 of the triangle.
	pub v_1: (f32, f32, f32),
	/// Vertex 2 of the triangle.
	pub v_2: (f32, f32, f32),
	/// Vertex 3 of the triangle.
	pub v_3: (f32, f32, f32),
}

#[inline(always)]
fn cross(a: (f32, f32, f32), b: (f32, f32, f32)) -> (f32, f32, f32) {
	(a.1 * b.2 - a.2 * b.1, a.2 * b.0 - a.0 * b.2, a.0 * b.1 - a.1 * b.0)
}

#[inline(always)]
fn dot(a: (f32, f32, f32), b: (f32, f32, f32)) -> f32 {
	a.0 * b.0 + a.1 * b.1 + a.2 * b.2
}

#[inline(always)]
fn len(a: (f32, f32, f32)) -> f32 {
	(a.0 * a.0 + a.1 * a.1 + a.2 * a.2).sqrt()
}

#[inline(always)]
fn normalized(a: (f32, f32, f32)) -> (f32, f32, f32) {
	let l = len(a);
	if l == 0.0 {
		(0.0, 0.0, 0.0)
	}else {
		(a.0 / l, a.1 / l, a.2 / l)
	}
}

#[inline(always)]
fn distance(a: (f32, f32, f32), b: (f32, f32, f32)) -> f32 {
	len((a.0 - b.0, a.1 - b.1, a.2 - b.2))
}

#[inline(always)]
fn distance_to_line(p: (f32, f32, f32), p0: (f32, f32, f32), d: (f32, f32, f32)) -> f32 {
	let len_d = len(d);
	if len_d == 0.0 {
		return distance(p, p0);
	}
	
	let delta = (p.0 - p0.0, p.1 - p0.1, p.2 - p0.2);
	let len = len(cross(delta, d));

	len / len_d
}

impl Triangle {
	fn normal_vector(&self) -> (f32, f32, f32) {
		let e_1 = (self.v_2.0 - self.v_1.0, self.v_2.1 - self.v_1.1, self.v_2.2 - self.v_1.2);
		let e_2 = (self.v_3.0 - self.v_1.0, self.v_3.1 - self.v_1.1, self.v_3.2 - self.v_1.2);
		let n = cross(e_1, e_2);
		let len_n = len(n);
		(n.0 / len_n, n.1 / len_n, n.2 / len_n)
	}
	
	fn mt_alg(&self, from: (f32, f32, f32), d: (f32, f32, f32)) -> Option<f32> {
		let e_1 = (self.v_2.0 - self.v_1.0, self.v_2.1 - self.v_1.1, self.v_2.2 - self.v_1.2);
		let e_2 = (self.v_3.0 - self.v_1.0, self.v_3.1 - self.v_1.1, self.v_3.2 - self.v_1.2);

		let p = cross(d, e_2);
		let det = dot(e_1, p);

		if det.abs() < 1e-6 {
			return None;
		}

		let inv_det = 1.0 / det;
		let t = (from.0 - self.v_1.0, from.1 - self.v_1.1, from.2 - self.v_1.2);
		let u = dot(t, p) * inv_det;

		if !(0.0..=1.0).contains(&u) {
			return None;
		}

		let q = cross(t, e_1);
		let v = dot(d, q) * inv_det;

		if v < 0.0 || u + v > 1.0 {
			return None;
		}

		let t = dot(e_2, q) * inv_det;

		if t <= 0.0 {
			None
		}else {
			Some(t)
		}
	}
}

#[cfg(feature = "room_gen")]
#[cfg_attr(docsrs, doc(cfg(feature = "room_gen")))]
/// A trait for generating point sources.
pub trait PointSampler {
	/// Sample a random point in the room.
	fn sample_point(&self) -> (f32, f32, f32);
}

const VELOCITY_OF_SOUND: f32 = 343.0;

/// A point generator for generating random points on a ball.
#[cfg(feature = "room_gen")]
#[cfg_attr(docsrs, doc(cfg(feature = "room_gen")))]
pub struct Ball {
	/// Radius of the ball.
	pub radius: f32,
}

#[cfg(feature = "room_gen")]
#[cfg_attr(docsrs, doc(cfg(feature = "room_gen")))]
impl PointSampler for Ball {
	fn sample_point(&self) -> (f32, f32, f32) {
		let theta = rand::random_range(0.0..=2.0 * PI);
		let phi = rand::random_range(0.0..=PI);
		(
			self.radius * phi.cos() * theta.cos(),
			self.radius * phi.cos() * theta.sin(),
			self.radius * phi.sin(),
		)
	}
}

#[cfg(feature = "room_gen")]
#[cfg_attr(docsrs, doc(cfg(feature = "room_gen")))]
impl RayTracing {
	/// Create a random **convex** room with the given number of points and decay factor.
	/// 
	/// Will place the sound source and listener at the center of the room.
	/// 
	/// Also we will set velocity of sound to 343 m/s (approximately the speed of sound in air).
	pub fn random_space(
		sampler: &impl PointSampler,
		max_points: usize,
		decay_factor: f32,
		roughness_factor: f32,
		hearable_distance: f32,
		sample_rate: usize,
	) -> Self {
		let points = (0..max_points).map(|_| sampler.sample_point()).collect::<Vec<_>>();
		Self::convex_hull_room(&points, decay_factor, roughness_factor, hearable_distance, sample_rate)
	}

	/// Create a convex hull room with the given points and decay factor.
	/// 
	/// Will place the sound source and listener at the center of the room.
	/// 
	/// Also we will set velocity of sound to 343 m/s (approximately the speed of sound in air).
	pub fn convex_hull_room(
		points: &[(f32, f32, f32)],
		decay_factor: f32,
		roughness_factor: f32,
		hearable_distance: f32,
		sample_rate: usize,
	) -> Self {
    	use parry3d::{math::Vec3, transformation::convex_hull};

		let points = points.iter().map(|p| Vec3::new(p.0, p.1, p.2)).collect::<Vec<_>>();

		let (vertex, indices) = convex_hull(&points);
		let mut tris = vec![];
		for [i, j, k] in indices {
			let i = i as usize;
			let j = j as usize;
			let k = k as usize;
			tris.push(Triangle {
				v_1: (vertex[i][0], vertex[i][1], vertex[i][2]),
				v_2: (vertex[j][0], vertex[j][1], vertex[j][2]),
				v_3: (vertex[k][0], vertex[k][1], vertex[k][2]),
			});
		}

		let mut center = (0.0, 0.0, 0.0);
		let vertex_len = vertex.len() as f32;

		for p in vertex {
			center.0 += p.x;
			center.1 += p.y;
			center.2 += p.z;
		}

		center.0 /= vertex_len;
		center.1 /= vertex_len;
		center.2 /= vertex_len;

		let sound_source = center;
		let listener = center;

		Self {
			sound_velocity: VELOCITY_OF_SOUND,
			sample_rate,
			shape: tris.into_iter().map(|t| Face { 
				tri: t, 
				decay_factor, 
				roughness_factor 
			}).collect(),
			sound_source,
			listener,
			hearable_distance,
			decay_factor,
		}
	}
}

impl RayTracing {
	/// Create a rectangular room with the given dimensions and decay factor.
	/// 
	/// Will place the sound source and listener at the center of the room.
	/// 
	/// Also we will set velocity of sound to 343 m/s (approximately the speed of sound in air).
	pub fn rectangular_room(
		width: f32, 
		height: f32, 
		depth: f32, 
		decay_factor: f32,
		hearable_distance: f32,
		sample_rate: usize,
	) -> Self {
		let sound_source = (width / 2.0, height / 2.0, depth / 2.0);
		let listener = (width / 2.0, height / 2.0, depth / 2.0);
		let v000 = (0.0, 0.0, 0.0);
		let v100 = (width, 0.0, 0.0);
		let v010 = (0.0, height, 0.0);
		let v001 = (0.0, 0.0, depth);
		let v110 = (width, height, 0.0);
		let v101 = (width, 0.0, depth);
		let v011 = (0.0, height, depth);
		let v111 = (width, height, depth);

		let tris = vec![
			Triangle { v_1: v000, v_2: v100, v_3: v110 },
			Triangle { v_1: v000, v_2: v110, v_3: v010 },
			Triangle { v_1: v101, v_2: v001, v_3: v011 },
			Triangle { v_1: v101, v_2: v011, v_3: v111 },
			Triangle { v_1: v000, v_2: v001, v_3: v011 },
			Triangle { v_1: v000, v_2: v011, v_3: v010 },
			Triangle { v_1: v100, v_2: v110, v_3: v111 },
			Triangle { v_1: v100, v_2: v111, v_3: v101 },
			Triangle { v_1: v000, v_2: v100, v_3: v101 },
			Triangle { v_1: v000, v_2: v101, v_3: v001 },
			Triangle { v_1: v010, v_2: v011, v_3: v111 },
			Triangle { v_1: v010, v_2: v111, v_3: v110 },
		];

		// The hand-written triangles above are not wound consistently: some of
		// them produce a normal that points *outwards*. Flip any face whose
		// normal points away from the room interior (the centroid) so that
		// normal_vector() always points into the room.
		let interior = (width / 2.0, height / 2.0, depth / 2.0);
		let tris = tris.into_iter().map(|tri| {
			let center = (
				(tri.v_1.0 + tri.v_2.0 + tri.v_3.0) / 3.0,
				(tri.v_1.1 + tri.v_2.1 + tri.v_3.1) / 3.0,
				(tri.v_1.2 + tri.v_2.2 + tri.v_3.2) / 3.0,
			);
			let n = tri.normal_vector();
			let inward = (interior.0 - center.0, interior.1 - center.1, interior.2 - center.2);
			if dot(n, inward) < 0.0 {
				// swap two vertices to reverse the winding (and thus the normal)
				Triangle { v_1: tri.v_1, v_2: tri.v_3, v_3: tri.v_2 }
			}else {
				tri
			}
		}).collect::<Vec<_>>();

		Self {
			sound_velocity: VELOCITY_OF_SOUND,
			sample_rate,
			shape: tris.into_iter().map(|t| Face { 
				tri: t, 
				decay_factor: 0.0, 
				roughness_factor: 0.0 
			}).collect(),
			sound_source,
			listener,
			hearable_distance,
			decay_factor,
		}
	}

	/// Generate the reverb impulse response.
	pub fn generate_ir(&self, maxium_ir_len: Option<usize>) -> Vec<f32> {
		const NUM_POINTS: usize = 10000;

		if let Some(max_ir_len) = maxium_ir_len && max_ir_len <= 1 {
			return vec![0.0; max_ir_len];
		}

		let max_ir_len = maxium_ir_len.map(|len| len as f32 / self.sample_rate as f32);

		let mut output = vec![];
		for _ in 0..NUM_POINTS {
			let theta = rand::random_range(-PI..=PI);
			let phi = rand::random_range(0.0..=PI);
			let ir = self.generate_single_point(theta, phi, 1.0, max_ir_len);
			for echo in ir {
				let index_f32 = echo.time * self.sample_rate as f32;
				let index = index_f32.floor() as usize;
				let frac = index_f32 - index as f32;
				if output.len() < index + 2 {
					output.resize(index + 2, 0.0);
				}
				output[index] += echo.amplitude * (1.0 - frac);
				output[index + 1] += echo.amplitude * frac;
			}
		}

		if output.is_empty() {
			return vec![];
		}

		let output_max = output
			.iter()
			.map(|inner| inner.abs())
			.max_by(|a, b| a.partial_cmp(b).unwrap())
			.unwrap();

		let output = output.into_iter().map(|inner| inner / output_max).collect::<Vec<_>>();

		if let Some(max_ir_len) = maxium_ir_len {
			let output = output.into_iter().take(max_ir_len).collect::<Vec<_>>();
			return output.into_iter().enumerate().map(|(i, inner)| {
				let t = i as f32 / (max_ir_len - 1) as f32;
				let window = (t * PI).cos() * 0.5 + 0.5;
				inner * window
			}).collect()
		}

		let mut trim_start = 0;
		while trim_start < output.len() && output[trim_start] <= 0.005 {
			trim_start += 1;
		}
		let mut trim_end = output.len() - 1;
		while trim_end > 0 && output[trim_end] <= 0.005 {
			trim_end -= 1;
		}
		output.into_iter().skip(trim_start).take(trim_end - trim_start + 1).collect()
	}

	/// Generate a stereo impulse response from the same reflections. Every echo
	/// is panned by the direction from the listener to its reflection point:
	/// with the listener facing +x and +y up, +z is the right-hand side. The far
	/// ear also gets a small ITD. No per-channel random gain is involved.
	pub fn generate_stereo_ir(&self, maxium_ir_len: Option<usize>) -> [Vec<f32>; 2] {
		if let Some(max_ir_len) = maxium_ir_len && max_ir_len <= 1 {
			return [vec![0.0; max_ir_len], vec![0.0; max_ir_len]];
		}

		let max_ir_len = maxium_ir_len.map(|len| len as f32 / self.sample_rate as f32);
		let right = (0.0, 0.0, 1.0);
		const MAX_ITD: f32 = 0.0006;
		let mut left = vec![];
		let mut right_channel = vec![];

		for _ in 0..10000 {
			let theta = rand::random_range(-PI..=PI);
			let phi = rand::random_range(0.0..=PI);
			for echo in self.generate_single_point(theta, phi, 1.0, max_ir_len) {
				let pan = dot(echo.direction, right).clamp(-1.0, 1.0);
				let angle = (pan + 1.0) * PI / 4.0;
				let itd = MAX_ITD * pan.abs();
				let left_time = echo.time + if pan > 0.0 { itd } else { 0.0 };
				let right_time = echo.time + if pan < 0.0 { itd } else { 0.0 };

				add_sample(
					&mut left,
					left_time * self.sample_rate as f32,
					echo.amplitude * angle.cos(),
				);
				add_sample(
					&mut right_channel,
					right_time * self.sample_rate as f32,
					echo.amplitude * angle.sin(),
				);
			}
		}

		if left.is_empty() && right_channel.is_empty() {
			return [vec![], vec![]];
		}

		let peak = left
			.iter()
			.chain(right_channel.iter())
			.map(|sample| sample.abs())
			.fold(0.0, f32::max);

		if peak > 0.0 {
			for sample in left.iter_mut().chain(right_channel.iter_mut()) {
				*sample /= peak;
			}
		}

		if let Some(max_ir_len) = maxium_ir_len {
			left.truncate(max_ir_len);
			right_channel.truncate(max_ir_len);
			let denominator = (max_ir_len - 1).max(1) as f32;

			for channel in [&mut left, &mut right_channel] {
				for (index, sample) in channel.iter_mut().enumerate() {
					let t = index as f32 / denominator;
					let window = (t * PI).cos() * 0.5 + 0.5;
					*sample *= window;
				}
			}

			return [left, right_channel];
		}

		let length = left.len().max(right_channel.len());
		let audible = |channel: &[f32], index: usize| {
			channel.get(index).is_some_and(|sample| sample.abs() > 0.005)
		};
		let start = (0..length).find(|&index| {
			audible(&left, index) || audible(&right_channel, index)
		});
		let end = (0..length).rev().find(|&index| {
			audible(&left, index) || audible(&right_channel, index)
		});
		let (Some(start), Some(end)) = (start, end) else {
			return [vec![], vec![]];
		};
		let length = end - start + 1;
		let mut left = left.into_iter().skip(start).take(length).collect::<Vec<_>>();
		let mut right_channel = right_channel
			.into_iter()
			.skip(start)
			.take(length)
			.collect::<Vec<_>>();
		left.resize(length, 0.0);
		right_channel.resize(length, 0.0);
		[left, right_channel]
	}

	fn generate_single_point(&self, theta: f32, phi: f32, amp: f32, maxium_ir_len: Option<f32>) -> Vec<Echo> {
		const MAX_T_VALUE: f32 = 50.0;
		const THRESHOLD: f32 = 0.001;

		let range = 0.0..=MAX_T_VALUE;
		let mut dir = (
			theta.cos() * phi.sin(),
			theta.sin() * phi.sin(),
			phi.cos()
		);
		
		let mut current_pos = self.sound_source;
		let mut output = vec![];
		let mut current_amp = amp;
		let mut current_phase = rand::random_range(0.0..2.0 * PI);
		let mut accumulated_distance = 0.0;

		loop {
			let mut min_t = f32::INFINITY;
			let mut normal = (0.0, 0.0, 0.0);
			let mut relative_factor = 0.0;
			let mut relative_roughness = 0.0;
			for face in &self.shape {
				if let Some(t) = face.tri.mt_alg(current_pos, dir) && t < min_t {
					min_t = t;
					normal = face.tri.normal_vector();
					relative_factor = face.decay_factor;
					relative_roughness = face.roughness_factor;
				}
			}

			// too far to be heard
			if !range.contains(&min_t) {
				break;
			}

			// The ray grazes the face (or re-hits the very surface it just
			// left): the intersection is numerically degenerate, stop instead
			// of accumulating thousands of pointless micro-reflections.
			if min_t < 1e-4 {
				break;
			}

			// The ray always travels inside a convex room, so the wall normal it
			// hits must point *towards* the interior (opposing the incoming
			// direction). Triangle winding is not guaranteed to agree across all
			// faces, so flip the computed normal when needed. This keeps the
			// diffuse-hemisphere sampling and the reflection law consistent.
			if dot(normal, dir) > 0.0 {
				normal = (-normal.0, -normal.1, -normal.2);
			}

			current_pos = (
				current_pos.0 + dir.0 * min_t,
				current_pos.1 + dir.1 * min_t,
				current_pos.2 + dir.2 * min_t
			);

			accumulated_distance += min_t;

			// Pick a random direction on the hemisphere facing the room interior
			// (towards the flipped normal).
			let random_dir = loop {
				let random_theta = rand::random_range(0.0..=2.0 * PI);
				let random_phi = rand::random_range(0.0..=PI);
				let random_dir = (
					random_theta.cos() * random_phi.sin(),
					random_theta.sin() * random_phi.sin(),
					random_phi.cos()
				);
	
				let normal_dot_dir = dot(normal, random_dir);

				if normal_dot_dir > 0.0 {
					break random_dir;
				}else if normal_dot_dir < 0.0 {
					break (
						-random_dir.0,
						-random_dir.1,
						-random_dir.2
					)
				}
			};

			// Specular reflection off the (inward) surface normal, then blend
			// towards the diffuse direction by the roughness. Normalize afterwards
			// so `dir` stays a unit vector: `mt_alg`'s t is then a real distance,
			// and accumulated_distance / time / decay stay physically correct.
			let normal_dot_dir = dot(normal, dir);
			dir = (
				dir.0 - 2.0 * normal_dot_dir * normal.0,
				dir.1 - 2.0 * normal_dot_dir * normal.1,
				dir.2 - 2.0 * normal_dot_dir * normal.2
			);

			let dir = normalized((
				dir.0 * (1.0 - relative_roughness) + random_dir.0 * relative_roughness,
				dir.1 * (1.0 - relative_roughness) + random_dir.1 * relative_roughness,
				dir.2 * (1.0 - relative_roughness) + random_dir.2 * relative_roughness
			));

			current_amp *= (1.0 - relative_factor).sqrt();
			// an experiential phase shift formula
			let phase_shift = PI * (1.0 - relative_factor).powf(0.7);
			current_phase += phase_shift;
			current_phase %= 2.0 * PI;

			let distance_to_listener = distance_to_line(self.listener, current_pos, dir);
			let sound_to_listener = distance(current_pos, self.listener);
			let decay_factor = - self.decay_factor * (accumulated_distance + sound_to_listener);
			let decayed_amp = current_amp * decay_factor.exp();

			// Only record an echo when the listener lies ahead of the reflection
			// point along the outgoing ray (distance_to_line alone cannot tell
			// whether the listener is in front of or behind the ray origin).
			let to_listener = (
				self.listener.0 - current_pos.0,
				self.listener.1 - current_pos.1,
				self.listener.2 - current_pos.2,
			);
			if distance_to_listener <= self.hearable_distance && dot(dir, to_listener) > 0.0 {
				let time = (accumulated_distance + sound_to_listener) / self.sound_velocity;
				if let Some(max_ir_len) = maxium_ir_len && time > max_ir_len {
					break;
				}
				let direction = normalized((
					current_pos.0 - self.listener.0,
					current_pos.1 - self.listener.1,
					current_pos.2 - self.listener.2,
				));
				output.push(Echo {
					time,
					amplitude: decayed_amp * current_phase.cos(),
					direction,
				});
			}

			if decayed_amp < THRESHOLD {
				break;
			}
		}

		output
	}
}

impl GenIr for RayTracing {
	fn gen_ir(&self, maxium_ir_len: Option<usize>) -> Vec<f32> {
		self.generate_ir(maxium_ir_len)
	}

	fn gen_stereo_ir(&self, maxium_ir_len: Option<usize>) -> [Vec<f32>; 2] {
		self.generate_stereo_ir(maxium_ir_len)
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, _: String) -> bool {
		use egui::Slider;

		let mut sound_velocity = self.sound_velocity;
		let mut hearable_distance = self.hearable_distance;
		let mut decay_factor = self.decay_factor;

		ui.add(Slider::new(&mut sound_velocity, 1.0..=4000.0).logarithmic(true).text("Sound velocity"));
		ui.add(Slider::new(&mut hearable_distance, 0.1..=10.0).logarithmic(true).text("Hearable distance"));
		ui.add(Slider::new(&mut decay_factor, 0.001..=10.0).logarithmic(true).text("Decay factor"));

		if sound_velocity != self.sound_velocity ||
			hearable_distance != self.hearable_distance ||
			decay_factor != self.decay_factor {
			self.sound_velocity = sound_velocity;
			self.hearable_distance = hearable_distance;
			self.decay_factor = decay_factor;
			return true;
		}

		false
	}
}

#[derive(Parameters)]
/// A reverb effect based on impulse response.
pub struct IrBasedReverb<T: GenIr, const CHANNELS: usize> {
	/// The reverb impulse response generator.
	#[sub_param]
	pub ir_gen: T,
	#[sub_param]
	/// The convolver for applying the reverb effect.
	pub convolver: FftConvolver<CHANNELS>,
}

impl<T: GenIr, const CHANNELS: usize> IrBasedReverb<T, CHANNELS> {
	/// Create a new `IrBasedReverb` with the given reverb impulse response generator.
	pub fn new(ir_gen: T, sample_rate: usize) -> Self {
		let stereo = ir_gen.gen_stereo_ir(None);
		let convolver = FftConvolver::new(
			core::array::from_fn(|channel| {
				let mut ir = stereo[channel % 2].clone();
				ir.truncate(48000);
				ir
			}),
			sample_rate
		);
		Self { 
			ir_gen, 
			convolver
		}
	}
}

impl<T: GenIr + Send + Sync, const CHANNELS: usize> Effect<CHANNELS> for IrBasedReverb<T, CHANNELS> {
	fn delay(&self) -> usize {
		self.convolver.delay()
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"Reverb (IR-based)"
	}

	fn process(
		&mut self, 
		samples: &mut [f32; CHANNELS], 
		other: &[&[f32; CHANNELS]],
		process_context: &mut Box<dyn crate::ProcessContext>,
	) {
		self.convolver.process(samples, other, process_context);	
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
    	use crate::tools::ui_tools::gain_ui;

		if self.ir_gen.demo_ui(ui, format!("{}_ir_gen", id_prefix)) {
			let stereo = self.ir_gen.gen_stereo_ir(None);
			self.convolver.replace_ir(
				core::array::from_fn(|channel| {
					let mut ir = stereo[channel % 2].clone();
					ir.truncate(48000);
					ir
				})
			);
		};
		gain_ui(ui, &mut self.convolver.dry_gain, Some("Dry gain".to_string()), false);
		gain_ui(ui, &mut self.convolver.wet_gain, Some("Wet gain".to_string()), true);
	}
}

mod tests {

	#[test]
	fn room_normals_point_inwards() {
		// 10x10x10 房间，声源/听众在 (5,5,5)。验证每个面的法向指向房间内侧
		// （法向与 面中心->内部一点 的方向点积 > 0）。
		use super::*;
		let rt = RayTracing::rectangular_room(10.0, 10.0, 10.0, 0.001, 3.0, 48000);
		let interior = (5.0, 5.0, 5.0);

		for face in &rt.shape {
			let tri = &face.tri;
			let center = (
				(tri.v_1.0 + tri.v_2.0 + tri.v_3.0) / 3.0,
				(tri.v_1.1 + tri.v_2.1 + tri.v_3.1) / 3.0,
				(tri.v_1.2 + tri.v_2.2 + tri.v_3.2) / 3.0,
			);
			let n = tri.normal_vector();
			// 从面中心指向房间内部点的向量
			let inward = (interior.0 - center.0, interior.1 - center.1, interior.2 - center.2);
			assert!(dot(n, inward) > 0.0, "face normal points outwards: n={n:?}");
		}
	}

	#[test]
	fn test_ir_has_peaks_and_early_reflection() {
		use super::*;
		let rt = RayTracing::rectangular_room(10.0, 10.0, 10.0, 0.001, 6.0, 48000);
		let ir = rt.generate_ir(Some(48000));
		// 归一化后峰值应为 1.0
		let max = ir.iter().cloned().fold(0.0f32, f32::max);
		assert!(max > 0.5, "IR peak too small: {max}");
		// 有非零样本（房间反射产生回声）
		let nonzero = ir.iter().filter(|&&x| x.abs() > 1e-4).count();
		assert!(nonzero > 10, "IR has too few nonzero taps: {nonzero}");
	}
	/// A source on the right must produce more energy in the right IR channel.
	#[test]
	fn the_stereo_ir_pans_reflections_by_side() {
		use super::*;

		let mut rt = RayTracing::rectangular_room(10.0, 10.0, 10.0, 0.001, 6.0, 48000);
		rt.listener = (5.0, 5.0, 5.0);
		rt.sound_source = (5.0, 5.0, 8.0);

		let irs = rt.generate_stereo_ir(Some(4_800));
		let energy = |ir: &[f32]| ir.iter().map(|sample| sample * sample).sum::<f32>();
		let left = energy(&irs[0]);
		let right = energy(&irs[1]);

		assert!(right > left * 1.05, "right {right} should lead left {left}");
	}

	#[test]
	fn test_ray_tracing() {
		use crate::{effects::reverb_ir_gen::RayTracing, tools::pcm_data::save_pcm_data};
		let mut rt = RayTracing::rectangular_room(
			10.0, 
			10.0, 
			10.0, 
			0.001, 
			3.0, 
			48000
		);
		rt.sound_velocity = 200.0;
		for shape in &mut rt.shape {
			shape.decay_factor = rand::random_range(0.0..=1.0);
			shape.roughness_factor = rand::random_range(0.0..=1.0);
		}
		let ir_1 = rt.generate_ir(None);
		let ir_2 = rt.generate_ir(None);
		assert!(save_pcm_data("./test_ray_tracing.wav", &[ir_1, ir_2], 48000).is_ok());
	}

	#[test]
	#[cfg(feature = "room_gen")]
	fn test_room_gen() {
    	use crate::{prelude::{Ball, RayTracing}, tools::pcm_data::save_pcm_data};

		let mut rt = RayTracing::random_space(
			&Ball { radius: 10.0 }, 
			100, 
			0.001, 
			0.0, 
			3.0, 
			48000
		);
		rt.sound_velocity = 200.0;
		for shape in &mut rt.shape {
			shape.decay_factor = rand::random_range(0.0..=1.0);
			shape.roughness_factor = rand::random_range(0.0..=1.0);
		}
		let ir_1 = rt.generate_ir(None);
		let ir_2 = rt.generate_ir(None);
		assert!(save_pcm_data("./test_room_gen.wav", &[ir_1, ir_2], 48000).is_ok());
	}
}