//! Implmentation of the reverb impulse response generator.

use i_am_dsp_derive::Parameters;
use std::f32::consts::PI;

use crate::{
	Effect,
	prelude::{FftConvolver, Parameters},
};

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

/// How many initial rays are traced. The directions are evenly spaced on the
/// sphere (Fibonacci), so a low count still covers the room without the clumps
/// of random sampling.
const TRACE_POINTS: usize = 2500;

/// One evenly spaced direction on the sphere, in the theta/phi convention the
/// tracer uses.
fn fibonacci_direction(index: usize, count: usize) -> (f32, f32) {
	let i = index as f32 + 0.5;
	let z = 1.0 - 2.0 * i / count as f32;
	let phi = z.clamp(-1.0, 1.0).acos();
	let theta = PI * (1.0 + 5.0_f32.sqrt()) * i;

	(theta, phi)
}

/// The longest half-window used for the listener passage. The ray-capture
/// tolerance (hearable_distance) is not a physical head size: using it as the
/// aperture smeared every echo over tens of milliseconds and low-passed the
/// whole impulse response. Keep only a point-like interval so the highs survive;
/// the span is still wider than one sample.
const MAX_APERTURE_SECONDS: f32 = 0.00005;

/// One traced reflection: when it reaches the listener, how loud it is, and
/// which side of the listener it arrives from.
struct Echo {
	time: f32,
	amplitude: f32,
	/// Unit direction from the listener towards the reflection point. With the
	/// listener facing +x and +y up, +z is the right-hand side.
	direction: (f32, f32, f32),
}

/// The equal-power gains for a reflection on the listener's right (+z).
fn stereo_gains(direction: (f32, f32, f32)) -> (f32, f32) {
	let pan = dot(direction, (0.0, 0.0, 1.0)).clamp(-1.0, 1.0);
	let angle = (pan + 1.0) * PI / 4.0;

	(angle.cos(), angle.sin())
}

/// A unit-energy Hann window for the time the ray spends inside the listener
/// sphere. A shorter chord gets fewer taps; a longer chord gets more.
fn aperture_window(half_time: f32, sample_rate: usize) -> Vec<f32> {
	let samples = ((2.0 * half_time * sample_rate as f32).round() as usize).clamp(1, 1024);
	let mut weights = Vec::with_capacity(samples);
	let mut energy = 0.0;

	for sample in 0..samples {
		let t = (sample as f32 + 0.5) / samples as f32;
		let weight = 0.5 - 0.5 * (t * 2.0 * PI).cos();
		energy += weight * weight;
		weights.push(weight);
	}

	if energy > 0.0 {
		let scale = 1.0 / energy.sqrt();
		for weight in weights.iter_mut() {
			*weight *= scale;
		}
	}

	weights
}

/// Remove the zero-lag Dirac from a normalised IR. The peak of the generated
/// response can land on sample zero; without this the wet convolver would pass
/// the dry signal at zero latency.
fn fade_in(ir: &mut [f32], sample_rate: usize) {
	let samples = ((0.002 * sample_rate as f32) as usize).clamp(1, 512);
	let count = samples.min(ir.len());

	for (index, sample) in ir.iter_mut().take(count).enumerate() {
		let window = 0.5 - 0.5 * (PI * index as f32 / count as f32).cos();
		*sample *= window;
	}
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
	(
		a.1 * b.2 - a.2 * b.1,
		a.2 * b.0 - a.0 * b.2,
		a.0 * b.1 - a.1 * b.0,
	)
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
	} else {
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
		let e_1 = (
			self.v_2.0 - self.v_1.0,
			self.v_2.1 - self.v_1.1,
			self.v_2.2 - self.v_1.2,
		);
		let e_2 = (
			self.v_3.0 - self.v_1.0,
			self.v_3.1 - self.v_1.1,
			self.v_3.2 - self.v_1.2,
		);
		let n = cross(e_1, e_2);
		let len_n = len(n);
		(n.0 / len_n, n.1 / len_n, n.2 / len_n)
	}

	fn mt_alg(&self, from: (f32, f32, f32), d: (f32, f32, f32)) -> Option<f32> {
		let e_1 = (
			self.v_2.0 - self.v_1.0,
			self.v_2.1 - self.v_1.1,
			self.v_2.2 - self.v_1.2,
		);
		let e_2 = (
			self.v_3.0 - self.v_1.0,
			self.v_3.1 - self.v_1.1,
			self.v_3.2 - self.v_1.2,
		);

		let p = cross(d, e_2);
		let det = dot(e_1, p);

		if det.abs() < 1e-6 {
			return None;
		}

		let inv_det = 1.0 / det;
		let t = (
			from.0 - self.v_1.0,
			from.1 - self.v_1.1,
			from.2 - self.v_1.2,
		);
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

		if t <= 0.0 { None } else { Some(t) }
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

mod trace;

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

		ui.add(
			Slider::new(&mut sound_velocity, 1.0..=4000.0)
				.logarithmic(true)
				.text("Sound velocity"),
		);
		ui.add(
			Slider::new(&mut hearable_distance, 0.1..=10.0)
				.logarithmic(true)
				.text("Hearable distance"),
		);
		ui.add(
			Slider::new(&mut decay_factor, 0.001..=10.0)
				.logarithmic(true)
				.text("Decay factor"),
		);

		if sound_velocity != self.sound_velocity
			|| hearable_distance != self.hearable_distance
			|| decay_factor != self.decay_factor
		{
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
		let mut convolver = FftConvolver::new(
			core::array::from_fn(|channel| {
				let mut ir = stereo[channel % 2].clone();
				ir.truncate(48000);
				ir
			}),
			sample_rate,
		);
		// The convolver's own default is dry 1 / wet 0.0125, which passes the
		// original and barely any reverb. The IR-based reverb is a wet effect.
		convolver.dry_gain = 0.0;
		convolver.wet_gain = 1.0;
		convolver.normalize_energy();
		Self { ir_gen, convolver }
	}
}

impl<T: GenIr + Send + Sync, const CHANNELS: usize> Effect<CHANNELS>
	for IrBasedReverb<T, CHANNELS>
{
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
			self.convolver.replace_ir(core::array::from_fn(|channel| {
				let mut ir = stereo[channel % 2].clone();
				ir.truncate(48000);
				ir
			}));
			self.convolver.normalize_energy();
		};
		gain_ui(
			ui,
			&mut self.convolver.dry_gain,
			Some("Dry gain".to_string()),
			false,
		);
		gain_ui(
			ui,
			&mut self.convolver.wet_gain,
			Some("Wet gain".to_string()),
			true,
		);
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
			let inward = (
				interior.0 - center.0,
				interior.1 - center.1,
				interior.2 - center.2,
			);
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
	/// The pan is positional: a reflection on the right goes right, on the left
	/// goes left, straight ahead is centred.
	#[test]
	fn the_stereo_pan_follows_the_reflection_side() {
		use super::*;

		let (left, right) = stereo_gains((0.0, 0.0, 1.0));
		assert!(right > left * 10.0, "right {right} left {left}");

		let (left, right) = stereo_gains((0.0, 0.0, -1.0));
		assert!(left > right * 10.0, "right {right} left {left}");

		let (left, right) = stereo_gains((1.0, 0.0, 0.0));
		assert!((left - right).abs() < 1e-4, "right {right} left {left}");
	}

	/// A longer chord through the listener gets more taps, and each window
	/// keeps its energy at one instead of turning into a louder hit.
	#[test]
	fn the_aperture_window_covers_the_chord() {
		use super::*;

		let short = aperture_window(0.001, 48_000);
		let long = aperture_window(0.010, 48_000);

		assert!(long.len() > short.len());

		for window in [&short, &long] {
			let energy: f32 = window.iter().map(|weight| weight * weight).sum();
			assert!((energy - 1.0).abs() < 1e-4, "energy {energy}");
		}
	}

	/// The normalised response must not start with a unit Dirac: that would turn
	/// the wet convolver into a zero-latency dry path.
	#[test]
	fn the_ir_does_not_start_with_a_dirac() {
		use super::*;

		let rt = RayTracing::rectangular_room(10.0, 10.0, 10.0, 0.001, 6.0, 48000);
		let ir = rt.generate_ir(Some(4_800));
		let stereo = rt.generate_stereo_ir(Some(4_800));

		assert!(ir[0].abs() < 0.01, "mono first sample {}", ir[0]);
		assert!(
			stereo[0][0].abs() < 0.01,
			"left first sample {}",
			stereo[0][0]
		);
		assert!(
			stereo[1][0].abs() < 0.01,
			"right first sample {}",
			stereo[1][0]
		);
	}

	#[test]
	fn test_ray_tracing() {
		use crate::{effects::reverb_ir_gen::RayTracing, tools::pcm_data::save_pcm_data};
		let mut rt = RayTracing::rectangular_room(10.0, 10.0, 10.0, 0.001, 3.0, 48000);
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
		use crate::{
			prelude::{Ball, RayTracing},
			tools::pcm_data::save_pcm_data,
		};

		let mut rt = RayTracing::random_space(&Ball { radius: 10.0 }, 100, 0.001, 0.0, 3.0, 48000);
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
