//! A feedback delay network reverb.

use std::mem::size_of;

use bitvec::{bitvec, order::Lsb0};
use i_am_dsp_derive::Parameters;

use crate::{
	Effect, ProcessContext,
	prelude::PureDelay,
	tools::matrix::{Mat, mat_mul_vec, random_householder_vector},
};

/// How many allpass stages smear the input before it reaches the delay lines.
const INPUT_DIFFUSION_STAGES: usize = 4;

/// The delay of every input diffusion stage, in seconds.
///
/// The lengths are mutually close to coprime, so the stages do not line up and
/// the diffusion stays smooth instead of building a second comb.
const INPUT_DIFFUSION_DELAYS: [f32; INPUT_DIFFUSION_STAGES] = [0.0017, 0.0033, 0.0061, 0.0113];

/// The allpass feedback of the diffusion stages at full diffusion.
const INPUT_DIFFUSION_FEEDBACK: f32 = 0.7;

/// Feedback values below this are flushed to zero.
///
/// The loop keeps multiplying the tail by a gain below one, so it walks straight
/// through the subnormal range, where every operation costs an order of
/// magnitude more on x86. Flushing here keeps the tail affordable for callers
/// that cannot turn on flush to zero for their audio thread.
const DENORMAL_THRESHOLD: f32 = 1e-25;

mod bytes;
#[cfg(test)]
mod tests;

use bytes::*;

/// The feedback matrix of one channel.
#[derive(Debug, Clone, Copy)]
pub(crate) enum FeedbackMatrix<const DELAY_LINES: usize> {
	/// The identity minus twice the outer product of a unit vector, which is a
	/// lossless reflection and can be applied in O(DELAY_LINES) instead of
	/// O(DELAY_LINES²). A zero vector is the identity matrix.
	Householder([f32; DELAY_LINES]),
	/// Any other matrix, applied in O(DELAY_LINES²).
	Dense(Mat<DELAY_LINES>),
}

impl<const DELAY_LINES: usize> Default for FeedbackMatrix<DELAY_LINES> {
	fn default() -> Self {
		Self::Householder([0.0; DELAY_LINES])
	}
}

impl<const DELAY_LINES: usize> FeedbackMatrix<DELAY_LINES> {
	/// A random lossless reflection.
	fn householder() -> Self {
		Self::Householder(random_householder_vector())
	}

	/// The identity: a network of independent delay lines.
	fn identity() -> Self {
		Self::Householder([0.0; DELAY_LINES])
	}

	/// Applies the matrix to 'input', writing the result into 'output'.
	#[inline]
	fn mix(&self, input: &[f32; DELAY_LINES], output: &mut [f32; DELAY_LINES]) {
		match self {
			Self::Householder(vector) => {
				// (I - 2 v vᵀ) x = x - 2 v (vᵀ x)
				let mut projection = 0.0;
				for line in 0..DELAY_LINES {
					projection += vector[line] * input[line];
				}
				let scale = 2.0 * projection;
				for line in 0..DELAY_LINES {
					output[line] = input[line] - scale * vector[line];
				}
			}
			Self::Dense(matrix) => {
				*output = mat_mul_vec(matrix, input);
			}
		}
	}

	/// The matrix as a plain dense matrix.
	fn to_dense(self) -> Mat<DELAY_LINES> {
		match self {
			Self::Householder(vector) => core::array::from_fn(|i| {
				core::array::from_fn(|j| {
					let identity = if i == j { 1.0 } else { 0.0 };
					identity - 2.0 * vector[i] * vector[j]
				})
			}),
			Self::Dense(matrix) => matrix,
		}
	}

	/// Recognizes a stored dense matrix that is a Householder reflection, so
	/// presets saved with the default matrix keep the O(DELAY_LINES) mix.
	fn from_dense(matrix: Mat<DELAY_LINES>) -> Self {
		// An off diagonal element is -2 u_i u_j, so the column of the largest
		// component fixes the sign of every component.
		let mut pivot = 0;
		let mut largest = 0.0;
		for (i, row) in matrix.iter().enumerate() {
			if row[i].abs() > largest {
				largest = row[i].abs();
				pivot = i;
			}
		}
		let pivot_vector = ((1.0 - matrix[pivot][pivot]) / 2.0).max(0.0).sqrt();
		if pivot_vector < 1e-6 {
			// The matrix is the identity.
			return Self::identity();
		}
		let mut vector = [0.0; DELAY_LINES];
		for i in 0..DELAY_LINES {
			vector[i] = -matrix[i][pivot] / (2.0 * pivot_vector);
		}
		let is_householder = matrix.iter().enumerate().all(|(i, row)| {
			row.iter().enumerate().all(|(j, elem)| {
				let identity = if i == j { 1.0 } else { 0.0 };
				(*elem - (identity - 2.0 * vector[i] * vector[j])).abs() <= 1e-4
			})
		});
		if is_householder {
			Self::Householder(vector)
		} else {
			Self::Dense(matrix)
		}
	}
}

/// A delay line with an explicit write position.
///
/// The feedback loop touches these once per sample, and the modulo a ring buffer
/// needs per access is a measurable part of that, so the position is wrapped
/// with a comparison instead.
#[derive(Default)]
struct DelayLine {
	buffer: Vec<f32>,
	position: usize,
}

impl DelayLine {
	fn new(capacity: usize) -> Self {
		Self {
			buffer: vec![0.0; capacity.max(1)],
			position: 0,
		}
	}

	/// The oldest sample, which is what a delay of the whole buffer returns.
	#[inline]
	fn read(&self) -> f32 {
		self.buffer[self.position]
	}

	#[inline]
	fn write(&mut self, value: f32) {
		self.buffer[self.position] = value;
		self.position += 1;
		if self.position == self.buffer.len() {
			self.position = 0;
		}
	}

	fn clear(&mut self) {
		self.buffer.fill(0.0);
		self.position = 0;
	}

	fn resize(&mut self, capacity: usize) {
		self.buffer = vec![0.0; capacity.max(1)];
		self.position = 0;
	}
}

/// One Schroeder allpass stage of the input diffusion.
///
/// An allpass passes every frequency at unit magnitude, so it smears the input
/// in time without colouring it.
#[derive(Default)]
struct Allpass {
	line: DelayLine,
	feedback: f32,
}

impl Allpass {
	#[inline]
	fn process(&mut self, input: f32) -> f32 {
		let delayed = self.line.read();
		let state = input + self.feedback * delayed;
		// Flushing the state keeps everything the line hands out normal or
		// zero, which is what keeps subnormal inputs from slowing the whole
		// network down.
		let state = if state.abs() > DENORMAL_THRESHOLD {
			state
		} else {
			0.0
		};
		self.line.write(state);
		delayed - self.feedback * state
	}
}

/// Everything the per sample loop derives from the parameters.
#[derive(Clone, Copy)]
struct Coefficients<const DELAY_LINES: usize> {
	/// Per line feedback gain that gives that delay line the requested T60.
	line_gain: [f32; DELAY_LINES],
	/// The one pole damping in the feedback path, as a 0..1 blend per sample.
	damping: f32,
	/// The cosine and sine of the channel rotation that shares the feedback.
	spread_cos: f32,
	spread_sin: f32,
	/// Whether the values are up to date.
	valid: bool,
	/// The parameters they were computed from.
	decay_time: f32,
	damping_cutoff: f32,
	spread: f32,
	sample_rate: usize,
}

impl<const DELAY_LINES: usize> Default for Coefficients<DELAY_LINES> {
	fn default() -> Self {
		Self {
			line_gain: [0.0; DELAY_LINES],
			damping: 0.0,
			spread_cos: 1.0,
			spread_sin: 0.0,
			valid: false,
			decay_time: 0.0,
			damping_cutoff: 0.0,
			spread: 0.0,
			sample_rate: 0,
		}
	}
}

/// A reverb effect based on a feedback delay network.
///
/// Every channel runs its own network: the delay lines are read, damped and
/// weighted with the channel's feedback matrix, and written back together with
/// the diffused input. The output is the weighted sum of the delay line outputs,
/// so 'output_factors' shapes which lines are heard.
///
/// The network itself has no latency; the pre delay in front of it does, and is
/// reported through 'Effect::delay'.
#[derive(Parameters)]
pub struct Reverb<
	TailEffect: Effect<CHANNELS>,
	const DELAY_LINES: usize = 8,
	const CHANNELS: usize = 2,
> {
	/// The tail effect of the reverb.
	#[sub_param]
	pub tail: TailEffect,
	/// The delay lines, one set per channel.
	#[skip]
	delay_lines: [[DelayLine; DELAY_LINES]; CHANNELS],
	/// The state of the damping filter that sits in the feedback path.
	#[skip]
	damping_state: [[f32; DELAY_LINES]; CHANNELS],
	/// The allpass stages that diffuse the input.
	#[skip]
	diffusor: [[Allpass; INPUT_DIFFUSION_STAGES]; CHANNELS],
	/// The feedback matrix of every channel.
	#[persist(
		serialize = "format_feedback_matrix",
		deserialize = "parse_feedback_matrix"
	)]
	feedback_matrix: [FeedbackMatrix<DELAY_LINES>; CHANNELS],
	/// The gain every delay line is excited with.
	///
	/// The lines are excited with alternating signs so that their early echoes
	/// do not add up into a discrete reflection, at unit total energy.
	#[skip]
	input_factors: [f32; DELAY_LINES],
	/// The delay time of the reverb, in samples.
	#[persist(serialize = "format_delay_time", deserialize = "parse_delay_time")]
	delay_time: [usize; DELAY_LINES],
	/// The pre delay in front of the network.
	#[sub_param]
	pure_delay: PureDelay<CHANNELS>,
	/// The weights the delay lines are summed up with.
	#[persist(
		serialize = "format_output_factors",
		deserialize = "parse_output_factors"
	)]
	pub output_factors: [f32; DELAY_LINES],
	/// The input gain of the reverb, saves in linear scale.
	#[range(min = 0.01, max = 1.0)]
	#[logarithmic]
	#[default(1.0)]
	pub reverbed_gain: f32,
	/// How much the input diffusion smears the input, between 0 and 1.
	#[default(1.0)]
	pub diffusion: f32,
	/// The decay time of the reverb, saves in milliseconds.
	#[unit(milliseconds)]
	#[default(1000.0)]
	pub decay_time: f32,
	/// The cutoff of the damping in the feedback path, in Hz.
	///
	/// This is what makes the tail lose its high frequencies first, like the air
	/// absorption of a room.
	#[range(min = 200.0, max = 20000.0)]
	#[logarithmic]
	#[unit(hertz)]
	#[default(12000.0)]
	pub damping_cutoff: f32,
	/// How much of the feedback the channels share, between 0 and 1.
	///
	/// At 0 every channel is an independent network, at 1 the channels are
	/// rotated fully into each other, which makes the two sides of a stereo
	/// signal excite the same set of modes. A rotation is used instead of a
	/// blend so that the sum and the difference of the channels keep decaying
	/// at the same rate.
	#[range(min = 0.0, max = 1.0)]
	#[default(0.5)]
	#[unit(percent)]
	pub spread: f32,
	/// The center delay of the reverb, saves in milliseconds.
	center_delay: f32,
	/// The sample rate of the audio.
	#[skip]
	pub sample_rate: usize,
	/// The values derived from the parameters above.
	#[skip]
	coefficients: Coefficients<DELAY_LINES>,

	#[cfg(feature = "real_time_demo")]
	#[skip]
	prime_start_pos: usize,
}

#[derive(Debug, Clone, Default)]
/// The way to generate the feedback matrix.
pub enum FeedbackMatrixGen<const DELAY_LINES: usize> {
	/// A random Householder matrix.
	#[default]
	HouseHolder,
	/// An identity matrix.
	Identity,
	/// A custom matrix.
	Custom(Mat<DELAY_LINES>),
}

impl<const DELAY_LINES: usize> FeedbackMatrixGen<DELAY_LINES> {
	/// Generate the feedback matrix.
	pub fn generate(&self) -> Mat<DELAY_LINES> {
		match self {
			Self::HouseHolder => FeedbackMatrix::householder().to_dense(),
			Self::Identity => FeedbackMatrix::identity().to_dense(),
			Self::Custom(mat) => *mat,
		}
	}
}

impl<TailEffect: Effect<CHANNELS>, const DELAY_LINES: usize, const CHANNELS: usize>
	Reverb<TailEffect, DELAY_LINES, CHANNELS>
{
	/// Create a new reverb effect.
	///
	/// # Panics
	///
	/// 1. 'DELAY_LINES' is 0,
	/// 2. 'CHANNELS' is 0,
	/// 3. 'delay_time_start' is 0.
	pub fn new(
		tail: TailEffect,
		sample_rate: usize,
		mat_gen: FeedbackMatrixGen<DELAY_LINES>,
		delay_time_start: usize,
		center_delay: f32,
	) -> Self {
		assert!(DELAY_LINES > 0, "Delay lines must be greater than 0");
		assert!(CHANNELS > 0, "Channels must be greater than 0");
		let delay_time: [usize; DELAY_LINES] = prime(delay_time_start);
		let feedback_matrix =
			core::array::from_fn(|_| FeedbackMatrix::from_dense(mat_gen.generate()));
		// Unit energy per channel, without the common component that would make
		// every line echo at the same time.
		let scale = 1.0 / (DELAY_LINES as f32).sqrt();
		let input_factors = core::array::from_fn(|line| if line % 2 == 0 { scale } else { -scale });
		let center_delay_samples = (center_delay / 1000.0 * sample_rate as f32) as usize;
		let delay_lines = core::array::from_fn(|_| {
			core::array::from_fn(|line| DelayLine::new(delay_time[line] + center_delay_samples))
		});

		let mut result = Self {
			tail,
			delay_lines,
			damping_state: [[0.0; DELAY_LINES]; CHANNELS],
			diffusor: core::array::from_fn(|_| {
				core::array::from_fn(|stage| Allpass {
					line: DelayLine::new(input_diffusion_delay(stage, sample_rate)),
					feedback: INPUT_DIFFUSION_FEEDBACK,
				})
			}),
			feedback_matrix,
			input_factors,
			delay_time,
			pure_delay: PureDelay::new(65536, 10.0, sample_rate),
			output_factors: [scale; DELAY_LINES],
			reverbed_gain: 1.0,
			diffusion: 1.0,
			decay_time: 1000.0,
			damping_cutoff: 12000.0,
			spread: 0.5,
			center_delay,
			sample_rate,
			coefficients: Coefficients::default(),

			#[cfg(feature = "real_time_demo")]
			prime_start_pos: delay_time_start,
		};
		result.refresh_coefficients();
		result
	}

	/// Get the center delay of the reverb, in milliseconds.
	pub fn center_delay(&self) -> f32 {
		self.center_delay
	}

	/// set the center delay of the reverb, in milliseconds.
	pub fn set_center_delay(&mut self, center_delay: f32) {
		self.center_delay = center_delay;
		self.resize_delay_lines();
	}

	/// Get the pre delay in front of the network, in milliseconds.
	pub fn pre_delay(&self) -> f32 {
		self.pure_delay.delay_time
	}

	/// Set the pre delay in front of the network, in milliseconds. The pure
	/// delay's buffer is a fixed size, so the value is clamped to a second.
	pub fn set_pre_delay(&mut self, pre_delay: f32) {
		self.pure_delay.delay_time = pre_delay.clamp(0.0, 1000.0);
	}

	/// Set the delay time of the reverb using the prime numbers.
	pub fn set_delay_use_prime(&mut self, start: usize) {
		self.delay_time = prime(start);
		self.resize_delay_lines();
	}

	/// Set the delay time of the reverb.
	pub fn set_delay_time(&mut self, delay_time: [usize; DELAY_LINES]) {
		self.delay_time = delay_time;
		self.resize_delay_lines();
	}

	/// Get the delay time of the reverb, in samples.
	pub fn get_delay_time(&self) -> [usize; DELAY_LINES] {
		self.delay_time
	}

	/// Clear the delay lines to stop the reverb.
	pub fn clear_delay_lines(&mut self) {
		for channel in 0..CHANNELS {
			for line in self.delay_lines[channel].iter_mut() {
				line.clear();
			}
			for stage in self.diffusor[channel].iter_mut() {
				stage.line.clear();
			}
			self.damping_state[channel] = [0.0; DELAY_LINES];
		}

		// The pre-delay and the damping tail are separate states in front of and
		// inside the network.
		self.pure_delay.clear_history();
		self.tail.clear_tail();
	}

	/// Recomputes the values the per sample loop derives from the parameters.
	fn refresh_coefficients(&mut self) {
		let sample_rate = self.sample_rate.max(1) as f32;
		let decay_seconds = self.decay_time.max(1.0) / 1000.0;
		let center_delay_samples = (self.center_delay / 1000.0 * sample_rate) as usize;
		for line in 0..DELAY_LINES {
			// A line of D samples passes the signal through once every D
			// samples, so its gain is the one that reaches -60 dB after the
			// decay time. Every mode then decays at the same rate, instead of
			// the short lines dying first.
			let delay = (self.delay_time[line] + center_delay_samples).max(1) as f32;
			self.coefficients.line_gain[line] =
				10.0_f32.powf(-3.0 * delay / sample_rate / decay_seconds);
		}
		// y += g (x - y) has its -3 dB point at about g * sample_rate / 2 pi.
		let cutoff = self.damping_cutoff.clamp(1.0, sample_rate / 2.0);
		self.coefficients.damping =
			1.0 - (-2.0 * core::f32::consts::PI * cutoff / sample_rate).exp();
		let angle = self.spread.clamp(0.0, 1.0) * core::f32::consts::FRAC_PI_4;
		self.coefficients.spread_cos = angle.cos();
		self.coefficients.spread_sin = angle.sin();
		self.coefficients.decay_time = self.decay_time;
		self.coefficients.damping_cutoff = self.damping_cutoff;
		self.coefficients.spread = self.spread;
		self.coefficients.sample_rate = self.sample_rate;
		self.coefficients.valid = true;
	}

	/// Whether the derived values no longer match the parameters.
	#[inline]
	fn coefficients_stale(&self) -> bool {
		!self.coefficients.valid
			|| self.coefficients.decay_time != self.decay_time
			|| self.coefficients.damping_cutoff != self.damping_cutoff
			|| self.coefficients.spread != self.spread
			|| self.coefficients.sample_rate != self.sample_rate
	}

	fn resize_delay_lines(&mut self) {
		let center_delay_samples = (self.center_delay / 1000.0 * self.sample_rate as f32) as usize;
		for delay_lines in self.delay_lines.iter_mut() {
			for (line, delay_line) in delay_lines.iter_mut().enumerate() {
				delay_line.resize(self.delay_time[line] + center_delay_samples);
			}
		}
		// The line lengths changed, so the decay gains have to follow.
		self.coefficients.valid = false;
		self.clear_delay_lines();
	}
}

/// The delay of one input diffusion stage, in samples.
fn input_diffusion_delay(stage: usize, sample_rate: usize) -> usize {
	(INPUT_DIFFUSION_DELAYS[stage] * sample_rate as f32)
		.round()
		.max(1.0) as usize
}

fn prime<const N: usize>(start: usize) -> [usize; N] {
	assert!(N > 0, "N must be greater than 0");
	assert!(start > 0, "Start must be greater than 0");

	let last_index = start + N - 1;

	let upper_bound = if last_index < 6 {
		20
	} else {
		let i_f64 = last_index as f64;
		let ln_i = i_f64.ln();
		let ln_ln_i = ln_i.ln();
		(i_f64 * (ln_i + ln_ln_i - 0.5)).ceil() as usize
	};

	let mut sieve = bitvec![u8, Lsb0; 0; upper_bound + 1];
	sieve.set(0, true);
	sieve.set(1, true);

	let sqrt_n = (upper_bound as f64).sqrt() as usize + 1;

	for i in 2..=sqrt_n {
		if !sieve[i] {
			let mut j = i * i;
			while j <= upper_bound {
				sieve.set(j, true);
				j += i;
			}
		}
	}

	let mut primes = vec![];
	for i in 2..=upper_bound {
		if !sieve[i] {
			primes.push(i);
		}
	}

	let mut result = [0; N];
	for i in 0..N {
		result[N - 1 - i] = primes[primes.len() - 1 - i];
	}

	result
}

impl<TailEffect: Effect<CHANNELS>, const DELAY_LINES: usize, const CHANNELS: usize> Effect<CHANNELS>
	for Reverb<TailEffect, DELAY_LINES, CHANNELS>
{
	fn delay(&self) -> usize {
		// The network has no latency of its own, the pre delay does.
		self.pure_delay.delay()
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"FDN Reverb"
	}

	fn process(
		&mut self,
		samples: &mut [f32; CHANNELS],
		other: &[&[f32; CHANNELS]],
		ctx: &mut Box<dyn ProcessContext>,
	) {
		if self.coefficients_stale() {
			self.refresh_coefficients();
		}

		let input = *samples;
		let mut output = [0.0; CHANNELS];
		let damping = self.coefficients.damping;
		let diffusion = INPUT_DIFFUSION_FEEDBACK * self.diffusion.clamp(0.0, 1.0);

		// Diffuse the input of every channel first: feeding the lines directly
		// makes the early response a train of discrete echoes.
		let mut diffused = [0.0; CHANNELS];
		for (channel, diffused) in diffused.iter_mut().enumerate() {
			let mut value = input[channel] * self.reverbed_gain;
			for stage in self.diffusor[channel].iter_mut() {
				stage.feedback = diffusion;
				value = stage.process(value);
			}
			*diffused = value;
		}

		// Read, damp and weight every line of every channel before anything is
		// written back, because the channels are mixed into each other first.
		let mut feedback = [[0.0; DELAY_LINES]; CHANNELS];
		for (channel, feedback) in feedback.iter_mut().enumerate() {
			for (line, value) in feedback.iter_mut().enumerate() {
				let delayed = self.delay_lines[channel][line].read();
				output[channel] += delayed * self.output_factors[line];
				// Damping belongs inside the loop: a filter after the sum can
				// only colour the tail, it cannot make it decay darker.
				let state = &mut self.damping_state[channel][line];
				*state += damping * (delayed - *state);
				if state.abs() <= DENORMAL_THRESHOLD {
					*state = 0.0;
				}
				*value = *state * self.coefficients.line_gain[line];
			}
		}

		// Rotate the channels into each other: sharing part of the feedback is
		// what makes this one stereo network instead of two independent reverbs.
		if CHANNELS > 1 {
			let cos = self.coefficients.spread_cos;
			let sin = self.coefficients.spread_sin;
			for channel in (0..CHANNELS).step_by(2) {
				if channel + 1 >= CHANNELS {
					break;
				}
				let (head, tail) = feedback.split_at_mut(channel + 1);
				let left = &mut head[channel];
				let right = &mut tail[0];
				for (left, right) in left.iter_mut().zip(right.iter_mut()) {
					let (first, second) = (*left, *right);
					// A rotation and not a blend: its eigenvalues sit on the unit
					// circle, so the loop stays lossless at every spread.
					*left = cos * first - sin * second;
					*right = sin * first + cos * second;
				}
			}
		}

		for (channel, diffused) in diffused.iter().enumerate() {
			let mut mixed = [0.0; DELAY_LINES];
			self.feedback_matrix[channel].mix(&feedback[channel], &mut mixed);

			for (line, mixed) in mixed.iter().enumerate() {
				let value = *mixed + *diffused * self.input_factors[line];
				self.delay_lines[channel][line].write(if value.abs() > DENORMAL_THRESHOLD {
					value
				} else {
					0.0
				});
			}
		}

		self.tail.process(&mut output, other, ctx);
		self.pure_delay.process(&mut output, other, ctx);
		*samples = output;

		if ctx.should_stop() {
			self.clear_delay_lines();
		}
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		use crate::tools::ui_tools::gain_ui;

		ui.add(egui::Slider::new(&mut self.diffusion, 0.0..=1.0).text("Diffusion"));
		ui.add(egui::Slider::new(&mut self.spread, 0.0..=1.0).text("Stereo spread"));
		ui.add(egui::Slider::new(&mut self.decay_time, 1.0..=10000.0).text("Decay time (ms)"));
		ui.add(
			egui::Slider::new(&mut self.damping_cutoff, 200.0..=20000.0)
				.logarithmic(true)
				.text("Damping (Hz)"),
		);
		let mut center_delay = self.center_delay;
		ui.add(egui::Slider::new(&mut center_delay, 1.0..=50.0).text("Center delay (ms)"));
		self.pure_delay
			.demo_ui(ui, format!("{}pre_delay", id_prefix));
		if center_delay != self.center_delay {
			self.set_center_delay(center_delay);
		}

		let mut prime_start = self.prime_start_pos;
		ui.add(egui::Slider::new(&mut prime_start, 0..=100).text("Prime start"));
		if prime_start != self.prime_start_pos {
			if prime_start == 0 {
				self.set_delay_time([0; DELAY_LINES]);
			} else {
				self.set_delay_use_prime(prime_start);
			}
		}
		self.prime_start_pos = prime_start;

		gain_ui(
			ui,
			&mut self.reverbed_gain,
			Some("Reverbed gain".to_string()),
			true,
		);

		ui.horizontal(|ui| {
			if ui.button("random matrix").clicked() {
				self.feedback_matrix = core::array::from_fn(|_| FeedbackMatrix::householder());
			}
			if ui.button("identity matrix").clicked() {
				self.feedback_matrix = core::array::from_fn(|_| FeedbackMatrix::identity());
			}
		});

		self.tail.demo_ui(ui, format!("{}tail", id_prefix));
	}
}
