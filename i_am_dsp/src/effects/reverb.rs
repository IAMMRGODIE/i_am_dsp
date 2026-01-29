//! Simple reverb effect.

use bitvec::{bitvec, order::Lsb0};
use i_am_dsp_derive::Parameters;
use crate::{
	prelude::PureDelay, 
	tools::{
		matrix::{identity, mat_add, mat_mul_vec, mat_scale, random_householder_matrix, same, scalar_multiply, vector_add, Mat}, 
		ring_buffer::RingBuffer
	}, 
	Effect
};

fn format_feedback_matrix<const DELAY_LINES: usize, const CHANNELS: usize>(mat: &[Mat<DELAY_LINES>; CHANNELS]) -> Vec<u8> {
	let mut result = vec![];
	for channel in mat {
		for row in channel {
			for elem in row {
				let bits = f32::to_le_bytes(*elem);
				result.extend_from_slice(&bits);
			}
		}
	}
	result
}

fn parse_feedback_matrix<const DELAY_LINES: usize, const CHANNELS: usize>(data: Vec<u8>) -> [Mat<DELAY_LINES>; CHANNELS] {
	let mut result = [[[0.0; DELAY_LINES]; DELAY_LINES]; CHANNELS];
	let mut i = 0;
	for channel in result.iter_mut() {
		for row in channel.iter_mut() {
			for elem in row.iter_mut() {
				let bits = [data[i], data[i+1], data[i+2], data[i+3]];
				*elem = f32::from_le_bytes(bits);
				i += 4;
			}
		}
	}
	result
}

fn format_delay_time<const DELAY_LINES: usize>(delay_time: &[usize; DELAY_LINES]) -> Vec<u8> {
	let mut result = vec![];
	for delay in delay_time {
		let bits = (*delay as u32).to_le_bytes();
		result.extend_from_slice(&bits);
	}
	result
}

fn parse_delay_time<const DELAY_LINES: usize>(data: Vec<u8>) -> [usize; DELAY_LINES] {
	let mut result = [0; DELAY_LINES];
	let mut i = 0;
	for delay in result.iter_mut() {
		let bits = [data[i], data[i+1], data[i+2], data[i+3]];
		*delay = u32::from_le_bytes(bits) as usize;
		i += 4;
	}
	result
}

fn format_output_factors<const DELAY_LINES: usize>(output_factors: &[f32; DELAY_LINES]) -> Vec<u8> {
	let mut result = vec![];
	for factor in output_factors {
		let bits = f32::to_le_bytes(*factor);
		result.extend_from_slice(&bits);
	}
	result
}

fn parse_output_factors<const DELAY_LINES: usize>(data: Vec<u8>) -> [f32; DELAY_LINES] {
	let mut result = [0.0; DELAY_LINES];
	let mut i = 0;
	for factor in result.iter_mut() {
		let bits = [data[i], data[i+1], data[i+2], data[i+3]];
		*factor = f32::from_le_bytes(bits);
		i += 4;
	}
	result
}

/// A reverb effect based on feedback delay network.
#[derive(Parameters)]
pub struct Reverb<
	TailEffect: Effect<CHANNELS>,
	const DELAY_LINES: usize = 8, 
	const CHANNELS: usize = 2
> {
	/// The tail effect of the reverb.
	#[sub_param]
	pub tail: TailEffect,
	#[skip]
	delay_lines: [[RingBuffer<f32>; DELAY_LINES]; CHANNELS],
	#[persist(serialize = "format_feedback_matrix", deserialize = "parse_feedback_matrix")]
	feedback_matrix: [Mat<DELAY_LINES>; CHANNELS],
	#[persist(serialize = "format_delay_time", deserialize = "parse_delay_time")]
	delay_time: [usize; DELAY_LINES],
	#[sub_param]
	pure_delay: PureDelay<CHANNELS>,
	/// The weights when we summing up the delay lines.
	#[persist(serialize = "format_output_factors", deserialize = "parse_output_factors")]
	pub output_factors: [f32; DELAY_LINES],
	/// The input gain of the reverb, saves in linear scale.
	#[range(min = 0.01, max = 1.0)]
	#[logarithmic]
	pub reverbed_gain: f32,
	/// A parameter that controls the amount of diffusion in the reverb.
	/// 
	/// Should be between 0 and 1.
	pub diffusion: f32,
	/// The decay time of the reverb, saves in milliseconds
	pub decay_time: f32,
	/// The center delay of the reverb, saves in milliseconds
	center_delay: f32,
	/// The sample rate of the audio.
	#[skip]
	pub sample_rate: usize,

	#[cfg(feature = "real_time_demo")]
	prime_start_pos: usize,
}

#[derive(Debug, Clone, Default)]
/// The way to generate the feedback matrix.
pub enum FeedbackMatrixGen<const DELAY_LINES: usize> {
	/// A random Householder matrix.
	#[default] HouseHolder,
	/// An identity matrix.
	Identity,
	/// A custom matrix.
	Custom(Mat<DELAY_LINES>),
}

impl<const DELAY_LINES: usize> FeedbackMatrixGen<DELAY_LINES> {
	/// Generate the feedback matrix.
	pub fn generate(&self) -> Mat<DELAY_LINES> {
		match self {
			FeedbackMatrixGen::HouseHolder => {
				random_householder_matrix()
			},
			FeedbackMatrixGen::Identity => {
				identity()
			},
			FeedbackMatrixGen::Custom(mat) => *mat,
		}
	}
}

impl<
	TailEffect: Effect<CHANNELS>,
	const DELAY_LINES: usize,
	const CHANNELS: usize
> Reverb<TailEffect, DELAY_LINES, CHANNELS> {
	/// Create a new reverb effect.
	/// 
	/// # Panics
	/// 1. `DELAY_LINES` is 0,
	/// 2. `CHANNELS` is 0,
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
		let feedback_matrix = core::array::from_fn(|_| mat_gen.generate());
		let center_delay_samples = (center_delay / 1000.0 * sample_rate as f32) as usize;
		let delay_lines = core::array::from_fn(|_| {
			core::array::from_fn(|i| {
				RingBuffer::new(delay_time[i] + center_delay_samples)
			})
		});

		// println!("delay_time: {:?}", delay_time);

		Self {
			tail,
			delay_lines,
			delay_time,
			output_factors: [1.0 / DELAY_LINES as f32; DELAY_LINES],
			feedback_matrix,
			reverbed_gain: 1.0,
			diffusion: 1.0,
			decay_time: 1000.0,
			center_delay,
			sample_rate,
			pure_delay: PureDelay::new(65536, 10.0, sample_rate),

			#[cfg(feature = "real_time_demo")]
			prime_start_pos: delay_time_start,
		}
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

	/// Set the delay time of the reverb using the prime numbers.
	pub fn set_delay_use_prime(&mut self, start: usize) {
		let delay_time: [usize; DELAY_LINES] = prime(start);
		self.delay_time = delay_time;
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
		for delay_lines in self.delay_lines.iter_mut() {
			for delay_line in delay_lines.iter_mut() {
				delay_line.clear();
			}
		}
	}

	fn resize_delay_lines(&mut self) {
		let center_delay_samples = (self.center_delay / 1000.0 * self.sample_rate as f32) as usize;
		for delay_lines in self.delay_lines.iter_mut() {
			for (i, delay_line) in delay_lines.iter_mut().enumerate() {
				delay_line.resize(self.delay_time[i] + center_delay_samples);
			}
		}
	}
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
		result[N -1 - i] = primes[primes.len() - 1 - i];
	}

	result
}

impl<
	TailEffect: Effect<CHANNELS>,
	const DELAY_LINES: usize,
	const CHANNELS: usize
> Effect<CHANNELS> for Reverb<TailEffect, DELAY_LINES, CHANNELS> {
	fn delay(&self) -> usize {
		0
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"FDN Reverb"
	}

	fn process(
		&mut self, 
		samples: &mut [f32; CHANNELS], 
		other: &[&[f32; CHANNELS]],
		ctx: &mut Box<dyn crate::ProcessContext>,
	) {
		let sample_rate = self.sample_rate as f32;

		let center_delay_samples = self.center_delay / 1000.0 * sample_rate;
		let n_avg = self.delay_time.iter().map(|x| *x as f32).sum::<f32>() / DELAY_LINES as f32 + center_delay_samples;
		let decay_factor = 10.0_f32.powf(- 3.0 * n_avg / sample_rate / self.decay_time * 1e3);
		// println!("decay_factor: {}", decay_factor);

		let mut output = [0.0; CHANNELS];

		for (i, delay_lines) in self.delay_lines.iter_mut().enumerate() {
			let mat_1 = mat_scale(&self.feedback_matrix[i], self.diffusion);
			let mat_2 = mat_scale(&identity::<DELAY_LINES>(), 1.0 - self.diffusion);
			let mat = mat_add(&mat_1, &mat_2);

			let input: [f32; DELAY_LINES] = same(samples[i] * self.reverbed_gain);
			let mut delay_line_read = [0.0; DELAY_LINES];
			for (i, val) in delay_line_read.iter_mut().enumerate() {
				*val = delay_lines[i][0]; 
			}
			output[i] = delay_line_read.iter().map(|val| *val * self.output_factors[i]).sum::<f32>();
			delay_line_read = mat_mul_vec(&mat, &delay_line_read);
			delay_line_read = scalar_multiply(&delay_line_read, decay_factor);
			let update = vector_add(&delay_line_read, &input);
			for (i, val) in update.iter().enumerate() {
				delay_lines[i].push(*val);
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
		ui.add(egui::Slider::new(&mut self.decay_time, 1.0..=10000.0).text("Decay time (ms)"));
		let mut center_delay = self.center_delay;
		ui.add(egui::Slider::new(&mut center_delay, 1.0..=50.0).text("Center delay (ms)"));
		self.pure_delay.demo_ui(ui, format!("{}tail", id_prefix));
		if center_delay != self.center_delay {
			self.set_center_delay(center_delay);
		}
		let mut prime_start = self.prime_start_pos;
		ui.add(egui::Slider::new(&mut prime_start, 0..=100).text("Prime start"));
		if prime_start != self.prime_start_pos {
			if prime_start == 0 {
				self.set_delay_time([0; DELAY_LINES]);
			}else {
				self.set_delay_use_prime(prime_start);
			}
		}
		self.prime_start_pos = prime_start;

		gain_ui(ui, &mut self.reverbed_gain, Some("Reverbed gain".to_string()), true);

		ui.horizontal(|ui| {
			if ui.button("random matrix").clicked() {
				self.feedback_matrix = core::array::from_fn(|_| random_householder_matrix());
				// println!("Feedback matrix: {:?}", self.feedback_matrix);
			}
			if ui.button("identity matrix").clicked() {
				self.feedback_matrix = core::array::from_fn(|_| identity());
			}
		});

		self.tail.demo_ui(ui, format!("{}tail", id_prefix));
	}

}