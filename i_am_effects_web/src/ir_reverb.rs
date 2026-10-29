//! The ray-traced IR reverb: a generated room impulse response convolved with
//! an FFT convolver.
//!
//! The IR is built on a worker thread, so moving Room, Speed or Decay does not
//! stall the audio thread. The generator normalises the response and fades its
//! first few milliseconds, and the convolver is wet-only; the wrapper owns the
//! dry/wet pair.

use crossbeam_channel::{Receiver, Sender, unbounded};
use i_am_dsp::prelude::{Effect, FftConvolver, ProcessContext, RayTracing};
use i_am_dsp_derive::Parameters;

/// One IR request; the worker generates the newest one and discards the rest.
#[derive(Clone, Copy)]
struct IrRequest {
	size: f32,
	decay_time: f32,
	sound_velocity: f32,
	sample_rate: usize,
}

/// The background IR builder and its result channel.
struct IrWorker {
	requests: Sender<IrRequest>,
	results: Receiver<FftConvolver<2>>,
}

impl IrWorker {
	fn new() -> Self {
		let (requests, request_rx) = unbounded::<IrRequest>();
		let (result_tx, results) = unbounded::<FftConvolver<2>>();

		std::thread::spawn(move || {
			while let Ok(mut request) = request_rx.recv() {
				// A knob drag queues several requests; only the last one matters.
				while let Ok(next) = request_rx.try_recv() {
					request = next;
				}

				let convolver = build_convolver(request);
				if result_tx.send(convolver).is_err() {
					break;
				}
			}
		});

		Self { requests, results }
	}
}

fn generator(request: IrRequest) -> RayTracing {
	// The ray tracer decays by exp(-decay * distance); this puts T60 at the
	// requested time using the current speed of sound.
	let decay = (6.9 / ((request.decay_time.max(1.0) / 1_000.0) * request.sound_velocity))
		.clamp(1e-4, 1.0);
	let mut generator = RayTracing::rectangular_room(
		request.size,
		request.size * 0.7,
		request.size * 0.9,
		decay,
		4.0,
		request.sample_rate,
	);
	generator.sound_velocity = request.sound_velocity;

	generator
}

fn build_convolver(request: IrRequest) -> FftConvolver<2> {
	// Follow the requested T60 instead of truncating every IR to one second.
	let length = ((request.decay_time / 1_000.0) * request.sample_rate as f32).round() as usize;
	let length = length.clamp(4_800, 192_000);
	let stereo = generator(request).generate_stereo_ir(Some(length));
	let mut convolver = FftConvolver::new(
		core::array::from_fn(|channel| {
			let mut ir = stereo[channel % 2].clone();
			ir.truncate(length);
			ir
		}),
		request.sample_rate,
	);
	convolver.dry_gain = 0.0;
	convolver.wet_gain = 1.0;
	convolver.normalize_energy();

	convolver
}

/// The IR reverb with a room, speed, decay and a dry/wet pair.
#[derive(Parameters)]
pub struct IrReverbEffect {
	/// The room size in metres. The generated IR follows it.
	#[range(min = 5.0, max = 20.0)]
	#[default(10.0)]
	#[id(display = "Room")]
	#[layout(group = "IR Reverb", order = 0, widget = knob, hero = "tail")]
	pub size: f32,

	/// The speed of sound in m/s; it controls the arrival times in the IR.
	#[range(min = 50.0, max = 2_000.0)]
	#[default(343.0)]
	#[id(display = "Speed")]
	#[layout(group = "IR Reverb", order = 1, widget = knob)]
	pub sound_velocity: f32,

	/// The T60 decay time.
	#[range(min = 200.0, max = 5_000.0)]
	#[unit(milliseconds)]
	#[default(2_200.0)]
	#[id(display = "Decay")]
	#[layout(group = "IR Reverb", order = 2, widget = knob)]
	pub decay_time: f32,

	/// The dry gain of the effect.
	#[range(min = 0.0, max = 1.0)]
	#[unit(percent)]
	#[default(0.0)]
	#[id(display = "Dry")]
	#[layout(group = "IR Reverb", order = 3, widget = knob)]
	pub dry_gain: f32,

	/// The wet gain of the effect.
	#[range(min = 0.0, max = 1.0)]
	#[unit(percent)]
	#[default(1.0)]
	#[id(display = "Wet")]
	#[layout(group = "IR Reverb", order = 4, widget = knob)]
	pub wet_gain: f32,

	#[skip]
	convolver: FftConvolver<2>,
	#[skip]
	worker: IrWorker,
	#[skip]
	sample_rate: usize,
	#[skip]
	last_size: f32,
	#[skip]
	last_velocity: f32,
	#[skip]
	last_decay: f32,
}

impl IrReverbEffect {
	/// A 10 m room, 343 m/s, 2.2 s decay and a fully wet convolver.
	pub fn new(sample_rate: usize) -> Self {
		let request = IrRequest {
			size: 10.0,
			decay_time: 2_200.0,
			sound_velocity: 343.0,
			sample_rate,
		};
		let convolver = build_convolver(request);

		Self {
			size: request.size,
			sound_velocity: request.sound_velocity,
			decay_time: request.decay_time,
			dry_gain: 0.0,
			wet_gain: 1.0,
			convolver,
			worker: IrWorker::new(),
			sample_rate,
			last_size: request.size,
			last_velocity: request.sound_velocity,
			last_decay: request.decay_time,
		}
	}
}

impl Effect<2> for IrReverbEffect {
	fn delay(&self) -> usize {
		self.convolver.delay()
	}

	fn process(
		&mut self,
		samples: &mut [f32; 2],
		other: &[&[f32; 2]],
		context: &mut Box<dyn ProcessContext>,
	) {
		while let Ok(convolver) = self.worker.results.try_recv() {
			self.convolver = convolver;
		}

		if (self.size - self.last_size).abs() > 1e-4
			|| (self.sound_velocity - self.last_velocity).abs() > 1e-4
			|| (self.decay_time - self.last_decay).abs() > 1e-4
		{
			let request = IrRequest {
				size: self.size,
				decay_time: self.decay_time,
				sound_velocity: self.sound_velocity,
				sample_rate: self.sample_rate,
			};
			let _ = self.worker.requests.send(request);
			self.last_size = self.size;
			self.last_velocity = self.sound_velocity;
			self.last_decay = self.decay_time;
		}

		let dry = *samples;
		self.convolver.process(samples, other, context);

		for (index, sample) in samples.iter_mut().enumerate() {
			*sample = dry[index] * self.dry_gain + *sample * self.wet_gain;
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// With the default fully wet setting, the impulse must not come straight
	/// back as the dry signal.
	#[test]
	fn the_ir_reverb_does_not_pass_the_dry() {
		let mut effect = IrReverbEffect::new(48_000);
		let mut context: Box<dyn ProcessContext> = Box::new(());
		let mut first = 0.0;

		for index in 0..512 {
			let input = if index == 0 { 1.0 } else { 0.0 };
			let mut samples = [input, input];
			effect.process(&mut samples, &[], &mut context);

			if index == 0 {
				first = samples[0];
			}

			assert!(samples.iter().all(|sample| sample.is_finite()), "{samples:?}");
		}

		assert!(first.abs() < 0.01, "dry leaked at sample 0: {first}");
	}
}
