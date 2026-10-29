//! The particle wider with a small product surface: the vocoder's depth and
//! band plus the mid/side width after it.

use i_am_dsp::prelude::{Effect, ParticleWider, ProcessContext};
use i_am_dsp_derive::Parameters;

#[derive(Parameters)]
pub struct ParticleWiderEffect {
	/// How strongly each partial is scattered.
	#[range(min = 0.0, max = 1.0)]
	#[unit(percent)]
	#[default(0.5)]
	#[id(display = "Depth")]
	#[layout(group = "Particle", order = 0, widget = knob, hero = "particles")]
	pub depth: f32,

	/// The lowest frequency that is scattered.
	#[range(min = 20.0, max = 20_000.0)]
	#[logarithmic]
	#[unit(hertz)]
	#[default(200.0)]
	#[id(display = "Low")]
	#[layout(group = "Particle", order = 1, widget = knob)]
	pub low_frequency: f32,

	/// The highest frequency that is scattered.
	#[range(min = 20.0, max = 20_000.0)]
	#[logarithmic]
	#[unit(hertz)]
	#[default(8_000.0)]
	#[id(display = "High")]
	#[layout(group = "Particle", order = 2, widget = knob)]
	pub high_frequency: f32,

	/// The mid gain after the widener; 1 is neutral.
	#[range(min = 0.0, max = 2.0)]
	#[unit(percent)]
	#[default(1.0)]
	#[id(display = "Mid")]
	#[layout(group = "Particle", order = 3, widget = knob)]
	pub mid_gain: f32,

	/// The side gain after the widener; 1 is neutral.
	#[range(min = 0.0, max = 2.0)]
	#[unit(percent)]
	#[default(1.0)]
	#[id(display = "Side")]
	#[layout(group = "Particle", order = 4, widget = knob)]
	pub side_gain: f32,

	/// The phase-vocoder window shape.
	#[range(min = 0.0, max = 1.0)]
	#[default(0.54)]
	#[id(display = "Window")]
	#[layout(group = "Particle", order = 5, widget = knob)]
	pub window_factor: f32,

	/// The FFT size, picked from the header.
	#[choices("512", "1024", "2048", "4096")]
	#[default(2)]
	#[id(display = "FFT")]
	#[layout(group = "Particle", order = 6, widget = choice)]
	pub fft_size: i32,

	/// Whether a scattered magnitude may be inverted as well.
	#[default(false)]
	#[id(display = "Negative")]
	#[layout(group = "Particle", order = 7, widget = toggle)]
	pub allow_negative: bool,

	#[skip]
	inner: ParticleWider,
	#[skip]
	last_window_factor: f32,
	#[skip]
	last_fft_size: i32,
}

impl ParticleWiderEffect {
	pub fn new(sample_rate: usize) -> Self {
		Self {
			depth: 0.5,
			low_frequency: 200.0,
			high_frequency: 8_000.0,
			mid_gain: 1.0,
			side_gain: 1.0,
			window_factor: 0.54,
			fft_size: 2,
			allow_negative: false,
			inner: ParticleWider::new(sample_rate),
			last_window_factor: 0.54,
			last_fft_size: 2,
		}
	}
}

impl Effect<2> for ParticleWiderEffect {
	fn delay(&self) -> usize {
		self.inner.delay()
	}

	fn process(
		&mut self,
		samples: &mut [f32; 2],
		other: &[&[f32; 2]],
		context: &mut Box<dyn ProcessContext>,
	) {
		if (self.window_factor - self.last_window_factor).abs() > 1e-4 {
			self.inner.vocoder.window_factor = self.window_factor.clamp(0.0, 1.0);
			self.inner.compensate_overlap_gain();
			self.last_window_factor = self.window_factor;
		}

		if self.fft_size != self.last_fft_size {
			let size = [512, 1024, 2048, 4096][self.fft_size.clamp(0, 3) as usize];
			if self.inner.vocoder.renew_window_size(size).is_some() {
				self.inner.compensate_overlap_gain();
			}
			self.last_fft_size = self.fft_size;
		}

		let mapper = &mut self.inner.vocoder.mapper;
		mapper.depth = self.depth;
		mapper.low_frequency = self.low_frequency;
		mapper.high_frequency = self.high_frequency;
		mapper.allow_negative = self.allow_negative;
		self.inner.stereo_control.mid_gain = self.mid_gain;
		self.inner.stereo_control.side_gain = self.side_gain;
		self.inner.process(samples, other, context);
	}
}
