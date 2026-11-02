//! The common downward dynamics: a gate, a downward expander, a peak limiter
//! and a transient shaper.
//!
//! They all read a linked peak detector and shape a gain, but the gain computer
//! is what makes them different: a gate closes below a threshold, an expander
//! falls further the quieter the signal is, a limiter caps the level, and the
//! transient shaper splits the detector into a fast and a slow envelope.

use i_am_dsp_derive::Parameters;

use crate::{Effect, ProcessContext};

/// Linear amplitude for a level in dB.
fn db_to_linear(db: f32) -> f32 {
	10.0_f32.powf(db / 20.0)
}

/// Level in dB for a linear amplitude, with a floor instead of -infinity.
fn linear_to_db(value: f32) -> f32 {
	20.0 * value.max(1e-9).log10()
}

/// The one-pole coefficient that reaches about 63 % of a step in the given time.
fn coefficient(milliseconds: f32, sample_rate: f32) -> f32 {
	(-1.0 / (milliseconds.max(0.01) * 0.001 * sample_rate)).exp()
}

/// The loudest absolute sample in a frame, so the detector is linked across channels.
fn peak<const CHANNELS: usize>(frame: &[f32; CHANNELS]) -> f32 {
	frame
		.iter()
		.fold(0.0_f32, |peak, sample| peak.max(sample.abs()))
}

/// A one-pole gain smoother with separate attack and release coefficients.
fn ballistics(current: f32, target: f32, attack: f32, release: f32) -> f32 {
	let coefficient = if target < current { attack } else { release };
	coefficient * current + (1.0 - coefficient) * target
}

/// A peak envelope with an instant attack and a one-pole release.
fn follow(current: f32, level: f32, release: f32) -> f32 {
	if level > current {
		level
	} else {
		release * current + (1.0 - release) * level
	}
}

/// A peak envelope with separate attack and release times.
fn follow_timed(current: f32, level: f32, attack: f32, release: f32) -> f32 {
	let coefficient = if level > current { attack } else { release };
	coefficient * current + (1.0 - coefficient) * level
}

/// A gate: the signal is let through above the threshold and attenuated by
/// Range below it, with a hold time so a decaying note is not chopped.
#[derive(Parameters)]
pub struct Gate<const CHANNELS: usize = 2> {
	/// The level where the gate opens.
	#[range(min = -80.0, max = 0.0)]
	#[unit(decibels)]
	#[default(-40.0)]
	#[id(display = "Threshold")]
	#[layout(group = "Gate", order = 0, widget = knob, hero = "gate")]
	pub threshold: f32,

	/// How much the signal is attenuated while the gate is closed.
	#[range(min = 0.0, max = 80.0)]
	#[unit(decibels)]
	#[default(60.0)]
	#[id(display = "Range")]
	#[layout(group = "Gate", order = 1, widget = knob)]
	pub range: f32,

	/// How quickly the gate opens.
	#[range(min = 0.1, max = 50.0)]
	#[unit(milliseconds)]
	#[default(1.0)]
	#[id(display = "Attack")]
	#[layout(group = "Gate", order = 2, widget = knob)]
	pub attack: f32,

	/// How long the gate stays open after the signal falls below the threshold.
	#[range(min = 0.0, max = 500.0)]
	#[unit(milliseconds)]
	#[default(50.0)]
	#[id(display = "Hold")]
	#[layout(group = "Gate", order = 3, widget = knob)]
	pub hold: f32,

	/// How quickly the gate closes.
	#[range(min = 5.0, max = 2_000.0)]
	#[unit(milliseconds)]
	#[default(150.0)]
	#[id(display = "Release")]
	#[layout(group = "Gate", order = 4, widget = knob)]
	pub release: f32,

	/// The gap between the open and close thresholds, so a signal near the
	/// threshold does not chatter.
	#[range(min = 0.0, max = 12.0)]
	#[unit(decibels)]
	#[default(3.0)]
	#[id(display = "Hysteresis")]
	#[layout(group = "Gate", order = 5, widget = knob)]
	pub hysteresis: f32,

	/// Follow the sidechain input instead of the main input.
	#[default(false)]
	#[id(display = "Sidechain")]
	#[layout(group = "Gate", order = 6, widget = toggle)]
	pub sidechain: bool,

	#[skip]
	sample_rate: f32,
	#[skip]
	detector: f32,
	#[skip]
	gain: f32,
	#[skip]
	open: bool,
	#[skip]
	hold_counter: usize,
}

impl<const CHANNELS: usize> Gate<CHANNELS> {
	/// A gate in its starting state: closed, at full attenuation.
	pub fn new(sample_rate: usize) -> Self {
		Self {
			threshold: -40.0,
			range: 60.0,
			attack: 1.0,
			hold: 50.0,
			release: 150.0,
			hysteresis: 3.0,
			sidechain: false,
			sample_rate: sample_rate as f32,
			detector: 0.0,
			gain: db_to_linear(-60.0),
			open: false,
			hold_counter: 0,
		}
	}
}

impl<const CHANNELS: usize> Effect<CHANNELS> for Gate<CHANNELS> {
	fn delay(&self) -> usize {
		0
	}

	fn process(
		&mut self,
		samples: &mut [f32; CHANNELS],
		other: &[&[f32; CHANNELS]],
		_: &mut Box<dyn ProcessContext>,
	) {
		let frame = if self.sidechain {
			other.get(1).copied().unwrap_or(&*samples)
		} else {
			&*samples
		};
		let release = coefficient(5.0, self.sample_rate);
		self.detector = follow(self.detector, peak(frame), release);
		let level_db = linear_to_db(self.detector);
		let hold = (self.hold * 0.001 * self.sample_rate).round() as usize;

		if level_db >= self.threshold {
			self.open = true;
			self.hold_counter = hold;
		} else if self.open {
			if self.hold_counter > 0 {
				self.hold_counter -= 1;
			} else if level_db < self.threshold - self.hysteresis {
				self.open = false;
			}
		}

		let target = if self.open {
			1.0
		} else {
			db_to_linear(-self.range)
		};
		let attack = coefficient(self.attack, self.sample_rate);
		let release = coefficient(self.release, self.sample_rate);
		self.gain = ballistics(self.gain, target, attack, release);

		for sample in samples.iter_mut() {
			*sample *= self.gain;
			if !sample.is_finite() {
				*sample = 0.0;
			}
		}
	}

	fn clear_tail(&mut self) {
		self.detector = 0.0;
		self.gain = db_to_linear(-60.0);
		self.open = false;
		self.hold_counter = 0;
	}
}

/// A downward expander: below the threshold the level is pushed down by the
/// ratio, so quiet details get quieter instead of being cut off like a gate.
#[derive(Parameters)]
pub struct Expander<const CHANNELS: usize = 2> {
	/// The level where downward expansion starts.
	#[range(min = -80.0, max = 0.0)]
	#[unit(decibels)]
	#[default(-40.0)]
	#[id(display = "Threshold")]
	#[layout(group = "Expander", order = 0, widget = knob, hero = "expander")]
	pub threshold: f32,

	/// How much further below the threshold the level falls.
	#[range(min = 1.0, max = 10.0)]
	#[default(2.0)]
	#[id(display = "Ratio")]
	#[layout(group = "Expander", order = 1, widget = knob)]
	pub ratio: f32,

	/// The deepest attenuation the expander may apply.
	#[range(min = 0.0, max = 80.0)]
	#[unit(decibels)]
	#[default(40.0)]
	#[id(display = "Range")]
	#[layout(group = "Expander", order = 2, widget = knob)]
	pub range: f32,

	/// How quickly the gain falls when the signal drops.
	#[range(min = 0.1, max = 50.0)]
	#[unit(milliseconds)]
	#[default(5.0)]
	#[id(display = "Attack")]
	#[layout(group = "Expander", order = 3, widget = knob)]
	pub attack: f32,

	/// How quickly the gain comes back when the signal rises.
	#[range(min = 10.0, max = 2_000.0)]
	#[unit(milliseconds)]
	#[default(150.0)]
	#[id(display = "Release")]
	#[layout(group = "Expander", order = 4, widget = knob)]
	pub release: f32,

	/// Follow the sidechain input instead of the main input.
	#[default(false)]
	#[id(display = "Sidechain")]
	#[layout(group = "Expander", order = 5, widget = toggle)]
	pub sidechain: bool,

	#[skip]
	sample_rate: f32,
	#[skip]
	detector: f32,
	#[skip]
	gain: f32,
}

impl<const CHANNELS: usize> Expander<CHANNELS> {
	/// An expander at a gentle 2:1 with 40 dB of range.
	pub fn new(sample_rate: usize) -> Self {
		Self {
			threshold: -40.0,
			ratio: 2.0,
			range: 40.0,
			attack: 5.0,
			release: 150.0,
			sidechain: false,
			sample_rate: sample_rate as f32,
			detector: 0.0,
			gain: 1.0,
		}
	}
}

impl<const CHANNELS: usize> Effect<CHANNELS> for Expander<CHANNELS> {
	fn delay(&self) -> usize {
		0
	}

	fn process(
		&mut self,
		samples: &mut [f32; CHANNELS],
		other: &[&[f32; CHANNELS]],
		_: &mut Box<dyn ProcessContext>,
	) {
		let frame = if self.sidechain {
			other.get(1).copied().unwrap_or(&*samples)
		} else {
			&*samples
		};
		let release = coefficient(5.0, self.sample_rate);
		self.detector = follow(self.detector, peak(frame), release);
		let level_db = linear_to_db(self.detector);
		let reduction = if level_db < self.threshold {
			(self.threshold - level_db) * (self.ratio.max(1.0) - 1.0)
		} else {
			0.0
		};
		let reduction = reduction.min(self.range).max(0.0);
		let target = db_to_linear(-reduction);
		let attack = coefficient(self.attack, self.sample_rate);
		let release = coefficient(self.release, self.sample_rate);
		self.gain = ballistics(self.gain, target, attack, release);

		for sample in samples.iter_mut() {
			*sample *= self.gain;
			if !sample.is_finite() {
				*sample = 0.0;
			}
		}
	}

	fn clear_tail(&mut self) {
		self.detector = 0.0;
		self.gain = 1.0;
	}
}

/// The 4x polyphase true-peak interpolator from ITU-R BS.1770 Annex 2.
///
/// Twelve taps per phase is the smallest causal realization of the standard's
/// 48-tap, 4x oversampling filter. The phases are stored in the standard's
/// phase-major order and are applied newest-sample-first.
const TRUE_PEAK_TAPS: usize = 12;
const TRUE_PEAK_PHASES: [[f32; TRUE_PEAK_TAPS]; 4] = [
	[
		0.001_708_984_4,
		0.010_986_328,
		-0.019_653_32,
		0.033_203_125,
		-0.059_448_242,
		0.137_329_1,
		0.972_167_97,
		-0.102_294_92,
		0.047_607_422,
		-0.026_611_328,
		0.014_892_578,
		-0.008_300_781,
	],
	[
		-0.029_174_805,
		0.029_296_875,
		-0.051_757_812,
		0.089_111_33,
		-0.166_503_9,
		0.465_087_9,
		0.779_785_16,
		-0.200_317_38,
		0.101_562_5,
		-0.058_227_54,
		0.033_081_055,
		-0.018_920_898,
	],
	[
		-0.018_920_898,
		0.033_081_055,
		-0.058_227_54,
		0.101_562_5,
		-0.200_317_38,
		0.779_785_16,
		0.465_087_9,
		-0.166_503_9,
		0.089_111_33,
		-0.051_757_812,
		0.029_296_875,
		-0.029_174_805,
	],
	[
		-0.008_300_781,
		0.014_892_578,
		-0.026_611_328,
		0.047_607_422,
		-0.102_294_92,
		0.972_167_97,
		0.137_329_1,
		-0.059_448_242,
		0.033_203_125,
		-0.019_653_32,
		0.010_986_328,
		0.001_708_984_4,
	],
];

/// A peak limiter with a lookahead delay: the detector lowers the gain before
/// the transient reaches the output. In True Peak mode the detector runs the
/// input through the 4x, 48-tap interpolator from ITU-R BS.1770 Annex 2, so
/// inter-sample peaks are caught as well as sample peaks.
#[derive(Parameters)]
pub struct Limiter<const CHANNELS: usize = 2> {
	/// The highest level that leaves the limiter.
	#[range(min = -30.0, max = 0.0)]
	#[unit(decibels)]
	#[default(-0.1)]
	#[id(display = "Ceiling")]
	#[layout(group = "Limiter", order = 0, widget = knob, hero = "limiter")]
	pub ceiling: f32,

	/// How quickly the gain returns after a peak.
	#[range(min = 1.0, max = 1_000.0)]
	#[unit(milliseconds)]
	#[default(50.0)]
	#[id(display = "Release")]
	#[layout(group = "Limiter", order = 1, widget = knob)]
	pub release: f32,

	/// How far ahead the detector looks.
	#[range(min = 0.0, max = 5.0)]
	#[unit(milliseconds)]
	#[default(1.5)]
	#[id(display = "Lookahead")]
	#[layout(group = "Limiter", order = 2, widget = knob)]
	pub lookahead: f32,

	/// Gain added after limiting.
	#[range(min = -12.0, max = 12.0)]
	#[unit(decibels)]
	#[default(0.0)]
	#[id(display = "Output")]
	#[layout(group = "Limiter", order = 3, widget = knob)]
	pub output: f32,

	/// Measure inter-sample peaks through the BS.1770 4x interpolator.
	#[default(false)]
	#[id(display = "True Peak")]
	#[layout(group = "Limiter", order = 4, widget = toggle)]
	pub true_peak: bool,

	#[skip]
	sample_rate: f32,
	#[skip]
	detector: f32,
	#[skip]
	detector_hold: usize,
	#[skip]
	gain: f32,
	#[skip]
	delay: [i_am_dsp::tools::ring_buffer::RingBuffer<f32>; CHANNELS],
	#[skip]
	true_peak_history: [[f32; TRUE_PEAK_TAPS]; CHANNELS],
}

impl<const CHANNELS: usize> Limiter<CHANNELS> {
	/// A limiter at -0.1 dB with 1.5 ms of lookahead.
	pub fn new(sample_rate: usize) -> Self {
		let lookahead = (sample_rate as f32 * 0.0015).round() as usize;

		Self {
			ceiling: -0.1,
			release: 50.0,
			lookahead: 1.5,
			output: 0.0,
			true_peak: false,
			sample_rate: sample_rate as f32,
			detector: 0.0,
			detector_hold: 0,
			gain: 1.0,
			delay: core::array::from_fn(|_| {
				i_am_dsp::tools::ring_buffer::RingBuffer::new(lookahead)
			}),
			true_peak_history: core::array::from_fn(|_| [0.0; TRUE_PEAK_TAPS]),
		}
	}

	fn lookahead_samples(&self) -> usize {
		(self.lookahead * 0.001 * self.sample_rate).round() as usize
	}

	/// Keep the interpolator primed even while sample-peak mode is selected, so
	/// turning True Peak on does not start with an empty filter.
	fn feed_true_peak_history(&mut self, samples: &[f32; CHANNELS]) {
		for (channel, sample) in samples.iter().enumerate() {
			let history = &mut self.true_peak_history[channel];
			history.rotate_left(1);
			history[TRUE_PEAK_TAPS - 1] = *sample;
		}
	}

	/// The linked maximum of the four BS.1770 interpolated phases, plus the raw
	/// sample as a lower bound. This is the level the limiter's ceiling sees.
	fn true_peak_level(&mut self, samples: &[f32; CHANNELS]) -> f32 {
		let mut level = 0.0_f32;

		for (channel, sample) in samples.iter().enumerate() {
			let history = &mut self.true_peak_history[channel];
			history.rotate_left(1);
			history[TRUE_PEAK_TAPS - 1] = *sample;
			level = level.max(sample.abs());

			for phase in &TRUE_PEAK_PHASES {
				let mut interpolated = 0.0_f32;

				for (tap, coefficient) in phase.iter().enumerate() {
					interpolated += coefficient * history[TRUE_PEAK_TAPS - 1 - tap];
				}

				level = level.max(interpolated.abs());
			}
		}

		level
	}
}

impl<const CHANNELS: usize> Effect<CHANNELS> for Limiter<CHANNELS> {
	fn delay(&self) -> usize {
		self.lookahead_samples()
	}

	fn process(
		&mut self,
		samples: &mut [f32; CHANNELS],
		_other: &[&[f32; CHANNELS]],
		_: &mut Box<dyn ProcessContext>,
	) {
		let lookahead = self.lookahead_samples();

		if self.delay[0].capacity() != lookahead {
			for buffer in self.delay.iter_mut() {
				buffer.resize(lookahead);
			}
		}

		let level = if self.true_peak {
			self.true_peak_level(samples)
		} else {
			self.feed_true_peak_history(samples);
			peak(samples)
		};

		// Hold the detector's peak for the lookahead, so the gain that was
		// computed when the peak arrived is still down when it reaches the output.
		if level >= self.detector {
			self.detector = level;
			self.detector_hold = lookahead;
		} else if self.detector_hold > 0 {
			self.detector_hold -= 1;
		} else {
			let release = coefficient(5.0, self.sample_rate);
			self.detector = release * self.detector + (1.0 - release) * level;
		}

		let level_db = linear_to_db(self.detector);
		let needed = if level_db > self.ceiling {
			db_to_linear(self.ceiling - level_db)
		} else {
			1.0
		};

		// Attack is instant and the detector is a lookahead ahead of the audio, so
		// the gain is already down when the peak arrives. Only the return is slow.
		if needed < self.gain {
			self.gain = needed;
		} else {
			let release = coefficient(self.release, self.sample_rate);
			self.gain = (release * self.gain + (1.0 - release)).min(needed);
		}

		let output = db_to_linear(self.output);

		if lookahead == 0 {
			for sample in samples.iter_mut() {
				*sample *= self.gain * output;
			}
		} else {
			for (channel, sample) in samples.iter_mut().enumerate() {
				let delayed = self.delay[channel][0];
				self.delay[channel].push(*sample);
				*sample = delayed * self.gain * output;
			}
		}
	}

	fn clear_tail(&mut self) {
		self.detector = 0.0;
		self.detector_hold = 0;
		self.gain = 1.0;

		for buffer in self.delay.iter_mut() {
			buffer.clear();
		}

		self.true_peak_history.fill([0.0; TRUE_PEAK_TAPS]);
	}
}

/// A transient shaper: a fast and a slow peak envelope are compared, and the
/// difference boosts or cuts the attack and the decaying tail separately.
#[derive(Parameters)]
pub struct TransientShaper<const CHANNELS: usize = 2> {
	/// How much the attack is boosted (positive) or softened (negative).
	#[range(min = -1.0, max = 1.0)]
	#[unit(percent)]
	#[default(0.3)]
	#[id(display = "Attack")]
	#[layout(group = "Transient", order = 0, widget = knob, hero = "waveform")]
	pub attack: f32,

	/// How much the tail is boosted (positive) or shortened (negative).
	#[range(min = -1.0, max = 1.0)]
	#[unit(percent)]
	#[default(0.0)]
	#[id(display = "Sustain")]
	#[layout(group = "Transient", order = 1, widget = knob)]
	pub sustain: f32,

	/// Gain after shaping.
	#[range(min = -12.0, max = 12.0)]
	#[unit(decibels)]
	#[default(0.0)]
	#[id(display = "Output")]
	#[layout(group = "Transient", order = 2, widget = knob)]
	pub output: f32,

	/// The dry/wet mix.
	#[range(min = 0.0, max = 1.0)]
	#[unit(percent)]
	#[default(1.0)]
	#[id(display = "Mix")]
	#[layout(group = "Transient", order = 3, widget = knob)]
	pub mix: f32,

	#[skip]
	fast: f32,
	#[skip]
	slow: f32,
	#[skip]
	gain: f32,
	#[skip]
	fast_attack: f32,
	#[skip]
	slow_attack: f32,
	#[skip]
	fast_release: f32,
	#[skip]
	slow_release: f32,
	#[skip]
	smoothing: f32,
}

impl<const CHANNELS: usize> TransientShaper<CHANNELS> {
	/// A transient shaper with a moderate attack boost.
	pub fn new(sample_rate: usize) -> Self {
		let rate = sample_rate as f32;

		Self {
			attack: 0.3,
			sustain: 0.0,
			output: 0.0,
			mix: 1.0,
			fast: 0.0,
			slow: 0.0,
			gain: 1.0,
			fast_attack: coefficient(1.0, rate),
			slow_attack: coefficient(30.0, rate),
			fast_release: coefficient(8.0, rate),
			slow_release: coefficient(160.0, rate),
			smoothing: coefficient(1.0, rate),
		}
	}
}

impl<const CHANNELS: usize> Effect<CHANNELS> for TransientShaper<CHANNELS> {
	fn delay(&self) -> usize {
		0
	}

	fn process(
		&mut self,
		samples: &mut [f32; CHANNELS],
		_other: &[&[f32; CHANNELS]],
		_: &mut Box<dyn ProcessContext>,
	) {
		let dry = *samples;
		let level = peak(&dry);

		// The fast envelope follows the hit, the slow one follows the body.
		self.fast = follow_timed(self.fast, level, self.fast_attack, self.fast_release);
		self.slow = follow_timed(self.slow, level, self.slow_attack, self.slow_release);

		let floor = self.slow.max(1e-6);
		let transient = ((self.fast - self.slow) / floor).clamp(0.0, 4.0);
		let tail = ((self.slow - self.fast) / floor).clamp(0.0, 1.0);
		let gain_db =
			(self.attack * 12.0 * transient + self.sustain * 12.0 * tail).clamp(-60.0, 24.0);
		let target = db_to_linear(gain_db);
		self.gain = self.smoothing * self.gain + (1.0 - self.smoothing) * target;
		let output = db_to_linear(self.output);
		let mix = self.mix.clamp(0.0, 1.0);

		for (sample, dry) in samples.iter_mut().zip(dry.iter()) {
			let value = dry * (1.0 - mix) + *sample * self.gain * output * mix;
			*sample = if value.is_finite() { value } else { 0.0 };
		}
	}

	fn clear_tail(&mut self) {
		self.fast = 0.0;
		self.slow = 0.0;
		self.gain = 1.0;
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn process<const CHANNELS: usize, E: Effect<CHANNELS>>(
		effect: &mut E,
		frames: impl Iterator<Item = [f32; CHANNELS]>,
	) -> Vec<[f32; CHANNELS]> {
		let mut context: Box<dyn ProcessContext> = Box::new(());
		let mut output = Vec::new();

		for mut frame in frames {
			effect.process(&mut frame, &[], &mut context);
			output.push(frame);
		}

		output
	}

	fn sine(amplitude: f32, seconds: f32) -> impl Iterator<Item = [f32; 2]> {
		let frames = (seconds * 48_000.0) as usize;

		(0..frames).map(move |index| {
			let phase = 2.0 * std::f32::consts::PI * 220.0 * index as f32 / 48_000.0;
			let value = amplitude * phase.sin();
			[value, value]
		})
	}

	fn energy<const CHANNELS: usize>(frames: &[[f32; CHANNELS]]) -> f32 {
		frames
			.iter()
			.flat_map(|frame| frame.iter())
			.map(|sample| sample * sample)
			.sum()
	}

	/// A quiet signal is held down, a loud one opens the gate.
	#[test]
	fn the_gate_closes_on_a_quiet_signal() {
		let quiet = process(
			&mut {
				let mut gate = Gate::<2>::new(48_000);
				gate.threshold = -20.0;
				gate.hold = 0.0;
				gate.release = 20.0;
				gate
			},
			sine(0.001, 0.2),
		);
		let loud = process(
			&mut {
				let mut gate = Gate::<2>::new(48_000);
				gate.threshold = -20.0;
				gate.hold = 0.0;
				gate.release = 20.0;
				gate
			},
			sine(0.5, 0.2),
		);

		let quiet_energy = energy(&quiet[2_048..]);
		let loud_energy = energy(&loud[2_048..]);
		assert!(
			quiet_energy < loud_energy * 1e-4,
			"quiet {quiet_energy}, loud {loud_energy}"
		);
	}

	/// An expander leaves the loud part alone and pushes the quiet part down.
	#[test]
	fn the_expander_pushes_a_quiet_signal_down() {
		let render = |amplitude: f32| {
			let mut expander = Expander::<2>::new(48_000);
			expander.threshold = -20.0;
			expander.ratio = 4.0;
			expander.range = 60.0;
			expander.attack = 1.0;
			expander.release = 20.0;
			energy(&process(&mut expander, sine(amplitude, 0.4))[4_096..])
		};

		let quiet = render(0.001);
		let loud = render(0.5);
		assert!(quiet < loud * 1e-4, "quiet {quiet}, loud {loud}");
	}

	/// The ceiling is a hard limit on the output peak.
	#[test]
	fn the_limiter_holds_the_ceiling() {
		let mut limiter = Limiter::<2>::new(48_000);
		limiter.ceiling = -6.0;
		limiter.release = 20.0;
		limiter.lookahead = 1.0;
		let frames = process(&mut limiter, sine(0.9, 0.5));
		let peak = frames[8_000..]
			.iter()
			.flat_map(|frame| frame.iter())
			.fold(0.0_f32, |peak, sample| peak.max(sample.abs()));
		let ceiling = db_to_linear(-6.0);

		assert!(peak <= ceiling * 1.01, "peak {peak}, ceiling {ceiling}");
	}

	/// A 12 kHz sine at 45 degrees has a sample peak of 0.636 but a true peak
	/// of 0.9. Sample-peak limiting therefore misses it; BS.1770 mode must not.
	#[test]
	fn the_true_peak_limiter_catches_an_inter_sample_peak() {
		let render = |true_peak: bool| {
			let mut limiter = Limiter::<1>::new(48_000);
			limiter.ceiling = -3.0;
			limiter.release = 20.0;
			limiter.lookahead = 1.5;
			limiter.true_peak = true_peak;
			let mut context: Box<dyn ProcessContext> = Box::new(());
			let mut output = Vec::new();

			for index in 0..4_800 {
				let phase =
					std::f32::consts::FRAC_PI_2 * index as f32 + std::f32::consts::FRAC_PI_4;
				let mut frame = [0.9 * phase.sin()];
				limiter.process(&mut frame, &[], &mut context);
				output.push(frame[0]);
			}

			output[1_024..]
				.iter()
				.fold(0.0_f32, |peak, sample| peak.max(sample.abs()))
		};

		let sample_peak = render(false);
		let true_peak = render(true);

		assert!(
			true_peak < sample_peak * 0.95,
			"sample {sample_peak}, true {true_peak}"
		);
	}

	/// A transient shaper makes the attack louder or softer, and is finite.
	#[test]
	fn the_transient_shaper_moves_the_attack() {
		let render = |attack: f32| {
			let mut shaper = TransientShaper::<2>::new(48_000);
			shaper.attack = attack;
			let mut context: Box<dyn ProcessContext> = Box::new(());
			let mut output = Vec::new();

			for index in 0..4_096 {
				let value = if (200..260).contains(&index) {
					0.8 * ((index as f32 * 0.3).sin())
				} else {
					0.0
				};
				let mut frame = [value, value];
				shaper.process(&mut frame, &[], &mut context);
				output.push(frame);
			}

			energy(&output[200..500])
		};

		let boosted = render(1.0);
		let cut = render(-1.0);

		assert!(boosted > cut, "boosted {boosted}, cut {cut}");
		assert!(boosted.is_finite() && cut.is_finite());
	}

	/// Both the gate and the expander run in mono.
	#[test]
	fn the_dynamics_are_generic_over_channels() {
		let mut gate = Gate::<1>::new(48_000);
		let mut expander = Expander::<1>::new(48_000);
		let mut limiter = Limiter::<1>::new(48_000);
		let mut shaper = TransientShaper::<1>::new(48_000);
		let mut context: Box<dyn ProcessContext> = Box::new(());

		for _ in 0..1_024 {
			let mut frame = [0.5];
			gate.process(&mut frame, &[], &mut context);
			let mut frame = [0.5];
			expander.process(&mut frame, &[], &mut context);
			let mut frame = [0.5];
			limiter.process(&mut frame, &[], &mut context);
			let mut frame = [0.5];
			shaper.process(&mut frame, &[], &mut context);
		}
	}
}
