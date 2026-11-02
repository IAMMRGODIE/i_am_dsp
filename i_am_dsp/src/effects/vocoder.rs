//! The classic band vocoder: a 24 dB/oct bandpass bank on the carrier and on
//! the modulator, an envelope follower per band, and the carrier bands shaped by
//! those envelopes. The modulator is the sidechain when one is connected, and the
//! main input otherwise, which makes the self-vocoded "Vocodex" sound.

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::Arc;

use i_am_dsp_derive::Parameters;

use rustfft::{Fft, FftPlanner, num_complex::Complex};

use crate::{
	Effect, ProcessContext,
	effects::{filter::Biquad, phase_vocoder::overlap_add_gain},
	tools::ring_buffer::RingBuffer,
};

/// The most bands the bank can hold.
pub const MAX_BANDS: usize = 128;
/// The bottom of the band bank.
pub const MIN_HZ: f32 = 20.0;
/// The top of the band bank.
pub const MAX_HZ: f32 = 20_000.0;

/// One band: two carrier stages and two modulator stages, plus the envelope.
struct Band<const CHANNELS: usize> {
	carrier: [Biquad<CHANNELS>; 2],
	modulator: [Biquad<CHANNELS>; 2],
	envelope: f32,
}

impl<const CHANNELS: usize> Band<CHANNELS> {
	fn new(sample_rate: usize) -> Self {
		Self {
			carrier: [Biquad::new(sample_rate), Biquad::new(sample_rate)],
			modulator: [Biquad::new(sample_rate), Biquad::new(sample_rate)],
			envelope: 0.0,
		}
	}
}

/// The mean of a frame, so a mono detector sees a stereo frame as one signal.
fn mono<const CHANNELS: usize>(frame: &[f32; CHANNELS]) -> f32 {
	frame.iter().sum::<f32>() / CHANNELS as f32
}

/// The band levels the page draws, one pair per band.
pub struct VocoderMeters {
	input: [AtomicU32; MAX_BANDS],
	output: [AtomicU32; MAX_BANDS],
	bands: AtomicUsize,
}

impl Default for VocoderMeters {
	fn default() -> Self {
		Self {
			input: std::array::from_fn(|_| AtomicU32::new(0)),
			output: std::array::from_fn(|_| AtomicU32::new(0)),
			bands: AtomicUsize::new(24),
		}
	}
}

impl VocoderMeters {
	fn set_bands(&self, bands: usize) {
		self.bands.store(bands.min(MAX_BANDS), Ordering::Relaxed);

		for meter in &self.input[bands.min(MAX_BANDS)..] {
			meter.store(0, Ordering::Relaxed);
		}
		for meter in &self.output[bands.min(MAX_BANDS)..] {
			meter.store(0, Ordering::Relaxed);
		}
	}

	fn set(&self, band: usize, input: f32, output: f32) {
		if band >= MAX_BANDS {
			return;
		}

		self.input[band].store(input.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
		self.output[band].store(output.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
	}

	/// Copies the current levels into the slices, returning the band count.
	pub fn read(&self, input: &mut [f32], output: &mut [f32]) -> usize {
		let bands = self.bands.load(Ordering::Relaxed).min(MAX_BANDS);

		for band in 0..bands.min(input.len()).min(output.len()) {
			input[band] = f32::from_bits(self.input[band].load(Ordering::Relaxed));
			output[band] = f32::from_bits(self.output[band].load(Ordering::Relaxed));
		}

		bands
	}

	/// The current pair of arrays, for the page route.
	pub fn snapshot(&self) -> (usize, Vec<f32>, Vec<f32>) {
		let bands = self.bands.load(Ordering::Relaxed).min(MAX_BANDS);
		let input = (0..bands)
			.map(|band| f32::from_bits(self.input[band].load(Ordering::Relaxed)))
			.collect();
		let output = (0..bands)
			.map(|band| f32::from_bits(self.output[band].load(Ordering::Relaxed)))
			.collect();

		(bands, input, output)
	}
}

/// The vocoder's DSP.
#[derive(Parameters)]
pub struct VocoderEffect<const CHANNELS: usize = 2> {
	/// How many bands the bank has, from 4 to 128.
	#[range(min = 4, max = 128)]
	#[default(24)]
	#[id(display = "Bands")]
	#[layout(group = "Vocoder", order = 0, widget = knob, step = 1.0, hero = "vocodex")]
	pub bands: i32,

	/// The width of the middle bands, as a multiple of their spacing.
	#[range(min = 0.25, max = 4.0)]
	#[unit(percent)]
	#[default(1.0)]
	#[id(display = "Bandwidth")]
	#[layout(group = "Vocoder", order = 1, widget = knob)]
	pub bandwidth: f32,

	/// The envelope attack.
	#[range(min = 1.0, max = 200.0)]
	#[unit(milliseconds)]
	#[default(5.0)]
	#[id(display = "Attack")]
	#[layout(group = "Vocoder", order = 2, widget = knob)]
	pub attack: f32,

	/// The envelope release.
	#[range(min = 10.0, max = 1_000.0)]
	#[unit(milliseconds)]
	#[default(80.0)]
	#[id(display = "Release")]
	#[layout(group = "Vocoder", order = 3, widget = knob)]
	pub release: f32,

	/// The wet/dry mix.
	#[range(min = 0.0, max = 1.0)]
	#[unit(percent)]
	#[default(1.0)]
	#[id(display = "Mix")]
	#[layout(group = "Vocoder", order = 4, widget = knob)]
	pub mix: f32,

	/// The output gain.
	#[range(min = -24.0, max = 24.0)]
	#[unit(db)]
	#[default(0.0)]
	#[id(display = "Output")]
	#[layout(group = "Vocoder", order = 5, widget = knob)]
	pub output: f32,

	/// Use the sidechain as the modulator; off, the effect vocodes the input
	/// with itself.
	#[default(false)]
	#[id(display = "Sidechain")]
	#[layout(group = "Vocoder", order = 6, widget = toggle)]
	pub sidechain: bool,

	#[skip]
	bands_state: Vec<Band<CHANNELS>>,
	#[skip]
	active: usize,
	#[skip]
	sample_rate: f32,
	#[skip]
	attack_coefficient: f32,
	#[skip]
	release_coefficient: f32,
	#[skip]
	last_attack: f32,
	#[skip]
	last_release: f32,
	#[skip]
	last_bandwidth: f32,
	#[skip]
	meters: Arc<VocoderMeters>,
	#[skip]
	meter_counter: usize,
}

impl<const CHANNELS: usize> VocoderEffect<CHANNELS> {
	/// A 24-band bank at the host's rate.
	pub fn new(sample_rate: usize, meters: Arc<VocoderMeters>) -> Self {
		let mut effect = Self {
			bands: 24,
			bandwidth: 1.0,
			attack: 5.0,
			release: 80.0,
			mix: 1.0,
			output: 0.0,
			sidechain: false,
			bands_state: (0..MAX_BANDS).map(|_| Band::new(sample_rate)).collect(),
			active: 0,
			sample_rate: sample_rate as f32,
			attack_coefficient: 0.0,
			release_coefficient: 0.0,
			last_attack: 5.0,
			last_release: 80.0,
			last_bandwidth: 1.0,
			meters,
			meter_counter: 0,
		};

		effect.update_bands();
		effect.update_times();
		effect
	}

	fn update_bands(&mut self) {
		let bands = self.bands.clamp(4, 128) as usize;
		let bandwidth = self.bandwidth.clamp(0.25, 4.0);

		if bands == self.active && (bandwidth - self.last_bandwidth).abs() < 1e-4 {
			return;
		}

		self.active = bands;
		self.last_bandwidth = bandwidth;
		let ratio = MAX_HZ / MIN_HZ;
		let octaves = ratio.log2() / bands as f32;
		let q = (1.0 / (2.0_f32.powf(octaves) - 1.0)).clamp(0.6, 24.0);

		for (index, band) in self.bands_state.iter_mut().enumerate() {
			if index >= bands {
				break;
			}

			let lower = MIN_HZ * ratio.powf(index as f32 / bands as f32);
			let upper = MIN_HZ * ratio.powf((index + 1) as f32 / bands as f32);
			let center = (lower * upper).sqrt();
			let width = center / q * bandwidth;

			for filter in &mut band.carrier {
				filter.set_to_bandpass(center, width);
			}
			for filter in &mut band.modulator {
				filter.set_to_bandpass(center, width);
			}
		}

		self.meters.set_bands(bands);
		self.meter_counter = 0;
	}

	fn update_times(&mut self) {
		let fs = self.sample_rate;
		self.attack_coefficient = (-1.0 / (self.attack.max(0.1) * 0.001 * fs)).exp();
		self.release_coefficient = (-1.0 / (self.release.max(1.0) * 0.001 * fs)).exp();
		self.last_attack = self.attack;
		self.last_release = self.release;
	}
}

impl<const CHANNELS: usize> Effect<CHANNELS> for VocoderEffect<CHANNELS> {
	fn band_meters(&self, input: &mut [f32], output: &mut [f32]) -> usize {
		self.meters.read(input, output)
	}

	fn delay(&self) -> usize {
		0
	}

	fn process(
		&mut self,
		samples: &mut [f32; CHANNELS],
		other: &[&[f32; CHANNELS]],
		_: &mut Box<dyn ProcessContext>,
	) {
		// Both the band count and the bandwidth change the coefficients.
		self.update_bands();

		if self.attack != self.last_attack || self.release != self.last_release {
			self.update_times();
		}

		let dry = *samples;
		let modulator_frame = if self.sidechain {
			other.get(1).copied().unwrap_or(&dry)
		} else {
			&dry
		};
		let modulator = mono(modulator_frame);
		let output_gain = 10.0_f32.powf(self.output / 20.0);
		let mix = self.mix.clamp(0.0, 1.0);
		let bands = self.active;
		let meters = self.meters.clone();

		self.meter_counter += 1;
		let write_meters = self.meter_counter >= 32;

		if write_meters {
			self.meter_counter = 0;
		}

		let mut wet = [0.0_f32; CHANNELS];

		for band in 0..bands {
			let state = &mut self.bands_state[band];

			let mut carrier = dry;
			carrier = state.carrier[0].process_sample(carrier);
			carrier = state.carrier[1].process_sample(carrier);

			let mut modulator_band =
				state.modulator[0].process_sample([modulator; CHANNELS]);
			modulator_band = state.modulator[1].process_sample(modulator_band);
			let level = modulator_band
				.iter()
				.fold(0.0_f32, |loudest, sample| loudest.max(sample.abs()));

			let coefficient = if level > state.envelope {
				self.attack_coefficient
			} else {
				self.release_coefficient
			};
			state.envelope = coefficient * state.envelope + (1.0 - coefficient) * level;

			let envelope = state.envelope;

			for (channel, sample) in wet.iter_mut().enumerate() {
				*sample += carrier[channel] * envelope;
			}

			if write_meters {
				let level = carrier
					.iter()
					.map(|sample| sample.abs())
					.sum::<f32>()
					/ CHANNELS as f32;
				meters.set(band, envelope, level * envelope);
			}
		}

		for channel in 0..CHANNELS {
			let sample = dry[channel] * (1.0 - mix) + wet[channel] * output_gain * mix;
			samples[channel] = if sample.is_finite() { sample } else { 0.0 };
		}
	}
}


/// The STFT vocoder: two analyses (carrier and modulator) and one synthesis,
/// with the modulator's spectral shape imposed on the carrier's magnitudes and
/// the carrier's phase kept for the inverse transform.
#[derive(Parameters)]
pub struct StftVocoderEffect<const CHANNELS: usize = 2> {
	/// The FFT window size.
	#[choices("512", "1024", "2048", "4096")]
	#[default(2)]
	#[id(display = "Window Size")]
	#[layout(group = "STFT Vocoder", order = 0, widget = choice, hero = "spectra")]
	pub window_size: i32,

	/// The wet/dry mix.
	#[range(min = 0.0, max = 1.0)]
	#[unit(percent)]
	#[default(1.0)]
	#[id(display = "Mix")]
	#[layout(group = "STFT Vocoder", order = 1, widget = knob)]
	pub mix: f32,

	/// The output gain.
	#[range(min = -24.0, max = 24.0)]
	#[unit(db)]
	#[default(0.0)]
	#[id(display = "Output")]
	#[layout(group = "STFT Vocoder", order = 2, widget = knob)]
	pub output: f32,

	/// Use the sidechain as the modulator; off, the effect vocodes the input
	/// with itself.
	#[default(false)]
	#[id(display = "Sidechain")]
	#[layout(group = "STFT Vocoder", order = 3, widget = toggle)]
	pub sidechain: bool,

	#[skip]
	window: usize,
	#[skip]
	hop: usize,
	#[skip]
	fft: Arc<dyn Fft<f32>>,
	#[skip]
	ifft: Arc<dyn Fft<f32>>,
	#[skip]
	window_table: Vec<f32>,
	#[skip]
	gain: f32,
	#[skip]
	carrier_input: [RingBuffer<f32>; CHANNELS],
	#[skip]
	modulator_input: RingBuffer<f32>,
	#[skip]
	output_buffer: [RingBuffer<f32>; CHANNELS],
	#[skip]
	carrier_spectrum: [Vec<Complex<f32>>; CHANNELS],
	#[skip]
	modulator_spectrum: Vec<Complex<f32>>,
	#[skip]
	output_spectrum: [Vec<Complex<f32>>; CHANNELS],
	#[skip]
	input_count: usize,
	#[skip]
	output_count: usize,
	#[skip]
	last_window: usize,
}

impl<const CHANNELS: usize> StftVocoderEffect<CHANNELS> {
	/// An STFT vocoder with a 2048 sample window.
	pub fn new(_sample_rate: usize) -> Self {
		let mut effect = Self {
			window_size: 2,
			mix: 1.0,
			output: 0.0,
			sidechain: false,
			window: 0,
			hop: 0,
			fft: FftPlanner::new().plan_fft_forward(2),
			ifft: FftPlanner::new().plan_fft_inverse(2),
			window_table: Vec::new(),
			gain: 1.0,
			carrier_input: core::array::from_fn(|_| RingBuffer::new(2)),
			modulator_input: RingBuffer::new(2),
			output_buffer: core::array::from_fn(|_| RingBuffer::new(2)),
			carrier_spectrum: core::array::from_fn(|_| vec![Complex::ZERO; 2]),
			modulator_spectrum: vec![Complex::ZERO; 2],
			output_spectrum: core::array::from_fn(|_| vec![Complex::ZERO; 2]),
			input_count: 0,
			output_count: 0,
			last_window: 0,
		};

		effect.renew(2_048);
		effect
	}

	fn renew(&mut self, window: usize) {
		let window = window.clamp(512, 4_096);
		let hop = window / 4;
		let mut planner = FftPlanner::new();
		self.window = window;
		self.hop = hop;
		self.fft = planner.plan_fft_forward(window);
		self.ifft = planner.plan_fft_inverse(window);
		self.window_table = (0..window)
			.map(|index| {
				let phase = 2.0 * std::f32::consts::PI * index as f32 / window as f32;
				0.5 * (0.6 - 0.4 * phase.cos())
			})
			.collect();
		self.gain = 1.0 / overlap_add_gain(0.6);
		self.carrier_input = core::array::from_fn(|_| RingBuffer::new(window));
		self.modulator_input = RingBuffer::new(window);
		self.output_buffer = core::array::from_fn(|_| RingBuffer::new(window));
		self.carrier_spectrum = core::array::from_fn(|_| vec![Complex::ZERO; window]);
		self.modulator_spectrum = vec![Complex::ZERO; window];
		self.output_spectrum = core::array::from_fn(|_| vec![Complex::ZERO; window]);
		self.input_count = 0;
		self.output_count = 0;
		self.last_window = window;
	}

	fn process_frame(&mut self) {
		let window = self.window;
		let fft = self.fft.clone();
		let ifft = self.ifft.clone();

		for channel in 0..CHANNELS {
			for index in 0..window {
				self.carrier_spectrum[channel][index] =
					Complex::new(self.window_table[index] * self.carrier_input[channel][index], 0.0);
			}
			fft.process(&mut self.carrier_spectrum[channel]);
		}

		for index in 0..window {
			self.modulator_spectrum[index] =
				Complex::new(self.window_table[index] * self.modulator_input[index], 0.0);
		}
		fft.process(&mut self.modulator_spectrum);

		// Normalise the modulator by its loudest bin, not its mean: a sine
		// modulator has almost all of its energy in one bin, and dividing by the
		// mean would make that bin tens of times louder than the carrier.
		let peak = (1..window / 2)
			.map(|bin| self.modulator_spectrum[bin].norm())
			.fold(0.0_f32, f32::max);
		let denominator = peak + 1e-9;
		let modulator = &self.modulator_spectrum;
		let gain = self.gain;
		let window_table = &self.window_table;

		for channel in 0..CHANNELS {
			let carrier = &self.carrier_spectrum[channel];
			let output = &mut self.output_spectrum[channel];

			output[0] = carrier[0];
			for bin in 1..window / 2 {
				let scale = modulator[bin].norm() / denominator;
				output[bin] = Complex::from_polar(carrier[bin].norm() * scale, carrier[bin].arg());
			}
			let nyquist = window / 2;
			let scale = modulator[nyquist].norm() / denominator;
			output[nyquist] = Complex::new(carrier[nyquist].re * scale, 0.0);
			for bin in 1..window / 2 {
				output[window - bin] = output[bin].conj();
			}

			ifft.process(output);

			for index in 0..window {
				self.output_buffer[channel][index] +=
					output[index].re * window_table[index] / window as f32 * gain;
			}
		}
	}
}

impl<const CHANNELS: usize> Effect<CHANNELS> for StftVocoderEffect<CHANNELS> {
	fn delay(&self) -> usize {
		self.window / 2
	}

	fn process(
		&mut self,
		samples: &mut [f32; CHANNELS],
		other: &[&[f32; CHANNELS]],
		_: &mut Box<dyn ProcessContext>,
	) {
		let size = [512, 1_024, 2_048, 4_096]
			[self.window_size.clamp(0, 3) as usize];

		if size != self.last_window {
			self.renew(size);
		}

		let dry = *samples;
		let modulator_frame = if self.sidechain {
			other.get(1).copied().unwrap_or(&dry)
		} else {
			&dry
		};
		let modulator = mono(modulator_frame);

		for (channel, sample) in samples.iter_mut().enumerate() {
			self.carrier_input[channel].push(*sample);
			*sample = self.output_buffer[channel][self.output_count];
		}
		self.modulator_input.push(modulator);

		self.output_count = (self.output_count + 1) % self.window;
		self.input_count += 1;

		if self.input_count >= self.hop {
			for buffer in &mut self.output_buffer {
				buffer.extend_defaults(self.hop);
			}
			self.input_count -= self.hop;
			self.output_count = (self.output_count + self.window - self.hop) % self.window;
			self.process_frame();
		}

		let output_gain = 10.0_f32.powf(self.output / 20.0);
		let mix = self.mix.clamp(0.0, 1.0);

		for channel in 0..CHANNELS {
			let sample = dry[channel] * (1.0 - mix) + samples[channel] * output_gain * mix;
			samples[channel] = if sample.is_finite() { sample } else { 0.0 };
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn render(effect: &mut VocoderEffect, frames: usize, sidechain: Option<f32>) -> [f32; 2] {
		let mut context: Box<dyn ProcessContext> = Box::new(());
		let mut energy = [0.0; 2];

		for frame in 0..frames {
			let t = frame as f32 / 48_000.0;
			let input = 0.5 * (2.0 * std::f32::consts::PI * 220.0 * t).sin();
			let main = [input, input];
			let mut samples = main;

			if let Some(level) = sidechain {
				let value = level * (2.0 * std::f32::consts::PI * 330.0 * t).sin();
				let side = [value, value];
				effect.process(&mut samples, &[&main, &side], &mut context);
			} else {
				effect.process(&mut samples, &[], &mut context);
			}

			energy[0] += samples[0] * samples[0];
			energy[1] += samples[1] * samples[1];
		}

		energy
	}

	#[test]
	fn every_band_count_is_finite() {
		for bands in [4, 8, 24, 64, 128] {
			let meters = Arc::new(VocoderMeters::default());
			let mut effect = VocoderEffect::new(48_000, meters);
			effect.bands = bands;
			let energy = render(&mut effect, 1_024, None);

			assert!(energy.iter().all(|value| value.is_finite()), "{bands}");
		}
	}

	#[test]
	fn a_sidechain_changes_the_envelopes() {
		let meters = Arc::new(VocoderMeters::default());
		let mut self_vocoded = VocoderEffect::new(48_000, meters.clone());
		self_vocoded.sidechain = false;
		self_vocoded.mix = 1.0;
		let self_energy = render(&mut self_vocoded, 2_048, None);

		let meters = Arc::new(VocoderMeters::default());
		let mut sidechained = VocoderEffect::new(48_000, meters);
		sidechained.sidechain = true;
		sidechained.mix = 1.0;
		let side_energy = render(&mut sidechained, 2_048, Some(0.5));

		assert!(
			(self_energy[0] - side_energy[0]).abs() > 1e-4,
			"{self_energy:?} vs {side_energy:?}"
		);
	}

	#[test]
	fn the_bandwidth_control_keeps_the_bank_finite() {
		let meters = Arc::new(VocoderMeters::default());
		let mut effect = VocoderEffect::new(48_000, meters);

		for bandwidth in [0.25, 0.5, 1.0, 2.0, 4.0] {
			effect.bandwidth = bandwidth;
			let energy = render(&mut effect, 1_024, None);
			assert!(energy.iter().all(|value| value.is_finite()), "{bandwidth}");
		}
	}

	#[test]
	fn bandwidth_moves_the_filter_coefficients() {
		let meters = Arc::new(VocoderMeters::default());
		let mut effect = VocoderEffect::new(48_000, meters);
		let mut context: Box<dyn ProcessContext> = Box::new(());
		let mut frame = [0.0; 2];
		effect.process(&mut frame, &[], &mut context);

		let before = effect.bands_state[12].carrier[0].b;
		effect.bandwidth = 2.0;
		effect.process(&mut frame, &[], &mut context);
		let after = effect.bands_state[12].carrier[0].b;

		assert_ne!(before, after, "bandwidth did not redesign the filters");
	}

	#[test]
	fn the_edge_bands_keep_the_high_frequencies() {
		let meters = Arc::new(VocoderMeters::default());
		let mut effect = VocoderEffect::new(48_000, meters);
		effect.bands = 24;
		effect.sidechain = true;
		let mut context: Box<dyn ProcessContext> = Box::new(());
		let mut energy = 0.0;

		for frame in 0..4_096 {
			let t = frame as f32 / 48_000.0;
			let input = 0.5 * (2.0 * std::f32::consts::PI * 15_000.0 * t).sin();
			let main = [input, input];
			let mut samples = main;
			effect.process(&mut samples, &[&main, &main], &mut context);
			energy += samples[0] * samples[0];
		}

		assert!(energy > 1e-6, "15 kHz was lost: {energy}");
	}

	fn stft_energy(effect: &mut StftVocoderEffect, frames: usize, sidechain: Option<f32>) -> f32 {
		let mut context: Box<dyn ProcessContext> = Box::new(());
		let mut energy = 0.0;

		for frame in 0..frames {
			let t = frame as f32 / 48_000.0;
			let input = 0.5 * (2.0 * std::f32::consts::PI * 220.0 * t).sin();
			let main = [input, input];
			let mut samples = main;

			if let Some(level) = sidechain {
				let value = level * (2.0 * std::f32::consts::PI * 330.0 * t).sin();
				let side = [value, value];
				effect.process(&mut samples, &[&main, &side], &mut context);
			} else {
				effect.process(&mut samples, &[], &mut context);
			}

			energy += samples[0] * samples[0] + samples[1] * samples[1];
		}

		energy
	}

	#[test]
	fn the_stft_vocoder_is_finite_at_every_window() {
		for window in 0..4 {
			let mut effect = StftVocoderEffect::new(48_000);
			effect.window_size = window;
			let energy = stft_energy(&mut effect, 8_192, None);

			assert!(energy.is_finite(), "{window}: {energy}");
		}
	}

	#[test]
	fn the_stft_vocoder_uses_the_sidechain() {
		let mut effect = StftVocoderEffect::new(48_000);
		effect.window_size = 2;
		effect.sidechain = true;
		let energy = stft_energy(&mut effect, 8_192, Some(0.5));

		assert!(energy > 1e-9, "the sidechained STFT was silent: {energy}");
	}

	/// Both vocoders are generic over their channel count, so a mono instance
	/// processes on one channel without reaching for a second one.
	#[test]
	fn both_vocoders_process_mono() {
		let meters = Arc::new(VocoderMeters::default());
		let mut band: VocoderEffect<1> = VocoderEffect::new(48_000, meters);
		let mut stft: StftVocoderEffect<1> = StftVocoderEffect::new(48_000);
		let mut context: Box<dyn ProcessContext> = Box::new(());
		let mut band_energy = 0.0;
		let mut stft_energy = 0.0;

		for frame in 0..8_192 {
			let t = frame as f32 / 48_000.0;
			let input = 0.5 * (2.0 * std::f32::consts::PI * 220.0 * t).sin();
			let mut samples = [input];

			band.process(&mut samples, &[], &mut context);
			band_energy += samples[0] * samples[0];

			let mut samples = [input];
			stft.process(&mut samples, &[], &mut context);
			stft_energy += samples[0] * samples[0];
		}

		assert!(band_energy.is_finite(), "{band_energy}");
		assert!(stft_energy.is_finite(), "{stft_energy}");
		assert!(stft_energy > 1e-9, "the mono STFT was silent: {stft_energy}");
	}

	#[test]
	#[ignore = "performance check"]
	fn the_128_band_bank_processes_a_second() {
		let meters = Arc::new(VocoderMeters::default());
		let mut effect = VocoderEffect::new(48_000, meters);
		effect.bands = 128;
		let mut context: Box<dyn ProcessContext> = Box::new(());
		let started = std::time::Instant::now();

		for frame in 0..48_000 {
			let t = frame as f32 / 48_000.0;
			let input = 0.5 * (2.0 * std::f32::consts::PI * 220.0 * t).sin();
			let mut samples = [input, input];
			effect.process(&mut samples, &[], &mut context);
		}

		println!("128 bands, one second: {:?}", started.elapsed());
	}
}
