//! The classic band vocoder: a 24 dB/oct bandpass bank on the carrier and on
//! the modulator, an envelope follower per band, and the carrier bands shaped by
//! those envelopes. The modulator is the sidechain when one is connected, and the
//! main input otherwise, which makes the self-vocoded "Vocodex" sound.

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::Arc;

use i_am_dsp_derive::Parameters;

use crate::{Effect, ProcessContext, effects::filter::Biquad};

/// The most bands the bank can hold.
pub const MAX_BANDS: usize = 128;
/// The bottom of the band bank.
pub const MIN_HZ: f32 = 80.0;
/// The top of the band bank.
pub const MAX_HZ: f32 = 12_000.0;

/// One band: two carrier stages and two modulator stages, plus the envelope.
struct Band {
	carrier: [Biquad<2>; 2],
	modulator: [Biquad<2>; 2],
	envelope: f32,
}

impl Band {
	fn new(sample_rate: usize) -> Self {
		Self {
			carrier: [Biquad::new(sample_rate), Biquad::new(sample_rate)],
			modulator: [Biquad::new(sample_rate), Biquad::new(sample_rate)],
			envelope: 0.0,
		}
	}

	fn clear(&mut self) {
		for filter in &mut self.carrier {
			filter.clear_state();
		}
		for filter in &mut self.modulator {
			filter.clear_state();
		}
		self.envelope = 0.0;
	}
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
pub struct VocoderEffect {
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
	bands_state: Vec<Band>,
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

impl VocoderEffect {
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

			if index == 0 {
				for filter in &mut band.carrier {
					filter.set_to_lowpass(upper, Biquad::<2>::Q1);
				}
				for filter in &mut band.modulator {
					filter.set_to_lowpass(upper, Biquad::<2>::Q1);
				}
			} else if index + 1 == bands {
				for filter in &mut band.carrier {
					filter.set_to_highpass(lower, Biquad::<2>::Q1);
				}
				for filter in &mut band.modulator {
					filter.set_to_highpass(lower, Biquad::<2>::Q1);
				}
			} else {
				let center = (lower * upper).sqrt();
				let width = center / q * bandwidth;

				for filter in &mut band.carrier {
					filter.set_to_bandpass(center, width);
				}
				for filter in &mut band.modulator {
					filter.set_to_bandpass(center, width);
				}
			}

			band.clear();
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

impl Effect<2> for VocoderEffect {
	fn delay(&self) -> usize {
		0
	}

	fn process(
		&mut self,
		samples: &mut [f32; 2],
		other: &[&[f32; 2]],
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
		let modulator = (modulator_frame[0] + modulator_frame[1]) * 0.5;
		let output_gain = 10.0_f32.powf(self.output / 20.0);
		let mix = self.mix.clamp(0.0, 1.0);
		let bands = self.active;
		let meters = self.meters.clone();

		self.meter_counter += 1;
		let write_meters = self.meter_counter >= 32;

		if write_meters {
			self.meter_counter = 0;
		}

		let mut wet = [0.0_f32; 2];

		for band in 0..bands {
			let state = &mut self.bands_state[band];

			let mut carrier = [dry[0], dry[1]];
			carrier = state.carrier[0].process_sample(carrier);
			carrier = state.carrier[1].process_sample(carrier);

			let mut modulator_band = state.modulator[0].process_sample([modulator, modulator]);
			modulator_band = state.modulator[1].process_sample(modulator_band);
			let level = modulator_band[0].abs();

			let coefficient = if level > state.envelope {
				self.attack_coefficient
			} else {
				self.release_coefficient
			};
			state.envelope = coefficient * state.envelope + (1.0 - coefficient) * level;

			let envelope = state.envelope;
			let out_left = carrier[0] * envelope;
			let out_right = carrier[1] * envelope;

			wet[0] += out_left;
			wet[1] += out_right;

			if write_meters {
				meters.set(band, envelope, (out_left.abs() + out_right.abs()) * 0.5);
			}
		}

		wet[0] *= output_gain;
		wet[1] *= output_gain;

		for channel in 0..2 {
			let sample = dry[channel] * (1.0 - mix) + wet[channel] * mix;
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
