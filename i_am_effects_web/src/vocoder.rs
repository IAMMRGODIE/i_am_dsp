//! The classic band vocoder: a 24 dB/oct bandpass bank on the carrier and on
//! the modulator, an envelope follower per band, and the carrier bands shaped by
//! those envelopes. The modulator is the sidechain when one is connected, and the
//! main input otherwise, which makes the self-vocoded "Vocodex" sound.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::Arc;

use i_am_dsp::prelude::{Effect, ParamMap, Parameter, ProcessContext, SetValue};
use i_am_dsp_derive::Parameters;
use i_am_effects_core::processor::{EffectMessage, EffectProcessor};
use i_am_plugin::{
	AspectRatioStrategy, Descriptor, Editor, GuiResizeHints, Plugin, WindowOptions,
};
use i_am_web_gui::{ProtocolRequest, ProtocolResponse, Scope};

/// The most bands the bank can hold.
const MAX_BANDS: usize = 128;
/// The bottom of the band bank.
const MIN_HZ: f32 = 80.0;
/// The top of the band bank.
const MAX_HZ: f32 = 12_000.0;

/// A second-order bandpass, shared by both stages of a 24 dB/oct band.
#[derive(Clone, Copy, Default)]
struct Biquad {
	b0: f32,
	b1: f32,
	b2: f32,
	a1: f32,
	a2: f32,
}

#[derive(Clone, Copy, Default)]
struct BiquadState {
	z1: f32,
	z2: f32,
}

impl Biquad {
	fn bandpass(sample_rate: f32, center: f32, q: f32) -> Self {
		let center = center.clamp(20.0, sample_rate * 0.45);
		let omega = 2.0 * std::f32::consts::PI * center / sample_rate;
		let cos = omega.cos();
		let alpha = omega.sin() / (2.0 * q.max(0.1));
		let a0 = 1.0 + alpha;

		Self {
			b0: alpha / a0,
			b1: 0.0,
			b2: -alpha / a0,
			a1: -2.0 * cos / a0,
			a2: (1.0 - alpha) / a0,
		}
	}

	/// Transposed direct form II: two state values instead of four.
	fn process(&self, state: &mut BiquadState, input: f32) -> f32 {
		let output = self.b0 * input + state.z1;
		state.z1 = self.b1 * input - self.a1 * output + state.z2;
		state.z2 = self.b2 * input - self.a2 * output;
		output
	}
}

#[derive(Clone, Copy, Default)]
struct BandState {
	carrier_left: [BiquadState; 2],
	carrier_right: [BiquadState; 2],
	modulator: [BiquadState; 2],
	envelope: f32,
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

	fn snapshot(&self) -> (usize, Vec<f32>, Vec<f32>) {
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

	/// The envelope attack.
	#[range(min = 1.0, max = 200.0)]
	#[unit(milliseconds)]
	#[default(5.0)]
	#[id(display = "Attack")]
	#[layout(group = "Vocoder", order = 1, widget = knob)]
	pub attack: f32,

	/// The envelope release.
	#[range(min = 10.0, max = 1_000.0)]
	#[unit(milliseconds)]
	#[default(80.0)]
	#[id(display = "Release")]
	#[layout(group = "Vocoder", order = 2, widget = knob)]
	pub release: f32,

	/// The wet/dry mix.
	#[range(min = 0.0, max = 1.0)]
	#[unit(percent)]
	#[default(1.0)]
	#[id(display = "Mix")]
	#[layout(group = "Vocoder", order = 3, widget = knob)]
	pub mix: f32,

	/// The output gain.
	#[range(min = -24.0, max = 24.0)]
	#[unit(db)]
	#[default(0.0)]
	#[id(display = "Output")]
	#[layout(group = "Vocoder", order = 4, widget = knob)]
	pub output: f32,

	/// Use the sidechain as the modulator; off, the effect vocodes the input
	/// with itself.
	#[default(false)]
	#[id(display = "Sidechain")]
	#[layout(group = "Vocoder", order = 5, widget = toggle)]
	pub sidechain: bool,

	#[skip]
	coefficients: [Biquad; MAX_BANDS],
	#[skip]
	band_states: [BandState; MAX_BANDS],
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
	meters: Arc<VocoderMeters>,
	#[skip]
	meter_counter: usize,
}

impl VocoderEffect {
	/// A 24-band bank at the host's rate.
	pub fn new(sample_rate: usize, meters: Arc<VocoderMeters>) -> Self {
		let mut effect = Self {
			bands: 24,
			attack: 5.0,
			release: 80.0,
			mix: 1.0,
			output: 0.0,
			sidechain: false,
			coefficients: [Biquad::default(); MAX_BANDS],
			band_states: [BandState::default(); MAX_BANDS],
			active: 0,
			sample_rate: sample_rate as f32,
			attack_coefficient: 0.0,
			release_coefficient: 0.0,
			last_attack: 5.0,
			last_release: 80.0,
			meters,
			meter_counter: 0,
		};

		effect.update_bands();
		effect.update_times();
		effect
	}

	fn update_bands(&mut self) {
		let bands = self.bands.clamp(4, 128) as usize;

		if bands == self.active {
			return;
		}

		self.active = bands;
		let octaves = (MAX_HZ / MIN_HZ).log2() / bands as f32;
		let q = (1.0 / (2.0_f32.powf(octaves) - 1.0)).clamp(0.6, 24.0);

		for (index, coefficient) in self.coefficients.iter_mut().enumerate().take(bands) {
			let t = (index as f32 + 0.5) / bands as f32;
			let center = MIN_HZ * (MAX_HZ / MIN_HZ).powf(t);
			*coefficient = Biquad::bandpass(self.sample_rate, center, q);
		}

		self.band_states = [BandState::default(); MAX_BANDS];
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
		if self.bands.clamp(4, 128) as usize != self.active {
			self.update_bands();
		}

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
			let coefficient = self.coefficients[band];
			let state = &mut self.band_states[band];

			let mut left = coefficient.process(&mut state.carrier_left[0], dry[0]);
			left = coefficient.process(&mut state.carrier_left[1], left);
			let mut right = coefficient.process(&mut state.carrier_right[0], dry[1]);
			right = coefficient.process(&mut state.carrier_right[1], right);

			let mut modulator_band = coefficient.process(&mut state.modulator[0], modulator);
			modulator_band = coefficient.process(&mut state.modulator[1], modulator_band);

			let level = modulator_band.abs();
			let coefficient_time = if level > state.envelope {
				self.attack_coefficient
			} else {
				self.release_coefficient
			};
			state.envelope =
				coefficient_time * state.envelope + (1.0 - coefficient_time) * level;

			let envelope = state.envelope;
			let out_left = left * envelope;
			let out_right = right * envelope;

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

fn route(request: &ProtocolRequest<'_>, meters: &VocoderMeters) -> Option<ProtocolResponse> {
	if request.method != "GET" || request.path != "/bands" {
		return None;
	}

	let (bands, input, output) = meters.snapshot();
	let mut json = format!("{{\"bands\":{bands},\"input\":[");

	for (index, value) in input.iter().enumerate() {
		if index > 0 {
			json.push(',');
		}
		let _ = std::fmt::Write::write_fmt(&mut json, format_args!("{value:.4}"));
	}

	json.push_str("],\"output\":[");

	for (index, value) in output.iter().enumerate() {
		if index > 0 {
			json.push(',');
		}
		let _ = std::fmt::Write::write_fmt(&mut json, format_args!("{value:.4}"));
	}

	json.push_str("]}");
	Some(ProtocolResponse::json(json))
}

/// The product.
pub struct VocoderWeb {
	processor: EffectProcessor<VocoderEffect>,
	bypass: Arc<AtomicBool>,
	scope: Arc<Scope>,
	meters: Arc<VocoderMeters>,
}

impl Default for VocoderWeb {
	fn default() -> Self {
		let meters = Arc::new(VocoderMeters::default());
		let build_meters = meters.clone();

		Self {
			processor: EffectProcessor::with_build(move |rate| {
				VocoderEffect::new(rate, build_meters.clone())
			}),
			bypass: Arc::new(AtomicBool::new(false)),
			scope: Arc::new(Scope::new()),
			meters,
		}
	}
}

impl i_am_dsp::prelude::Parameters for VocoderWeb {
	fn get_parameters(&self) -> Vec<Parameter> {
		self.processor.parameters()
	}

	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		self.processor.set_parameter(identifier, value)
	}
}

impl i_am_dsp::Processor for VocoderWeb {
	type Message = EffectMessage;

	fn delay(&self) -> usize {
		i_am_dsp::Processor::delay(&self.processor)
	}

	fn process(
		&mut self,
		samples: &mut [f32; 2],
		other: &[[f32; 2]],
		process_context: &mut Box<dyn ProcessContext>,
	) {
		let input = *samples;
		let input_mono = (input[0] + input[1]) * 0.5;
		let sidechain = other
			.get(1)
			.map(|frame| (frame[0] + frame[1]) * 0.5)
			.unwrap_or(input_mono);

		if self.bypass.load(Ordering::Relaxed) {
			self.scope.push_stereo(input, input, sidechain);
			return;
		}

		i_am_dsp::Processor::process(&mut self.processor, samples, other, process_context);

		let output = *samples;
		self.scope.push_stereo(input, output, sidechain);
	}

	fn on_message(&self, message: Self::Message) {
		i_am_dsp::Processor::on_message(&self.processor, message);
	}
}

impl Plugin for VocoderWeb {
	const DESCRIPTOR: Descriptor =
		Descriptor::new("iamdsp.effects.vocoder.web", "I Am Vocoder")
			.with_vendor("iamplugins")
			.with_version(env!("CARGO_PKG_VERSION"));

	const HAS_GUI: bool = true;

	const INPUT_PORTS: &'static [i_am_plugin::AudioPort] = &[
		i_am_plugin::AudioPort { name: "Main" },
		i_am_plugin::AudioPort { name: "Sidechain" },
	];

	fn new() -> Self {
		Self::default()
	}

	fn window_options() -> WindowOptions {
		const FACE_FRACTION: f64 = 0.85;

		WindowOptions::new()
			.with_size((700.0 / FACE_FRACTION, 380.0 / FACE_FRACTION))
			.with_title("I Am Vocoder")
			.with_resize_hints(GuiResizeHints {
				can_resize_horizontally: true,
				can_resize_vertically: true,
				strategy: AspectRatioStrategy::Disregard,
			})
	}

	fn param_map(&self) -> ParamMap {
		self.processor.param_map()
	}

	fn sync_params(&mut self) {
		self.processor.sync_params();
	}

	fn on_activate(&mut self, sample_rate: usize) {
		self.processor.activate(sample_rate);
	}

	fn editor(&self) -> Option<Box<dyn Editor<Self>>> {
		let face = i_am_web_gui::Faceplate::new("I Am Vocoder", "#e0a45a", (700.0, 380.0));
		let meters = self.meters.clone();

		i_am_web_gui::editor_with(
			self,
			face,
			self.bypass.clone(),
			self.scope.clone(),
			self.processor.sample_rate(),
			move |request: &ProtocolRequest<'_>| route(request, &meters),
			|_: &str| false,
		)
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

	#[test]
	fn the_band_route_reports_a_level_per_band() {
		let meters = Arc::new(VocoderMeters::default());
		let mut effect = VocoderEffect::new(48_000, meters.clone());
		effect.bands = 8;
		let _ = render(&mut effect, 2_048, Some(0.5));

		let request = ProtocolRequest {
			method: "GET",
			path: "/bands",
			query: None,
			body: &[],
		};
		let response = route(&request, &meters).expect("the bands route");
		let body = String::from_utf8(response.body).expect("utf-8");

		assert!(body.contains("\"bands\":8"), "{body}");
		assert!(body.contains("\"input\""), "{body}");
	}
}
