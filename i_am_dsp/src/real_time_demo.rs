//! This is a simple real-time audio processing demo.
//!
//! To start with, see [`DspDemo`]

use crate::{
	prelude::*,
	tools::{ring_buffer::RingBuffer, ui_tools::gain_ui},
};

#[cfg(feature = "string")]
use crate::generators::string_engine::StringInstrument;
use std::{
	collections::HashMap,
	sync::{Arc, Mutex},
};

#[cfg(feature = "standalone")]
use std::collections::HashSet;

#[cfg(feature = "standalone")]
use cpal::{
	StreamConfig,
	traits::{DeviceTrait, HostTrait, StreamTrait},
};
#[cfg(feature = "standalone")]
use crossbeam_channel::{Receiver, TryRecvError};

#[cfg(feature = "standalone")]
use egui::Key;

use crate::{Effect, Generator};

type GneratorConstructor = Box<dyn (Fn(usize) -> Box<dyn Generator>) + Send + Sync>;
type EffectConstructor = Box<dyn (Fn(usize) -> Box<dyn Effect>) + Send + Sync>;

const CHANGE_HISTORY_LEN: usize = 64;

// #[cfg(feature = "standalone")]
mod context;
mod menus;

pub use context::SimpleContext;

#[cfg(feature = "standalone")]
use self::menus::simulate_midi;

lazy_static::lazy_static! {
	/// The List of available Generators
	///
	/// You can also add your own Generator by adding a tuple to this list.
	pub static ref GENERATOR_LIST: Vec<(String, GneratorConstructor)> = {
		let mut list: Vec<(String, GneratorConstructor)> = vec![
			(
				"Resampler".to_string(),
				Box::new(|sample_rate| {
					let sampler = Sampler::new(sample_rate);
					Box::new(Adsr::new(sampler, EqualTemperament, sample_rate)) as Box<dyn Generator>
				})
			),
			(
				"Player".to_string(),
				Box::new(|sample_rate| {
					Box::new(Sampler::new(sample_rate)) as Box<dyn Generator>
				})
			),
			(
				"Basic Shapes".to_string(),
				Box::new(|sample_rate| {
					let tables: Vec<Box<dyn WaveTable + Send + Sync>> = vec![
						Box::new(SineWave),
						Box::new(TriangleWave),
						Box::new(SawWave),
						Box::new(SquareWave),
					];
					let smoother = WaveTableSmoother::new(tables, 0.0);
					Box::new(Adsr::new(smoother, EqualTemperament, sample_rate)) as Box<dyn Generator>
				})
			),
			(
				"BilinearSmoother".to_string(),
				Box::new(|sample_rate| {
					let smoother = BilinearSmoother::new(SineWave, TriangleWave, SawWave, SquareWave);
					Box::new(Adsr::new(smoother, EqualTemperament, sample_rate)) as Box<dyn Generator>
				})
			),
			(
				"Noise Wave".to_string(),
				Box::new(|sample_rate| Box::new(Adsr::new(TableOsc(NoiseWave), EqualTemperament, sample_rate)) as Box<dyn Generator>)
			),
			(
				"Additive(Saw Bend)".to_string(),
				Box::new(|sample_rate| {
					// Enough partials for a saw to reach the top of the audible range in the bass
					let gen_freq = BendedSawGen::default();
					let add: AdditiveOsc<BendedSawGen, 256> = AdditiveOsc::new(gen_freq, 256);
					Box::new(Adsr::new(add, EqualTemperament, sample_rate)) as Box<dyn Generator>
				})
			),
			(
				"Stereo Canvas".to_string(),
				Box::new(|sample_rate| {
					let canvas = MultiStereoGenerators::new();
					Box::new(Adsr::new(canvas, EqualTemperament, sample_rate)) as Box<dyn Generator>
				})
			),
			(
				"Plucked String".to_string(),
				Box::new(|sample_rate| {
					let tables: Vec<Box<dyn WaveTable + Send + Sync>> = vec![
						Box::new(-1.0),
						Box::new(1.0),
						Box::new(SineWave),
						Box::new(TriangleWave),
						Box::new(SawWave),
						Box::new(SquareWave),
					];
					let smoother = WaveTableSmoother::new(tables, 0.0);

					let fill_tables: Vec<Box<dyn WaveTable + Send + Sync>> = vec![
						Box::new(SineWave),
						Box::new(TriangleWave),
						Box::new(SawWave),
						Box::new(SquareWave),
						Box::new(NoiseWave),
					];

					let fill_smoother = WaveTableSmoother::new(fill_tables, 0.0);

					let plucked = SimplePluck::new(sample_rate, fill_smoother, smoother, 1000.0, 1000.0);
					Box::new(Adsr::new(plucked, EqualTemperament, sample_rate)) as Box<dyn Generator>
				})
			)
		];

		#[cfg(feature = "string")]
		{
			list.push((
				"Modelled Strings".to_string(),
				Box::new(|sample_rate| {
					Box::new(StringInstrument::new(sample_rate)) as Box<dyn Generator>
				})
			));
		}

		#[cfg(feature = "rhai")]
		{
			list.push((
				"Additive(Rhai)".to_string(),
				Box::new(|sample_rate| {
					let rhai_gen = RhaiFreqGen::new(sample_rate);
					let add: AdditiveOsc<RhaiFreqGen> = AdditiveOsc::new(rhai_gen, 64);
					Box::new(Adsr::new(add, EqualTemperament, sample_rate)) as Box<dyn Generator>
				})
			));
		}

		list.sort_by(|a, b| a.0.cmp(&b.0));

		list
	};

	/// The List of available Effects
	///
	/// You can also add your own Effect by adding a tuple to this list.
	pub static ref EFFECT_LIST: Vec<(String, Vec<(String, EffectConstructor)>)> = {
		let mut filter_effects: Vec<(String, EffectConstructor)> = vec![
			(
				"Biquad".to_string(),
				Box::new(|sample_rate| Box::new(Biquad::new(sample_rate)) as Box<dyn Effect>)
			),
			(
				"Disperser".to_string(),
				Box::new(|sample_rate| Box::new(Disperser::new(sample_rate)) as Box<dyn Effect>)
			),
			(
				"Lowpass".to_string(),
				Box::new(|sample_rate| Box::new(Lowpass::new(sample_rate, 1000.0, Biquad::<2>::Q1)) as Box<dyn Effect>)
			),
			(
				"Highpass".to_string(),
				Box::new(|sample_rate| Box::new(Highpass::new(sample_rate, 1000.0, Biquad::<2>::Q1)) as Box<dyn Effect>)
			),
			(
				"Bandpass".to_string(),
				Box::new(|sample_rate| Box::new(Bandpass::new(sample_rate, 1000.0, 200.0)) as Box<dyn Effect>)
			),
			(
				"Bandstop".to_string(),
				Box::new(|sample_rate| Box::new(Bandstop::new(sample_rate, 1000.0, 200.0)) as Box<dyn Effect>)
			),
			(
				"Peak".to_string(),
				Box::new(|sample_rate| Box::new(Peak::new(sample_rate, 1000.0, -3.01, 200.0)) as Box<dyn Effect>)
			),
			(
				"HighShelf".to_string(),
				Box::new(|sample_rate| Box::new(HighShelf::new(sample_rate, 1000.0, -3.01, 1.0)) as Box<dyn Effect>)
			),
			(
				"LowShelf".to_string(),
				Box::new(|sample_rate| Box::new(LowShelf::new(sample_rate, 1000.0, -3.01, 1.0)) as Box<dyn Effect>)
			)
		];

		filter_effects.sort_by(|a, b| a.0.cmp(&b.0));

		let mut convolver_effects: Vec<(String, EffectConstructor)> = vec![
			(
				"Convolver".to_string(),
				Box::new(|_| {
					let hilbert_transform = hilbert_transform::<2>(511);
					let output = Convolver::new(hilbert_transform, &DelyaCaculateMode::Fir);
					Box::new(output) as Box<dyn Effect>
				})
			),
			(
				"FftConvolver".to_string(),
				Box::new(|sample_rate| {
					let id = convolve_identity::<2>(8192);
					let output = FftConvolver::<2, 256>::new(id, sample_rate);
					Box::new(output) as Box<dyn Effect>
				})
			),
			(
				"Reverb".to_string(),
				Box::new(|sample_rate| {
					let effect: Reverb<Biquad<2>, 8, 2> = Reverb::new(
						Biquad::new(sample_rate),
						sample_rate,
						Default::default(),
						10,
						10.0
					);
					Box::new(effect) as Box<dyn Effect>
				})
			),
			(
				"Reverb(IR-Based)".to_string(),
				Box::new(|sample_rate| {
					let ir_gen = RayTracing::rectangular_room(
						10.0,
						10.0,
						10.0,
						0.005,
						1.0,
						sample_rate
					);
					let reverb = IrBasedReverb::new(ir_gen, sample_rate);
					Box::new(reverb) as Box<dyn Effect>
				})
			),
			(
				"Hybrid Reverb".to_string(),
				Box::new(|sample_rate| {
					let ir_gen = RayTracing::rectangular_room(
						10.0,
						10.0,
						10.0,
						0.005,
						1.0,
						sample_rate
					);
					let reverb = HybridReverb::<2, 8>::new(ir_gen, sample_rate, 1.0, 1.0);
					Box::new(reverb) as Box<dyn Effect>
				})
			)
		];

		convolver_effects.sort_by(|a, b| a.0.cmp(&b.0));

		let mut other_effects: Vec<(String, EffectConstructor)> = vec![
			(
				"IIR Frequency Shifter".to_string(),
				Box::new(|sample_rate| {
					let effect = IIRFreqShifter::<10>::new(sample_rate, 0.0);
					Box::new(effect) as Box<dyn Effect>
				})
			),
			(
				"FIR Frequency Shifter".to_string(),
				Box::new(|sample_rate| {
					let effect = FIRFreqShifter::new(sample_rate, 0.0);
					Box::new(effect) as Box<dyn Effect>
				})
			),
			(
				"FFT Frequency Shifter".to_string(),
				Box::new(|sample_rate| {
					let effect = PhaseVocoder::new(FreqShift::default(), 1024, sample_rate);
					Box::new(effect) as Box<dyn Effect>
				})
			),
			(
				"Stereo Controller ".to_string(),
				Box::new(|_| Box::new(StereoController::default()) as Box<dyn Effect>)
			),
			(
				"Gain".to_string(),
				Box::new(|_| Box::new(Gain::default()) as Box<dyn Effect>)
			),
			(
				"Phaser".to_string(),
				Box::new(|sample_rate| {
					let effect = Phaser::new(SineWave, 10, sample_rate);
					Box::new(effect) as Box<dyn Effect>
				})
			),
			(
				"LrController".to_string(),
				Box::new(|_| Box::new(LrControl::default()) as Box<dyn Effect>)
			),
			(
				"Pitch Shifter(WSOLA)".to_string(),
				Box::new(|_| Box::new(PitchShifter::default()) as Box<dyn Effect>)
			),
			(
				"Pitch Shifter(Phase Vocoder)".to_string(),
				Box::new(|sample_rate| Box::new(PhaseVocoder::new(PitchShift::default(), 1024, sample_rate)) as Box<dyn Effect>)
			),
			(
				"Particle Wider".to_string(),
				Box::new(|sample_rate| Box::new(ParticleWider::new(sample_rate)) as Box<dyn Effect>)
			),
			(
				"Unstablizer".to_string(),
				Box::new(|sample_rate| Box::new(Unstablizer::new(sample_rate)) as Box<dyn Effect>)
			),
		];

		other_effects.sort_by(|a, b| a.0.cmp(&b.0));

		let mut compresser_effects: Vec<(String, EffectConstructor)> = vec![
			(
				"Compressor(IIR Hilbert Transform)".to_string(),
				Box::new(|sample_rate| {
					let env = IIRHilbertEnvelope::<10>::new(sample_rate);
					let env = EnvelopeWithHistory::new(env, 32768);
					let effect = Compressor::new(env, sample_rate, 46.0, 200.0, -3.0, 5.0);
					Box::new(effect) as Box<dyn Effect>
				})
			),
			(
				"Compressor(FIR Hilbert Transform)".to_string(),
				Box::new(|sample_rate| {
					let env = FIRHilbertEnvelope::new(127);
					let env = EnvelopeWithHistory::new(env, 32768);
					let effect = Compressor::new(env, sample_rate, 46.0, 200.0, -3.0, 5.0);
					Box::new(effect) as Box<dyn Effect>
				})
			),
			(
				"Compressor(Rectify Envelope)".to_string(),
				Box::new(|sample_rate| {
					let env = RectifyEnvelope::new(sample_rate, 1000.0);
					let env = EnvelopeWithHistory::new(env, 32768);
					let effect = Compressor::new(env, sample_rate, 46.0, 200.0, -3.0, 5.0);
					Box::new(effect) as Box<dyn Effect>
				})
			),
			(
				"Compressor(Peak Detector)".to_string(),
				Box::new(|sample_rate| {
					let env = PeakDetector::new(sample_rate, 46.0, 200.0);
					let env = EnvelopeWithHistory::new(env, 32768);
					let effect = Compressor::new(env, sample_rate, 46.0, 200.0, -3.0, 5.0);
					Box::new(effect) as Box<dyn Effect>
				})
			),
			(
				"Compressor(TKEO Envelope)".to_string(),
				Box::new(|sample_rate| {
					let env = TkeoEnvelope::new();
					let env = EnvelopeWithHistory::new(env, 32768);
					let effect = Compressor::new(env, sample_rate, 46.0, 200.0, -3.0, 5.0);
					Box::new(effect) as Box<dyn Effect>
				})
			),
		];

		compresser_effects.sort_by(|a, b| a.0.cmp(&b.0));

		let mut distortion_effects: Vec<(String, EffectConstructor)> = vec![
			(
				"Hard Clipper".to_string(),
				Box::new(|_| Box::new(HardClipper::new(0.8, 1.0)) as Box<dyn Effect>)
			),
			(
				"Soft Clipper".to_string(),
				Box::new(|_| Box::new(SoftClipper::new(0.8, 1.0)) as Box<dyn Effect>)
			),
			(
				"Overdrive".to_string(),
				Box::new(|_| Box::new(Overdrive::new(1.0, 1.0)) as Box<dyn Effect>)
			),
			(
				"Bit Crusher".to_string(),
				Box::new(|_| Box::new(BitCrusher::new(131072, 1.0)) as Box<dyn Effect>)
			),
			(
				"Saturator".to_string(),
				Box::new(|_| Box::new(Saturator::new(5.0, 2.0, 1.0)) as Box<dyn Effect>)
			),
			(
				"Downsampler".to_string(),
				Box::new(|sample_rate| Box::new(Downsampler::new(sample_rate)) as Box<dyn Effect>)
			),
			(
				"Crusher".to_string(),
				Box::new(|sample_rate| Box::new(Crusher::new(sample_rate)) as Box<dyn Effect>)
			)
		];

		distortion_effects.sort_by(|a, b| a.0.cmp(&b.0));

		let mut delay_effects: Vec<(String, EffectConstructor)> = vec![
			(
				"Delay".to_string(),
				Box::new(|sample_rate| {
					let delay = Delay::new((), 65536, 50.0, sample_rate);
					Box::new(delay) as Box<dyn Effect>
				})
			),
			(
				"Pure Delay".to_string(),
				Box::new(|sample_rate| {
					let delay = PureDelay::new(65536, 50.0, sample_rate);
					Box::new(delay) as Box<dyn Effect>
				})
			),
			(
				"Flanger".to_string(),
				Box::new(|sample_rate| {
					let effect = Flanger::new(8192, sample_rate, SineWave);
					Box::new(effect) as Box<dyn Effect>
				})
			),
			(
				"Chorus".to_string(),
				Box::new(|sample_rate| {
					let effect = Chorus::new(8192, 10, sample_rate, SineWave);
					Box::new(effect) as Box<dyn Effect>
				})
			),
		];

		delay_effects.sort_by(|a, b| a.0.cmp(&b.0));

		let mut visual_effects: Vec<(String, EffectConstructor)> = vec![
			(
				"Waveform".to_string(),
				Box::new(|_| {
					let effect = Waveform::<2>::new(32768);
					Box::new(effect) as Box<dyn Effect>
				})
			),
		];

		visual_effects.sort_by(|a, b| a.0.cmp(&b.0));

		vec![
			("Compressor".to_string(), compresser_effects),
			("Delay".to_string(), delay_effects),
			("Distortion".to_string(), distortion_effects),
			// ("Envelope".to_string(), envelope_effects),
			("Filter".to_string(), filter_effects),
			("Reverb".to_string(), convolver_effects),
			("Other".to_string(), other_effects),
			("Visual".to_string(), visual_effects),
			// ("Vocoder".to_string(), vocoder_effects),
		]
	};
}

/// The DspDemo struct
///
/// Saves the state of the DSP demo and provides methods to update and draw the UI.
pub struct DspDemo {
	#[cfg(feature = "standalone")]
	_stream: cpal::Stream,
	#[cfg(feature = "standalone")]
	error_receiver: Receiver<String>,
	current_error: Option<String>,

	shared_data: Arc<Mutex<SharedData>>,
	sample_rate: usize,
	#[cfg(feature = "standalone")]
	key_holding: HashSet<usize>,
	#[cfg(feature = "standalone")]
	note_shift: isize,

	changed_history: RingBuffer<Option<ListenedValue>>,
	current_track_effect: usize,
}

impl serde::Serialize for DspDemo {
	fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
	where
		S: serde::Serializer,
	{
		use serde::ser::Error;
		let data = match self.persist_data() {
			Ok(data) => data,
			Err(e) => {
				return std::result::Result::Err(S::Error::custom(format!(
					"Failed to serialize DspDemo: {}",
					e
				)));
			}
		};
		data.serialize(serializer)
	}
}

impl<'de> serde::Deserialize<'de> for DspDemo {
	fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
	where
		D: serde::Deserializer<'de>,
	{
		use serde::de::Error;
		let data: Vec<PersistData> = Vec::deserialize(deserializer)?;
		let mut demo = match DspDemo::new(Some(48000)) {
			Ok(demo) => demo,
			Err(e) => {
				return std::result::Result::Err(D::Error::custom(format!(
					"Failed to create DspDemo: {}",
					e
				)));
			}
		};
		match demo.load_persist_data(data) {
			Ok(_) => Ok(demo),
			Err(e) => std::result::Result::Err(D::Error::custom(format!(
				"Failed to load persist data: {}",
				e
			))),
		}
	}
}

mod data;

use data::*;

mod ui;

#[cfg(feature = "standalone")]
impl eframe::App for DspDemo {
	fn ui(&mut self, ui: &mut egui::Ui, _: &mut eframe::Frame) {
		egui::CentralPanel::default().show(ui, |ui| {
			self.ui(ui);
		});

		ui.ctx().request_repaint();
	}
}

#[cfg(test)]
mod tests;
