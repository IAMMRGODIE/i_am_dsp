//! This is a simple real-time audio processing demo.
//! 
//! To start with, see [`DspDemo`]

use crate::{prelude::*, tools::{ring_buffer::RingBuffer, ui_tools::gain_ui}};
use std::{collections::HashMap, sync::{Arc, Mutex}};

#[cfg(feature = "standalone")]
use std::collections::HashSet;


use anyhow::{anyhow, Result};
#[cfg(feature = "standalone")]
use cpal::{traits::{DeviceTrait, HostTrait, StreamTrait}, StreamConfig};
#[cfg(feature = "standalone")]
use crossbeam_channel::{Receiver, TryRecvError};
use egui::{vec2, Grid, Slider};

#[cfg(feature = "standalone")]
use egui::Key;

use crate::{Effect, Generator};

type GneratorConstructor = Box<dyn (Fn(usize) -> Box<dyn Generator>) + Send + Sync>;
type EffectConstructor = Box<dyn (Fn(usize) -> Box<dyn Effect>) + Send + Sync>;

const CHANGE_HISTORY_LEN: usize = 64;

// #[cfg(feature = "standalone")]
/// A simple context that storage all necessary information for processing.
pub struct SimpleContext {
	/// The process infos
	pub info: ProcessInfos,
	/// The midi events
	pub midi_events: Vec<NoteEvent>,
}

impl ProcessContext for SimpleContext {
	fn infos(&self) -> &ProcessInfos {
		&self.info
	}

	fn next_event(&mut self) -> Option<NoteEvent> {
		self.midi_events.pop()
	}

	fn send_event(&mut self, event: NoteEvent) {
		self.midi_events.push(event);
	}

	fn events(&self) -> &[NoteEvent] {
		&self.midi_events
	}
}

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
				"Noise Wave".to_string(),
				Box::new(|sample_rate| Box::new(Adsr::new(NoiseWave, EqualTemperament, sample_rate)) as Box<dyn Generator>)
			),
			(
				"Additive(Saw Bend)".to_string(),
				Box::new(|sample_rate| {
					let gen_freq = BendedSawGen::default();
					let add: AdditiveOsc<BendedSawGen> = AdditiveOsc::new(gen_freq, 64);
					Box::new(Adsr::new(add, EqualTemperament, sample_rate)) as Box<dyn Generator>
				})
			)
		];

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
				"IIR Hilbert Transform".to_string(),
				Box::new(|sample_rate| Box::new(HilbertTransform::<8>::new(sample_rate)) as Box<dyn Effect>)
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
			)
		];

		convolver_effects.sort_by(|a, b| a.0.cmp(&b.0));

		let mut other_effects: Vec<(String, EffectConstructor)> = vec![
			(
				"IIR Frequency Shifter".to_string(),
				Box::new(|sample_rate| {
					let effect = IIRFreqShifter::<8>::new(sample_rate, 0.0);
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
		];

		other_effects.sort_by(|a, b| a.0.cmp(&b.0));

		let mut compresser_effects: Vec<(String, EffectConstructor)> = vec![
			(
				"Compressor(IIR Hilbert Transform)".to_string(),
				Box::new(|sample_rate| {
					let env = IIRHilbertEnvelope::<8>::new(sample_rate);
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
			)
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
			("Convolver".to_string(), convolver_effects),
			("Delay/Reverb".to_string(), delay_effects),
			("Distortion".to_string(), distortion_effects),
			// ("Envelope".to_string(), envelope_effects),
			("Filter".to_string(), filter_effects), 
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
	key_holding: HashSet<u8>,
	#[cfg(feature = "standalone")]
	note_shift: i8,

	changed_history: RingBuffer<Option<ListenedValue>>,
	current_track_effect: usize,
}

impl serde::Serialize for DspDemo {
	fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
	where
		S: serde::Serializer 
	{
		use serde::ser::Error;
		let data = match self.persist_data() {
			Ok(data) => data,
			Err(e) => {
				return std::result::Result::Err(S::Error::custom(format!("Failed to serialize DspDemo: {}", e)));
			}
		};
		data.serialize(serializer)	
	}
}

impl<'de> serde::Deserialize<'de> for DspDemo {
	fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
	where
		D: serde::Deserializer<'de> 
	{
		use serde::de::Error;
		let data: Vec<PersistData> = Vec::deserialize(deserializer)?;
		let mut demo = match DspDemo::new(Some(48000)) {
			Ok(demo) => demo,
			Err(e) => {
				return std::result::Result::Err(D::Error::custom(format!("Failed to create DspDemo: {}", e)));
			}
		};
		match demo.load_persist_data(data) {
			Ok(_) => Ok(demo),
			Err(e) => {
				std::result::Result::Err(D::Error::custom(format!("Failed to load persist data: {}", e)))
			}
		}
	}
}

/// The listender for parameter changes.
pub struct ListenedValue {
	history: ChangedValue,
	range: Option<(FloatOrInt, FloatOrInt, bool)>,
}

impl ListenedValue {
	/// Get the change history of the parameter.
	pub fn changed_value(&self) -> &ChangedValue {
		&self.history
	}

	/// Get the range of changed parameter, will return None if parameter is non-numeric.
	pub fn range(&self) -> Option<(FloatOrInt, FloatOrInt)> {
		if let Some((min, max, _)) = &self.range {
			Some((*min, *max))
		} else {
			None
		}
	}

	/// Returns true if the parameter is logarithmic, false if parameter is non-numeric or linear.
	pub fn is_logarithmic(&self) -> bool {
		if let Some((_, _, log)) = &self.range {
			*log
		} else {
			false
		}
	}
}

/// The Change history that emits by the UI returns to the host.
#[derive(Debug, Clone, PartialEq)]
pub enum ChangedValue {
	/// The Ui has changed the value of a generator parameter.
	Generator {
		/// the index of the generator
		index: usize,
		/// the parameter identifier, same as [`Parameter`] trait.
		identifier: String,
		/// the new value of the parameter
		change_to: SetValue,
		// /// is the parameter controlled by an LFO?
		// is_lfo_controlling: bool,
	},
	/// The Ui has changed the value of an effect parameter.
	Effect {
		/// the track id of the effect
		track: usize,
		/// the index of the effect
		index: usize,
		/// the parameter identifier, same as [`Parameter`] trait.
		identifier: String,
		/// the new value of the parameter
		change_to: SetValue,
		// /// is the parameter controlled by an LFO?
		// is_lfo_controlling: bool,
	}
}

impl ChangedValue {
	fn is_param_same(&self, other: &Self) -> bool {
		match (self, other) {
			(ChangedValue::Generator { 
				index, 
				identifier, 
				.. 
			}, ChangedValue::Generator { 
				index: other_index, 
				identifier: other_identifier, 
				.. 
			}) => {
				*index == *other_index && identifier == other_identifier
			},
			(ChangedValue::Effect { 
				track, 
				index, 
				identifier, 
				.. 
			}, ChangedValue::Effect { 
				track: other_track, 
				index: other_index, 
				identifier: other_identifier,
				.. 
			}) => {
				*track == *other_track && *index == *other_index && identifier == other_identifier
			},
			_ => false,
		}
	}
}

// last for master
const MAX_OUTPUT_TRACKS: usize = 4;

struct GeneratorWarpper {
	path: String,
	generator: Box<dyn Generator>,
	output_track: usize,
	gain: f32,
}

struct EffectWarpper {
	path: (String, String),
	effect: Box<dyn Effect>,
	mix: f32,
}

struct SharedData {
	generators: Vec<GeneratorWarpper>,
	effects: [Vec<EffectWarpper>; MAX_OUTPUT_TRACKS + 1],
	#[cfg(feature = "standalone")]
	ctx: Option<SimpleContext>,
	lfos: Vec<LfoedParams>,
	sample_rate: usize,
	current_phase: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
/// A helper to storage the f32/i32 value of a parameter.
pub enum FloatOrInt {
	/// A f32 value.
	Float(f32),
	/// An i32 value.
	Int(i32),
}

impl FloatOrInt {
	fn as_binary(&self) -> Vec<u8> {
		let mut binary = vec![];
		match self {
			FloatOrInt::Float(value) => {
				binary.push(1u8);
				binary.extend_from_slice(&value.to_le_bytes());
			},
			FloatOrInt::Int(value) => {
				binary.push(0u8);
				binary.extend_from_slice(&value.to_le_bytes());
			}
		}
		binary
	}

	fn from_binary(binary: Vec<u8>) -> Self {
		if binary[0] == 1u8 {
			let mut bytes = [0u8; 4];
			bytes.copy_from_slice(&binary[1..]);
			FloatOrInt::Float(f32::from_le_bytes(bytes))
		}else {
			let mut bytes = [0u8; 4];
			bytes.copy_from_slice(&binary[1..]);
			FloatOrInt::Int(i32::from_le_bytes(bytes))
		}
	}
}

struct LfoedParams {
	lfo: EditableLfo,
	params: Vec<LfoedParam>,
	phase: f32,
	trig: bool,
}

struct LfoedParam {
	// none for generator
	track: Option<usize>,
	index: usize,
	identifier: String,
	min: FloatOrInt,
	max: FloatOrInt,
	start: FloatOrInt,
	end: FloatOrInt,
	logarithmic: bool,
}

impl LfoedParam {
	fn as_binary(&self) -> Vec<u8> {
		let mut binary = vec![];
		if let Some(track) = self.track {
			binary.push(1u8);
			binary.extend_from_slice(&(track as u64).to_le_bytes());
		}else {
			binary.push(0u8);
		}
		binary.extend_from_slice(&(self.index as u64).to_le_bytes());
		let len = self.identifier.len() as u64;
		binary.extend_from_slice(&len.to_le_bytes());
		binary.extend_from_slice(self.identifier.as_bytes());
		binary.extend_from_slice(&self.min.as_binary());
		binary.extend_from_slice(&self.max.as_binary());
		binary.extend_from_slice(&self.start.as_binary());
		binary.extend_from_slice(&self.end.as_binary());
		binary.push(if self.logarithmic { 1u8 } else { 0u8 });

		binary
	}

	fn from_binary(mut binary: Vec<u8>) -> Self {
		let mut track = None;
		if binary[0] == 1u8 {
			let current = binary.split_off(5);
			let mut bytes = [0u8; 8];
			bytes.copy_from_slice(&binary[1..]);
			track = Some(usize::from_le_bytes(bytes));
			binary = current;
		}else {
			let current = binary.split_off(1);
			binary = current;
		}

		let mut index = [0u8; 8];
		index.copy_from_slice(&binary[0..8]);
		let index = u64::from_le_bytes(index) as usize;
		let identifier_len = u64::from_le_bytes(binary[8..16].try_into().unwrap()) as usize;
		let identifier = String::from_utf8_lossy(&binary[16..16 + identifier_len])
			.to_string();
		let current = binary.split_off(16 + identifier_len);
		binary = current;
		let mut current = binary.split_off(5);
		let min = FloatOrInt::from_binary(binary);
		let mut current_2 = current.split_off(5);
		let max = FloatOrInt::from_binary(current);
		let mut current_3 = current_2.split_off(5);
		let start = FloatOrInt::from_binary(current_2);
		let current_4 = current_3.split_off(5);
		let end = FloatOrInt::from_binary(current_3);
		let logarithmic = current_4[0] != 0u8;

		LfoedParam {
			track,
			index,
			identifier,
			min,
			max,
			start,
			end,
			logarithmic,
		}
	}
}

/// A helper to storage the value of a Generator/Effect/Lfo data.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq)]
pub enum PersistData {
	/// The data of a generator.
	Generator {
		/// The path where we should create the generator.
		path: String,
		/// The output track of the generator.
		output_track: usize,
		/// The gain of the generator.
		gain: f32,
		/// The parameters of the generator.
		params: Vec<Parameter>
	},
	/// The data of an effect.
	Effect {
		/// The path where we should create the effect.
		path: (String, String),
		/// The mix of the effect.
		mix: f32,
		/// The parameters of the effect.
		params: Vec<Parameter>,
		/// The number of track of the effect.
		track: usize,
		/// The index of the effect in the track.
		index: usize
	},
	/// The data of an LFO.
	Lfo {
		/// The parameters of the LFO.
		params: Vec<Parameter>,
		/// The lfoed parameters, in binary format.
		lfoed_params: Vec<Vec<u8>>,
		/// Whether the LFO is in trig mode.
		trig: bool,
	}
}

impl SharedData {
	fn delay(&self) -> usize {
		self.effects.iter().map(|g| {
			g.iter().map(|e| e.effect.delay()).sum::<usize>()
		}).sum::<usize>()
	}

	fn get_persist_data(&self) -> Vec<PersistData> {
		let mut generators = vec![];
		for generator in self.generators.iter() {
			let params = generator.generator.get_parameters();
			let persist_data = PersistData::Generator {
				path: generator.path.clone(),
				output_track: generator.output_track,
				gain: generator.gain,
				params
			};
			generators.push(persist_data);
		}

		let mut effects = vec![];
		for (i, effects_in_track) in self.effects.iter().enumerate() {
			for (j, effect) in effects_in_track.iter().enumerate() {
				let params = effect.effect.get_parameters();
				let persist_data = PersistData::Effect {
					path: effect.path.clone(),
					mix: effect.mix,
					params,
					track: i,
					index: j
				};
				effects.push(persist_data);
			}
		}

		let mut lfo = vec![];
		for lfo_params in self.lfos.iter() {
			let params = lfo_params.lfo.get_parameters();
			let lfoed_params = lfo_params.params.iter().map(|param| param.as_binary()).collect();
			let persist_data = PersistData::Lfo {
				params,
				lfoed_params,
				trig: lfo_params.trig,
			};
			lfo.push(persist_data);
		}

		let mut persist_data = vec![];
		persist_data.extend(generators);
		persist_data.extend(effects);
		persist_data.extend(lfo);

		persist_data
	}

	fn from_persist_data(data: Vec<PersistData>, sample_rate: usize) -> Self {
		let mut generators = vec![];
		let mut effects: [HashMap<usize, EffectWarpper>; MAX_OUTPUT_TRACKS + 1] = std::array::from_fn(|_| HashMap::new());
		let mut lfos = vec![];

		let generator_list = GENERATOR_LIST.iter().map(|(path, generator)| {
			(path.clone(), generator)
		}).collect::<HashMap<_, _>>();

		let effect_list = EFFECT_LIST.iter().map(|(path, effect)| {
			(path.clone(), effect.iter().map(|(key, value)| (key.clone(), value)).collect::<HashMap<_, _>>())
		}).collect::<HashMap<_, _>>();

		for data in data {
			match data {
				PersistData::Generator {
					path,
					output_track,
					gain,
					params
				} => {
					let Some(generator) = generator_list.get(&path) else {
						continue;
					};
					let mut generator = generator(sample_rate);
					for param in params {
						generator.set_parameter(&param.identifier, param.value.to_set_value());
					}
					generators.push(GeneratorWarpper {
						path,
						generator,
						output_track,
						gain,
					});
				},
				PersistData::Effect {
					path,
					mix,
					params,
					track,
					index
				} => {
					let Some(effect) = effect_list.get(&path.0) else {
						continue;
					};
					let Some(effect) = effect.get(&path.1) else {
						continue;
					};
					let mut effect = effect(sample_rate);
					for param in params {
						effect.set_parameter(&param.identifier, param.value.to_set_value());
					}
					let effect_warpper = EffectWarpper {
						path,
						effect,
						mix,
					};
					effects[track].insert(index, effect_warpper);
				},
				PersistData::Lfo { 
					params, 
					lfoed_params ,
					trig,
				} => {
					let mut lfo = EditableLfo::new();
					let mut lfoed_params_list = vec![];
					for param in params {
						lfo.set_parameter(&param.identifier, param.value.to_set_value());
					}

					for binary in lfoed_params {
						let lfoed_param = LfoedParam::from_binary(binary);
						lfoed_params_list.push(lfoed_param);
					}

					let lfoed_params = LfoedParams {
						lfo,
						params: lfoed_params_list,
						phase: 0.0,
						trig,
					};
					lfos.push(lfoed_params);
				}
			}
		}

		let mut out_effects = std::array::from_fn(|_| vec![]);
		for (i, effects_in_track) in effects.into_iter().enumerate() {
			let mut effects_vec = effects_in_track.into_iter().collect::<Vec<_>>();
			effects_vec.sort_by(|(a, _), (b, _)| a.cmp(b));
			let effects = effects_vec.into_iter().map(|(_, effect)| effect).collect();
			out_effects[i] = effects;
		}

		Self { 
			generators, 
			lfos, 
			sample_rate, 
			current_phase: 0.0,
			effects: out_effects, 
			#[cfg(feature = "standalone")]
			ctx: Some(SimpleContext {
				info: ProcessInfos {
					sample_rate,
					trustable: false,
					playing: true,
					tempo: Some(150.0),
					current_bar_number: None,
					time_signature: None,
					current_time: 0.0,
				},
				midi_events: vec![],
			}),
		}
	}

	fn generate(&mut self, ctx: &mut Box<dyn ProcessContext>) -> [f32; 2] {
		let need_retrig = ctx.events().iter().any(|event| {
			matches!(event, NoteEvent::NoteOn { .. })
		});
		for lfo in &mut self.lfos {
			if need_retrig && lfo.trig {
				lfo.phase = 0.0;
			}

			let value = lfo.lfo.sample(lfo.phase / lfo.lfo.lfo_frequency, 0);
			lfo.params.retain(|param| {
				let set_value = if let (
					FloatOrInt::Float(start), 
					FloatOrInt::Float(end)
				) = (&param.start, &param.end) {
					let start = *start;
					let end = *end;
					let value = if param.logarithmic {
						(start.ln() + (end.ln() - start.ln()) * value).exp()
					}else {
						start + (end - start) * value
					};
					SetValue::Float(value)
				}else if let (
					FloatOrInt::Int(start), 
					FloatOrInt::Int(end)
				) = (&param.start, &param.end) {
					let start = *start as f32;
					let end = *end as f32;
					let value = if param.logarithmic {
						(start.ln() + (end.ln() - start.ln()) * value).exp()
					}else {
						start + (end - start) * value
					};
					SetValue::Int(value as i32)
				}else {
					SetValue::Nothing
				};
				if let Some(track) = &param.track {
					if let Some(inner) = self.effects[*track].get_mut(param.index) {
						inner.effect.set_parameter(&param.identifier, set_value)
					}else {
						false
					}
				}else if let Some(inner) = self.generators.get_mut(param.index) {
					inner.generator.set_parameter(&param.identifier, set_value)
				}else {
					false
				}
			});
			lfo.phase += lfo.lfo.lfo_frequency / self.sample_rate as f32;
			if lfo.lfo.one_shot {
				lfo.phase = lfo.phase.clamp(0.0, 1.0);
			}else {
				lfo.phase %= 1.0;
			}
		}

		self.current_phase += 1.0 / self.sample_rate as f32;
		self.current_phase %= 1.0;

		let mut output_tracks = [[0.0; 2]; MAX_OUTPUT_TRACKS];
		for generator in self.generators.iter_mut() {
			let track = generator.output_track;
			for (i, sample) in generator.generator.generate(ctx).into_iter().enumerate() {
				output_tracks[track][i] += sample * generator.gain;
			}
		}

		for (i, effects) in self.effects.iter_mut().enumerate() {
			if i == MAX_OUTPUT_TRACKS {
				break;
			}

			let other_tracks = (0..MAX_OUTPUT_TRACKS).filter_map(|id| {
				if id != i {
					Some(output_tracks[id])
				}else {
					None
				}
			}).collect::<Vec<_>>();

			let other_tracks_ref = other_tracks.iter().collect::<Vec<_>>();

			
			for effect in effects.iter_mut() {
				let before_backup = output_tracks[i];
				effect.effect.process(&mut output_tracks[i], &other_tracks_ref, ctx);
				output_tracks[i][0] = before_backup[0] * (1.0 - effect.mix) + output_tracks[i][0] * effect.mix;
				output_tracks[i][1] = before_backup[1] * (1.0 - effect.mix) + output_tracks[i][1] * effect.mix;
			}
		}

		let mut output = [0.0; 2];
		for (i, track) in output_tracks.iter().enumerate() {
			if i == MAX_OUTPUT_TRACKS {
				break;
			}
			output[0] += track[0];
			output[1] += track[1];
		}
		for effect in &mut self.effects[MAX_OUTPUT_TRACKS] {
			let before_backup = output;
			effect.effect.process(&mut output, &[], ctx);
			output[0] = before_backup[0] * (1.0 - effect.mix) + output[0] * effect.mix;
			output[1] = before_backup[1] * (1.0 - effect.mix) + output[1] * effect.mix;
		}

		output
	}
}

impl DspDemo {
	/// Create a new DspDemo instance
	/// 
	/// Error may occur if no output device is available or if the stream cannot be created.
	/// 
	/// User should provide the sample rate if `standalone` feature is disabled.
	pub fn new(sample_rate: Option<usize>) -> Result<Self> {
		// let mut delay_buffers = [RingBuffer::new(0), RingBuffer::new(0)];

		
		cfg_if::cfg_if! {
			if #[cfg(feature = "standalone")] {
				let _ = sample_rate;
				let host = cpal::default_host();
				let device = host.default_output_device()
				.ok_or_else(|| anyhow!("No output device available"))?;
			
				let config = device.default_output_config()?;
				let config = StreamConfig::from(config);
				let sample_rate = config.sample_rate.0 as usize;
				let (error_sender, error_receiver) = crossbeam_channel::unbounded();
			}else {
				let Some(sample_rate) = sample_rate else {
					return Err(anyhow!("No sample rate provided"));
				};
			}
		}

		let shared_data = Arc::new(Mutex::new(SharedData {
			generators: vec![],
			effects: std::array::from_fn(|_| vec![]),
			sample_rate,
			current_phase: 0.0,
			lfos: vec![],

			#[cfg(feature = "standalone")]
			ctx: Some(SimpleContext {
				info: ProcessInfos {
					sample_rate,
					trustable: false,
					playing: true,
					tempo: Some(150.0),
					current_bar_number: None,
					time_signature: None,
					current_time: 0.0,
				},
				midi_events: vec![],
			}),
		}));

		cfg_if::cfg_if! {
			if #[cfg(feature = "standalone")] {
				let stream_error_sender = error_sender.clone();

				let shared_data_stream = shared_data.clone();

				let stream = device.build_output_stream(
					&config,
					move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
						let lock_result = shared_data_stream.lock();
						let mut shared_data = match lock_result {
							Ok(data) => data,
							// Err(TryLockError::WouldBlock) => return,
							Err(_) => {
								stream_error_sender.send("Shared data has been poisoned".to_string()).expect("canot send error message");
								return;
							},
						};

						let mut ctx: Box<dyn ProcessContext> = Box::new(shared_data.ctx.take().unwrap());

						for val in data.chunks_mut(2) {
							if val.len() != 2 {
								stream_error_sender.send("Received invalid number of samples".to_string()).expect("canot send error message");
								break;
							}

							let output = shared_data.generate(&mut ctx);

							val[0] = output[0];
							val[1] = output[1];
						}

						shared_data.ctx = Some(SimpleContext {
							info: ProcessInfos {
								sample_rate,
								trustable: false,
								playing: true,
								tempo: Some(150.0),
								current_bar_number: None,
								time_signature: None,
								current_time: 0.0,
							},
							midi_events: vec![],
						});
					},
					move |err| {
						error_sender.send(format!("Stream error: {}", err)).expect("canot send error message");
					},
					None 
				)?;

				stream.play()?;
			}
		}

		Ok(Self {
			#[cfg(feature = "standalone")]
			_stream: stream,
			#[cfg(feature = "standalone")]
			error_receiver,
			current_error: None,
			shared_data,
			sample_rate,
			#[cfg(feature = "standalone")]
			key_holding: HashSet::new(),
			#[cfg(feature = "standalone")]
			note_shift: 0,
			changed_history: RingBuffer::new(CHANGE_HISTORY_LEN),
			current_track_effect: 0,
		})
	}

	/// Get the delay of the current state.
	pub fn delay(&self) -> Result<usize> {
		let lock_result = self.shared_data.lock();
		let shared_data = match lock_result {
			Ok(data) => data,
			// Err(TryLockError::WouldBlock) => return,
			Err(_) => {
				return Err(anyhow!("Shared data has been poisoned"));
			},
		};
		Ok(shared_data.delay())
	}

	/// Generate a single sample.
	pub fn generate(&mut self, ctx: &mut Box<dyn ProcessContext>) -> [f32; 2] {
		let lock_result = self.shared_data.lock();
		let mut shared_data = match lock_result {
			Ok(data) => data,
			// Err(TryLockError::WouldBlock) => return,
			Err(_) => {
				self.current_error = Some("Shared data has been poisoned".to_string());
				return [0.0, 0.0];
			},
		};

		shared_data.generate(ctx)
	}

	/// Get an iterator of parameter changes.
	pub fn change_history(&self) -> impl Iterator<Item = &ListenedValue> {
		self.changed_history.range(0..CHANGE_HISTORY_LEN).filter_map(|inner| inner.as_ref())
	}

	/// Get a value from a generator, only numerical values are supported.
	/// 
	/// Return value is in normalized range [0.0, 1.0].
	pub fn get_generator_value_normalized(&self, index: usize, identifier: &str) -> Result<Option<f32>> {
		let lock_result = self.shared_data.lock();
		let shared_data = match lock_result {
			Ok(data) => data,
			// Err(TryLockError::WouldBlock) => return,
			Err(_) => {
				return Err(anyhow!("Shared data has been poisoned"));
			},
		};
		if let Some(generator) = shared_data.generators.get(index) {
			let parameters = generator.generator.get_parameters();
			if let Some(parameter) = parameters.iter().find(|param| param.identifier == identifier) {
				return match &parameter.value {
					Value::Float { value, range, .. } => {
						let normalized = (*value - *range.start()) / (*range.end() - *range.start());
						Ok(Some(normalized))
					},
					Value::Int { value, range, .. } => {
						let normalized = (*value as f32 - *range.start() as f32) / (*range.end() - *range.start()) as f32;
						Ok(Some(normalized))
					},
					_ => Ok(None),
				};
			}
		}
		Ok(None)
	}

	/// Get a value from an effect, only numerical values are supported.
	pub fn get_effect_value_normalized(&self, track: usize, index: usize, identifier: &str) -> Result<Option<f32>> {
		let lock_result = self.shared_data.lock();
		let shared_data = match lock_result {
			Ok(data) => data,
			// Err(TryLockError::WouldBlock) => return,
			Err(_) => {
				return Err(anyhow!("Shared data has been poisoned"));
			},
		};
		if let Some(effects) = shared_data.effects.get(track) {
			if let Some(effect) = effects.get(index) {
				let parameters = effect.effect.get_parameters();
				if let Some(parameter) = parameters.iter().find(|param| param.identifier == identifier) {
					return match &parameter.value {
						Value::Float { value, range, .. } => {
							let normalized = (*value - *range.start()) / (*range.end() - *range.start());
							Ok(Some(normalized))
						},
						Value::Int { value, range, .. } => {
							let normalized = (*value as f32 - *range.start() as f32) / (*range.end() - *range.start()) as f32;
							Ok(Some(normalized))
						},
						_ => Ok(None),
					};
				}
			}
		}
		Ok(None)
	}

	/// Change the value of a parameter.
	/// 
	/// Returns `true` if the value is changed, `false` 
	/// false will occur if the parameter is not found.
	pub fn change_value(&mut self, value: ChangedValue) -> Result<bool>  {
		let lock_result = self.shared_data.lock();
		let mut shared_data = match lock_result {
			Ok(data) => data,
			// Err(TryLockError::WouldBlock) => return,
			Err(_) => {
				return Err(anyhow!("Shared data has been poisoned"));
			},
		};
		match value {
			ChangedValue::Effect { 
				track, 
				index, 
				identifier, 
				change_to 
			} => {
				if track > MAX_OUTPUT_TRACKS {
					return Ok(false);
				}

				let Some(effects) = shared_data.effects[track].get_mut(index) else {
					return Ok(false);
				};

				Ok(effects.effect.set_parameter(&identifier, change_to))
			},
			ChangedValue::Generator { 
				index, 
				identifier, 
				change_to 
			} => {
				let Some(generators) = shared_data.generators.get_mut(index) else {
					return Ok(false);
				};

				Ok(generators.generator.set_parameter(&identifier, change_to))
			},
		}
	}

	/// Get persist data of the current state.
	pub fn persist_data(&self) -> Result<Vec<PersistData>> {
		let lock_result = self.shared_data.lock();
		let shared_data = match lock_result {
			Ok(data) => data,
			// Err(TryLockError::WouldBlock) => return,
			Err(_) => {
				return Err(anyhow!("Shared data has been poisoned"));
			},
		};
		Ok(shared_data.get_persist_data())
	}

	/// Change the sample rate of the current state.
	/// 
	/// Note: this function will recreate the state with the new sample rate, so it may take some time to complete.
	pub fn change_sample_rate(&mut self, sample_rate: usize) -> Result<()> {
		if self.sample_rate == sample_rate {
			return Ok(());
		}

		let lock_result = self.shared_data.lock();
		let mut shared_data = match lock_result {
			Ok(data) => data,
			// Err(TryLockError::WouldBlock) => return,
			Err(_) => {
				return Err(anyhow!("Shared data has been poisoned"));
			},
		};
		let shard_persisted_data = shared_data.get_persist_data();
		*shared_data = SharedData::from_persist_data(shard_persisted_data, sample_rate);
		self.sample_rate = sample_rate;

		Ok(())
	}

	/// Load persist data to the current state.
	pub fn load_persist_data(&mut self, data: Vec<PersistData>) -> Result<()> {
		let shared_data = SharedData::from_persist_data(data, self.sample_rate);
		let lock_result = self.shared_data.lock();
		let mut shared_data_old = match lock_result {
			Ok(data) => data,
			// Err(TryLockError::WouldBlock) => return,
			Err(_) => {
				return Err(anyhow!("Shared data has been poisoned"));
			},
		};
		*shared_data_old = shared_data;
		Ok(())
	}

	/// Draw the UI
	pub fn ui(&mut self, ui: &mut egui::Ui) {
		if let Some(err_msg) = &self.current_error {
			ui.colored_label(egui::Color32::RED, format!("ERR: {err_msg}"));
			if ui.button("Clear Error").clicked() {
				self.current_error = None;
			}
			ui.separator();
		}

		cfg_if::cfg_if! {
			if #[cfg(feature = "standalone")] {
				let recv_result = self.error_receiver.try_recv();
		
				match recv_result {
					Ok(err_msg) => self.current_error = Some(err_msg),
					Err(TryRecvError::Empty) => {}
					Err(TryRecvError::Disconnected) => self.current_error = Some("Error Channels disconnected".to_string()),
				};
			}
		}
		let lock_result = self.shared_data.lock();
		let mut shared_data = match lock_result {
			Ok(data) => data,
			// Err(TryLockError::WouldBlock) => return,
			Err(_) => {
				self.current_error = Some("Shared data has been poisoned".to_string());
				return;
			},
		};

		egui::TopBottomPanel::top("state_pan").show_inside(ui, |ui| {
			ui.horizontal(|ui| {
				for track in 0..=MAX_OUTPUT_TRACKS {
					let text = if track == MAX_OUTPUT_TRACKS {
						"Master".to_string()
					}else {
						format!("Track {track}")
					};
					ui.selectable_value(&mut self.current_track_effect, track, text);
				}
			});
		});

		#[cfg(feature = "standalone")]
		let mut clear_all_notes = false;
		egui::TopBottomPanel::bottom("lfo_panel").resizable(true).show_inside(ui, |ui| {
			
			egui::TopBottomPanel::bottom("midi_panel").show_inside(ui, |ui| {
				#[cfg(feature = "standalone")]
				{
					let mut key_holding = self.key_holding.iter().copied().collect::<Vec<_>>();
					key_holding.sort();
					let keys = key_holding.into_iter().map(format_note).collect::<Vec<_>>();
					ui.horizontal(|ui| {
						let mut shift = self.note_shift;
						ui.add(Slider::new(&mut shift, -36..=36).text("Note Shift"));
						if shift != self.note_shift {
							self.note_shift = shift;
							clear_all_notes = true;
						}
	
						if ui.button("Clear All Notes").clicked() {
							clear_all_notes = true;
						}
						if keys.is_empty() {
							ui.label("no note playing")
						}else {
							ui.label(format!("Playing: [{}]", keys.join(", ")))
						}
					});
				}
				
				ui.separator();
			});
			let current_phase = shared_data.current_phase;
			egui::CentralPanel::default().show_inside(ui, |ui| {
				egui::ScrollArea::vertical().show(ui, |ui| {
					let mut should_remove_lfo = None;
					for (i, lfo) in shared_data.lfos.iter_mut().enumerate() {
						let id_prefix = format!("Lfo {i}: ");
						ui.collapsing(format!("Lfo {i}"), |ui| {
							lfo.lfo.demo_ui(ui, id_prefix);
							let mut should_remove = None;
							ui.collapsing("Params", |ui| {
								Grid::new(format!("Lfo {i} Params")).show(ui, |ui| {
									for (j, param) in lfo.params.iter_mut().enumerate() {
										let name = if let Some(track) = &param.track {
											format!("Effect {} (at track {track})", param.index)
										}else {
											format!("Generator {}", param.index)
										};
										ui.label(name);
										ui.label(&param.identifier);
										if let (
											FloatOrInt::Float(start), 
											FloatOrInt::Float(end),
											FloatOrInt::Float(max), 
											FloatOrInt::Float(min),
										) = (
											&mut param.start, 
											&mut param.end,
											&param.max, 
											&param.min,
										) {
											let range = *min..=*max;
											ui.add(Slider::new(start, range.clone()).text("Start"));
											ui.add(Slider::new(end, range).text("End"));
										}else if let (
											FloatOrInt::Int(start), 
											FloatOrInt::Int(end),
											FloatOrInt::Int(max), 
											FloatOrInt::Int(min),
										) = (
											&mut param.start, 
											&mut param.end,
											&param.max, 
											&param.min,
										) {
											let range = *min..=*max;
											ui.add(Slider::new(start, range.clone()).text("Start"));
											ui.add(Slider::new(end, range).text("End"));
										}	

										if ui.button("Remove Param").clicked() {
											should_remove = Some(j)
										}
										ui.end_row();
									}
								});
								if let Some(Some(inner)) = ui.menu_button("Add Param", |ui| {
									draw_change_history(ui, &self.changed_history)
								}).inner {
									lfo.params.push(inner);
								}
							});
							if ui.selectable_label(lfo.trig, "Trig").clicked() {
								lfo.trig = !lfo.trig;
								if !lfo.trig {
									lfo.phase = current_phase;
								}
							}
							if let Some(j) = should_remove {
								lfo.params.remove(j);
							}
							if ui.button("Remove Lfo").clicked() {
								should_remove_lfo = Some(i);
							}
						});
					}
					if ui.button("Add Lfo").clicked() {
						let current_phase = shared_data.current_phase;
						shared_data.lfos.push(LfoedParams {
							lfo: EditableLfo::default(),
							params: vec![],
							phase: current_phase,
							trig: false,
						});
					}
	
					if let Some(i) = should_remove_lfo {
						shared_data.lfos.remove(i);
					}
				});
			});
		});
		#[cfg(feature = "standalone")]
		{
			let mut ctx = shared_data.ctx.take().unwrap();
			simulate_midi(ui, &mut ctx, &mut self.key_holding, clear_all_notes, self.note_shift);
			shared_data.ctx = Some(ctx);
		}

		ui.allocate_space(vec2(0.0, 4.0));

		egui::ScrollArea::vertical().show(ui, |ui| {
			let mut should_remove = None;
			let mut should_swap = None;
			let mut should_replace = None;
			let generators_len = shared_data.generators.len();
			let track_available = self.current_track_effect.clamp(0, MAX_OUTPUT_TRACKS - 1);

			for (i, generator) in shared_data.generators.iter_mut().enumerate() {
				let name = generator.generator.name().to_string();
				ui.collapsing(format!("Generator {i}: {name}"), |ui| {
					let prefix = format!("Generator: {}", name);
					let params_old = generator.generator.get_parameters();
					generator.generator.demo_ui(ui, prefix);
					let params_new = generator.generator.get_parameters();
					for (param_old, param_new) in params_old.into_iter().zip(params_new.into_iter()) {
						if param_old != param_new {
							let range = get_range(&param_new.value);
							let history = ListenedValue { 
								history: ChangedValue::Generator { 
									index: i, 
									identifier: param_old.identifier, 
									change_to: param_new.value.to_set_value(), 
								},
								range
							};
							let len = self.changed_history.capacity();
							if let Some(history_old) = &self.changed_history[len - 1] {
								if !(history_old.history.is_param_same(&history.history)) {
									self.changed_history.push(Some(history));
								}
							}else {
								self.changed_history.push(Some(history));
							}
						}
					}
					gain_ui(ui, &mut generator.gain, None, false);
					ui.add(egui::Slider::new(&mut generator.output_track, 0..=MAX_OUTPUT_TRACKS - 1).text("Output Track"));
					if ui.button("Remove Generator").clicked() {
						should_remove = Some(i);
					}
					if i > 0 {
						let btn = ui.button("Move Up");
						if btn.clicked() {
							should_swap = Some((i, i - 1));
						}
					}
					if i < generators_len - 1 {
						let btn = ui.button("Move Down");
						if btn.clicked() {
							should_swap = Some((i, i + 1));
						}
					}
					if let Some(Some(inner)) = ui.menu_button("Replace", |ui| {
						generator_add_menu(ui, self.sample_rate, track_available)
					}).inner {
						should_replace = Some((i, inner));
					};
				});
			}

			if let Some((i, generator)) = should_replace {
				shared_data.generators[i].generator = generator.generator;
				shared_data.generators[i].path = generator.path;
			}
			if let Some((i, j)) = should_swap {
				shared_data.generators.swap(i, j);
			}
			if let Some(i) = should_remove {
				shared_data.generators.remove(i);
			}
			if let Some(Some(inner)) = ui.menu_button("Add Generator", |ui| {
				generator_add_menu(ui, self.sample_rate, track_available)
			}).inner {
				shared_data.generators.push(inner);
			}
			
			ui.separator();

			ui.heading("Effects");
			let effects_len = shared_data.effects.len();
			let current_track = self.current_track_effect;
			let mut should_remove = None;
			let mut should_swap = None;
			let mut should_replace = None;

			for (i, effect) in shared_data.effects[current_track].iter_mut().enumerate() {
				let name = effect.effect.name().to_string();
				ui.collapsing(format!("Effect {i}: {name}"), |ui| {
					let prefix = format!("Effect {i}: {name}");
					
					let params_old = effect.effect.get_parameters();
					effect.effect.demo_ui(ui, prefix);
					let params_new = effect.effect.get_parameters();

					for (param_old, param_new) in params_old.into_iter().zip(params_new.into_iter()) {
						if param_old != param_new {
							let range = get_range(&param_new.value);
							let history = ListenedValue { 
								history: ChangedValue::Effect { 
									track: self.current_track_effect, 
									index: i, 
									identifier: param_old.identifier, 
									change_to: param_new.value.to_set_value(), 
								},
								range,
							};

							let len = self.changed_history.capacity();
							if let Some(history_old) = &self.changed_history[len - 1] {
								if !(history_old.history.is_param_same(&history.history)) {
									self.changed_history.push(Some(history));
								}
							}else {
								self.changed_history.push(Some(history));
							}
						}
					}
					if i > 0 {
						let btn = ui.button("Move Up");
						if btn.clicked() {
							should_swap = Some((i, i - 1));
						}
					}
					if i < effects_len - 1 {
						let btn = ui.button("Move Down");
						if btn.clicked() {
							should_swap = Some((i, i + 1));
						}
					}
					ui.add(egui::Slider::new(&mut effect.mix, 0.0..=1.0).text("Mix"));
					if ui.button("Remove").clicked() {
						should_remove = Some(i);
					}
					if let Some(Some(inner)) = ui.menu_button("Replace", |ui| {
						effect_add_menu(ui, self.sample_rate)
					}).inner {
						should_replace = Some((i, inner));
					};
				});
				ui.separator();
			}
			if let Some((i, effect)) = should_replace {
				shared_data.effects[current_track][i].effect = effect.effect;
				shared_data.effects[current_track][i].path = effect.path;
			}
			if let Some((i, j)) = should_swap {
				shared_data.effects[current_track].swap(i, j);
			}
			if let Some(i) = should_remove {
				shared_data.effects[current_track].remove(i);
			}
			if let Some(Some(inner)) = ui.menu_button("Add Effect", |ui| {
				effect_add_menu(ui, self.sample_rate)
			}).inner {
				shared_data.effects[current_track].push(inner);
			}
		});
	}

	/// Run the DSP demo
	#[cfg(feature = "standalone")]
	pub fn run(self, native_options: eframe::NativeOptions) -> Result<()> {
		eframe::run_native("Dsp Demo", native_options, Box::new(|_| Box::new(self)))
			.map_err(|e| anyhow!("{}", e))?;
		Ok(())
	}
}

#[cfg(feature = "standalone")]
impl eframe::App for DspDemo {
	fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
		egui::CentralPanel::default().show(ctx, |ui| {
			self.ui(ui);
		});
	
		ctx.request_repaint();
	}
}

fn generator_add_menu(ui: &mut egui::Ui, sample_rate: usize, output_track: usize) -> Option<GeneratorWarpper> {
	let mut output = None;
	let mut path = None;
	for (name, generator) in GENERATOR_LIST.iter() {
		if ui.button(name).clicked() {
			output = Some(generator(sample_rate));
			path = Some(name.clone());
			ui.close_menu();
			break;
		}
	}

	if ui.button("Close").clicked() {
		ui.close_menu();
	}

	if let (Some(generator), Some(path)) = (output, path) {
		Some(GeneratorWarpper { 
			path,
			generator, 
			output_track, 
			gain: 1.0 
		})
	}else {
		None
	}
}

fn effect_add_menu(ui: &mut egui::Ui, sample_rate: usize) -> Option<EffectWarpper> {
	let mut output = None;
	let mut path = None;

	for (name_1, effect) in EFFECT_LIST.iter() {
		ui.menu_button(name_1, |ui| {
			for (name, effect) in effect.iter() {
				if ui.button(name).clicked() {
					output = Some(effect(sample_rate));
					path = Some((name_1.clone(), name.clone()));
					ui.close_menu();
					break;
				}
			}
		});
	}

	if ui.button("Close").clicked() {
		ui.close_menu();
	}

	if let (Some(effect), Some((path_1, path_2))) = (output, path) {
		Some(EffectWarpper { 
			path: (path_1, path_2), 
			effect, 
			mix: 1.0 
		})
	}else {
		None
	}
}

#[cfg(feature = "standalone")]
lazy_static::lazy_static! {
	static ref NoteMap: HashMap<Key, u8> = {
		const C5: u8 = 60;
		const C4: u8 = 60 - 12;

		HashMap::from([
			(Key::Q, C5),
			(Key::Num2, C5 + 1),
			(Key::W, C5 + 2),
			(Key::Num3, C5 + 3),
			(Key::E, C5 + 4),
			(Key::R, C5 + 5),
			(Key::Num5, C5 + 6),
			(Key::T, C5 + 7),
			(Key::Num6, C5 + 8),
			(Key::Y, C5 + 9),
			(Key::Num7, C5 + 10),
			(Key::U, C5 + 11),
			(Key::I, C5 + 12),
			(Key::Num9, C5 + 13),
			(Key::O, C5 + 14),
			(Key::Num0, C5 + 15),
			(Key::P, C5 + 16),

			(Key::Z, C4),
			(Key::S, C4 + 1),
			(Key::X, C4 + 2),
			(Key::D, C4 + 3),
			(Key::C, C4 + 4),
			(Key::V, C4 + 5),
			(Key::G, C4 + 6),
			(Key::B, C4 + 7),
			(Key::H, C4 + 8),
			(Key::N, C4 + 9),
			(Key::J, C4 + 10),
			(Key::M, C4 + 11),
		])
	};
}

#[cfg(feature = "standalone")]
fn simulate_midi(
	ui: &mut egui::Ui, 
	ctx: &mut SimpleContext, 
	key_holding: &mut HashSet<u8>,
	clear_all: bool,
	note_shift: i8,
) {
	ui.input(|input| {
		for (key, note) in NoteMap.iter() {
			if input.key_down(*key) {
				let note = if note_shift >= 0 {
					note + note_shift as u8
				}else {
					note.saturating_sub(note_shift.unsigned_abs())
				};
				if key_holding.insert(note) {
					// println!("semi-note on: {}", note);
					ctx.send_event(NoteEvent::NoteOn { time: 0, channel: 0, note, velocity: 1.0 });
				}
			}else if input.key_released(*key) {
				let note = if note_shift >= 0 {
					note + note_shift as u8
				}else {
					note.saturating_sub(note_shift.unsigned_abs())
				};
				if key_holding.remove(&note) {
					// println!("semi-note off: {}", note);
					ctx.send_event(NoteEvent::NoteOff { time: 0, channel: 0, note, velocity: 1.0 });
				}
			}
		}
	});
	if clear_all {
		key_holding.clear();
		ctx.send_event(NoteEvent::ImmediateStop);
	}
}

fn get_range(value: &Value) -> Option<(FloatOrInt, FloatOrInt, bool)> {
	match value {
		Value::Float { range, logarithmic,.. } => {
			Some((
				FloatOrInt::Float(*range.start()), 
				FloatOrInt::Float(*range.end()),
				*logarithmic
			))
		},
		Value::Int { range, logarithmic, .. } => {
			Some((
				FloatOrInt::Int(*range.start()), 
				FloatOrInt::Int(*range.end()),
				*logarithmic
			))
		},
		_ => None,
	}
}

fn draw_change_history(
	ui: &mut egui::Ui, 
	history: &RingBuffer<Option<ListenedValue>>,
) -> Option<LfoedParam> {
	let mut output = None;
	for history in history.range(0..CHANGE_HISTORY_LEN) {
		let Some(history) = history else {
			continue;
		};
		let Some(range) = history.range else {
			continue;
		};

		match &history.history {
			ChangedValue::Generator { index, identifier, .. } => {
				if ui.button(format!("Generator {}: {}", index, identifier)).clicked() {
					output = Some(LfoedParam {
						track: None,
						index: *index,
						identifier: identifier.clone(),
						min: range.0,
						max: range.1,
						start: range.0,
						end: range.1,
						logarithmic: range.2
					});
					ui.close_menu();
					break;
				} 
			},
			ChangedValue::Effect { track, index, identifier, .. } => {
				if ui.button(format!("Effect {}: {}", index, identifier)).clicked() {
					output = Some(LfoedParam {
						track: Some(*track),
						index: *index,
						identifier: identifier.clone(),
						min: range.0,
						max: range.1,
						start: range.0,
						end: range.1,
						logarithmic: range.2
					});
					ui.close_menu();
					break;
				}
			}
		}
	}
	output
}