use super::*;

/// The listender for parameter changes.
pub struct ListenedValue {
	pub(super) history: ChangedValue,
	pub(super) range: Option<(FloatOrInt, FloatOrInt, bool)>,
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
	},
}

impl ChangedValue {
	pub(super) fn is_param_same(&self, other: &Self) -> bool {
		match (self, other) {
			(
				ChangedValue::Generator {
					index, identifier, ..
				},
				ChangedValue::Generator {
					index: other_index,
					identifier: other_identifier,
					..
				},
			) => *index == *other_index && identifier == other_identifier,
			(
				ChangedValue::Effect {
					track,
					index,
					identifier,
					..
				},
				ChangedValue::Effect {
					track: other_track,
					index: other_index,
					identifier: other_identifier,
					..
				},
			) => *track == *other_track && *index == *other_index && identifier == other_identifier,
			_ => false,
		}
	}
}

// last for master
pub(super) const MAX_OUTPUT_TRACKS: usize = 4;

pub(super) struct GeneratorWarpper {
	pub(super) path: String,
	pub(super) generator: Box<dyn Generator>,
	pub(super) output_track: usize,
	pub(super) gain: f32,
}

pub(super) struct EffectWarpper {
	pub(super) path: (String, String),
	pub(super) effect: Box<dyn Effect>,
	pub(super) mix: f32,
	/// Per-channel delay lines used to align the dry signal with the wet
	/// signal, whose latency is reported by Effect::delay().
	pub(super) dry_delay: [RingBuffer<f32>; 2],
}

impl EffectWarpper {
	/// Delays the dry signal by the effect's latency so that dry and wet are
	/// time-aligned when mixed. The delay line is resized automatically when
	/// the effect latency changes at runtime.
	pub(super) fn align_dry(&mut self, dry: [f32; 2]) -> [f32; 2] {
		let latency = self.effect.delay();
		if self.dry_delay[0].capacity() != latency {
			for buffer in self.dry_delay.iter_mut() {
				buffer.resize(latency);
			}
		}

		if latency == 0 {
			return dry;
		}

		let delayed = [self.dry_delay[0][0], self.dry_delay[1][0]];
		self.dry_delay[0].push(dry[0]);
		self.dry_delay[1].push(dry[1]);
		delayed
	}
}
pub(super) struct SharedData {
	pub(super) generators: Vec<GeneratorWarpper>,
	pub(super) effects: [Vec<EffectWarpper>; MAX_OUTPUT_TRACKS + 1],
	pub(super) track_delay: [[RingBuffer<f32>; 2]; MAX_OUTPUT_TRACKS],
	#[cfg(feature = "standalone")]
	pub(super) ctx: Option<SimpleContext>,
	pub(super) lfos: Vec<LfoedParams>,
	pub(super) sample_rate: usize,
	pub(super) current_phase: f32,
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
	pub(super) fn as_binary(&self) -> Vec<u8> {
		let mut binary = vec![];
		match self {
			FloatOrInt::Float(value) => {
				binary.push(1u8);
				binary.extend_from_slice(&value.to_le_bytes());
			}
			FloatOrInt::Int(value) => {
				binary.push(0u8);
				binary.extend_from_slice(&value.to_le_bytes());
			}
		}
		binary
	}

	pub(super) fn from_binary(binary: Vec<u8>) -> Self {
		if binary[0] == 1u8 {
			let mut bytes = [0u8; 4];
			bytes.copy_from_slice(&binary[1..]);
			FloatOrInt::Float(f32::from_le_bytes(bytes))
		} else {
			let mut bytes = [0u8; 4];
			bytes.copy_from_slice(&binary[1..]);
			FloatOrInt::Int(i32::from_le_bytes(bytes))
		}
	}
}

pub(super) struct LfoedParams {
	pub(super) lfo: EditableLfo,
	pub(super) params: Vec<LfoedParam>,
	pub(super) phase: f32,
	pub(super) trig: bool,
}

pub(super) struct LfoedParam {
	// none for generator
	pub(super) track: Option<usize>,
	pub(super) index: usize,
	pub(super) identifier: String,
	pub(super) min: FloatOrInt,
	pub(super) max: FloatOrInt,
	pub(super) start: FloatOrInt,
	pub(super) end: FloatOrInt,
	pub(super) logarithmic: bool,
}

impl LfoedParam {
	pub(super) fn as_binary(&self) -> Vec<u8> {
		let mut binary = vec![];
		if let Some(track) = self.track {
			binary.push(1u8);
			binary.extend_from_slice(&(track as u64).to_le_bytes());
		} else {
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

	pub(super) fn from_binary(mut binary: Vec<u8>) -> Self {
		let mut track = None;
		if binary[0] == 1u8 {
			let current = binary.split_off(5);
			let mut bytes = [0u8; 8];
			bytes.copy_from_slice(&binary[1..]);
			track = Some(usize::from_le_bytes(bytes));
			binary = current;
		} else {
			let current = binary.split_off(1);
			binary = current;
		}

		let mut index = [0u8; 8];
		index.copy_from_slice(&binary[0..8]);
		let index = u64::from_le_bytes(index) as usize;
		let identifier_len = u64::from_le_bytes(binary[8..16].try_into().unwrap()) as usize;
		let identifier = String::from_utf8_lossy(&binary[16..16 + identifier_len]).to_string();
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
		params: Vec<Parameter>,
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
		index: usize,
	},
	/// The data of an LFO.
	Lfo {
		/// The parameters of the LFO.
		params: Vec<Parameter>,
		/// The lfoed parameters, in binary format.
		lfoed_params: Vec<Vec<u8>>,
		/// Whether the LFO is in trig mode.
		trig: bool,
	},
}

impl SharedData {
	/// Total latency of the current signal graph, in samples.
	///
	/// The output tracks are mixed in parallel, so the graph latency is the
	/// latency of the slowest output track plus the latency of the master
	/// bus effects, which every track passes through.
	pub(super) fn delay(&self) -> usize {
		let max_track_delay = (0..MAX_OUTPUT_TRACKS)
			.map(|i| {
				self.effects[i]
					.iter()
					.map(|e| e.effect.delay())
					.sum::<usize>()
			})
			.max()
			.unwrap_or(0);
		let master_delay = self.effects[MAX_OUTPUT_TRACKS]
			.iter()
			.map(|e| e.effect.delay())
			.sum::<usize>();
		max_track_delay + master_delay
	}

	pub(super) fn get_persist_data(&self) -> Vec<PersistData> {
		let mut generators = vec![];
		for generator in self.generators.iter() {
			let params = generator.generator.get_parameters();
			let persist_data = PersistData::Generator {
				path: generator.path.clone(),
				output_track: generator.output_track,
				gain: generator.gain,
				params,
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
					index: j,
				};
				effects.push(persist_data);
			}
		}

		let mut lfo = vec![];
		for lfo_params in self.lfos.iter() {
			let params = lfo_params.lfo.get_parameters();
			let lfoed_params = lfo_params
				.params
				.iter()
				.map(|param| param.as_binary())
				.collect();
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

	pub(super) fn from_persist_data(data: Vec<PersistData>, sample_rate: usize) -> Self {
		let mut generators = vec![];
		let mut effects: [HashMap<usize, EffectWarpper>; MAX_OUTPUT_TRACKS + 1] =
			std::array::from_fn(|_| HashMap::new());
		let mut lfos = vec![];

		let generator_list = GENERATOR_LIST
			.iter()
			.map(|(path, generator)| (path.clone(), generator))
			.collect::<HashMap<_, _>>();

		let effect_list = EFFECT_LIST
			.iter()
			.map(|(path, effect)| {
				(
					path.clone(),
					effect
						.iter()
						.map(|(key, value)| (key.clone(), value))
						.collect::<HashMap<_, _>>(),
				)
			})
			.collect::<HashMap<_, _>>();

		for data in data {
			match data {
				PersistData::Generator {
					path,
					output_track,
					gain,
					params,
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
				}
				PersistData::Effect {
					path,
					mix,
					params,
					track,
					index,
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
						dry_delay: std::array::from_fn(|_| RingBuffer::new(0)),
					};
					effects[track].insert(index, effect_warpper);
				}
				PersistData::Lfo {
					params,
					lfoed_params,
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
			effects_vec.sort_by_key(|(a, _)| *a);
			let effects = effects_vec.into_iter().map(|(_, effect)| effect).collect();
			out_effects[i] = effects;
		}

		Self {
			generators,
			lfos,
			sample_rate,
			current_phase: 0.0,
			effects: out_effects,
			track_delay: std::array::from_fn(|_| std::array::from_fn(|_| RingBuffer::new(0))),
			#[cfg(feature = "standalone")]
			ctx: Some(SimpleContext {
				info: ProcessInfos {
					sample_rate,
					trustable: false,
					playing: true,
					tempo: Some(150.0),
					time_signature: None,
					current_time: 0.0,
					// `position_beats` (and the deprecated `current_beat_number`) stay at their
					// default: a standalone context has no host timeline to report yet.
					..ProcessInfos::default()
				},
				midi_events: vec![],
			}),
		}
	}

	pub(super) fn generate(&mut self, ctx: &mut Box<dyn ProcessContext>) -> [f32; 2] {
		let need_retrig = ctx
			.events()
			.iter()
			.any(|event| matches!(event, NoteEvent::NoteOn { .. }));
		for lfo in &mut self.lfos {
			if need_retrig && lfo.trig {
				lfo.phase = 0.0;
			}

			let value = lfo.lfo.sample(lfo.phase / lfo.lfo.lfo_frequency, 0);
			lfo.params.retain(|param| {
				let set_value = if let (FloatOrInt::Float(start), FloatOrInt::Float(end)) =
					(&param.start, &param.end)
				{
					let start = *start;
					let end = *end;
					let value = if param.logarithmic {
						(start.ln() + (end.ln() - start.ln()) * value).exp()
					} else {
						start + (end - start) * value
					};
					SetValue::Float(value)
				} else if let (FloatOrInt::Int(start), FloatOrInt::Int(end)) =
					(&param.start, &param.end)
				{
					let start = *start as f32;
					let end = *end as f32;
					let value = if param.logarithmic {
						(start.ln() + (end.ln() - start.ln()) * value).exp()
					} else {
						start + (end - start) * value
					};
					SetValue::Int(value as i32)
				} else {
					SetValue::Nothing
				};
				if let Some(track) = &param.track {
					if let Some(inner) = self.effects[*track].get_mut(param.index) {
						inner.effect.set_parameter(&param.identifier, set_value)
					} else {
						false
					}
				} else if let Some(inner) = self.generators.get_mut(param.index) {
					inner.generator.set_parameter(&param.identifier, set_value)
				} else {
					false
				}
			});
			lfo.phase += lfo.lfo.lfo_frequency / self.sample_rate as f32;
			if lfo.lfo.one_shot {
				lfo.phase = lfo.phase.clamp(0.0, 1.0);
			} else {
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

		// Total latency of each output track: the latencies of its effect chain
		// add up, because each effect's output is delayed by its own latency.
		let mut track_total_delay = [0usize; MAX_OUTPUT_TRACKS];
		let mut max_track_delay = 0usize;
		for (i, effects) in self.effects.iter().enumerate() {
			if i == MAX_OUTPUT_TRACKS {
				break;
			}
			track_total_delay[i] = effects.iter().map(|e| e.effect.delay()).sum::<usize>();
			max_track_delay = max_track_delay.max(track_total_delay[i]);
		}

		for (i, effects) in self.effects.iter_mut().enumerate() {
			if i == MAX_OUTPUT_TRACKS {
				break;
			}

			let other_tracks = (0..MAX_OUTPUT_TRACKS)
				.filter_map(|id| {
					if id != i {
						Some(output_tracks[id])
					} else {
						None
					}
				})
				.collect::<Vec<_>>();

			let other_tracks_ref = other_tracks.iter().collect::<Vec<_>>();

			for effect in effects.iter_mut() {
				// The effect's process() output is delayed by effect.delay(), so
				// delay the dry signal by the same amount before mixing, otherwise
				// dry and wet are time-misaligned and the mix comb-filters.
				let dry = output_tracks[i];
				let dry_delayed = effect.align_dry(dry);
				effect
					.effect
					.process(&mut output_tracks[i], &other_tracks_ref, ctx);
				output_tracks[i][0] =
					dry_delayed[0] * (1.0 - effect.mix) + output_tracks[i][0] * effect.mix;
				output_tracks[i][1] =
					dry_delayed[1] * (1.0 - effect.mix) + output_tracks[i][1] * effect.mix;
			}
		}

		// Align every track to the slowest one before summing, so parallel
		// tracks with different effect-chain latencies stay time-aligned.
		for (i, track) in output_tracks.iter_mut().enumerate() {
			let extra_delay = max_track_delay - track_total_delay[i];
			if extra_delay > 0 {
				if self.track_delay[i][0].capacity() != extra_delay {
					for buffer in self.track_delay[i].iter_mut() {
						buffer.resize(extra_delay);
					}
				}
				let delayed = [self.track_delay[i][0][0], self.track_delay[i][1][0]];
				self.track_delay[i][0].push(track[0]);
				self.track_delay[i][1].push(track[1]);
				*track = delayed;
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
			// Master effects are common to every track, so their latency only
			// shifts the whole mix; the dry signal still has to be aligned for
			// a correct dry/wet mix.
			let dry = output;
			let dry_delayed = effect.align_dry(dry);
			effect.effect.process(&mut output, &[], ctx);
			output[0] = dry_delayed[0] * (1.0 - effect.mix) + output[0] * effect.mix;
			output[1] = dry_delayed[1] * (1.0 - effect.mix) + output[1] * effect.mix;
		}

		output
	}
}
