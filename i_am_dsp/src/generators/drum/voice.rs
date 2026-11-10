use super::*;

/// Where a hit sits in the stereo field.
#[derive(Clone, Copy)]
pub(super) struct Stereo {
	pub(super) pan_l: f32,
	pub(super) pan_r: f32,
	pub(super) width: f32,
}

/// The per-hit variation the humaniser adds.
#[derive(Clone, Copy)]
pub(super) struct Humanized {
	pub(super) level: f32,
	pub(super) detune: f32,
}

/// One sine phase modulating a bank, which turns each mode into a cluster.
pub(super) struct Modulator {
	pub(super) phase: f32,
	pub(super) frequency: f32,
	pub(super) index: f32,
}

impl Modulator {
	pub(super) fn new(frequency: f32, index: f32, rng: &mut Rng) -> Self {
		Self {
			phase: rng.unit(),
			frequency: frequency.max(1.0),
			index,
		}
	}

	#[inline]
	pub(super) fn next(&mut self, rate: f32) -> f32 {
		let value = (self.phase * std::f32::consts::TAU).sin() * self.index;

		self.phase = (self.phase + self.frequency / rate).fract();

		value
	}
}

/// The drum a GM note asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Drum {
	/// The kick.
	Kick,
	/// The snare.
	Snare,
	/// The clap.
	Clap,
	/// A closed hi-hat.
	HatClosed,
	/// The pedal hi-hat.
	HatPedal,
	/// An open hi-hat.
	HatOpen,
	/// One of the toms.
	Tom,
	/// The crash.
	Crash,
	/// The ride.
	Ride,
}

/// The General MIDI drum map, as far as this kit can answer it.
///
/// 35/36 kick, 37/38/40 snare, 39 clap, 41-50 toms, 42/44/46 hats, and the
/// cymbals: 49/52/55/57 are crashes, 51/53/59 rides. Every other note is silent.
pub(crate) fn drum_of(note: u8) -> Option<Drum> {
	Some(match note {
		35 | 36 => Drum::Kick,
		37 | 38 | 40 => Drum::Snare,
		39 => Drum::Clap,
		42 => Drum::HatClosed,
		44 => Drum::HatPedal,
		46 => Drum::HatOpen,
		41 | 43 | 45 | 47 | 48 | 50 => Drum::Tom,
		49 | 52 | 55 | 57 => Drum::Crash,
		51 | 53 | 59 => Drum::Ride,
		_ => return None,
	})
}

/// What a layer is made of.
pub(super) enum Source {
	/// A bank of decaying partials.
	Partials(Vec<Partial>),
	/// Bands of filtered noise.
	Noise(Vec<NoiseBand>),
	/// The 808's six square waves through a highpass: the metal of a cymbal.
	Metal {
		frequencies: [f32; 6],
		t60: f32,
		cutoff: f32,
	},
	/// Noise with a decorrelated second stream: the wide parts of a drum.
	NoiseWide(Vec<NoiseBand>),
	/// An imported one-shot, played once from the hit.
	Sample { data: Arc<SampleData>, level: f32 },
	/// A partial bank whose phases one sine modulates: dense sidebands.
	Fm {
		partials: Vec<Partial>,
		modulator: f32,
		index: f32,
	},
}

/// One layer of a hit: a source, how far it follows the note, and its level.
///
/// The kick is the first drum written this way: the body, the click and the
/// noise edge are three layers of one voice, which is what lets a snare be a
/// body plus air plus wires without a second engine.
pub(super) struct Layer {
	pub(super) source: Source,
	/// 1 follows the pitch curve, 0 stays where the source sits.
	pub(super) keytrack: f32,
	/// How the layer grows with velocity: the body answers a hard hit faster
	/// than the click does.
	pub(super) velocity: f32,
	/// The layer's own level, applied before the master amp curve.
	pub(super) gain: f32,
}

/// Everything one hit needs, built from the controls.
pub(super) struct Recipe {
	pub(super) layers: Vec<Layer>,
	pub(super) pitch: Curve,
	pub(super) amp: Curve,
}

/// One noise band and, for a wide layer, its decorrelated partner.
pub(super) struct NoiseVoice {
	pub(super) band: Band,
	pub(super) second: Option<Band>,
}

impl NoiseVoice {
	/// The band's next sample: a mono pair, or two channels at the given width.
	#[inline]
	pub(super) fn next(&mut self, rng: &mut Rng, width: f32) -> (f32, f32) {
		let left = self.band.next(rng);

		match &mut self.second {
			Some(second) => {
				let right = second.next(rng);
				let cross = 0.5 - width * 0.5;

				(
					left * (1.0 - cross) + right * cross,
					right * (1.0 - cross) + left * cross,
				)
			}
			None => (left, left),
		}
	}
}

/// One running layer of a hit.
pub(super) struct VoiceLayer {
	pub(super) partials: Vec<Running>,
	pub(super) noise: Vec<NoiseVoice>,
	pub(super) metal: Option<MetalVoice>,
	pub(super) sample: Option<SamplePlayback>,
	pub(super) modulator: Option<Modulator>,
	pub(super) keytrack: f32,
}

/// A one-shot being read, at the pace the host asks for it.
pub(super) struct SamplePlayback {
	pub(super) data: Arc<SampleData>,
	pub(super) position: f32,
	pub(super) step: f32,
	pub(super) gain: f32,
}

impl SamplePlayback {
	pub(super) fn new(data: Arc<SampleData>, rate: f32, gain: f32) -> Self {
		Self {
			step: data.rate.max(1) as f32 / rate.max(1.0),
			data,
			position: 0.0,
			gain,
		}
	}

	#[inline]
	pub(super) fn next(&mut self) -> f32 {
		let pcm = &self.data.pcm;
		let index = self.position as usize;

		if index + 1 >= pcm.len() {
			return 0.0;
		}

		let fraction = self.position - index as f32;
		let value = pcm[index] + (pcm[index + 1] - pcm[index]) * fraction;

		self.position += self.step;

		value * self.gain
	}
}

/// One running metal cluster: six square waves and the highpass after them.
pub(super) struct MetalVoice {
	pub(super) frequencies: [f32; 6],
	pub(super) phases: [f32; 6],
	pub(super) level: f32,
	pub(super) decay: f32,
	pub(super) highpass: f32,
	pub(super) previous_in: f32,
	pub(super) previous_out: f32,
}

impl MetalVoice {
	pub(super) fn new(frequencies: [f32; 6], t60: f32, gain: f32, cutoff: f32, rate: f32) -> Self {
		Self {
			frequencies,
			phases: [0.0; 6],
			level: gain,
			// A T60 in a per sample multiplier: -60 dB over the time.
			decay: 10f32.powf(-3.0 / (t60.max(0.005) * rate)),
			highpass: (-std::f32::consts::TAU * cutoff.clamp(200.0, rate * 0.45) / rate).exp(),
			previous_in: 0.0,
			previous_out: 0.0,
		}
	}

	#[inline]
	pub(super) fn next(&mut self, rate: f32) -> f32 {
		let mut sum = 0.0;

		for (phase, frequency) in self.phases.iter_mut().zip(self.frequencies) {
			*phase += frequency / rate;
			if *phase >= 1.0 {
				*phase -= 1.0;
			}
			sum += if *phase < 0.5 { 1.0 } else { -1.0 };
		}

		// One pole highpass: the six squares are the metal, the filter is the
		// hat. Without it the cluster is a low chord, not a cymbal.
		let out = sum - self.previous_in + self.highpass * self.previous_out;
		self.previous_in = sum;
		self.previous_out = out;

		let value = out * self.level * 0.15;
		self.level *= self.decay;
		value
	}
}

/// One hit: the running layers, and its place on the curves.
pub(super) struct Voice {
	pub(super) layers: Vec<VoiceLayer>,
	pub(super) pitch: Curve,
	pub(super) amp: Curve,
	pub(super) time: f32,
	/// 1 while the note rings, ramping to 0 when another hat chokes it.
	pub(super) choke: f32,
	pub(super) choking: bool,
	/// The drum it is, for the choke group.
	pub(super) drum: Drum,
	/// The hit's own level, from the humaniser.
	pub(super) level: f32,
	/// The hit's own detune, from the humaniser.
	pub(super) detune: f32,
	/// The pan gains of this hit.
	pub(super) pan_l: f32,
	pub(super) pan_r: f32,
	/// How wide the decorrelated noise sits.
	pub(super) width: f32,
}

impl Voice {
	pub(super) fn start(
		recipe: &Recipe,
		drum: Drum,
		rate: f32,
		velocity: f32,
		human: Humanized,
		stereo: Stereo,
		rng: &mut Rng,
	) -> Self {
		let velocity = velocity.clamp(0.0, 1.0);
		let mut layers = Vec::with_capacity(recipe.layers.len());

		for layer in &recipe.layers {
			let level = velocity.powf(layer.velocity) * layer.gain;

			match &layer.source {
				Source::Partials(partials) => layers.push(VoiceLayer {
					partials: partials
						.iter()
						.map(|partial| {
							Running::new(
								Partial {
									amplitude: partial.amplitude * level,
									..*partial
								},
								rate,
								rng,
							)
						})
						.collect(),
					noise: Vec::new(),
					metal: None,
					modulator: None,
					sample: None,
					keytrack: layer.keytrack,
				}),
				Source::Noise(bands) => layers.push(VoiceLayer {
					partials: Vec::new(),
					noise: bands
						.iter()
						.map(|band| NoiseVoice {
							band: Band::new(
								NoiseBand {
									gain: band.gain * level,
									..*band
								},
								rate,
							),
							second: None,
						})
						.collect(),
					metal: None,
					modulator: None,
					sample: None,
					keytrack: layer.keytrack,
				}),
				Source::Metal {
					frequencies,
					t60,
					cutoff,
				} => layers.push(VoiceLayer {
					partials: Vec::new(),
					noise: Vec::new(),
					metal: Some(MetalVoice::new(*frequencies, *t60, level, *cutoff, rate)),
					modulator: None,
					sample: None,
					keytrack: layer.keytrack,
				}),
				Source::NoiseWide(bands) => layers.push(VoiceLayer {
					partials: Vec::new(),
					noise: bands
						.iter()
						.map(|band| NoiseVoice {
							band: Band::new(
								NoiseBand {
									gain: band.gain * level,
									..*band
								},
								rate,
							),
							second: Some(Band::new(
								NoiseBand {
									gain: band.gain * level,
									..*band
								},
								rate,
							)),
						})
						.collect(),
					metal: None,
					modulator: None,
					sample: None,
					keytrack: layer.keytrack,
				}),
				Source::Sample { data, level } => layers.push(VoiceLayer {
					partials: Vec::new(),
					noise: Vec::new(),
					metal: None,
					sample: Some(SamplePlayback::new(data.clone(), rate, *level)),
					modulator: None,
					keytrack: layer.keytrack,
				}),
				Source::Fm {
					partials,
					modulator,
					index,
				} => layers.push(VoiceLayer {
					partials: partials
						.iter()
						.map(|partial| {
							Running::new(
								Partial {
									amplitude: partial.amplitude * level,
									..*partial
								},
								rate,
								rng,
							)
						})
						.collect(),
					noise: Vec::new(),
					metal: None,
					modulator: Some(Modulator::new(*modulator, *index, rng)),
					sample: None,
					keytrack: layer.keytrack,
				}),
			}
		}

		Self {
			layers,
			pitch: recipe.pitch.clone(),
			amp: recipe.amp.clone(),
			time: 0.0,
			choke: 1.0,
			choking: false,
			drum,
			level: human.level,
			detune: human.detune,
			pan_l: stereo.pan_l,
			pan_r: stereo.pan_r,
			width: stereo.width,
		}
	}

	pub(super) fn next(&mut self, rng: &mut Rng, rate: f32) -> ([f32; 2], [f32; 2], [f32; 2]) {
		let bend = 2f32.powf(self.pitch.octaves(self.time));
		let mut mono = 0.0;
		let mut wide_l = 0.0;
		let mut wide_r = 0.0;

		for layer in &mut self.layers {
			// A layer that does not follow the note keeps its own frequencies;
			// one that does gets the whole fall, which is how the click stays
			// material while the body dives.
			let ratio = ((1.0 + (bend - 1.0) * layer.keytrack).clamp(0.05, 1024.0)) * self.detune;

			let modulation = match &mut layer.modulator {
				Some(modulator) => modulator.next(rate),
				None => 0.0,
			};

			for partial in &mut layer.partials {
				mono += partial.next_fm(ratio, rate, modulation);
			}
			for noise in &mut layer.noise {
				let (left, right) = noise.next(rng, self.width);

				if noise.second.is_some() {
					wide_l += left;
					wide_r += right;
				} else {
					mono += left;
				}
			}
			if let Some(metal) = &mut layer.metal {
				mono += metal.next(rate);
			}
			if let Some(sample) = &mut layer.sample {
				mono += sample.next();
			}
		}

		let level = self.amp.value(self.time).max(0.0);
		self.time += 1.0 / rate;

		// Only the decorrelated parts are widened. A phase or time spread on the
		// body itself biases the image (the ear locks to the earlier, louder
		// transient), which is why Width lives on the noise layers.
		let mut left = (mono + wide_l) * level * self.pan_l;
		let mut right = (mono + wide_r) * level * self.pan_r;

		if self.choking {
			self.choke = (self.choke - 1.0 / (0.004 * rate)).max(0.0);
			left *= self.choke;
			right *= self.choke;
		}

		left *= self.level;
		right *= self.level;

		// A cymbal is its tail and a hat is its own dry until the IR takes
		// over; both feed a bus, and neither is heard directly.
		match self.drum {
			Drum::Crash | Drum::Ride => ([0.0, 0.0], [left, right], [0.0, 0.0]),
			Drum::HatClosed | Drum::HatPedal | Drum::HatOpen => {
				([0.0, 0.0], [0.0, 0.0], [left, right])
			}
			_ => ([left, right], [0.0, 0.0], [0.0, 0.0]),
		}
	}

	pub(super) fn finished(&self) -> bool {
		self.time > self.amp.end() + 0.02 || (self.choking && self.choke <= 0.0)
	}
}
