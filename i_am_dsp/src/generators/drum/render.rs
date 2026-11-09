use super::*;

impl DrumInstrument {
	/// Imports a one-shot that every hit layers under it, from WAV bytes.
	pub fn set_click_sample(&mut self, bytes: &[u8]) -> Result<(), String> {
		self.click_sample = Some(decode_click(bytes)?);
		Ok(())
	}

	/// Shares a slot with the editor, so a file the page loads reaches the
	/// voices without the page being allowed to write serialized parameters.
	pub fn with_click_slot(mut self, slot: Arc<ClickSlot>) -> Self {
		if let Some(data) = slot.load() {
			self.click_sample = Some(data);
		}
		self.click_seen = slot.generation();
		self.click_slot = Some(slot);
		self
	}

	/// Shares the six one-shot slots with the editor.
	pub fn with_samples(mut self, slots: Arc<Samples>) -> Self {
		self.samples = Some(slots.clone());
		self.samples_seen = slots.generation();
		self.adopt_samples();

		self
	}

	/// Shares the snare's shapes with the editor.
	pub fn with_snare(mut self, shared: Arc<Mutex<SnareCurves>>) -> Self {
		self.snare_shared = shared;
		self
	}

	/// The shell's envelope.
	fn snare_base(&self) -> Curve {
		self.snare_shared
			.lock()
			.map(|curves| curves.base.clone())
			.unwrap_or_default()
	}

	/// The wires' envelope.
	fn snare_noise(&self) -> Curve {
		self.snare_shared
			.lock()
			.map(|curves| curves.noise.clone())
			.unwrap_or_default()
	}

	/// The shell's pitch: octaves to a ratio.
	fn snare_pitch(&self) -> Curve {
		let octaves = self
			.snare_shared
			.lock()
			.map(|curves| curves.pitch.clone())
			.unwrap_or_default();

		Curve::new(
			octaves
				.points()
				.iter()
				.map(|point| Point {
					time: point.time,
					value: 2f32.powf(point.value.clamp(-4.0, 10.0)),
					bend: point.bend,
				})
				.collect(),
		)
	}

	/// Shares the amplitude shapes with the editor.
	pub fn with_amps(mut self, shared: Arc<Mutex<[Curve; 6]>>) -> Self {
		self.amps = shared;
		self
	}

	/// The amplitude shape of one drum, as drawn.
	fn amp_for(&self, drum: Drum) -> Curve {
		let fallback = default_amp_templates();

		self.amps
			.lock()
			.map(|amps| amps[drum.index()].clone())
			.unwrap_or_else(|_| fallback[drum.index()].clone())
	}

	/// When the amplitude shape ends, which is also the voice's length.
	fn amp_end(&self, drum: Drum) -> f32 {
		self.amp_for(drum).end().max(0.01)
	}

	/// The hat's shape: the closed template, stretched for an open hit.
	fn amp_for_hat(&self, open: bool, hat_open: f32) -> Curve {
		let template = self.amp_for(Drum::HatClosed);

		if !open {
			return template;
		}

		let scale = (hat_open.clamp(0.1, 2.0) * 1.3) / template.end().max(1e-4);

		Curve::new(
			template
				.points()
				.iter()
				.map(|point| Point {
					time: point.time * scale,
					value: point.value,
					bend: point.bend,
				})
				.collect(),
		)
	}

	/// Shares the fitted preset with the editor.
	pub fn with_fit_slot(mut self, slot: Arc<FitSlot>) -> Self {
		self.fit_slot = Some(slot.clone());
		self.fit_seen = slot.generation();
		self.adopt_fit();

		self
	}

	/// Installs a fitted bank directly, for a renderer.
	pub fn set_fitted(&mut self, preset: Option<DrumPreset>) {
		self.fitted = preset;
	}

	/// Takes a fitted preset the page posted since the last block.
	pub(super) fn adopt_fit(&mut self) {
		let Some(slot) = &self.fit_slot else {
			return;
		};

		let generation = slot.generation();

		if generation != self.fit_seen {
			self.fit_seen = generation;
			self.fitted = slot.load();
		}
	}

	/// The cymbal built from a fitted bank instead of the generated one.
	fn fitted_recipe(&self, preset: &DrumPreset) -> Recipe {
		let decay = preset.decay.clamp(0.02, 12.0);
		let partials: Vec<Partial> = preset
			.partials
			.iter()
			.filter(|partial| partial.frequency.is_finite() && partial.frequency > 0.0)
			.map(|partial| Partial {
				frequency: partial.frequency.clamp(20.0, 20_000.0),
				amplitude: partial.amplitude.max(0.0),
				t60: partial.t60.clamp(0.005, 20.0),
			})
			.collect();
		let noise: Vec<NoiseBand> = preset
			.noise
			.iter()
			.filter(|band| band.frequency.is_finite() && band.frequency > 0.0)
			.map(|band| NoiseBand {
				frequency: band.frequency.clamp(20.0, 20_000.0),
				q: band.q.clamp(0.1, 20.0),
				gain: band.gain.max(0.0),
				t60: band.t60.clamp(0.005, 20.0),
			})
			.collect();
		let amp = Curve::new(vec![
			Point {
				time: 0.0,
				value: 1.0,
				bend: 0.0,
			},
			Point {
				time: decay,
				value: 0.0,
				bend: 0.0,
			},
		]);

		Recipe {
			layers: vec![
				Layer {
					source: Source::Partials(partials),
					keytrack: 0.0,
					velocity: 1.0,
					gain: 0.6,
				},
				Layer {
					source: Source::Noise(noise),
					keytrack: 0.0,
					velocity: 1.0,
					gain: 1.0,
				},
			],
			pitch: flat_pitch(),
			amp,
		}
	}

	/// The level of one slot's one-shot.
	fn sample_level(&self, drum: Drum) -> f32 {
		match drum {
			Drum::Kick => self.sample_level_kick,
			Drum::Snare => self.sample_level_snare,
			Drum::HatClosed | Drum::HatPedal | Drum::HatOpen => self.sample_level_hat,
			Drum::Clap => self.sample_level_clap,
			Drum::Tom => self.sample_level_tom,
			Drum::Crash | Drum::Ride => self.sample_level_cymbal,
		}
		.clamp(0.0, 2.0)
	}

	/// Copies the shared slots in when the page has changed them.
	pub(super) fn adopt_samples(&mut self) {
		let Some(slots) = &self.samples else {
			return;
		};

		let generation = slots.generation();

		if generation == self.samples_seen {
			return;
		}

		self.samples_seen = generation;

		for (index, drum) in [
			Drum::Kick,
			Drum::Snare,
			Drum::HatClosed,
			Drum::Clap,
			Drum::Tom,
			Drum::Crash,
		]
		.iter()
		.enumerate()
		{
			self.loaded[index] = slots.get(*drum);
		}
	}

	/// A recipe that plays an imported one-shot instead of the synthesis.
	fn sample_recipe(&self, data: &SampleData, level: f32) -> Recipe {
		let duration = data.seconds().max(0.01) * 1.05;
		let amp = Curve::new(vec![
			Point {
				time: 0.0,
				value: 1.0,
				bend: 0.0,
			},
			Point {
				time: duration,
				value: 1.0,
				bend: 0.0,
			},
		]);

		Recipe {
			layers: vec![Layer {
				source: Source::Sample {
					data: Arc::new(data.clone()),
					level: level.clamp(0.0, 2.0),
				},
				keytrack: 0.0,
				velocity: 1.0,
				gain: 1.0,
			}],
			pitch: flat_pitch(),
			amp,
		}
	}

	/// Takes a click the page imported since the last block.
	pub(super) fn adopt_click(&mut self) {
		let Some(slot) = &self.click_slot else {
			return;
		};

		let generation = slot.generation();

		if generation != self.click_seen {
			self.click_seen = generation;

			if let Some(data) = slot.load() {
				self.click_sample = Some(data);
			}
		}
	}

	fn pan_for(&self, drum: Drum) -> f32 {
		match drum {
			Drum::Kick => self.pan_kick,
			Drum::Snare => self.pan_snare,
			Drum::HatClosed | Drum::HatPedal | Drum::HatOpen => self.pan_hat,
			Drum::Clap => self.pan_clap,
			Drum::Tom => self.pan_tom,
			Drum::Crash | Drum::Ride => self.pan_cymbal,
		}
		.clamp(-1.0, 1.0)
	}

	fn level_for(&self, drum: Drum) -> f32 {
		match drum {
			Drum::Kick => self.level_kick,
			Drum::Snare => self.level_snare,
			Drum::HatClosed | Drum::HatPedal | Drum::HatOpen => self.level_hat,
			Drum::Clap => self.level_clap,
			Drum::Tom => self.level_tom,
			Drum::Crash | Drum::Ride => self.level_cymbal,
		}
		.clamp(0.0, 2.0)
	}

	/// Writes a kit preset into the controls.
	pub fn apply_kit(&mut self, index: i32) {
		self.kit = index.clamp(0, 1);

		for (identifier, value) in kit_values(self.kit) {
			let _ = <Self as crate::parameters::Parameters>::set_parameter(
				self,
				identifier,
				crate::parameters::SetValue::Float(*value),
			);
		}
	}

	/// The values a kit preset writes, for a host that applies them itself.
	pub fn kit_preset(index: i32) -> &'static [(&'static str, f32)] {
		kit_values(index)
	}

	/// A silent instrument, built for one sample rate.
	pub fn new(rate: usize) -> Self {
		let rate = if rate > 0 { rate } else { 48_000 };
		let mut instrument = Self {
			rate: rate as f32,
			..Self::default()
		};
		instrument.shared = Arc::new(Mutex::new(instrument.pitch_curve.clone()));

		// The cymbal tail is one convolver for the whole kit, built once.
		let ir = load_cymbal_ir(rate);
		let mut convolver = FftConvolver::<2>::new([ir.clone(), ir], rate);

		convolver.dry_gain = 0.0;
		convolver.wet_gain = 1.0;
		// Scale the wet by the IR's energy, so a longer or denser IR does not
		// come out louder than a short one at the same mix.
		convolver.normalize_energy();
		instrument.tail_convolver = Some(Box::new(convolver));

		let hat_ir = load_hat_ir(rate);
		let mut hat = FftConvolver::<2>::new([hat_ir.clone(), hat_ir], rate);

		hat.dry_gain = 0.0;
		hat.wet_gain = 1.0;
		hat.normalize_energy();
		instrument.hat_convolver = Some(Box::new(hat));

		instrument
	}

	/// Use a curve the editor also holds, so a drag is heard on the next hit.
	pub fn with_shared(mut self, shared: Arc<Mutex<Curve>>) -> Self {
		if let Ok(curve) = shared.lock() {
			self.pitch_curve = curve.clone();
		}
		self.shared = shared;
		self
	}

	/// The controls, turned into one hit.
	pub(super) fn kick_recipe(&self) -> Recipe {
		let tune = self.tune.clamp(20.0, 200.0);
		let decay = self.amp_end(Drum::Kick);

		// Circular membrane ratios: the body is not a sine, and it is not a
		// harmonic series either.
		let ratios = [1.0, 1.593, 2.135, 2.295, 2.653, 2.918];
		let gains = [1.0, 0.42, 0.22, 0.15, 0.09, 0.05];
		let t60 = [1.4, 1.0, 0.7, 0.6, 0.45, 0.35];
		let body = ratios
			.iter()
			.zip(gains.iter())
			.zip(t60.iter())
			.map(|((ratio, gain), t60)| Partial {
				frequency: tune * ratio,
				amplitude: *gain,
				t60: decay * t60,
			})
			.collect();

		let tone = self.click_tone.clamp(200.0, 10_000.0);
		let click_level = self.click.clamp(0.0, 1.0) * self.brightness;
		let click = if click_level <= 0.0 {
			Vec::new()
		} else {
			// Three close, very short resonances: material, not a beep.
			[(0.65, 1.0, 0.012), (1.0, 0.75, 0.009), (1.6, 0.4, 0.006)]
				.iter()
				.map(|(ratio, gain, t60)| Partial {
					frequency: tone * ratio,
					amplitude: gain * click_level,
					t60: *t60,
				})
				.collect()
		};

		let mut noise = Vec::new();
		if click_level > 0.0 {
			// The noise edge of the click, and nothing else: this is a kick.
			noise.push(NoiseBand {
				frequency: 6_000.0,
				q: 0.7,
				gain: 0.5 * click_level * self.brightness,
				t60: 0.0015,
			});
		}

		// The editor writes the shared copy; the saved parameter is the mirror.
		let pitch = self
			.shared
			.lock()
			.map(|curve| curve.clone())
			.unwrap_or_else(|_| self.pitch_curve.clone());
		let amp = self.amp_for(Drum::Kick);

		Recipe {
			layers: vec![
				Layer {
					source: Source::Partials(body),
					keytrack: 1.0,
					velocity: 1.5,
					gain: 0.9,
				},
				Layer {
					source: Source::Partials(click),
					keytrack: 0.0,
					velocity: 0.7,
					gain: 1.0,
				},
				Layer {
					source: Source::Noise(noise),
					keytrack: 0.0,
					velocity: 0.7,
					gain: 1.0,
				},
			],
			pitch,
			amp,
		}
	}

	/// The snare's shell: two mode clusters and the stick's click, with their
	/// own envelope and pitch.
	fn snare_base_recipe(&self) -> Recipe {
		let tune = self.snare_tune.clamp(60.0, 600.0);
		let decay = self.snare_base().end().max(0.02);
		let click_level = self.snare_click.clamp(0.0, 1.0) * self.brightness;
		let low: Vec<Partial> = [(1.0, 1.0, 1.0), (1.593, 0.5, 0.7), (2.135, 0.25, 0.5)]
			.iter()
			.map(|(ratio, gain, t60)| Partial {
				frequency: tune * ratio,
				amplitude: *gain,
				t60: decay * t60,
			})
			.collect();
		let high: Vec<Partial> = [(1.83, 0.5, 0.6), (2.4, 0.25, 0.4)]
			.iter()
			.map(|(ratio, gain, t60)| Partial {
				frequency: tune * ratio,
				amplitude: *gain,
				t60: decay * t60,
			})
			.collect();
		let click: Vec<Partial> = if click_level <= 0.0 {
			Vec::new()
		} else {
			[(2_600.0, 0.5, 0.008), (4_200.0, 0.3, 0.005)]
				.iter()
				.map(|(frequency, gain, t60)| Partial {
					frequency: *frequency,
					amplitude: gain * click_level,
					t60: *t60,
				})
				.collect()
		};

		Recipe {
			layers: vec![
				Layer {
					source: Source::Partials(low),
					keytrack: 1.0,
					velocity: 1.4,
					gain: 0.7,
				},
				Layer {
					source: Source::Partials(high),
					keytrack: 1.0,
					velocity: 1.4,
					gain: 0.35,
				},
				Layer {
					source: Source::Partials(click),
					keytrack: 0.0,
					velocity: 0.8,
					gain: 1.0,
				},
			],
			pitch: self.snare_pitch(),
			amp: self.snare_base(),
		}
	}

	/// The snare's wires: the air and the wire it throws off, with their own
	/// envelope and no pitch.
	fn snare_noise_recipe(&self) -> Recipe {
		let wires = self.snare_wires.clamp(0.0, 1.0) * self.brightness;
		let amp = self.snare_noise();
		let decay = amp.end().max(0.02);
		let air = NoiseBand {
			frequency: 4_200.0,
			q: 0.7,
			gain: 0.45 * wires,
			t60: decay * 0.75,
		};
		let wire = NoiseBand {
			frequency: 6_800.0,
			q: 1.4,
			gain: 0.35 * wires,
			t60: decay,
		};

		Recipe {
			layers: vec![Layer {
				source: Source::NoiseWide(vec![air, wire]),
				keytrack: 0.0,
				velocity: 1.0,
				gain: 1.0,
			}],
			pitch: flat_pitch(),
			amp,
		}
	}

	/// The hi-hat: six square waves through a highpass, with a short noise
	/// edge. Closed, pedal and open differ only in how long the cluster rings.
	fn hat_recipe(&self, drum: Drum) -> Recipe {
		let open = drum == Drum::HatOpen;
		let tune = self.hat_tune.clamp(0.4, 2.5);
		let t60 = if open {
			self.hat_open.clamp(0.1, 2.0)
		} else {
			self.amp_end(Drum::HatClosed) / 1.3
		};
		let cutoff = (self.hat_cutoff * self.brightness).clamp(1_500.0, 12_000.0);
		let level = 1.0;
		let base: [f32; 6] = [205.3, 304.4, 369.6, 522.7, 540.0, 800.0];
		let frequencies = base.map(|frequency| frequency * tune);

		let pitch = Curve::new(vec![
			Point {
				time: 0.0,
				value: 1.0,
				bend: 0.0,
			},
			Point {
				time: t60,
				value: 1.0,
				bend: 0.0,
			},
		]);
		let amp = self.amp_for_hat(open, self.hat_open);

		Recipe {
			layers: vec![
				Layer {
					source: Source::Metal {
						frequencies,
						t60,
						cutoff,
					},
					keytrack: 0.0,
					velocity: 1.0,
					gain: level,
				},
				Layer {
					source: Source::NoiseWide(vec![NoiseBand {
						frequency: 9_000.0,
						q: 0.6,
						gain: 0.25 * level,
						t60: (t60 * 0.4).max(0.01),
					}]),
					keytrack: 0.0,
					velocity: 1.0,
					gain: 1.0,
				},
			],
			pitch,
			amp,
		}
	}

	/// The clap: one noise source, retriggered four times and then let go.
	///
	/// The bursts are the amp curve, not four voices: a clap is a hand, four
	/// hands close together, and the tail they leave on the room.
	fn clap_recipe(&self) -> Recipe {
		let tone = self.clap_tone.clamp(400.0, 8_000.0);
		let decay = self.amp_end(Drum::Clap);

		let amp = self.amp_for(Drum::Clap);
		let pitch = flat_pitch();

		Recipe {
			layers: vec![
				Layer {
					source: Source::NoiseWide(vec![NoiseBand {
						frequency: tone,
						q: 1.1,
						gain: 0.55,
						t60: (decay * 0.6).max(0.05),
					}]),
					keytrack: 0.0,
					velocity: 1.0,
					gain: 1.0,
				},
				Layer {
					source: Source::Noise(vec![NoiseBand {
						frequency: (tone * 3.0).min(12_000.0),
						q: 0.8,
						gain: 0.3 * self.brightness,
						t60: (decay * 0.4).max(0.03),
					}]),
					keytrack: 0.0,
					velocity: 1.0,
					gain: 1.0,
				},
			],
			pitch,
			amp,
		}
	}

	/// The toms: one membrane, tuned by the note.
	fn tom_recipe(&self, note: u8) -> Recipe {
		let frequency =
			440.0 * 2f32.powf((note as f32 - 69.0) / 12.0) * self.tom_tune.clamp(0.4, 2.5);
		let decay = self.amp_end(Drum::Tom);
		let ratios = [1.0, 1.593, 2.135, 2.296, 2.653];
		let gains = [1.0, 0.55, 0.3, 0.18, 0.1];
		let t60 = [1.0, 0.7, 0.5, 0.4, 0.3];

		let partials: Vec<Partial> = ratios
			.iter()
			.zip(gains.iter())
			.zip(t60.iter())
			.map(|((ratio, gain), t60)| Partial {
				frequency: frequency * ratio,
				amplitude: *gain,
				t60: decay * t60,
			})
			.collect();

		let amp = self.amp_for(Drum::Tom);

		Recipe {
			layers: vec![Layer {
				source: Source::Partials(partials),
				keytrack: 1.0,
				velocity: 1.4,
				gain: 0.9,
			}],
			pitch: flat_pitch(),
			amp,
		}
	}

	/// The cymbals: a dense, inharmonic bank, excited by a noise burst.
	fn cymbal_recipe(&self, drum: Drum) -> Recipe {
		if let Some(preset) = &self.fitted {
			return self.fitted_recipe(preset);
		}

		let ride = drum == Drum::Ride;
		let decay = self.amp_end(Drum::Crash) / 0.95 * if ride { 0.45 } else { 1.0 };
		let level = 1.0;
		let count = 64usize;
		let mut partials = Vec::with_capacity(count);

		for index in 0..count {
			let step = index as f32;
			// A plate's modes are dense and irregular: log spaced, then jittered.
			let scatter = ((step * 12.9898).sin() * 43_758.545).fract().abs();
			let frequency = 220.0 * 1.105f32.powf(step) * (0.94 + 0.12 * scatter);
			let amplitude = (1.0 / (1.0 + step).powf(0.55)) * (0.6 + 0.4 * scatter);
			let t60 = decay * (1.0 - 0.55 * (step / count as f32));

			partials.push(Partial {
				// The bank reaches the top of the band; the voice fades a partial
				// out as it approaches Nyquist instead of folding it back.
				frequency: frequency.min(self.rate * 0.45),
				amplitude,
				t60: t60.max(0.05),
			});
		}

		let amp = self.amp_for(Drum::Crash);

		Recipe {
			layers: vec![
				Layer {
					source: Source::Fm {
						partials,
						modulator: 173.0,
						index: 120.0 * self.cymbal_fm.clamp(0.0, 1.0) * self.brightness,
					},
					keytrack: 0.0,
					velocity: 1.0,
					gain: 0.5 * level,
				},
				Layer {
					source: Source::NoiseWide(vec![
						NoiseBand {
							frequency: 8_000.0,
							q: 0.5,
							gain: 0.55 * level * self.brightness,
							t60: (decay * 0.12).max(0.05),
						},
						// Air: the top octave the plate throws off first.
						NoiseBand {
							frequency: 14_000.0,
							q: 0.6,
							gain: 0.42 * level * self.brightness,
							t60: (decay * 0.06).max(0.03),
						},
						NoiseBand {
							frequency: 19_000.0,
							q: 0.8,
							gain: 0.2 * level * self.brightness,
							t60: (decay * 0.04).max(0.02),
						},
					]),
					keytrack: 0.0,
					velocity: 1.0,
					gain: 1.0,
				},
			],
			pitch: flat_pitch(),
			amp,
		}
	}

	/// Start one hit, if the note is the kick's.
	pub fn start(&mut self, note: u8, velocity: f32) -> bool {
		self.adopt_samples();
		self.adopt_fit();

		let Some(drum) = drum_of(note) else {
			return false;
		};

		// A harder hit opens the drum up: the material and the FM index follow
		// the brightness, the body only gets louder.
		let velocity = velocity.clamp(0.0, 1.0);
		self.brightness = 1.0 + (velocity - 0.5) * 2.0 * self.velocity_brightness.clamp(0.0, 1.0);

		// A closed or pedal hat chokes the open one: that is the pair.
		if matches!(drum, Drum::HatClosed | Drum::HatPedal) {
			for voice in &mut self.playing {
				if voice.drum == Drum::HatOpen {
					voice.choking = true;
				}
			}
		}

		if self.playing.len() >= MAX_VOICES {
			self.playing.remove(0);
		}

		// A loaded one-shot replaces this drum's synthesis; the click is only
		// layered under a synthesised drum.
		let loaded = self.loaded[drum.index()].clone();
		let mut recipes: Vec<Recipe> = if let Some(data) = &loaded {
			vec![self.sample_recipe(data, self.sample_level(drum))]
		} else if drum == Drum::Snare {
			// The shell and the wires are separate voices, so their envelopes
			// and the shell's pitch are their own.
			vec![self.snare_base_recipe(), self.snare_noise_recipe()]
		} else {
			vec![match drum {
				Drum::Kick => self.kick_recipe(),
				Drum::Clap => self.clap_recipe(),
				Drum::HatClosed | Drum::HatPedal | Drum::HatOpen => self.hat_recipe(drum),
				Drum::Tom => self.tom_recipe(note),
				Drum::Crash | Drum::Ride => self.cymbal_recipe(drum),
				Drum::Snare => self.snare_base_recipe(),
			}]
		};

		if loaded.is_none()
			&& drum != Drum::Snare
			&& let Some(sample) = &self.click_sample
			&& self.click_sample_level > 0.0
		{
			recipes[0].layers.push(Layer {
				source: Source::Sample {
					data: Arc::new(sample.clone()),
					level: self.click_sample_level.clamp(0.0, 1.0),
				},
				keytrack: 0.0,
				velocity: 1.0,
				gain: 1.0,
			});
		}

		if matches!(drum, Drum::HatClosed | Drum::HatPedal | Drum::HatOpen) {
			// The hat's attack starts dry and hands over to the IR.
			self.hat_gate = 0.0;
		}

		// A drummer never hits twice the same: the humaniser moves the level and
		// the pitch of each hit, and zero leaves the kit exact.
		let humanize = self.humanize.clamp(0.0, 1.0);
		let level = (1.0 + self.rng.bipolar() * 0.3 * humanize) * self.level_for(drum);
		let cents = self.rng.bipolar() * 20.0 * humanize;
		let detune = 2f32.powf(cents / 1_200.0);
		let human = Humanized { level, detune };

		for recipe in &recipes {
			if self.playing.len() >= MAX_VOICES {
				self.playing.remove(0);
			}

			self.playing.push(Voice::start(
				recipe,
				drum,
				self.rate,
				velocity,
				human,
				{
					let pan = self.pan_for(drum).clamp(-1.0, 1.0);
					let angle = (pan + 1.0) * std::f32::consts::FRAC_PI_4;

					Stereo {
						pan_l: angle.cos(),
						pan_r: angle.sin(),
						width: self.width.clamp(0.0, 1.0),
					}
				},
				&mut self.rng,
			));
		}

		true
	}

	/// Renders one hit on its own, without a host.
	///
	/// The instrument is left with nothing playing, so this is for exporting a
	/// one-shot rather than for playing.
	pub fn render(&mut self, note: u8, velocity: f32, seconds: f32) -> Vec<f32> {
		self.playing.clear();
		let _ = self.start(note, velocity);
		let count = (seconds.max(0.01) * self.rate) as usize;

		(0..count).map(|_| self.next_sample()).collect()
	}

	/// Renders one hit as stereo frames, for a host that wants the width.
	pub fn render_stereo(&mut self, note: u8, velocity: f32, seconds: f32) -> Vec<[f32; 2]> {
		self.playing.clear();
		let _ = self.start(note, velocity);
		let count = (seconds.max(0.01) * self.rate) as usize;

		(0..count).map(|_| self.next_frame()).collect()
	}

	/// One sample of the summed voices, driven and levelled.
	///
	/// Public so an offline renderer can pull the instrument without a host.
	pub fn next_sample(&mut self) -> f32 {
		let frame = self.next_frame();

		(frame[0] + frame[1]) * 0.5
	}

	/// One stereo frame of the summed voices, driven and levelled.
	pub fn next_frame(&mut self) -> [f32; 2] {
		let rate = self.rate;
		let rng = &mut self.rng;
		let mut dry = [0.0f32; 2];
		let mut cymbal_send = [0.0f32; 2];
		let mut hat_send = [0.0f32; 2];

		for voice in &mut self.playing {
			let (frame, cymbal, hat) = voice.next(rng, rate);
			dry[0] += frame[0];
			dry[1] += frame[1];
			cymbal_send[0] += cymbal[0];
			cymbal_send[1] += cymbal[1];
			hat_send[0] += hat[0];
			hat_send[1] += hat[1];
		}
		self.playing.retain(|voice| !voice.finished());

		self.tail_send = (cymbal_send[0] + cymbal_send[1]) * 0.5;

		// The tails run while a voice of theirs is alive, and a little longer so
		// the IR's own ring can finish after the send stops.
		if self
			.playing
			.iter()
			.any(|voice| matches!(voice.drum, Drum::Crash | Drum::Ride))
		{
			self.tail_remaining = (rate * 0.3) as usize;
		}
		if self
			.playing
			.iter()
			.any(|voice| matches!(voice.drum, Drum::HatClosed | Drum::HatPedal | Drum::HatOpen))
		{
			self.hat_remaining = (rate * 0.05) as usize;
		}

		let direct = self.hat_direct.clamp(0.0, 10.0) * 0.001;

		if direct <= 0.0 {
			self.hat_gate = 1.0;
		} else if self.hat_gate < 1.0 {
			self.hat_gate = (self.hat_gate + 1.0 / (direct * rate)).min(1.0);
		}

		let mut out = dry;

		if self.tail_remaining > 0
			&& let Some(convolver) = &mut self.tail_convolver
		{
			self.tail_remaining -= 1;
			let wet = convolver.frame(cymbal_send);
			out[0] += wet[0];
			out[1] += wet[1];
		}

		if self.hat_remaining > 0
			&& let Some(convolver) = &mut self.hat_convolver
		{
			self.hat_remaining -= 1;
			let wet = convolver.frame(hat_send);
			out[0] += hat_send[0] * (1.0 - self.hat_gate) + wet[0] * self.hat_gate;
			out[1] += hat_send[1] * (1.0 - self.hat_gate) + wet[1] * self.hat_gate;
		} else if hat_send != [0.0, 0.0] {
			out[0] += hat_send[0];
			out[1] += hat_send[1];
		}

		let drive = self.drive.clamp(1.0, 8.0);
		let gain = self.gain.clamp(0.0, 1.0);

		[
			(out[0] * drive).tanh() * gain,
			(out[1] * drive).tanh() * gain,
		]
	}
}
