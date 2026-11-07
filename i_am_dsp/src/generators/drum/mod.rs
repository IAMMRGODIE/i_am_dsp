//! A drum voice, and the kick that proves it.
//!
//! M0 plays note 36. The body is a bank of decaying partials: a sine is just a
//! bank with one entry, and the inharmonic ratios below are membrane modes. The
//! click is a much shorter bank plus a band of noise, started with random phases
//! so it reads as material rather than a tone. Pitch and amplitude are the same
//! breakpoint curves the editor will draw.

pub mod curve;
pub mod sources;

use std::sync::{Arc, Mutex};

use i_am_dsp_derive::Parameters;

use crate::{Generator, NoteEvent};
use curve::{Curve, Point};
use sources::{Band, NoiseBand, Partial, Rng, Running};

/// The note the kick answers to, in the host's numbering.
pub const KICK: u8 = 36;
/// The note the snare answers to.
pub const SNARE: u8 = 38;

/// How many hits may ring at once.
const MAX_VOICES: usize = 16;

/// What a layer is made of.
enum Source {
	/// A bank of decaying partials.
	Partials(Vec<Partial>),
	/// Bands of filtered noise.
	Noise(Vec<NoiseBand>),
}

/// One layer of a hit: a source, how far it follows the note, and its level.
///
/// The kick is the first drum written this way: the body, the click and the
/// noise edge are three layers of one voice, which is what lets a snare be a
/// body plus air plus wires without a second engine.
struct Layer {
	source: Source,
	/// 1 follows the pitch curve, 0 stays where the source sits.
	keytrack: f32,
	/// How the layer grows with velocity: the body answers a hard hit faster
	/// than the click does.
	velocity: f32,
	/// The layer's own level, applied before the master amp curve.
	gain: f32,
}

/// Everything one hit needs, built from the controls.
struct Recipe {
	layers: Vec<Layer>,
	pitch: Curve,
	amp: Curve,
}

/// One running layer of a hit.
struct VoiceLayer {
	partials: Vec<Running>,
	noise: Vec<Band>,
	keytrack: f32,
}

/// One hit: the running layers, and its place on the curves.
struct Voice {
	layers: Vec<VoiceLayer>,
	pitch: Curve,
	amp: Curve,
	time: f32,
}

impl Voice {
	fn start(recipe: &Recipe, rate: f32, velocity: f32, rng: &mut Rng) -> Self {
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
					keytrack: layer.keytrack,
				}),
				Source::Noise(bands) => layers.push(VoiceLayer {
					partials: Vec::new(),
					noise: bands
						.iter()
						.map(|band| {
							Band::new(
								NoiseBand {
									gain: band.gain * level,
									..*band
								},
								rate,
							)
						})
						.collect(),
					keytrack: layer.keytrack,
				}),
			}
		}

		Self {
			layers,
			pitch: recipe.pitch.clone(),
			amp: recipe.amp.clone(),
			time: 0.0,
		}
	}

	fn next(&mut self, rng: &mut Rng, rate: f32) -> f32 {
		let bend = self.pitch.value(self.time);
		let mut sum = 0.0;

		for layer in &mut self.layers {
			// A layer that does not follow the note keeps its own frequencies;
			// one that does gets the whole fall, which is how the click stays
			// material while the body dives.
			let ratio = (1.0 + (bend - 1.0) * layer.keytrack).clamp(0.05, 8.0);

			for partial in &mut layer.partials {
				sum += partial.next(ratio, rate);
			}
			for band in &mut layer.noise {
				sum += band.next(rng);
			}
		}

		let level = self.amp.value(self.time).max(0.0);
		self.time += 1.0 / rate;

		sum * level
	}

	fn finished(&self) -> bool {
		self.time > self.amp.end() + 0.02
	}
}

/// A kick drum synthesised from partials, a click and noise.
#[derive(Parameters)]
pub struct DrumInstrument {
	#[skip]
	playing: Vec<Voice>,
	#[skip]
	rate: f32,
	#[skip]
	rng: Rng,
	/// The pitch curve, saved with the plug-in state. The voice reads the
	/// shared copy, which the editor keeps in step while the window is open.
	#[serde]
	#[id(display = "Pitch Curve")]
	pub pitch_curve: Curve,
	#[skip]
	shared: Arc<Mutex<Curve>>,
	/// The fundamental of the kick, in Hz.
	#[range(min = 30.0, max = 120.0)]
	#[default(52.0)]
	#[unit(hertz)]
	#[id(display = "Tune")]
	#[layout(group = "Kick", order = 0, widget = knob, hero = "fall")]
	pub tune: f32,
	/// How long the body rings, in seconds.
	#[range(min = 0.05, max = 2.0)]
	#[default(0.45)]
	#[id(display = "Decay")]
	#[layout(group = "Kick", order = 2, widget = knob)]
	pub decay: f32,
	/// The click level.
	#[range(min = 0.0, max = 1.0)]
	#[default(0.5)]
	#[id(display = "Click")]
	#[layout(group = "Kick", order = 3, widget = knob)]
	pub click: f32,
	/// Where the click's material sits, in Hz.
	#[range(min = 300.0, max = 6000.0)]
	#[default(1800.0)]
	#[unit(hertz)]
	#[id(display = "Click Tone")]
	#[layout(group = "Kick", order = 4, widget = knob)]
	pub click_tone: f32,
	/// The drive into the output stage.
	#[range(min = 1.0, max = 8.0)]
	#[default(1.5)]
	#[id(display = "Drive")]
	#[layout(group = "Kick", order = 5, widget = knob)]
	pub drive: f32,
	/// The output level.
	#[range(min = 0.0, max = 1.0)]
	#[default(0.8)]
	#[id(display = "Gain")]
	#[layout(group = "Kick", order = 6, widget = knob)]
	pub gain: f32,
	/// The fundamental of the snare's shell, in Hz.
	#[range(min = 120.0, max = 400.0)]
	#[default(180.0)]
	#[unit(hertz)]
	#[id(display = "Snare Tune")]
	#[layout(group = "Snare", order = 0, widget = knob)]
	pub snare_tune: f32,
	/// How long the shell rings, in seconds.
	#[range(min = 0.05, max = 1.0)]
	#[default(0.18)]
	#[id(display = "Snare Decay")]
	#[layout(group = "Snare", order = 1, widget = knob)]
	pub snare_decay: f32,
	/// How much air and wire the hit throws off.
	#[range(min = 0.0, max = 1.0)]
	#[default(0.5)]
	#[id(display = "Wires")]
	#[layout(group = "Snare", order = 2, widget = knob)]
	pub snare_wires: f32,
	/// The snare's stick click.
	#[range(min = 0.0, max = 1.0)]
	#[default(0.4)]
	#[id(display = "Snare Click")]
	#[layout(group = "Snare", order = 3, widget = knob)]
	pub snare_click: f32,
}

/// The pitch fall a fresh kick starts with: a deep dive in the first 30 ms.
pub fn default_pitch() -> Curve {
	Curve::new(vec![
		Point {
			time: 0.0,
			value: 4.5,
			bend: -2.0,
		},
		Point {
			time: 0.03,
			value: 1.0,
			bend: 0.0,
		},
	])
}

impl Default for DrumInstrument {
	fn default() -> Self {
		Self {
			playing: Vec::new(),
			rate: 48_000.0,
			rng: Rng::default(),
			pitch_curve: default_pitch(),
			shared: Arc::new(Mutex::new(default_pitch())),
			tune: 52.0,
			decay: 0.45,
			click: 0.5,
			click_tone: 1800.0,
			drive: 1.5,
			gain: 0.8,
			snare_tune: 180.0,
			snare_decay: 0.18,
			snare_wires: 0.5,
			snare_click: 0.4,
		}
	}
}

impl DrumInstrument {
	/// A silent instrument, built for one sample rate.
	pub fn new(rate: usize) -> Self {
		let mut instrument = Self {
			rate: if rate > 0 { rate as f32 } else { 48_000.0 },
			..Self::default()
		};
		instrument.shared = Arc::new(Mutex::new(instrument.pitch_curve.clone()));
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
	fn kick_recipe(&self) -> Recipe {
		let tune = self.tune.clamp(20.0, 200.0);
		let decay = self.decay.clamp(0.02, 4.0);

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
		let click_level = self.click.clamp(0.0, 1.0);
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
				gain: 0.5 * click_level,
				t60: 0.0015,
			});
		}

		// The editor writes the shared copy; the saved parameter is the mirror.
		let pitch = self
			.shared
			.lock()
			.map(|curve| curve.clone())
			.unwrap_or_else(|_| self.pitch_curve.clone());
		let amp = Curve::new(vec![
			Point {
				time: 0.0,
				value: 1.0,
				bend: -2.5,
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

	/// The note the kick answers to, and its page.
	fn is_kick(note: u8) -> bool {
		note == KICK
	}

	/// The note the snare answers to.
	fn is_snare(note: u8) -> bool {
		note == SNARE
	}

	/// The snare: two shell clusters, the air and wire it throws off, and a
	/// stick click. Same engine as the kick, different layers.
	fn snare_recipe(&self) -> Recipe {
		let tune = self.snare_tune.clamp(60.0, 600.0);
		let decay = self.snare_decay.clamp(0.03, 2.0);
		let wires = self.snare_wires.clamp(0.0, 1.0);
		let click_level = self.snare_click.clamp(0.0, 1.0);

		// Two membrane clusters, the way the 808's two bridged-Ts sit an octave
		// and a fifth apart. Without the second the shell is a tom.
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
		let air = NoiseBand {
			frequency: 4_200.0,
			q: 0.7,
			gain: 0.45 * wires,
			t60: 0.12,
		};
		let wire = NoiseBand {
			frequency: 6_800.0,
			q: 1.4,
			gain: 0.35 * wires,
			t60: 0.16,
		};

		let pitch = Curve::new(vec![
			Point {
				time: 0.0,
				value: 1.6,
				bend: -1.0,
			},
			Point {
				time: 0.015,
				value: 1.0,
				bend: 0.0,
			},
		]);
		let amp = Curve::new(vec![
			Point {
				time: 0.0,
				value: 1.0,
				bend: -2.5,
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
				Layer {
					source: Source::Noise(vec![air, wire]),
					keytrack: 0.0,
					velocity: 1.0,
					gain: 1.0,
				},
			],
			pitch,
			amp,
		}
	}

	/// Start one hit, if the note is the kick's.
	pub fn start(&mut self, note: u8, velocity: f32) -> bool {
		if !Self::is_kick(note) && !Self::is_snare(note) {
			return false;
		}

		if self.playing.len() >= MAX_VOICES {
			self.playing.remove(0);
		}

		let recipe = if Self::is_snare(note) {
			self.snare_recipe()
		} else {
			self.kick_recipe()
		};
		self.playing
			.push(Voice::start(&recipe, self.rate, velocity, &mut self.rng));

		true
	}

	/// One sample of the summed voices, driven and levelled.
	///
	/// Public so an offline renderer can pull the instrument without a host.
	pub fn next_sample(&mut self) -> f32 {
		let rate = self.rate;
		let rng = &mut self.rng;
		let mut sum = 0.0;

		for voice in &mut self.playing {
			sum += voice.next(rng, rate);
		}
		self.playing.retain(|voice| !voice.finished());

		(sum * self.drive.clamp(1.0, 8.0)).tanh() * self.gain.clamp(0.0, 1.0)
	}
}

impl Generator<2> for DrumInstrument {
	fn generate(&mut self, context: &mut Box<dyn crate::ProcessContext>) -> [f32; 2] {
		let rate = context.infos().sample_rate as f32;
		if rate > 0.0 && (rate - self.rate).abs() > 0.5 {
			self.rate = rate;
			self.playing.clear();
		}

		let events = context.events().to_vec();
		context.clear_events();
		for event in &events {
			match event {
				NoteEvent::NoteOn { note, velocity, .. } => {
					self.start((*note).clamp(0, 127) as u8, *velocity);
				}
				NoteEvent::ImmediateStop => self.playing.clear(),
				_ => {}
			}
		}

		let sample = self.next_sample();

		[sample, sample]
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn render(kit: &mut DrumInstrument, count: usize) -> Vec<f32> {
		(0..count).map(|_| kit.next_sample()).collect()
	}

	fn rms(samples: &[f32]) -> f32 {
		(samples.iter().map(|sample| sample * sample).sum::<f32>()
			/ samples.len().max(1) as f32)
			.sqrt()
	}

	#[test]
	fn the_kick_sounds_and_decays() {
		let mut kit = DrumInstrument::new(48_000);
		assert!(kit.start(KICK, 1.0));

		let out = render(&mut kit, 48_000);
		let peak = out.iter().fold(0.0f32, |peak, sample| peak.max(sample.abs()));

		assert!(peak > 0.05, "the kick was silent: {peak}");
		assert!(rms(&out[..2_400]) > rms(&out[45_600..]) * 100.0);
	}

	#[test]
	fn the_snare_sounds_and_is_not_the_kick() {
		let mut snare = DrumInstrument::new(48_000);
		assert!(snare.start(SNARE, 1.0));
		let snare = render(&mut snare, 9_600);

		let mut kick = DrumInstrument::new(48_000);
		kick.start(KICK, 1.0);
		let kick = render(&mut kick, 9_600);

		let peak = snare.iter().fold(0.0f32, |peak, sample| peak.max(sample.abs()));
		assert!(peak > 0.05, "the snare was silent: {peak}");

		// The wires make the snare cross zero far more often than a 52 Hz shell.
		let crossings = |samples: &[f32]| {
			samples
				.windows(2)
				.filter(|pair| (pair[0] >= 0.0) != (pair[1] >= 0.0))
				.count()
		};

		assert!(
			crossings(&snare) > crossings(&kick) * 3,
			"the snare sounds like a kick: {} vs {}",
			crossings(&snare),
			crossings(&kick)
		);
	}

	#[test]
	fn a_note_the_kick_does_not_answer_is_silent() {
		let mut kit = DrumInstrument::new(48_000);

		assert!(!kit.start(60, 1.0));
		assert!(render(&mut kit, 480).iter().all(|sample| sample.abs() < 1e-12));
	}

	#[test]
	fn a_harder_hit_is_louder() {
		let mut soft = DrumInstrument::new(48_000);
		let mut hard = DrumInstrument::new(48_000);
		soft.start(KICK, 0.25);
		hard.start(KICK, 1.0);

		let soft = rms(&render(&mut soft, 4_800));
		let hard = rms(&render(&mut hard, 4_800));

		assert!(hard > soft * 2.0, "{hard} vs {soft}");
	}

	#[test]
	fn a_clickless_kick_has_no_click() {
		let mut kit = DrumInstrument::new(48_000);
		kit.click = 0.0;

		let recipe = kit.kick_recipe();
		let material = recipe.layers.iter().filter(|layer| layer.keytrack == 0.0);
		let mut any = false;

		for layer in material {
			match &layer.source {
				Source::Partials(partials) => any |= !partials.is_empty(),
				Source::Noise(bands) => any |= !bands.is_empty(),
			}
		}

		assert!(!any, "a clickless kick still has material layers");
	}

	#[test]
	fn the_editor_curve_is_heard() {
		let kit = DrumInstrument::new(48_000);
		let shared = Arc::new(Mutex::new(Curve::from_triples([
			(0.0, 7.0, 0.0),
			(0.1, 1.0, 0.0),
		])));

		let kit = kit.with_shared(shared);

		assert!((kit.kick_recipe().pitch.value(0.0) - 7.0).abs() < 1e-5);
	}
}
