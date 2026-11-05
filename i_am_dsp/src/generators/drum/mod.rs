//! A drum voice, and the kick that proves it.
//!
//! M0 plays note 36. The body is a bank of decaying partials: a sine is just a
//! bank with one entry, and the inharmonic ratios below are membrane modes. The
//! click is a much shorter bank plus a band of noise, started with random phases
//! so it reads as material rather than a tone. Pitch and amplitude are the same
//! breakpoint curves the editor will draw.

pub mod curve;
pub mod sources;

use i_am_dsp_derive::Parameters;

use crate::{Generator, NoteEvent};
use curve::{Curve, Point};
use sources::{Band, NoiseBand, Partial, Rng, Running};

/// The note the kick answers to, in the host's numbering.
pub const KICK: u8 = 36;

/// How many hits may ring at once.
const MAX_VOICES: usize = 16;

/// Everything one hit needs, built from the controls.
struct Recipe {
	body: Vec<Partial>,
	click: Vec<Partial>,
	noise: Vec<NoiseBand>,
	pitch: Curve,
	amp: Curve,
}

/// One hit: the running partials, the noise, and its place on the curves.
struct Voice {
	body: Vec<Running>,
	click: Vec<Running>,
	noise: Vec<Band>,
	pitch: Curve,
	amp: Curve,
	time: f32,
}

impl Voice {
	fn start(recipe: &Recipe, rate: f32, velocity: f32, rng: &mut Rng) -> Self {
		let velocity = velocity.clamp(0.0, 1.0);
		let body_gain = velocity.powf(1.5) * 0.9;
		let click_gain = velocity.powf(0.7);

		let scaled = |partials: &[Partial], gain: f32| {
			partials
				.iter()
				.map(|partial| Partial {
					amplitude: partial.amplitude * gain,
					..*partial
				})
				.collect::<Vec<_>>()
		};

		let body = scaled(&recipe.body, body_gain);
		let click = scaled(&recipe.click, click_gain);

		Self {
			body: body
				.iter()
				.map(|partial| Running::new(*partial, rate, rng))
				.collect(),
			click: click
				.iter()
				.map(|partial| Running::new(*partial, rate, rng))
				.collect(),
			noise: recipe
				.noise
				.iter()
				.map(|band| Band::new(*band, rate))
				.collect(),
			pitch: recipe.pitch.clone(),
			amp: recipe.amp.clone(),
			time: 0.0,
		}
	}

	fn next(&mut self, rng: &mut Rng, rate: f32) -> f32 {
		let ratio = self.pitch.value(self.time).clamp(0.05, 8.0);
		let mut sum = 0.0;

		// The body follows the pitch curve; the click is material, so it keeps
		// its own frequencies.
		for partial in &mut self.body {
			sum += partial.next(ratio, rate);
		}
		for partial in &mut self.click {
			sum += partial.next(1.0, rate);
		}
		for band in &mut self.noise {
			sum += band.next(rng);
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
	/// The fundamental of the kick, in Hz.
	#[range(min = 30.0, max = 120.0)]
	#[default(52.0)]
	#[unit(hertz)]
	#[id(display = "Tune")]
	#[layout(group = "Kick", order = 0, widget = knob)]
	pub tune: f32,
	/// How far the pitch falls in the first milliseconds.
	#[range(min = 1.0, max = 6.0)]
	#[default(3.2)]
	#[id(display = "Punch")]
	#[layout(group = "Kick", order = 1, widget = knob)]
	pub punch: f32,
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
}

impl Default for DrumInstrument {
	fn default() -> Self {
		Self {
			playing: Vec::new(),
			rate: 48_000.0,
			rng: Rng::default(),
			tune: 52.0,
			punch: 3.2,
			decay: 0.45,
			click: 0.5,
			click_tone: 1800.0,
			drive: 1.5,
			gain: 0.8,
		}
	}
}

impl DrumInstrument {
	/// A silent instrument, built for one sample rate.
	pub fn new(rate: usize) -> Self {
		Self {
			rate: if rate > 0 { rate as f32 } else { 48_000.0 },
			..Self::default()
		}
	}

	/// The controls, turned into one hit.
	fn recipe(&self) -> Recipe {
		let tune = self.tune.clamp(20.0, 200.0);
		let decay = self.decay.clamp(0.02, 4.0);
		let punch = self.punch.clamp(1.0, 8.0);

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

		let pitch = Curve::new(vec![
			Point {
				time: 0.0,
				value: punch,
				bend: -2.0,
			},
			Point {
				time: 0.03,
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
			body,
			click,
			noise,
			pitch,
			amp,
		}
	}

	/// Start one hit, if the note is the kick's.
	pub fn start(&mut self, note: u8, velocity: f32) -> bool {
		if note != KICK {
			return false;
		}

		if self.playing.len() >= MAX_VOICES {
			self.playing.remove(0);
		}

		let recipe = self.recipe();
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

		assert!(kit.recipe().click.is_empty());
		assert!(kit.recipe().noise.is_empty());
	}
}
