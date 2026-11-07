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
/// The closed hi-hat's note.
pub const HAT_CLOSED: u8 = 42;
/// The pedal hi-hat's note.
pub const HAT_PEDAL: u8 = 44;
/// The open hi-hat's note.
pub const HAT_OPEN: u8 = 46;
/// The clap's note.
pub const CLAP: u8 = 39;
/// The lowest tom's note.
pub const TOM_LOW: u8 = 41;
/// The highest tom's note.
pub const TOM_HIGH: u8 = 48;
/// The crash's note.
pub const CRASH: u8 = 49;
/// The ride's note.
pub const RIDE: u8 = 51;

/// How many hits may ring at once.
const MAX_VOICES: usize = 16;

/// A diffuse, slowly shifting tail for the cymbals.
///
/// The bank on its own is a comb: modes that are close to evenly spaced and all
/// ring for the same time read as points rather than as metal. Four combs and
/// two allpasses scatter that into a tail, and the frequency shifter inside the
/// loop moves every pass a few hertz so the tail never settles into a harmonic
/// series.
struct CymbalTail {
	combs: Vec<Comb>,
	allpasses: Vec<Allpass>,
	shifter: crate::effects::prelude::HilbertTransform<4, 1>,
	phase: f32,
	shift: f32,
	feedback: f32,
	loop_state: f32,
}

/// A damped comb, the classic Schroeder building block.
struct Comb {
	buffer: Vec<f32>,
	index: usize,
	feedback: f32,
	damping: f32,
	state: f32,
}

impl Comb {
	fn new(seconds: f32, t60: f32, rate: f32) -> Self {
		let samples = ((seconds * rate).round() as usize).max(16);
		let feedback = 10f32.powf(-3.0 * (samples as f32 / rate) / t60.max(0.05));

		Self {
			buffer: vec![0.0; samples],
			index: 0,
			feedback: feedback.clamp(0.05, 0.97),
			damping: (-std::f32::consts::TAU * 8_000.0 / rate).exp(),
			state: 0.0,
		}
	}

	#[inline]
	fn process(&mut self, input: f32) -> f32 {
		let out = self.buffer[self.index];

		self.state += (out - self.state) * self.damping;
		self.buffer[self.index] = input + self.state * self.feedback;
		self.index += 1;

		if self.index >= self.buffer.len() {
			self.index = 0;
		}

		out
	}
}

/// A Schroeder allpass, which turns an echo into a smear.
struct Allpass {
	buffer: Vec<f32>,
	index: usize,
	gain: f32,
}

impl Allpass {
	fn new(seconds: f32, rate: f32) -> Self {
		Self {
			buffer: vec![0.0; ((seconds * rate).round() as usize).max(8)],
			index: 0,
			gain: 0.5,
		}
	}

	#[inline]
	fn process(&mut self, input: f32) -> f32 {
		let buffered = self.buffer[self.index];
		let out = buffered - self.gain * input;

		self.buffer[self.index] = input + self.gain * out;
		self.index += 1;

		if self.index >= self.buffer.len() {
			self.index = 0;
		}

		out
	}
}

impl CymbalTail {
	fn new(rate: f32, t60: f32, shift: f32) -> Self {
		Self {
			combs: [0.0297, 0.0371, 0.0411, 0.0437]
				.iter()
				.map(|seconds| Comb::new(*seconds, t60, rate))
				.collect(),
			allpasses: [0.0050, 0.0017]
				.iter()
				.map(|seconds| Allpass::new(*seconds, rate))
				.collect(),
			shifter: crate::effects::prelude::HilbertTransform::new(rate as usize),
			phase: 0.0,
			shift: shift.max(0.0),
			feedback: 0.45,
			loop_state: 0.0,
		}
	}

	#[inline]
	fn process(&mut self, input: f32, rate: f32) -> f32 {
		let loop_in = input + self.loop_state * self.feedback;
		let mut wet = 0.0;

		for comb in &mut self.combs {
			wet += comb.process(loop_in);
		}
		wet *= 0.25;

		for allpass in &mut self.allpasses {
			wet = allpass.process(wet);
		}

		let analytic = self.shifter.apply_transform_single(wet, 0);

		self.phase += std::f32::consts::TAU * self.shift / rate;
		if self.phase > std::f32::consts::TAU {
			self.phase -= std::f32::consts::TAU;
		}

		let (sin, cosine) = self.phase.sin_cos();
		self.loop_state = analytic.real * cosine - analytic.imag * sin;

		wet
	}
}

/// What the tail of a note needs when it is a cymbal.
#[derive(Clone, Copy)]
struct TailSpec {
	t60: f32,
	shift: f32,
	level: f32,
}

/// What a layer is made of.
enum Source {
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
	metal: Option<MetalVoice>,
	keytrack: f32,
}

/// One running metal cluster: six square waves and the highpass after them.
struct MetalVoice {
	frequencies: [f32; 6],
	phases: [f32; 6],
	level: f32,
	decay: f32,
	highpass: f32,
	previous_in: f32,
	previous_out: f32,
}

impl MetalVoice {
	fn new(frequencies: [f32; 6], t60: f32, gain: f32, cutoff: f32, rate: f32) -> Self {
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
	fn next(&mut self, rate: f32) -> f32 {
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
struct Voice {
	layers: Vec<VoiceLayer>,
	pitch: Curve,
	amp: Curve,
	time: f32,
	/// The diffuse tail under a cymbal, if this note has one.
	tail: Option<CymbalTail>,
	/// How long the tail is allowed to ring after the amp curve ends.
	tail_grace: f32,
	/// How loud the tail sits under the bank.
	tail_level: f32,
	/// 1 while the note rings, ramping to 0 when another hat chokes it.
	choke: f32,
	choking: bool,
	/// The note this voice was started by, so a hat can choke the other one.
	note: u8,
}

impl Voice {
	fn start(
		recipe: &Recipe,
		note: u8,
		rate: f32,
		velocity: f32,
		tail: Option<TailSpec>,
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
					metal: None,
					keytrack: layer.keytrack,
				}),
				Source::Metal {
					frequencies,
					t60,
					cutoff,
				} => layers.push(VoiceLayer {
					partials: Vec::new(),
					noise: Vec::new(),
					metal: Some(MetalVoice::new(
						*frequencies,
						*t60,
						level,
						*cutoff,
						rate,
					)),
					keytrack: layer.keytrack,
				}),
			}
		}

		Self {
			layers,
			tail: tail.map(|spec| CymbalTail::new(rate, spec.t60, spec.shift)),
			tail_grace: tail.map(|spec| spec.t60 * 0.8).unwrap_or(0.0),
			tail_level: tail.map(|spec| spec.level).unwrap_or(0.0),
			pitch: recipe.pitch.clone(),
			amp: recipe.amp.clone(),
			time: 0.0,
			choke: 1.0,
			choking: false,
			note,
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
			if let Some(metal) = &mut layer.metal {
				sum += metal.next(rate);
			}
		}

		let level = self.amp.value(self.time).max(0.0);
		self.time += 1.0 / rate;

		let mut out = sum * level;

		if let Some(tail) = &mut self.tail {
			out += tail.process(sum, rate) * self.tail_level;
		}

		if self.choking {
			self.choke = (self.choke - 1.0 / (0.004 * rate)).max(0.0);
			out *= self.choke;
		}

		out
	}

	fn finished(&self) -> bool {
		self.time > self.amp.end() + self.tail_grace + 0.02
			|| (self.choking && self.choke <= 0.0)
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
	/// The metal cluster's tone, as a multiplier on the six frequencies.
	#[range(min = 0.4, max = 2.5)]
	#[default(1.0)]
	#[id(display = "Hat Tone")]
	#[layout(group = "Hat", order = 0, widget = knob)]
	pub hat_tune: f32,
	/// How long a closed hat rings, in seconds.
	#[range(min = 0.01, max = 0.3)]
	#[default(0.05)]
	#[id(display = "Closed")]
	#[layout(group = "Hat", order = 1, widget = knob)]
	pub hat_decay: f32,
	/// How long an open hat rings, in seconds.
	#[range(min = 0.1, max = 2.0)]
	#[default(0.45)]
	#[id(display = "Open")]
	#[layout(group = "Hat", order = 2, widget = knob)]
	pub hat_open: f32,
	/// The highpass that makes the cluster metal rather than a chord.
	#[range(min = 1_500.0, max = 12_000.0)]
	#[default(6_500.0)]
	#[unit(hertz)]
	#[id(display = "Hat Cutoff")]
	#[layout(group = "Hat", order = 3, widget = knob)]
	pub hat_cutoff: f32,
	/// The hat's level.
	#[range(min = 0.0, max = 1.0)]
	#[default(0.6)]
	#[id(display = "Hat Level")]
	#[layout(group = "Hat", order = 4, widget = knob)]
	pub hat_gain: f32,
	/// Where the clap's noise sits, in Hz.
	#[range(min = 600.0, max = 5_000.0)]
	#[default(1_600.0)]
	#[unit(hertz)]
	#[id(display = "Clap Tone")]
	#[layout(group = "Clap", order = 0, widget = knob)]
	pub clap_tone: f32,
	/// How long the clap's tail lasts, in seconds.
	#[range(min = 0.08, max = 1.0)]
	#[default(0.35)]
	#[id(display = "Clap Decay")]
	#[layout(group = "Clap", order = 1, widget = knob)]
	pub clap_decay: f32,
	/// How long the toms ring, in seconds.
	#[range(min = 0.1, max = 1.5)]
	#[default(0.5)]
	#[id(display = "Tom Decay")]
	#[layout(group = "Tom", order = 0, widget = knob)]
	pub tom_decay: f32,
	/// A multiplier on the toms' shell pitch.
	#[range(min = 0.5, max = 2.0)]
	#[default(1.0)]
	#[id(display = "Tom Tune")]
	#[layout(group = "Tom", order = 1, widget = knob)]
	pub tom_tune: f32,
	/// How long the cymbals ring, in seconds.
	#[range(min = 0.5, max = 6.0)]
	#[default(2.5)]
	#[id(display = "Cymbal Decay")]
	#[layout(group = "Cymbal", order = 0, widget = knob)]
	pub cymbal_decay: f32,
	/// The cymbals' level.
	#[range(min = 0.0, max = 1.0)]
	#[default(0.5)]
	#[id(display = "Cymbal Level")]
	#[layout(group = "Cymbal", order = 1, widget = knob)]
	pub cymbal_gain: f32,
	/// How much of the diffuse, shifted tail the cymbals have.
	#[range(min = 0.0, max = 1.0)]
	#[default(0.8)]
	#[id(display = "Cymbal Tail")]
	#[layout(group = "Cymbal", order = 2, widget = knob)]
	pub cymbal_tail: f32,
}

/// A pitch curve that stays at one, for the drums whose note is their pitch.
fn flat_pitch() -> Curve {
	Curve::new(vec![
		Point {
			time: 0.0,
			value: 1.0,
			bend: 0.0,
		},
		Point {
			time: 4.0,
			value: 1.0,
			bend: 0.0,
		},
	])
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
			hat_tune: 1.0,
			hat_decay: 0.05,
			hat_open: 0.45,
			hat_cutoff: 6_500.0,
			hat_gain: 0.6,
			clap_tone: 1_600.0,
			clap_decay: 0.35,
			tom_decay: 0.5,
			tom_tune: 1.0,
			cymbal_decay: 2.5,
			cymbal_gain: 0.5,
			cymbal_tail: 0.8,
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

	/// Whether a note is one of the hat's three.
	fn is_hat(note: u8) -> bool {
		matches!(note, HAT_CLOSED | HAT_PEDAL | HAT_OPEN)
	}

	/// The hi-hat: six square waves through a highpass, with a short noise
	/// edge. Closed and open differ only in how long the cluster rings.
	fn hat_recipe(&self, note: u8) -> Recipe {
		let open = note == HAT_OPEN;
		let tune = self.hat_tune.clamp(0.4, 2.5);
		let t60 = if open {
			self.hat_open.clamp(0.1, 2.0)
		} else {
			self.hat_decay.clamp(0.01, 0.3)
		};
		let cutoff = self.hat_cutoff.clamp(1_500.0, 12_000.0);
		let level = self.hat_gain.clamp(0.0, 1.0);
		// The 808's six, which are not a harmonic series and not a chord.
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
		let amp = Curve::new(vec![
			Point {
				time: 0.0,
				value: 1.0,
				bend: 0.0,
			},
			Point {
				time: t60 * 1.3,
				value: 0.0,
				bend: 0.0,
			},
		]);

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
					source: Source::Noise(vec![NoiseBand {
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

	/// Whether a note is the clap.
	fn is_clap(note: u8) -> bool {
		note == CLAP
	}

	/// Whether a note is one of the toms.
	fn is_tom(note: u8) -> bool {
		(TOM_LOW..=TOM_HIGH).contains(&note)
	}

	/// Whether a note is one of the cymbals.
	fn is_cymbal(note: u8) -> bool {
		note == CRASH || note == RIDE
	}

	/// The clap: one noise source, retriggered four times and then let go.
	///
	/// The bursts are the amp curve, not four voices: a clap is a hand, four
	/// hands close together, and the tail they leave on the room.
	fn clap_recipe(&self) -> Recipe {
		let tone = self.clap_tone.clamp(400.0, 8_000.0);
		let decay = self.clap_decay.clamp(0.05, 2.0);

		let amp = Curve::new(vec![
			Point {
				time: 0.0,
				value: 1.0,
				bend: -3.0,
			},
			Point {
				time: 0.009,
				value: 0.0,
				bend: 0.0,
			},
			Point {
				time: 0.012,
				value: 0.9,
				bend: -3.0,
			},
			Point {
				time: 0.021,
				value: 0.0,
				bend: 0.0,
			},
			Point {
				time: 0.024,
				value: 0.8,
				bend: -2.5,
			},
			Point {
				time: 0.033,
				value: 0.0,
				bend: 0.0,
			},
			Point {
				time: 0.036,
				value: 0.7,
				bend: -1.5,
			},
			Point {
				time: 0.06,
				value: 0.32,
				bend: 0.0,
			},
			Point {
				time: decay,
				value: 0.0,
				bend: 0.0,
			},
		]);
		let pitch = flat_pitch();

		Recipe {
			layers: vec![
				Layer {
					source: Source::Noise(vec![NoiseBand {
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
						gain: 0.3,
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
		let frequency = 440.0
			* 2f32.powf((note as f32 - 69.0) / 12.0)
			* self.tom_tune.clamp(0.4, 2.5);
		let decay = self.tom_decay.clamp(0.05, 2.0);
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
	fn cymbal_recipe(&self, note: u8) -> Recipe {
		let ride = note == RIDE;
		let decay = self.cymbal_decay.clamp(0.3, 8.0) * if ride { 0.45 } else { 1.0 };
		let level = self.cymbal_gain.clamp(0.0, 1.0);
		let count = 56usize;
		let mut partials = Vec::with_capacity(count);

		for index in 0..count {
			let step = index as f32;
			// A plate's modes are dense and irregular: log spaced, then jittered.
			let scatter = ((step * 12.9898).sin() * 43_758.545).fract().abs();
			let frequency = 220.0 * 1.105f32.powf(step) * (0.94 + 0.12 * scatter);
			let amplitude = (1.0 / (1.0 + step).powf(0.9)) * (0.6 + 0.4 * scatter);
			let t60 = decay * (1.0 - 0.55 * (step / count as f32));

			partials.push(Partial {
				frequency: frequency.min(16_000.0),
				amplitude,
				t60: t60.max(0.05),
			});
		}

		let amp = Curve::new(vec![
			Point {
				time: 0.0,
				value: 1.0,
				bend: 0.0,
			},
			Point {
				time: decay * 0.95,
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
					gain: 0.5 * level,
				},
				Layer {
					source: Source::Noise(vec![NoiseBand {
						frequency: 9_000.0,
						q: 0.5,
						gain: 0.35 * level,
						t60: (decay * 0.12).max(0.05),
					}]),
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
		let hat = Self::is_hat(note);

		if !Self::is_kick(note)
			&& !Self::is_snare(note)
			&& !hat
			&& !Self::is_clap(note)
			&& !Self::is_tom(note)
			&& !Self::is_cymbal(note)
		{
			return false;
		}

		// A closed or pedal hat chokes the open one: that is the pair.
		if hat && note != HAT_OPEN {
			for voice in &mut self.playing {
				if voice.note == HAT_OPEN {
					voice.choking = true;
				}
			}
		}

		if self.playing.len() >= MAX_VOICES {
			self.playing.remove(0);
		}

		let recipe = if hat {
			self.hat_recipe(note)
		} else if Self::is_snare(note) {
			self.snare_recipe()
		} else if Self::is_clap(note) {
			self.clap_recipe()
		} else if Self::is_tom(note) {
			self.tom_recipe(note)
		} else if Self::is_cymbal(note) {
			self.cymbal_recipe(note)
		} else {
			self.kick_recipe()
		};
		let tail = if Self::is_cymbal(note) {
			let ride = note == RIDE;
			let decay = self.cymbal_decay.clamp(0.3, 8.0) * if ride { 0.45 } else { 1.0 };

			let level = self.cymbal_tail.clamp(0.0, 1.0);

			(level > 0.0).then_some(TailSpec {
				t60: decay,
				shift: if ride { 5.0 } else { 8.0 },
				level,
			})
		} else {
			None
		};

		self.playing.push(Voice::start(
			&recipe,
			note,
			self.rate,
			velocity,
			tail,
			&mut self.rng,
		));

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
	fn the_hats_sound_and_the_open_one_rings_longer() {
		let mut closed = DrumInstrument::new(48_000);
		assert!(closed.start(HAT_CLOSED, 1.0));
		let closed = render(&mut closed, 24_000);

		let mut open = DrumInstrument::new(48_000);
		assert!(open.start(HAT_OPEN, 1.0));
		let open = render(&mut open, 24_000);

		let peak = closed
			.iter()
			.fold(0.0f32, |peak, sample| peak.max(sample.abs()));
		assert!(peak > 0.02, "the closed hat was silent: {peak}");
		assert!(
			rms(&open[12_000..]) > rms(&closed[12_000..]) * 5.0,
			"the open hat died like a closed one: {} vs {}",
			rms(&open[12_000..]),
			rms(&closed[12_000..])
		);
	}

	#[test]
	fn a_closed_hat_chokes_an_open_one() {
		let mut free = DrumInstrument::new(48_000);
		free.start(HAT_OPEN, 1.0);
		let free = render(&mut free, 24_000);

		let mut choked = DrumInstrument::new(48_000);
		choked.start(HAT_OPEN, 1.0);
		render(&mut choked, 4_800);
		// The choke is the closed hat's gesture, not its sound: a silent one
		// still stops the open hat, and then the tail is the only thing left.
		choked.hat_gain = 0.0;
		choked.start(HAT_CLOSED, 1.0);
		let after = render(&mut choked, 4_800);

		let open_tail = rms(&free[4_800..6_240]);
		let choked_tail = rms(&after[0..1_440]);

		assert!(
			choked_tail < open_tail * 0.6,
			"the closed hat did not choke the open one: {choked_tail} vs {open_tail}"
		);
	}

	#[test]
	fn the_clap_retriggers() {
		let mut kit = DrumInstrument::new(48_000);
		assert!(kit.start(CLAP, 1.0));
		let out = render(&mut kit, 12_000);

		// A burst, a near silent gap, another burst: the four hands.
		let early = rms(&out[0..240]);
		let gap = rms(&out[432..480]);

		assert!(early > 0.01, "the clap was silent: {early}");
		assert!(
			gap < early * 0.25,
			"the clap did not retrigger: {early} then {gap}"
		);
	}

	#[test]
	fn the_toms_follow_the_note() {
		let crossings = |note: u8| {
			let mut kit = DrumInstrument::new(48_000);
			assert!(kit.start(note, 1.0));
			let out = render(&mut kit, 4_800);

			out.windows(2)
				.filter(|pair| (pair[0] >= 0.0) != (pair[1] >= 0.0))
				.count()
		};

		assert!(
			crossings(TOM_HIGH) > crossings(TOM_LOW) * 13 / 10,
			"the toms do not follow the note: {} vs {}",
			crossings(TOM_HIGH),
			crossings(TOM_LOW)
		);
	}

	#[test]
	fn the_cymbals_ring_longer_than_a_hat() {
		let tail = |note: u8| {
			let mut kit = DrumInstrument::new(48_000);
			kit.start(note, 1.0);
			let out = render(&mut kit, 96_000);

			rms(&out[48_000..])
		};

		let crash = tail(CRASH);
		let closed = tail(HAT_CLOSED);
		let ride = tail(RIDE);

		assert!(crash > 0.001, "the crash died: {crash}");
		assert!(
			crash > closed * 20.0,
			"the crash rang like a hat: {crash} vs {closed}"
		);
		assert!(ride < crash, "the ride outlasts the crash");
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
				Source::Metal { .. } => any = true,
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
