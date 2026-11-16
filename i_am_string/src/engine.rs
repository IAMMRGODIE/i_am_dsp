//! Polyphony, note lifecycle and the render loop.

use crate::Rng;
use crate::body::Body;
use crate::contact::{BowLaw, HammerLaw};
use crate::core::{Core, StringCore};
use crate::instrument::{Excitation, Preset, VoiceSpec};
use crate::string::WaveguideString;
use crate::sympathetic::SympatheticBank;

mod events;
mod params;

pub use events::Event;
pub use params::EngineParams;

mod voices;

use voices::*;

/// A polyphonic engine over the string cores in [crate::core].
pub struct StringEngine {
	sample_rate: f32,
	preset: Preset,
	params: EngineParams,
	voices: Vec<Voice>,
	max_voices: usize,
	age: u64,
	dc: [[f32; 2]; 2],
	/// The one pole of the DC blocker, computed for this host rate.
	dc_coefficient: f32,
	/// A last DC blocker on the output, after the body and the gain.
	output_dc: [[f32; 2]; 2],
	/// Its pole, lower than the one before the body, so it only removes an
	/// offset and leaves the bottom of the tail alone.
	output_dc_coefficient: f32,
	/// Start the DC blocker on the first sample of a note that begins in
	/// silence, so the string's static force is not heard as a thump.
	prime_dc: bool,
	body_left: Body,
	body_right: Body,
	/// A few milliseconds on one side, to give the sympathetic strings width.
	spread: crate::delay::DelayLine,
	dispersion_cache: std::collections::HashMap<u8, Vec<f32>>,
	/// The undamped strings, which ring sympathetically and carry the damper.
	sympathetic: SympatheticBank,
	/// The sustain pedal, from 0 (up) to 1 (fully down).
	pedal: f32,
	/// Which keys are still held down.
	keys_held: [bool; 128],
	control_counter: u32,
	attack: Attack,
}

/// The bridge force a full velocity note is normalised to.
const TARGET_FORCE: f32 = 0.6;

/// A note held for less than this is released as a stab, in seconds.
const SHORT_NOTE: f32 = 0.12;

/// The fade a bowed note starts with, in seconds.
///
/// The bow's first slip is a step in the force, and a bowed string has no
/// transient to hide it behind: the fade gives the note a start instead of a
/// click, short enough that the bow still bites.
const BOW_FADE_SECONDS: f32 = 0.08;

/// How often the vibrato's pitch ratio is recomputed, in samples.
///
/// Sixteen is three kilohertz at 48, far above the five hertz a hand can shake,
/// and a sixteenth of the `powf` the sample rate would otherwise ask for.
const BEND_INTERVAL: u64 = 16;

/// How many low partials each sympathetic string contributes.
const SYMPATHETIC_PARTIALS: usize = 2;
/// The lowest note that has a sympathetic string.
const SYMPATHETIC_LOWEST_NOTE: u8 = 21;
/// The highest note that has a sympathetic string.
const SYMPATHETIC_HIGHEST_NOTE: u8 = 108;
/// Undamped strings are not allowed to ring forever; this caps their T60.
///
/// A resonator takes its own T60 divided by 6.9 to reach two thirds of its
/// steady state, so a twelve second cap made the sympathetic bank arrive a
/// third of a second after the note and push the peak of the attack out to
/// 130 ms. A real instrument's undamped strings are coupled to the bridge
/// and answer within tens of milliseconds, so the cap is four seconds.
const SYMPATHETIC_MAX_T60: f32 = 4.0;
/// The T60 of a string with the felt fully resting on it.
const SYMPATHETIC_DAMPED_T60: f32 = 0.12;
/// The dampers, the pedal and the sympathetic bank are updated this often.
const CONTROL_INTERVAL: u32 = 64;

/// The per-round-trip loop gain of a damper felt that is `engagement` down.
///
/// Zero engagement is a free string, so the gain is one; one is felt resting on
/// the string, which is the damped T60 the preset asks for. In between, the
/// damping rises faster than the pedal falls, because on a real instrument the
/// dampers lift over the first part of the pedal travel and the useful half
/// pedal lives near the bottom of it. A linear map would make half pedalling
/// almost inaudible.
fn damper_gain(frequency: f32, engagement: f32, bowed: bool) -> f32 {
	let engagement = engagement.clamp(0.0, 1.0);
	if engagement <= 0.0 {
		return 1.0;
	}
	let brightness = (frequency / 4186.0).sqrt().clamp(0.0, 1.0);
	let damped_t60 = if bowed {
		0.30
	} else {
		0.25 - 0.13 * brightness
	};
	let extra = 6.907_755 / damped_t60 * engagement.powf(1.5);
	(-extra / frequency.max(1.0)).exp().clamp(0.0, 1.0)
}

impl StringEngine {
	/// Create an engine that renders at `sample_rate` hertz.
	pub fn new(sample_rate: f32) -> Self {
		let sample_rate = if sample_rate > 0.0 {
			sample_rate
		} else {
			48_000.0
		};
		let preset = Preset::Violin;
		let mut engine = Self {
			sample_rate,
			preset,
			params: preset.defaults(),
			voices: Vec::new(),
			max_voices: 24,
			age: 0,
			dc: [[0.0; 2]; 2],
			// Eleven hertz: below every note's fundamental, so only the static
			// force is removed, and the same corner whatever the host rate.
			dc_coefficient: (-std::f32::consts::TAU * 11.0 / sample_rate).exp(),
			output_dc: [[0.0; 2]; 2],
			output_dc_coefficient: (-std::f32::consts::TAU * 4.0 / sample_rate).exp(),
			prime_dc: false,
			body_left: Body::for_preset(preset, sample_rate, -1.0),
			body_right: Body::for_preset(preset, sample_rate, 1.0),
			spread: crate::delay::DelayLine::new((sample_rate * 0.002).round()),
			dispersion_cache: std::collections::HashMap::new(),
			sympathetic: SympatheticBank::new(),
			pedal: 0.0,
			keys_held: [false; 128],
			control_counter: 0,
			attack: Attack::new(),
		};
		engine.attack.prepare(sample_rate);
		engine.rebuild_sympathetic();
		engine
	}

	/// The sample rate the engine renders at.
	pub fn sample_rate(&self) -> f32 {
		self.sample_rate
	}

	/// The current preset.
	pub fn preset(&self) -> Preset {
		self.preset
	}

	/// Switch preset, keeping the current player controls.
	pub fn set_preset(&mut self, preset: Preset) {
		self.preset = preset;
		// Which core is in use is a choice about the model, not about the
		// instrument, so changing instrument keeps it. Otherwise comparing the
		// two cores would mean setting the switch again after every preset.
		let core = self.params.core;
		self.params = preset.defaults();
		self.params.core = core;
		self.body_left = Body::for_preset(preset, self.sample_rate, -1.0);
		self.body_right = Body::for_preset(preset, self.sample_rate, 1.0);
		self.dispersion_cache.clear();
		self.rebuild_sympathetic();
	}

	/// Switch preset, copying the new preset's defaults into the controls.
	pub fn preset_params(&self) -> EngineParams {
		self.preset.defaults()
	}

	/// The player controls.
	pub fn params(&self) -> &EngineParams {
		&self.params
	}

	/// The player controls, mutably.
	pub fn params_mut(&mut self) -> &mut EngineParams {
		&mut self.params
	}

	/// Adopt a new set of controls.
	///
	/// Fitted dispersion filters are dropped only when a control that actually
	/// changes the string design has moved, and the sympathetic bank is
	/// rebuilt only when its level wakes up or falls asleep.
	pub fn apply_params(&mut self, params: EngineParams) {
		if params.affects_string_design(&self.params) {
			self.dispersion_cache.clear();
		}
		let was_audible = self.params.sympathetic > 0.0;
		let is_audible = params.sympathetic > 0.0;
		let retune = params.affects_string_design(&self.params);
		self.params = params;
		if was_audible != is_audible || retune {
			self.rebuild_sympathetic();
		}
	}

	/// The sustain pedal position, from 0 (up) to 1 (fully down).
	pub fn pedal(&self) -> f32 {
		self.pedal
	}

	/// Move the sustain pedal.
	///
	/// Anything in between is a half pedal: the felt only brushes the strings,
	/// so they decay rather than stopping.
	pub fn set_pedal(&mut self, position: f32) {
		self.pedal = position.clamp(0.0, 1.0);
	}

	/// The maximum number of simultaneous notes.
	pub fn set_max_voices(&mut self, voices: usize) {
		self.max_voices = voices.max(1);
	}

	/// The number of notes currently sounding.
	pub fn active_voices(&self) -> usize {
		self.voices.len()
	}

	/// Start a note.
	pub fn note_on(&mut self, note: u8, velocity: f32) {
		let velocity = velocity.clamp(0.0, 1.0);
		if velocity <= 0.0 {
			self.note_off(note);
			return;
		}
		self.keys_held[note.min(127) as usize] = true;

		// A note that starts in silence carries the string's static force as a
		// step; starting the blocker on that step keeps it out of the thump.
		// A tail below a thousandth of full scale is inaudible, so starting the
		// blocker on this note's step cannot click the one it replaces.
		if self.voices.iter().all(|voice| voice.peak < 1e-3) {
			self.prime_dc = true;
		}

		self.release_voice(note);
		while self.voices.len() >= self.max_voices {
			self.steal_voice();
		}
		let spec = self.preset.spec(note, velocity, &self.params);
		// A bow needs none of this: it has its own rosin noise.
		let scale = TARGET_FORCE * self.params.gain;
		match spec.excitation {
			// The mechanism noise grows with the dynamics, but less steeply than
			// the note itself, which is why a very soft note is mostly mechanism.
			Excitation::Hammer => {
				self.attack.trigger(
					300.0 + 500.0 * velocity,
					0.8,
					scale * 0.30 * velocity.powf(1.6),
					0.010,
					self.sample_rate,
				);
				// The hard edge of the hammer on the string, gone in under two
				// milliseconds. A felt hammer is soft, but it is not a filter.
				self.attack.trigger_tick(
					2600.0 + 1800.0 * velocity,
					0.6,
					scale * 0.30 * velocity.powf(1.2),
					0.0018,
					self.sample_rate,
				);
				// The keybed and the board knock. From G#4 upwards it belongs
				// to the note rather than to the room, and a chord of them is
				// the sound of the action, so it fades in over an octave.
				let body = ((note as f32 - 68.0) / 12.0).clamp(0.0, 1.0);
				self.attack.trigger_knock(scale * 0.20 * velocity * body);
				// The felt lets go of the string over a few milliseconds; the
				// scrape is the noise of it, highpassed and gone in 25 ms.
				self.attack.trigger_scrape(
					2500.0,
					0.7,
					scale * 0.02 * velocity.powf(1.8),
					0.025,
					self.sample_rate,
				);
			}
			Excitation::Pluck => {
				self.attack.trigger(
					1500.0 + 2500.0 * velocity,
					0.9,
					scale * 0.40 * velocity.powf(1.6),
					0.004,
					self.sample_rate,
				);
				self.attack.trigger_tick(
					4000.0 + 2000.0 * velocity,
					0.5,
					scale * 0.06 * velocity.powf(1.5),
					0.0012,
					self.sample_rate,
				);
			}
			// A bow bites rather than clicks, so its noise is brighter and
			// lasts longer than a hammer's.
			Excitation::Bow => self.attack.trigger(
				2200.0 + 1800.0 * velocity,
				0.6,
				scale * 0.22 * velocity.powf(1.4),
				0.045,
				self.sample_rate,
			),
		}
		let voice = self.build_voice(note, &spec);
		self.voices.push(voice);
	}

	/// Release a note, letting its damper fall.
	pub fn note_off(&mut self, note: u8) {
		self.keys_held[note.min(127) as usize] = false;
		self.release_voice(note);
	}

	/// Release every note.
	pub fn all_notes_off(&mut self) {
		self.keys_held = [false; 128];
		// A piano's damper is a mechanism: the felt falls the same way however
		// long the key was held, and how far it falls is the pedal's business.
		// A hand on a guitar or a bowed string is not a mechanism, so on those
		// the release follows the gesture.
		let expressive = true;
		for voice in &mut self.voices {
			Self::engage_damper(voice, self.sample_rate, expressive);
		}
	}

	/// Apply an [Event].
	pub fn handle_event(&mut self, event: Event) {
		match event {
			Event::NoteOn { note, velocity } => self.note_on(note, velocity),
			Event::NoteOff { note } => self.note_off(note),
			Event::AllNotesOff => self.all_notes_off(),
			Event::Sustain { position } => self.set_pedal(position),
		}
	}

	/// Render one stereo frame.
	pub fn next_frame(&mut self) -> [f32; 2] {
		let dt = 1.0 / self.sample_rate;
		let mut left = 0.0;
		let mut right = 0.0;
		// The bank is excited by the strings as they are, not by the AC part:
		// its own resonances are what the undamped strings answer.
		let mut raw_left = 0.0;
		let mut raw_right = 0.0;
		let params = &self.params;
		for voice in &mut self.voices {
			voice.damper += (voice.damper_target - voice.damper) * voice.damper_rate;

			voice.samples += 1;
			voice.age_seconds += dt;

			// A powf and a sin per sample per voice is the most expensive thing
			// in the loop, and the hand moves far slower than the sample rate.
			if voice.samples % BEND_INTERVAL == 0 || voice.samples == 1 {
				let cents = Self::bend(voice, params, dt * BEND_INTERVAL as f32);

				voice.ratio = 2f32.powf(cents / 1200.0);
			}

			let ratio = voice.ratio;
			let mut value_left = 0.0;
			let mut value_right = 0.0;
			for (index, string) in voice.strings.iter_mut().enumerate() {
				string.set_damper(voice.damper);
				string.set_pitch_ratio(ratio);
				let sample = string.step();
				let (panned_left, panned_right) = voice.pans[index];
				value_left += sample * panned_left;
				value_right += sample * panned_right;
			}
			for string in &voice.strings {
				voice.peak = voice.peak.max(string.envelope());
			}
			// A bow bites the string; its first slip is a step too, and a short
			// fade in takes the edge off without slowing the bow's own attack.
			if voice.fade_in > 0 && voice.samples < voice.fade_in {
				let phase = voice.samples as f32 / voice.fade_in as f32;
				let fade = 0.5 - 0.5 * (std::f32::consts::PI * phase).cos();

				value_left *= fade;
				value_right *= fade;
			}

			// The string's static force is a step at the start of every note.
			// Blocking it per voice means a note played over a ringing tail is
			// cleaned up too, and starting the blocker on the step keeps its own
			// ring out of the attack.
			if !voice.dc_primed {
				voice.dc_primed = true;
				voice.dc = [[value_left, 0.0], [value_right, 0.0]];
			}

			let blocked_left = value_left - voice.dc[0][0] + self.dc_coefficient * voice.dc[0][1];
			voice.dc[0] = [value_left, blocked_left];
			let blocked_right = value_right - voice.dc[1][0] + self.dc_coefficient * voice.dc[1][1];
			voice.dc[1] = [value_right, blocked_right];

			left += blocked_left * voice.gain;
			right += blocked_right * voice.gain;
			raw_left += value_left * voice.gain;
			raw_right += value_right * voice.gain;
			if voice.bowed && voice.released {
				for string in &mut voice.strings {
					string.release_ports(dt);
				}
			}
		}
		// A hammer takes a millisecond to bite and a bow longer still, so a
		// voice that has never made a sound yet gets a grace period before it is
		// reaped as silent.
		let grace = self.sample_rate as u64;
		self.voices.retain(|voice| {
			let quiet = voice.strings.iter().all(|s| s.is_quiet());
			!quiet || (voice.peak <= 0.0 && voice.samples < grace)
		});

		// The dampers and the sympathetic strings move at a control rate; a
		// felt does not need 48 kHz resolution and the resonators only need
		// their coefficients reprogrammed when a damper has actually moved.
		self.control_counter += 1;
		if self.control_counter >= CONTROL_INTERVAL {
			let elapsed = dt * self.control_counter as f32;
			self.control_counter = 0;
			self.update_dampers(elapsed);
		}

		// The undamped strings answer everything the played strings do. The
		// coupling is one way on purpose: nothing is fed back into the voices,
		// so this can only add a tail, never instability.
		let sympathetic_amount = self.params.sympathetic;
		let sympathetic =
			self.sympathetic.process((raw_left + raw_right) * 0.5) * sympathetic_amount;

		// The undamped strings are all over the instrument, so they are given
		// width rather than a position.
		let spread = self.spread.tap(1.0);
		self.spread.push(sympathetic);
		let noise = self.attack.process();
		let sources = [
			left + sympathetic + noise,
			right + spread * 0.6 + sympathetic * 0.8 + noise,
		];

		let amount = self.params.body.clamp(0.0, 1.0);
		let prime = std::mem::take(&mut self.prime_dc);
		let mut frame = [0.0; 2];
		for (channel, source) in sources.into_iter().enumerate() {
			// A DC blocker, because an asymmetric contact law can leave an
			// offset on the string that would otherwise show up as a thump. It
			// sits before the body: the board has no DC response either, and
			// feeding it the step is what turns the offset into a knock.
			if prime {
				self.dc[channel] = [source, 0.0];
			}

			let blocked = source - self.dc[channel][0] + self.dc_coefficient * self.dc[channel][1];
			self.dc[channel][0] = source;
			self.dc[channel][1] = blocked;

			let body = if channel == 0 {
				&mut self.body_left
			} else {
				&mut self.body_right
			};
			let shaped = body.process(blocked);
			let mixed = blocked + (shaped - blocked) * amount;
			let output = mixed * self.params.gain;
			// Nothing downstream should see an offset either: the gain, the body
			// mix and a quiet tail can all leave a little, and it is free to
			// remove it once more at the very end.
			let final_blocked = output - self.output_dc[channel][0]
				+ self.output_dc_coefficient * self.output_dc[channel][1];
			self.output_dc[channel] = [output, final_blocked];
			frame[channel] = final_blocked;
		}
		frame
	}

	/// Render one sample, mixed down to mono.
	pub fn next_sample(&mut self) -> f32 {
		let frame = self.next_frame();
		0.5 * (frame[0] + frame[1])
	}

	fn build_voice(&mut self, note: u8, spec: &VoiceSpec) -> Voice {
		let count = spec.strings.max(1);
		let core = self.params.core;
		// Only the waveguide has a fitted filter to share, and designing one
		// costs milliseconds, which is why it is cached per note.
		let coefficients = match core {
			Core::Waveguide => Some(self.fitted_dispersion(note, spec)),
			Core::Modal => None,
		};
		let mut strings = Vec::with_capacity(count);
		for i in 0..count {
			let offset = unison_offset(i, count) * spec.detune_cents;
			let mut config = spec.string.clone();
			// The unison strings share one filter: a detune of a few cents does
			// not move the dispersion curve meaningfully.
			config.dispersion_coefficients = coefficients.clone();
			config.frequency = spec.string.frequency * 2f32.powf(offset / 1200.0);
			let mut string = StringCore::new(core, self.sample_rate, &config);
			// A hammer is not perfectly parallel to the strings: it reaches the
			// three of a unison a fraction of a millimetre apart. The same
			// excitation on all three gives an attack that is phase-locked and
			// reads as synthetic, so each string is struck at its own point.
			let mut struck = spec.clone();
			struck.position =
				(spec.position + unison_offset(i, count) * 0.0015 + 0.0007 * (i as f32).sin())
					.clamp(0.02, 0.96);
			excite(&mut string, &struck);
			strings.push(string);
		}

		let gain = TARGET_FORCE / spec.reference_force * self.params.gain;
		// No panning by register: a piano is not a wide instrument, and pushing
		// the bass left and the treble right is a mixing trick, not acoustics.
		// The width comes from the unison strings and from the body.
		let centre = 0.0;
		let mut pans = Vec::with_capacity(count);
		for index in 0..count {
			let spread = if count > 1 {
				(index as f32 - (count as f32 - 1.0) * 0.5) * 0.55
			} else {
				0.0
			};
			let pan = (centre + spread).clamp(-0.95, 0.95);
			let angle = (pan + 1.0) * std::f32::consts::FRAC_PI_4;
			pans.push((angle.cos(), angle.sin()));
		}
		self.age += 1;
		Voice {
			note,
			frequency: spec.string.frequency,
			strings,
			damper: 1.0,
			damper_target: 1.0,
			release_engagement: 1.0,
			damper_rate: 0.004,
			gain,
			bowed: spec.excitation == Excitation::Bow,
			released: false,
			age: self.age,
			samples: 0,
			peak: 0.0,
			pans,
			age_seconds: 0.0,
			// Deliberately not zero. Every note of a chord starting its shake
			// from the same place sounds like one hand playing all of them.
			vibrato_phase: (note as f32 * 0.37).fract() * std::f32::consts::TAU,
			ratio: 1.0,
			fade_in: if spec.excitation == Excitation::Bow {
				(self.sample_rate * BOW_FADE_SECONDS) as u64
			} else {
				0
			},
			dc: [[0.0; 2]; 2],
			dc_primed: false,
		}
	}

	/// How far the left hand has bent this note, in cents.
	///
	/// A player does not switch a vibrato on. The note starts with a settled
	/// hand, the hand starts moving a moment later, and it takes a moment more
	/// to reach full depth - which is most of why a real vibrato sounds like
	/// expression and a sinusoidal one sounds like a fault.
	fn bend(voice: &mut Voice, params: &EngineParams, dt: f32) -> f32 {
		let rate = params.vibrato_rate.clamp(0.0, 20.0);
		voice.vibrato_phase += std::f32::consts::TAU * rate * dt;
		if voice.vibrato_phase > std::f32::consts::TAU {
			voice.vibrato_phase -= std::f32::consts::TAU;
		}
		let arrival = ((voice.age_seconds - params.vibrato_delay.max(0.0))
			/ params.vibrato_onset.max(1e-4))
		.clamp(0.0, 1.0);
		params.vibrato_depth.clamp(0.0, 200.0) * arrival * voice.vibrato_phase.sin()
	}

	/// Fit the dispersion filter for a note once, and reuse it afterwards.
	fn fitted_dispersion(&mut self, note: u8, spec: &VoiceSpec) -> Vec<f32> {
		if let Some(coefficients) = self.dispersion_cache.get(&note) {
			return coefficients.clone();
		}
		let probe = WaveguideString::new(self.sample_rate, spec.string.clone());
		let coefficients = probe.dispersion_coefficients();
		self.dispersion_cache.insert(note, coefficients.clone());
		coefficients
	}

	fn release_voice(&mut self, note: u8) {
		let sample_rate = self.sample_rate;
		let expressive = true;
		for voice in &mut self.voices {
			if voice.note == note && !voice.released {
				Self::engage_damper(voice, sample_rate, expressive);
			}
		}
	}

	fn engage_damper(voice: &mut Voice, sample_rate: f32, expressive: bool) {
		voice.released = true;
		let dt = 1.0 / sample_rate;
		// How a player lets go depends on how long the note was held. A note
		// that was only stabbed is cut with the hand already on the string,
		// in a couple of milliseconds and all the way down. One that has been
		// ringing for a while is released into the board: the fade is slower
		// and it stops short of a mute, which is the difference between a
		// plucked note that is damped and one that is simply let go.
		let (engagement, rate) = if voice.bowed {
			(1.0, 0.004)
		} else if !expressive {
			// The mechanised damper: full felt, at the rate the pedal asks for.
			(1.0, 0.004)
		} else if voice.age_seconds < SHORT_NOTE {
			(1.0, 0.012)
		} else {
			(0.35, 0.0015)
		};
		voice.release_engagement = engagement;
		voice.damper_rate = rate;
		// The felt starts falling at once; how far it gets depends on the pedal,
		// which update_dampers works out every control interval.
		voice.damper_target = damper_gain(voice.frequency, engagement, voice.bowed);
		for string in &mut voice.strings {
			string.release_ports(dt);
		}
	}

	/// Rebuild the sympathetic bank from the current preset and controls.
	fn rebuild_sympathetic(&mut self) {
		self.sympathetic.clear();
		if self.params.sympathetic <= 0.0 {
			return;
		}
		let sample_rate = self.sample_rate;
		let nyquist = sample_rate * 0.45;
		let open_floor = 6.907_755 / SYMPATHETIC_MAX_T60;
		let damped = 6.907_755 / SYMPATHETIC_DAMPED_T60;
		for note in SYMPATHETIC_LOWEST_NOTE..=SYMPATHETIC_HIGHEST_NOTE {
			let spec = self.preset.spec(note, 1.0, &self.params);
			let mut partials = Vec::with_capacity(SYMPATHETIC_PARTIALS);
			for index in 1..=SYMPATHETIC_PARTIALS {
				let order = index as f32;
				let frequency = spec.string.frequency
					* order
					* (1.0 + spec.string.inharmonicity * order * order).sqrt();
				if frequency >= nyquist {
					break;
				}
				let decay = spec.string.sigma0 + spec.string.sigma1 * frequency * frequency;
				partials.push((frequency, decay.max(open_floor), damped));
			}
			if !partials.is_empty() {
				self.sympathetic.set_string(note, partials, sample_rate);
			}
		}
	}

	/// Work out where every damper is, and move it.
	fn update_dampers(&mut self, dt: f32) {
		let open = (1.0 - self.pedal).clamp(0.0, 1.0);
		for voice in &mut self.voices {
			let engagement = if voice.released {
				voice.release_engagement * open
			} else {
				0.0
			};
			voice.damper_target = damper_gain(voice.frequency, engagement, voice.bowed);
		}
		let held = &self.keys_held;
		self.sympathetic
			.set_engagement_by(|note| if held[note as usize] { 0.0 } else { open });
		self.sympathetic.update(dt);
	}

	fn steal_voice(&mut self) {
		if self.voices.is_empty() {
			return;
		}
		let mut best = 0;
		let mut best_score = f32::INFINITY;
		for (i, voice) in self.voices.iter().enumerate() {
			let envelope = voice
				.strings
				.iter()
				.map(|s| s.envelope())
				.fold(0.0f32, f32::max);
			let score = if voice.released {
				envelope * 0.25
			} else {
				envelope
			};
			// Quietest first, and the oldest when two are equally quiet.
			if score < best_score || (score <= best_score && voice.age < self.voices[best].age) {
				best_score = score;
				best = i;
			}
		}
		self.voices.remove(best);
	}
}

/// The detune multiplier of string `index` out of `count`.
fn unison_offset(index: usize, count: usize) -> f32 {
	if count <= 1 {
		return 0.0;
	}
	index as f32 - (count as f32 - 1.0) * 0.5
}

fn excite(string: &mut StringCore, spec: &VoiceSpec) {
	match spec.excitation {
		Excitation::Pluck => {
			string.pluck_with_release(spec.position, spec.pluck_amplitude, spec.pluck_release)
		}
		Excitation::Hammer => string.add_port(
			spec.position,
			Box::new(HammerLaw::new(
				spec.hammer_mass,
				spec.hammer_velocity,
				spec.hammer_exponent,
				spec.hammer_stiffness,
				0.0,
			)),
			spec.contact_substeps,
		),
		Excitation::Bow => {
			let mut law = BowLaw::new(spec.bow_force, spec.bow_velocity);
			// Rosin is not smooth: the surface noise is what starts the
			// stick-slip cycle, and it doubles as the bow's own noise floor.
			law.noise = 0.0015;
			string.add_port(spec.position, Box::new(law), spec.contact_substeps);
		}
	}
}
