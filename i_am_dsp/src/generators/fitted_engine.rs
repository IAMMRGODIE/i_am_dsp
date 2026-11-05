//! A fitted instrument: a mode table played back, not a string model.
//!
//! The adapter for `i_am_string::fitted`: it turns [NoteEvent]s into voices,
//! applies the velocity curve and the release, and exposes the player controls.

use i_am_dsp_derive::Parameters;

use i_am_string::fitted;

use crate::generators::adsr::bend;
use crate::{Generator, NoteEvent};

const MAX_VOICES: usize = 32;

/// The table is numbered the way the library numbers notes: A4 is 57.
fn to_table(note: usize) -> u8 {
	note.clamp(0, 127) as u8
}

struct Playing {
	voice: fitted::Voice,
	note: u8,
	fade: f32,
	releasing: bool,
}

/// A polyphonic instrument played from a fitted mode table.
#[derive(Parameters)]
pub struct FittedInstrument {
	#[skip]
	playing: Vec<Playing>,
	#[skip]
	sample_rate: f32,
	/// The overall level.
	#[range(min = 0.0, max = 1.0)]
	#[default(0.8)]
	#[id(display = "Gain")]
	#[layout(group = "Voice", order = 0, widget = knob)]
	pub gain: f32,
	/// The velocity curve: negative is a soft keyboard, positive a hard one.
	#[range(min = -4.0, max = 4.0)]
	#[default(0.0)]
	#[id(display = "Velocity Curve")]
	#[layout(group = "Voice", order = 1, widget = knob)]
	pub velocity_curve: f32,
	/// How long a released note takes to fade, in seconds.
	#[range(min = 0.02, max = 2.0)]
	#[default(0.15)]
	#[logarithmic]
	#[id(display = "Release")]
	#[layout(group = "Voice", order = 2, widget = knob)]
	pub release: f32,
}

impl Default for FittedInstrument {
	fn default() -> Self {
		Self {
			playing: Vec::new(),
			sample_rate: 48_000.0,
			gain: 0.8,
			velocity_curve: 0.0,
			release: 0.15,
		}
	}
}

impl FittedInstrument {
	/// A silent instrument, built for one sample rate.
	pub fn new(rate: usize) -> Self {
		Self {
			sample_rate: if rate > 0 { rate as f32 } else { 48_000.0 },
			..Self::default()
		}
	}

	fn start(&mut self, note: u8, velocity: f32) {
		if self.playing.len() >= MAX_VOICES {
			self.playing.remove(0);
		}
		let shaped = bend(velocity.clamp(0.0, 1.0), self.velocity_curve) * self.gain;
		self.playing.push(Playing {
			voice: fitted::Voice::start(note, velocity.clamp(0.0, 1.0), shaped, self.sample_rate),
			note,
			fade: 1.0,
			releasing: false,
		});
	}

	fn release(&mut self, note: u8) {
		for voice in &mut self.playing {
			if voice.note == note {
				voice.releasing = true;
			}
		}
	}

	fn release_all(&mut self) {
		for voice in &mut self.playing {
			voice.releasing = true;
		}
	}
}

impl Generator<2> for FittedInstrument {
	fn generate(&mut self, process_context: &mut Box<dyn crate::ProcessContext>) -> [f32; 2] {
		let rate = process_context.infos().sample_rate as f32;
		if rate > 0.0 && (rate - self.sample_rate).abs() > 0.5 {
			self.sample_rate = rate;
			self.playing.clear();
		}

		let events = process_context.events().to_vec();
		process_context.clear_events();
		for event in &events {
			match event {
				NoteEvent::NoteOn { note, velocity, .. } => self.start(to_table(*note), *velocity),
				NoteEvent::NoteOff { note, .. } | NoteEvent::Stop { note, .. } => {
					self.release(to_table(*note))
				}
				NoteEvent::ImmediateStop => self.release_all(),
				_ => {}
			}
		}

		let step = 1.0 / (self.release.max(0.02) * self.sample_rate);
		let mut mixed = 0.0f32;
		for voice in &mut self.playing {
			let sample = voice.voice.next_sample();
			if voice.releasing {
				voice.fade = (voice.fade - step).max(0.0);
			}
			mixed += sample * voice.fade;
		}
		self.playing.retain(|voice| !voice.releasing || voice.fade > 0.0);
		[mixed, mixed]
	}
}