//! A fitted instrument: a mode table played back, not a string model.
//!
//! The adapter for `i_am_string::fitted`: it turns [NoteEvent]s into voices,
//! applies the velocity curve and the release, and exposes the player controls.

use i_am_dsp_derive::Parameters;

use serde::{Deserialize, Serialize};
use std::sync::Arc;

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
	/// Seconds since the note started, for the attack ramp.
	age: f32,
}

/// A table the user brought, saved with the plug-in state.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImportedTable {
	/// The file name, for the interface.
	pub name: String,
	/// The bytes the fitter wrote.
	pub bytes: Vec<u8>,
}

/// A polyphonic instrument played from a fitted mode table.
#[derive(Parameters)]
pub struct FittedInstrument {
	#[skip]
	playing: Vec<Playing>,
	#[skip]
	sample_rate: f32,
	#[skip]
	cached: Option<(u64, Arc<fitted::Instrument>)>,
	/// A table the user imported. Saved with the plug-in state.
	#[serde]
	#[id(display = "Imported Table")]
	pub imported: Option<ImportedTable>,
	/// How long the note takes to reach full level, in seconds. Short enough
	/// to be an attack rather than a swell, long enough to stop the click at
	/// the start of a note.
	#[range(min = 0.0, max = 0.08)]
	#[default(0.001)]
	#[id(display = "Attack")]
	#[layout(group = "Voice", order = 3, widget = knob)]
	pub attack: f32,
	/// Which fitted table is playing.
	#[range(min = 0, max = 5)]
	#[choices("The Gentleman", "Peel Guitar", "Bass", "Celeste", "Harp", "Custom")]
	#[default(0)]
	#[id(display = "Instrument")]
	#[layout(group = "Voice", order = 5, widget = choice, hero = "modal")]
	pub instrument: i32,
	/// The overall level.
	#[range(min = 0.0, max = 1.0)]
	#[default(0.8)]
	#[id(display = "Gain")]
	#[layout(group = "Voice", order = 0, widget = knob, hero = "modal")]
	pub gain: f32,
	/// The velocity curve: negative is a soft keyboard, positive a hard one.
	#[range(min = -10.0, max = 10.0)]
	#[default(0.0)]
	#[id(display = "Velocity Curve")]
	#[layout(group = "Voice", order = 1, widget = knob)]
	pub velocity_curve: f32,
	/// How far apart the two channels are placed, 0 for mono.
	#[range(min = 0.0, max = 1.0)]
	#[default(0.4)]
	#[id(display = "Width")]
	#[layout(group = "Voice", order = 4, widget = knob)]
	pub width: f32,
	/// Whether a note the table does not hold is derived from its neighbours.
	#[default(false)]
	#[id(display = "Derive Notes")]
	#[layout(group = "Voice", order = 6, widget = toggle)]
	pub derive: bool,
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
			instrument: 0,
			cached: None,
			imported: None,
			gain: 0.8,
			velocity_curve: 0.0,
			release: 0.15,
			attack: 0.004,
			width: 0.4,
			derive: false,
		}
	}
}

impl FittedInstrument {
	/// A silent instrument, built for one sample rate.
	pub fn new(rate: usize) -> Self {
		fitted::install_from_config();
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
		let custom = if self.instrument == 5 {
			self.custom()
		} else {
			None
		};
		let table: &fitted::Instrument = match self.instrument {
			1 => fitted::peel(),
			2 => fitted::bass(),
			3 => fitted::celeste(),
			4 => fitted::harp(),
			5 => match &custom {
				Some(custom) => custom.as_ref(),
				None => return,
			},
			_ => fitted::instrument(),
		};
		self.playing.push(Playing {
			voice: fitted::Voice::start_from(
				table,
				note,
				velocity.clamp(0.0, 1.0),
				shaped,
				self.sample_rate,
				self.derive,
			),
			note,
			fade: 1.0,
			releasing: false,
			age: 0.0,
		});
	}

	/// The user's table, parsed once per change of bytes.
	fn custom(&mut self) -> Option<Arc<fitted::Instrument>> {
		let imported = self.imported.as_ref()?;
		let mut hash = 1469598103934665603u64;
		for byte in imported.bytes.iter() {
			hash = (hash ^ *byte as u64).wrapping_mul(1099511628211);
		}
		if let Some((cached, table)) = &self.cached {
			if *cached == hash {
				return Some(table.clone());
			}
		}
		let table = Arc::new(fitted::parse_table(&imported.bytes)?);
		self.cached = Some((hash, table.clone()));
		Some(table)
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

		let dt = 1.0 / self.sample_rate;
		let step = 1.0 / (self.release.max(0.02) * self.sample_rate);
		let mut left = 0.0f32;
		let mut right = 0.0f32;
		let width = self.width.clamp(0.0, 1.0);
		for voice in &mut self.playing {
			let [ear_left, ear_right] = voice.voice.next_frame();
			let attack = if self.attack > 1e-4 {
				(voice.age / self.attack).min(1.0)
			} else {
				1.0
			};
			voice.age += dt;
			if voice.releasing {
				voice.fade = (voice.fade - step).max(0.0);
			}
			let level = attack * voice.fade;
			// The two channels are fitted separately; Width blends them back
			// toward the mono sum instead of widening a mono voice.
			let mono = 0.5 * (ear_left + ear_right);
			left += (mono + (ear_left - mono) * width) * level;
			right += (mono + (ear_right - mono) * width) * level;
		}
		self.playing.retain(|voice| !voice.releasing || voice.fade > 0.0);
		[left, right]
	}
}