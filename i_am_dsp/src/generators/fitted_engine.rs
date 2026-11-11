//! A fitted instrument: a mode table played back, not a string model.
//!
//! The adapter for `i_am_string::fitted`: it turns [NoteEvent]s into voices,
//! applies the velocity curve and the release, and exposes the player controls.

use i_am_dsp_derive::Parameters;

use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

use i_am_string::fitted;

use crate::generators::adsr::bend;
use crate::{Generator, NoteEvent};

const MAX_VOICES: usize = 32;

/// The table is numbered the way the library numbers notes: A4 is 57.
fn to_table(note: usize) -> u8 {
	note.clamp(0, 127) as u8
}

/// A cheap identity for a table's bytes, so a note does not parse it twice.
fn hash_of(bytes: &[u8]) -> u64 {
	let mut hash = 1469598103934665603u64;
	for byte in bytes {
		hash = (hash ^ *byte as u64).wrapping_mul(1099511628211);
	}
	hash
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

/// What the page and the audio thread share: the table the page fitted.
///
/// A serialized parameter cannot travel through the parameter map, and the fit
/// runs on the page's thread, so the table goes into a slot the instrument
/// reads. A save copies it into the `imported` parameter, which is the copy the
/// host stores.
#[derive(Clone, Default)]
pub struct TableSlot(Arc<Mutex<SlotInner>>);

/// The slot's contents, with a number that changes on every write.
#[derive(Default)]
struct SlotInner {
	table: Option<Arc<ImportedTable>>,
	version: u64,
}

impl TableSlot {
	/// Stores a table; the next note picks it up.
	pub fn set(&self, table: ImportedTable) {
		let mut inner = self.0.lock().unwrap_or_else(|error| error.into_inner());
		inner.table = Some(Arc::new(table));
		inner.version = inner.version.wrapping_add(1);
	}

	/// The table the page last fitted, if any.
	pub fn current(&self) -> Option<Arc<ImportedTable>> {
		self.0
			.lock()
			.unwrap_or_else(|error| error.into_inner())
			.table
			.clone()
	}

	/// A number that changes whenever the table does, so the plug-in knows when
	/// the parameter it saves is out of date.
	pub fn version(&self) -> u64 {
		self.0
			.lock()
			.unwrap_or_else(|error| error.into_inner())
			.version
	}

	/// Forgets the table, so the imported parameter is what plays again.
	pub fn clear(&self) {
		let mut inner = self.0.lock().unwrap_or_else(|error| error.into_inner());
		inner.table = None;
		inner.version = inner.version.wrapping_add(1);
	}
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
	#[skip]
	slot: TableSlot,
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
	/// How far the whole instrument is shifted, in semitones. A sample fitted
	/// an octave off is fixed here instead of being fitted again.
	#[range(min = -24, max = 24)]
	#[default(0)]
	#[id(display = "Transpose")]
	#[layout(group = "Voice", order = 7, widget = knob, bipolar, step = 1.0)]
	pub transpose: i32,
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
			transpose: 0,
			slot: TableSlot::default(),
		}
	}
}

impl FittedInstrument {
	/// A silent instrument, built for one sample rate.
	pub fn new(rate: usize) -> Self {
		Self::with_slot(rate, TableSlot::default())
	}

	/// A silent instrument over a slot the page can write.
	pub fn with_slot(rate: usize, slot: TableSlot) -> Self {
		fitted::install_from_config();
		Self {
			sample_rate: if rate > 0 { rate as f32 } else { 48_000.0 },
			slot,
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
				self.played(note),
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

	/// The note the table is asked for: the key the player pressed, shifted by
	/// the transpose.
	fn played(&self, note: u8) -> u8 {
		(note as i32 + self.transpose).clamp(0, 127) as u8
	}

	/// The user's table, parsed once per change of bytes.
	///
	/// The slot wins over the parameter: a table the page just fitted is in the
	/// slot, and the parameter only carries the one the host saved.
	fn custom(&mut self) -> Option<Arc<fitted::Instrument>> {
		let hash = match self.slot.current() {
			Some(table) => hash_of(&table.bytes),
			None => hash_of(&self.imported.as_ref()?.bytes),
		};
		if let Some((cached, table)) = &self.cached
			&& *cached == hash
		{
			return Some(table.clone());
		}
		let bytes = match self.slot.current() {
			Some(table) => table.bytes.clone(),
			None => self.imported.as_ref()?.bytes.clone(),
		};
		let table = Arc::new(fitted::parse_table(&bytes)?);
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

/// One sample, fitted into a table the instrument can play.
#[derive(Debug)]
pub struct FittedSample {
	/// The note the table is keyed to.
	pub note: u8,
	/// The fundamental the fit found, in hertz.
	pub frequency: f32,
	/// How many partials the note kept.
	pub modes: usize,
	/// The table, in the format [ImportedTable] carries.
	pub bytes: Vec<u8>,
}

/// Fit one recorded note into a table the instrument can play.
///
/// `wav` is the file as the page sent it. The two channels are fitted
/// separately, so a stereo recording keeps its own image; a mono one is fitted
/// once and copied to both ears, and so is a stereo file whose channels are the
/// same, because fitting a mono recording twice would invent a difference
/// between the ears.
///
/// A table fitted this way holds one note. Every other key is derived from it
/// when the instrument is played, which is why a single sample is enough to
/// play a scale.
pub fn fit_sample(wav: &[u8], note: Option<u8>) -> Result<FittedSample, String> {
	let pcm = crate::tools::pcm_data::load_from_bytes::<2>(wav.to_vec())
		.map_err(|error| error.to_string())?;
	let rate = pcm.sample_rate as f32;
	if rate <= 0.0 {
		return Err("the file has no sample rate".to_string());
	}
	let left = i_am_string::fit::fit(&pcm.pcm_data[0], rate, note)
		.ok_or_else(|| "the sample holds no note to fit".to_string())?;
	let right = match pcm.pcm_data.get(1) {
		Some(channel)
			if channel.len() == pcm.pcm_data[0].len() && !same(channel, &pcm.pcm_data[0]) =>
		{
			i_am_string::fit::fit(channel, rate, Some(left.note))
				.map(|fitted| fitted.modes)
				.unwrap_or_else(|| left.modes.clone())
		}
		_ => left.modes.clone(),
	};
	let note = left.note;
	let frequency = left.frequency;
	let modes = left.modes.len();
	let table = fitted::Note {
		note,
		left: left.modes,
		right,
	};
	// One velocity layer: a single sample says nothing about how the
	// instrument behaves at other dynamics, so every layer plays the levels
	// that were measured.
	let bytes = fitted::write_table(&[table], &[0.0], 1, 1);
	Ok(FittedSample {
		note,
		frequency,
		modes,
		bytes,
	})
}

/// Whether two channels carry the same recording.
fn same(left: &[f32], right: &[f32]) -> bool {
	left.iter().zip(right).all(|(a, b)| (a - b).abs() < 1e-6)
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
		self.playing
			.retain(|voice| !voice.releasing || voice.fade > 0.0);
		[left, right]
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// A 16 bit mono WAV holding one note, as a page would post it.
	fn wav(samples: &[f32], rate: u32) -> Vec<u8> {
		let data = (samples.len() * 2) as u32;
		let mut out = Vec::new();
		out.extend_from_slice(b"RIFF");
		out.extend_from_slice(&(36 + data).to_le_bytes());
		out.extend_from_slice(b"WAVEfmt ");
		out.extend_from_slice(&16u32.to_le_bytes());
		out.extend_from_slice(&1u16.to_le_bytes());
		out.extend_from_slice(&1u16.to_le_bytes());
		out.extend_from_slice(&rate.to_le_bytes());
		out.extend_from_slice(&(rate * 2).to_le_bytes());
		out.extend_from_slice(&2u16.to_le_bytes());
		out.extend_from_slice(&16u16.to_le_bytes());
		out.extend_from_slice(b"data");
		out.extend_from_slice(&data.to_le_bytes());
		for sample in samples {
			let value = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
			out.extend_from_slice(&value.to_le_bytes());
		}
		out
	}

	/// One decaying note with a fundamental and a fifth above it.
	fn note(rate: u32, seconds: f32) -> Vec<f32> {
		let count = (seconds * rate as f32) as usize;
		(0..count)
			.map(|index| {
				let time = index as f32 / rate as f32;
				let decay = (-3.0 * time).exp();
				decay
					* (0.5 * (std::f32::consts::TAU * 220.0 * time).sin()
						+ 0.25 * (std::f32::consts::TAU * 330.0 * time).sin())
			})
			.collect()
	}

	#[test]
	fn a_sample_becomes_a_playable_table() {
		let rate = 44_100;
		let fitted = fit_sample(&wav(&note(rate, 1.2), rate), Some(57)).expect("a fitted table");

		assert_eq!(fitted.note, 57);
		assert!(fitted.modes >= 2, "only {} partials", fitted.modes);

		let table = fitted::parse_table(&fitted.bytes).expect("the table parses");
		assert_eq!(table.notes().len(), 1);
		assert_eq!(table.notes()[0].note, 57);

		let mut voice = fitted::Voice::start_from(&table, 57, 1.0, 0.8, rate as f32, false);
		let mut peak = 0.0f32;
		for _ in 0..4_800 {
			peak = peak.max(voice.next_sample().abs());
		}
		assert!(peak > 1e-4, "the fitted table is silent");
	}

	#[test]
	fn a_silent_file_is_refused() {
		let rate = 44_100;
		let error = fit_sample(&wav(&vec![0.0; rate as usize], rate), None)
			.expect_err("a silent file has no table");
		assert!(error.contains("no note"), "{error}");
	}

	#[test]
	fn the_transpose_shifts_which_note_is_played() {
		let mut instrument = FittedInstrument::new(48_000);
		assert_eq!(instrument.played(60), 60);

		instrument.transpose = 12;
		assert_eq!(instrument.played(60), 72);

		instrument.transpose = -24;
		assert_eq!(instrument.played(10), 0, "a transposed key is clamped");
		instrument.transpose = 24;
		assert_eq!(instrument.played(120), 127);
	}

	#[test]
	fn something_that_is_not_a_wav_is_refused() {
		assert!(fit_sample(b"not a wave file", None).is_err());
	}
}
