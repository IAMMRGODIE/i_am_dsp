//! This is a simple dsp library for real-time audio processing.
//! 
//! TODO: 
//! 1. Visual Effects
//! 2. SIMD support
//! 3. BPM sync
//! 4. Bypass Support
//! 5. parameter reset hotkey.

#![warn(missing_docs)]

use crate::prelude::{Parameter, Parameters};

extern crate self as i_am_dsp;

#[cfg(feature = "real_time_demo")]
pub mod real_time_demo;

pub mod parameters;

pub mod generators;
pub mod effects;
pub mod tools;
pub mod prelude;

#[doc(hidden)]
pub use phf;

// /// re-export lazy_static for derive macro
// /// 
// /// See https://crates.io/crates/lazy_static for more information.
// pub use lazy_static::lazy_static;

/// Main trait for effects.
pub trait Effect<const CHANNELS: usize = 2>: Send + Sync + Parameters {
	/// Process the given samples.
	/// 
	/// `samples` is the input and output buffer, user should modify it in place.
	/// `other` contains other input buffers, you may use it to imply effect like sidechain.
	/// `process_context` contains information about the process, like sample rate, tempo, etc.
	fn process(
		&mut self, 
		samples: &mut [f32; CHANNELS], 
		other: &[&[f32; CHANNELS]],
		process_context: &mut Box<dyn ProcessContext>,
	);

	/// Returns the delay of the effect in samples.
	fn delay(&self) -> usize;
	

	/// Returns the name of the effect.
	/// 
	/// Used to identify the effect in the real-time demo.
	#[cfg(feature = "real_time_demo")]
	#[cfg_attr(docsrs, doc(cfg(feature = "real_time_demo")))]
	fn name(&self) -> &str {
		"Unnamed"
	}

	#[cfg(feature = "real_time_demo")]
	#[cfg_attr(docsrs, doc(cfg(feature = "real_time_demo")))]
	/// Real time demo UI for the effect
	/// 
	/// By default, it just shows a label with the name of the effect.
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		let _ = id_prefix;
		ui.label("nothing");
	}
}

/// Main trait for generators.
pub trait Generator<const CHANNELS: usize = 2>: Send + Sync + Parameters {
	/// Generate a new sample.
	/// 
	/// `process_context` contains information about the process, like sample rate, tempo, etc.
	fn generate(&mut self, process_context: &mut Box<dyn ProcessContext>) -> [f32; CHANNELS];

	#[cfg(feature = "real_time_demo")]
	#[cfg_attr(docsrs, doc(cfg(feature = "real_time_demo")))]
	/// Returns the name of the generator.
	/// 
	/// Used to identify the generator in the real-time demo.
	fn name(&self) -> &str {
		"Unnamed"
	}

	#[cfg(feature = "real_time_demo")]
	#[cfg_attr(docsrs, doc(cfg(feature = "real_time_demo")))]
	/// Real time demo UI for the generator.
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		let _ = id_prefix;
		ui.label("nothing");
	}
}

impl<const CHANNELS: usize, T: Effect<CHANNELS>> Effect<CHANNELS> for Vec<T> {
	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"Anonymous Effect Chain"
	}

	fn delay(&self) -> usize {
		self.iter().map(|effect| effect.delay()).sum()
	}

	fn process(&mut self, samples: &mut [f32; CHANNELS], other: &[&[f32; CHANNELS]], process_context: &mut Box<dyn ProcessContext>) {
		for effect in self {
			effect.process(samples, other, process_context);
		}
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		for (i, effect) in self.iter_mut().enumerate() {
			let effect_id = format!("{}_effect_{}", id_prefix, i);
			ui.collapsing(effect_id.clone(), |ui| {
				effect.demo_ui(ui, effect_id);
			});
		}
	}
}

impl<const CHANNELS: usize> Parameters for Vec<Box<dyn Effect<CHANNELS>>> {
	fn get_parameters(&self) -> Vec<Parameter> {
		let mut result = Vec::new();
		for (i, p) in self.iter().enumerate() {
			for mut param in p.get_parameters() {
				param.identifier = format!("{i}.{}", param.identifier);
				result.push(param);
			}
		}
		result
	}

	fn set_parameter(&mut self, identifier: &str, value: prelude::SetValue) -> bool {
		let mut parts = identifier.split(".").collect::<Vec<&str>>();
		let index = parts.remove(0).parse::<usize>().expect("Invalid index");
		let rest_identifier = parts.join(".");
		if index >= self.len() {
			return false;
		}
		self[index].set_parameter(&rest_identifier, value)
	}
}

impl<const CHANNELS: usize> Effect<CHANNELS> for Vec<Box<dyn Effect<CHANNELS>>> {
	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"Anonymous Effect Chain"
	}

	fn delay(&self) -> usize {
		self.iter().map(|effect| effect.delay()).sum()
	}

	fn process(&mut self, samples: &mut [f32; CHANNELS], other: &[&[f32; CHANNELS]], process_context: &mut Box<dyn ProcessContext>) {
		for effect in self {
			effect.process(samples, other, process_context);
		}
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		for (i, effect) in self.iter_mut().enumerate() {
			let effect_id = format!("{}_effect_{}", id_prefix, i);
			ui.collapsing(effect_id.clone(), |ui| {
				effect.demo_ui(ui, effect_id);
			});
		}
	}
}

impl<const CHANNELS: usize> Generator<CHANNELS> for () {
	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"Null Generator"
	}

	fn generate(&mut self, _: &mut Box<dyn ProcessContext>) -> [f32; CHANNELS] {
		[0.0; CHANNELS]
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		let _ = (ui, id_prefix);
	}
}

impl<const CHANNELS: usize> Effect<CHANNELS> for () {
	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"Null Effect"
	}

	fn delay(&self) -> usize {
		0
	}

	fn process(
		&mut self, 
		samples: &mut [f32; CHANNELS], 
		other: &[&[f32; CHANNELS]], 
		_: &mut Box<dyn ProcessContext>
	) {
		let _ = (samples, other);
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		let _ = (ui, id_prefix);
	}
}

/// Main trait for process context.
pub trait ProcessContext: Send + Sync {
	/// Returns the process infos.
	fn infos(&self) -> &ProcessInfos;
	/// Returns the next note event in the process.
	/// 
	/// this is useful for midi processing.
	fn next_event(&mut self) -> Option<NoteEvent>;
	/// Send a note event to the process.
	/// 
	/// This may be used for midi effects.
	fn send_event(&mut self, event: NoteEvent);
	/// Returns the note events sent to the process, but not consumed them.
	fn events(&self) -> &[NoteEvent];

	/// Whether the process should stop immediately or not.
	fn should_stop(&self) -> bool {
		self.events().contains(&NoteEvent::ImmediateStop)
	}
}

lazy_static::lazy_static! {
	static ref DEFAULT_PROCESS_INFO: ProcessInfos = ProcessInfos::default();
}

impl ProcessContext for () {
	fn infos(&self) -> &ProcessInfos {
		&DEFAULT_PROCESS_INFO
	}

	fn next_event(&mut self) -> Option<NoteEvent> {
		None
	}

	fn send_event(&mut self, _: NoteEvent) {}

	fn events(&self) -> &[NoteEvent] {
		&[]
	}
}

impl<T: ProcessContext> ProcessContext for Box<T> {
	fn infos(&self) -> &ProcessInfos {
		self.as_ref().infos()
	}

	fn next_event(&mut self) -> Option<NoteEvent> {
		self.as_mut().next_event()
	}

	fn send_event(&mut self, event: NoteEvent) {
		self.as_mut().send_event(event)
	}

	fn events(&self) -> &[NoteEvent] {
		self.as_ref().events()
	}
}

#[derive(Clone, Debug, PartialEq, Default)]
#[non_exhaustive]
/// Information about the process in current frame.
pub struct ProcessInfos {
	/// Whether the process info is trustable or not.
	/// 
	/// If it's not trustable, the user shouldn't rely the info on it,
	pub trustable: bool,
	/// The sample rate of the process.
	pub sample_rate: usize,
	/// Whether the host is playing or not.
	pub playing: bool,
	/// The tempo of the process.
	pub tempo: Option<f32>,
	/// The time signature of the process.
	/// 
	/// The trunc part is current bar number. 
	/// the fract part is how long have passed in the current bar(in percentage).
	pub current_bar_number: Option<f32>,
	/// The time signature of the process.
	/// 
	/// The first element is the numerator, the second element is the denominator.
	pub time_signature: Option<(usize, usize)>,
	/// The current time of the process, in seconds.
	pub current_time: f32,
}

impl ProcessInfos {
	/// Create a new process infos with default values.
	pub fn new() -> Self {
		Self::default()
	}
}

// #[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
/// A note event that can be used to represent a midi event.
/// 
/// May change in the future.
pub enum NoteEvent {
	/// Note on event.
	NoteOn {
		/// the time of the event, in samples.
		time: usize,
		/// the channel of the event.
		/// 
		/// in 0..16 for most cases.
		channel: u8,
		/// The pitch of the note.
		/// 
		/// in 0..128 for most cases, 60 = middle C.
		note: u8,
		/// The velocity of the note.
		/// 
		/// in 0.0..=1.0 for most cases.
		velocity: f32,	
	},
	/// Note off event.
	NoteOff {
		/// the time of the event, in samples.
		time: usize,
		/// the channel of the event.
		/// 
		/// in 0..16 for most cases.
		channel: u8,
		/// The pitch of the note.
		/// 
		/// in 0..128 for most cases, 60 = middle C.
		note: u8,
		/// The velocity of the note.
		/// 
		/// in 0.0..=1.0 for most cases.
		velocity: f32,	
	},
	/// Stop note event.
	/// 
	/// If you receive this event, you should stop the note attached to it immediately.
	Stop {
		/// the time of the event, in samples.
		time: usize,
		/// the channel of the event.
		channel: u8,
		/// The pitch of the note.
		note: u8,
	},
	/// Midi CC event.
	MidiCC {
		/// the time of the event, in samples.
		time: usize,
		/// the channel of the event.
		/// 
		/// in 0..16 for most cases.
		channel: u8,
		/// The controller number.
		cc: u8,
		/// The value of the controller.
		value: f32,
	},
	/// Stop all sound event.
	/// 
	/// If you receive this event, you should stop all sound immediately.
	ImmediateStop,
}

const NOTE_NAMES: [&str; 12] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

/// Format a note number as a string.
pub fn format_note(note: u8) -> String {
	let note_index = (note % 12) as usize;
	let octave = (note / 12) as i32;
	format!("{}{}", NOTE_NAMES[note_index], octave)
}