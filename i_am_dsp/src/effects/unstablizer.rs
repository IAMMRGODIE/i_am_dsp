//! A pitch shifter that keeps the note that is playing in tune.
//!
//! This is the DSP of the `i_am_unstablizer` plug-in: every bin of a short time
//! Fourier transform is moved by a frequency mapping that has one fixed point,
//! the note that is being played, so that note stays in tune while everything
//! else drifts away from it. The transform, the windowing and the phase
//! propagation are the library's [`PhaseVocoder`] driven by a
//! [`NoteAnchoredShift`] mapper; this effect adds the note tracking and the
//! mapping from a note number to a frequency.
//!
//! The plug-in stops outputting frames while no note is held. This effect keeps
//! the anchor at the last note it saw instead, so that it can be used without
//! note input at all (set [`Unstablizer::follow_notes`] to false and the
//! mapper's own `anchor` is used).

use i_am_dsp_derive::Parameters;

use crate::{
	effects::phase_vocoder::{NoteAnchoredShift, PhaseVocoder},
	Effect, NoteEvent, ProcessContext,
};

/// The window size the plug-in starts with, in samples.
const DEFAULT_WINDOW_SIZE: usize = 2048;
/// The window factor the plug-in starts with.
const DEFAULT_WINDOW_FACTOR: f32 = 0.6;
/// The highest note number that is tracked.
const MAX_NOTE: usize = 128;
/// The note number of A4, which the reference pitch is the frequency of.
const A4_NOTE: f32 = 69.0;

/// Returns the frequency of a note number, in Hz.
///
/// `reference_pitch` is the frequency of A4, and the note numbers follow the
/// MIDI convention, where 69 is A4 and every step is a semitone.
pub fn note_frequency(reference_pitch: f32, note: f32) -> f32 {
	reference_pitch * 2.0_f32.powf((note - A4_NOTE) / 12.0)
}

/// The notes that are held right now, in the order they were pressed.
///
/// This is the set the plug-in keeps, without the heap: releasing a note swaps
/// the last one into its place, so the last entry is always the most recently
/// pressed note that is still down.
struct HeldNotes {
	notes: [usize; MAX_NOTE],
	len: usize,
}

impl HeldNotes {
	const fn new() -> Self {
		Self {
			notes: [0; MAX_NOTE],
			len: 0,
		}
	}

	fn position(&self, note: usize) -> Option<usize> {
		self.notes[..self.len].iter().position(|held| *held == note)
	}

	/// Adds a note, ignoring one that is already held or out of range.
	fn insert(&mut self, note: usize) {
		if note >= MAX_NOTE || self.position(note).is_some() {
			return;
		}

		self.notes[self.len] = note;
		self.len += 1;
	}

	/// Removes a note, if it is held.
	fn remove(&mut self, note: usize) {
		let Some(position) = self.position(note) else {
			return;
		};

		self.len -= 1;
		self.notes[position] = self.notes[self.len];
	}

	fn clear(&mut self) {
		self.len = 0;
	}

	fn last(&self) -> Option<usize> {
		self.notes.get(self.len.checked_sub(1)?).copied()
	}
}

/// A phase vocoder that shifts every bin around the note that is playing.
#[derive(Parameters)]
pub struct Unstablizer<const CHANNELS: usize = 2> {
	/// The frequency of A4, used to turn a played note into a frequency, in Hz.
	#[range(min = 400.0, max = 480.0)]
	pub reference_pitch: f32,
	/// How far the anchor is transposed from the played note, in semitones.
	#[range(min = -24.0, max = 24.0)]
	pub transpose: f32,
	/// Whether the notes played through the effect set the anchor.
	///
	/// While this is on, the note that is held sets the anchor of the mapper, and
	/// the last note that was played keeps it after all notes are released. Turn
	/// it off to leave [`NoteAnchoredShift::anchor`] alone.
	pub follow_notes: bool,
	/// The phase vocoder that moves the bins.
	#[sub_param]
	pub vocoder: PhaseVocoder<NoteAnchoredShift, CHANNELS>,
	#[skip]
	held_notes: HeldNotes,
	#[skip]
	last_note: Option<usize>,
}

impl<const CHANNELS: usize> Unstablizer<CHANNELS> {
	/// Creates a new unstablizer for the given sample rate.
	///
	/// # Panics
	///
	/// Panics if `CHANNELS` is 0.
	pub fn new(sample_rate: usize) -> Self {
		assert!(CHANNELS > 0, "CHANNELS must be greater than 0");

		let mut vocoder = PhaseVocoder::new(
			NoteAnchoredShift::default(),
			DEFAULT_WINDOW_SIZE,
			sample_rate,
		);
		vocoder.window_factor = DEFAULT_WINDOW_FACTOR;

		Self {
			reference_pitch: 440.0,
			transpose: 0.0,
			follow_notes: true,
			vocoder,
			held_notes: HeldNotes::new(),
			last_note: None,
		}
	}

	/// Updates the held notes from the events of the frame being processed.
	///
	/// The events are read, not consumed: an effect can share them with a
	/// generator or another effect that needs them as well.
	fn track_events(&mut self, process_context: &mut Box<dyn ProcessContext>) {
		for event in process_context.events() {
			match event {
				NoteEvent::NoteOn { note, .. } => self.held_notes.insert(*note),
				NoteEvent::NoteOff { note, .. } | NoteEvent::Stop { note, .. } => self.held_notes.remove(*note),
				NoteEvent::ImmediateStop => self.held_notes.clear(),
				_ => {},
			}
		}

		if let Some(note) = self.held_notes.last() {
			self.last_note = Some(note);
		}
	}
}

impl<const CHANNELS: usize> Effect<CHANNELS> for Unstablizer<CHANNELS> {
	fn delay(&self) -> usize {
		self.vocoder.window_size()
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"Unstablizer"
	}

	fn process(
		&mut self,
		samples: &mut [f32; CHANNELS],
		other: &[&[f32; CHANNELS]],
		process_context: &mut Box<dyn ProcessContext>,
	) {
		if self.follow_notes {
			self.track_events(process_context);

			if let Some(note) = self.held_notes.last().or(self.last_note) {
				self.vocoder.mapper.anchor = note_frequency(self.reference_pitch, note as f32 + self.transpose);
			}
		}

		self.vocoder.process(samples, other, process_context);
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		use egui::Slider;

		self.vocoder.demo_ui(ui, format!("{}_vocoder", id_prefix));
		ui.add(Slider::new(&mut self.reference_pitch, 400.0..=480.0).text("Reference Pitch (Hz)"));
		ui.add(Slider::new(&mut self.transpose, -24.0..=24.0).text("Transpose (semitones)"));
		ui.checkbox(&mut self.follow_notes, "Follow Notes");
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{Effect, ProcessContext, ProcessInfos};

	struct TestContext {
		events: Vec<NoteEvent>,
	}


	impl ProcessContext for TestContext {
		fn infos(&self) -> &ProcessInfos {
			&crate::DEFAULT_PROCESS_INFO
		}

		fn next_event(&mut self) -> Option<NoteEvent> {
			if self.events.is_empty() { None } else { Some(self.events.remove(0)) }
		}

		fn send_event(&mut self, event: NoteEvent) {
			self.events.push(event);
		}

		fn events(&self) -> &[NoteEvent] {
			&self.events
		}

		fn clear_events(&mut self) {
			self.events.clear();
		}
	}

	fn note_on(note: usize) -> NoteEvent {
		NoteEvent::NoteOn { channel: 0, note, velocity: 1.0 }
	}

	fn note_off(note: usize) -> NoteEvent {
		NoteEvent::NoteOff { channel: 0, note, velocity: 1.0 }
	}

	/// Runs one frame through the effect with the given events.
	fn process_with_events(unstablizer: &mut Unstablizer<1>, events: Vec<NoteEvent>) {
		let context = TestContext { events };
		let mut context: Box<dyn ProcessContext> = Box::new(context);
		let mut frame = [0.0];
		unstablizer.process(&mut frame, &[], &mut context);
	}

	#[test]
	fn a_played_note_sets_the_anchor() {
		let mut unstablizer = Unstablizer::<1>::new(48000);

		process_with_events(&mut unstablizer, vec![note_on(69)]);
		assert!((unstablizer.vocoder.mapper.anchor - 440.0).abs() < 0.01);

		// An octave down is half the frequency.
		process_with_events(&mut unstablizer, vec![note_off(69), note_on(57)]);
		assert!((unstablizer.vocoder.mapper.anchor - 220.0).abs() < 0.01);

		// The anchor follows the note that is still held.
		process_with_events(&mut unstablizer, vec![note_on(60), note_off(60)]);
		assert!((unstablizer.vocoder.mapper.anchor - 220.0).abs() < 0.01);

		// A different reference pitch and a transposition move it as well.
		unstablizer.reference_pitch = 432.0;
		unstablizer.transpose = 12.0;
		process_with_events(&mut unstablizer, vec![note_off(57), note_on(60)]);
		let expected = note_frequency(432.0, 60.0 + 12.0);
		assert!((unstablizer.vocoder.mapper.anchor - expected).abs() < 0.01);
	}

	#[test]
	fn the_anchor_is_left_alone_when_the_notes_are_not_followed() {
		let mut unstablizer = Unstablizer::<1>::new(48000);
		unstablizer.follow_notes = false;
		unstablizer.vocoder.mapper.anchor = 123.0;

		process_with_events(&mut unstablizer, vec![note_on(69)]);
		assert_eq!(unstablizer.vocoder.mapper.anchor, 123.0);
	}

	#[test]
	fn the_parameters_are_reachable_by_their_identifiers() {
		use crate::prelude::{Parameters, SetValue};

		let mut unstablizer = Unstablizer::<2>::new(48000);
		assert!(unstablizer.set_parameter("vocoder.mapper.shift", SetValue::Float(2.0)));
		assert_eq!(unstablizer.vocoder.mapper.shift, 2.0);
		assert!(unstablizer.set_parameter("vocoder.window_factor", SetValue::Float(0.5)));
		assert_eq!(unstablizer.vocoder.window_factor, 0.5);
		assert!(unstablizer.set_parameter("transpose", SetValue::Float(12.0)));
		assert_eq!(unstablizer.transpose, 12.0);
		assert!(unstablizer.set_parameter("follow_notes", SetValue::Bool(false)));
		assert!(!unstablizer.follow_notes);
	}

	/// The magnitude of a frequency in the given samples.
	fn magnitude_at(samples: &[f32], sample_rate: f32, frequency: f32) -> f32 {
		let omega = 2.0 * std::f32::consts::PI * frequency / sample_rate;
		let coefficient = 2.0 * omega.cos();
		let (mut last, mut second_last) = (0.0f32, 0.0f32);

		for sample in samples {
			let current = *sample + coefficient * last - second_last;
			second_last = last;
			last = current;
		}

		(last * last + second_last * second_last - coefficient * last * second_last)
			.max(0.0)
			.sqrt()
	}

	#[test]
	fn the_shift_moves_a_tone_away_from_the_anchor() {
		let sample_rate = 48000.0;
		let mut unstablizer = Unstablizer::<1>::new(48000);
		// No notes: the mapper anchor is the one that is set, and it is what the
		// tone is shifted around.
		unstablizer.follow_notes = false;
		unstablizer.vocoder.mapper.anchor = 468.75;
		unstablizer.vocoder.mapper.shift = 2.0;

		// The frequencies sit on the analysis bins of a 2048 sample window
		// (23.4375 Hz apart), so the window leakage does not pollute the
		// measurement: 750 Hz is bin 32, the anchor is bin 20, and the mapping
		// puts the tone at 750 * 2 + 468.75 * (1 - 2) = 1031.25 Hz, bin 44.
		let window_size = unstablizer.vocoder.window_size();
		let mut context: Box<dyn ProcessContext> = Box::new(());
		let mut output = Vec::new();
		for i in 0..window_size * 8 {
			let sample = (2.0 * std::f32::consts::PI * 750.0 * i as f32 / sample_rate).sin();
			let mut frame = [sample];
			unstablizer.process(&mut frame, &[], &mut context);
			output.push(frame[0]);
		}

		let moved = magnitude_at(&output[window_size..], sample_rate, 1031.25);
		let left_behind = magnitude_at(&output[window_size..], sample_rate, 750.0);
		assert!(
			moved > left_behind * 20.0,
			"the tone was not moved: 1031.25 Hz has {moved}, 750 Hz has {left_behind}",
		);
	}
}
