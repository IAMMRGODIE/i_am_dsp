/// A note level event.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Event {
	/// Start a note.
	NoteOn {
		/// The MIDI note number, with 69 = A4.
		note: u8,
		/// The velocity in 0..1.
		velocity: f32,
	},
	/// Release a note.
	NoteOff {
		/// The MIDI note number.
		note: u8,
	},
	/// Release everything.
	AllNotesOff,
	/// Move the sustain pedal, from 0 (up) to 1 (fully down).
	Sustain {
		/// The pedal position.
		position: f32,
	},
}
