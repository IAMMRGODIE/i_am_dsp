//! The context the demo hands its processors: sample rate, tempo and a note queue.

use super::*;

/// A simple context that storage all necessary information for processing.
pub struct SimpleContext {
	/// The process infos
	pub info: ProcessInfos,
	/// The midi events
	pub midi_events: Vec<NoteEvent>,
}

impl ProcessContext for SimpleContext {
	fn infos(&self) -> &ProcessInfos {
		&self.info
	}

	fn next_event(&mut self) -> Option<NoteEvent> {
		self.midi_events.pop()
	}

	fn send_event(&mut self, event: NoteEvent) {
		self.midi_events.push(event);
	}

	fn events(&self) -> &[NoteEvent] {
		&self.midi_events
	}

	fn clear_events(&mut self) {
		self.midi_events.clear();
	}
}
