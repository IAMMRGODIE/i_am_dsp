//! What runs the audio, with no interface attached.
//!
//! A processor is the part of an application or a plug-in that turns the audio it is
//! handed into the audio it hands back, and receives the messages an interface sends
//! it. It says nothing about how those messages are made or drawn: a GUI toolkit adds
//! a view on top of it, a plug-in wrapper adds a host around it, and an application
//! that only wants the audio can use it on its own.

use std::time::Instant;

use crate::prelude::Parameters;
use crate::{NoteEvent, ProcessContext};

/// A message a processor can receive.
///
/// This is the lowest common shape of what an interface sends the audio side: a note
/// event the host delivered, a periodic tick, and whatever else a toolkit turns into a
/// message of its own.
pub trait Message: Clone + Send + Sync + 'static {
	/// Converts a note event into a message.
	fn from_note_event(event: NoteEvent) -> Self;

	/// Converts a message into a note event, if it is one.
	fn note_event(&self) -> Option<NoteEvent>;

	/// The value to send when a view wants a periodic update.
	fn tick(instant: Instant) -> Self;
}

impl Message for () {
	fn from_note_event(_: NoteEvent) -> Self {}

	fn note_event(&self) -> Option<NoteEvent> {
		None
	}

	fn tick(_: Instant) -> Self {}
}

/// The audio and message side of an application or a plug-in.
///
/// A GUI toolkit is not part of this: `i_am_web_gui` extends it with a view, and
/// `i_am_plugin` wraps it in a CLAP/VST3 host. Implement this directly for a processor
/// that has no interface, or for one an interface of some other kind drives.
pub trait Processor: Parameters + Send + Sync + 'static {
	/// The message the processor accepts.
	type Message: Message;

	/// Processes the input samples and writes the output into `samples`.
	///
	/// `other` holds the additional input buffers (side-chains, for example), and
	/// `process_context` carries everything the host said about the current block.
	fn process(
		&mut self,
		samples: &mut [f32; 2],
		other: &[[f32; 2]],
		process_context: &mut Box<dyn ProcessContext>,
	);

	/// The delay of the processor in samples.
	fn delay(&self) -> usize;

	/// Called when the interface sends a message.
	///
	/// The processor is behind a shared reference here because the audio thread holds
	/// it mutably at the same time; state the two sides share has to be behind
	/// interior mutability.
	fn on_message(&self, message: Self::Message);
}

impl<P: Processor> Processor for crate::prelude::Paramed<P> {
	type Message = P::Message;

	fn process(
		&mut self,
		samples: &mut [f32; 2],
		other: &[[f32; 2]],
		process_context: &mut Box<dyn ProcessContext>,
	) {
		self.value.process(samples, other, process_context)
	}

	fn delay(&self) -> usize {
		self.value.delay()
	}

	fn on_message(&self, message: Self::Message) {
		self.value.on_message(message)
	}
}
