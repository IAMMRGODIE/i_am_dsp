//! An iced gui made for `i_am_dsp`
//! 
//! Note: currently, we only focus on 2 channeled audio processing, even though the library supports more.

use std::time::Instant;

use i_am_dsp::{NoteEvent, ProcessContext};
use iced::Element;

pub mod styles;
pub mod tools;
pub mod plugins;
#[cfg(feature = "standalone")]
pub mod demo;

/// A trait for views that can be synced with the processor.
pub trait SyncedView {
	type Message;

	/// Updates the view with the current state of the processor.
	fn update(&mut self, message: &Self::Message);

	/// The view function for the iced gui.
	fn view(&self) -> Element<'_, Self::Message>;
}

/// A trait for messages that can be converted from `NoteEvent`s.
pub trait Message: Clone + Send + Sync + 'static {
	/// Converts a `NoteEvent` to a message.
	fn from_note_event(event: NoteEvent) -> Self;
	/// Converts a message to a `NoteEvent`, if possible.
	fn note_event(&self) -> Option<NoteEvent>;
	/// The tick function for the processor.
	fn tick(instant: Instant) -> Self;
}

/// A trait for processors that can be used in the iced gui.
pub trait Processsor: Send + Sync + 'static {
	/// The message type used by the processor.
	type Message: Message;
	/// The view type used by the processor.
	type SyncedView: SyncedView<Message = Self::Message>;

	/// Processes the input samples and sends the output samples to the output buffer.
	fn process(&mut self, samples: &mut [f32; 2], other: &[&[f32; 2]], process_context: &mut Box<dyn ProcessContext>);

	/// The delay of the processor in samples.
	fn delay(&self) -> usize;

	/// The number of input channels of the processor.
	fn on_message(&mut self, message: Self::Message);

	/// The view for the processor.
	fn synced_view(&mut self) -> Self::SyncedView;
}