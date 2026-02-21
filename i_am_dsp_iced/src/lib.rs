//! An iced gui made for `i_am_dsp`
//! 
//! Note: currently, we only focus on 2 channeled audio processing, even though the library supports more.
//! 
//! TODO: Documentation

use std::time::Instant;

use i_am_dsp::{NoteEvent, ProcessContext, prelude::{Paramed, Parameters}};
use iced::{Element, Subscription};

pub mod styles;
pub mod tools;
pub mod plugins;
#[cfg(feature = "standalone")]
pub mod demo;

/// Re-exports of the `iced` crate.
pub use iced;

/// A function that returns a subscription that ticks the processor every 16ms.
pub fn timer<P: Processor>() -> Subscription<P::Message> {
	iced::time::every(std::time::Duration::from_millis(16)).map(P::Message::tick)
}

/// A trait for views that can be synced with the processor.
pub trait SyncedView: Send {
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
pub trait Processor: Parameters + Send + Sync + 'static {
	/// The message type used by the processor.
	type Message: Message;
	/// The view type used by the processor.
	type SyncedView: SyncedView<Message = Self::Message>;

	/// Processes the input samples and sends the output samples to the output buffer.
	fn process(&mut self, samples: &mut [f32; 2], other: &[[f32; 2]], process_context: &mut Box<dyn ProcessContext>);

	/// The delay of the processor in samples.
	fn delay(&self) -> usize;

	/// The function that is called when a message is received.
	/// 
	/// You need to use internel mutability to modify the processor's state, therefor the function takes a mutable reference to the processor.
	fn on_message(&self, message: Self::Message);

	/// The view for the processor.
	fn synced_view(&self) -> Self::SyncedView;
}

impl<P: Processor> Processor for Paramed<P> {
	type Message = P::Message;
	type SyncedView = P::SyncedView;

	fn process(&mut self, samples: &mut [f32; 2], other: &[[f32; 2]], process_context: &mut Box<dyn ProcessContext>) {
		self.value.process(samples, other, process_context)
	}

	fn delay(&self) -> usize {
		self.value.delay()
	}

	fn on_message(&self, message: Self::Message) {
		self.value.on_message(message)
	}

	fn synced_view(&self) -> Self::SyncedView {
		self.value.synced_view()
	}
}
