//! An iced gui made for `i_am_dsp`
//! 
//! Note: currently, we only focus on 2 channeled audio processing, even though the library supports more.
//! 
//! This crate adds a view to the processors of `i_am_dsp`: the
//! [Processor](crate::Processor) here is that crate's processor with a
//! [SyncedView] attached. A processor that wants no view at all implements
//! `i_am_dsp::Processor` directly, which is also the shape the plug-in wrapper in
//! `i_am_plugin` asks for.

#![warn(missing_docs)]

use iced::{Element, Subscription};

pub mod styles;
#[cfg(feature = "faceplate")]
pub mod faceplate;
pub mod tools;
#[cfg(feature = "standalone")]
pub mod demo;

/// Re-exports of the `iced` crate.
pub use iced;

/// The message trait of `i_am_dsp`, re-exported here so a view can be written
/// without naming that crate.
pub use i_am_dsp::{Message, Processor as CoreProcessor};

/// A function that returns a subscription that ticks the processor every 16ms.
pub fn timer<P: Processor>() -> Subscription<P::Message> {
	iced::time::every(std::time::Duration::from_millis(16)).map(P::Message::tick)
}

/// A message that wants to be told when the keyboard modifiers change.
///
/// It is a trait of its own because the modifiers are an iced concept and the
/// processor's [Message] is not: only an application that listens for them has to
/// implement it, and the empty body is all most of them need.
pub trait ModifierMessage: Message {
	/// Called when the keyboard modifiers change, for controls that read them:
	/// Shift to fine-tune a value, Ctrl/Cmd to reset it to the default.
	fn from_modifiers(_modifiers: iced::keyboard::Modifiers) -> Option<Self> {
		None
	}
}

impl ModifierMessage for () {}

/// A subscription that keeps [tools::input::modifiers] up to date.
///
/// iced sends keyboard events to the focused widget only, and a mouse event
/// carries no modifiers of its own, so this is the one place a control can learn
/// that Shift is being held. Add it next to [timer] in the application's
/// subscriptions:
///
/// ```ignore
/// Subscription::batch([timer::<MyProcessor>(), modifiers_subscription::<MyProcessor>()])
/// ```
pub fn modifiers_subscription<P: Processor>() -> Subscription<P::Message>
where
	P::Message: ModifierMessage,
{
	iced::event::listen_with(|event, _status, _window| match event {
		iced::Event::Keyboard(iced::keyboard::Event::ModifiersChanged(modifiers)) => {
			tools::input::set_modifiers(modifiers);
			P::Message::from_modifiers(modifiers)
		},
		_ => None,
	})
}

/// A trait for views that can be synced with the processor.
pub trait SyncedView: Send {
	/// The message type used by the view.
	type Message;

	/// Updates the view with the current state of the processor.
	fn update(&mut self, message: &Self::Message);

	/// The view function for the iced gui.
	fn view(&self) -> Element<'_, Self::Message>;
}

/// A trait for processors that can be used in the iced gui.
///
/// It is a processor from `i_am_dsp` plus the view that draws it. A processor that
/// has no view implements `i_am_dsp::Processor` alone.
pub trait Processor: i_am_dsp::Processor {
	/// The view type used by the processor.
	type SyncedView: SyncedView<Message = Self::Message>;

	/// The view for the processor.
	fn synced_view(&self) -> Self::SyncedView;
}

impl<P: Processor> Processor for i_am_dsp::prelude::Paramed<P> {
	type SyncedView = P::SyncedView;

	fn synced_view(&self) -> Self::SyncedView {
		self.value.synced_view()
	}
}
