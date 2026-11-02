//! What runs the audio, with no interface attached.
//!
//! A processor is the part of an application or a plug-in that turns the audio it is
//! handed into the audio it hands back, and receives the messages an interface sends
//! it. It says nothing about how those messages are made or drawn: a GUI toolkit adds
//! a view on top of it, a plug-in wrapper adds a host around it, and an application
//! that only wants the audio can use it on its own.

use std::time::Instant;

use crate::parameters::{Parameter, SetValue};
use crate::prelude::Parameters;
use crate::{Effect, Generator, NoteEvent, ProcessContext};

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

/// A generator, played where an effect is expected.
///
/// The two halves of this library disagree about the direction of the audio: an
/// [Effect] reads the buffer it is handed and writes the result back in place, a
/// [Generator] returns the sample it made. A host stage -- a plug-in wrapper, an
/// effect slot that plays an instrument -- only knows the first shape, and this is
/// the whole difference between them.
///
/// Nothing else is adapted. The parameter list, the note events, the tempo and the
/// sample rate reach the generator exactly as they reach an effect, because the
/// adapter forwards all of them; only the buffer changes hands.
///
/// A generator has no latency to report: it makes the sound of the sample it is
/// asked for, rather than answering with a delayed one.
pub struct GeneratorEffect<G, const CHANNELS: usize = 2>(pub G);

impl<G, const CHANNELS: usize> GeneratorEffect<G, CHANNELS> {
	/// Play a generator wherever an effect goes.
	pub const fn new(generator: G) -> Self {
		Self(generator)
	}

	/// The generator inside.
	pub fn generator(&self) -> &G {
		&self.0
	}

	/// The generator inside, mutably.
	pub fn generator_mut(&mut self) -> &mut G {
		&mut self.0
	}

	/// Take the generator back out.
	pub fn into_inner(self) -> G {
		self.0
	}
}

impl<G: Parameters, const CHANNELS: usize> Parameters for GeneratorEffect<G, CHANNELS> {
	fn get_parameters(&self) -> Vec<Parameter> {
		self.0.get_parameters()
	}

	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		self.0.set_parameter(identifier, value)
	}

	fn stable_id_overrides(&self) -> Vec<(String, u32)> {
		self.0.stable_id_overrides()
	}
}

impl<G: Generator<CHANNELS>, const CHANNELS: usize> Effect<CHANNELS>
	for GeneratorEffect<G, CHANNELS>
{
	fn process(
		&mut self,
		samples: &mut [f32; CHANNELS],
		_other: &[&[f32; CHANNELS]],
		process_context: &mut Box<dyn ProcessContext>,
	) {
		*samples = self.0.generate(process_context);
	}

	fn delay(&self) -> usize {
		0
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::prelude::Value;

	/// A generator with one control, so the delegation can be seen.
	struct Level(f32);

	impl Parameters for Level {
		fn get_parameters(&self) -> Vec<Parameter> {
			vec![Parameter::new(
				"level",
				Value::Float {
					value: self.0,
					range: 0.0..=1.0,
					logarithmic: false,
				},
			)]
		}

		fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
			if identifier == "level"
				&& let SetValue::Float(level) = value
			{
				self.0 = level;
				return true;
			}

			false
		}
	}

	impl Generator<2> for Level {
		fn generate(&mut self, _: &mut Box<dyn ProcessContext>) -> [f32; 2] {
			[self.0, -self.0]
		}
	}

	#[test]
	fn a_generator_writes_over_the_buffer_it_is_handed() {
		let mut effect = GeneratorEffect::<_, 2>::new(Level(0.5));
		let mut samples = [7.0, 7.0];
		let mut context: Box<dyn ProcessContext> = Box::new(());

		effect.process(&mut samples, &[], &mut context);

		assert_eq!(samples, [0.5, -0.5]);
	}

	#[test]
	fn a_generator_has_no_latency() {
		let effect = GeneratorEffect::<_, 2>::new(Level(0.0));

		assert_eq!(Effect::delay(&effect), 0);
	}

	#[test]
	fn the_parameters_are_the_generators() {
		let mut effect = GeneratorEffect::<_, 2>::new(Level(0.5));

		assert_eq!(effect.get_parameters().len(), 1);
		assert!(effect.set_parameter("level", SetValue::Float(0.25)));
		assert_eq!(effect.generator().0, 0.25);
	}
}
