//! Processor plumbing for an effect, with no interface attached.

use std::time::Instant;

use i_am_dsp::prelude::{Effect, ParamMap, Paramed, Parameter, Parameters, SetValue};
use i_am_dsp::{NoteEvent, ProcessContext};

/// The rate an effect is built for before the host has said what it runs at.
///
/// The effects take their rate when they are built, and the ones whose coefficients
/// depend on it -- a filter, a reverb, a shaped drive -- are built wrong when it is
/// wrong. Rebuilding them once the host has said would be the fix, and it cannot be
/// done here: the parameter map a host reads is made together with the effect, and
/// has to stay the same map for the plug-in's life, or the host's automation ends up
/// in a map nothing is reading.
pub const DEFAULT_SAMPLE_RATE: usize = 44_100;

impl<E: Effect<2> + Parameters> EffectProcessor<E> {
	/// Drops the effect's history, for a clear-tail button.
	pub fn clear_tail(&mut self) {
		self.effect.value.clear_tail();
	}
}

/// A message from the interface.
#[derive(Debug, Clone)]
pub enum EffectMessage {
	/// A parameter moved.
	///
	/// The interface writes the new value into the shared map itself, which is what
	/// the audio thread reads, so the message carries nothing: a faceplate is built
	/// from parameters, and the parameters are the interface.
	Changed,
}

impl i_am_dsp::Message for EffectMessage {
	fn from_note_event(_: NoteEvent) -> Self {
		Self::Changed
	}

	fn note_event(&self) -> Option<NoteEvent> {
		None
	}

	fn tick(_: Instant) -> Self {
		Self::Changed
	}
}

/// An effect, with the interface its metadata describes.
pub struct EffectProcessor<E: Parameters> {
	/// The effect, kept inside a `Paramed` because that is what holds the parameter
	/// map the host and the interface both write into.
	///
	/// The map is made with the effect and has to stay the same map for the plug-in's
	/// whole life, so the effect is what moves when the rate changes: `Paramed` keeps
	/// its map when the value inside it is replaced.
	effect: Paramed<E>,
	/// How to build the effect for a sample rate.
	build: Box<dyn Fn(usize) -> E + Send + Sync>,
	sample_rate: usize,
}

impl<E: Parameters> Parameters for EffectProcessor<E> {
	fn get_parameters(&self) -> Vec<Parameter> {
		self.effect.get_parameters()
	}

	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		self.effect.set_parameter(identifier, value)
	}
}

impl<E: Parameters + 'static> EffectProcessor<E> {
	/// Wraps an effect, given how to build it for a sample rate.
	///
	/// It is built once for the placeholder rate, and rebuilt by
	/// [`activate`](Self::activate) when the host says what it really runs at.
	pub fn new(build: fn(usize) -> E) -> Self {
		Self::with_build(build)
	}

	/// Wraps an effect whose builder captures the state it has to keep.
	///
	/// An effect that owns shared state -- a loaded impulse response, say -- has
	/// to reconnect to it when [`activate`](Self::activate) replaces the effect for
	/// the host's rate, and a plain function pointer cannot carry that state.
	pub fn with_build(build: impl Fn(usize) -> E + Send + Sync + 'static) -> Self {
		let effect = build(DEFAULT_SAMPLE_RATE);

		Self {
			effect: Paramed::new(effect),
			build: Box::new(build),
			sample_rate: DEFAULT_SAMPLE_RATE,
		}
	}

	/// The effect itself, for a faceplate that publishes state the parameter map
	/// does not carry.
	pub fn effect(&self) -> &E {
		&self.effect.value
	}

	/// The rate the effect is currently built for.
	pub fn sample_rate(&self) -> usize {
		self.sample_rate
	}

	/// Rebuilds the effect for the rate the host just announced.
	///
	/// The parameter map is deliberately left alone: it is the one the host is
	/// holding, and swapping the effect inside it is the whole trick. The values are
	/// pushed into the new effect so nothing the interface moved is lost.
	pub fn activate(&mut self, sample_rate: usize) {
		if sample_rate == self.sample_rate || sample_rate == 0 {
			return;
		}

		self.effect.value = (self.build)(sample_rate);
		self.effect.sync_params();
		self.sample_rate = sample_rate;
	}

	/// The parameters, as a host presents them.
	pub fn parameters(&self) -> Vec<Parameter> {
		self.effect.get_parameters()
	}

	/// Where the interface writes and the audio thread reads.
	pub fn param_map(&self) -> ParamMap {
		self.effect.param_map()
	}

	/// Pushes the map's latest values into the effect without waiting for an
	/// audio block. Saving a project can happen between a knob move and the next
	/// process call; without this the saved state would carry the old value.
	pub fn sync_params(&mut self) {
		self.effect.sync_params();
	}

	/// Sets a parameter directly, which is what a test does and what a host does
	/// through the map.
	pub fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		self.effect.set_parameter(identifier, value)
	}
}

impl<E: Effect<2> + Parameters + Send + Sync + 'static> i_am_dsp::Processor for EffectProcessor<E> {
	type Message = EffectMessage;

	fn delay(&self) -> usize {
		self.effect.delay()
	}

	fn process(
		&mut self,
		samples: &mut [f32; 2],
		other: &[[f32; 2]],
		process_context: &mut Box<dyn ProcessContext>,
	) {
		// An effect takes its side-chains as slices of slices, which is the shape a
		// generator's output has; the processor is handed plain frames.
		let sides: Vec<&[f32; 2]> = other.iter().collect();

		// Paramed reads the shared map into the effect before this runs, which is how
		// a value the interface moved reaches the audio thread.
		self.effect.process(samples, &sides, process_context);
	}

	fn glide_from(&mut self, channel: u8, note: usize) {
		self.effect.glide_from(channel, note);
	}

	fn on_message(&self, _: EffectMessage) {}
}
