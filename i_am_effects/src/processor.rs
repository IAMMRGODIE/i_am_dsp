//! What every effect in this bundle has in common.
//!
//! An effect here is three things: the DSP that does the work, the parameters it
//! declares, and the interface those parameters describe. Only the first is written
//! once per effect -- the other two are this module, and that is the whole point of
//! the metadata the DSP carries.

use std::sync::atomic::Ordering;
use std::time::Instant;

use i_am_dsp::prelude::{Effect, ParamMap, Parameter, Paramed, Parameters, SetValue};
use i_am_dsp::{NoteEvent, ProcessContext};
use i_am_dsp_iced::iced::widget::canvas;
use i_am_dsp_iced::iced::{Element, Length};
use i_am_dsp_iced::tools::{spectrum::Spectrum, transfer::transfer};
use i_am_dsp_iced::{Message as IcedMessage, Processor, SyncedView, faceplate, styles};

/// The rate an effect is built for before the host has said what it runs at.
///
/// The effects take their rate when they are built, and the ones whose coefficients
/// depend on it -- a filter, a reverb, a shaped drive -- are built wrong when it is
/// wrong. Rebuilding them once the host has said would be the fix, and it cannot be
/// done here: the parameter map a host reads is made together with the effect, and
/// has to stay the same map for the plug-in's life, or the host's automation ends up
/// in a map nothing is reading.
pub const DEFAULT_SAMPLE_RATE: usize = 44_100;

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

impl IcedMessage for EffectMessage {
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
	build: fn(usize) -> E,
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

impl<E: Parameters> EffectProcessor<E> {
	/// Wraps an effect, given how to build it for a sample rate.
	///
	/// It is built once for the placeholder rate, and rebuilt by
	/// [`activate`](Self::activate) when the host says what it really runs at.
	pub fn new(build: fn(usize) -> E) -> Self {
		Self {
			effect: Paramed::new(build(DEFAULT_SAMPLE_RATE)),
			build,
			sample_rate: DEFAULT_SAMPLE_RATE,
		}
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

	/// Sets a parameter directly, which is what a test does and what a host does
	/// through the map.
	pub fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		self.effect.set_parameter(identifier, value)
	}
}

impl<E: Effect<2> + Parameters + Send + Sync + 'static> Processor for EffectProcessor<E> {
	type Message = EffectMessage;
	type SyncedView = EffectView;

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

	fn on_message(&self, _: EffectMessage) {}

	fn synced_view(&self) -> EffectView {
		EffectView {
			parameters: self.parameters(),
			param_map: self.param_map(),
		}
	}
}

/// The interface: one faceplate, built from the parameters it was given.
pub struct EffectView {
	parameters: Vec<Parameter>,
	param_map: ParamMap,
}

impl SyncedView for EffectView {
	type Message = EffectMessage;

	fn update(&mut self, _: &Self::Message) {
		// The controls write into the map and are drawn from the snapshot, so the
		// snapshot has to be taken again for a knob to follow the drag.
		faceplate::refresh(&mut self.parameters, &self.param_map);
	}

	fn view(&self) -> Element<'_, EffectMessage> {
		let map = &self.param_map;

		faceplate::faceplate_with_heroes(
			&self.parameters,
			|identifier, value| {
				map[identifier].store(value, Ordering::SeqCst);

				EffectMessage::Changed
			},
			|kind| hero(kind, map),
		)
	}
}


/// The first parameter of the given names that this effect has.
fn value_of(map: &ParamMap, names: &[&str]) -> Option<f32> {
	names.iter().find_map(|name| {
		map.query_param_index(name)
			.and_then(|index| map[index].load(Ordering::SeqCst).float())
	})
}

/// The picture a section asked for, drawn from the values it holds.
///
/// This is a host's half of a hero: the metadata says a section is about a
/// "transfer" curve and this decides what that looks like, because only the host
/// knows what it can draw. A name nobody here recognises draws nothing, and the
/// section is laid out as controls -- a hero is an addition, never a requirement.
fn hero<'a>(kind: &str, map: &ParamMap) -> Option<Element<'a, EffectMessage>> {
	match kind {
		// A shaper is its curve: the more it is driven, the harder it bends.
		"transfer" => {
			let drive = value_of(map, &["drive", "amount", "threshold"]).unwrap_or(0.5);
			let gain = 1.0 + drive.clamp(0.0, 1.0) * 6.0;

			Some(
				canvas(transfer(move |input| (input * gain).tanh()))
					.width(Length::Fill)
					.height(styles::u(14.0))
					.into(),
			)
		},
		// A filter is its response: where it is told to cut, and how much it rings.
		"spectrum" => {
			let cutoff = value_of(map, &["cutoff"]).unwrap_or(1_000.0);
			let q = value_of(map, &["q"]).unwrap_or(0.707);
			let response = Spectrum::new();

			response.update(&lowpass_response(cutoff, q));

			Some(
				canvas(response)
					.width(Length::Fill)
					.height(styles::u(14.0))
					.into(),
			)
		},
		_ => None,
	}
}

/// The magnitude response of the filter this effect is, on the logarithmic axis a
/// spectrum draws: 0 to 1, low frequencies on the left.
///
/// The shape is the second-order transfer function the biquad implements rather than a
/// curve drawn to look like one: flat below the corner, twelve decibels an octave below
/// it, and the resonance the filter was given showing as a peak at the corner. Drawing
/// it from the distance to the corner instead -- which is what this did at first -- falls
/// away on both sides and reads as a bandpass, which is not what the effect does.
///
/// Vertically it is decibels over a fixed range, because a filter falls away by orders of
/// magnitude: on a linear axis everything past the corner is a cliff against the floor.
fn lowpass_response(cutoff: f32, q: f32) -> Vec<f32> {
	const POINTS: usize = 64;
	const LOW: f32 = 20.0;
	const HIGH: f32 = 20_000.0;
	/// How far below the loudest point the picture goes.
	const RANGE_DB: f32 = 40.0;

	let corner = cutoff.clamp(LOW, HIGH);
	let resonance = q.max(0.05);

	let magnitudes: Vec<f32> = (0..POINTS)
		.map(|index| {
			let position = index as f32 / (POINTS - 1) as f32;
			let frequency = LOW * (HIGH / LOW).powf(position);
			let ratio = frequency / corner;

			// |H| of a second-order lowpass at this frequency.
			let real = 1.0 - ratio * ratio;
			let imaginary = ratio / resonance;

			1.0 / (real * real + imaginary * imaginary).sqrt()
		})
		.collect();

	// One scale for every curve, whatever its resonance: the loudest point of this one
	// is the top of the picture and everything else is read against it.
	let loudest = magnitudes.iter().copied().fold(f32::MIN, f32::max).max(1e-6);

	magnitudes
		.into_iter()
		.map(|magnitude| {
			let decibels = 20.0 * (magnitude / loudest).max(1e-6).log10();

			(1.0 + decibels / RANGE_DB).clamp(0.0, 1.0)
		})
		.collect()
}

/// A descriptor for one effect in the bundle, and the plumbing that turns it into a
/// plug-in.
///
/// Everything a plug-in has to say about itself is here or in the effect's
/// parameters: the DSP says what its controls are and where they belong, and this
/// says what the plug-in is called.
#[macro_export]
macro_rules! effect_plugin {
	($name:ident, $effect:ty, $build:expr, $id:literal, $title:literal, $size:expr) => {
		#[doc = concat!("The ", $title, " plug-in.")]
		pub struct $name($crate::processor::EffectProcessor<$effect>);

		impl Default for $name {
			fn default() -> Self {
				Self($crate::processor::EffectProcessor::new($build))
			}
		}

		impl $name {
			/// The rate the effect inside is currently built for.
			pub fn sample_rate(&self) -> usize {
				self.0.sample_rate()
			}
		}

		impl i_am_dsp::prelude::Parameters for $name {
			fn get_parameters(&self) -> ::std::vec::Vec<i_am_dsp::prelude::Parameter> {
				self.0.parameters()
			}

			fn set_parameter(
				&mut self,
				identifier: &str,
				value: i_am_dsp::prelude::SetValue,
			) -> bool {
				self.0.set_parameter(identifier, value)
			}
		}

		impl i_am_dsp_iced::Processor for $name {
			type Message = $crate::processor::EffectMessage;
			type SyncedView = $crate::processor::EffectView;

			fn delay(&self) -> usize {
				i_am_dsp_iced::Processor::delay(&self.0)
			}

			fn process(
				&mut self,
				samples: &mut [f32; 2],
				other: &[[f32; 2]],
				process_context: &mut ::std::boxed::Box<dyn i_am_dsp::ProcessContext>,
			) {
				i_am_dsp_iced::Processor::process(&mut self.0, samples, other, process_context)
			}

			fn on_message(&self, message: Self::Message) {
				i_am_dsp_iced::Processor::on_message(&self.0, message)
			}

			fn synced_view(&self) -> Self::SyncedView {
				i_am_dsp_iced::Processor::synced_view(&self.0)
			}
		}

		impl i_am_plugin::Plugin for $name {
			const DESCRIPTOR: i_am_plugin::Descriptor = i_am_plugin::Descriptor::new($id, $title);

			fn on_activate(&mut self, sample_rate: usize) {
				self.0.activate(sample_rate);
			}

			fn new() -> Self {
				Self::default()
			}

			fn window_options() -> i_am_plugin::WindowOptions {
				i_am_plugin::WindowOptions::new()
					.with_size($size)
					.with_title($title)
			}

			fn param_map(&self) -> i_am_dsp::prelude::ParamMap {
				self.0.param_map()
			}
		}
	};
}

#[cfg(test)]
mod tests {
	use super::*;
	use i_am_dsp::prelude::Value;

	/// A lowpass is flat below its corner and falls away above it. This is the shape the
	/// picture has to have: drawing it from the distance to the corner falls away on both
	/// sides, and a bump in the middle is a bandpass.
	#[test]
	fn the_filter_picture_is_a_lowpass_and_not_a_bump() {
		let response = lowpass_response(1_000.0, 0.707);
		let last = response.len() - 1;
		let at = |frequency: f32| {
			let position = (frequency / 20.0).ln() / (20_000.0_f32 / 20.0).ln();

			response[(position * last as f32).round() as usize]
		};

		assert!(
			(at(20.0) - at(200.0)).abs() < 0.05,
			"the passband is flat, not a slope up to the corner"
		);
		assert!(
			at(200.0) > at(20_000.0) + 0.5,
			"and it falls away above the corner, which is what makes it a lowpass"
		);
	}

	/// A control draws the value it was handed, so the view has to hand it a new one
	/// when the map changes. This is the bug that made every knob in the demo sit still
	/// while its value moved underneath.
	#[test]
	fn a_view_follows_the_values_in_the_map() {
		let effect = EffectProcessor::new(|_rate| i_am_dsp::prelude::HardClipper::new(0.8, 1.0));
		let mut view = effect.synced_view();
		let map = effect.param_map();

		let drawn = |view: &EffectView| match view.parameters[0].value {
			Value::Float { value, .. } => value,
			_ => panic!("the shaper's first control is a float"),
		};

		let before = drawn(&view);

		// What a drag does: the callback writes into the shared map.
		map[view.parameters[0].identifier.as_str()].store(SetValue::Float(0.25), Ordering::SeqCst);
		view.update(&EffectMessage::Changed);

		assert_ne!(before, 0.25, "the test is pointless if it started there");
		assert_eq!(
			drawn(&view),
			0.25,
			"the snapshot the controls are drawn from followed the map"
		);
	}
}
