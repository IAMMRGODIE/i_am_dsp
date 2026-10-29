//! The iced faceplate, and the macro that turns an effect into a plug-in.
//!
//! The processor itself lives in `i_am_effects_core`; this module adds the view the
//! parameters describe, and the product macro every plug-in in the bundle shares.

use std::sync::atomic::Ordering;

use i_am_dsp::prelude::{ParamMap, Parameter, Parameters};
use i_am_dsp_iced::iced::widget::canvas;
use i_am_dsp_iced::iced::{Element, Length};
use i_am_dsp_iced::tools::{spectrum::Spectrum, transfer::transfer};
use i_am_dsp_iced::{SyncedView, faceplate, styles};

use crate::processor::{EffectMessage, EffectProcessor};

/// The iced view of a processor: one faceplate, built from the parameters it was given.
///
/// This is a free function rather than an `i_am_dsp_iced::Processor` implementation
/// on the processor: the processor lives in another crate, and only the product type
/// (which is local) may implement a foreign trait for it.
#[doc(hidden)]
pub fn view_for<E: Parameters>(processor: &EffectProcessor<E>) -> EffectView {
	EffectView {
		parameters: processor.parameters(),
		param_map: processor.param_map(),
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
		// A shaper is its curve: the mode, the amount and the output gain are the
		// three things the picture depends on, so the picture reads them.
		"transfer" => {
			let mode = value_of(map, &["mode"]).unwrap_or(0.0) as i32;
			let amount = value_of(map, &["threshold", "amount", "drive"])
				.unwrap_or(0.5)
				.clamp(0.01, 4.0);
			let output = value_of(map, &["gain"]).unwrap_or(1.0).clamp(0.01, 4.0);

			Some(
				canvas(transfer(move |input| {
					let shaped = if mode == 1 {
						(input / amount).tanh()
					} else {
						input.clamp(-amount, amount)
					};

					(shaped * output).clamp(-1.0, 1.0)
				}))
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

		// The audio side: no view, no toolkit.
		impl i_am_dsp::Processor for $name {
			type Message = $crate::processor::EffectMessage;

			fn delay(&self) -> usize {
				i_am_dsp::Processor::delay(&self.0)
			}

			fn process(
				&mut self,
				samples: &mut [f32; 2],
				other: &[[f32; 2]],
				process_context: &mut ::std::boxed::Box<dyn i_am_dsp::ProcessContext>,
			) {
				i_am_dsp::Processor::process(&mut self.0, samples, other, process_context)
			}

			fn on_message(&self, message: Self::Message) {
				i_am_dsp::Processor::on_message(&self.0, message)
			}
		}

		// The view, for the iced toolkit.
		impl i_am_dsp_iced::Processor for $name {
			type SyncedView = $crate::EffectView;

			fn synced_view(&self) -> Self::SyncedView {
				$crate::view_for(&self.0)
			}
		}

		impl i_am_plugin::Plugin for $name {
			const DESCRIPTOR: i_am_plugin::Descriptor = i_am_plugin::Descriptor::new($id, $title);

			const HAS_GUI: bool = true;

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

			fn editor(&self) -> ::std::option::Option<::std::boxed::Box<dyn i_am_plugin::Editor<Self>>> {
				::std::option::Option::Some(::std::boxed::Box::new(
					i_am_plugin_iced::IcedEditor::new(
						i_am_dsp_iced::Processor::synced_view(self),
					),
				))
			}
		}
	};
}

#[cfg(test)]
mod tests {
	use super::*;
	use i_am_dsp::prelude::{SetValue, Value};

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
		let mut view = view_for(&effect);
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