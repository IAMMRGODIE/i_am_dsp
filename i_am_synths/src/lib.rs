//! The synthesizer products.
//!
//! An instrument is a plug-in with no audio input and a note input, and its face
//! is the same web faceplate the effects use: the parameter metadata is the whole
//! layout, and i_am_web_gui draws it. The one thing an instrument needs that an
//! effect does not is a preset: choosing an instrument has to move the other
//! controls as well, which a single parameter cannot do, so the editor answers the
//! preset choice itself before the generic command parser sees it.
//!
//! Each product lives in its own module; this one only keeps the body they share
//! and the names the bundle exports. The CLAP and VST3 entry points live in
//! i_am_synths_bundle, because clap_entry is a global symbol.

/// Implements the two traits every instrument forwards unchanged.
///
/// A product owns its window, its descriptor and its editor -- that is what it
/// is -- and keeps its `processor`, `scope` and `bypass` fields. The parameter
/// map a host reads and the note-to-sound path `play_synth` plays are the same
/// for all of them, so they are written here once. A product with a custom
/// parameter setter -- the drummer writes a kit's knobs when the kit changes --
/// takes `synth_processor!` alone and keeps its own `Parameters` impl.
///
/// The macro has to stay above the `mod` declarations: a `macro_rules!` is
/// visible only to the items that follow it.
macro_rules! synth_processor {
	($name:ident) => {
		impl i_am_dsp::Processor for $name {
			type Message = i_am_plugin::processor::EffectMessage;

			fn delay(&self) -> usize {
				i_am_dsp::Processor::delay(&self.processor)
			}

			fn process(
				&mut self,
				samples: &mut [f32; 2],
				other: &[[f32; 2]],
				process_context: &mut Box<dyn i_am_dsp::ProcessContext>,
			) {
				$crate::play_synth(
					&mut self.processor,
					&self.scope,
					&self.bypass,
					samples,
					other,
					process_context,
				);
			}

			fn on_message(&self, message: Self::Message) {
				i_am_dsp::Processor::on_message(&self.processor, message);
			}
		}
	};
}

macro_rules! synth_product {
	($name:ident) => {
		impl i_am_dsp::prelude::Parameters for $name {
			fn get_parameters(&self) -> Vec<i_am_dsp::prelude::Parameter> {
				self.processor.parameters()
			}

			fn set_parameter(
				&mut self,
				identifier: &str,
				value: i_am_dsp::prelude::SetValue,
			) -> bool {
				self.processor.set_parameter(identifier, value)
			}
		}

		synth_processor!($name);
	};
}

mod additive;
mod canvas;
mod drummer;
mod fitted;
mod sampler;
mod script;
mod shapes;
mod spectrum;
mod strings;

pub use additive::{AdditiveWeb, additive_faceplate};
pub use canvas::{CanvasWeb, canvas_faceplate, canvas_shape};
pub use drummer::{DrummerWeb, faceplate as drummer_faceplate};
pub use fitted::{FittedWeb, faceplate as fitted_faceplate};
pub use sampler::{SamplerWeb, sampler_faceplate};
pub use script::{ScriptWeb, script_faceplate};
pub use shapes::{ShapesWeb, shapes_faceplate};
pub use spectrum::{SpectrumWeb, spectrum_faceplate};
pub use strings::{StringWeb, faceplate};

use std::sync::atomic::Ordering;

use i_am_dsp::prelude::{ParamMap, Parameters, StringInstrument};
use i_am_dsp::{NoteEvent, ProcessContext, Processor};
use i_am_web_gui::Scope;

pub(crate) use i_am_web_gui::json_escape;

/// The body every instrument shares: the note badge, the bypass and the audio.
///
/// An instrument has nothing to pass through, so bypass is silence, and the scope
/// always sees silence as its input: the pictures draw the sound the voice made.
fn play_synth<P: Processor>(
	processor: &mut P,
	scope: &Scope,
	bypass: &std::sync::atomic::AtomicBool,
	samples: &mut [f32; 2],
	other: &[[f32; 2]],
	process_context: &mut Box<dyn ProcessContext>,
) {
	// The note badge: the page shows the note the host is playing.
	for event in process_context.events() {
		match event {
			NoteEvent::NoteOn { note, .. } => scope.set_note(*note as i32),
			NoteEvent::NoteOff { note, .. } | NoteEvent::Stop { note, .. } => {
				if scope.note() == *note as i32 {
					scope.set_note(-1);
				}
			}
			NoteEvent::ImmediateStop => scope.set_note(-1),
			_ => {}
		}
	}

	if bypass.load(Ordering::Relaxed) {
		*samples = [0.0; 2];
		scope.push_stereo([0.0; 2], [0.0; 2], 0.0);
		return;
	}

	Processor::process(processor, samples, other, process_context);

	scope.push_stereo([0.0; 2], *samples, 0.0);
}

/// Answers the preset choice before the generic parameter parser sees it.
///
/// The DSP follows the preset parameter every block, so it switches the string
/// and the body on its own, but the other controls are independent parameters:
/// loading a patch means moving all of them. The map is the shared state the page
/// and the audio thread both read, so writing them here is the whole operation.
fn apply_preset_command(map: &ParamMap, rate: usize, body: &str) -> bool {
	let Some(value) = body.strip_prefix("preset=") else {
		return false;
	};

	let Ok(index) = value.trim().parse::<i32>() else {
		return false;
	};

	apply_preset(map, rate, index);

	true
}

/// Moves every control to the preset's values, except the performance controls.
fn apply_preset(map: &ParamMap, rate: usize, index: i32) {
	let mut instrument = StringInstrument::new(rate);
	instrument.select_preset(index);

	for parameter in instrument.get_parameters() {
		// The transpose is a performance control, not part of an instrument; the
		// DSP's own preset loading leaves it alone and so does this.
		if parameter.identifier == "transpose" {
			continue;
		}

		let identifier = parameter.identifier;
		let value = parameter.value.to_set_value();
		map.set(&identifier, value, Ordering::Relaxed);
	}
}

#[cfg(test)]
mod tests;

#[cfg(test)]
/// A mono 16-bit WAV file, for the tests that import one.
pub(crate) fn wav(samples: &[f32], rate: u32) -> Vec<u8> {
	use hound::{SampleFormat, WavSpec, WavWriter};
	use std::io::Cursor;

	let spec = WavSpec {
		channels: 1,
		sample_rate: rate,
		bits_per_sample: 16,
		sample_format: SampleFormat::Int,
	};
	let mut cursor = Cursor::new(Vec::new());

	{
		let mut writer = WavWriter::new(&mut cursor, spec).expect("a writer");

		for sample in samples {
			writer
				.write_sample((sample.clamp(-1.0, 1.0) * 32767.0) as i16)
				.expect("a sample");
		}

		writer.finalize().expect("a finished file");
	}

	cursor.into_inner()
}
