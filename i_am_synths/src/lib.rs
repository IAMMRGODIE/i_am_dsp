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

mod additive;
mod canvas;
mod script;
mod shapes;
mod spectrum;
mod strings;

pub use additive::{AdditiveWeb, additive_faceplate};
pub use canvas::{CanvasWeb, canvas_faceplate, canvas_shape};
pub use script::{ScriptWeb, script_faceplate};
pub use shapes::{ShapesWeb, shapes_faceplate};
pub use spectrum::{SpectrumWeb, spectrum_faceplate};
pub use strings::{StringWeb, faceplate};

use std::sync::atomic::Ordering;

use i_am_dsp::prelude::{ParamMap, Parameters, StringInstrument};
use i_am_dsp::{NoteEvent, ProcessContext, Processor};
use i_am_web_gui::Scope;

fn json_escape(text: &str) -> String {
	let mut output = String::with_capacity(text.len());

	for character in text.chars() {
		match character {
			'"' => output.push_str("\\\""),
			'\\' => output.push_str("\\\\"),
			'\n' => output.push_str("\\n"),
			'\r' => output.push_str("\\r"),
			'\t' => output.push_str("\\t"),
			character if (character as u32) < 0x20 => {
				output.push_str(&format!("\\u{:04x}", character as u32));
			}
			character => output.push(character),
		}
	}

	output
}

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
