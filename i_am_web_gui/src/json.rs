use i_am_plugin::Plugin;

use crate::face;
use crate::{Control, Faceplate};

/// The controls a parameter list describes, flattened across sections.
///
/// A rack shows one card per module and gives each card the controls its effect
/// already declares; the conversion is the same one the faceplate page uses.
pub fn controls(parameters: &[i_am_dsp::prelude::Parameter]) -> Vec<Control> {
	face::sections(parameters)
		.into_iter()
		.flat_map(|section| section.controls)
		.collect()
}

/// The faceplate JSON for a plug-in, for a test or a preview.
pub fn face_json<P: Plugin>(processor: &P, face: &Faceplate) -> String {
	face::face_json(face, &face::sections(&processor.get_parameters()))
}

/// The state JSON for a plug-in, for a test or a preview.
pub fn state_json<P: Plugin>(processor: &P) -> String {
	let sections = face::sections(&processor.get_parameters());

	face::state_json(&processor.param_map(), &sections)
}

/// Applies one id=value command, for a test.
pub fn apply_command<P: Plugin>(processor: &P, body: &str) -> bool {
	let sections = face::sections(&processor.get_parameters());

	face::apply_command(&processor.param_map(), &sections, body)
}
