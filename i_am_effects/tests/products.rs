//! What this bundle promises.
//!
//! The interesting part of a product like this is not that it has plug-ins -- it is
//! that each plug-in is described by the effect it wraps, so the interface, the host's
//! parameter list and the sections all come from one place. These two tests hold that
//! down: every effect is a plug-in with sections, and the two pictures this crate can
//! draw are the ones the effects ask for.

use i_am_dsp::prelude::{Parameter, Parameters};
use i_am_dsp_iced::faceplate;
use i_am_effects::{Crusher, Drive, Filter};

/// The three plug-ins, as (name, parameters).
fn products() -> Vec<(&'static str, Vec<Parameter>)> {
	vec![
		("crusher", Crusher::new(44_100).get_parameters()),
		("drive", Drive::new(44_100).get_parameters()),
		("filter", Filter::new(44_100).get_parameters()),
	]
}

/// Every plug-in has controls, and the controls say where they belong.
#[test]
fn every_plug_in_is_laid_out_from_its_parameters() {
	for (name, parameters) in products() {
		assert!(!parameters.is_empty(), "{name} has parameters");

		let sections = faceplate::arrange(&parameters);

		assert!(!sections.is_empty(), "{name} is laid out in sections");
		assert!(
			sections.iter().all(|(_, members)| !members.is_empty()),
			"{name} has no empty sections"
		);
	}
}

/// The pictures a faceplate can be asked for are the ones the effects name.
#[test]
fn the_effects_ask_for_the_pictures_this_crate_draws() {
	let heroes = |parameters: &[Parameter]| -> Vec<String> {
		parameters
			.iter()
			.filter_map(|parameter| parameter.meta.layout.hero.as_deref().map(str::to_owned))
			.collect()
	};

	let drive = Drive::new(44_100).get_parameters();
	let filter = Filter::new(44_100).get_parameters();

	assert!(
		heroes(&drive).contains(&"transfer".to_string()),
		"a shaper is its curve, and it says so"
	);
	assert!(
		heroes(&filter).contains(&"spectrum".to_string()),
		"a filter is its response, and it says so"
	);
}
