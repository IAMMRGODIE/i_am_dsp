//! The synth products, as the patcher's generators.
//!
//! A descriptor carries what a card needs (name, colour, faceplate size) and the
//! parameter schema the product itself declares, so the card and the instrument
//! cannot drift apart.

use i_am_dsp::prelude::{Parameter, Parameters};
use i_am_synths::{
	AdditiveWeb, CanvasWeb, DrummerWeb, FittedWeb, SamplerWeb, ScriptWeb, ShapesWeb, SpectrumWeb,
	StringWeb,
};

/// One synth product the patcher can host.
pub struct SynthDescriptor {
	/// The id a `synth:<id>` node kind uses.
	pub id: &'static str,
	/// The name the card shows.
	pub name: &'static str,
	/// The product's faceplate, which says the title, colour and size.
	pub faceplate: fn() -> i_am_web_gui::Faceplate,
	/// The parameter schema a fresh instrument starts with.
	pub parameters: fn() -> Vec<Parameter>,
}

impl SynthDescriptor {
	/// The kind id the graph stores.
	pub fn kind(&self) -> String {
		format!("synth:{}", self.id)
	}
}

fn additive_parameters() -> Vec<Parameter> {
	AdditiveWeb::default().get_parameters()
}

fn canvas_parameters() -> Vec<Parameter> {
	CanvasWeb::default().get_parameters()
}

fn drummer_parameters() -> Vec<Parameter> {
	DrummerWeb::default().get_parameters()
}

fn fitted_parameters() -> Vec<Parameter> {
	FittedWeb::default().get_parameters()
}

fn sampler_parameters() -> Vec<Parameter> {
	SamplerWeb::default().get_parameters()
}

fn script_parameters() -> Vec<Parameter> {
	ScriptWeb::default().get_parameters()
}

fn shapes_parameters() -> Vec<Parameter> {
	ShapesWeb::default().get_parameters()
}

fn spectrum_parameters() -> Vec<Parameter> {
	SpectrumWeb::default().get_parameters()
}

fn strings_parameters() -> Vec<Parameter> {
	StringWeb::default().get_parameters()
}

/// Every synth product, in the order the menu shows them.
pub const ALL: &[SynthDescriptor] = &[
	SynthDescriptor {
		id: "additive",
		name: "Additive",
		faceplate: i_am_synths::additive_faceplate,
		parameters: additive_parameters,
	},
	SynthDescriptor {
		id: "canvas",
		name: "Canvas",
		faceplate: i_am_synths::canvas_faceplate,
		parameters: canvas_parameters,
	},
	SynthDescriptor {
		id: "drummer",
		name: "Drummer",
		faceplate: i_am_synths::drummer_faceplate,
		parameters: drummer_parameters,
	},
	SynthDescriptor {
		id: "fitted",
		name: "Fitted",
		faceplate: i_am_synths::fitted_faceplate,
		parameters: fitted_parameters,
	},
	SynthDescriptor {
		id: "sampler",
		name: "Sampler",
		faceplate: i_am_synths::sampler_faceplate,
		parameters: sampler_parameters,
	},
	SynthDescriptor {
		id: "script",
		name: "Script",
		faceplate: i_am_synths::script_faceplate,
		parameters: script_parameters,
	},
	SynthDescriptor {
		id: "shapes",
		name: "Shapes",
		faceplate: i_am_synths::shapes_faceplate,
		parameters: shapes_parameters,
	},
	SynthDescriptor {
		id: "spectrum",
		name: "Spectrum",
		faceplate: i_am_synths::spectrum_faceplate,
		parameters: spectrum_parameters,
	},
	SynthDescriptor {
		id: "strings",
		name: "Strings",
		faceplate: i_am_synths::faceplate,
		parameters: strings_parameters,
	},
];

/// The descriptor for a `synth:<id>` kind's id part.
pub fn synth(id: &str) -> Option<&'static SynthDescriptor> {
	ALL.iter().find(|descriptor| descriptor.id == id)
}
