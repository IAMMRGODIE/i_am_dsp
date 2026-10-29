//! I Am Crusher, with a WebView2 editor.
//!
//! This is a separate product from the iced Crusher in `i_am_effects`: its own crate,
//! its own descriptor, and no iced dependency. It reuses the GUI-neutral
//! `EffectProcessor` from `i_am_effects_core` and the Crusher DSP, and puts the
//! `WryEditor` from `i_am_plugin_wry` in front of them with the page in `gui.html`.
//!
//! The editor is resizable: the page scales its fixed 400x640 stage as one piece, and
//! the product asks the host to preserve that aspect ratio while it does.
//!
//! The plug-in id is `iamdsp.effects.crusher.web`, so a host can keep it next to the
//! released `iamdsp.effects.crusher` without confusing the two.
//!
//! ```text
//! cargo build -p i_am_crusher_wry
//! cargo bundle -p i_am_crusher_wry --features vst3
//! cargo run -p i_am_crusher_wry --example crusher_web
//! ```

mod gui;

use i_am_dsp::prelude::{Crusher as CrusherDsp, ParamMap, Parameter, Parameters, SetValue};
use i_am_dsp::ProcessContext;
use i_am_effects_core::processor::{EffectMessage, EffectProcessor};
use i_am_plugin::{
	Descriptor, Editor, Plugin, Tag, WindowOptions, export_clap, AspectRatioStrategy,
	GuiResizeHints,
};

/// The Crusher, with the parameters and DSP of the released one and a web editor.
pub struct CrusherWeb(EffectProcessor<CrusherDsp<2>>);

impl Parameters for CrusherWeb {
	fn get_parameters(&self) -> Vec<Parameter> {
		self.0.parameters()
	}

	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		self.0.set_parameter(identifier, value)
	}
}

impl i_am_dsp::Processor for CrusherWeb {
	type Message = EffectMessage;

	fn delay(&self) -> usize {
		i_am_dsp::Processor::delay(&self.0)
	}

	fn process(
		&mut self,
		samples: &mut [f32; 2],
		other: &[[f32; 2]],
		process_context: &mut Box<dyn ProcessContext>,
	) {
		i_am_dsp::Processor::process(&mut self.0, samples, other, process_context)
	}

	fn on_message(&self, message: Self::Message) {
		i_am_dsp::Processor::on_message(&self.0, message)
	}
}

impl Plugin for CrusherWeb {
	const DESCRIPTOR: Descriptor = Descriptor::new("iamdsp.effects.crusher.web", "I Am Crusher Web")
		.with_tags(&[Tag::AudioEffect, Tag::Distortion])
		.with_vendor("iamplugins")
		.with_version(env!("CARGO_PKG_VERSION"));

	const HAS_GUI: bool = true;

	fn new() -> Self {
		Self(EffectProcessor::new(CrusherDsp::<2>::new))
	}

	fn window_options() -> WindowOptions {
		WindowOptions::new()
			.with_size((400.0, 640.0))
			.with_title("I Am Crusher Web")
			// The page scales its whole 400x640 stage to the window, so the host may
			// resize it as long as the aspect ratio is kept.
			.with_resize_hints(GuiResizeHints {
				can_resize_horizontally: true,
				can_resize_vertically: true,
				strategy: AspectRatioStrategy::Preserve {
					width: 400,
					height: 640,
				},
			})
	}

	fn param_map(&self) -> ParamMap {
		self.0.param_map()
	}

	fn on_activate(&mut self, sample_rate: usize) {
		self.0.activate(sample_rate);
	}

	fn editor(&self) -> Option<Box<dyn Editor<Self>>> {
		gui::editor(self)
	}
}

export_clap!(CrusherWeb);

#[cfg(test)]
mod tests {
	use super::*;

	/// The page scales its 400x640 stage, and the host is told it may resize the
	/// editor as long as it keeps that aspect ratio.
	#[test]
	fn the_window_is_resizable_along_the_design_aspect() {
		let hints = CrusherWeb::window_options()
			.resize_hints
			.expect("the web editor is resizable");

		assert!(hints.can_resize_horizontally);
		assert!(hints.can_resize_vertically);
		assert!(matches!(
			hints.strategy,
			AspectRatioStrategy::Preserve {
				width: 400,
				height: 640
			}
		));
	}
}
