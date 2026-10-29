//! One library that exports two plug-ins, for the host probe to load.
//!
//! Both are headless: the probe scans and processes audio, and a GUI is not part of
//! that. Keeping them free of a toolkit is also what makes this fixture a check that
//! `i_am_plugin` needs no GUI crate at all.

use i_am_dsp::Processor;
use i_am_dsp::prelude::{Parameter, Parameters, SetValue};
use i_am_dsp::ProcessContext;
use i_am_plugin::{Descriptor, Plugin, export_clap};

macro_rules! test_plugin {
	($name:ident, $id:literal, $title:literal) => {
		struct $name;

		impl Parameters for $name {
			fn get_parameters(&self) -> Vec<Parameter> {
				Vec::new()
			}

			fn set_parameter(&mut self, _: &str, _: SetValue) -> bool {
				false
			}
		}

		impl Processor for $name {
			type Message = ();

			fn delay(&self) -> usize {
				0
			}

			fn process(&mut self, _: &mut [f32; 2], _: &[[f32; 2]], _: &mut Box<dyn ProcessContext>) {}

			fn on_message(&self, _: ()) {}
		}

		impl Plugin for $name {
			const DESCRIPTOR: Descriptor = Descriptor::new($id, $title);

			fn new() -> Self {
				Self
			}

			fn param_map(&self) -> i_am_dsp::prelude::ParamMap {
				i_am_dsp::prelude::ParamMap::default()
			}
		}
	};
}

test_plugin!(Crusher, "dev.i-am-dsp.probe.crusher", "Probe Crusher");
test_plugin!(Filter, "dev.i-am-dsp.probe.filter", "Probe Filter");

// One call, one library, two plug-ins.
export_clap!(Crusher, Filter);
