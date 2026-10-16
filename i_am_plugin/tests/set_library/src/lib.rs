//! One library that exports two plug-ins, for the host probe to load.

use i_am_dsp::prelude::{Parameter, Parameters, SetValue};
use i_am_dsp::ProcessContext;
use i_am_dsp_iced::iced::widget::text;
use i_am_dsp_iced::{Processor, SyncedView};
use i_am_plugin::{Descriptor, Plugin, export_clap};

struct View;

impl SyncedView for View {
	type Message = ();

	fn update(&mut self, _: &()) {}

	fn view(&self) -> i_am_dsp_iced::iced::Element<'_, ()> {
		text("").into()
	}
}

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
			type SyncedView = View;

			fn delay(&self) -> usize {
				0
			}

			fn process(&mut self, _: &mut [f32; 2], _: &[[f32; 2]], _: &mut Box<dyn ProcessContext>) {}

			fn on_message(&self, _: ()) {}

			fn synced_view(&self) -> Self::SyncedView {
				View
			}
		}

		impl Plugin for $name {
			const DESCRIPTOR: Descriptor = Descriptor::new($id, $title);

			fn new() -> Self {
				Self
			}

			fn window_options() -> i_am_plugin::WindowOptions {
				i_am_plugin::WindowOptions::default()
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
