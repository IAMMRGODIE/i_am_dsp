//! One library, two plug-ins.
//!
//! A bundle ships as one binary while every plug-in in it is a plug-in of its own to
//! the host. That is what the set factory is for, and this is the smallest thing
//! that can be checked without a host: the library lists both, each with its own
//! name and id.

use i_am_dsp::prelude::{Parameter, Parameters, SetValue};
use i_am_dsp::ProcessContext;
use i_am_dsp_iced::iced::widget::text;
use i_am_dsp_iced::{Processor, SyncedView};
use i_am_plugin::{Descriptor, Plugin, export_clap};

/// The smallest view a processor can have.
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

test_plugin!(First, "dev.i-am-dsp.test.first", "First Test");
test_plugin!(Second, "dev.i-am-dsp.test.second", "Second Test");

// One call, one library, two plug-ins.
export_clap!(First, Second);

#[test]
fn the_library_exports_every_plugin_it_was_given() {
	use i_am_plugin::__private::PluginSetSource;

	let entries = PluginSet::entries();

	assert_eq!(entries.len(), 2, "a set of two lists two");

	let names: Vec<_> = entries
		.iter()
		.map(|entry| entry.descriptor().name().unwrap_or_default().to_string_lossy().into_owned())
		.collect();

	assert_eq!(names, vec!["First Test", "Second Test"], "in the order they were listed");
	assert_ne!(
		entries[0].descriptor().id(),
		entries[1].descriptor().id(),
		"each keeps its own id, which is what a host names it by"
	);
}
