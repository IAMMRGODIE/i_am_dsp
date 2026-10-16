//! clap_plugin

use crate::shared::*;
use crate::plugin::{Plugin, PluginAuExt};
use crate::main_thread::{PluginMain};
use crate::latency::{LatencyWatch};
use crate::audio::{AudioProcessor};

/// A wrapper around a plugin that implements the `clack_plugin::Plugin` trait.
#[derive(Default)]
pub struct ClapPlugin<P: Plugin>(std::marker::PhantomData<P>);

impl<P: Plugin> DefaultPluginFactory for ClapPlugin<P> 
where 
	<P as i_am_dsp_iced::Processor>::Message: std::fmt::Debug
{
	fn get_descriptor() -> clack_plugin::prelude::PluginDescriptor {
		let descriptor = P::DESCRIPTOR;
		let mut out = clack_plugin::prelude::PluginDescriptor::new(descriptor.id, descriptor.name);
		if let Some(vendor) = descriptor.vendor {
			out = out.with_vendor(vendor)
		}
		if let Some(url) = descriptor.url {
			out = out.with_url(url)
		}
		if let Some(vendor) = descriptor.vendor {
			out = out.with_vendor(vendor)
		}
		if let Some(url) = descriptor.url {
			out = out.with_url(url)
		}
		if let Some(manual_url) = descriptor.manual_url {
			out = out.with_manual_url(manual_url)
		}
		if let Some(support_url) = descriptor.support_url {
			out = out.with_support_url(support_url)
		}
		if let Some(version) = descriptor.version {
			out = out.with_version(version)
		}
		if let Some(description) = descriptor.description {
			out = out.with_description(description)
		}
		out = out.with_features(descriptor.tags.iter().map(|inner| inner.as_cstr()));
		out
	}

	fn new_shared(_: clack_plugin::prelude::HostSharedHandle<'_>) -> Result<Self::Shared<'_>, PluginError> {
		Ok(())
	}

	fn new_main_thread<'a>(
		host: clack_plugin::prelude::HostMainThreadHandle<'a>,
		_: &'a Self::Shared<'a>,
	) -> Result<Self::MainThread<'a>, PluginError> {
		let options = P::window_options();
		let processor = P::new();
		let param_map = processor.param_map();
		// The plugin starts at the latency it reports before it has processed anything, which
		// is what a host compensates for when it activates the plugin.
		let latency = LatencyWatch::new(processor.delay() as u32);

		// SAFETY: the host owns the plugin, so the handle it hands us here outlives this
		// struct; the lifetime is erased only because `PluginMain` is not parameterized by it.
		let host = unsafe { host.with_arbitrary_lifetime() };

		Ok(PluginMain {
			window_size: options.window_size,
			scale_factor: 1.0,
			title: options.title,
			resize_hints: options.resize_hints,
			param_map,
			processor: Box::pin(processor), 
			host: Some(host),
			latency,
			parent: None,
			handle: None,
			message_receivers: vec![],
		})
	}
}

impl<P: Plugin> PluginLatencyImpl for PluginMain<P> {
	fn get(&mut self) -> u32 {
		// What the audio thread last ran at, which is the latency the host has to compensate.
		self.latency.current()
	}
}

impl<P: Plugin> PluginNotePortsImpl for PluginMain<P> {
	fn count(&mut self, is_input: bool) -> u32 {
		if is_input {
			P::INPUT_MIDI_PORTS.len() as u32
		}else {
			P::OUTPUT_MIDI_PORTS.len() as u32
		}
	}

	fn get(&mut self, index: u32, is_input: bool, writer: &mut clack_extensions::note_ports::NotePortInfoWriter) {
		let ports = if is_input {
			P::INPUT_MIDI_PORTS
		}else {
			P::OUTPUT_MIDI_PORTS
		};

		if index >= ports.len() as u32 {
			return;
		}

		writer.set(&NotePortInfo {
			id: index.into(),
			name: ports[index as usize].name.as_bytes(),
			supported_dialects: NoteDialects::CLAP,
			preferred_dialect: Some(NoteDialect::Clap),
		});
	}
}

impl<P: Plugin> clack_plugin::plugin::Plugin for ClapPlugin<P> 
where 
	<P as i_am_dsp_iced::Processor>::Message: std::fmt::Debug
{
	type Shared<'a> = ();
	type AudioProcessor<'a> = AudioProcessor<'a, P>;
	type MainThread<'a> = PluginMain<P>;

	fn declare_extensions(builder: &mut clack_plugin::prelude::PluginExtensions<Self>, _: Option<&Self::Shared<'_>>) {
		builder
			.register::<PluginAudioPorts>()
			.register::<PluginParams>()
			.register::<PluginState>()
			.register::<PluginAsVST3>()
			.register::<PluginLatency>()
			.register::<PluginNotePorts>()
			.register::<PluginGui>();
	}
}

impl<P: Plugin> ClapPlugin<P> {
	const VST3_ID: [u8; 16] = xxh3_128(P::DESCRIPTOR.id.as_bytes()).to_be_bytes();

	pub(crate) const VST3_PLUGIN_INFO: PluginInfoAsVST3<'static> = PluginInfoAsVST3::new(
		if let Some(vendor) = P::DESCRIPTOR.vendor {
			if let Ok(vendor) = CStr::from_bytes_until_nul(vendor.as_bytes()) {
				Some(vendor)
			}else {
				None
			}
		}else {
			None
		},
		Some(&Self::VST3_ID),
		None,
	);
}

impl<P: Plugin> PluginAsVST3Impl for PluginMain<P> {
	fn num_midi_channels(&mut self, _: u32) -> u32 {
		16
	}

	fn supported_note_expressions(&mut self) -> clack_extensions::clap_wrapper::vst3::SupportedNoteExpressions {
		clack_extensions::clap_wrapper::vst3::SupportedNoteExpressions::empty()
	}
}

impl<P: Plugin> PluginFactoryAsVST3Impl for ClapPlugin<P> 
{
	fn get_vst3_info(&self, index: u32) -> Option<&PluginInfoAsVST3<'_>> {
		if index == 0 {
			Some(&Self::VST3_PLUGIN_INFO)
		}else {
			None
		}
	}
}

impl<P: PluginAuExt> PluginFactoryAsAUv2Impl for ClapPlugin<P> {
	fn get_auv2_info(&self, index: u32) -> Option<clack_extensions::clap_wrapper::auv2::PluginInfoAsAUv2> {
		if index == 0 {
			// A four-character code is not guaranteed to be valid UTF-8, so this
			// has to be a checked conversion rather than `from_utf8_unchecked`.
			Some(clack_extensions::clap_wrapper::auv2::PluginInfoAsAUv2::new(
				str::from_utf8(&P::AU_TYPE).expect("`AU_TYPE` must be valid UTF-8"),
				str::from_utf8(&P::AU_SUBTYPE).expect("`AU_SUBTYPE` must be valid UTF-8")
			))
		}else {
			None
		}
	}
}

/// Re-exported so that [`export_clap`] can add the VST3 entry points to the exported
/// library without the plug-in crate having to depend on `clap-wrapper` itself.
#[cfg(feature = "vst3")]
#[doc(hidden)]
pub use clap_wrapper::export_vst3 as __export_vst3;
