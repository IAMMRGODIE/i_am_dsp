//! Exporting more than one plug-in from one library.
//!
//! A CLAP host asks a library for its factory, and a factory may list as many
//! plug-ins as it likes. That is what lets a bundle ship as one binary while every
//! snap-in still appears in the host as a plug-in of its own, and it is why this
//! exists at all: the convenient one-plug-in entry type covers the common case, and
//! this covers the rest.
//!
//! The work is all here rather than in the macro, so that the macro is only a list
//! of types and the behaviour can be read in one place.

use std::ffi::CStr;
use std::marker::PhantomData;
use std::sync::OnceLock;

use clack_extensions::clap_wrapper::vst3::{PluginFactoryAsVST3Impl, PluginInfoAsVST3};
use clack_plugin::entry::{DefaultPluginFactory, Entry, EntryFactories, EntryLoadError};
use clack_plugin::factory::plugin::{PluginFactoryImpl, PluginFactoryWrapper};
use clack_plugin::host::HostInfo;
use clack_plugin::plugin::{PluginDescriptor, PluginInstance};

use crate::clap_plugin::ClapPlugin;
use crate::plugin::Plugin;

/// One plug-in of a set: what it is called, how to make it, and what the wrapped
/// formats call it.
pub struct SetEntry {
	descriptor: PluginDescriptor,
	create: for<'a> fn(HostInfo<'a>, &'a PluginDescriptor) -> Option<PluginInstance<'a>>,
	vst3: PluginInfoAsVST3<'static>,
}

impl SetEntry {
	/// Describes a plug-in type.
	///
	/// Called once per type when the factory is first asked for anything.
	pub fn of<P>() -> Self
	where
		P: Plugin,
		<P as i_am_dsp_iced::Processor>::Message: std::fmt::Debug,
	{
		Self {
			descriptor: ClapPlugin::<P>::get_descriptor(),
			create: |host_info, descriptor| {
				Some(PluginInstance::new::<ClapPlugin<P>>(
					host_info,
					descriptor,
					ClapPlugin::<P>::new_shared,
					ClapPlugin::<P>::new_main_thread,
				))
			},
			vst3: ClapPlugin::<P>::VST3_PLUGIN_INFO,
		}
	}

	/// What the host is told this plug-in is.
	pub fn descriptor(&self) -> &PluginDescriptor {
		&self.descriptor
	}

	/// What a VST3 host is told it is.
	pub fn vst3_info(&self) -> &PluginInfoAsVST3<'static> {
		&self.vst3
	}
}

/// The plug-ins a library exports, and how to build them.
///
/// Implemented by the [`export_clap`](crate::export_clap) macro. A crate that has
/// to assemble its list some other way -- from a registry, or from a feature flag --
/// implements this instead and points the macro's entry type at it.
pub trait PluginSetSource: Send + Sync + 'static {
	/// The plug-ins to export, in the order the host should see them.
	fn entries() -> Vec<SetEntry>;
}

/// The factory a set of plug-ins is registered behind.
pub struct SetFactory<S: PluginSetSource> {
	entries: OnceLock<Vec<SetEntry>>,
	_source: PhantomData<fn() -> S>,
}

impl<S: PluginSetSource> Default for SetFactory<S> {
	fn default() -> Self {
		Self {
			entries: OnceLock::new(),
			_source: PhantomData,
		}
	}
}

impl<S: PluginSetSource> SetFactory<S> {
	/// The list, built on the first question anyone asks about it.
	fn entries(&self) -> &[SetEntry] {
		self.entries.get_or_init(S::entries)
	}
}

impl<S: PluginSetSource> PluginFactoryImpl for SetFactory<S> {
	fn plugin_count(&self) -> u32 {
		self.entries().len() as u32
	}

	fn plugin_descriptor(&self, index: u32) -> Option<&PluginDescriptor> {
		self.entries().get(index as usize).map(SetEntry::descriptor)
	}

	fn create_plugin<'a>(
		&'a self,
		host_info: HostInfo<'a>,
		plugin_id: &CStr,
	) -> Option<PluginInstance<'a>> {
		// The host names the plug-in it wants by id, not by index: a library that
		// lists three of them is asked for whichever one the session saved.
		self.entries()
			.iter()
			.find(|entry| entry.descriptor.id().unwrap_or_default() == plugin_id)
			.and_then(|entry| (entry.create)(host_info, &entry.descriptor))
	}
}

impl<S: PluginSetSource> PluginFactoryAsVST3Impl for SetFactory<S> {
	fn get_vst3_info(&self, index: u32) -> Option<&PluginInfoAsVST3<'_>> {
		self.entries().get(index as usize).map(SetEntry::vst3_info)
	}
}

/// The library's way in: the entry a host calls to find the factory.
///
/// Nothing about the AU wrapper is reported for a set, because a set has no single
/// type to take the two four-character codes from. A crate that needs AU support
/// exports one plug-in per library and implements
/// [`PluginAuExt`](crate::PluginAuExt) for it.
pub struct SetEntryPoint<S: PluginSetSource> {
	factory: PluginFactoryWrapper<SetFactory<S>>,
}

impl<S: PluginSetSource> Entry for SetEntryPoint<S> {
	fn new(_plugin_path: Option<&CStr>) -> Result<Self, EntryLoadError> {
		Ok(Self {
			factory: PluginFactoryWrapper::new(SetFactory::<S>::default()),
		})
	}

	fn declare_factories<'a>(&'a self, builder: &mut EntryFactories<'a>) {
		builder.register_factory(&self.factory);
	}
}
