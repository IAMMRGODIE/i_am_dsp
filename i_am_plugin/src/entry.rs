//! entry

/// Emitted by [`export_clap`] to add the entry points of every wrapper format this
/// crate was built with. With no wrapper format enabled it expands to nothing.
#[cfg(feature = "vst3")]
#[doc(hidden)]
#[macro_export]
macro_rules! __export_wrapped_entry_points {
	() => {
		$crate::__export_vst3!();
	};
}

/// No wrapper format is enabled, so there is nothing to add.
#[cfg(not(feature = "vst3"))]
#[doc(hidden)]
#[macro_export]
macro_rules! __export_wrapped_entry_points {
	() => {};
}

/// Exports a plugin for the CLAP audio plugin host.
/// 
/// With the `vst3` feature of this crate enabled, the exported library also carries
/// the VST3 entry points, ready to be packed into a `.vst3` bundle by the
/// `i_am_bundler` tool (`cargo bundle`).
#[macro_export] macro_rules! export_clap {
	($plugin_ty: ty) => {
		i_am_plugin::clack_export_entry!(i_am_plugin::SinglePluginEntry<i_am_plugin::ClapPlugin<$plugin_ty>>);
		$crate::__export_wrapped_entry_points!();
	}
}
