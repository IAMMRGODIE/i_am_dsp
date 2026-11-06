//! entry

/// Emitted by [`export_plugin`] to add the entry points of every wrapper format this
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

/// Exports the plug-ins of this library for the CLAP audio plugin host.
///
/// One library may export as many as it likes, and the host sees each of them as a
/// plug-in of its own while the bundle they ship in is one binary:
///
/// ```ignore
/// export_plugin!(MyProcessor);
/// export_plugin!(MyProcessor, MyOtherProcessor);
/// ```
///
/// With the `vst3` feature of this crate enabled, the exported library also carries
/// the VST3 entry points, ready to be packed into a `.vst3` bundle by the
/// `i_am_bundler` tool (`cargo bundle`).
#[macro_export]
macro_rules! export_plugin {
	($($plugin_ty: ty),+ $(,)?) => {
		/// The plug-ins this library exports.
		#[doc(hidden)]
		pub struct PluginSet;

		impl $crate::__private::PluginSetSource for PluginSet {
			fn entries() -> ::std::vec::Vec<$crate::__private::SetEntry> {
				::std::vec![$($crate::__private::SetEntry::of::<$plugin_ty>()),+]
			}
		}

		$crate::clack_export_entry!($crate::__private::SetEntryPoint<PluginSet>);
		$crate::__export_wrapped_entry_points!();
	};
}

/// The name [`export_plugin`] had when it only spoke CLAP.
///
/// It is kept so a crate written against the old name still builds; new code
/// should call [`export_plugin`], which also exports the VST3 entry points when
/// the `vst3` feature is on.
#[deprecated(note = "use export_plugin")]
#[doc(hidden)]
#[macro_export]
macro_rules! export_clap {
	($($plugin_ty: ty),+ $(,)?) => {
		$crate::export_plugin!($($plugin_ty),+);
	};
}
