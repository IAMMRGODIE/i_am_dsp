//! helper library for `i_am_dsp_iced` crate to export CLAP plugin based on `clack`
//! 
//! # Minimal Example
//! 
//! ```ignore
//! // first, build a `Processor` based on `i_am_dsp_iced`
//! struct MyProcessor;
//! 
//! impl Processor for MyProcessor {
//!     // see more in `i_am_dsp_iced` crate
//!     //...
//! }
//! 
//! // The imply `Plugin` trait for the `Processor`
//! impl Plugin for MyProcessor {
//!     const DESCRIPTOR: Descriptor = Descriptor::new("i.am.example.processor", "My Processor");
//!     
//!     fn new() -> Self {
//!         MyProcessor
//!     }
//!     
//!     fn window_options() -> WindowOptions {
//!         WindowOptions::default()
//!     }
//! 
//!     fn param_map(&self) -> ParamMap {
//!         ParamMap::default()
//!     }
//! }
//! 
//! // Finally, export the plugin as a CLAP plugin
//! export_clap!(MyProcessor);
//! 
//! // Additionaly, you can impl `PluginAuExt` to prepare the plugin for AU support
//! //
//! // You should check AU's manual for meaning and usage of `AU_TYPE` and `AU_SUBTYPE` fields.
//! impl PluginAuExt for MyProcessor {
//!     const AU_TYPE: [u8; 4] = *b"aufx";
//!     const AU_SUBTYPE: [u8; 4] = *b"mypr";
//! }
//! 
//! # Exporting a VST3 plug-in
//! 
//! [`export_clap`] always exports the CLAP entry point (`clap_entry`). It also exports the
//! VST3 entry points (`GetPluginFactory`, `InitDll`, `ExitDll`) when the `vst3` feature of
//! this crate is enabled, which links `clap-wrapper` and the VST3 SDK into your library.
//! Turn it on from your plug-in crate:
//! 
//! ```toml
//! [features]
//! vst3 = ["i_am_plugin/vst3"]
//! 
//! [package.metadata.i_am_dsp]
//! name = "My Processor"               # bundle name, defaults to the package name
//! vendor = "My Company"               # optional
//! id = "com.mycompany.myprocessor"    # optional, used by the macOS Info.plist
//! ```
//! 
//! ```sh
//! cargo build --release --features vst3
//! cargo bundle -p my_plugin --features vst3
//! ```
//! 
//! `cargo bundle` is the `i_am_bundler` tool of this workspace: it writes
//! `My Processor.vst3` and `My Processor.clap` next to the built library, and it refuses
//! to write a VST3 bundle for a library that has no VST3 entry points.
//! 
//! Do **NOT** forget to add following settings to your `Cargo.toml` file:
//! 
//! ```toml
//! [lib]
//! crate-type = ["cdylib"]
//! ```
//! 
//! Without the `vst3` feature you can still build with `cargo build --release`, rename the
//! suffix to `.clap` and load the plugin in a CLAP host.


/// Re-exported so that [`export_clap`] can add the VST3 entry points to the exported
/// library without the plug-in crate having to depend on `clap-wrapper` itself.
#[cfg(feature = "vst3")]
#[doc(hidden)]
pub use clap_wrapper::export_vst3 as __export_vst3;

/// Re-exported for the [`export_clap`] macro to name the entry type by.
#[doc(hidden)]
pub use clack_plugin::{clack_export_entry, entry::SinglePluginEntry};

extern crate self as i_am_plugin;

mod audio;
mod clap_plugin;
mod context;
mod descriptor;
mod entry;
mod events;
mod latency;
mod main_thread;
mod plugin;
mod shared;

#[cfg(test)]
mod tests;

pub use audio::AudioProcessor;
pub use clap_plugin::ClapPlugin;
pub use context::ClapContext;
pub use descriptor::{AudioPort, Descriptor, MidiPort, Tag};
pub use main_thread::{GuiProgram, PluginMain};
pub use plugin::{Plugin, PluginAuExt, WindowOptions};
