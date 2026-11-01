//! A plug-in wrapper, with the GUI kept at arm's length.
//! 
//! This crate turns a processor from `i_am_dsp` into a CLAP plug-in (and, with the
//! `vst3` feature, a VST3 one) using `clack`. It knows what a host asks a plug-in:
//! descriptors, ports, parameters, state, latency, and how to run the audio thread.
//! It does not know what draws the editor.
//!
//! A GUI is connected through the [Editor] trait. A plug-in says which editor it
//! uses in [Plugin::editor], and the editor is handed the host's window as a raw
//! handle plus a channel for its messages; everything else is the backend's business.
//! This crate's tests use the default: no editor at all. The WebView2 backend is the
//! [wry] module; another toolkit would be another module behind the same seam.
//!
//! # Minimal Example
//! 
//! ```ignore
//! use i_am_plugin::{Descriptor, Plugin, WindowOptions, export_clap};
//! 
//! // first, build a processor from `i_am_dsp`
//! struct MyProcessor;
//! 
//! impl i_am_dsp::Processor for MyProcessor {
//!     type Message = ();
//!     // see more in `i_am_dsp::Processor`
//!     //...
//! }
//! 
//! impl Plugin for MyProcessor {
//!     const DESCRIPTOR: Descriptor = Descriptor::new("i.am.example.processor", "My Processor");
//!     const HAS_GUI: bool = true;
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
//!
//!     // the only line that names the toolkit
//!     fn editor(&self) -> Option<Box<dyn i_am_plugin::Editor<Self>>> {
//!         Some(Box::new(i_am_plugin::wry::WryEditor::new(
//!             "<html>...</html>",
//!             |body| Some(MyMessage::from_js(body)),
//!         )))
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
//! ```
//! 
//! A plug-in with no GUI simply leaves [Plugin::editor] and `HAS_GUI` alone:
//!
//! ```ignore
//! impl Plugin for MyHeadlessProcessor {
//!     const DESCRIPTOR: Descriptor = Descriptor::new("i.am.example.headless", "My Headless");
//!
//!     fn new() -> Self { MyHeadlessProcessor }
//!     fn param_map(&self) -> ParamMap { self.param_map.clone() }
//! }
//! ```
//! 
//! # Writing another backend
//!
//! An editor is a type that implements [Editor] for one plug-in type:
//!
//! ```ignore
//! struct MyEditor { /* the toolkit's window */ }
//!
//! impl<P: Plugin> Editor<P> for MyEditor {
//!     fn open(&mut self, context: EditorContext<P>) -> Result<(), EditorError> {
//!         // build the window inside context.parent, size it with context.size and
//!         // context.scale_factor, and keep context.messages for what the user does
//!         Ok(())
//!     }
//!
//!     fn show(&mut self) -> Result<(), EditorError> { /* ... */ }
//!     fn hide(&mut self) -> Result<(), EditorError> { /* ... */ }
//!     fn close(&mut self) { /* ... */ }
//! }
//! ```
//!
//! The backend moves the view to its own thread if it has one, sends
//! [Message](i_am_dsp::Message) values over the channel, and is otherwise free: the
//! wrapper never calls into it except through the trait.
//! 
//! # Exporting a VST3 plug-in
//! 
//! [export_clap] always exports the CLAP entry point (`clap_entry`). It also exports the
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
mod editor;
mod entry;
mod events;
mod factory;
mod latency;
mod log;
mod main_thread;
mod plugin;
mod shared;

/// The GUI-neutral processor plumbing: an effect and the parameter map a host
/// and the interface both read.
pub mod processor;

/// The WebView2 editor backend, formerly the separate i_am_plugin_wry crate.
pub mod wry;

#[cfg(test)]
mod tests;

pub use audio::AudioProcessor;
pub use factory::{PluginSetSource, SetEntry, SetEntryPoint};
pub use clap_plugin::ClapPlugin;
pub use context::ClapContext;
pub use descriptor::{AudioPort, Descriptor, MidiPort, Tag};
pub use editor::{Editor, EditorContext, EditorError, ParentWindow};
pub use main_thread::PluginMain;
pub use plugin::{Plugin, PluginAuExt, WindowOptions, WindowSize};

/// The resize hints a [WindowOptions] carries, re-exported so a plug-in can describe
/// a resizable window without also depending on `clack-extensions`.
pub use clack_extensions::gui::{AspectRatioStrategy, GuiResizeHints};

/// The processor and message traits of `i_am_dsp`, re-exported so a plug-in crate
/// can name them without also depending on that crate.
pub use i_am_dsp::{Message, Processor};

/// The clap-wrapper static library references the clap_entry data symbol even in
/// test binaries, which do not export a plug-in. The real entry point is only
/// produced by the product cdylib; this zeroed local symbol lets the test build
/// link without pretending to be a VST3 plug-in.
#[cfg(all(test, feature = "vst3"))]
#[unsafe(no_mangle)]
pub static clap_entry: [u8; 64] = [0; 64];

/// The names the export macros need to be able to reach.
///
/// A macro expands in the crate that calls it, so everything it mentions has to be
/// nameable from there; this is that list, and nothing here is meant to be used by
/// hand.
#[doc(hidden)]
pub mod __private {
	pub use crate::factory::{PluginSetSource, SetEntry, SetEntryPoint};
}
