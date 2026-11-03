//! The CLAP and VST3 entry points of the synthesizer products.
//!
//! Everything a product *is* lives in `i_am_synths`: the DSP, the web faceplate
//! and the `Plugin` implementation. This crate adds only what makes the bundle
//! loadable — `clap_entry` and, with the `vst3` feature, the VST3 entry points.
//!
//! It is a crate of its own because `clap_entry` is a global symbol: a library
//! that exported one could not also be linked into another plug-in.
//!
//! ```text
//! cargo build -p i_am_synths_bundle --features vst3
//! cargo bundle -p i_am_synths_bundle --features vst3
//! ```

pub use i_am_synths::*;

use i_am_plugin::export_clap;

export_clap!(StringWeb, ShapesWeb, AdditiveWeb, SpectrumWeb);
