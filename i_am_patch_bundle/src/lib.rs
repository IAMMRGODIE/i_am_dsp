//! The CLAP and VST3 entry points of the patcher products.

pub use i_am_patch_product::*;

use i_am_plugin::export_plugin;

export_plugin!(PatcherEffect, PatcherGenerator);
