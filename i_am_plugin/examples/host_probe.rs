//! What a host sees when it loads a library.
//!
//! This is the other end of [`export_clap`](i_am_plugin::export_clap): it loads a
//! built plug-in library through clack's host API and asks its factory what it has.
//! Nothing here knows what is inside the library -- which is the point, because that
//! is also true of the host that will load it in the real world.
//!
//! Run it against a built library:
//!
//! ```text
//! cargo run -p i_am_plugin --example host_probe -- path/to/library.clap
//! ```
//!
//! There is a library to point it at in `tests/set_library`: a fixture crate that
//! exports two plug-ins from one binary, with its own manifest so that it can be a
//! cdylib without being a member of this workspace. From the workspace root:
//!
//! ```text
//! cargo build --manifest-path i_am_plugin/tests/set_library/Cargo.toml
//! cargo run -p i_am_plugin --example host_probe -- \
//!     i_am_plugin/tests/set_library/target/debug/set_library.dll
//! ```
//!
//! ...and it prints both of them, which is the whole of what "a bundle exports more
//! than one plug-in" means to a host.

use clack_host::prelude::PluginEntry;

fn main() -> Result<(), Box<dyn std::error::Error>> {
	let path = std::env::args().nth(1).expect("a path to a plug-in library");

	// SAFETY: loading a library runs its code. That is what a host does when it
	// scans a plug-in folder, and it is why a plug-in is trusted software.
	let entry = unsafe { PluginEntry::load(path) }?;
	let factory = entry.get_plugin_factory().expect("a CLAP factory");

	println!("the library exports {} plug-in(s)", factory.plugin_count());

	for index in 0..factory.plugin_count() {
		let descriptor = factory.plugin_descriptor(index).expect("a descriptor");

		println!(
			"  {index}: {} ({})",
			descriptor.name().unwrap_or_default().to_string_lossy(),
			descriptor.id().unwrap_or_default().to_string_lossy(),
		);
	}

	Ok(())
}
