//! Detection of the VST3 entry points a host looks up in the plug-in binary.
//!
//! A library only becomes a VST3 plug-in once something (in this workspace,
//! `clap-wrapper` behind the `vst3` feature) exports `GetPluginFactory`.
//! Bundling a library that does not is a silent failure in every DAW, so the
//! bundler checks for it and refuses unless `--force` is given.

use std::fs;
use std::path::Path;

/// What the check managed to establish.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Presence {
	/// The entry point is definitely there.
	Present,
	/// The entry point is definitely absent.
	Missing,
	/// The binary format could not be inspected; assume it might be fine.
	Unknown,
}

/// The result of inspecting a library.
#[derive(Debug, Clone)]
pub struct EntryPointCheck {
	pub presence: Presence,
	/// How the conclusion was reached, for the summary output.
	pub detail: String,
	/// The exported names, when they could be read (PE only).
	pub exported: Vec<String>,
}

/// Checks whether `library` exports the VST3 entry points.
pub fn check_vst3(library: &Path) -> EntryPointCheck {
	let bytes = match fs::read(library) {
		Ok(bytes) => bytes,
		Err(e) => {
			return EntryPointCheck {
				presence: Presence::Unknown,
				detail: format!("cannot read the library: {e}"),
				exported: Vec::new(),
			};
		},
	};

	if bytes.starts_with(b"MZ") {
		match pe_export_names(&bytes) {
			Some(names) => {
				let present = names.iter().any(|name| name == "GetPluginFactory");
				EntryPointCheck {
					presence: if present { Presence::Present } else { Presence::Missing },
					detail: format!(
						"PE export table read ({} symbol{} exported)",
						names.len(),
						if names.len() == 1 { "" } else { "s" }
					),
					exported: names,
				}
			},
			None => EntryPointCheck {
				presence: Presence::Unknown,
				detail: "the file starts with `MZ` but its export table could not be read".into(),
				exported: Vec::new(),
			},
		}
	} else if contains(&bytes, b"GetPluginFactory") {
		EntryPointCheck {
			presence: Presence::Present,
			detail: "found the symbol name in the dynamic string table".into(),
			exported: Vec::new(),
		}
	} else {
		EntryPointCheck {
			presence: Presence::Unknown,
			detail: "cannot inspect this platform's binary format; not checking".into(),
			exported: Vec::new(),
		}
	}
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
	haystack.windows(needle.len()).any(|window| window == needle)
}

fn read_u16(bytes: &[u8], offset: usize) -> Option<u16> {
	let slice = bytes.get(offset..offset.checked_add(2)?)?;
	Some(u16::from_le_bytes(slice.try_into().ok()?))
}

fn read_u32(bytes: &[u8], offset: usize) -> Option<u32> {
	let slice = bytes.get(offset..offset.checked_add(4)?)?;
	Some(u32::from_le_bytes(slice.try_into().ok()?))
}

fn read_c_string(bytes: &[u8], offset: usize) -> Option<String> {
	let rest = bytes.get(offset..)?;
	let end = rest.iter().take(4096).position(|b| *b == 0)?;
	Some(String::from_utf8_lossy(&rest[..end]).into_owned())
}

/// Reads the names in a PE image's export directory. Returns `None` when the file
/// is not a PE image that can be walked safely.
fn pe_export_names(bytes: &[u8]) -> Option<Vec<String>> {
	let pe_offset = read_u32(bytes, 0x3c)? as usize;
	if bytes.get(pe_offset..pe_offset + 4)? != b"PE\0\0" {
		return None;
	}

	let coff = pe_offset + 4;
	let section_count = read_u16(bytes, coff + 2)? as usize;
	let optional_header_size = read_u16(bytes, coff + 16)? as usize;
	let optional_header = coff + 20;

	let directory_offset = match read_u16(bytes, optional_header)? {
		0x10b => optional_header + 96,  // PE32
		0x20b => optional_header + 112, // PE32+
		_ => return None,
	};

	let export_rva = read_u32(bytes, directory_offset)? as usize;
	if export_rva == 0 {
		return Some(Vec::new());
	}

	let sections_offset = optional_header + optional_header_size;
	let mut sections = Vec::with_capacity(section_count);
	for index in 0..section_count {
		let section = sections_offset + index * 40;
		let virtual_size = read_u32(bytes, section + 8)? as usize;
		let virtual_address = read_u32(bytes, section + 12)? as usize;
		let raw_size = read_u32(bytes, section + 16)? as usize;
		let raw_pointer = read_u32(bytes, section + 20)? as usize;
		sections.push((virtual_address, virtual_size.max(raw_size), raw_pointer));
	}

	let to_offset = |rva: usize| -> Option<usize> {
		sections
			.iter()
			.find(|(address, size, _)| rva >= *address && rva < address + size)
			.map(|(address, _, raw)| raw + (rva - address))
	};

	let export_dir = to_offset(export_rva)?;
	let name_count = read_u32(bytes, export_dir + 24)? as usize;
	let names_rva = read_u32(bytes, export_dir + 32)? as usize;
	let names_offset = to_offset(names_rva)?;

	let mut names = Vec::with_capacity(name_count.min(4096));
	for index in 0..name_count {
		let name_rva = read_u32(bytes, names_offset + index * 4)? as usize;
		let name_offset = to_offset(name_rva)?;
		names.push(read_c_string(bytes, name_offset)?);
	}
	Some(names)
}
