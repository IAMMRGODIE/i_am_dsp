//! The on-disk layouts audio hosts load, plus installing them.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

/// The platform whose bundle layout we produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
	Windows,
	MacOs,
	Linux,
}

impl Format {
	/// The platform this binary was compiled for.
	pub fn host() -> Self {
		if cfg!(windows) {
			Self::Windows
		} else if cfg!(target_os = "macos") {
			Self::MacOs
		} else {
			Self::Linux
		}
	}
}

/// The architecture sub-folder to use inside a bundle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
	X86_64,
	X86,
	Aarch64,
	Arm,
}

impl Arch {
	/// The architecture this binary was compiled for.
	pub fn host() -> Self {
		match std::env::consts::ARCH {
			"x86_64" => Self::X86_64,
			"x86" | "i686" => Self::X86,
			"aarch64" => Self::Aarch64,
			_ => Self::Arm,
		}
	}

	/// The name VST3 gives this architecture's sub-folder, e.g. `x86_64-win`.
	fn vst3_dir(self, format: Format) -> &'static str {
		match (format, self) {
			(Format::Windows, Self::X86_64) => "x86_64-win",
			(Format::Windows, Self::X86) => "x86-win",
			(Format::Windows, _) => "arm64-win",
			(Format::Linux, Self::X86_64) => "x86_64-linux",
			(Format::Linux, Self::X86) => "i386-linux",
			(Format::Linux, Self::Aarch64) => "aarch64-linux",
			(Format::Linux, Self::Arm) => "armv7l-linux",
			(Format::MacOs, _) => "MacOS",
		}
	}
}

/// Which plug-in format a produced bundle is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
	Vst3,
	Clap,
}

impl Kind {
	pub fn label(self) -> &'static str {
		match self {
			Self::Vst3 => "vst3",
			Self::Clap => "clap",
		}
	}

	fn folder(self) -> &'static str {
		match self {
			Self::Vst3 => "VST3",
			Self::Clap => "CLAP",
		}
	}
}

/// The metadata written into a bundle.
pub struct PluginMeta<'a> {
	pub name: &'a str,
	pub version: Option<&'a str>,
	pub id: &'a str,
}

/// Writes the VST3 bundle (or single file) for `library` and returns its path.
pub fn write_vst3(
	library: &Path,
	out_dir: &Path,
	meta: &PluginMeta<'_>,
	format: Format,
	arch: Arch,
	single_file: bool,
) -> Result<PathBuf, String> {
	fs::create_dir_all(out_dir).map_err(|e| format!("cannot create `{}`: {e}", out_dir.display()))?;

	if single_file {
		if format != Format::Windows {
			return Err("--single-file is only valid for Windows VST3 plug-ins".into());
		}
		let destination = out_dir.join(format!("{}.vst3", meta.name));
		copy_file(library, &destination)?;
		return Ok(destination);
	}

	let root = out_dir.join(format!("{}.vst3", meta.name));
	remove_bundle(&root)?;

	let destination = match format {
		Format::MacOs => root.join("Contents").join("MacOS").join(meta.name),
		Format::Windows => root
			.join("Contents")
			.join(arch.vst3_dir(format))
			.join(format!("{}.vst3", meta.name)),
		Format::Linux => root
			.join("Contents")
			.join(arch.vst3_dir(format))
			.join(format!("{}.so", meta.name)),
	};
	copy_file(library, &destination)?;

	if format == Format::MacOs {
		write_plist(&root.join("Contents").join("Info.plist"), meta)?;
	}

	Ok(root)
}

/// Writes the CLAP plug-in and returns its path.
pub fn write_clap(
	library: &Path,
	out_dir: &Path,
	meta: &PluginMeta<'_>,
	format: Format,
) -> Result<PathBuf, String> {
	fs::create_dir_all(out_dir).map_err(|e| format!("cannot create `{}`: {e}", out_dir.display()))?;

	if format != Format::MacOs {
		let destination = out_dir.join(format!("{}.clap", meta.name));
		copy_file(library, &destination)?;
		return Ok(destination);
	}

	let root = out_dir.join(format!("{}.clap", meta.name));
	remove_bundle(&root)?;
	copy_file(library, &root.join("Contents").join("MacOS").join(meta.name))?;
	write_plist(&root.join("Contents").join("Info.plist"), meta)?;
	Ok(root)
}

/// The directory hosts scan for plug-ins of `kind`, if this platform defines one.
pub fn plugin_dir(format: Format, kind: Kind, system: bool) -> Option<PathBuf> {
	match format {
		Format::Windows => {
			if system {
				std::env::var_os("CommonProgramFiles")
					.map(|root| PathBuf::from(root).join(kind.folder()))
			} else {
				std::env::var_os("LOCALAPPDATA").map(|root| {
					PathBuf::from(root)
						.join("Programs")
						.join("Common")
						.join(kind.folder())
				})
			}
		},
		Format::MacOs => {
			if system {
				Some(PathBuf::from("/Library/Audio/Plug-Ins").join(kind.folder()))
			} else {
				home().map(|home| {
					home.join("Library")
						.join("Audio")
						.join("Plug-Ins")
						.join(kind.folder())
				})
			}
		},
		Format::Linux => {
			if system {
				Some(PathBuf::from("/usr/lib").join(kind.folder().to_ascii_lowercase()))
			} else {
				home().map(|home| home.join(format!(".{}", kind.folder().to_ascii_lowercase())))
			}
		},
	}
}

fn home() -> Option<PathBuf> {
	std::env::var_os("HOME")
		.or_else(|| std::env::var_os("USERPROFILE"))
		.filter(|value| !value.is_empty())
		.map(PathBuf::from)
}

/// Copies a bundle (file or folder) into `destination_root`, replacing anything
/// with the same name.
pub fn install(source: &Path, destination_root: &Path) -> Result<PathBuf, String> {
	let file_name = source
		.file_name()
		.ok_or_else(|| format!("`{}` has no file name", source.display()))?;

	fs::create_dir_all(destination_root)
		.map_err(|e| format!("cannot create `{}`: {e}", destination_root.display()))?;

	let destination = destination_root.join(file_name);
	if destination.is_dir() {
		fs::remove_dir_all(&destination)
			.map_err(|e| format!("cannot remove `{}`: {e}", destination.display()))?;
	} else if destination.is_file() {
		fs::remove_file(&destination)
			.map_err(|e| format!("cannot remove `{}`: {e}", destination.display()))?;
	}

	if source.is_dir() {
		copy_dir(source, &destination)?;
	} else {
		copy_file(source, &destination)?;
	}

	Ok(destination)
}

fn copy_file(source: &Path, destination: &Path) -> Result<(), String> {
	if let Some(parent) = destination.parent() {
		fs::create_dir_all(parent)
			.map_err(|e| format!("cannot create `{}`: {e}", parent.display()))?;
	}
	fs::copy(source, destination).map_err(|e| {
		format!(
			"cannot copy `{}` to `{}`: {e}",
			source.display(),
			destination.display()
		)
	})?;
	Ok(())
}

fn copy_dir(source: &Path, destination: &Path) -> Result<(), String> {
	fs::create_dir_all(destination)
		.map_err(|e| format!("cannot create `{}`: {e}", destination.display()))?;

	let entries = fs::read_dir(source).map_err(|e| format!("cannot read `{}`: {e}", source.display()))?;
	for entry in entries {
		let entry = entry.map_err(|e| format!("cannot read `{}`: {e}", source.display()))?;
		let target = destination.join(entry.file_name());
		let file_type = entry
			.file_type()
			.map_err(|e| format!("cannot inspect `{}`: {e}", entry.path().display()))?;
		if file_type.is_dir() {
			copy_dir(&entry.path(), &target)?;
		} else {
			copy_file(&entry.path(), &target)?;
		}
	}
	Ok(())
}

/// Removes an existing bundle folder so stale files cannot linger. Only paths that
/// actually look like a bundle are ever removed.
fn remove_bundle(path: &Path) -> Result<(), String> {
	let looks_like_a_bundle = path
		.extension()
		.and_then(|extension| extension.to_str())
		.is_some_and(|extension| extension == "vst3" || extension == "clap");

	if !looks_like_a_bundle {
		return Err(format!("refusing to remove `{}`", path.display()));
	}

	if path.is_dir() {
		fs::remove_dir_all(path).map_err(|e| format!("cannot remove `{}`: {e}", path.display()))?;
	}
	Ok(())
}

fn write_plist(path: &Path, meta: &PluginMeta<'_>) -> Result<(), String> {
	let version = meta.version.unwrap_or("1.0");
	let name = xml_escape(meta.name);
	let id = xml_escape(meta.id);

	let xml = format!(
		r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>CFBundleDevelopmentRegion</key>
	<string>English</string>
	<key>CFBundleExecutable</key>
	<string>{name}</string>
	<key>CFBundleIdentifier</key>
	<string>{id}</string>
	<key>CFBundleInfoDictionaryVersion</key>
	<string>6.0</string>
	<key>CFBundleName</key>
	<string>{name}</string>
	<key>CFBundleDisplayName</key>
	<string>{name}</string>
	<key>CFBundlePackageType</key>
	<string>BNDL</string>
	<key>CFBundleSignature</key>
	<string>????</string>
	<key>CFBundleShortVersionString</key>
	<string>{version}</string>
	<key>CFBundleVersion</key>
	<string>{version}</string>
	<key>NSHighResolutionCapable</key>
	<true/>
</dict>
</plist>
"#
	);

	if let Some(parent) = path.parent() {
		fs::create_dir_all(parent)
			.map_err(|e| format!("cannot create `{}`: {e}", parent.display()))?;
	}
	let mut file =
		fs::File::create(path).map_err(|e| format!("cannot create `{}`: {e}", path.display()))?;
	file.write_all(xml.as_bytes())
		.map_err(|e| format!("cannot write `{}`: {e}", path.display()))?;
	Ok(())
}

fn xml_escape(value: &str) -> String {
	value
		.replace('&', "&amp;")
		.replace('<', "&lt;")
		.replace('>', "&gt;")
		.replace('"', "&quot;")
}
