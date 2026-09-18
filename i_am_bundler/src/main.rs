//! `i_am_bundler` turns a built `i_am_plugin` library into the on-disk layouts audio
//! hosts load:
//!
//! * `<name>.vst3` - the VST3 bundle (folder form everywhere, or a single renamed
//!   library on Windows with `--single-file`).
//! * `<name>.clap` - the CLAP plug-in (a single file on Windows and Linux, a bundle
//!   on macOS).
//!
//! The plug-in name, vendor and id come from `[package.metadata.i_am_dsp]` in the
//! plug-in crate's manifest and can be overridden on the command line:
//!
//! ```toml
//! [package.metadata.i_am_dsp]
//! name = "My Synth"
//! vendor = "My Company"
//! id = "com.mycompany.mysynth"
//! ```
//!
//! Typical use, from the workspace root, through the `cargo bundle` alias:
//!
//! ```text
//! cargo bundle -p plugin_example --features vst3
//! ```

mod bundle;
mod exports;
mod manifest;
mod target;

use std::path::{Path, PathBuf};
use std::process::Command;

use bundle::{Arch, Format, Kind, PluginMeta};
use exports::Presence;
use manifest::CrateInfo;

const USAGE: &str = r#"i_am_bundler - bundle an i_am_plugin library as .vst3 / .clap

USAGE:
    i_am_bundler [OPTIONS] [LIBRARY]

ARGS:
    LIBRARY                 Built plug-in library. When omitted, the library is
                            looked up in cargo's target directory after building.

OPTIONS:
    -p, --package NAME      Cargo package to build and bundle
        --manifest-path P   Manifest of the plug-in crate (default: ./Cargo.toml)
        --release           Build the release profile (default)
        --debug             Build the debug profile
        --features LIST     Features handed to "cargo build" (for example "vst3")
        --target TRIPLE     Cross-compilation target
        --target-dir DIR    Cargo target directory
        --no-build          Do not run "cargo build"
        --name NAME         Bundle name (overrides package.metadata.i_am_dsp)
        --vendor VENDOR     Vendor (used for the default bundle id)
        --version VERSION   Version (written into the macOS Info.plist)
        --id ID             Bundle identifier (written into the macOS Info.plist)
        --out-dir DIR       Where bundles are written (default: next to the library)
        --format FORMAT     windows, macos or linux (default: this machine)
        --arch ARCH         x86_64, x86, aarch64 or arm (default: this machine)
        --single-file       Write NAME.vst3 as one renamed library (Windows only)
        --no-vst3           Skip the VST3 bundle
        --no-clap           Skip the .clap plug-in
        --install           Also install into the user plug-in directory
        --install-system    Also install into the system plug-in directory
        --force             Bundle even when GetPluginFactory is missing
    -h, --help              Print this help
"#;

fn main() {
	match Args::parse(std::env::args().skip(1).collect()) {
		Ok(Parse::Help) => print!("{}", USAGE),
		Ok(Parse::Run(args)) => {
			if let Err(error) = run(*args) {
				eprintln!("error: {error}");
				std::process::exit(1);
			}
		},
		Err(error) => {
			eprintln!("error: {error}");
			eprintln!();
			print!("{}", USAGE);
			std::process::exit(2);
		},
	}
}

enum Parse {
	Help,
	Run(Box<Args>),
}

struct Args {
	library: Option<PathBuf>,
	package: Option<String>,
	manifest_path: Option<PathBuf>,
	release: bool,
	features: Option<String>,
	target: Option<String>,
	target_dir: Option<PathBuf>,
	no_build: bool,
	name: Option<String>,
	vendor: Option<String>,
	version: Option<String>,
	id: Option<String>,
	out_dir: Option<PathBuf>,
	format: Option<Format>,
	arch: Option<Arch>,
	single_file: bool,
	no_vst3: bool,
	no_clap: bool,
	install: bool,
	install_system: bool,
	force: bool,
}

impl Default for Args {
	fn default() -> Self {
		Self {
			library: None,
			package: None,
			manifest_path: None,
			release: true,
			features: None,
			target: None,
			target_dir: None,
			no_build: false,
			name: None,
			vendor: None,
			version: None,
			id: None,
			out_dir: None,
			format: None,
			arch: None,
			single_file: false,
			no_vst3: false,
			no_clap: false,
			install: false,
			install_system: false,
			force: false,
		}
	}
}

impl Args {
	fn parse(argv: Vec<String>) -> Result<Parse, String> {
		let mut args = Args::default();
		let mut iter = argv.into_iter();

		while let Some(raw) = iter.next() {
			let (flag, inline) = match raw.split_once('=') {
				Some((flag, value)) if flag.starts_with("--") => {
					(flag.to_string(), Some(value.to_string()))
				},
				_ => (raw.clone(), None),
			};

			macro_rules! value {
				($name:expr) => {
					match inline.clone() {
						Some(value) => value,
						None => iter
							.next()
							.ok_or_else(|| format!("{} needs a value", $name))?,
					}
				};
			}

			match flag.as_str() {
				"-h" | "--help" => return Ok(Parse::Help),
				"-p" | "--package" => args.package = Some(value!("--package")),
				"--manifest-path" => {
					args.manifest_path = Some(PathBuf::from(value!("--manifest-path")));
				},
				"--release" => args.release = true,
				"--debug" => args.release = false,
				"--features" => args.features = Some(value!("--features")),
				"--target" => args.target = Some(value!("--target")),
				"--target-dir" => args.target_dir = Some(PathBuf::from(value!("--target-dir"))),
				"--no-build" => args.no_build = true,
				"--name" => args.name = Some(value!("--name")),
				"--vendor" => args.vendor = Some(value!("--vendor")),
				"--version" => args.version = Some(value!("--version")),
				"--id" => args.id = Some(value!("--id")),
				"--out-dir" => args.out_dir = Some(PathBuf::from(value!("--out-dir"))),
				"--format" => args.format = Some(parse_format(&value!("--format"))?),
				"--arch" => args.arch = Some(parse_arch(&value!("--arch"))?),
				"--single-file" => args.single_file = true,
				"--no-vst3" => args.no_vst3 = true,
				"--no-clap" => args.no_clap = true,
				"--install" => args.install = true,
				"--install-system" => args.install_system = true,
				"--force" => args.force = true,
				other if other.starts_with('-') => return Err(format!("unknown option `{other}`")),
				other => {
					if args.library.is_some() {
						return Err(format!("unexpected extra argument `{other}`"));
					}
					args.library = Some(PathBuf::from(other));
				},
			}
		}

		Ok(Parse::Run(Box::new(args)))
	}
}

fn parse_format(value: &str) -> Result<Format, String> {
	match value.to_ascii_lowercase().as_str() {
		"windows" | "win" => Ok(Format::Windows),
		"macos" | "mac" | "osx" => Ok(Format::MacOs),
		"linux" | "unix" => Ok(Format::Linux),
		other => Err(format!(
			"unknown format `{other}` (expected windows, macos or linux)"
		)),
	}
}

fn parse_arch(value: &str) -> Result<Arch, String> {
	match value.to_ascii_lowercase().as_str() {
		"x86_64" | "x64" | "amd64" => Ok(Arch::X86_64),
		"x86" | "i686" | "win32" => Ok(Arch::X86),
		"aarch64" | "arm64" => Ok(Arch::Aarch64),
		"arm" | "armv7" => Ok(Arch::Arm),
		other => Err(format!(
			"unknown architecture `{other}` (expected x86_64, x86, aarch64 or arm)"
		)),
	}
}

fn run(args: Args) -> Result<(), String> {
	let cwd = std::env::current_dir().map_err(|e| format!("cannot read the current directory: {e}"))?;
	let cwd_manifest = cwd.join("Cargo.toml");

	// Which crate are we dealing with? This is best effort when the caller passes
	// the library explicitly.
	let info = if let Some(path) = &args.manifest_path {
		Some(CrateInfo::load(path)?)
	} else if let Some(package) = &args.package {
		Some(CrateInfo::load(&manifest::resolve_package_manifest(
			&cwd_manifest,
			package,
		)?)?)
	} else {
		CrateInfo::load(&cwd_manifest).ok().filter(|info| info.is_package)
	};
	let info = info.as_ref();

	if !args.no_build {
		let Some(info) = info else {
			return Err(
				"cannot tell which crate to build; pass -p PACKAGE or --manifest-path PATH".into(),
			);
		};
		build(&args, info)?;
	}

	let library = match &args.library {
		Some(path) => {
			if !path.is_file() {
				return Err(format!("`{}` is not a file", path.display()));
			}
			path.clone()
		},
		None => {
			let Some(info) = info else {
				return Err("pass the built library as an argument, or -p PACKAGE".into());
			};
			if !info.is_cdylib() {
				eprintln!(
					"warning: `{}` does not declare a `cdylib` target, so building it will not produce a loadable plug-in",
					info.manifest_path.display()
				);
			}
			lookup_library(&args, info)?
		},
	};

	let format = args.format.unwrap_or_else(Format::host);
	let arch = args.arch.unwrap_or_else(Arch::host);

	let name = args
		.name
		.clone()
		.or_else(|| info.and_then(|info| info.plugin_name.clone()))
		.or_else(|| info.and_then(|info| info.package_name.clone()))
		.or_else(|| library.file_stem().map(|stem| stem.to_string_lossy().into_owned()))
		.filter(|name| !name.trim().is_empty())
		.ok_or_else(|| "cannot determine the plug-in name; pass --name NAME".to_string())?;
	validate_name(&name)?;

	let vendor = args
		.vendor
		.clone()
		.or_else(|| info.and_then(|info| info.vendor.clone()));
	let version = args
		.version
		.clone()
		.or_else(|| info.and_then(|info| info.version.clone()))
		.or_else(|| info.and_then(|info| info.package_version.clone()));
	let id = args
		.id
		.clone()
		.or_else(|| info.and_then(|info| info.id.clone()))
		.unwrap_or_else(|| {
			format!(
				"com.{}.{}",
				slug(vendor.as_deref().unwrap_or("iamdsp")),
				slug(&name)
			)
		});

	let out_dir = args.out_dir.clone().unwrap_or_else(|| {
		library
			.parent()
			.map(Path::to_path_buf)
			.unwrap_or_else(|| PathBuf::from("."))
	});

	let meta = PluginMeta {
		name: &name,
		version: version.as_deref(),
		id: &id,
	};

	field("plugin", &name);
	field("library", library.display());
	field("platform", format!("{} / {}", format_label(format), arch_label(arch)));

	let mut written: Vec<(Kind, PathBuf)> = Vec::new();

	if !args.no_vst3 {
		let check = exports::check_vst3(&library);
		match check.presence {
			Presence::Present => {
				let entry_points: Vec<&str> = check
					.exported
					.iter()
					.map(String::as_str)
					.filter(|name| {
						name.starts_with("Get") || name.starts_with("Init") || name.starts_with("Exit")
					})
					.collect();
				if entry_points.is_empty() {
					field("vst3", &check.detail);
				} else {
					field("vst3", format!("exported {}", entry_points.join(", ")));
				}
			},
			Presence::Unknown => field("vst3", &check.detail),
			Presence::Missing => {
				if !args.force {
					return Err(format!(
						"`{}` does not export GetPluginFactory, so hosts will not load it as a VST3 plug-in.\n\
						 Rebuild with the vst3 feature (which pulls in clap-wrapper), for example:\n\
						     cargo build --release --features vst3\n\
						 or pass --force to write the bundle anyway.",
						library.display()
					));
				}
				eprintln!(
					"warning: `{}` does not export GetPluginFactory; writing the bundle anyway (--force)",
					library.display()
				);
			},
		}

		let path = bundle::write_vst3(&library, &out_dir, &meta, format, arch, args.single_file)?;
		field("vst3 bundle", path.display());
		written.push((Kind::Vst3, path));
	}

	if !args.no_clap {
		let path = bundle::write_clap(&library, &out_dir, &meta, format)?;
		field("clap", path.display());
		written.push((Kind::Clap, path));
	}

	if args.install || args.install_system {
		for (kind, path) in &written {
			for system in [false, true] {
				if system && !args.install_system {
					continue;
				}
				if !system && !args.install {
					continue;
				}
				let Some(root) = bundle::plugin_dir(format, *kind, system) else {
					eprintln!(
						"warning: cannot determine the {} plug-in directory on this platform",
						kind.label()
					);
					continue;
				};
				let installed = bundle::install(path, &root)?;
				field("installed", installed.display());
			}
		}
	}

	Ok(())
}

fn build(args: &Args, info: &CrateInfo) -> Result<(), String> {
	let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
	let mut command = Command::new(cargo);
	command
		.arg("build")
		.arg("--manifest-path")
		.arg(&info.manifest_path);
	if args.release {
		command.arg("--release");
	}
	if let Some(features) = &args.features {
		command.arg("--features").arg(features);
	}
	if let Some(target) = &args.target {
		command.arg("--target").arg(target);
	}
	if let Some(dir) = &args.target_dir {
		command.arg("--target-dir").arg(dir);
	}

	println!(
		"building {} ({})...",
		info.package_name.as_deref().unwrap_or("<unknown package>"),
		if args.release { "release" } else { "debug" }
	);
	let status = command.status().map_err(|e| format!("cannot run cargo: {e}"))?;
	if !status.success() {
		return Err(format!("cargo build failed ({status})"));
	}
	Ok(())
}

fn lookup_library(args: &Args, info: &CrateInfo) -> Result<PathBuf, String> {
	let file_name = info
		.library_file_name()
		.ok_or_else(|| "cannot determine the library file name; pass the library path".to_string())?;
	let workspace_root = manifest::find_workspace_root(&info.dir);
	let target_dir = target::target_dir(
		args.target_dir.as_deref(),
		workspace_root.as_deref(),
		&info.dir,
	);
	let profile = if args.release { "release" } else { "debug" };
	target::find_library(&target_dir, &file_name, profile, args.target.as_deref())
}

fn validate_name(name: &str) -> Result<(), String> {
	const FORBIDDEN: [char; 9] = ['<', '>', ':', '"', '/', '\\', '|', '?', '*'];
	if name.is_empty() || name.trim() != name {
		return Err(format!(
			"the plug-in name `{name}` must not be empty or padded with spaces"
		));
	}
	if name.contains(FORBIDDEN) {
		return Err(format!(
			"the plug-in name `{name}` contains a character that cannot appear in a file name"
		));
	}
	Ok(())
}

/// A lower-case, ASCII, dash-separated version of `value`, for default bundle ids.
fn slug(value: &str) -> String {
	let mut out = String::new();
	let mut separator = false;
	for c in value.chars() {
		if c.is_ascii_alphanumeric() {
			if separator && !out.is_empty() {
				out.push('-');
			}
			separator = false;
			out.push(c.to_ascii_lowercase());
		} else {
			separator = true;
		}
	}
	if out.is_empty() {
		"plugin".to_string()
	} else {
		out
	}
}

fn format_label(format: Format) -> &'static str {
	match format {
		Format::Windows => "windows",
		Format::MacOs => "macos",
		Format::Linux => "linux",
	}
}

fn arch_label(arch: Arch) -> &'static str {
	match arch {
		Arch::X86_64 => "x86_64",
		Arch::X86 => "x86",
		Arch::Aarch64 => "aarch64",
		Arch::Arm => "arm",
	}
}

fn field(label: &str, value: impl std::fmt::Display) {
	println!("{label:<11}: {value}");
}
