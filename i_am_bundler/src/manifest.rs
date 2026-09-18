//! A deliberately tiny reader for the few `Cargo.toml` fields the bundler needs.
//!
//! It understands tables, `key = "string"`, `key = 'string'`, string arrays and
//! multi-line strings. It is *not* a TOML implementation and makes no attempt to be
//! one: it only has to be predictable for the handful of keys the bundler looks up.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

/// The manifest fields the bundler cares about.
#[derive(Debug, Default, Clone)]
pub struct CrateInfo {
	/// Path of the manifest this was read from.
	pub manifest_path: PathBuf,
	/// Directory holding the manifest.
	pub dir: PathBuf,
	/// `[package] name`
	pub package_name: Option<String>,
	/// `[package] version`
	pub package_version: Option<String>,
	/// `[lib] name` (defaults to the package name with `-` replaced by `_`)
	pub lib_name: Option<String>,
	/// Raw text of `[lib] crate-type`.
	pub crate_types: Option<String>,
	/// `[package.metadata.i_am_dsp] name`
	pub plugin_name: Option<String>,
	/// `[package.metadata.i_am_dsp] vendor`
	pub vendor: Option<String>,
	/// `[package.metadata.i_am_dsp] version`
	pub version: Option<String>,
	/// `[package.metadata.i_am_dsp] id`
	pub id: Option<String>,
	/// `[workspace] members`, when this manifest is a workspace root.
	pub workspace_members: Vec<String>,
	/// Whether the manifest has a `[package]` table.
	pub is_package: bool,
	/// Whether the manifest has a `[workspace]` table.
	pub is_workspace: bool,
}

impl CrateInfo {
	/// Reads and parses a manifest.
	pub fn load(path: &Path) -> Result<Self, String> {
		let text = fs::read_to_string(path)
			.map_err(|e| format!("cannot read `{}`: {e}", path.display()))?;
		Ok(Self::from_text(path, &text))
	}

	fn from_text(path: &Path, text: &str) -> Self {
		let toml = TomlLite::parse(text);
		Self {
			manifest_path: path.to_path_buf(),
			dir: path.parent().unwrap_or_else(|| Path::new(".")).to_path_buf(),
			package_name: toml.get_string("package.name"),
			package_version: toml.get_string("package.version"),
			lib_name: toml.get_string("lib.name"),
			crate_types: toml.get("lib.crate-type").map(str::to_string),
			plugin_name: toml.get_string("package.metadata.i_am_dsp.name"),
			vendor: toml.get_string("package.metadata.i_am_dsp.vendor"),
			version: toml.get_string("package.metadata.i_am_dsp.version"),
			id: toml.get_string("package.metadata.i_am_dsp.id"),
			workspace_members: toml.get_array("workspace.members"),
			is_package: toml.has_table("package"),
			is_workspace: toml.has_table("workspace"),
		}
	}

	/// Whether the crate declares a `cdylib` target.
	pub fn is_cdylib(&self) -> bool {
		self.crate_types
			.as_deref()
			.is_some_and(|types| types.contains("cdylib"))
	}

	/// The name of the built library file, without a directory.
	pub fn library_file_name(&self) -> Option<String> {
		let package = self.package_name.as_deref()?;
		let stem = self.lib_name.as_deref().unwrap_or(package).replace('-', "_");
		Some(library_file_name(&stem))
	}
}

/// The file name a cdylib with this stem gets on the current platform.
pub fn library_file_name(stem: &str) -> String {
	if cfg!(windows) {
		format!("{stem}.dll")
	} else if cfg!(target_os = "macos") {
		format!("lib{stem}.dylib")
	} else {
		format!("lib{stem}.so")
	}
}

/// Walks upwards from `start` looking for the manifest of the workspace that owns it.
pub fn find_workspace_root(start: &Path) -> Option<PathBuf> {
	let mut current = Some(start.to_path_buf());
	while let Some(dir) = current {
		let manifest = dir.join("Cargo.toml");
		if manifest.is_file()
			&& let Ok(info) = CrateInfo::load(&manifest)
			&& info.is_workspace
		{
			return Some(dir);
		}
		current = dir.parent().map(Path::to_path_buf);
	}
	None
}

/// Resolves a cargo package name to its manifest, starting from `cwd_manifest`.
pub fn resolve_package_manifest(cwd_manifest: &Path, package: &str) -> Result<PathBuf, String> {
	// The manifest we were pointed at may be the package itself...
	if let Ok(info) = CrateInfo::load(cwd_manifest) {
		if info.package_name.as_deref() == Some(package) {
			return Ok(cwd_manifest.to_path_buf());
		}
		// ...or a workspace listing it in `members`...
		let root = cwd_manifest.parent().unwrap_or_else(|| Path::new("."));
		let mut candidates: Vec<PathBuf> = info
			.workspace_members
			.iter()
			.filter(|member| !member.contains('*'))
			.map(|member| root.join(member).join("Cargo.toml"))
			.collect();

		// ...or just one of the workspace root's own directories, which covers
		// members we could not read from the manifest (globs, `exclude`, ...).
		if let Ok(entries) = fs::read_dir(root) {
			for entry in entries.flatten() {
				let candidate = entry.path().join("Cargo.toml");
				if candidate.is_file() {
					candidates.push(candidate);
				}
			}
		}

		for candidate in candidates {
			if let Ok(member_info) = CrateInfo::load(&candidate)
				&& member_info.package_name.as_deref() == Some(package)
			{
				return Ok(candidate);
			}
		}
	}

	Err(format!(
		"cannot find the package `{package}` from `{}`; pass --manifest-path <PLUGIN/Cargo.toml>",
		cwd_manifest.display()
	))
}

/// The tiny TOML subset reader described in the module documentation.
struct TomlLite {
	values: BTreeMap<String, String>,
	tables: BTreeSet<String>,
}

impl TomlLite {
	fn parse(text: &str) -> Self {
		let mut values = BTreeMap::new();
		let mut tables = BTreeSet::new();
		let mut table = String::new();
		let mut lines = text.lines();

		while let Some(raw) = lines.next() {
			let stripped = strip_comment(raw);
			let line = stripped.trim();
			if line.is_empty() {
				continue;
			}

			if let Some(header) = line.strip_prefix('[') {
				let header = header.trim_start_matches('[').trim_end_matches(']').trim();
				table = header.to_string();
				tables.insert(table.clone());
				continue;
			}

			let Some(eq) = line.find('=') else {
				continue;
			};
			let key = line[..eq].trim().to_string();
			let mut value = line[eq + 1..].trim().to_string();

			// Arrays are commonly spread over several lines; keep reading until the
			// closing bracket shows up.
			if value.starts_with('[') && !value.contains(']') {
				for next in lines.by_ref() {
					let next = strip_comment(next);
					value.push(' ');
					value.push_str(next.trim());
					if value.contains(']') {
						break;
					}
				}
			}

			// Swallow the body of a multi-line string so it can never be mistaken
			// for further keys.
			if let Some(terminator) = multiline_terminator(&value) {
				for next in lines.by_ref() {
					if next.contains(terminator) {
						break;
					}
				}
			}

			let full = if table.is_empty() {
				key
			} else {
				format!("{table}.{key}")
			};
			values.insert(full, value);
		}

		Self { values, tables }
	}

	fn get(&self, path: &str) -> Option<&str> {
		self.values.get(path).map(String::as_str)
	}

	fn get_string(&self, path: &str) -> Option<String> {
		self.get(path).and_then(parse_string)
	}

	fn get_array(&self, path: &str) -> Vec<String> {
		let Some(raw) = self.get(path) else {
			return Vec::new();
		};
		let mut out = Vec::new();
		let mut rest = raw;
		while let Some(start) = rest.find('"') {
			let after = &rest[start + 1..];
			let Some(end) = after.find('"') else { break };
			out.push(after[..end].to_string());
			rest = &after[end + 1..];
		}
		out
	}

	fn has_table(&self, path: &str) -> bool {
		self.tables.contains(path)
	}
}

/// Removes a trailing `#` comment, ignoring `#` inside quoted strings.
fn strip_comment(line: &str) -> String {
	let mut out = String::new();
	let mut quote: Option<char> = None;
	let mut chars = line.chars();
	while let Some(c) = chars.next() {
		match quote {
			Some(q) => {
				out.push(c);
				if c == '\\' && q == '"' {
					if let Some(escaped) = chars.next() {
						out.push(escaped);
					}
				} else if c == q {
					quote = None;
				}
			},
			None => {
				if c == '#' {
					break;
				}
				if c == '"' || c == '\'' {
					quote = Some(c);
				}
				out.push(c);
			},
		}
	}
	out
}

/// Returns the terminator of an unterminated multi-line string value.
fn multiline_terminator(value: &str) -> Option<&'static str> {
	for terminator in ["\"\"\"", "'''"] {
		if let Some(rest) = value.strip_prefix(terminator)
			&& !rest.contains(terminator)
		{
			return Some(terminator);
		}
	}
	None
}

/// Reads a basic or literal string value.
fn parse_string(value: &str) -> Option<String> {
	let value = value.trim();
	if let Some(rest) = value.strip_prefix("\"\"\"") {
		return Some(rest.trim_end_matches('"').to_string());
	}
	if let Some(rest) = value.strip_prefix("'''") {
		return Some(rest.trim_end_matches('\'').to_string());
	}

	match value.chars().next()? {
		'"' => unescape(value),
		'\'' => {
			let end = value[1..].find('\'')?;
			Some(value[1..1 + end].to_string())
		},
		_ => None,
	}
}

fn unescape(value: &str) -> Option<String> {
	let mut out = String::new();
	let mut chars = value[1..].chars();
	while let Some(c) = chars.next() {
		match c {
			'"' => return Some(out),
			'\\' => match chars.next()? {
				'n' => out.push('\n'),
				't' => out.push('\t'),
				'r' => out.push('\r'),
				'0' => out.push('\0'),
				'\\' => out.push('\\'),
				'"' => out.push('"'),
				other => out.push(other),
			},
			other => out.push(other),
		}
	}
	None
}
