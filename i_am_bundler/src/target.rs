//! Locating the cdylib cargo just built.

use std::path::{Path, PathBuf};

/// The directory cargo places build artifacts in.
pub fn target_dir(explicit: Option<&Path>, workspace_root: Option<&Path>, manifest_dir: &Path) -> PathBuf {
	if let Some(dir) = explicit {
		return dir.to_path_buf();
	}
	if let Some(dir) = std::env::var_os("CARGO_TARGET_DIR")
		&& !dir.is_empty()
	{
		return PathBuf::from(dir);
	}
	workspace_root.unwrap_or(manifest_dir).join("target")
}

/// Finds `<target-dir>/[<triple>/]<profile>/<file-name>`.
pub fn find_library(
	target_dir: &Path,
	file_name: &str,
	profile: &str,
	triple: Option<&str>,
) -> Result<PathBuf, String> {
	let mut candidates = Vec::new();
	if let Some(triple) = triple {
		candidates.push(target_dir.join(triple).join(profile).join(file_name));
	}
	candidates.push(target_dir.join(profile).join(file_name));

	for candidate in &candidates {
		if candidate.is_file() {
			return Ok(candidate.clone());
		}
	}

	let mut message = format!("cannot find `{file_name}`; looked at:\n");
	for candidate in &candidates {
		message.push_str(&format!("  - {}\n", candidate.display()));
	}
	message.push_str("build the plug-in first, or pass the library path as an argument");
	Err(message)
}
