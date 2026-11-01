//! Bounded diagnostics for the WebView2 editor.
//!
//! A host swallows the error a GUI open returns, so the editor leaves a short
//! trail under the temp directory: which lifecycle call ran, on which thread,
//! and how far an open got. The trail is deliberately small. open.log rotates
//! once when it reaches 256 KiB (override with I_AM_PLUGIN_WRY_LOG_BYTES), and
//! last-error.log is capped and rewritten in place. The backup of an open log
//! is kept next to it for one generation.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// The default cap for the open log. The rotated backup doubles the on-disk
/// cost, so the default pair is bounded to roughly half a megabyte.
const DEFAULT_OPEN_LOG_BYTES: u64 = 256 * 1024;

/// The error log is a single short note, not an append-only history.
const ERROR_LOG_BYTES: u64 = 64 * 1024;

/// A process has one UI thread in the normal case, but a host may move the
/// editor between threads. Serialising writes keeps two such calls from
/// interleaving half-lines or rotating at the same time.
static WRITE_LOCK: Mutex<()> = Mutex::new(());

/// The directory the editor writes its own diagnostics into.
pub(crate) fn log_dir() -> PathBuf {
	std::env::temp_dir().join("i_am_plugin_wry")
}

/// Append one timestamped line to open.log, rotating it when it is full.
pub(crate) fn note(message: &str) {
	let path = log_dir().join("open.log");
	let line = format!(
		"{} [pid:{} tid:{}] {}\n",
		timestamp(),
		process_id(),
		thread_id(),
		message
	);
	let _ = append_rotated(&path, &line, open_log_limit());
}

/// Record a failure where a host's log cannot be read.
///
/// The error log keeps the most recent notes, but is truncated rather than
/// growing without bound. A successful open removes it.
pub(crate) fn error(message: &str) {
	let path = log_dir().join("last-error.log");
	let line = format!(
		"{} [pid:{} tid:{}] {}\n",
		timestamp(),
		process_id(),
		thread_id(),
		message
	);
	let _ = write_bounded(&path, &line, ERROR_LOG_BYTES);
}

/// Remove the error log after an editor opened successfully.
pub(crate) fn clear_error() {
	let _ = fs::remove_file(log_dir().join("last-error.log"));
}

fn open_log_limit() -> u64 {
	std::env::var("I_AM_PLUGIN_WRY_LOG_BYTES")
		.ok()
		.and_then(|value| value.parse::<u64>().ok())
		.filter(|value| *value >= 4096)
		.unwrap_or(DEFAULT_OPEN_LOG_BYTES)
}

fn append_rotated(path: &Path, line: &str, max_bytes: u64) -> std::io::Result<()> {
	let _guard = WRITE_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());

	if let Some(parent) = path.parent() {
		fs::create_dir_all(parent)?;
	}

	if let Ok(metadata) = fs::metadata(path)
		&& metadata.len().saturating_add(line.len() as u64) > max_bytes
	{
		rotate(path)?;
	}

	let mut file = OpenOptions::new().create(true).append(true).open(path)?;
	file.write_all(line.as_bytes())
}

fn rotate(path: &Path) -> std::io::Result<()> {
	let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
		return Ok(());
	};
	let backup = path.with_file_name(format!("{name}.1"));
	let _ = fs::remove_file(&backup);
	fs::rename(path, &backup)
}

fn write_bounded(path: &Path, line: &str, max_bytes: u64) -> std::io::Result<()> {
	let _guard = WRITE_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());

	if let Some(parent) = path.parent() {
		fs::create_dir_all(parent)?;
	}

	let mut body = String::new();
	if let Ok(existing) = fs::read_to_string(path)
		&& (existing.len() as u64).saturating_add(line.len() as u64) <= max_bytes
	{
		body.push_str(&existing);
	}
	body.push_str(line);
	fs::write(path, body)
}

#[cfg(windows)]
fn timestamp() -> String {
	use windows_sys::Win32::Foundation::SYSTEMTIME;
	use windows_sys::Win32::System::SystemInformation::GetLocalTime;

	// SAFETY: GetLocalTime writes into the struct it is given; the struct is
	// local and fully initialised by the call.
	let mut time: SYSTEMTIME = unsafe { std::mem::zeroed() };
	unsafe { GetLocalTime(&mut time) };

	format!(
		"{:04}-{:02}-{:02} {:02}:{:02}:{:02}.{:03}",
		time.wYear,
		time.wMonth,
		time.wDay,
		time.wHour,
		time.wMinute,
		time.wSecond,
		time.wMilliseconds
	)
}

#[cfg(not(windows))]
fn timestamp() -> String {
	let now = std::time::SystemTime::now()
		.duration_since(std::time::UNIX_EPOCH)
		.unwrap_or_default();
	format!("unix-{}.{:03}", now.as_secs(), now.subsec_millis())
}

#[cfg(windows)]
fn process_id() -> u32 {
	// SAFETY: both calls take no pointers and cannot fail.
	unsafe { windows_sys::Win32::System::Threading::GetCurrentProcessId() }
}

#[cfg(windows)]
fn thread_id() -> u32 {
	// SAFETY: the call takes no arguments and returns this thread's id.
	unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() }
}

#[cfg(not(windows))]
fn process_id() -> u32 {
	std::process::id()
}

#[cfg(not(windows))]
fn thread_id() -> u32 {
	0
}

#[cfg(test)]
mod tests {
	use super::{append_rotated, write_bounded};
	use std::fs;
	use std::path::PathBuf;

	fn test_dir(name: &str) -> PathBuf {
		let stamp = std::time::SystemTime::now()
			.duration_since(std::time::UNIX_EPOCH)
			.unwrap()
			.as_nanos();
		let path = std::env::temp_dir().join(format!(
			"i_am_plugin_wry_{name}_{}_{}",
			std::process::id(),
			stamp
		));
		fs::create_dir_all(&path).unwrap();
		path
	}

	#[test]
	fn a_full_open_log_is_rotated_once() {
		let dir = test_dir("rotate");
		let path = dir.join("open.log");

		for _ in 0..10 {
			append_rotated(&path, "0123456789\n", 32).unwrap();
		}

		assert!(path.exists(), "the current log is recreated after rotation");
		assert!(path.with_file_name("open.log.1").exists(), "the old log is kept");
		assert!(
			fs::metadata(path.with_file_name("open.log.1")).unwrap().len() <= 32,
			"the backup is not larger than the cap"
		);

		let _ = fs::remove_dir_all(&dir);
	}

	#[test]
	fn an_error_log_is_capped_and_rewritten() {
		let dir = test_dir("error");
		let path = dir.join("last-error.log");

		write_bounded(&path, "first\n", 24).unwrap();
		write_bounded(&path, "second\n", 24).unwrap();
		assert!(fs::read_to_string(&path).unwrap().contains("first"));

		write_bounded(&path, "this line is too long for the cap\n", 24).unwrap();
		let body = fs::read_to_string(&path).unwrap();
		assert_eq!(body, "this line is too long for the cap\n");

		let _ = fs::remove_dir_all(&dir);
	}
}
