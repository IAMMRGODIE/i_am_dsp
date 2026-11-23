//! The page's Save button: a native save-as dialog beside the editor.
//!
//! The dialog is a modal window on a worker thread, so the page asks for a save and
//! then asks how it ended: the answer outlives the request that started it.

use std::sync::{Arc, Mutex};

use serde_json::json;

/// What a request to save was answered with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Asked {
	/// A dialog is on screen.
	Started,
	/// One was already on screen.
	Busy,
	/// This build has no native dialog.
	Unavailable,
}

/// Where one save has got to.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum State {
	/// Nothing has been asked for since the last save ended.
	#[default]
	Idle,
	/// A dialog is on screen.
	Pending,
	/// The text was written, at this path.
	Saved(String),
	/// The dialog was dismissed without a file.
	Cancelled,
	/// The file could not be written.
	Failed(String),
}

/// The save the page is waiting on.
#[derive(Default, Clone)]
pub struct SaveSlot {
	state: Arc<Mutex<State>>,
}

impl SaveSlot {
	/// The state, for the page.
	pub fn state(&self) -> State {
		self.state
			.lock()
			.map(|state| state.clone())
			.unwrap_or_default()
	}

	fn set(&self, state: State) {
		if let Ok(mut current) = self.state.lock() {
			*current = state;
		}
	}

	/// Writes the text to a chosen path and records it; false when it failed.
	pub fn write_to(&self, path: &std::path::Path, text: &str) -> bool {
		match std::fs::write(path, text) {
			Ok(()) => {
				self.set(State::Saved(path.display().to_string()));
				true
			}
			Err(error) => {
				self.set(State::Failed(error.to_string()));
				false
			}
		}
	}

	/// Records what the dialog answered: a path to write, or none for a cancel.
	pub fn settle(&self, path: Option<std::path::PathBuf>, text: &str) {
		match path {
			Some(path) => {
				self.write_to(&path, text);
			}
			None => self.set(State::Cancelled),
		}
	}

	/// Asks for the native save-as dialog for this text, under this name.
	pub fn ask(&self, text: String, suggested: &str) -> Asked {
		if !i_am_effects::dialog::CAN_SAVE {
			return Asked::Unavailable;
		}

		{
			let Ok(mut state) = self.state.lock() else {
				return Asked::Busy;
			};

			if matches!(*state, State::Pending) {
				return Asked::Busy;
			}

			*state = State::Pending;
		}

		let slot = self.clone();

		i_am_effects::dialog::save(suggested, ("Patch text", &["txt"]), move |path| {
			slot.settle(path, &text);
		});

		Asked::Started
	}

	/// The JSON the page polls.
	pub fn json(&self) -> String {
		match self.state() {
			State::Idle => json!({ "state": "idle" }).to_string(),
			State::Pending => json!({ "state": "pending" }).to_string(),
			State::Cancelled => json!({ "state": "cancelled" }).to_string(),
			State::Saved(path) => json!({ "state": "saved", "path": path }).to_string(),
			State::Failed(error) => json!({ "state": "failed", "error": error }).to_string(),
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// A save writes the text where the dialog said, and the page can read the path.
	#[test]
	fn a_save_writes_the_text_and_says_where() {
		let slot = SaveSlot::default();
		let path = std::env::temp_dir().join("i_am_patch_save_slot_test.txt");
		let text = "node 0 out:audio 0 0\n";

		assert_eq!(slot.state(), State::Idle);
		assert!(slot.write_to(&path, text));

		assert_eq!(slot.state(), State::Saved(path.display().to_string()));
		assert_eq!(std::fs::read_to_string(&path).expect("the file"), text);
		assert!(slot.json().contains("\"state\":\"saved\""));

		let _ = std::fs::remove_file(&path);
	}

	/// The dialog's two answers: a path, which is written, or a cancel, which is not.
	#[test]
	fn the_dialog_answer_is_what_writes_the_file() {
		let slot = SaveSlot::default();
		let path = std::env::temp_dir().join("i_am_patch_save_settle_test.txt");
		let text = "node 0 out:audio 0 0\n";

		slot.settle(Some(path.clone()), text);
		assert_eq!(slot.state(), State::Saved(path.display().to_string()));
		assert_eq!(std::fs::read_to_string(&path).expect("the file"), text);

		slot.settle(None, text);
		assert_eq!(slot.state(), State::Cancelled);
		assert!(slot.json().contains("\"cancelled\""));

		let _ = std::fs::remove_file(&path);
	}

	/// A path that cannot be written is reported instead of pretended.
	#[test]
	fn a_save_that_cannot_write_says_so() {
		let slot = SaveSlot::default();

		assert!(!slot.write_to(&std::env::temp_dir(), "x"));
		assert!(matches!(slot.state(), State::Failed(_)));
		assert!(slot.json().contains("\"state\":\"failed\""));
	}
}
