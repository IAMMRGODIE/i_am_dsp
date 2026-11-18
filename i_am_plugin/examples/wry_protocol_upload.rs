//! Does a custom-protocol route receive the body of a POST?
//!
//! The Convolver product uploads an impulse response this way: the page POSTs the
//! picked file to `/ir`, and the Rust side decodes it. The page cannot read the
//! body back if WebView2 drops it, so this example checks the one thing the Rust
//! tests cannot: that the bytes cross the WebView2 boundary.
//!
//! Run: cargo run -p i_am_plugin --example protocol_upload
//!
//! A window opens, the page posts a short body, and its echo is printed. It closes
//! itself after five seconds.

#[cfg(not(windows))]
fn main() {
	eprintln!("this smoke test is Windows-only; the backend also runs on Linux and macOS");
}

#[cfg(windows)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
	imp::main()
}

#[cfg(windows)]
mod imp {
	use std::ffi::c_void;
	use std::time::{Duration, Instant};

	use i_am_dsp::prelude::{ParamMap, Parameter, Parameters, SetValue};
	use i_am_dsp::{NoteEvent, ProcessContext};
	use i_am_plugin::wry::{ProtocolRequest, ProtocolResponse, WryEditor};
	use i_am_plugin::{Descriptor, Editor, EditorContext, ParentWindow, Plugin, WindowSize};
	use windows_sys::Win32::Foundation::HWND;
	use windows_sys::Win32::Graphics::Gdi::{COLOR_WINDOW, UpdateWindow};
	use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
	use windows_sys::Win32::UI::WindowsAndMessaging::{
		CW_USEDEFAULT, CreateWindowExW, DefWindowProcW, DispatchMessageW, MSG, PM_REMOVE,
		PeekMessageW, RegisterClassW, SW_SHOW, ShowWindow, TranslateMessage, WNDCLASSW,
		WS_OVERLAPPEDWINDOW, WS_VISIBLE,
	};

	/// The body the page posts. Its length is what the echo proves.
	const BODY: &str = "hello upload";

	const HTML: &str = r#"<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>upload probe</title></head>
<body style="font: 13px monospace; background: #111; color: #ddd; padding: 16px">
<p>Posting a body to the custom protocol...</p>
<script>
fetch("/upload", { method: "POST", body: "hello upload" })
  .then(function (response) { return response.text(); })
  .then(function (text) { window.ipc.postMessage("result:" + text); })
  .catch(function (error) { window.ipc.postMessage("error:" + error); });
</script>
</body>
</html>
"#;

	#[derive(Clone, Debug)]
	pub struct DemoMessage(pub String);

	impl i_am_plugin::Message for DemoMessage {
		fn from_note_event(_: NoteEvent) -> Self {
			Self(String::new())
		}

		fn note_event(&self) -> Option<NoteEvent> {
			None
		}

		fn tick(_: Instant) -> Self {
			Self(String::new())
		}
	}

	pub struct DemoProcessor {
		data_directory: std::path::PathBuf,
	}

	impl Parameters for DemoProcessor {
		fn get_parameters(&self) -> Vec<Parameter> {
			Vec::new()
		}

		fn set_parameter(&mut self, _: &str, _: SetValue) -> bool {
			false
		}
	}

	impl i_am_plugin::Processor for DemoProcessor {
		type Message = DemoMessage;

		fn delay(&self) -> usize {
			0
		}

		fn process(&mut self, _: &mut [f32; 2], _: &[[f32; 2]], _: &mut Box<dyn ProcessContext>) {}

		fn on_message(&self, _: Self::Message) {}
	}

	impl Plugin for DemoProcessor {
		const DESCRIPTOR: Descriptor =
			Descriptor::new("dev.i-am-dsp.wry.upload", "I Am Wry Upload Probe");
		const HAS_GUI: bool = true;

		fn new() -> Self {
			Self {
				data_directory: std::path::PathBuf::from("."),
			}
		}

		fn param_map(&self) -> ParamMap {
			ParamMap::default()
		}

		// The route echoes what arrived, so the length is visible from the outside.
		fn editor(&self) -> Option<Box<dyn Editor<Self>>> {
			Some(Box::new(
				WryEditor::new("", |message| Some(DemoMessage(message.to_owned())))
					.with_data_directory(self.data_directory.clone())
					.with_custom_protocol_request("demo", move |request: &ProtocolRequest<'_>| {
						match request.path {
							"/" => ProtocolResponse::html(HTML),
							"/upload" => ProtocolResponse::text(format!(
								"method={} length={} body={}",
								request.method,
								request.body.len(),
								String::from_utf8_lossy(request.body),
							)),
							_ => ProtocolResponse::not_found(),
						}
					})
					.with_url("demo://localhost/"),
			))
		}
	}

	fn wide(text: &str) -> Vec<u16> {
		text.encode_utf16().chain(std::iter::once(0)).collect()
	}

	pub fn main() -> Result<(), Box<dyn std::error::Error>> {
		let data_directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
			.join("..")
			.join("target")
			.join("wry-upload-data");
		std::fs::create_dir_all(&data_directory)?;

		let class_name = wide("IAmPluginWryUpload");
		let title = wide("i_am_plugin + wry upload");

		unsafe {
			let hinstance = GetModuleHandleW(std::ptr::null());
			let class = WNDCLASSW {
				lpfnWndProc: Some(DefWindowProcW),
				hInstance: hinstance,
				hbrBackground: (COLOR_WINDOW as usize + 1) as *mut c_void,
				lpszClassName: class_name.as_ptr(),
				..Default::default()
			};

			if RegisterClassW(&class) == 0 {
				return Err("RegisterClassW failed".into());
			}

			let hwnd: HWND = CreateWindowExW(
				0,
				class_name.as_ptr(),
				title.as_ptr(),
				WS_OVERLAPPEDWINDOW | WS_VISIBLE,
				CW_USEDEFAULT,
				CW_USEDEFAULT,
				520,
				260,
				std::ptr::null_mut(),
				std::ptr::null_mut(),
				hinstance,
				std::ptr::null(),
			);

			if hwnd.is_null() {
				return Err("CreateWindowExW failed".into());
			}

			ShowWindow(hwnd, SW_SHOW);
			UpdateWindow(hwnd);

			let parent = ParentWindow::from_win32_hwnd(hwnd).expect("a nonzero HWND");
			let (sender, receiver) = crossbeam_channel::unbounded::<DemoMessage>();
			let mut editor = DemoProcessor {
				data_directory: data_directory.clone(),
			}
			.editor()
			.expect("the probe has an editor");

			editor.open(EditorContext {
				parent,
				size: WindowSize::new(500.0, 220.0),
				scale_factor: Some(1.0),
				title: "upload probe".to_string(),
				resizable: false,
				messages: sender,
			})?;

			println!("waiting for the page to post {BODY:?}");

			let deadline = Instant::now() + Duration::from_secs(5);

			while Instant::now() < deadline {
				let mut message: MSG = std::mem::zeroed();

				while PeekMessageW(&mut message, std::ptr::null_mut(), 0, 0, PM_REMOVE) != 0 {
					TranslateMessage(&message);
					DispatchMessageW(&message);
				}

				while let Ok(message) = receiver.try_recv() {
					println!("ipc: {}", message.0);
				}

				std::thread::sleep(Duration::from_millis(8));
			}

			editor.close();
			Ok(())
		}
	}
}

/// The example is not a plug-in, but a vst3-enabled workspace still links the
/// clap-wrapper static library that expects this data symbol.
#[cfg(feature = "vst3")]
#[unsafe(no_mangle)]
static clap_entry: [u8; 64] = [0; 64];
