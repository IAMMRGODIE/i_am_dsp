//! A live smoke test for the wry editor.
//!
//! It creates a plain Win32 parent window (the part a CLAP host would own), puts a
//! [WryEditor] inside it with a small HTML page, pumps the window's messages for a
//! few seconds, and prints every string the page sends back over IPC. The page posts
//! "ready" as it loads and a button posts more, which is the whole path a real
//! plug-in GUI would use.
//!
//! Run: cargo run -p i_am_plugin --example standalone
//!
//! A window opens for six seconds and closes itself.

#[cfg(not(windows))]
fn main() {
	eprintln!("the wry backend is only implemented on Windows");
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
	use i_am_plugin::wry::WryEditor;
	use i_am_plugin::{Descriptor, Editor, EditorContext, ParentWindow, Plugin, WindowSize};
	use windows_sys::Win32::Foundation::HWND;
	use windows_sys::Win32::Graphics::Gdi::{COLOR_WINDOW, UpdateWindow};
	use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
	use windows_sys::Win32::UI::WindowsAndMessaging::{
		CW_USEDEFAULT, CreateWindowExW, DefWindowProcW, DestroyWindow, DispatchMessageW, MSG,
		PM_REMOVE, PeekMessageW, RegisterClassW, SW_SHOW, ShowWindow, TranslateMessage, WNDCLASSW,
		WS_OVERLAPPEDWINDOW, WS_VISIBLE,
	};

	/// The page the editor shows. It proves two things the backend has to do: the
	/// HTML is really rendered, and its IPC reaches Rust.
	const HTML: &str = r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>i_am_plugin + wry</title>
<style>
  :root { color-scheme: dark; }
  html, body { height: 100%; margin: 0; }
  body {
    display: grid;
    place-items: center;
    background: radial-gradient(circle at 30% 20%, #22223b, #0b0b12 70%);
    color: #ececf1;
    font-family: ui-sans-serif, system-ui, sans-serif;
    overflow: hidden;
  }
  main { text-align: center; transform: translateY(-8px); }
  h1 { margin: 0 0 8px; font-size: 30px; font-weight: 650; letter-spacing: -0.02em; }
  h1 span { color: #a78bfa; }
  p { margin: 0 0 26px; color: #9ca3af; font-size: 14px; }
  .orb {
    width: 130px; height: 130px; margin: 0 auto 28px;
    border-radius: 50%;
    background: conic-gradient(from 0deg, #a78bfa, #22d3ee, #a78bfa);
    filter: blur(2px);
    animation: spin 4s linear infinite;
    box-shadow: 0 0 60px -10px #8b5cf6;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  button {
    border: 1px solid #3f3f56; border-radius: 10px;
    background: #1b1b2b; color: #ececf1;
    padding: 10px 18px; font-size: 14px; cursor: pointer;
  }
  button:hover { background: #26263c; }
  #count { color: #22d3ee; font-variant-numeric: tabular-nums; }
</style>
</head>
<body>
<main>
  <div class="orb"></div>
  <h1>i_am_plugin <span>&times;</span> wry</h1>
  <p>This page is a plug-in editor, embedded through the Editor seam.</p>
  <button id="send">Send a message to the processor</button>
  <p style="margin-top:18px">messages sent: <span id="count">0</span></p>
</main>
<script>
  let count = 0;
  const counter = document.getElementById('count');
  function post(text) {
    count += 1;
    counter.textContent = String(count);
    if (window.ipc && window.ipc.postMessage) {
      window.ipc.postMessage(text);
    }
  }
  document.getElementById('send').addEventListener('click', function () {
    post('hello from the page');
  });
  post('ready');
</script>
</body>
</html>
"#;

	/// The message type the page produces.
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

	/// The processor behind the page. It never runs audio here; the example is about
	/// the editor seam, not the DSP.
	pub struct DemoProcessor {
		data_directory: std::path::PathBuf,
	}

	impl DemoProcessor {
		fn with_data_directory(path: impl Into<std::path::PathBuf>) -> Self {
			Self {
				data_directory: path.into(),
			}
		}
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
		const DESCRIPTOR: Descriptor = Descriptor::new("dev.i-am-dsp.wry.demo", "I Am Wry Demo");
		const HAS_GUI: bool = true;

		fn new() -> Self {
			Self {
				data_directory: std::path::PathBuf::from("."),
			}
		}

		fn param_map(&self) -> ParamMap {
			ParamMap::default()
		}

		// Exactly what the wrapper calls when the host opens the GUI: the plug-in
		// picks a backend, and nothing above this line knows which one it is.
		fn editor(&self) -> Option<Box<dyn Editor<Self>>> {
			Some(Box::new(
				WryEditor::new(HTML, |body| Some(DemoMessage(body.to_owned())))
					.with_data_directory(self.data_directory.clone()),
			))
		}
	}

	fn wide(text: &str) -> Vec<u16> {
		text.encode_utf16().chain(std::iter::once(0)).collect()
	}

	pub fn main() -> Result<(), Box<dyn std::error::Error>> {
		// WebView2 keeps a profile directory. Its default lives under %LOCALAPPDATA%,
		// which a sandboxed build cannot write; keeping it in the workspace also
		// makes the example self-contained.
		let data_directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
			.join("..")
			.join("..")
			.join("target")
			.join("wry-webview2-data");
		std::fs::create_dir_all(&data_directory)?;

		let class_name = wide("IAmPluginWryHost");
		let title = wide("i_am_plugin + wry");

		// SAFETY: every call is a plain Win32 call on this thread, and the pointers
		// passed come from locals that outlive the calls.
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
				800,
				600,
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
			let mut editor = DemoProcessor::with_data_directory(data_directory)
				.editor()
				.expect("the demo plug-in has an editor");

			editor.open(EditorContext {
				parent,
				size: WindowSize::new(760.0, 540.0),
				scale_factor: Some(1.0),
				title: "i_am_plugin + wry".to_string(),
				resizable: true,
				messages: sender,
			})?;

			println!("editor opened; pumping messages for six seconds");

			let deadline = Instant::now() + Duration::from_secs(6);
			let mut ready = false;

			while Instant::now() < deadline {
				let mut message: MSG = std::mem::zeroed();

				while PeekMessageW(&mut message, std::ptr::null_mut(), 0, 0, PM_REMOVE) != 0 {
					TranslateMessage(&message);
					DispatchMessageW(&message);
				}

				while let Ok(message) = receiver.try_recv() {
					println!("ipc: {}", message.0);

					if message.0 == "ready" {
						ready = true;
					}
				}

				std::thread::sleep(Duration::from_millis(8));
			}

			editor.close();
			DestroyWindow(hwnd);

			println!("closed; the page reported ready = {ready}");
			Ok(())
		}
	}
}

/// The example is not a plug-in, but a vst3-enabled workspace still links the
/// clap-wrapper static library that expects this data symbol.
#[cfg(feature = "vst3")]
#[unsafe(no_mangle)]
static clap_entry: [u8; 64] = [0; 64];
