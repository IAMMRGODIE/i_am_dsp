//! A one-shot check that the wry editor really follows a size change.
//!
//! Run with: cargo run -p i_am_plugin --example wry_resize

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
	use i_am_dsp::ProcessContext;
	use i_am_plugin::wry::WryEditor;
	use i_am_plugin::{Descriptor, Editor, EditorContext, ParentWindow, Plugin, Processor, WindowSize};
	use windows_sys::Win32::Foundation::HWND;
	use windows_sys::Win32::Graphics::Gdi::{COLOR_WINDOW, UpdateWindow};
	use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
	use windows_sys::Win32::UI::WindowsAndMessaging::{
		CW_USEDEFAULT, CreateWindowExW, DefWindowProcW, DestroyWindow, DispatchMessageW, MSG,
		PM_REMOVE, PeekMessageW, RegisterClassW, SW_SHOW, SetWindowPos, ShowWindow, TranslateMessage,
		SWP_NOZORDER, WNDCLASSW, WS_OVERLAPPEDWINDOW, WS_VISIBLE,
	};

	const HTML: &str = r#"<!doctype html>
<html><head><meta charset="utf-8"><style>html,body{margin:0;height:100%;background:#111;color:#eee;font:16px system-ui}</style></head>
<body><main id="size">-</main>
<script>
  const send = () => window.ipc.postMessage(String(window.innerWidth) + "x" + String(window.innerHeight));
  window.addEventListener("resize", send);
  setTimeout(send, 60);
  send();
</script></body></html>"#;

	#[derive(Clone, Debug)]
	pub struct DemoMessage(String);

	impl i_am_plugin::Message for DemoMessage {
		fn from_note_event(_: i_am_dsp::NoteEvent) -> Self {
			Self(String::new())
		}

		fn note_event(&self) -> Option<i_am_dsp::NoteEvent> {
			None
		}

		fn tick(_: Instant) -> Self {
			Self(String::new())
		}
	}

	pub struct DemoProcessor;


	impl Parameters for DemoProcessor {
		fn get_parameters(&self) -> Vec<Parameter> {
			Vec::new()
		}

		fn set_parameter(&mut self, _: &str, _: SetValue) -> bool {
			false
		}
	}

	impl Processor for DemoProcessor {
		type Message = DemoMessage;

		fn delay(&self) -> usize {
			0
		}

		fn process(&mut self, _: &mut [f32; 2], _: &[[f32; 2]], _: &mut Box<dyn ProcessContext>) {}

		fn on_message(&self, _: Self::Message) {}
	}

	impl Plugin for DemoProcessor {
		const DESCRIPTOR: Descriptor = Descriptor::new("dev.i-am-dsp.wry.resize", "Wry Resize");
		const HAS_GUI: bool = true;

		fn new() -> Self {
			Self
		}

		fn param_map(&self) -> ParamMap {
			ParamMap::default()
		}

		fn editor(&self) -> Option<Box<dyn Editor<Self>>> {
			Some(Box::new(WryEditor::new(HTML, |body| {
				Some(DemoMessage(body.to_owned()))
			})))
		}
	}

	fn wide(text: &str) -> Vec<u16> {
		text.encode_utf16().chain(std::iter::once(0)).collect()
	}

	pub fn main() -> Result<(), Box<dyn std::error::Error>> {
		let class_name = wide("IAmPluginWryResizeHost");
		let title = wide("wry resize check");

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
				820,
				620,
				std::ptr::null_mut(),
				std::ptr::null_mut(),
				hinstance,
				std::ptr::null(),
			);

			if hwnd.is_null() {
				let code = windows_sys::Win32::Foundation::GetLastError();
				return Err(format!("CreateWindowExW failed: {code}").into());
			}

			ShowWindow(hwnd, SW_SHOW);
			UpdateWindow(hwnd);

			let parent = ParentWindow::from_win32_hwnd(hwnd).expect("a nonzero HWND");
			let (sender, receiver) = crossbeam_channel::unbounded::<DemoMessage>();
			let mut editor = DemoProcessor.editor().expect("the demo plug-in has an editor");

			editor.open(EditorContext {
				parent,
				size: WindowSize::new(760.0, 540.0),
				scale_factor: Some(1.0),
				title: "wry resize check".to_string(),
				resizable: true,
				messages: sender,
			})?;

			let pump = |millis: u64| {
				let deadline = Instant::now() + Duration::from_millis(millis);

				while Instant::now() < deadline {
					let mut message: MSG = std::mem::zeroed();

					while PeekMessageW(&mut message, std::ptr::null_mut(), 0, 0, PM_REMOVE) != 0 {
						TranslateMessage(&message);
						DispatchMessageW(&message);
					}

					while let Ok(message) = receiver.try_recv() {
						if !message.0.is_empty() {
							println!("page viewport: {}", message.0);
						}
					}

					std::thread::sleep(Duration::from_millis(5));
				}
			};

			pump(1200);
			println!("-- host window to 1100x780, editor to 1100x780 --");

			SetWindowPos(hwnd, std::ptr::null_mut(), 0, 0, 1100, 780, SWP_NOZORDER);
			editor.set_size(WindowSize::new(1100.0, 780.0))?;
			pump(1500);

			editor.close();
			DestroyWindow(hwnd);
			Ok(())
		}
	}
}

#[cfg(feature = "vst3")]
#[unsafe(no_mangle)]
static clap_entry: [u8; 64] = [0; 64];
