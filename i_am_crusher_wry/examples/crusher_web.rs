//! I Am Crusher Web, in a plain window.
//!
//! It drives the same `Plugin::editor()` -> `Editor::open()` path a host uses, with a
//! real HWND standing in for the DAW's parent window. Resizing the window calls the
//! editor's `set_size`, as a host does, so the page's scaling can be seen.
//!
//! Run: cargo run -p i_am_crusher_wry --example crusher_web
//!
//! Optional arguments are the hold time in seconds and the outer window size:
//!
//! ```text
//! cargo run -p i_am_crusher_wry --example crusher_web -- 12 700 1000
//! ```

#[cfg(not(windows))]
fn main() {
	eprintln!("crusher_web is only implemented on Windows");
}

#[cfg(windows)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
	imp::main()
}

#[cfg(windows)]
mod imp {
	use std::ffi::c_void;
	use std::time::{Duration, Instant};

	use i_am_crusher_wry::CrusherWeb;
	use i_am_plugin::{EditorContext, ParentWindow, Plugin, WindowSize};
	use windows_sys::Win32::Foundation::{HWND, RECT};
	use windows_sys::Win32::Graphics::Gdi::{COLOR_WINDOW, UpdateWindow};
	use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
	use windows_sys::Win32::UI::WindowsAndMessaging::{
		CW_USEDEFAULT, CreateWindowExW, DefWindowProcW, DestroyWindow, DispatchMessageW,
		GetClientRect, MSG, PM_REMOVE, PeekMessageW, RegisterClassW, SW_SHOW, ShowWindow,
		TranslateMessage, WNDCLASSW, WS_OVERLAPPEDWINDOW, WS_VISIBLE,
	};

	fn wide(text: &str) -> Vec<u16> {
		text.encode_utf16().chain(std::iter::once(0)).collect()
	}

	pub fn main() -> Result<(), Box<dyn std::error::Error>> {
		let arguments: Vec<String> = std::env::args().skip(1).collect();
		let mut numbers = arguments
			.iter()
			.filter_map(|argument| argument.parse::<u32>().ok());
		let hold_seconds = numbers.next().map(u64::from).unwrap_or(12);
		let window_width = numbers.next().unwrap_or(440) as i32;
		let window_height = numbers.next().unwrap_or(700) as i32;

		let class_name = wide("IAmCrusherWebHost");
		let title = wide("I Am Crusher Web");

		// SAFETY: plain Win32 calls on this thread; the locals outlive each call.
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
				window_width,
				window_height,
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
			let crusher = CrusherWeb::new();
			let mut editor = Plugin::editor(&crusher).expect("the Crusher has a web editor");

			editor.open(EditorContext {
				parent,
				size: WindowSize::new(400.0, 640.0),
				scale_factor: Some(1.0),
				title: "I Am Crusher Web".to_string(),
				resizable: true,
				messages: crossbeam_channel::unbounded().0,
			})?;

			println!("crusher web editor opened; pumping messages for {hold_seconds} seconds");

			let deadline = Instant::now() + Duration::from_secs(hold_seconds);
			let mut last_client = (0i32, 0i32);

			while Instant::now() < deadline {
				let mut message: MSG = std::mem::zeroed();

				while PeekMessageW(&mut message, std::ptr::null_mut(), 0, 0, PM_REMOVE) != 0 {
					TranslateMessage(&message);
					DispatchMessageW(&message);
				}

				// A host calls `set_size` when it resizes the editor; the demo is its own
				// host, so it follows the window and the page scales its stage to fit.
				let mut client: RECT = std::mem::zeroed();
				GetClientRect(hwnd, &mut client);
				let client = (client.right - client.left, client.bottom - client.top);

				if client != last_client && client.0 > 0 && client.1 > 0 {
					last_client = client;
					editor.set_size(WindowSize::new(client.0 as f32, client.1 as f32))?;
				}

				std::thread::sleep(Duration::from_millis(8));
			}

			editor.close();
			DestroyWindow(hwnd);

			println!("crusher web editor closed");
			Ok(())
		}
	}
}
