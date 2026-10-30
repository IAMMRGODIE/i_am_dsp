//! Does the Convolver's picker path really put a dialog on screen?
//!
//! The product opens its IR picker on a worker thread, with the DAW window as the
//! dialog's owner. This probe does the same against a plain Win32 window, then
//! looks for the dialog window (class `#32770`) in this process and closes it.
//!
//! Run: cargo run -p i_am_effects --example rfd_probe
//!
//! A small window opens; the file dialog should appear a moment later. The probe
//! prints whether it found it.

#[cfg(not(windows))]
fn main() {
	eprintln!("the probe is Windows-only");
}

#[cfg(windows)]
fn main() {
	imp::main();
}

#[cfg(windows)]
mod imp {
	use std::ffi::c_void;
	use std::time::{Duration, Instant};

	use raw_window_handle::{
		DisplayHandle, HandleError, HasDisplayHandle, HasWindowHandle, RawDisplayHandle,
		RawWindowHandle, WindowHandle, Win32WindowHandle, WindowsDisplayHandle,
	};
	use windows_sys::Win32::Foundation::HWND;
	use windows_sys::Win32::Graphics::Gdi::COLOR_WINDOW;
	use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
	use windows_sys::Win32::UI::WindowsAndMessaging::{
		CW_USEDEFAULT, CreateWindowExW, DefWindowProcW, DispatchMessageW, FindWindowW,
		GetWindowThreadProcessId, MSG, PM_REMOVE, PeekMessageW, PostMessageW, RegisterClassW,
		SW_SHOW, ShowWindow, TranslateMessage, WM_CLOSE, WNDCLASSW, WS_OVERLAPPEDWINDOW, WS_VISIBLE,
	};

	struct Owner(isize);

	impl HasWindowHandle for Owner {
		fn window_handle(&self) -> Result<WindowHandle<'_>, HandleError> {
			let hwnd = std::num::NonZeroIsize::new(self.0).ok_or(HandleError::Unavailable)?;
			let handle = RawWindowHandle::Win32(Win32WindowHandle::new(hwnd));

			// SAFETY: the window outlives the dialog.
			Ok(unsafe { WindowHandle::borrow_raw(handle) })
		}
	}

	impl HasDisplayHandle for Owner {
		fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
			// SAFETY: the Windows display handle carries no state.
			Ok(unsafe {
				DisplayHandle::borrow_raw(RawDisplayHandle::Windows(WindowsDisplayHandle::new()))
			})
		}
	}

	fn wide(text: &str) -> Vec<u16> {
		text.encode_utf16().chain(std::iter::once(0)).collect()
	}

	pub fn main() {
		unsafe {
			let instance = GetModuleHandleW(std::ptr::null());
			let class_name = wide("IAmRfdProbe");
			let title = wide("rfd probe");
			let class = WNDCLASSW {
				lpfnWndProc: Some(DefWindowProcW),
				hInstance: instance,
				hbrBackground: (COLOR_WINDOW as usize + 1) as *mut c_void,
				lpszClassName: class_name.as_ptr(),
				..Default::default()
			};

			if RegisterClassW(&class) == 0 {
				eprintln!("RegisterClassW failed");
				return;
			}

			let hwnd: HWND = CreateWindowExW(
				0,
				class_name.as_ptr(),
				title.as_ptr(),
				WS_OVERLAPPEDWINDOW | WS_VISIBLE,
				CW_USEDEFAULT,
				CW_USEDEFAULT,
				420,
				180,
				std::ptr::null_mut(),
				std::ptr::null_mut(),
				instance,
				std::ptr::null(),
			);

			if hwnd.is_null() {
				eprintln!("CreateWindowExW failed");
				return;
			}

			ShowWindow(hwnd, SW_SHOW);
			// Give the probe window the foreground so the dialog has a real owner.
			windows_sys::Win32::UI::WindowsAndMessaging::SetForegroundWindow(hwnd);

			let owner = Owner(hwnd as isize);

			std::thread::spawn(move || {
				std::thread::sleep(Duration::from_millis(200));

				let dialog = rfd::FileDialog::new()
					.add_filter("Impulse response", &["wav"])
					.set_parent(&owner);

				println!("picker returned: {:?}", dialog.pick_file());
			});

			let deadline = Instant::now() + Duration::from_secs(6);
			let mut found = false;

			while Instant::now() < deadline {
				let mut message: MSG = std::mem::zeroed();

				while PeekMessageW(&mut message, std::ptr::null_mut(), 0, 0, PM_REMOVE) != 0 {
					TranslateMessage(&message);
					DispatchMessageW(&message);
				}

				let dialog = FindWindowW(wide("#32770").as_ptr(), std::ptr::null());

				if !dialog.is_null() {
					let mut pid = 0;
					GetWindowThreadProcessId(dialog, &mut pid);

					if pid == std::process::id() {
						println!("dialog found; closing it");
						PostMessageW(dialog, WM_CLOSE, 0, 0);
						found = true;
						break;
					}
				}

				std::thread::sleep(Duration::from_millis(50));
			}

			if !found {
				println!("no dialog appeared");
			}

			// Let the worker's picker return after the close, then leave.
			std::thread::sleep(Duration::from_millis(500));
			windows_sys::Win32::UI::WindowsAndMessaging::DestroyWindow(hwnd);
		}
	}
}
