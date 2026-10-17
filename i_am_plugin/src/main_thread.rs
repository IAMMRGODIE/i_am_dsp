//! main_thread

use std::cell::RefCell;

use crate::shared::*;
use crate::plugin::{Plugin};
use crate::latency::{LatencyWatch};

/// A struct to hold the parent window handle and the window handle for the plugin on main thread.
/// 
/// Public mainly for `export_clap` macro. At most case, you don't need to use it directly.
pub struct PluginMain<P: Plugin> {
	/// The window Options used to create the plugin GUI.
	pub(crate) window_size: Size,
	/// The scale factor the host told us about, and whether it ever did.
	///
	/// A host that never calls `set_scale` leaves the window to the platform's own
	/// DPI handling; one that does is taken at its word.
	pub(crate) scale_factor: f64,
	pub(crate) scale_set: bool,
	/// The title of the plugin GUI.
	pub(crate) title: String,
	/// The resize hints of the plugin GUI.
	/// 
	/// None for can't resize.
	pub(crate) resize_hints: Option<GuiResizeHints>,

	pub(crate) param_map: ParamMap,
	pub(crate) processor: Pin<Box<P>>,
	/// The host, kept so the plugin can tell it when the latency moves.
	///
	/// The handle is valid for as long as the plugin lives: `new_main_thread` receives it from
	/// the host and the host owns the plugin, so its lifetime outlasts this struct.
	pub(crate) host: Option<HostMainThreadHandle<'static>>,
	/// What the audio thread is running at, and what the host was last told.
	pub(crate) latency: LatencyWatch,
	/// The window the host handed us, kept until the GUI is opened.
	pub(crate) parent: Option<ParentWindowHandle>,
	/// The open GUI window, if it is open.
	pub(crate) handle: Option<Window>,
	/// The zoom the running iced program renders at, shared with it.
	///
	/// It is the host's scale factor divided by the one the window reports, because
	/// iced_baseview multiplies the two and the host's is the one that counts.
	pub(crate) zoom: Option<Arc<AtomicU32>>,
	pub(crate) message_receivers: Vec<Receiver<P::Message>>,
}

impl<P: Plugin> PluginMain<P> {
	/// Open the plugin GUI in the window the host provided.
	pub fn open(&mut self) -> Result<(), PluginError> 
	where
		P::Message: Message + Debug,
	{
		let Some(parent) = self.parent.clone() else {
			return Err(PluginError::Message("No parent window provided"));
		};

		let synced_view = self.processor.synced_view();
		let (message_sender, message_receiver) = crossbeam_channel::unbounded();

		// `window_size` is the logical size the faceplate is laid out for. The physical
		// size is that times the host's scale factor, which is the same number the host
		// sized its parent window with; the settings fallback only matters to a platform
		// that cannot tell baseview which scale it is running at.
		let mut settings = IcedWindowSettings::new()
			.with_title(self.title.clone())
			.with_size(LogicalSize::new(self.window_size.width as f64, self.window_size.height as f64))
			.with_fallback_scale_factor(self.scale_factor)
			.with_resizable(self.resize_hints.is_some())
			.with_always_redraw(true);
		settings.parent = Some(parent);

		// The zoom iced draws at. iced_baseview scales the viewport by the window's own
		// scale factor times this one, so dividing the host's factor by the window's
		// leaves the host's factor as the one the faceplate is drawn with.
		let zoom = Arc::new(AtomicU32::new(1.0f32.to_bits()));
		let zoom_for_build = Arc::clone(&zoom);
		// A host that never reported a scale leaves the window at the platform's own
		// DPI; the iced program then runs at its default zoom of 1.
		let host_scale = self.scale_set.then_some(self.scale_factor);

		let program = GuiProgram::<P> {
			state: RefCell::new(Some(GuiState::<P> {
				synced_view,
				message_sender,
			})),
			zoom: Arc::clone(&zoom),
		};

		let (window, _message_sender) = create_window(
			settings,
			PollSubNotifier::new(),
			move |window_size| {
				if let Some(host_scale) = host_scale {
					let window_scale = window_size.get().scale_factor;

					if window_scale > 0.0 && window_scale.is_finite() {
						zoom_for_build.store(
							((host_scale / window_scale) as f32).to_bits(),
							Ordering::Relaxed,
						);
					}
				}

				Ok(program)
			},
			None,
		).map_err(|error| PluginError::Error(Box::new(error)))?;

		if let Some(host_scale) = host_scale {
			// The host laid its parent out for `window_size * host_scale` pixels. A host
			// that manages the scale itself (a per-monitor-aware v1 parent) is given
			// exactly that; one whose window carries a different DPI asks for the same
			// size here.
			let physical_size = PhysicalSize::new(
				(self.window_size.width as f64 * host_scale).round() as u32,
				(self.window_size.height as f64 * host_scale).round() as u32,
			);
			window.resize(physical_size)
				.map_err(|error| PluginError::Error(Box::new(error)))?;
		}

		window.show().map_err(|error| PluginError::Error(Box::new(error)))?;

		// Only a GUI that is actually running has messages worth collecting; a failed
		// open leaves no receiver behind for it.
		self.message_receivers.push(message_receiver);
		self.zoom = Some(zoom);
		self.handle = Some(window);
		Ok(())
	}

	/// Destroy the plugin GUI and its state.
	///
	/// Hiding only leaves the window off screen; this is what the host calls
	/// `destroy` for, and it is the only place the iced program is thrown away.
	pub fn close(&mut self) {
		if let Some(window) = self.handle.take() {
			window.close();
		}

		self.zoom = None;
	}

	/// Hide the plugin GUI, keeping it ready to be shown again.
	pub fn hide(&mut self) -> Result<(), PluginError> {
		if let Some(window) = &self.handle {
			window.hide()
				.map_err(|error| PluginError::Error(Box::new(error)))?;
		}

		Ok(())
	}

	/// Set the parent window handle.
	///
	/// The handle belongs to the host and is copied out of it, so it may outlive the
	/// window it was taken from exactly as long as the host keeps that window open.
	pub fn set_parent(&mut self, parent: GuiWindow<'_>) -> Result<(), PluginError> {
		// SAFETY: the handle is borrowed only long enough to copy the platform window out
		// of it; `ParentWindowHandle` owns what it needs from then on.
		let handle = unsafe { parent.borrow_handle_unchecked() }
			.map_err(|_| PluginError::Message("Unsupported window API"))?;

		self.parent = Some(ParentWindowHandle::from_window(&handle));
		Ok(())
	}
}

/// The state the GUI keeps between messages: the view it draws, and where the
/// messages the user makes are sent.
pub struct GuiState<P: Plugin> {
	synced_view: P::SyncedView,
	message_sender: Sender<P::Message>,
}

/// The `iced` program the plugin GUI runs.
///
/// The state is held in the program until `boot` asks for it: `Program::boot` receives a
/// reference, so the one place the state can be moved out of is here.
pub struct GuiProgram<P: Plugin> {
	state: RefCell<Option<GuiState<P>>>,
	/// The scale iced draws the faceplate at, shared with the host side so that a
	/// changed host scale reaches the running program (see [`Program::scale_factor`]).
	zoom: Arc<AtomicU32>,
}

impl<P: Plugin> Program for GuiProgram<P> 
where
	P::Message: Message + Debug,
{
	type State = GuiState<P>;
	type Message = P::Message;
	type Theme = Theme;
	type Renderer = iced_baseview::Renderer;
	type Executor = executor::Default;

	fn name() -> &'static str {
		"i_am_plugin"
	}

	fn settings(&self) -> iced_baseview::Settings {
		iced_baseview::Settings::default()
	}

	fn window(&self) -> Option<iced_baseview::window::Settings> {
		None
	}

	fn boot(&self) -> (Self::State, Task<Self::Message>) {
		let state = self.state.borrow_mut()
			.take()
			.expect("the plugin GUI is booted exactly once");

		let task = match P::EMBEDDED_FONT {
			Some(bytes) => iced_baseview::font::load(bytes)
				.map(|_| P::Message::tick(i_am_dsp_iced::iced::time::Instant::now())),
			None => Task::none(),
		};

		(state, task)
	}

	fn subscription(&self, _state: &Self::State) -> Subscription<Self::Message> {
		timer::<P>()
	}

	fn scale_factor(&self, _state: &Self::State, _window: iced_baseview::window::Id) -> f32 {
		f32::from_bits(self.zoom.load(Ordering::Relaxed))
	}

	fn theme(&self, _state: &Self::State, _window: iced_baseview::window::Id) -> Option<Self::Theme> {
		Some(i_am_dsp_iced::styles::theme())
	}

	fn update(&self, state: &mut Self::State, message: Self::Message) -> Task<Self::Message> {
		state.synced_view.update(&message);
		let _ = state.message_sender.send(message);
		Task::none()
	}

	fn view<'a>(&self, state: &'a Self::State, _window: iced_baseview::window::Id) -> Element<'a, Self::Message> {
		state.synced_view.view()
	}
}

impl<P: Plugin> PluginGuiImpl for PluginMain<P> 
where 
	P::Message: Message + Debug,
{
	fn is_api_supported(&mut self, configuration: GuiConfiguration) -> bool {
		configuration.api_type
			== GuiApiType::default_for_current_platform().expect("Unsupported platform")
			&& !configuration.is_floating
	}

	fn get_preferred_api(&mut self) -> Option<GuiConfiguration<'_>> {
		Some(GuiConfiguration {
            api_type: GuiApiType::default_for_current_platform().expect("Unsupported platform"),
            is_floating: false,
        })
	}

	fn create(&mut self, _: GuiConfiguration) -> Result<(), clack_plugin::prelude::PluginError> {
		Ok(())	
	}

	fn destroy(&mut self) {
		self.close();
	}

	fn set_scale(&mut self, scale: f64) -> Result<(), clack_plugin::prelude::PluginError> {
		self.scale_factor = scale;
		self.scale_set = true;

		if let (Some(window), Some(zoom)) = (&self.handle, &self.zoom) {
			let window_scale = window.size().scale_factor;

			if window_scale > 0.0 && window_scale.is_finite() {
				zoom.store(((scale / window_scale) as f32).to_bits(), Ordering::Relaxed);
			}

			let physical_size = PhysicalSize::new(
				(self.window_size.width as f64 * scale).round() as u32,
				(self.window_size.height as f64 * scale).round() as u32,
			);
			window.resize(physical_size)
				.map_err(|error| PluginError::Error(Box::new(error)))?;
		}

		Ok(())
	}

	fn can_resize(&mut self) -> bool {
		self.resize_hints.is_some()
	}

	fn get_resize_hints(&mut self) -> Option<GuiResizeHints> {
		self.resize_hints
	}

	fn get_size(&mut self) -> Option<clack_extensions::gui::GuiSize> {
		Some(clack_extensions::gui::GuiSize {
			width: (self.window_size.width as f64 * self.scale_factor) as u32,
			height: (self.window_size.height as f64 * self.scale_factor) as u32,
		})
	}

	fn set_size(&mut self, size: clack_extensions::gui::GuiSize) -> Result<(), clack_plugin::prelude::PluginError> {
		let scale_factor = self.scale_factor;
		self.window_size = Size::new(
			(size.width as f64 / scale_factor) as f32,
			(size.height as f64 / scale_factor) as f32,
		);
		Ok(())
	}

	fn set_parent(&mut self, window: clack_extensions::gui::Window) -> Result<(), clack_plugin::prelude::PluginError> {
		PluginMain::set_parent(self, window)?;
		Ok(())
	}

	fn set_transient(&mut self, window: clack_extensions::gui::Window) -> Result<(), clack_plugin::prelude::PluginError> {
		PluginMain::set_parent(self, window)?;
		Ok(())
	}

	fn show(&mut self) -> Result<(), clack_plugin::prelude::PluginError> {
		if let Some(window) = &self.handle {
			return window.show()
				.map_err(|error| PluginError::Error(Box::new(error)));
		}

		self.open()
	}

	fn hide(&mut self) -> Result<(), clack_plugin::prelude::PluginError> {
		PluginMain::hide(self)
	}

	fn adjust_size(&mut self, size: clack_extensions::gui::GuiSize) -> Option<clack_extensions::gui::GuiSize> {
		self.window_size = Size::new(size.width as f32, size.height as f32);
		Some(size)
	}

	fn suggest_title(&mut self, title: &str) {
		self.title = title.to_string();
	}
}
