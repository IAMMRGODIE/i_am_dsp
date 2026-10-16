//! main_thread

use crate::shared::*;
use crate::plugin::{Plugin};
use crate::latency::{LatencyWatch};

/// A struct to hold the parent window handle and the window handle for the plugin on main thread.
/// 
/// Public mainly for `export_clap` macro. At most case, you don't need to use it directly.
pub struct PluginMain<P: Plugin> {
	/// The window Options used to create the plugin GUI.
	pub(crate) window_size: Size,
	pub(crate) scale_factor: f64,
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
	pub(crate) parent: Option<RawWindowHandle>,
	pub(crate) handle: Option<WindowHandle<P::Message>>,
	pub(crate) message_receivers: Vec<Receiver<P::Message>>,
}

// impl<M: Message> Default for PluginGui<M> {
// 	fn default() -> Self {
// 		Self { 
// 			title: "Plugin GUI".to_string(),
// 			window_size: Size::new(640.0, 480.0),
// 			scale_factor: WindowScalePolicy::SystemScaleFactor,
// 			resize_hints: None,
// 			parent: None, 
// 			handle: None 
// 		}
// 	}
// }

// SAFETY: the handle is a copy of the parent window handle the host handed us in
// `set_parent`, and it stays valid for as long as the host keeps that window open.
unsafe impl<P: Plugin> HasRawWindowHandle for PluginMain<P> {
	fn raw_window_handle(&self) -> RawWindowHandle {
		self.parent.expect("No parent window provided")
	}
}

impl<P: Plugin> PluginMain<P> {
	/// Open the plugin GUI in current gui.
	pub fn open<App>(&mut self) -> Result<(), PluginError> 
	where 
		App: Application<Message = P::Message, Flags = (P::SyncedView, Sender<P::Message>)> + Send + 'static,
	{
		if self.parent.is_none() {
			return Err(PluginError::Message("No parent window provided"));
		}

		let settings = Settings { 
			window: WindowOpenOptions {
				size: self.window_size,
				scale: WindowScalePolicy::ScaleFactor(self.scale_factor),
				title: self.title.clone(),
			}, 
			iced_baseview: iced_baseview::IcedBaseviewSettings { 
				ignore_non_modifier_keys: false, 
				always_redraw: true, 
			}, 
			graphics_settings: Default::default(), 
			fonts: vec![],
		};

		let synced_view = self.processor.synced_view();
		let (message_sender, message_receiver) = crossbeam_channel::unbounded();
		self.message_receivers.push(message_receiver);

		self.handle = Some(open_parented::<App, Self>(self, (synced_view, message_sender), settings));
		Ok(())
	}

	/// Close the plugin GUI.
	pub fn close(&mut self) {
		if let Some(mut handle) = self.handle.take() {
			handle.close_window();
		}
	}

	/// Set the parent window handle.
	pub fn set_parent<T: HasRawWindowHandle>(&mut self, parent: T) {
		self.parent = Some(parent.raw_window_handle());
	}
}

/// A struct to hold a [`Processor`]'s [`SyncedView`] for displaying the plugin GUI.
/// 
/// Public mainly for `export_clap` macro. At most case, you don't need to use it directly.
pub struct GuiProgram<P: Plugin> {
	synced_view: P::SyncedView,
	message_sender: Sender<P::Message>,
	// _phantom: PhantomData<&'a mut P>,
}

impl<P: Plugin> Application for GuiProgram<P> 
where
	P::Message: Message + Debug,
{
	type Executor = iced_baseview::executor::Default;
	type Message = P::Message;
	type Flags = (P::SyncedView, Sender<P::Message>);
	type Theme = Theme;

	fn subscription(
		&self,
		_window_subs: &mut iced_baseview::WindowSubs<Self::Message>,
	) -> iced_baseview::futures::Subscription<Self::Message> {
		timer::<P>()
	}

	fn new(flags: Self::Flags) -> (Self, iced_baseview::Task<Self::Message>) {
		let (synced_view, message_sender) = flags;
		let program = GuiProgram {
			synced_view,
			message_sender,
		};

		let mut tasks = vec![];
		if let Some(bytes) = P::EMBEDDED_FONT {
			tasks.push(
				i_am_dsp_iced::iced::font::load(bytes)
					.map(|_| P::Message::tick(i_am_dsp_iced::iced::time::Instant::now())),
			);
		}
		let task = if tasks.is_empty() {
			iced_baseview::Task::none()
		}else {
			iced_baseview::Task::batch(tasks)
		};
		(program, task)
	}

	fn theme(&self) -> Self::Theme {
		i_am_dsp_iced::styles::theme()
	}

	fn update(&mut self, message: Self::Message) -> iced_baseview::Task<Self::Message> {
		self.synced_view.update(&message);
		let _ = self.message_sender.send(message);
		iced_baseview::Task::none()
	}

	fn view(&self) -> iced_baseview::Element<'_, Self::Message> {
		self.synced_view.view()
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

	fn destroy(&mut self) {}

	fn set_scale(&mut self, scale: f64) -> Result<(), clack_plugin::prelude::PluginError> {
		self.scale_factor = scale;
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
			width: (self.window_size.width * self.scale_factor) as u32,
			height: (self.window_size.height * self.scale_factor) as u32,
		})
	}

	fn set_size(&mut self, size: clack_extensions::gui::GuiSize) -> Result<(), clack_plugin::prelude::PluginError> {
		let scale_factor = self.scale_factor;
		self.window_size = Size::new(size.width as f64 / scale_factor, size.height as f64 / scale_factor);
		Ok(())
	}

	fn set_parent(&mut self, window: clack_extensions::gui::Window) -> Result<(), clack_plugin::prelude::PluginError> {
		self.set_parent(window);
		Ok(())
	}

	fn set_transient(&mut self, window: clack_extensions::gui::Window) -> Result<(), clack_plugin::prelude::PluginError> {
		self.set_parent(window);
		Ok(())
	}

	fn show(&mut self) -> Result<(), clack_plugin::prelude::PluginError> {
		self.open::<GuiProgram<P>>()
	}

	fn hide(&mut self) -> Result<(), clack_plugin::prelude::PluginError> {
		self.close();
		Ok(())
	}

	fn adjust_size(&mut self, size: clack_extensions::gui::GuiSize) -> Option<clack_extensions::gui::GuiSize> {
		self.window_size = Size::new(size.width as f64, size.height as f64);
		Some(size)
	}

	fn suggest_title(&mut self, title: &str) {
		self.title = title.to_string();
	}
}