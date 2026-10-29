//! iced as a GUI backend for `i_am_plugin`.
//!
//! `i_am_plugin` knows how to be a plug-in, and nothing about what draws its window.
//! This crate is the other half: [IcedEditor] implements the wrapper's
//! [Editor](i_am_plugin::Editor) trait with an iced program running in the host's
//! window through `iced_baseview`, and keeps the same message flow the wrapper used
//! to have built in -- the view updates itself from each message, and the message
//! goes on to the processor.
//!
//! A plug-in picks this backend by returning it from
//! [Plugin::editor](i_am_plugin::Plugin::editor):
//!
//! ```ignore
//! use i_am_dsp_iced::Processor as _; // for synced_view
//!
//! impl i_am_plugin::Plugin for MyProcessor {
//!     const HAS_GUI: bool = true;
//!
//!     fn editor(&self) -> Option<Box<dyn i_am_plugin::Editor<Self>>> {
//!         Some(Box::new(i_am_plugin_iced::IcedEditor::new(self.synced_view())))
//!     }
//!
//!     // ...
//! }
//! ```
//!
//! Nothing else in the crate names iced: an editor for another toolkit is a crate of
//! its own with the same shape.

use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use crossbeam_channel::Sender;
use i_am_dsp_iced::iced::time;
use i_am_dsp_iced::styles;
use i_am_dsp_iced::SyncedView;
use i_am_plugin::{Editor, EditorContext, EditorError, Message, Plugin, WindowSize};
use iced_baseview::{
	Element, IcedWindowSettings, PollSubNotifier, Program, Subscription, Task, Theme,
	baseview::{ParentWindowHandle, Window, dpi::{LogicalSize, PhysicalSize}},
	create_window, executor,
};

/// The editors this crate gives a processor.
///
/// The view is boxed so the program can be moved into the window's thread without
/// the plug-in wrapper knowing the concrete view type.
type BoxedView<M> = Box<dyn SyncedView<Message = M>>;

/// An iced editor: the host's window with an iced program running the plug-in's view.
///
/// It is created with the view the processor built, and opened by the wrapper when
/// the host asks for the GUI. A closed editor is thrown away; the wrapper asks the
/// processor for a new one the next time the window opens.
pub struct IcedEditor<P: Plugin> {
	view: Option<BoxedView<P::Message>>,
	window: Option<Window>,
	/// The host's scale factor, if it reported one.
	host_scale: Option<f64>,
	/// The zoom the running iced program renders at, shared with it.
	///
	/// It is the host's scale factor divided by the one the window reports, because
	/// iced_baseview multiplies the two and the host's is the one that counts.
	zoom: Arc<AtomicU32>,
	/// The logical size the plug-in asked for.
	size: WindowSize,
	/// A font to load when the GUI opens, if the plug-in has one.
	font: Option<&'static [u8]>,
}

impl<P: Plugin> IcedEditor<P> {
	/// Build an editor for a view.
	pub fn new<V>(view: V) -> Self
	where
		V: SyncedView<Message = P::Message> + 'static,
	{
		Self {
			view: Some(Box::new(view)),
			window: None,
			host_scale: None,
			zoom: Arc::new(AtomicU32::new(1.0f32.to_bits())),
			size: WindowSize::default(),
			font: None,
		}
	}

	/// Load a font when the GUI opens, so text drawn with the family it carries uses it.
	pub fn with_font(mut self, bytes: &'static [u8]) -> Self {
		self.font = Some(bytes);
		self
	}

	/// The iced window, if it is open.
	fn window_mut(&mut self) -> Result<&Window, EditorError> {
		self.window
			.as_ref()
			.ok_or_else(|| "the iced editor has no window".into())
	}

	/// Recompute the zoom from the host's scale factor and the window's own, and size
	/// the window to the physical pixels the host laid its parent out for.
	fn apply_scale(&mut self, scale: f64) -> Result<(), EditorError> {
		let Some(window) = &self.window else {
			return Ok(());
		};

		let window_scale = window.size().scale_factor;

		if window_scale > 0.0 && window_scale.is_finite() {
			self.zoom.store(((scale / window_scale) as f32).to_bits(), Ordering::Relaxed);
		}

		let physical_size = PhysicalSize::new(
			(self.size.width as f64 * scale).round() as u32,
			(self.size.height as f64 * scale).round() as u32,
		);

		window.resize(physical_size).map_err(iced_error)
	}
}

impl<P: Plugin> Editor<P> for IcedEditor<P> {
	fn open(&mut self, context: EditorContext<P>) -> Result<(), EditorError> {
		let EditorContext {
			parent,
			size,
			scale_factor,
			title,
			resizable,
			messages,
		} = context;

		self.size = size;
		self.host_scale = scale_factor;

		let Some(view) = self.view.take() else {
			return Err("the iced editor was opened more than once".into());
		};

		let zoom = Arc::new(AtomicU32::new(1.0f32.to_bits()));
		let zoom_for_build = Arc::clone(&zoom);
		let font = self.font;

		// `window_size` is the logical size the faceplate is laid out for. The physical
		// size is that times the host's scale factor, which is the same number the host
		// sized its parent window with; the settings fallback only matters to a platform
		// that cannot tell baseview which scale it is running at.
		let mut settings = IcedWindowSettings::new()
			.with_title(title)
			.with_size(LogicalSize::new(size.width as f64, size.height as f64))
			.with_fallback_scale_factor(scale_factor)
			.with_resizable(resizable)
			.with_always_redraw(true);
		settings.parent = Some(ParentWindowHandle::from_window(&parent));

		let (window, _message_sender) = create_window(
			settings,
			PollSubNotifier::new(),
			move |window_size| {
				if let Some(host_scale) = scale_factor {
					let window_scale = window_size.get().scale_factor;

					if window_scale > 0.0 && window_scale.is_finite() {
						zoom_for_build.store(
							((host_scale / window_scale) as f32).to_bits(),
							Ordering::Relaxed,
						);
					}
				}

				Ok(GuiProgram::<P> {
					state: RefCell::new(Some(GuiState {
						view,
						message_sender: messages,
					})),
					zoom: zoom_for_build,
					font,
				})
			},
			None,
		).map_err(iced_error)?;

		if let Some(host_scale) = scale_factor {
			// The host laid its parent out for `window_size * host_scale` pixels. A host
			// that manages the scale itself (a per-monitor-aware v1 parent) is given
			// exactly that; one whose window carries a different DPI asks for the same
			// size here.
			let physical_size = PhysicalSize::new(
				(size.width as f64 * host_scale).round() as u32,
				(size.height as f64 * host_scale).round() as u32,
			);
			window.resize(physical_size).map_err(iced_error)?;
		}

		window.show().map_err(iced_error)?;

		self.zoom = zoom;
		self.window = Some(window);
		Ok(())
	}

	fn show(&mut self) -> Result<(), EditorError> {
		self.window_mut()?.show().map_err(iced_error)
	}

	fn hide(&mut self) -> Result<(), EditorError> {
		self.window_mut()?.hide().map_err(iced_error)
	}

	fn close(&mut self) {
		if let Some(window) = self.window.take() {
			window.close();
		}
	}

	fn set_parent(&mut self, parent: i_am_plugin::ParentWindow) -> Result<(), EditorError> {
		let window = self.window_mut()?;
		let parent = ParentWindowHandle::from_window(&parent);

		window.set_parent(parent).map_err(iced_error)
	}

	fn set_scale_factor(&mut self, scale_factor: f64) -> Result<(), EditorError> {
		self.host_scale = Some(scale_factor);
		self.apply_scale(scale_factor)
	}

	fn set_size(&mut self, size: WindowSize) -> Result<(), EditorError> {
		self.size = size;

		if self.window.is_some() {
			// The host asked in physical pixels and the wrapper handed us the logical
			// size it stood for; sizing the window in those pixels keeps the parent and
			// the child on the same grid even when the window's own DPI differs.
			let scale = self.host_scale.unwrap_or(1.0);
			let physical_size = PhysicalSize::new(
				(size.width as f64 * scale).round() as u32,
				(size.height as f64 * scale).round() as u32,
			);

			self.window_mut()?.resize(physical_size).map_err(iced_error)?;
		}

		Ok(())
	}
}

/// Turn a baseview error into the kind the editor trait reports.
fn iced_error(error: iced_baseview::baseview::Error) -> EditorError {
	EditorError::new(error)
}

/// The state the GUI keeps between messages: the view it draws, and where the
/// messages the user makes are sent.
struct GuiState<P: Plugin> {
	view: BoxedView<P::Message>,
	message_sender: Sender<P::Message>,
}

/// The iced program the plugin GUI runs.
///
/// The state is held in the program until `boot` asks for it: `Program::boot` receives
/// a reference, so the one place the state can be moved out of is here.
struct GuiProgram<P: Plugin> {
	state: RefCell<Option<GuiState<P>>>,
	zoom: Arc<AtomicU32>,
	font: Option<&'static [u8]>,
}

impl<P: Plugin> Program for GuiProgram<P> {
	type State = GuiState<P>;
	type Message = P::Message;
	type Theme = Theme;
	type Renderer = iced_baseview::Renderer;
	type Executor = executor::Default;

	fn name() -> &'static str {
		"i_am_plugin_iced"
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

		let task = match self.font {
			Some(bytes) => iced_baseview::font::load(bytes)
				.map(|_| P::Message::tick(std::time::Instant::now())),
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
		Some(styles::theme())
	}

	fn update(&self, state: &mut Self::State, message: Self::Message) -> Task<Self::Message> {
		state.view.update(&message);
		let _ = state.message_sender.send(message);
		Task::none()
	}

	fn view<'a>(&self, state: &'a Self::State, _window: iced_baseview::window::Id) -> Element<'a, Self::Message> {
		state.view.view()
	}
}

/// A subscription that ticks the processor every 16ms.
///
/// It is the same cadence the iced application's own timer uses, so a view that
/// redraws itself from a tick behaves the same in the standalone demo and in a host.
fn timer<P: Plugin>() -> Subscription<P::Message> {
	time::every(Duration::from_millis(16)).map(P::Message::tick)
}
