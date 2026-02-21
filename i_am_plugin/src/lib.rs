//! helper library for `i_am_dsp_iced` crate to export CLAP plugin based on `clack`
//! 
//! # Minimal Example
//! 
//! ```ignore
//! // first, build a `Processor` based on `i_am_dsp_iced`
//! struct MyProcessor;
//! 
//! impl Processor for MyProcessor {
//!     // see more in `i_am_dsp_iced` crate
//!     //...
//! }
//! 
//! // The imply `Plugin` trait for the `Processor`
//! impl Plugin for MyProcessor {
//!     const DESCRIPTOR: Descriptor = Descriptor::new("i.am.example.processor", "My Processor");
//!     
//!     fn new() -> Self {
//!         MyProcessor
//!     }
//!     
//!     fn window_options() -> WindowOptions {
//!         WindowOptions::default()
//!     }
//! 
//!     fn param_map(&self) -> ParamMap {
//!         ParamMap::default()
//!     }
//! }
//! 
//! // Finally, export the plugin as a CLAP plugin
//! export_clap!(MyProcessor);
//! 
//! // Additionaly, you can impl `PluginAuExt` to prepare the plugin for AU support
//! //
//! // You should check AU's manual for meaning and usage of `AU_TYPE` and `AU_SUBTYPE` fields.
//! impl PluginAuExt for MyProcessor {
//!     const AU_TYPE: [u8; 4] = *b"aufx";
//!     const AU_SUBTYPE: [u8; 4] = *b"mypr";
//! }
//! 
//! // And use `clap_wrapper` crate to prepare for VST3 and AUv2 support
//! // 
//! // You should check VST3 or AUv2's manual for packing your clap plugin as a VST3 or AUv2 plugin.
//! //
//! // Note: `clap_wrapper` crate is not included in `i_am_plugin` crate. 
//! // You need to add it to your `Cargo.toml` file.
//! //
//! // Also, you should always use `export_clap` macro even if you dont need to export clap plugin.
//! clap_wrapper::export_auv2!();
//! clap_wrapper::export_vst3!();
//! ```
//! 
//! Do **NOT** forget to add following settings to your `Cargo.toml` file:
//! 
//! ```toml
//! [lib]
//! crate-type = ["cdylib"]
//! ```
//! 
//! Finally, you can build the plugin with `cargo build --release` and rename the suffix to `.clap`.
//! Then you should be able to load the plugin in your DAW.

use std::{any::Any, ffi::CStr, fmt::Debug, io::{Read, Write}, pin::Pin, slice::from_raw_parts, sync::{Arc, RwLock}};

use clack_extensions::{
	audio_ports::{AudioPortFlags, AudioPortInfo, AudioPortType, PluginAudioPorts, PluginAudioPortsImpl}, 
	clap_wrapper::{
		auv2::PluginFactoryAsAUv2Impl, 
		vst3::{PluginAsVST3, PluginAsVST3Impl, PluginFactoryAsVST3Impl, PluginInfoAsVST3}
	}, 
	gui::{GuiApiType, GuiConfiguration, GuiResizeHints, PluginGui, PluginGuiImpl}, 
	latency::{PluginLatency, PluginLatencyImpl}, 
	note_ports::{NoteDialect, NoteDialects, NotePortInfo, PluginNotePorts, PluginNotePortsImpl}, 
	params::{ParamInfo, ParamInfoFlags, PluginAudioProcessorParams, PluginMainThreadParams, PluginParams}, 
	state::{PluginState, PluginStateImpl}
};
use clack_plugin::{
	entry::DefaultPluginFactory, 
	events::{Event, Match, Pckn, 
		event_types::{NoteChokeEvent, NoteOffEvent, NoteOnEvent, ParamModEvent, ParamValueEvent, TransportEvent}, 
		spaces::CoreEventSpace
	}, 
	plugin::{PluginAudioProcessor, PluginError, PluginMainThread}, 
	prelude::{OutputEvents, SampleType}, 
	process::{Audio, Events, Process, ProcessStatus}
};
use crossbeam_channel::{Receiver, Sender};
use i_am_dsp::{ProcessContext, ProcessInfos, prelude::{AtomicValue, ParamMap, SetValue, from_binary, to_binary}};
use i_am_dsp_iced::{Message, Processor, SyncedView, timer};
use iced_baseview::{Application, Settings, Theme, baseview::{Size, WindowOpenOptions, WindowScalePolicy}, open_parented, window::WindowHandle};
use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};

extern crate self as i_am_plugin;

#[doc(hidden)]
pub use clack_plugin::{clack_export_entry, entry::SinglePluginEntry};
use xxhash_rust::const_xxh3::xxh3_128;

/// A struct to hold a plugin's audio port.
pub struct AudioPort {
	/// The name of the audio port.
	pub name: &'static str,
}

/// A struct to hold a plugin's MIDI port.
pub struct MidiPort {
	/// The name of the Midi port.
	pub name: &'static str,
}

/// The plugin's descriptor.
pub struct Descriptor {
	pub id: &'static str,
	pub name: &'static str,
	pub vendor: Option<&'static str>,
	pub url: Option<&'static str>,
	pub manual_url: Option<&'static str>,
	pub support_url: Option<&'static str>,
	pub version: Option<&'static str>,
	pub description: Option<&'static str>,
	pub tags: &'static [Tag],
}

/// A tag for the plugin.
pub enum Tag {
	Instrument,
	AudioEffect,
	NoteEffect,
	Analyzer,
	Synthesizer,
	Sampler,
	Drum,
	DrumMachine,
	Filter,
	Phaser,
	Equalizer,
	Deesser,
	PhaseVocoder,
	Granular,
	FrequencyShifter,
	PitchShifter,
	Distortion,
	TransientShaper,
	Compressor,
	Limiter,
}

impl Tag {
	fn as_cstr(&self) -> &CStr {
		match self {
			Self::Instrument => clack_plugin::plugin::features::INSTRUMENT,
			Self::AudioEffect => clack_plugin::plugin::features::AUDIO_EFFECT,
			Self::NoteEffect => clack_plugin::plugin::features::NOTE_EFFECT,
			Self::Analyzer => clack_plugin::plugin::features::ANALYZER,
			Self::Synthesizer => clack_plugin::plugin::features::SYNTHESIZER,
			Self::Sampler => clack_plugin::plugin::features::SAMPLER,
			Self::Drum => clack_plugin::plugin::features::DRUM,
			Self::DrumMachine => clack_plugin::plugin::features::DRUM_MACHINE,
			Self::Filter => clack_plugin::plugin::features::FILTER,
			Self::Phaser => clack_plugin::plugin::features::PHASER,
			Self::Equalizer => clack_plugin::plugin::features::EQUALIZER,
			Self::Deesser => clack_plugin::plugin::features::DEESSER,
			Self::PhaseVocoder => clack_plugin::plugin::features::PHASE_VOCODER,
			Self::Granular => clack_plugin::plugin::features::GRANULAR,
			Self::FrequencyShifter => clack_plugin::plugin::features::FREQUENCY_SHIFTER,
			Self::PitchShifter => clack_plugin::plugin::features::PITCH_SHIFTER,
			Self::Distortion => clack_plugin::plugin::features::DISTORTION,
			Self::TransientShaper => clack_plugin::plugin::features::TRANSIENT_SHAPER,
			Self::Compressor => clack_plugin::plugin::features::COMPRESSOR,
			Self::Limiter => clack_plugin::plugin::features::LIMITER,
		}
	}
}

impl Descriptor {
	/// Create a new descriptor with the given values.
	pub const fn new(
		id: &'static str,
		name: &'static str,
	) -> Self {
		Self {
			id,
			name,
			vendor: None,
			url: None,
			manual_url: None,
			support_url: None,
			version: None,
			description: None,
			tags: &[],
		}
	}

	/// Set the vendor of the plugin.
	pub const fn with_vendor(
		self,
		vendor: &'static str,
	) -> Self {
		Self {
			vendor: Some(vendor),
			..self
		}
	}

	/// Set the URL of the plugin's website.
	pub const fn with_url(
		self,
		url: &'static str,
	) -> Self {
		Self {
			url: Some(url),
			..self
		}
	}

	/// Set the URL of the plugin's manual.
	pub const fn with_manual_url(
		self,
		manual_url: &'static str,
	) -> Self {
		Self {
			manual_url: Some(manual_url),
			..self
		}
	}

	/// Set the URL of the plugin's support.
	pub const fn with_support_url(
		self,
		support_url: &'static str,
	) -> Self {
		Self {
			support_url: Some(support_url),
			..self
		}
	}

	/// Set the version of the plugin.
	pub const fn with_version(
		self,
		version: &'static str,
	) -> Self {
		Self {
			version: Some(version),
			..self
		}
	}

	/// Set the description of the plugin.
	pub const fn with_description(
		self,
		description: &'static str,
	) -> Self {
		Self {
			description: Some(description),
			..self
		}
	}

	/// Set the tags of the plugin.
	pub const fn with_tags(
		self,
		tags: &'static [Tag],
	) -> Self {
		Self {
			tags,
			..self
		}
	}
}

/// A trait for plugins that can be used with the `i_am_dsp_iced` crate.
/// 
/// Will automatically generate the id for audio ports.
/// 
/// Currently dynamical audio ports are not supported.
pub trait Plugin: Processor {
	/// The plugin's descriptor.
	const DESCRIPTOR: Descriptor;
	/// The input ports of the plugin.
	const INPUT_PORTS: &'static [AudioPort] = &[AudioPort {
		name: "Main",
	}];
	/// The output ports of the plugin.
	const OUTPUT_PORTS: &'static [AudioPort] = &[AudioPort {
		name: "Main",
	}];
	/// The MIDI input ports of the plugin.
	const INPUT_MIDI_PORTS: &'static [MidiPort] = &[];
	/// The MIDI output ports of the plugin.
	const OUTPUT_MIDI_PORTS: &'static [MidiPort] = &[];

	/// Create a new instance of the plugin.
	fn new() -> Self;

	/// Get window options for the plugin GUI.
	fn window_options() -> WindowOptions;

	/// Get the plugin's parameter map.
	fn param_map(&self) -> ParamMap;
}

/// A trait for plugin to export as Auv2 plugin.
pub trait PluginAuExt: Plugin {
	/// The AU type of the plugin.
	const AU_TYPE: [u8; 4];
	/// The AU subtype of the plugin.
	const AU_SUBTYPE: [u8; 4];
}

/// The window options for the plugin GUI.
#[derive(Debug, Clone)]
pub struct WindowOptions {
	/// The window Options used to create the plugin GUI.
	pub window_size: Size,
	// /// The window scale policy used to create the plugin GUI.
	// pub scale_factor: WindowScalePolicy,
	/// The title of the plugin GUI.
	pub title: String,
	/// The resize hints of the plugin GUI.
	/// 
	/// None for can't resize.
	pub resize_hints: Option<GuiResizeHints>,
}

impl WindowOptions {
	/// Create a new `WindowOptions` with default values.
	pub fn new() -> Self {
		Self::default()
	}

	/// Set the window size of the plugin GUI.
	pub fn with_size(self, size: (f64, f64)) -> Self {
		Self {
			window_size: Size::new(size.0, size.1),
			..self
		}
	}

	/* /// Set the window size of the plugin GUI to the system scale factor.
	pub fn with_system_scale_factor(self) -> Self {
		Self {
			scale_factor: WindowScalePolicy::SystemScaleFactor,
			..self
		}
	}

	/// Set the window scale factor of the plugin GUI.
	pub fn with_scale_factor(self, scale_factor: f64) -> Self {
		Self {
			scale_factor: WindowScalePolicy::ScaleFactor(scale_factor),
			..self
		}
	} */

	/// Set the title of the plugin GUI.
	pub fn with_title(self, title: &str) -> Self {
		Self {
			title: title.to_string(),
			..self
		}
	}
}

impl Default for WindowOptions {
	fn default() -> Self {
		Self {
			window_size: Size::new(640.0, 480.0),
			// scale_factor: WindowScalePolicy::SystemScaleFactor,
			title: "Plugin GUI".to_string(),
			resize_hints: None,
		}
	}
}

/// A struct to hold the parent window handle and the window handle for the plugin on main thread.
/// 
/// Public mainly for `export_clap` macro. At most case, you don't need to use it directly.
pub struct PluginMain<P: Plugin> {
	/// The window Options used to create the plugin GUI.
	window_size: Size,
	scale_factor: f64,
	/// The title of the plugin GUI.
	title: String,
	/// The resize hints of the plugin GUI.
	/// 
	/// None for can't resize.
	resize_hints: Option<GuiResizeHints>,

	param_map: ParamMap,
	processor: Pin<Box<P>>,
	parent: Option<RawWindowHandle>,
	handle: Option<WindowHandle<P::Message>>,
	message_receivers: Vec<Receiver<P::Message>>,
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
			// gui,
			// _phantom: PhantomData,
		};
		(program, iced_baseview::Task::none())
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

/// A struct to hold a [`Processor`] for the plugin audio thread.
/// 
/// Public mainly for `export_clap` macro. At most case, you don't need to use it directly.
pub struct AudioProcessor<'a, P: Plugin> {
	processor: usize,
	temp_buffer_1: Vec<&'a [usize]>,
	temp_buffer_2: Vec<bool>,
	temp_buffer_3: Vec<[f32; 2]>,
	sample_rate: usize,
	last_timer: Option<u64>,
	events_buffer: Arc<RwLock<Vec<i_am_dsp::NoteEvent>>>,
	event_sender: Sender<(usize, i_am_dsp::NoteEvent)>,
	event_receiver: Receiver<(usize, i_am_dsp::NoteEvent)>,
	param_map: ParamMap,
	_phantom: std::marker::PhantomData<P>,
}

impl<P: Plugin> PluginMainThread<'_, ()> for PluginMain<P> {
	fn on_main_thread(&mut self) {
		self.message_receivers.retain_mut(|inner| {
			loop {
				match inner.try_recv() {
					Ok(message) => {
						self.processor.on_message(message);
					}
					Err(crossbeam_channel::TryRecvError::Empty) => {
						break;
					}
					Err(crossbeam_channel::TryRecvError::Disconnected) => {
						return false;
					}
				}
			}
			true
		});
	}
}

/// The context for the CLAP plugin.
/// 
/// Public mainly for `export_clap` macro. At most case, you don't need to use it directly.
pub struct ClapContext {
	current_event: usize,
	events_buffer: Arc<RwLock<Vec<i_am_dsp::NoteEvent>>>,
	info: Option<ProcessInfos>,
	current_sample: usize,
	event_sender: Sender<(usize, i_am_dsp::NoteEvent)>,
}

impl ProcessContext for ClapContext {
	fn infos(&self) -> &ProcessInfos {
		if let Some(inner) = &self.info {
			inner
		}else {
			().infos()
		}
	}

	fn events(&self) -> &[i_am_dsp::NoteEvent] {
		let events_buffer = self.events_buffer.try_read().expect("cannot read events");
		unsafe {
			from_raw_parts(events_buffer.as_ptr(), events_buffer.len())
		}
	}

	fn next_event(&mut self) -> Option<i_am_dsp::NoteEvent> {
		let events_buffer = self.events_buffer.try_read().expect("cannot read events");

		if self.current_event >= events_buffer.len() {
			return None;
		}

		let event = events_buffer[self.current_event].clone();
		self.current_event += 1;
		Some(event)
	}

	fn send_event(&mut self, event: i_am_dsp::NoteEvent) {
		self.event_sender.send((self.current_sample, event)).unwrap();
	}

	fn should_stop(&self) -> bool {
		false
	}
}

fn convert_transport(inner: &TransportEvent, playing: bool, sample_rate: usize) -> i_am_dsp::ProcessInfos {
	let mut info = ProcessInfos::default();
	info.sample_rate = sample_rate;
	info.playing = playing;
	info.tempo = Some(inner.tempo as f32);
	info.current_bar_number = Some(inner.bar_number as f32);
	info.time_signature = Some((
		inner.time_signature_numerator as usize, 
		inner.time_signature_denominator as usize
	));
	info.current_time = inner.song_pos_seconds.to_float() as f32;
	info
}

impl ClapContext {
	fn from_clap(
		sample_rate: usize,
		last_timer: Option<u64>,
		process: Process,
		events_buffer: Arc<RwLock<Vec<i_am_dsp::NoteEvent>>>,
		event_sender: Sender<(usize, i_am_dsp::NoteEvent)>,
	) -> Self {
		let info = if let Some(inner) = process.transport {
			let info = convert_transport(
				inner, 
				last_timer != process.steady_time || process.steady_time.is_none(), 
				sample_rate
			);
			Some(info)
		}else {
			None
		};
		Self {
			current_event: 0,
			event_sender,
			// process, 
			events_buffer,
			current_sample: 0,
			info 
		}
	}
}

impl<'a, P: Plugin> PluginAudioProcessor<'a, (), PluginMain<P>> for AudioProcessor<'a, P> {
	fn activate(
		_: clack_plugin::prelude::HostAudioProcessorHandle<'a>,
		main_thread: &mut PluginMain<P>,
		_: &'a (),
		config: clack_plugin::prelude::PluginAudioConfiguration,
	) -> Result<Self, PluginError> {
		let ptr = main_thread.processor.as_ref().get_ref() as *const P as usize;
		let (event_sender, event_receiver) = crossbeam_channel::unbounded();

		Ok(AudioProcessor { 
			processor: ptr,
			temp_buffer_1: vec![],
			temp_buffer_2: vec![],
			temp_buffer_3: vec![],
			sample_rate: config.sample_rate as usize,
			last_timer: None,
			events_buffer: Default::default(),
			event_sender,
			event_receiver,
			param_map: main_thread.processor.param_map(),
			_phantom: std::marker::PhantomData,
		})
	}

	fn process(
		&mut self,
		process: Process,
		mut audio: Audio,
		events: Events,
	) -> Result<ProcessStatus, PluginError> {
		let processor = unsafe { &mut *(self.processor as *mut P) };
		self.temp_buffer_1.clear();
		self.temp_buffer_2.clear();
		self.temp_buffer_3.clear();
		let mut context = Some(ClapContext::from_clap(
			self.sample_rate, 
			self.last_timer, 
			process,
			self.events_buffer.clone(), 
			self.event_sender.clone()
		));
		let mut output_temp = [0.0; 2];

		let buffer_size = audio.frames_count() as usize;

		// let mut min_buffer_size = output_port.frames_count() as usize;
		let input_ports = audio.input_ports();
		for input_port in input_ports {
			match input_port.channels()? {
				SampleType::F32(inner) | SampleType::Both(inner, _) => {
					let data_len = inner.frames_count() as usize;
					let ptr = inner.raw_data().as_ptr() as *const usize;
					unsafe {
						self.temp_buffer_1.push(from_raw_parts(ptr, data_len));
					}
					self.temp_buffer_2.push(false);
					// min_buffer_size = min_buffer_size.min(data_len);
				},
				SampleType::F64(inner) => {
					let data_len = inner.frames_count() as usize;
					let ptr = inner.raw_data().as_ptr() as *const usize;
					unsafe {
						self.temp_buffer_1.push(from_raw_parts(ptr, data_len));
					}
					self.temp_buffer_2.push(true);
					// min_buffer_size = min_buffer_size.min(data_len);
				}
			}
		}

		for _ in 0..self.temp_buffer_1.len() {
			self.temp_buffer_3.push([0.0, 0.0]);
		}

		let mut next_event_sample = events.input.batch().next().map(|batch| batch.first_sample()).unwrap_or(0);
		let mut bacthced = events.input.batch();

		for i in 0..buffer_size {
			if i >= next_event_sample && let (Some(batch), Some(ctx)) = (bacthced.next(), &mut context) {
				let mut events_buffer = ctx.events_buffer
					.try_write()
					.map_err(|_| PluginError::Message("cannot read events"))?;

				events_buffer.clear();
				for event in batch.events() {
					match event.as_core_event() {
						Some(CoreEventSpace::NoteOn(note)) => {
							events_buffer.push(i_am_dsp::NoteEvent::NoteOn { 
								time: note.time() as usize, 
								channel: note.pckn().channel.into_specific().unwrap_or_default() as u8, 
								note: note.pckn().key.into_specific().unwrap_or_default() as u8, 
								velocity: note.velocity() as f32, 
							});
						},
						Some(CoreEventSpace::NoteOff(note)) => {
							events_buffer.push(i_am_dsp::NoteEvent::NoteOff { 
								time: note.time() as usize, 
								channel: note.pckn().channel.into_specific().unwrap_or_default() as u8, 
								note: note.pckn().key.into_specific().unwrap_or_default() as u8, 
								velocity: note.velocity() as f32, 
							});
						},
						Some(CoreEventSpace::NoteChoke(note)) => {
							events_buffer.push(i_am_dsp::NoteEvent::Stop { 
								time: note.time() as usize, 
								channel: note.pckn().channel.into_specific().unwrap_or_default() as u8, 
								note: note.pckn().key.into_specific().unwrap_or_default() as u8, 
							});
						},
						Some(CoreEventSpace::NoteEnd(note)) => {
							events_buffer.push(i_am_dsp::NoteEvent::NoteOff { 
								time: note.time() as usize, 
								channel: note.pckn().channel.into_specific().unwrap_or_default() as u8, 
								note: note.pckn().key.into_specific().unwrap_or_default() as u8, 
								velocity: 1.0, 
							});
						},

						Some(CoreEventSpace::ParamValue(param)) => {
							handle_param_value_event(param, &self.param_map);
						},
						Some(CoreEventSpace::ParamMod(param)) => {
							handle_param_mod_event(param, &self.param_map);
						},
						Some(CoreEventSpace::ParamGestureBegin(_)) => {},
						Some(CoreEventSpace::ParamGestureEnd(_)) => {},
						Some(CoreEventSpace::Transport(param)) => {
							let info = convert_transport(
								param, 
								self.last_timer != process.steady_time || process.steady_time.is_none(), 
								self.sample_rate
							);
							ctx.info = Some(info);
						},
						
						Some(CoreEventSpace::Midi(_)) |
						Some(CoreEventSpace::Midi2(_)) |
						Some(CoreEventSpace::MidiSysEx(_)) |
						Some(CoreEventSpace::NoteExpression(_)) |
						None => {}
					}
				}
				ctx.current_event = 0;
				next_event_sample = batch.next_batch_first_sample().unwrap_or(buffer_size);
			}

			for (j, buffer) in self.temp_buffer_1.iter().enumerate() {
				for channel in 0..2 {
					if channel >= buffer.len() {
						continue;
					}

					unsafe {
						if self.temp_buffer_2[j] {
							let ptr = buffer[channel] as *const f64;
							let array = from_raw_parts(ptr, buffer_size);
							self.temp_buffer_3[j][channel] = array[i] as f32;
							if j == 0 {
								output_temp[channel] = array[i] as f32;
							}
						}else {
							let ptr = buffer[channel] as *const f32;
							let array = from_raw_parts(ptr, buffer_size);
							self.temp_buffer_3[j][channel] = array[i];

							if j == 0 {
								output_temp[channel] = array[i];
							}
						}
					}
				}
			}
			let mut context_took = context.take().unwrap();
			context_took.current_sample = i;
			let mut context_in = Box::new(context_took) as Box<dyn ProcessContext>;

			processor.process(&mut output_temp, &self.temp_buffer_3, &mut context_in);

			let context_back = context_in as Box<dyn Any>;
			let context_back = Box::<dyn std::any::Any>::downcast::<ClapContext>(context_back).unwrap();
			context = Some(*context_back);

			for mut output_ports in audio.output_ports() {
				match output_ports.channels()? {
					SampleType::F32(inner) | SampleType::Both(inner, _) => {
						for (channel, ptr) in inner.raw_data().iter().enumerate() {
							if channel >= 2 {
								break;
							}
							unsafe {
								let to_write = ptr.add(i);
								*to_write = output_temp[channel];
							}
						}
					},
					SampleType::F64(inner) => {
						for (channel, ptr) in inner.raw_data().iter().enumerate() {
							if channel >= 2 {
								break;
							}
							unsafe {
								let to_write = ptr.add(i);
								*to_write = output_temp[channel] as f64;
							}
						}
					}
				}
			}
		}

		for (offset, event) in self.event_receiver.try_iter() {
			note_event_to_clap(events.output, offset + process.steady_time.unwrap_or_default() as usize, event)?;
		}

		sync_params(events.output, &self.param_map);

		self.last_timer = process.steady_time;

		Ok(ProcessStatus::Continue)
	}
}

fn note_event_to_clap<'a>(to_send: &mut OutputEvents<'a>, time_stamp: usize, note_event: i_am_dsp::NoteEvent) -> Result<(), PluginError> {
	match note_event {
		i_am_dsp::NoteEvent::NoteOn { time, channel, note, velocity } => {
			to_send.try_push(NoteOnEvent::new(
				time as u32,
				Pckn::new(Match::All, channel as u16, note as u16, Match::All),
				velocity as f64,
			))?;
		},
		i_am_dsp::NoteEvent::ImmediateStop => {
			to_send.try_push(NoteChokeEvent::new(
				time_stamp as u32,
				Pckn::new(Match::All, Match::All, Match::All, Match::All),
			))?;
		}
		i_am_dsp::NoteEvent::MidiCC { .. } => {},
		i_am_dsp::NoteEvent::NoteOff { time, channel, note, velocity } => {
			to_send.try_push(NoteOffEvent::new(
				time as u32,
				Pckn::new(Match::All, channel as u16, note as u16, Match::All),
				velocity as f64,
			))?;
		},
		i_am_dsp::NoteEvent::Stop { time, channel, note } => {
			to_send.try_push(NoteOffEvent::new(
				time as u32,
				Pckn::new(Match::All, channel as u16, note as u16, Match::All),
				0.0,
			))?;
		}
	}

	Ok(())
}

impl<P: Plugin> PluginStateImpl for PluginMain<P> {
	fn load(&mut self, input: &mut clack_plugin::stream::InputStream) -> Result<(), PluginError> {
		let mut buf = vec![];
		input.read_to_end(&mut buf)?;
		let params: Vec<i_am_dsp::prelude::Parameter> = from_binary(buf)?;
		let processor = unsafe {
			self.processor.as_mut().get_unchecked_mut()
		};

		for param in params {
			if param.value.is_serialized() {
				processor.set_parameter(&param.identifier, param.value.to_set_value());
			}else {
				self.param_map.set(&param.identifier, param.value.to_set_value(), std::sync::atomic::Ordering::SeqCst);
			}
		}
		Ok(())
	}
	
	fn save(&mut self, output: &mut clack_plugin::stream::OutputStream) -> Result<(), PluginError> {
		let params = self.processor.get_parameters();
		let binary = to_binary(&params)?;
		output.write_all(&binary)?;
		Ok(())
	}
}

impl<P: Plugin> PluginMainThreadParams for PluginMain<P> {
	fn count(&mut self) -> u32 {
		self.param_map.len() as u32
	}

	fn get_info(&mut self, param_index: u32, info: &mut clack_extensions::params::ParamInfoWriter) {
		let index = param_index as usize;
		let Some(param) = self.param_map.get_by_index(index) else {
			return;
		};

		let flags = if param.is_float() {
			ParamInfoFlags::IS_AUTOMATABLE
		}else {
			ParamInfoFlags::IS_AUTOMATABLE | ParamInfoFlags::IS_ENUM
		};

		let (min_value, max_value) = match &*param {
			AtomicValue::Bool { .. } => (0.0, 1.0),
			AtomicValue::Nothing => (0.0, 0.0),
			AtomicValue::Float { range, .. } => (*range.start() as f64, *range.end() as f64),
			AtomicValue::Int { range, .. } => (*range.start() as f64, *range.end() as f64),
			_ => (0.0, 0.0),
		};

		info.set(&ParamInfo {
			id: param_index.into(),
			flags,
			cookie: Default::default(),
			name: self.param_map.query_param_id(index).unwrap().as_bytes(),
			module: Default::default(),
			min_value,
			max_value,
			default_value: min_value,
		});
	}

	fn get_value(&mut self, param_id: clack_plugin::prelude::ClapId) -> Option<f64> {
		let index = param_id.get() as usize;
		let param = self.param_map.get_by_index(index)?;
		let current = param.load(std::sync::atomic::Ordering::SeqCst);
		match current {
			SetValue::Bool(val) => if val { Some(1.0) } else { Some(0.0) },
			SetValue::Int(val) => Some(val as f64),
			SetValue::Float(val) => Some(val as f64),
			SetValue::Nothing => None,
			SetValue::Serialized(_) => None,
		}
	}

	fn value_to_text(
		&mut self,
		param_id: clack_plugin::prelude::ClapId,
		value: f64,
		writer: &mut clack_extensions::params::ParamDisplayWriter,
	) -> core::fmt::Result {
		use std::fmt::Write;

		if param_id.get() as usize >= self.param_map.len() {
			Err(std::fmt::Error)
		}else {
			write!(writer, "{:.2}", value)
		}

	}

	fn text_to_value(&mut self, param_id: clack_plugin::prelude::ClapId, text: &std::ffi::CStr) -> Option<f64> {
		if param_id.get() >= self.param_map.len() as u32 {
			return None
		}
		
		let text = text.to_str().ok()?;
		let value = text.parse::<f64>().ok()?;
		Some(value)
	}

	fn flush(
		&mut self,
		input_parameter_changes: &clack_plugin::prelude::InputEvents,
		output_parameter_changes: &mut OutputEvents,
	) {
		flush(input_parameter_changes, output_parameter_changes, &self.param_map);
	}
}

fn handle_param_value_event(
	param: &ParamValueEvent,
	param_map: &ParamMap,
) {
	if let Some(param_id) = param.param_id() {
		let id = param_id.get() as usize;
		let value = param.value() as f32;
		if let Some(inner) = param_map.get_by_index(id) {
			if inner.is_int() {
				inner.store(value as i32, std::sync::atomic::Ordering::SeqCst);
				inner.set_changed(false, std::sync::atomic::Ordering::SeqCst);
			}else if inner.is_float() {
				inner.store(value, std::sync::atomic::Ordering::SeqCst);
				inner.set_changed(false, std::sync::atomic::Ordering::SeqCst);
			}else if inner.is_bool() {
				inner.store(value > 0.5, std::sync::atomic::Ordering::SeqCst);
				inner.set_changed(false, std::sync::atomic::Ordering::SeqCst);
			}
		}
	}
}

fn handle_param_mod_event(
	param: &ParamModEvent,
	param_map: &ParamMap,
) {
	if let Some(param_id) = param.param_id() {
		let id = param_id.get() as usize;
		let value = param.amount() as f32;
		if let Some(inner) = param_map.get_by_index(id) {
			if inner.is_int() {
				let amount = inner.load(std::sync::atomic::Ordering::SeqCst).int().unwrap() + value as i32;
				inner.store(amount, std::sync::atomic::Ordering::SeqCst);
				inner.set_changed(false, std::sync::atomic::Ordering::SeqCst);
			}else if inner.is_float() {
				let amount = inner.load(std::sync::atomic::Ordering::SeqCst).float().unwrap() + value;
				inner.store(amount, std::sync::atomic::Ordering::SeqCst);
				inner.set_changed(false, std::sync::atomic::Ordering::SeqCst);
			}
		}
	}
}

fn sync_params(
	output_parameter_changes: &mut OutputEvents, 
	param_map: &ParamMap
) {
	for (index, val) in param_map.iter().enumerate() {
		if val.is_chanegd() {
			let value = match val.load(std::sync::atomic::Ordering::SeqCst) {
				SetValue::Bool(v) => if v { 1.0 } else { 0.0 },
				SetValue::Int(v) => v as f64,
				SetValue::Float(v) => v as f64,
				_ => continue
			};

			let _ = output_parameter_changes.try_push(ParamValueEvent::new(
				0,
				(index as u32).into(),
				Pckn::match_all(),
				value,
				Default::default(),
			));

			val.set_changed(false, std::sync::atomic::Ordering::SeqCst);
		}
	}
}

fn flush(
	input_parameter_changes: &clack_plugin::prelude::InputEvents,
	output_parameter_changes: &mut OutputEvents, 
	param_map: &ParamMap
) {
	for event in input_parameter_changes {
		match event.as_core_event() {
			Some(CoreEventSpace::ParamValue(param)) => {
				handle_param_value_event(param, param_map);
			},
			Some(CoreEventSpace::ParamMod(param)) => {
				handle_param_mod_event(param, param_map);
			},
			_ => {}
		}
	}

	sync_params(output_parameter_changes, param_map);
}

impl<P: Plugin> PluginAudioPortsImpl for PluginMain<P> {
	fn count(&mut self, is_input: bool) -> u32 {
		if is_input {
			P::INPUT_PORTS.len() as u32
		}else {
			P::OUTPUT_PORTS.len() as u32
		}
	}

	fn get(&mut self, index: u32, is_input: bool, writer: &mut clack_extensions::audio_ports::AudioPortInfoWriter) {
		let ports = if is_input {
			P::INPUT_PORTS
		}else {
			P::OUTPUT_PORTS
		};

		if index >= ports.len() as u32 {
			return;
		}

		writer.set(&AudioPortInfo {
			id: index.into(),
			name: ports[index as usize].name.as_bytes(),
			channel_count: 2,
			flags: if index == 0 {
				AudioPortFlags::IS_MAIN
			}else {
				AudioPortFlags::empty()
			},
			port_type: Some(AudioPortType::STEREO),
			in_place_pair: None,
		});
	}
}

impl<'a, P: Plugin> PluginAudioProcessorParams for AudioProcessor<'a, P> {
	fn flush(
		&mut self,
		input_parameter_changes: &clack_plugin::prelude::InputEvents,
		output_parameter_changes: &mut OutputEvents,
	) {
		flush(input_parameter_changes, output_parameter_changes, &self.param_map);
	}
}


/// A wrapper around a plugin that implements the `clack_plugin::Plugin` trait.
#[derive(Default)]
pub struct ClapPlugin<P: Plugin>(std::marker::PhantomData<P>);

impl<P: Plugin> DefaultPluginFactory for ClapPlugin<P> 
where 
	<P as i_am_dsp_iced::Processor>::Message: std::fmt::Debug
{
	fn get_descriptor() -> clack_plugin::prelude::PluginDescriptor {
		let descriptor = P::DESCRIPTOR;
		let mut out = clack_plugin::prelude::PluginDescriptor::new(descriptor.id, descriptor.name);
		if let Some(vendor) = descriptor.vendor {
			out = out.with_vendor(vendor)
		}
		if let Some(url) = descriptor.url {
			out = out.with_url(url)
		}
		if let Some(vendor) = descriptor.vendor {
			out = out.with_vendor(vendor)
		}
		if let Some(url) = descriptor.url {
			out = out.with_url(url)
		}
		if let Some(manual_url) = descriptor.manual_url {
			out = out.with_manual_url(manual_url)
		}
		if let Some(support_url) = descriptor.support_url {
			out = out.with_support_url(support_url)
		}
		if let Some(version) = descriptor.version {
			out = out.with_version(version)
		}
		if let Some(description) = descriptor.description {
			out = out.with_description(description)
		}
		out = out.with_features(descriptor.tags.iter().map(|inner| inner.as_cstr()));
		out
	}

	fn new_shared(_: clack_plugin::prelude::HostSharedHandle<'_>) -> Result<Self::Shared<'_>, PluginError> {
		Ok(())
	}

	fn new_main_thread<'a>(
		_: clack_plugin::prelude::HostMainThreadHandle<'a>,
		_: &'a Self::Shared<'a>,
	) -> Result<Self::MainThread<'a>, PluginError> {
		let options = P::window_options();
		let processor = P::new();
		let param_map = processor.param_map();

		Ok(PluginMain {
			window_size: options.window_size,
			scale_factor: 1.0,
			title: options.title,
			resize_hints: options.resize_hints,
			param_map,
			processor: Box::pin(processor), 
			parent: None,
			handle: None,
			message_receivers: vec![],
		})
	}
}

impl<P: Plugin> PluginLatencyImpl for PluginMain<P> {
	fn get(&mut self) -> u32 {
		self.processor.delay() as u32
	}
}

impl<P: Plugin> PluginNotePortsImpl for PluginMain<P> {
	fn count(&mut self, is_input: bool) -> u32 {
		if is_input {
			P::INPUT_MIDI_PORTS.len() as u32
		}else {
			P::OUTPUT_MIDI_PORTS.len() as u32
		}
	}

	fn get(&mut self, index: u32, is_input: bool, writer: &mut clack_extensions::note_ports::NotePortInfoWriter) {
		let ports = if is_input {
			P::INPUT_MIDI_PORTS
		}else {
			P::OUTPUT_MIDI_PORTS
		};

		if index >= ports.len() as u32 {
			return;
		}

		writer.set(&NotePortInfo {
			id: index.into(),
			name: ports[index as usize].name.as_bytes(),
			supported_dialects: NoteDialects::CLAP,
			preferred_dialect: Some(NoteDialect::Clap),
		});
	}
}

impl<P: Plugin> clack_plugin::plugin::Plugin for ClapPlugin<P> 
where 
	<P as i_am_dsp_iced::Processor>::Message: std::fmt::Debug
{
	type Shared<'a> = ();
	type AudioProcessor<'a> = AudioProcessor<'a, P>;
	type MainThread<'a> = PluginMain<P>;

	fn declare_extensions(builder: &mut clack_plugin::prelude::PluginExtensions<Self>, _: Option<&Self::Shared<'_>>) {
		builder
			.register::<PluginAudioPorts>()
			.register::<PluginParams>()
			.register::<PluginState>()
			.register::<PluginAsVST3>()
			.register::<PluginLatency>()
			.register::<PluginNotePorts>()
			.register::<PluginGui>();
	}
}

impl<P: Plugin> ClapPlugin<P> {
	const VST3_ID: [u8; 16] = xxh3_128(P::DESCRIPTOR.id.as_bytes()).to_be_bytes();

	const VST3_PLUGIN_INFO: PluginInfoAsVST3<'static> = PluginInfoAsVST3::new(
		if let Some(vendor) = P::DESCRIPTOR.vendor {
			if let Ok(vendor) = CStr::from_bytes_until_nul(vendor.as_bytes()) {
				Some(vendor)
			}else {
				None
			}
		}else {
			None
		},
		Some(&Self::VST3_ID),
		None,
	);
}

impl<P: Plugin> PluginAsVST3Impl for PluginMain<P> {
	fn num_midi_channels(&mut self, _: u32) -> u32 {
		16
	}

	fn supported_note_expressions(&mut self) -> clack_extensions::clap_wrapper::vst3::SupportedNoteExpressions {
		clack_extensions::clap_wrapper::vst3::SupportedNoteExpressions::empty()
	}
}

impl<P: Plugin> PluginFactoryAsVST3Impl for ClapPlugin<P> 
{
	fn get_vst3_info(&self, index: u32) -> Option<&PluginInfoAsVST3<'_>> {
		if index == 0 {
			Some(&Self::VST3_PLUGIN_INFO)
		}else {
			None
		}
	}
}

impl<P: PluginAuExt> PluginFactoryAsAUv2Impl for ClapPlugin<P> {
	fn get_auv2_info(&self, index: u32) -> Option<clack_extensions::clap_wrapper::auv2::PluginInfoAsAUv2> {
		if index == 0 {
			unsafe {
				Some(clack_extensions::clap_wrapper::auv2::PluginInfoAsAUv2::new(
					str::from_utf8_unchecked(&P::AU_TYPE), 
					str::from_utf8_unchecked(&P::AU_SUBTYPE)
				))
			}
		}else {
			None
		}
	}
}

/// Exports a plugin for the CLAP audio plugin host.
#[macro_export] macro_rules! export_clap {
	($plugin_ty: ty) => {
		i_am_plugin::clack_export_entry!(i_am_plugin::SinglePluginEntry<i_am_plugin::ClapPlugin<$plugin_ty>>);
	}
}