//! The imports every module of this crate works with.
//!
//! They used to sit at the top of a single file; a glob of this module is how the
//! modules that were split out of it keep seeing them, without each one repeating a
//! block that most of it would not use.

pub(crate) use std::{any::Any, ffi::CStr, fmt::Debug, io::{Read, Write}, pin::Pin, slice::from_raw_parts, sync::{Arc, atomic::{AtomicU32, Ordering}}};

pub(crate) use clap_sys::ext::latency::{CLAP_EXT_LATENCY, clap_host_latency};
pub(crate) use clack_extensions::{
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
pub(crate) use clack_plugin::{
	host::HostMainThreadHandle,
	entry::DefaultPluginFactory, events::{Match, Pckn, event_types::{NoteChokeEvent, NoteOffEvent, NoteOnEvent, ParamModEvent, ParamValueEvent, TransportEvent, TransportFlags}, spaces::CoreEventSpace
	}, plugin::{PluginAudioProcessor, PluginError, PluginMainThread}, prelude::{OutputEvents, SampleType}, process::{Audio, Events, Process, ProcessStatus}
};
pub(crate) use crossbeam_channel::{Receiver, Sender};
pub(crate) use i_am_dsp::{ProcessContext, ProcessInfos, prelude::{AtomicValue, BeatClock, ParamMap, SetValue, from_binary, to_binary}};
pub(crate) use i_am_dsp_iced::{Message, Processor, SyncedView, timer};
pub(crate) use iced_baseview::{Application, Settings, Theme, baseview::{Size, WindowOpenOptions, WindowScalePolicy}, open_parented, window::WindowHandle};
pub(crate) use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};

extern crate self as i_am_plugin;

#[doc(hidden)]
pub(crate) use xxhash_rust::const_xxh3::xxh3_128;

