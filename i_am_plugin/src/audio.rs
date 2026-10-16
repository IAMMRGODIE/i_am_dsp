//! audio

use crate::shared::*;
use crate::plugin::{Plugin};
use crate::main_thread::{PluginMain};
use crate::latency::{LatencyPublisher};

/// The processor itself is owned by [`PluginMain`] on the main thread; this struct only
/// keeps its address, stored as an `usize` because [`AudioProcessor`] has to be `Send`
/// while raw pointers are not.
/// 
/// While the audio thread uses `&mut P` here, the main thread can still reach the same
/// `P` through `&P` (`on_message`, `delay`, `synced_view`, …) or, when loading state,
/// through `&mut P`. Those references overlap, so a plugin must share state between the
/// two sides only through atomics or other interior mutability.
pub struct AudioProcessor<'a, P: Plugin> {
	pub(crate) processor: usize,
	pub(crate) temp_buffer_1: Vec<&'a [usize]>,
	pub(crate) temp_buffer_2: Vec<bool>,
	pub(crate) temp_buffer_3: Vec<[f32; 2]>,
	pub(crate) sample_rate: usize,
	pub(crate) last_available_info: Option<ProcessInfos>,
	/// The musical timeline, kept across blocks.
	///
	/// The context is rebuilt for every block, but a host is allowed to report the
	/// transport once and then leave the plugin to run on its own, so the clock has to
	/// outlive the context: it is moved in at the start of a block and taken back at the end.
	pub(crate) clock: BeatClock,
	/// How many samples have been processed since the plugin was activated.
	///
	/// The clock anchors and advances on absolute samples; a block offset is only a
	/// position inside one block, so it cannot be used to measure elapsed time.
	pub(crate) samples_processed: u64,
	pub(crate) event_sender: Sender<(usize, i_am_dsp::NoteEvent)>,
	pub(crate) event_receiver: Receiver<(usize, i_am_dsp::NoteEvent)>,
	pub(crate) param_map: ParamMap,
	/// Where the audio thread publishes the latency it is running at, for the main thread to
	/// hand to the host.
	pub(crate) latency: LatencyPublisher,
	pub(crate) _phantom: std::marker::PhantomData<P>,
}

impl<P: Plugin> PluginMain<P> {
	/// Tells the host that the latency has to be queried again.
	///
	/// CLAP hands this to a plugin as a host extension, and the bindings in use only wrap the
	/// host side of it, so the call goes through the raw struct.
	fn notify_latency_changed(&self) {
		let Some(host) = &self.host else {
			return;
		};

		if !self.processor.reports_latency_changes() {
			return;
		}

		// SAFETY: `host` is the handle the host handed us in `new_main_thread` and is still
		// alive; the extension it returns is owned by the host and lives as long as it does.
		unsafe {
			let raw = host.as_raw();
			let Some(get_extension) = raw.get_extension else {
				return;
			};
			let extension = get_extension(raw as *const _, CLAP_EXT_LATENCY.as_ptr());

			if extension.is_null() {
				return;
			}

			let latency = &*(extension as *const clap_host_latency);

			if let Some(changed) = latency.changed {
				changed(raw as *const _);
			}
		}
	}
}

impl<P: Plugin> PluginMainThread<'_, ()> for PluginMain<P> {
	fn on_main_thread(&mut self) {
		// The audio thread is the only one that knows what the tempo has done to the latency;
		// this is where that news reaches the host.
		if self.latency.take_change().is_some() {
			self.notify_latency_changed();
		}

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