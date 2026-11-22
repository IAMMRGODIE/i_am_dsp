//! context

use crate::audio::{AudioProcessor, OutputChannel};
use crate::events::{
	handle_param_mod_event, handle_param_value_event, note_event_to_clap, sync_params,
};
use crate::main_thread::PluginMain;
use crate::plugin::Plugin;
use crate::shared::*;

/// The context for the CLAP plugin.
///
/// Public mainly for `export_plugin` macro. At most case, you don't need to use it directly.
pub struct ClapContext {
	pub(crate) current_event: usize,
	/// The note events of the current process block.
	///
	/// They are produced and consumed exclusively by the audio thread inside a
	/// single call to [`AudioProcessor::process`], so the context owns them
	/// directly instead of sharing them behind a lock.
	pub(crate) events: Vec<i_am_dsp::NoteEvent>,
	/// The sample the events in [`Self::events`] are timestamped at.
	///
	/// CLAP delivers events in batches that share a timestamp, and a generator is only
	/// supposed to see the events of the sample it is currently rendering. Remembering
	/// that timestamp is what lets [`ProcessContext::events`] hide a batch again on the
	/// samples that follow, instead of handing the same note to the generator over and
	/// over for the rest of the block.
	pub(crate) events_sample: usize,
	pub(crate) info: Option<ProcessInfos>,
	/// What to report before the host has said anything.
	///
	/// A host is free to send no transport at all, and a processor that reads the rate
	/// from here would then see zero and divide by it: the additive voice turned into
	/// NaN from its first sample. The rate is known from activation, so it is reported
	/// whether or not the host ever describes a timeline.
	pub(crate) base: ProcessInfos,
	pub(crate) current_sample: usize,
	/// The musical timeline, taken from the audio processor for the duration of the block.
	pub(crate) clock: BeatClock,
	/// The absolute sample the current block started at.
	pub(crate) block_start: u64,
	pub(crate) event_sender: Sender<(usize, i_am_dsp::NoteEvent)>,
}

impl ProcessContext for ClapContext {
	fn infos(&self) -> &ProcessInfos {
		self.info.as_ref().unwrap_or(&self.base)
	}

	fn events(&self) -> &[i_am_dsp::NoteEvent] {
		if self.current_sample == self.events_sample {
			&self.events
		} else {
			&[]
		}
	}

	fn next_event(&mut self) -> Option<i_am_dsp::NoteEvent> {
		if self.current_sample != self.events_sample || self.current_event >= self.events.len() {
			return None;
		}

		let event = self.events[self.current_event].clone();
		self.current_event += 1;
		Some(event)
	}

	fn send_event(&mut self, event: i_am_dsp::NoteEvent) {
		self.event_sender
			.send((self.current_sample, event))
			.unwrap();
	}

	fn should_stop(&self) -> bool {
		false
	}

	fn clear_events(&mut self) {
		self.current_event = 0;
		self.events.clear();
	}

	fn set_sample_offset(&mut self, sample_offset: usize) {
		self.current_sample = sample_offset;

		// The clock measures absolute samples, so the block offset is added to where this
		// block started. The borrow of `self.clock` ends with the statement, which is what
		// lets the position be written into `self.info` right after.
		let position = self
			.clock
			.advance_to(self.block_start + sample_offset as u64);

		if let Some(info) = &mut self.info {
			info.set_position_beats(position);
		}
	}

	fn sample_offset(&self) -> usize {
		self.current_sample
	}
}

fn convert_transport(inner: &TransportEvent, sample_rate: usize) -> i_am_dsp::ProcessInfos {
	let mut info = ProcessInfos::default();
	info.sample_rate = sample_rate;
	info.playing = inner.flags.contains(TransportFlags::IS_PLAYING);
	info.tempo = if inner.flags.contains(TransportFlags::HAS_TEMPO) {
		Some(inner.tempo as f32)
	} else {
		None
	};
	// `position_beats` and `current_beat_number` are left to the clock, which is the single
	// writer of the musical position: it fills them per sample in `set_sample_offset`.
	info.time_signature = if inner.flags.contains(TransportFlags::HAS_TIME_SIGNATURE) {
		Some((
			inner.time_signature_numerator as usize,
			inner.time_signature_denominator as usize,
		))
	} else {
		None
	};
	info.current_time = inner.song_pos_seconds.to_float() as f32;
	info.trustable = true;
	info
}

impl ClapContext {
	/// Takes a transport report and anchors the clock at `sample` (an absolute sample count).
	///
	/// CLAP reports the transport twice: once per block in the process struct, which is the
	/// position at the start of the block, and — when the host wants a change to be
	/// sample-accurate — as an event inside the block. Both go through here, so a seek, a loop
	/// wrap or a tempo change moves the position instead of letting it run on.
	pub(crate) fn apply_transport(&mut self, event: &TransportEvent, sample: u64) {
		self.clock
			.set_running(event.flags.contains(TransportFlags::IS_PLAYING));

		if event.flags.contains(TransportFlags::HAS_BEATS_TIMELINE) {
			// A host that reports a beat timeline without a tempo flag is taken to sit at the
			// 120 bpm the CLAP wrapper itself assumes: without a rate the position could not
			// advance at all, and 120 is the tempo every host falls back to.
			let tempo = if event.flags.contains(TransportFlags::HAS_TEMPO) {
				event.tempo
			} else {
				120.0
			};
			self.clock.anchor(
				sample,
				event.song_pos_beats.to_float(),
				tempo,
				event.tempo_inc,
			);
		} else {
			// A tempo without a beat timeline still says how fast time passes, but there is no
			// position for an effect to be in phase with.
			self.clock.clear();
		}

		self.info = Some(convert_transport(event, self.clock.sample_rate()));
	}

	fn from_clap(
		last_available_info: &mut Option<ProcessInfos>,
		process: Process,
		event_sender: Sender<(usize, i_am_dsp::NoteEvent)>,
		block_start: u64,
		clock: BeatClock,
	) -> Self {
		let mut base = ProcessInfos::default();
		base.sample_rate = clock.sample_rate();

		let mut context = Self {
			current_event: 0,
			event_sender,
			events: Vec::new(),
			events_sample: 0,
			base,
			current_sample: 0,
			info: None,
			clock,
			block_start,
		};

		if let Some(transport) = process.transport {
			// The transport in the process struct is the position at the start of the block.
			context.apply_transport(transport, block_start);
			*last_available_info = context.info.clone();
		} else {
			// The host did not repeat the transport this block, so what it last said still
			// stands and the clock keeps running from the anchor it already holds.
			context.info = last_available_info.clone();
		}

		context
	}
}

impl<'a, P: Plugin> AudioProcessor<'a, P> {
	/// Whether every input channel of this block is exactly zero.
	///
	/// A plug-in with no audio input is silent by definition, which is what an instrument
	/// wants to hear; the scan is what a product that asked for a silent input is allowed
	/// to rely on. Comparing against zero is the right test: a denormal is as good as
	/// nothing to an effect, and the wrapper flushes those to zero anyway.
	fn inputs_are_silent(&self, frames: usize) -> bool {
		for (channels, wide) in self.temp_buffer_1.iter().zip(&self.temp_buffer_2) {
			for &pointer in channels.iter() {
				// SAFETY: the host's channel buffer holds `frames` samples of this type.
				let silent = unsafe {
					if *wide {
						from_raw_parts(pointer as *const f64, frames)
							.iter()
							.all(|sample| *sample == 0.0)
					} else {
						from_raw_parts(pointer as *const f32, frames)
							.iter()
							.all(|sample| *sample == 0.0)
					}
				};

				if !silent {
					return false;
				}
			}
		}

		true
	}
}

impl<'a, P: Plugin> PluginAudioProcessor<'a, (), PluginMain<P>> for AudioProcessor<'a, P> {
	fn activate(
		_: clack_plugin::prelude::HostAudioProcessorHandle<'a>,
		main_thread: &mut PluginMain<P>,
		_: &'a (),
		config: clack_plugin::prelude::PluginAudioConfiguration,
	) -> Result<Self, PluginError> {
		// The host says what it will run at here, and this is the first moment anyone
		// can know: everything was built before the session existed.
		//
		// SAFETY: the processor is pinned in the still-alive `PluginMain`, and this is
		// the same place that takes its address below. Pinning it stops it moving, not
		// being borrowed mutably, and the borrow ends before the address is taken.
		unsafe { main_thread.processor.as_mut().get_unchecked_mut() }
			.on_activate(config.sample_rate as usize);

		let ptr = main_thread.processor.as_ref().get_ref() as *const P as usize;
		let (event_sender, event_receiver) = crossbeam_channel::unbounded();
		let param_map = main_thread.processor.param_map();

		// The ramps are built here, on the main thread: the audio thread only walks them.
		#[cfg(feature = "smoothing")]
		let smoothing = crate::smoothing::Smoothing::new(
			&param_map,
			config.sample_rate as usize,
			P::SMOOTHED_PARAMETERS,
		);

		Ok(AudioProcessor {
			processor: ptr,
			temp_buffer_1: vec![],
			temp_buffer_2: vec![],
			temp_buffer_3: vec![],
			temp_buffer_4: vec![],
			sample_rate: config.sample_rate as usize,
			last_available_info: None,
			clock: BeatClock::new(config.sample_rate as usize),
			samples_processed: 0,
			event_sender,
			event_receiver,
			param_map,
			#[cfg(feature = "smoothing")]
			smoothing,
			latency: main_thread.latency.publisher(),
			ui_open: main_thread.ui_open.clone(),
			state_guard: main_thread.state_guard.clone(),
			message_receiver: main_thread.message_receiver.clone(),
			_phantom: std::marker::PhantomData,
		})
	}

	fn reset(&mut self) {
		self.temp_buffer_1.clear();
		self.temp_buffer_2.clear();
		self.temp_buffer_3.clear();
		self.last_available_info = None;
		// A reset drops the timeline: the host has to tell us where we are again.
		self.clock = BeatClock::new(self.sample_rate);
		self.samples_processed = 0;
		#[cfg(feature = "smoothing")]
		self.smoothing.resync();
		// SAFETY: `activate` stored the address of the `P` owned by the still-alive and
		// pinned `PluginMain`, so the pointer is valid and the value is never moved.
		let processor = unsafe { &mut *(self.processor as *mut P) };
		processor.on_reset();
	}

	fn stop_processing(&mut self) {
		self.temp_buffer_1.clear();
		self.temp_buffer_2.clear();
		self.temp_buffer_3.clear();
		self.last_available_info = None;
		// Processing stopped, so the timeline is dropped with it; the next block re-anchors
		// from whatever the host reports when it starts us again.
		self.clock = BeatClock::new(self.sample_rate);
		self.samples_processed = 0;
		// SAFETY: `activate` stored the address of the `P` owned by the still-alive and
		// pinned `PluginMain`, so the pointer is valid and the value is never moved.
		let processor = unsafe { &mut *(self.processor as *mut P) };
		processor.on_stop_processing();
	}

	fn start_processing(&mut self) -> Result<(), PluginError> {
		self.temp_buffer_1.clear();
		self.temp_buffer_2.clear();
		self.temp_buffer_3.clear();
		self.last_available_info = None;
		// SAFETY: `activate` stored the address of the `P` owned by the still-alive and
		// pinned `PluginMain`, so the pointer is valid and the value is never moved.
		let processor = unsafe { &mut *(self.processor as *mut P) };
		processor.on_start_processing();
		Ok(())
	}

	fn process(
		&mut self,
		process: Process,
		mut audio: Audio,
		events: Events,
	) -> Result<ProcessStatus, PluginError> {
		// The host owns this thread, so it may not have flushed subnormals for us: a decaying
		// reverb or convolution tail would then run an order of magnitude slower on x86.
		i_am_dsp::enable_flush_to_zero();

		// The main thread is saving or loading the project and has taken the
		// processor: this block is silence, and `P` is not touched at all.
		if self.state_guard.audio_should_yield() {
			let frames = audio.frames_count() as usize;

			for mut output_port in audio.output_ports() {
				match output_port.channels()? {
					SampleType::F32(inner) | SampleType::Both(inner, _) => {
						for ptr in inner.raw_data() {
							for index in 0..frames {
								// SAFETY: the host's channel buffer holds `frames` samples.
								unsafe {
									*ptr.add(index) = 0.0;
								}
							}
						}
					}
					SampleType::F64(inner) => {
						for ptr in inner.raw_data() {
							for index in 0..frames {
								// SAFETY: see the F32 arm above.
								unsafe {
									*ptr.add(index) = 0.0;
								}
							}
						}
					}
				}
			}

			self.samples_processed += frames as u64;

			return Ok(ProcessStatus::Continue);
		}
		// SAFETY: `activate` stored the address of the `P` owned by the still-alive and
		// pinned `PluginMain`, so the pointer is valid and the value is never moved.
		let processor = unsafe { &mut *(self.processor as *mut P) };

		// The user's own actions reach the product here, on the thread that owns it.
		while let Ok(message) = self.message_receiver.try_recv() {
			i_am_dsp::prelude::Processor::on_message(&*processor, message);
		}

		self.temp_buffer_1.clear();
		self.temp_buffer_2.clear();
		self.temp_buffer_3.clear();
		// The context is per block, but the timeline is not: a host may report the transport
		// once and then stop repeating it, so the clock is handed in and taken back out.
		let block_start = self.samples_processed;
		let clock = self.clock;
		let event_sender = self.event_sender.clone();
		let mut context = Some(Box::new(ClapContext::from_clap(
			&mut self.last_available_info,
			process,
			event_sender,
			block_start,
			clock,
		)));
		let mut output_temp = [0.0; 2];

		let buffer_size = audio.frames_count() as usize;

		// let mut min_buffer_size = output_port.frames_count() as usize;
		let input_ports = audio.input_ports();
		for input_port in input_ports {
			match input_port.channels()? {
				SampleType::F32(inner) | SampleType::Both(inner, _) => {
					// SAFETY: `raw_data()` returns the host's array of per-channel sample
					// pointers, so its length is the port's *channel count*, not the block's
					// frame count. A `*mut f32` and an `usize` have the same size and
					// alignment, so viewing the pointer array as a `&[usize]` of exactly
					// `raw_data().len()` elements stays in bounds.
					let channel_count = inner.raw_data().len();
					let ptr = inner.raw_data().as_ptr() as *const usize;
					unsafe {
						self.temp_buffer_1.push(from_raw_parts(ptr, channel_count));
					}
					self.temp_buffer_2.push(false);
				}
				SampleType::F64(inner) => {
					// SAFETY: see the F32 arm above.
					let channel_count = inner.raw_data().len();
					let ptr = inner.raw_data().as_ptr() as *const usize;
					unsafe {
						self.temp_buffer_1.push(from_raw_parts(ptr, channel_count));
					}
					self.temp_buffer_2.push(true);
				}
			}
		}

		for _ in 0..self.temp_buffer_1.len() {
			self.temp_buffer_3.push([0.0, 0.0]);
		}

		// The output channels, taken once per block: from here on the sample loop only
		// writes through these. Walking the port structures per sample was most of what
		// a silent plug-in cost.
		self.temp_buffer_4.clear();

		for mut output_port in audio.output_ports() {
			match output_port.channels()? {
				SampleType::F32(inner) | SampleType::Both(inner, _) => {
					for (channel, pointer) in inner.raw_data().iter().enumerate().take(2) {
						self.temp_buffer_4.push(OutputChannel {
							pointer: *pointer as *mut u8 as usize,
							channel,
							wide: false,
						});
					}
				}
				SampleType::F64(inner) => {
					for (channel, pointer) in inner.raw_data().iter().enumerate().take(2) {
						self.temp_buffer_4.push(OutputChannel {
							pointer: *pointer as *mut u8 as usize,
							channel,
							wide: true,
						});
					}
				}
			}
		}

		// A block with nothing to play does not have to reach the DSP: no event can start
		// a voice, no input carries a signal, and the product says nothing is left to hear.
		// The output is still zeroed and the timeline still moves, so a block that costs a
		// whole voice costs a memset instead.
		let wakes = events.input.batch().any(|batch| {
			batch.events().any(|event| {
				matches!(
					event.as_core_event(),
					Some(
						CoreEventSpace::NoteOn(_)
							| CoreEventSpace::NoteOff(_)
							| CoreEventSpace::NoteChoke(_)
							| CoreEventSpace::NoteEnd(_)
							| CoreEventSpace::ParamValue(_)
							| CoreEventSpace::ParamMod(_)
							| CoreEventSpace::Midi(_)
					)
				)
			})
		});
		// A page on screen is fed by this loop -- its scope, its meters -- so it is the one
		// case where a silent block is still worth rendering.
		let watching = self.ui_open.load(std::sync::atomic::Ordering::Relaxed);
		let idle =
			!watching && !wakes && processor.is_idle(true) && self.inputs_are_silent(buffer_size);
		let rendered = if idle { 0 } else { buffer_size };

		let mut next_event_sample = events
			.input
			.batch()
			.next()
			.map(|batch| batch.first_sample())
			.unwrap_or(0);
		let mut bacthced = events.input.batch();

		for i in 0..rendered {
			if i >= next_event_sample
				&& let (Some(batch), Some(ctx)) = (bacthced.next(), &mut context)
			{
				ctx.events.clear();
				for event in batch.events() {
					match event.as_core_event() {
						Some(CoreEventSpace::NoteOn(note)) => {
							ctx.events.push(i_am_dsp::NoteEvent::NoteOn {
								channel: note.pckn().channel.into_specific().unwrap_or_default()
									as u8,
								note: note.pckn().key.into_specific().unwrap_or_default() as usize,
								velocity: note.velocity() as f32,
							});
						}
						Some(CoreEventSpace::NoteOff(note)) => {
							ctx.events.push(i_am_dsp::NoteEvent::NoteOff {
								channel: note.pckn().channel.into_specific().unwrap_or_default()
									as u8,
								note: note.pckn().key.into_specific().unwrap_or_default() as usize,
								velocity: note.velocity() as f32,
							});
						}
						Some(CoreEventSpace::NoteChoke(note)) => {
							if note.pckn().key.is_all() {
								ctx.events.push(i_am_dsp::NoteEvent::ImmediateStop);
							} else {
								ctx.events.push(i_am_dsp::NoteEvent::Stop {
									channel: note.pckn().channel.into_specific().unwrap_or_default()
										as u8,
									note: note.pckn().key.into_specific().unwrap_or_default()
										as usize,
								});
							}
						}
						Some(CoreEventSpace::NoteEnd(note)) => {
							ctx.events.push(i_am_dsp::NoteEvent::NoteOff {
								channel: note.pckn().channel.into_specific().unwrap_or_default()
									as u8,
								note: note.pckn().key.into_specific().unwrap_or_default() as usize,
								velocity: 1.0,
							});
						}

						Some(CoreEventSpace::ParamValue(param)) => {
							handle_param_value_event(param, &self.param_map);
						}
						Some(CoreEventSpace::ParamMod(param)) => {
							handle_param_mod_event(param, &self.param_map);
						}
						Some(CoreEventSpace::ParamGestureBegin(_)) => {}
						Some(CoreEventSpace::ParamGestureEnd(_)) => {}
						Some(CoreEventSpace::Transport(param)) => {
							// A transport event inside the block is the sample-accurate version of
							// the same report, so the clock re-anchors at this sample rather than
							// at the start of the block.
							ctx.apply_transport(param, ctx.block_start + i as u64);
							self.last_available_info = ctx.info.clone();
						}

						Some(CoreEventSpace::Midi(midi)) => {
							// A host may talk to a note port in the MIDI dialect, and FL
							// Studio's MIDI Out route does exactly that: the bytes a
							// keyboard would send arrive unprocessed, so they have to be
							// decoded before a note-following effect can see them.
							if let Some(event) = note_event_from_midi(midi.data()) {
								ctx.events.push(event);
							}
						}
						Some(CoreEventSpace::Midi2(_))
						| Some(CoreEventSpace::MidiSysEx(_))
						| Some(CoreEventSpace::NoteExpression(_))
						| None => {}
					}
				}
				ctx.current_event = 0;
				ctx.events_sample = batch.first_sample();
				next_event_sample = batch.next_batch_first_sample().unwrap_or(buffer_size);
			}

			for (j, buffer) in self.temp_buffer_1.iter().enumerate() {
				for channel in 0..2 {
					if channel >= buffer.len() {
						continue;
					}

					// SAFETY: `buffer[channel]` is a per-channel sample pointer provided by the
					// host, valid for `buffer_size` samples, and `i` is inside this block.
					unsafe {
						if self.temp_buffer_2[j] {
							let ptr = buffer[channel] as *const f64;
							let array = from_raw_parts(ptr, buffer_size);
							self.temp_buffer_3[j][channel] = array[i] as f32;
							if j == 0 {
								output_temp[channel] = array[i] as f32;
							}
						} else {
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
			// Also advances the musical clock to this sample.
			if let Some(context) = context.as_mut() {
				context.set_sample_offset(i);
			}

			// After the events of this sample and before the DSP reads a parameter:
			// the ramp is what the product sees, not the host's raw jump.
			#[cfg(feature = "smoothing")]
			self.smoothing.tick();

			// The box moves into the trait object and back out again, which keeps the
			// same allocation: a host calls this once per sample, and boxing a fresh
			// context each time was a malloc and a free every twenty microseconds.
			let mut context_in: Box<dyn ProcessContext> = context.take().unwrap();

			processor.process(&mut output_temp, &self.temp_buffer_3, &mut context_in);

			let context_back = context_in as Box<dyn Any>;
			let context_back =
				Box::<dyn std::any::Any>::downcast::<ClapContext>(context_back).unwrap();
			context = Some(context_back);

			for output in &self.temp_buffer_4 {
				// SAFETY: each pointer was taken from the host's output buffers for this
				// block, which hold `buffer_size` samples of the type `i` is written as,
				// and `i` is inside this block.
				unsafe {
					if output.wide {
						*(output.pointer as *mut f64).add(i) = output_temp[output.channel] as f64;
					} else {
						*(output.pointer as *mut f32).add(i) = output_temp[output.channel];
					}
				}
			}
		}

		// Nothing was rendered, so there is nothing to hear: silence the output and move the
		// timeline anyway, because an effect that wakes up later has to find the song where
		// the host left it rather than where the last note ended.
		if idle {
			for output in &self.temp_buffer_4 {
				// SAFETY: each pointer was taken from this block's output buffers, which hold
				// `buffer_size` samples of the type it is written as.
				unsafe {
					if output.wide {
						from_raw_parts_mut(output.pointer as *mut f64, buffer_size).fill(0.0);
					} else {
						from_raw_parts_mut(output.pointer as *mut f32, buffer_size).fill(0.0);
					}
				}
			}

			if let Some(context) = context.as_mut() {
				context.set_sample_offset(buffer_size);
			}
		}

		// Hand the timeline the block ended with back to the processor — a host is free to
		// report the transport once and then leave the plugin running on it — and count the
		// samples this block consumed, so the next one measures from the right place.
		self.samples_processed += buffer_size as u64;
		if let Some(context) = context {
			self.clock = context.clock;
		}

		// A synced delay takes its length from the tempo, so the latency the host has to
		// compensate may have moved during this block. Publishing it is cheap; the main thread
		// decides whether the host has to hear about it.
		self.latency.publish(processor.delay() as u32);

		for (offset, event) in self.event_receiver.try_iter() {
			note_event_to_clap(events.output, offset, event)?;
		}

		sync_params(events.output, &self.param_map);

		// A block that was skipped is the one thing a host can act on: "processing should
		// continue, unless all outputs are quiet" is exactly what happened, and a host that
		// understands it may stop calling us until an event arrives. Not `Sleep`: that one
		// asks to be left alone entirely, and a note would then depend on the host waking a
		// plug-in it has parked.
		Ok(if idle {
			ProcessStatus::ContinueIfNotQuiet
		} else {
			ProcessStatus::Continue
		})
	}
}

/// Decodes one raw MIDI 1.0 message into the note events effects understand.
///
/// A note port may be driven in the MIDI dialect as well as the CLAP one. FL
/// Studio's MIDI Out route sends the raw bytes a keyboard would send, so a
/// note-following effect otherwise never hears the notes it is meant to follow.
/// Every message the host delivers is complete: running status is the host's job.
fn note_event_from_midi(data: [u8; 3]) -> Option<i_am_dsp::NoteEvent> {
	let channel = data[0] & 0x0F;

	match data[0] & 0xF0 {
		// A note on with no velocity is the running-status note off, and hosts use it.
		0x90 if data[2] > 0 => Some(i_am_dsp::NoteEvent::NoteOn {
			channel,
			note: data[1] as usize,
			velocity: data[2] as f32 / 127.0,
		}),
		0x90 => Some(i_am_dsp::NoteEvent::NoteOff {
			channel,
			note: data[1] as usize,
			velocity: 0.0,
		}),
		0x80 => Some(i_am_dsp::NoteEvent::NoteOff {
			channel,
			note: data[1] as usize,
			velocity: data[2] as f32 / 127.0,
		}),
		// All Sound Off and All Notes Off both mean "release everything".
		0xB0 if data[1] == 120 || data[1] == 123 => Some(i_am_dsp::NoteEvent::ImmediateStop),
		0xB0 => Some(i_am_dsp::NoteEvent::MidiCC {
			channel,
			cc: data[1],
			value: data[2] as f32 / 127.0,
		}),
		_ => None,
	}
}

#[cfg(test)]
mod tests {
	use super::note_event_from_midi;
	use i_am_dsp::NoteEvent;

	#[test]
	fn a_note_on_keeps_its_channel_and_velocity() {
		match note_event_from_midi([0x92, 60, 100]) {
			Some(NoteEvent::NoteOn {
				channel,
				note,
				velocity,
			}) => {
				assert_eq!(channel, 2);
				assert_eq!(note, 60);
				assert!((velocity - 100.0 / 127.0).abs() < 1e-6);
			}
			other => panic!("expected a note on, got {other:?}"),
		}
	}

	#[test]
	fn a_note_on_with_no_velocity_is_a_note_off() {
		assert_eq!(
			note_event_from_midi([0x90, 60, 0]),
			Some(NoteEvent::NoteOff {
				channel: 0,
				note: 60,
				velocity: 0.0
			})
		);
	}

	#[test]
	fn a_note_off_carries_the_release_velocity() {
		match note_event_from_midi([0x83, 52, 64]) {
			Some(NoteEvent::NoteOff {
				channel,
				note,
				velocity,
			}) => {
				assert_eq!(channel, 3);
				assert_eq!(note, 52);
				assert!((velocity - 64.0 / 127.0).abs() < 1e-6);
			}
			other => panic!("expected a note off, got {other:?}"),
		}
	}

	#[test]
	fn all_notes_off_stops_everything() {
		assert_eq!(
			note_event_from_midi([0xB5, 123, 0]),
			Some(NoteEvent::ImmediateStop)
		);
		assert_eq!(
			note_event_from_midi([0xB5, 120, 0]),
			Some(NoteEvent::ImmediateStop)
		);
	}

	#[test]
	fn another_controller_stays_a_controller_event() {
		match note_event_from_midi([0xB1, 1, 127]) {
			Some(NoteEvent::MidiCC { channel, cc, value }) => {
				assert_eq!(channel, 1);
				assert_eq!(cc, 1);
				assert!((value - 1.0).abs() < 1e-6);
			}
			other => panic!("expected a controller, got {other:?}"),
		}
	}

	#[test]
	fn a_message_that_is_not_a_note_or_controller_is_dropped() {
		assert_eq!(note_event_from_midi([0xE0, 0, 64]), None);
	}
}
