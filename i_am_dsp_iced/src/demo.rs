use std::collections::HashMap;
use std::time::Duration;
use cpal::StreamConfig;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use i_am_dsp::ProcessContext;
use i_am_dsp::{NoteEvent, ProcessInfos};
use iced::widget::{column, text};
use iced::{Element, Event, Subscription};
use iced::keyboard::key::{Code, Physical};

use crate::styles::ERROR_COLOR;
use crate::{Message, Processor, SyncedView};

lazy_static::lazy_static! {
	static ref NoteMap: HashMap<Code, u8> = {
		const C5: u8 = 60;
		const C4: u8 = 60 - 12;

		HashMap::from([
			(Code::KeyQ, C5),
			(Code::Digit2, C5 + 1),
			(Code::KeyW, C5 + 2),
			(Code::Digit3, C5 + 3),
			(Code::KeyE, C5 + 4),
			(Code::KeyR, C5 + 5),
			(Code::Digit5, C5 + 6),
			(Code::KeyT, C5 + 7),
			(Code::Digit6, C5 + 8),
			(Code::KeyY, C5 + 9),
			(Code::Digit7, C5 + 10),
			(Code::KeyU, C5 + 11),
			(Code::KeyI, C5 + 12),
			(Code::Digit9, C5 + 13),
			(Code::KeyO, C5 + 14),
			(Code::Digit0, C5 + 15),
			(Code::KeyP, C5 + 16),

			(Code::KeyZ, C4),
			(Code::KeyS, C4 + 1),
			(Code::KeyX, C4 + 2),
			(Code::KeyD, C4 + 3),
			(Code::KeyC, C4 + 4),
			(Code::KeyV, C4 + 5),
			(Code::KeyG, C4 + 6),
			(Code::KeyB, C4 + 7),
			(Code::KeyH, C4 + 8),
			(Code::KeyN, C4 + 9),
			(Code::KeyJ, C4 + 10),
			(Code::KeyM, C4 + 11),
		])
	};
}

pub struct SimpleContext {
	/// The process infos
	pub info: ProcessInfos,
	/// The midi events
	pub midi_events: Vec<NoteEvent>,
}

impl ProcessContext for SimpleContext {
	fn infos(&self) -> &ProcessInfos {
		&self.info
	}

	fn next_event(&mut self) -> Option<NoteEvent> {
		self.midi_events.pop()
	}

	fn send_event(&mut self, event: NoteEvent) {
		self.midi_events.push(event);
	}

	fn events(&self) -> &[NoteEvent] {
		&self.midi_events
	}
}



/// A simple runable demo based on a processor
pub struct Demo<P: Processor> {
	sender: crossbeam_channel::Sender<P::Message>,
	error_receiver: crossbeam_channel::Receiver<String>,
	current_error: Option<String>,
	view: P::SyncedView,

	_stream: cpal::Stream,
}

struct CpalInner<P: Processor> {
	processor: P,
	sample_rate: usize,
	midi_events: Vec<NoteEvent>,
}

impl<P: Processor> Demo<P> {
	/// Create a new demo
	pub fn new(builder: impl FnOnce(usize) -> P) -> Self {
		let (sender, receiver) = crossbeam_channel::unbounded::<P::Message>();
		let (error_sender, error_receiver) = crossbeam_channel::unbounded();
		let host = cpal::default_host();
		let device = host.default_output_device().expect("no output device available");
		let config = device.default_output_config().expect("no default output config");
		let config = StreamConfig::from(config);
		let sample_rate = config.sample_rate as usize;
		let mut processor = builder(sample_rate);
		let view = processor.synced_view();
		let mut shared_data = CpalInner {
			processor,
			sample_rate,
			midi_events: vec![],
		};

		let other_sender = error_sender.clone();

		let stream = device.build_output_stream(
			&config,
			move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
				loop {
					match receiver.try_recv() {
						Ok(message) => {
							if let Some(event) = message.note_event() {
								shared_data.midi_events.push(event);
							}else {
								shared_data.processor.on_message(message);
							}
						},
						Err(crossbeam_channel::TryRecvError::Empty) => break,
						Err(crossbeam_channel::TryRecvError::Disconnected) => {
							error_sender.send("Recevier Disconnected".to_string()).expect("cant send error message");
							break;
						}
					}
				}
				let mut process_info = ProcessInfos::default();
				process_info.sample_rate = shared_data.sample_rate;
				process_info.current_time = 0.0;
				process_info.playing = true;

				let mut process_context = Box::new(SimpleContext {
					info: process_info,
					midi_events: std::mem::take(&mut shared_data.midi_events),
				}) as Box<dyn ProcessContext>;

				for val in data.chunks_mut(2) {
					if val.len() != 2 {
						error_sender.send("Received invalid number of samples".to_string()).expect("canot send error message");
						break;
					}

					let mut output = [0.0; 2];

					shared_data.processor.process(&mut output, &[], &mut process_context);

					val[0] = output[0];
					val[1] = output[1];
				}
			},
			move |err| {
				other_sender.send(format!("Stream error: {}", err)).expect("canot send error message");
			},
			None 
		).expect("failed to build stream");

		stream.play().expect("failed to play stream");

		Self {
			sender,
			error_receiver,
			current_error: None,
			view,
			_stream: stream,
		}
	}

	/// A subscriber that should be used in application
	pub fn subscriber() -> Subscription<P::Message> {
		let keyborad = iced::event::listen().filter_map(|inner| {
			let event = if let Event::Keyboard(key_event) = inner {
				key_event
			}else {
				return None;
			};

			match event {
				iced::keyboard::Event::KeyPressed { physical_key, repeat, .. } => {
					if repeat {
						return None;
					}

					let Physical::Code(code) = physical_key else {
						return None;
					};

					if code == Code::Escape {
						return Some(P::Message::from_note_event(NoteEvent::ImmediateStop));
					}

					let note = *NoteMap.get(&code)?;

					Some(P::Message::from_note_event(NoteEvent::NoteOn { time: 0, channel: 0, note, velocity: 1.0 }))
				},
				iced::keyboard::Event::KeyReleased { physical_key, .. } => {
					let Physical::Code(code) = physical_key else {
						return None;
					};

					let note = *NoteMap.get(&code)?;

					Some(P::Message::from_note_event(NoteEvent::NoteOff { time: 0, channel: 0, note, velocity: 0.0 }))
				},
				_ => None,
			}
		});

		let timer = iced::time::every(Duration::from_millis(16)).map(P::Message::tick);

		Subscription::batch(vec![
			keyborad,
			timer
		])
	}

	/// The view function for iced
	pub fn view(&self) -> Element<'_, P::Message> {
		column![
			self.current_error.as_ref().map(|error| text(error).color(ERROR_COLOR)),
			self.view.view(),
		].into()
	}

	/// The update function for iced
	/// 
	/// # Panics
	/// 
	/// This function will panic if the sender is disconnected.
	pub fn update(&mut self, message: P::Message) {
		match self.error_receiver.try_recv() {
			Ok(error) => {
				self.current_error = Some(error);
			},
			Err(crossbeam_channel::TryRecvError::Empty) => {},
			Err(crossbeam_channel::TryRecvError::Disconnected) => {
				self.current_error = Some("Error: Sender disconnected".to_string());
			}
		}

		self.view.update(&message);
		self.sender.send(message).expect("Sender disconnected");
	}
}