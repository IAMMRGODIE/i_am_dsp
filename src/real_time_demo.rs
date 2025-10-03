//! This is a simple real-time audio processing demo.
//! 
//! To start with, see [`DspDemo`]

use crate::prelude::*;
use std::{collections::{HashMap, HashSet}, sync::{Arc, Mutex}};

use anyhow::{anyhow, Result};
use cpal::{traits::{DeviceTrait, HostTrait, StreamTrait}, StreamConfig};
use crossbeam_channel::{Receiver, TryRecvError};
use egui::{Key, Slider};

use crate::{Effect, Generator};

type GneratorConstructor = Box<dyn (Fn(usize) -> Box<dyn Generator>) + Send + Sync>;
type EffectConstructor = Box<dyn (Fn(usize) -> Box<dyn Effect>) + Send + Sync>;

struct SimpleContext {
	info: ProcessInfos,
	midi_events: Vec<NoteEvent>,
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

lazy_static::lazy_static! {
	/// The List of available Generators
	/// 
	/// You can also add your own Generator by adding a tuple to this list.
	pub static ref GENERATOR_LIST: Vec<(String, GneratorConstructor)> = {
		let mut list: Vec<(String, GneratorConstructor)> = vec![
			(
				"Resampler".to_string(),
				Box::new(|sample_rate| {
					let sampler = Sampler::new(sample_rate);
					Box::new(Adsr::new(sampler, EqualTemperament, sample_rate)) as Box<dyn Generator>
				})
			),
			(
				"Player".to_string(),
				Box::new(|sample_rate| {
					Box::new(Sampler::new(sample_rate)) as Box<dyn Generator>
				})
			),
			(
				"Basic Shapes".to_string(),
				Box::new(|sample_rate| {
					let tables: Vec<Box<dyn WaveTable + Send + Sync>> = vec![
						Box::new(SineWave),
						Box::new(TriangleWave),
						Box::new(SawWave),
						Box::new(SquareWave),
					];
					let smoother = WaveTableSmoother {
						tables,
						smooth_factor: 0.0,
						// linear_interp: true,
					};
					Box::new(Adsr::new(smoother, EqualTemperament, sample_rate)) as Box<dyn Generator>
				})
			),
			(
				"Noise Wave".to_string(),
				Box::new(|sample_rate| Box::new(Adsr::new(NoiseWave, EqualTemperament, sample_rate)) as Box<dyn Generator>)
			),
			(
				"Additive(Saw Bend)".to_string(),
				Box::new(|sample_rate| {
					let gen_freq = BendedSawGen::default();
					let add: AdditiveOsc<BendedSawGen> = AdditiveOsc::new(gen_freq, 64);
					Box::new(Adsr::new(add, EqualTemperament, sample_rate)) as Box<dyn Generator>
				})
			)
		];

		#[cfg(feature = "rhai")]
		{
			list.push((
				"Additive(Rhai)".to_string(),
				Box::new(|sample_rate| {
					let rhai_gen = RhaiFreqGen::new(sample_rate);
					let add: AdditiveOsc<RhaiFreqGen> = AdditiveOsc::new(rhai_gen, 64);
					Box::new(Adsr::new(add, EqualTemperament, sample_rate)) as Box<dyn Generator>
				})
			));
		}

		list.sort_by(|a, b| a.0.cmp(&b.0));

		list
	};

	/// The List of available Effects
	/// 
	/// You can also add your own Effect by adding a tuple to this list.
	pub static ref EFFECT_LIST: Vec<(String, Vec<(String, EffectConstructor)>)> = {
		let mut filter_effects: Vec<(String, EffectConstructor)> = vec![
			(
				"Biquad".to_string(),
				Box::new(|sample_rate| Box::new(Biquad::new(sample_rate)) as Box<dyn Effect>)
			),
			(
				"Disperser".to_string(),
				Box::new(|sample_rate| Box::new(Disperser::new(sample_rate)) as Box<dyn Effect>)
			),
			(
				"IIR Hilbert Transform".to_string(),
				Box::new(|sample_rate| Box::new(HilbertTransform::<8>::new(sample_rate)) as Box<dyn Effect>)
			),
		];

		filter_effects.sort_by(|a, b| a.0.cmp(&b.0));

		// let mut envelope_effects: Vec<(String, EffectConstructor)> = vec![
		// 	(
		// 		"IIR Hilbert Enveloper".to_string(),
		// 		Box::new(|sample_rate| {
		// 			let env = IIRHilbertEnvelope::<8>::new(sample_rate);
		// 			let env_history = EnvelopeWithHistory::new(env, 32768);
		// 			Box::new(env_history) as Box<dyn Effect> 
		// 		})
		// 	),
		// 	(
		// 		"FIR Hilbert Enveloper".to_string(),
		// 		Box::new(|_| {
		// 			let env = FIRHilbertEnvelope::new(127);
		// 			let env_history = EnvelopeWithHistory::new(env, 32768);
		// 			Box::new(env_history) as Box<dyn Effect> 
		// 		})
		// 	),
		// 	(
		// 		"Rectify Enveloper".to_string(),
		// 		Box::new(|sample_rate| {
		// 			let env = RectifyEnvelope::new(sample_rate, 1000.0);
		// 			let env_history = EnvelopeWithHistory::new(env, 32768);
		// 			Box::new(env_history) as Box<dyn Effect> 
		// 		})
		// 	),
		// 	(
		// 		"Peak Detector".to_string(),
		// 		Box::new(|sample_rate| {
		// 			let env = PeakDetector::new(sample_rate, 46.0, 200.0);
		// 			let env_history = EnvelopeWithHistory::new(env, 32768);
		// 			Box::new(env_history) as Box<dyn Effect> 
		// 		})
		// 	),
		// 	(
		// 		"TKEO Envelope".to_string(),
		// 		Box::new(|_| {
		// 			let env = TkeoEnvelope::new();
		// 			let env_history = EnvelopeWithHistory::new(env, 32768);
		// 			Box::new(env_history) as Box<dyn Effect> 
		// 		})
		// 	),
		// ];
		// envelope_effects.sort_by(|a, b| a.0.cmp(&b.0));
		
		let mut convolver_effects: Vec<(String, EffectConstructor)> = vec![
			(
				"Convolver".to_string(),
				Box::new(|_| {
					let hilbert_transform = hilbert_transform::<2>(511);
					let output = Convolver::new(hilbert_transform, &DelyaCaculateMode::Fir);
					Box::new(output) as Box<dyn Effect> 
				})
			)
		];

		convolver_effects.sort_by(|a, b| a.0.cmp(&b.0));

		let mut other_effects: Vec<(String, EffectConstructor)> = vec![
			(
				"IIR Frequency Shifter".to_string(),
				Box::new(|sample_rate| {
					let effect = IIRFreqShifter::<8>::new(sample_rate, 0.0);
					Box::new(effect) as Box<dyn Effect> 
				})
			),
			(
				"FIR Frequency Shifter".to_string(),
				Box::new(|sample_rate| {
					let effect = FIRFreqShifter::new(sample_rate, 0.0);
					Box::new(effect) as Box<dyn Effect> 
				})
			),
			(
				"FFT Frequency Shifter".to_string(),
				Box::new(|sample_rate| {
					let effect = PhaseVocoder::new(FreqShift::default(), 1024, sample_rate);
					Box::new(effect) as Box<dyn Effect> 
				})
			),
			(
				"Stereo Controller ".to_string(),
				Box::new(|_| Box::new(StereoController::default()) as Box<dyn Effect>)
			),
			(
				"Gain".to_string(),
				Box::new(|_| Box::new(Gain::default()) as Box<dyn Effect>)
			),
			(
				"Phaser".to_string(),
				Box::new(|sample_rate| {
					let effect = Phaser::new(SineWave, 10, sample_rate);
					Box::new(effect) as Box<dyn Effect> 
				})
			),
			(
				"LrComtroller".to_string(),
				Box::new(|_| Box::new(LrControl::default()) as Box<dyn Effect>)
			),
			(
				"Pitch Shifter(WSOLA)".to_string(),
				Box::new(|_| Box::new(PitchShifter::default()) as Box<dyn Effect>)
			),
			(
				"Pitch Shifter(Phase Vocoder)".to_string(),
				Box::new(|sample_rate| Box::new(PhaseVocoder::new(PitchShift::default(), 1024, sample_rate)) as Box<dyn Effect>)
			),
		];

		other_effects.sort_by(|a, b| a.0.cmp(&b.0));

		let mut compresser_effects: Vec<(String, EffectConstructor)> = vec![
			(
				"Compressor(IIR Hilbert Transform)".to_string(),
				Box::new(|sample_rate| {
					let env = IIRHilbertEnvelope::<8>::new(sample_rate);
					let env = EnvelopeWithHistory::new(env, 32768);
					let effect = Compressor::new(env, sample_rate, 46.0, 200.0, -3.0, 5.0);
					Box::new(effect) as Box<dyn Effect> 
				})
			),
			(
				"Compressor(FIR Hilbert Transform)".to_string(),
				Box::new(|sample_rate| {
					let env = FIRHilbertEnvelope::new(127);
					let env = EnvelopeWithHistory::new(env, 32768);
					let effect = Compressor::new(env, sample_rate, 46.0, 200.0, -3.0, 5.0);
					Box::new(effect) as Box<dyn Effect> 
				})
			),
			(
				"Compressor(Rectify Envelope)".to_string(),
				Box::new(|sample_rate| {
					let env = RectifyEnvelope::new(sample_rate, 1000.0);
					let env = EnvelopeWithHistory::new(env, 32768);
					let effect = Compressor::new(env, sample_rate, 46.0, 200.0, -3.0, 5.0);
					Box::new(effect) as Box<dyn Effect> 
				})
			),
			(
				"Compressor(Peak Detector)".to_string(),
				Box::new(|sample_rate| {
					let env = PeakDetector::new(sample_rate, 46.0, 200.0);
					let env = EnvelopeWithHistory::new(env, 32768);
					let effect = Compressor::new(env, sample_rate, 46.0, 200.0, -3.0, 5.0);
					Box::new(effect) as Box<dyn Effect> 
				})
			),
			(
				"Compressor(TKEO Envelope)".to_string(),
				Box::new(|sample_rate| {
					let env = TkeoEnvelope::new();
					let env = EnvelopeWithHistory::new(env, 32768);
					let effect = Compressor::new(env, sample_rate, 46.0, 200.0, -3.0, 5.0);
					Box::new(effect) as Box<dyn Effect> 
				})
			),
		];

		compresser_effects.sort_by(|a, b| a.0.cmp(&b.0));

		let mut distortion_effects: Vec<(String, EffectConstructor)> = vec![
			(
				"Hard Clipper".to_string(),
				Box::new(|_| Box::new(HardClipper::new(0.8, 1.0)) as Box<dyn Effect>)
			),
			(
				"Soft Clipper".to_string(),
				Box::new(|_| Box::new(SoftClipper::new(0.8, 1.0)) as Box<dyn Effect>)
			),
			(
				"Overdrive".to_string(),
				Box::new(|_| Box::new(Overdrive::new(1.0, 1.0)) as Box<dyn Effect>)
			),
			(
				"Bit Crusher".to_string(),
				Box::new(|_| Box::new(BitCrusher::new(131072, 1.0)) as Box<dyn Effect>)
			),
			(
				"Saturator".to_string(),
				Box::new(|_| Box::new(Saturator::new(5.0, 2.0, 1.0)) as Box<dyn Effect>)
			),
			(
				"Downsampler".to_string(),
				Box::new(|sample_rate| Box::new(Downsampler::new(sample_rate)) as Box<dyn Effect>)
			)
		];

		distortion_effects.sort_by(|a, b| a.0.cmp(&b.0));

		let mut delay_effects: Vec<(String, EffectConstructor)> = vec![
			(
				"Delay".to_string(),
				Box::new(|sample_rate| {
					let delay = Delay::new((), 65536, 50.0, sample_rate);
					Box::new(delay) as Box<dyn Effect> 
				})
			),
			(
				"Pure Delay".to_string(),
				Box::new(|sample_rate| {
					let delay = PureDelay::new(65536, 50.0, sample_rate);
					Box::new(delay) as Box<dyn Effect> 
				})
			),
			(
				"Flanger".to_string(),
				Box::new(|sample_rate| {
					let effect = Flanger::new(8192, sample_rate, SineWave);
					Box::new(effect) as Box<dyn Effect> 
				})
			),
			(
				"Chorus".to_string(),
				Box::new(|sample_rate| {
					let effect = Chorus::new(8192, 10, sample_rate, SineWave);
					Box::new(effect) as Box<dyn Effect> 
				})
			),
			(
				"Reverb".to_string(),
				Box::new(|sample_rate| {
					let effect: Reverb<Biquad<2>, 8, 2> = Reverb::new(
						Biquad::new(sample_rate), 
						sample_rate, 
						Default::default(),
						10,
						10.0
					);
					Box::new(effect) as Box<dyn Effect> 
				})
			)
		];

		delay_effects.sort_by(|a, b| a.0.cmp(&b.0));

		let mut visual_effects: Vec<(String, EffectConstructor)> = vec![
			(
				"Waveform".to_string(),
				Box::new(|_| {
					let effect = Waveform::<2>::new(32768);
					Box::new(effect) as Box<dyn Effect> 
				})
			),
		];

		visual_effects.sort_by(|a, b| a.0.cmp(&b.0));

		vec![
			("Compressor".to_string(), compresser_effects),
			("Convolver".to_string(), convolver_effects),
			("Delay/Reverb".to_string(), delay_effects),
			("Distortion".to_string(), distortion_effects),
			// ("Envelope".to_string(), envelope_effects),
			("Filter".to_string(), filter_effects), 
			("Other".to_string(), other_effects),
			("Visual".to_string(), visual_effects),
			// ("Vocoder".to_string(), vocoder_effects),
		]
	};
}

/// The DspDemo struct
/// 
/// Saves the state of the DSP demo and provides methods to update and draw the UI.
pub struct DspDemo {
	_stream: cpal::Stream,
	error_receiver: Receiver<String>,
	current_error: Option<String>,

	shared_data: Arc<Mutex<SharedData>>,
	sample_rate: usize,
	key_holding: HashSet<u8>,
	note_shift: i8,
}

struct SharedData {
	generator: Option<Box<dyn Generator>>,
	effects: Vec<(Box<dyn Effect>, f32)>,
	ctx: Option<SimpleContext>,
}

impl DspDemo {
	/// Create a new DspDemo instance
	/// 
	/// Error may occur if no output device is available or if the stream cannot be created.
	pub fn new() -> Result<Self> {
		// let mut delay_buffers = [RingBuffer::new(0), RingBuffer::new(0)];

		let (error_sender, error_receiver) = crossbeam_channel::unbounded();
		let stream_error_sender = error_sender.clone();
	

		let host = cpal::default_host();
		let device = host.default_output_device()
			.ok_or_else(|| anyhow!("No output device available"))?;

		let config = device.default_output_config()?;
		let config = StreamConfig::from(config);
		let sample_rate = config.sample_rate.0 as usize;

		let shared_data = Arc::new(Mutex::new(SharedData {
			generator: None,
			effects: vec![],
			ctx: Some(SimpleContext {
				info: ProcessInfos {
					sample_rate,
					trustable: true,
					playing: true,
					tempo: 150.0,
					current_bar_number: None,
					time_signature: None,
					current_time: 0.0,
				},
				midi_events: vec![],
			}),
		}));

		let shared_data_stream = shared_data.clone();
		let stream = device.build_output_stream(
			&config,
			move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
				let lock_result = shared_data_stream.lock();
				let mut shared_data = match lock_result {
					Ok(data) => data,
					// Err(TryLockError::WouldBlock) => return,
					Err(_) => {
						stream_error_sender.send("Shared data has been poisoned".to_string()).expect("canot send error message");
						return;
					},
				};

				// let delay = shared_data.effects.iter().map(|inner| inner.delay()).sum::<usize>();
				// if delay != delay_buffers[0].capacity() {
				// 	delay_buffers[0] = RingBuffer::new(delay);
				// 	delay_buffers[1] = RingBuffer::new(delay);
				// }

				for val in data.chunks_mut(2) {
					if val.len() != 2 {
						stream_error_sender.send("Received invalid number of samples".to_string()).expect("canot send error message");
						break;
					}

					let mut ctx: Box<dyn ProcessContext> = Box::new(shared_data.ctx.take().unwrap());
					let mut output = if let Some(generator) = &mut shared_data.generator {
						generator.generate(&mut ctx)
					}else {
						[0.0; 2]
					};

					for (effect, mix) in &mut shared_data.effects {
						let output_backup = output;
						effect.process(&mut output, &[], &mut ctx);
						output[0] = output[0] * *mix + output_backup[0] * (1.0 - *mix);
						output[1] = output[1] * *mix + output_backup[1] * (1.0 - *mix);
					}

					shared_data.ctx = Some(SimpleContext {
						info: ProcessInfos {
							sample_rate,
							trustable: true,
							playing: true,
							tempo: 150.0,
							current_bar_number: None,
							time_signature: None,
							current_time: 0.0,
						},
						midi_events: vec![],
					});

					// let (output_l, output_r) = if delay != 0 {
					// 	delay_buffers[0].push(output[0]);
					// 	delay_buffers[1].push(output[1]);
					// 	(delay_buffers[0][0], delay_buffers[1][0])
					// }else {
					// 	(output[0], output[1])
					// };
					val[0] = output[0];
					val[1] = output[1];
				}
			},
			move |err| {
				error_sender.send(format!("Stream error: {}", err)).expect("canot send error message");
			},
			None 
		)?;

		stream.play()?;

		Ok(Self {
			_stream: stream,
			error_receiver,
			current_error: None,
			shared_data,
			sample_rate,
			key_holding: HashSet::new(),
			note_shift: 0,
		})
	}

	/// Draw the UI
	pub fn ui(&mut self, ui: &mut egui::Ui) {
		if let Some(err_msg) = &self.current_error {
			ui.colored_label(egui::Color32::RED, format!("ERR: {err_msg}"));
			if ui.button("Clear Error").clicked() {
				self.current_error = None;
			}
			ui.separator();
		}
		let recv_result = self.error_receiver.try_recv();

		match recv_result {
			Ok(err_msg) => self.current_error = Some(err_msg),
			Err(TryRecvError::Empty) => {}
			Err(TryRecvError::Disconnected) => self.current_error = Some("Error CHANNELS disconnected".to_string()),
		};
		let lock_result = self.shared_data.lock();
		let mut shared_data = match lock_result {
			Ok(data) => data,
			// Err(TryLockError::WouldBlock) => return,
			Err(_) => {
				self.current_error = Some("Shared data has been poisoned".to_string());
				return;
			},
		};

		let mut clear_all_notes = false;
		egui::TopBottomPanel::bottom("midi_panel").show_inside(ui, |ui| {
			let mut key_holding = self.key_holding.iter().copied().collect::<Vec<_>>();
			key_holding.sort();
			let keys = key_holding.into_iter().map(format_note).collect::<Vec<_>>();
			ui.horizontal(|ui| {
				let mut shift = self.note_shift;
				ui.add(Slider::new(&mut shift, -36..=36).text("Note Shift"));
				if shift != self.note_shift {
					self.note_shift = shift;
					clear_all_notes = true;
				}

				if ui.button("Clear All Notes").clicked() {
					clear_all_notes = true;
				}
				if keys.is_empty() {
					ui.label("no note playing")
				}else {
					ui.label(format!("Playing: [{}]", keys.join(", ")))
				}
			});
		});
		
		let mut ctx = shared_data.ctx.take().unwrap();
		simulate_midi(ui, &mut ctx, &mut self.key_holding, clear_all_notes, self.note_shift);
		shared_data.ctx = Some(ctx);

		egui::ScrollArea::vertical().show(ui, |ui| {
			if let Some(generator) = &mut shared_data.generator {
				ui.heading(generator.name());
				let prefix = format!("Generator: {}", generator.name());
				generator.demo_ui(ui, prefix);
				if ui.button("Remove Generator").clicked() {
					shared_data.generator = None;
				}
				ui.menu_button("Replace Generator", |ui| {
					generator_add_menu(ui, &mut shared_data, self.sample_rate);
				});
			}else {
				ui.heading("Generator");
				ui.label("No generator Loaded");
				ui.menu_button("Add Generator", |ui| {
					generator_add_menu(ui, &mut shared_data, self.sample_rate);
				});
			}
			ui.separator();

			ui.heading("Effects");
			let effects_len = shared_data.effects.len();
			let mut should_remove = None;
			let mut should_swap = None;

			for (i, effect) in shared_data.effects.iter_mut().enumerate() {
				let name = effect.0.name().to_string();
				ui.collapsing(format!("Effect {i}: {name}"), |ui| {
					let prefix = format!("Effect {i}: {name}");
					effect.0.demo_ui(ui, prefix);
					if i > 0 {
						let btn = ui.button("Move Up");
						if btn.clicked() {
							should_swap = Some((i, i - 1));
						}
					}
					if i < effects_len - 1 {
						let btn = ui.button("Move Down");
						if btn.clicked() {
							should_swap = Some((i, i + 1));
						}
					}
					ui.add(egui::Slider::new(&mut effect.1, 0.0..=1.0).text("Mix"));
					if ui.button("Remove").clicked() {
						should_remove = Some(i);
					}
				});
				ui.separator();
			}
			if let Some(i) = should_remove {
				shared_data.effects.remove(i);
			}
			if let Some((i, j)) = should_swap {
				shared_data.effects.swap(i, j);
			}
			if let Some(Some(inner)) = ui.menu_button("Add Effect", |ui| {
				effect_add_menu(ui, self.sample_rate)
			}).inner {
				shared_data.effects.push((inner, 1.0));
			}
		});
	}

	/// Run the DSP demo
	pub fn run(self, native_options: eframe::NativeOptions) -> Result<()> {
		eframe::run_native("Dsp Demo", native_options, Box::new(|_| Ok(Box::new(self))))
			.map_err(|e| anyhow!("{}", e))?;
		Ok(())
	}
}

impl eframe::App for DspDemo {
	fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
		egui::CentralPanel::default().show(ctx, |ui| {
			self.ui(ui);
		});
	
		ctx.request_repaint();
	}
}

fn generator_add_menu(ui: &mut egui::Ui, shared_data: &mut SharedData, sample_rate: usize) {
	for (name, generator) in GENERATOR_LIST.iter() {
		if ui.button(name).clicked() {
			shared_data.generator = Some(generator(sample_rate));
			ui.close();
		}
	}

	if ui.button("Close").clicked() {
		ui.close();
	}
}

fn effect_add_menu(ui: &mut egui::Ui, sample_rate: usize) -> Option<Box<dyn Effect>> {
	let mut output = None;

	for (name, effect) in EFFECT_LIST.iter() {
		ui.menu_button(name, |ui| {
			for (name, effect) in effect.iter() {
				if ui.button(name).clicked() {
					output = Some(effect(sample_rate));
					ui.close();
				}
			}
		});
	}

	if ui.button("Close").clicked() {
		ui.close();
	}

	output
}

lazy_static::lazy_static! {
	static ref NoteMap: HashMap<Key, u8> = {
		const C5: u8 = 60;
		const C4: u8 = 60 - 12;

		HashMap::from([
			(Key::Q, C5),
			(Key::Num2, C5 + 1),
			(Key::W, C5 + 2),
			(Key::Num3, C5 + 3),
			(Key::E, C5 + 4),
			(Key::R, C5 + 5),
			(Key::Num5, C5 + 6),
			(Key::T, C5 + 7),
			(Key::Num6, C5 + 8),
			(Key::Y, C5 + 9),
			(Key::Num7, C5 + 10),
			(Key::U, C5 + 11),
			(Key::I, C5 + 12),
			(Key::Num9, C5 + 13),
			(Key::O, C5 + 14),
			(Key::Num0, C5 + 15),
			(Key::P, C5 + 16),

			(Key::Z, C4),
			(Key::S, C4 + 1),
			(Key::X, C4 + 2),
			(Key::D, C4 + 3),
			(Key::C, C4 + 4),
			(Key::V, C4 + 5),
			(Key::G, C4 + 6),
			(Key::B, C4 + 7),
			(Key::H, C4 + 8),
			(Key::N, C4 + 9),
			(Key::J, C4 + 10),
			(Key::M, C4 + 11),
		])
	};
}
	
fn simulate_midi(
	ui: &mut egui::Ui, 
	ctx: &mut SimpleContext, 
	key_holding: &mut HashSet<u8>,
	clear_all: bool,
	note_shift: i8,
) {
	ui.input(|input| {
		for (key, note) in NoteMap.iter() {
			if input.key_down(*key) {
				let note = if note_shift >= 0 {
					note + note_shift as u8
				}else {
					note.saturating_sub(note_shift.unsigned_abs())
				};
				if key_holding.insert(note) {
					// println!("semi-note on: {}", note);
					ctx.send_event(NoteEvent::NoteOn { time: 0, channel: 0, note, velocity: 1.0 });
				}
			}else if input.key_released(*key) {
				let note = if note_shift >= 0 {
					note + note_shift as u8
				}else {
					note.saturating_sub(note_shift.unsigned_abs())
				};
				if key_holding.remove(&note) {
					// println!("semi-note off: {}", note);
					ctx.send_event(NoteEvent::NoteOff { time: 0, channel: 0, note, velocity: 1.0 });
				}
			}
		}
	});
	if clear_all {
		key_holding.clear();
		ctx.send_event(NoteEvent::ImmediateStop);
	}
}