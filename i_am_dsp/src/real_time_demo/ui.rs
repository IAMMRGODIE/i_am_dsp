use anyhow::{Result, anyhow};
use egui::{Grid, Slider, vec2};

use super::menus::{draw_change_history, effect_add_menu, generator_add_menu, get_range};
use super::*;

impl DspDemo {
	/// Create a new DspDemo instance
	///
	/// Error may occur if no output device is available or if the stream cannot be created.
	///
	/// User should provide the sample rate if `standalone` feature is disabled.
	pub fn new(sample_rate: Option<usize>) -> Result<Self> {
		// let mut delay_buffers = [RingBuffer::new(0), RingBuffer::new(0)];

		cfg_if::cfg_if! {
			if #[cfg(feature = "standalone")] {
				let _ = sample_rate;
				let host = cpal::default_host();
				let device = host.default_output_device()
					.ok_or_else(|| anyhow!("No output device available"))?;

				let config = device.default_output_config()?;
				let config = StreamConfig::from(config);
				let sample_rate = config.sample_rate as usize;
				let (error_sender, error_receiver) = crossbeam_channel::unbounded();
			}else {
				let Some(sample_rate) = sample_rate else {
					return Err(anyhow!("No sample rate provided"));
				};
			}
		}

		let shared_data = Arc::new(Mutex::new(SharedData {
			generators: vec![],
			effects: std::array::from_fn(|_| vec![]),
			track_delay: std::array::from_fn(|_| std::array::from_fn(|_| RingBuffer::new(0))),
			sample_rate,
			current_phase: 0.0,
			lfos: vec![],

			#[cfg(feature = "standalone")]
			ctx: Some(SimpleContext {
				info: ProcessInfos {
					sample_rate,
					trustable: false,
					playing: true,
					tempo: Some(150.0),
					time_signature: None,
					current_time: 0.0,
					// `position_beats` (and the deprecated `current_beat_number`) stay at their
					// default: a standalone context has no host timeline to report yet.
					..ProcessInfos::default()
				},
				midi_events: vec![],
			}),
		}));

		cfg_if::cfg_if! {
			if #[cfg(feature = "standalone")] {
				let stream_error_sender = error_sender.clone();

				let shared_data_stream = shared_data.clone();

				let stream = device.build_output_stream(
					config,
					move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
						// Reverb and convolution tails decay through the subnormal range,
						// where x86 arithmetic is an order of magnitude slower.
						crate::enable_flush_to_zero();
						let lock_result = shared_data_stream.lock();
						let mut shared_data = match lock_result {
							Ok(data) => data,
							// Err(TryLockError::WouldBlock) => return,
							Err(_) => {
								stream_error_sender.send("Shared data has been poisoned".to_string()).expect("canot send error message");
								return;
							},
						};

						let mut ctx: Box<dyn ProcessContext> = Box::new(shared_data.ctx.take().unwrap());

						for val in data.chunks_mut(2) {
							if val.len() != 2 {
								stream_error_sender.send("Received invalid number of samples".to_string()).expect("canot send error message");
								break;
							}

							let output = shared_data.generate(&mut ctx);
							ctx.clear_events();

							val[0] = output[0];
							val[1] = output[1];
						}

						shared_data.ctx = Some(SimpleContext {
							info: ProcessInfos {
								sample_rate,
								trustable: false,
								playing: true,
								tempo: Some(150.0),
								time_signature: None,
								current_time: 0.0,
								..ProcessInfos::default()
							},
							midi_events: vec![],
						});
					},
					move |err| {
						error_sender.send(format!("Stream error: {}", err)).expect("canot send error message");
					},
					None
				)?;

				stream.play()?;
			}
		}

		Ok(Self {
			#[cfg(feature = "standalone")]
			_stream: stream,
			#[cfg(feature = "standalone")]
			error_receiver,
			current_error: None,
			shared_data,
			sample_rate,
			#[cfg(feature = "standalone")]
			key_holding: HashSet::new(),
			#[cfg(feature = "standalone")]
			note_shift: 0,
			changed_history: RingBuffer::new(CHANGE_HISTORY_LEN),
			current_track_effect: 0,
		})
	}

	/// Get the delay of the current state.
	pub fn delay(&self) -> Result<usize> {
		let lock_result = self.shared_data.lock();
		let shared_data = match lock_result {
			Ok(data) => data,
			// Err(TryLockError::WouldBlock) => return,
			Err(_) => {
				return Err(anyhow!("Shared data has been poisoned"));
			}
		};
		Ok(shared_data.delay())
	}

	/// Generate a single sample.
	pub fn generate(&mut self, ctx: &mut Box<dyn ProcessContext>) -> [f32; 2] {
		let lock_result = self.shared_data.lock();
		let mut shared_data = match lock_result {
			Ok(data) => data,
			// Err(TryLockError::WouldBlock) => return,
			Err(_) => {
				self.current_error = Some("Shared data has been poisoned".to_string());
				return [0.0, 0.0];
			}
		};

		shared_data.generate(ctx)
	}

	/// Get an iterator of parameter changes.
	pub fn change_history(&self) -> impl Iterator<Item = &ListenedValue> {
		self.changed_history
			.range(0..CHANGE_HISTORY_LEN)
			.filter_map(|inner| inner.as_ref())
	}

	/// Get a value from a generator, only numerical values are supported.
	///
	/// Return value is in normalized range [0.0, 1.0].
	pub fn get_generator_value_normalized(
		&self,
		index: usize,
		identifier: &str,
	) -> Result<Option<f32>> {
		let lock_result = self.shared_data.lock();
		let shared_data = match lock_result {
			Ok(data) => data,
			// Err(TryLockError::WouldBlock) => return,
			Err(_) => {
				return Err(anyhow!("Shared data has been poisoned"));
			}
		};
		if let Some(generator) = shared_data.generators.get(index) {
			let parameters = generator.generator.get_parameters();
			if let Some(parameter) = parameters
				.iter()
				.find(|param| param.identifier == identifier)
			{
				return match &parameter.value {
					Value::Float { value, range, .. } => {
						let normalized =
							(*value - *range.start()) / (*range.end() - *range.start());
						Ok(Some(normalized))
					}
					Value::Int { value, range, .. } => {
						let normalized = (*value as f32 - *range.start() as f32)
							/ (*range.end() - *range.start()) as f32;
						Ok(Some(normalized))
					}
					_ => Ok(None),
				};
			}
		}
		Ok(None)
	}

	/// Get a value from an effect, only numerical values are supported.
	pub fn get_effect_value_normalized(
		&self,
		track: usize,
		index: usize,
		identifier: &str,
	) -> Result<Option<f32>> {
		let lock_result = self.shared_data.lock();
		let shared_data = match lock_result {
			Ok(data) => data,
			// Err(TryLockError::WouldBlock) => return,
			Err(_) => {
				return Err(anyhow!("Shared data has been poisoned"));
			}
		};

		if let Some(effects) = shared_data.effects.get(track)
			&& let Some(effect) = effects.get(index)
		{
			let parameters = effect.effect.get_parameters();
			if let Some(parameter) = parameters
				.iter()
				.find(|param| param.identifier == identifier)
			{
				return match &parameter.value {
					Value::Float { value, range, .. } => {
						let normalized =
							(*value - *range.start()) / (*range.end() - *range.start());
						Ok(Some(normalized))
					}
					Value::Int { value, range, .. } => {
						let normalized = (*value as f32 - *range.start() as f32)
							/ (*range.end() - *range.start()) as f32;
						Ok(Some(normalized))
					}
					_ => Ok(None),
				};
			}
		}

		Ok(None)
	}

	/// Change the value of a parameter.
	///
	/// Returns `true` if the value is changed, `false`
	/// false will occur if the parameter is not found.
	pub fn change_value(&mut self, value: ChangedValue) -> Result<bool> {
		let lock_result = self.shared_data.lock();
		let mut shared_data = match lock_result {
			Ok(data) => data,
			// Err(TryLockError::WouldBlock) => return,
			Err(_) => {
				return Err(anyhow!("Shared data has been poisoned"));
			}
		};
		match value {
			ChangedValue::Effect {
				track,
				index,
				identifier,
				change_to,
			} => {
				if track > MAX_OUTPUT_TRACKS {
					return Ok(false);
				}

				let Some(effects) = shared_data.effects[track].get_mut(index) else {
					return Ok(false);
				};

				Ok(effects.effect.set_parameter(&identifier, change_to))
			}
			ChangedValue::Generator {
				index,
				identifier,
				change_to,
			} => {
				let Some(generators) = shared_data.generators.get_mut(index) else {
					return Ok(false);
				};

				Ok(generators.generator.set_parameter(&identifier, change_to))
			}
		}
	}

	/// Get persist data of the current state.
	pub fn persist_data(&self) -> Result<Vec<PersistData>> {
		let lock_result = self.shared_data.lock();
		let shared_data = match lock_result {
			Ok(data) => data,
			// Err(TryLockError::WouldBlock) => return,
			Err(_) => {
				return Err(anyhow!("Shared data has been poisoned"));
			}
		};
		Ok(shared_data.get_persist_data())
	}

	/// Change the sample rate of the current state.
	///
	/// Note: this function will recreate the state with the new sample rate, so it may take some time to complete.
	pub fn change_sample_rate(&mut self, sample_rate: usize) -> Result<()> {
		if self.sample_rate == sample_rate {
			return Ok(());
		}

		let lock_result = self.shared_data.lock();
		let mut shared_data = match lock_result {
			Ok(data) => data,
			// Err(TryLockError::WouldBlock) => return,
			Err(_) => {
				return Err(anyhow!("Shared data has been poisoned"));
			}
		};
		let shard_persisted_data = shared_data.get_persist_data();
		*shared_data = SharedData::from_persist_data(shard_persisted_data, sample_rate);
		self.sample_rate = sample_rate;

		Ok(())
	}

	/// Load persist data to the current state.
	pub fn load_persist_data(&mut self, data: Vec<PersistData>) -> Result<()> {
		let shared_data = SharedData::from_persist_data(data, self.sample_rate);
		let lock_result = self.shared_data.lock();
		let mut shared_data_old = match lock_result {
			Ok(data) => data,
			// Err(TryLockError::WouldBlock) => return,
			Err(_) => {
				return Err(anyhow!("Shared data has been poisoned"));
			}
		};
		*shared_data_old = shared_data;
		Ok(())
	}

	/// Draw the UI
	pub fn ui(&mut self, ui: &mut egui::Ui) {
		ui.ctx().set_visuals(egui::Visuals::dark());

		if let Some(err_msg) = &self.current_error {
			ui.colored_label(egui::Color32::RED, format!("ERR: {err_msg}"));
			if ui.button("Clear Error").clicked() {
				self.current_error = None;
			}
			ui.separator();
		}

		cfg_if::cfg_if! {
			if #[cfg(feature = "standalone")] {
				let recv_result = self.error_receiver.try_recv();

				match recv_result {
					Ok(err_msg) => self.current_error = Some(err_msg),
					Err(TryRecvError::Empty) => {}
					Err(TryRecvError::Disconnected) => self.current_error = Some("Error Channels disconnected".to_string()),
				};
			}
		}
		let lock_result = self.shared_data.lock();
		let mut shared_data = match lock_result {
			Ok(data) => data,
			// Err(TryLockError::WouldBlock) => return,
			Err(_) => {
				self.current_error = Some("Shared data has been poisoned".to_string());
				return;
			}
		};

		egui::Panel::top("state_pan").show(ui, |ui| {
			ui.horizontal(|ui| {
				for track in 0..=MAX_OUTPUT_TRACKS {
					let text = if track == MAX_OUTPUT_TRACKS {
						"Master".to_string()
					} else {
						format!("Track {track}")
					};
					ui.selectable_value(&mut self.current_track_effect, track, text);
				}
			});
		});

		#[cfg(feature = "standalone")]
		let mut clear_all_notes = false;
		egui::Panel::bottom("lfo_panel")
			.resizable(true)
			.show(ui, |ui| {
				egui::Panel::bottom("midi_panel").show(ui, |ui| {
					#[cfg(feature = "standalone")]
					{
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
							} else {
								ui.label(format!("Playing: [{}]", keys.join(", ")))
							}
						});
					}

					ui.separator();
				});
				let current_phase = shared_data.current_phase;
				egui::CentralPanel::default().show(ui, |ui| {
					egui::ScrollArea::vertical().show(ui, |ui| {
						let mut should_remove_lfo = None;
						for (i, lfo) in shared_data.lfos.iter_mut().enumerate() {
							let id_prefix = format!("Lfo {i}: ");
							ui.collapsing(format!("Lfo {i}"), |ui| {
								lfo.lfo.demo_ui(ui, id_prefix);
								let mut should_remove = None;
								ui.collapsing("Params", |ui| {
									Grid::new(format!("Lfo {i} Params")).show(ui, |ui| {
										for (j, param) in lfo.params.iter_mut().enumerate() {
											let name = if let Some(track) = &param.track {
												format!("Effect {} (at track {track})", param.index)
											} else {
												format!("Generator {}", param.index)
											};
											ui.label(name);
											ui.label(&param.identifier);
											if let (
												FloatOrInt::Float(start),
												FloatOrInt::Float(end),
												FloatOrInt::Float(max),
												FloatOrInt::Float(min),
											) = (
												&mut param.start,
												&mut param.end,
												&param.max,
												&param.min,
											) {
												let range = *min..=*max;
												ui.add(
													Slider::new(start, range.clone())
														.text("Start")
														.logarithmic(param.logarithmic),
												);
												ui.add(
													Slider::new(end, range)
														.text("End")
														.logarithmic(param.logarithmic),
												);
											} else if let (
												FloatOrInt::Int(start),
												FloatOrInt::Int(end),
												FloatOrInt::Int(max),
												FloatOrInt::Int(min),
											) = (
												&mut param.start,
												&mut param.end,
												&param.max,
												&param.min,
											) {
												let range = *min..=*max;
												ui.add(
													Slider::new(start, range.clone())
														.text("Start")
														.logarithmic(param.logarithmic),
												);
												ui.add(
													Slider::new(end, range)
														.text("End")
														.logarithmic(param.logarithmic),
												);
											}

											if ui.button("Remove Param").clicked() {
												should_remove = Some(j)
											}
											ui.end_row();
										}
									});
									if let Some(Some(inner)) = ui
										.menu_button("Add Param", |ui| {
											draw_change_history(ui, &self.changed_history)
										})
										.inner
									{
										lfo.params.push(inner);
									}
								});
								if ui.selectable_label(lfo.trig, "Trig").clicked() {
									lfo.trig = !lfo.trig;
									if !lfo.trig {
										lfo.phase = current_phase;
									}
								}
								if let Some(j) = should_remove {
									lfo.params.remove(j);
								}
								if ui.button("Remove Lfo").clicked() {
									should_remove_lfo = Some(i);
								}
							});
						}
						if ui.button("Add Lfo").clicked() {
							let current_phase = shared_data.current_phase;
							shared_data.lfos.push(LfoedParams {
								lfo: EditableLfo::default(),
								params: vec![],
								phase: current_phase,
								trig: false,
							});
						}

						if let Some(i) = should_remove_lfo {
							shared_data.lfos.remove(i);
						}
					});
				});
			});
		#[cfg(feature = "standalone")]
		{
			let mut ctx = shared_data.ctx.take().unwrap();
			simulate_midi(
				ui,
				&mut ctx,
				&mut self.key_holding,
				clear_all_notes,
				self.note_shift,
			);
			shared_data.ctx = Some(ctx);
		}

		ui.allocate_space(vec2(0.0, 4.0));

		egui::ScrollArea::vertical().show(ui, |ui| {
			let mut should_remove = None;
			let mut should_swap = None;
			let mut should_replace = None;
			let generators_len = shared_data.generators.len();
			let track_available = self.current_track_effect.clamp(0, MAX_OUTPUT_TRACKS - 1);

			for (i, generator) in shared_data.generators.iter_mut().enumerate() {
				let name = generator.generator.name().to_string();
				ui.collapsing(format!("Generator {i}: {name}"), |ui| {
					let prefix = format!("Generator: {}", name);
					let params_old = generator.generator.get_parameters();
					generator.generator.demo_ui(ui, prefix);
					let params_new = generator.generator.get_parameters();
					for (param_old, param_new) in params_old.into_iter().zip(params_new) {
						if param_old != param_new {
							let range = get_range(&param_new.value);
							let history = ListenedValue {
								history: ChangedValue::Generator {
									index: i,
									identifier: param_old.identifier,
									change_to: param_new.value.to_set_value(),
								},
								range,
							};
							let len = self.changed_history.capacity();
							if let Some(history_old) = &self.changed_history[len - 1] {
								if !(history_old.history.is_param_same(&history.history)) {
									self.changed_history.push(Some(history));
								}
							} else {
								self.changed_history.push(Some(history));
							}
						}
					}
					gain_ui(ui, &mut generator.gain, None, false);
					ui.add(
						egui::Slider::new(&mut generator.output_track, 0..=MAX_OUTPUT_TRACKS - 1)
							.text("Output Track"),
					);
					if ui.button("Remove Generator").clicked() {
						should_remove = Some(i);
					}
					if i > 0 {
						let btn = ui.button("Move Up");
						if btn.clicked() {
							should_swap = Some((i, i - 1));
						}
					}
					if i < generators_len - 1 && generators_len != 1 {
						let btn = ui.button("Move Down");
						if btn.clicked() {
							should_swap = Some((i, i + 1));
						}
					}
					if let Some(Some(inner)) = ui
						.menu_button("Replace", |ui| {
							generator_add_menu(ui, self.sample_rate, track_available)
						})
						.inner
					{
						should_replace = Some((i, inner));
					};
				});
			}

			if let Some((i, generator)) = should_replace {
				shared_data.generators[i].generator = generator.generator;
				shared_data.generators[i].path = generator.path;
			}
			if let Some((i, j)) = should_swap {
				shared_data.generators.swap(i, j);
			}
			if let Some(i) = should_remove {
				shared_data.generators.remove(i);
			}
			if let Some(Some(inner)) = ui
				.menu_button("Add Generator", |ui| {
					generator_add_menu(ui, self.sample_rate, track_available)
				})
				.inner
			{
				shared_data.generators.push(inner);
			}

			ui.separator();

			ui.heading("Effects");
			let effects_len = shared_data.effects.len();
			let current_track = self.current_track_effect;
			let mut should_remove = None;
			let mut should_swap = None;
			let mut should_replace = None;

			for (i, effect) in shared_data.effects[current_track].iter_mut().enumerate() {
				let name = effect.effect.name().to_string();
				ui.collapsing(format!("Effect {i}: {name}"), |ui| {
					let prefix = format!("Effect {i}: {name}");

					let params_old = effect.effect.get_parameters();
					effect.effect.demo_ui(ui, prefix);
					let params_new = effect.effect.get_parameters();

					for (param_old, param_new) in params_old.into_iter().zip(params_new) {
						if param_old != param_new {
							let range = get_range(&param_new.value);
							let history = ListenedValue {
								history: ChangedValue::Effect {
									track: self.current_track_effect,
									index: i,
									identifier: param_old.identifier,
									change_to: param_new.value.to_set_value(),
								},
								range,
							};

							let len = self.changed_history.capacity();
							if let Some(history_old) = &self.changed_history[len - 1] {
								if !(history_old.history.is_param_same(&history.history)) {
									self.changed_history.push(Some(history));
								}
							} else {
								self.changed_history.push(Some(history));
							}
						}
					}
					if i > 0 {
						let btn = ui.button("Move Up");
						if btn.clicked() {
							should_swap = Some((i, i - 1));
						}
					}
					if i < effects_len - 1 && effects_len != 1 {
						let btn = ui.button("Move Down");
						if btn.clicked() {
							should_swap = Some((i, i + 1));
						}
					}
					ui.add(egui::Slider::new(&mut effect.mix, 0.0..=1.0).text("Mix"));
					if ui.button("Remove").clicked() {
						should_remove = Some(i);
					}
					if let Some(Some(inner)) = ui
						.menu_button("Replace", |ui| effect_add_menu(ui, self.sample_rate))
						.inner
					{
						should_replace = Some((i, inner));
					};
				});
				ui.separator();
			}
			if let Some((i, effect)) = should_replace {
				shared_data.effects[current_track][i].effect = effect.effect;
				shared_data.effects[current_track][i].path = effect.path;
			}
			if let Some((i, j)) = should_swap {
				shared_data.effects[current_track].swap(i, j);
			}
			if let Some(i) = should_remove {
				shared_data.effects[current_track].remove(i);
			}
			if let Some(Some(inner)) = ui
				.menu_button("Add Effect", |ui| effect_add_menu(ui, self.sample_rate))
				.inner
			{
				shared_data.effects[current_track].push(inner);
			}
		});
	}

	/// Run the DSP demo
	#[cfg(feature = "standalone")]
	pub fn run(self, native_options: eframe::NativeOptions) -> Result<()> {
		eframe::run_native("Dsp Demo", native_options, Box::new(|_| Ok(Box::new(self))))
			.map_err(|e| anyhow!("{}", e))?;
		Ok(())
	}
}
