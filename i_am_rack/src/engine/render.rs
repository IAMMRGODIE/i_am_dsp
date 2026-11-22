use super::*;

impl Engine {
	/// Builds the bands described by a state, with scopes of its own. The
	/// plug-in builds through `with_shared`; this is for tests.
	#[cfg(test)]
	pub fn new(sample_rate: usize, state: &RackState) -> Self {
		Self::with_scopes(sample_rate, state, Scopes::default())
	}

	/// Builds the bands a plug-in runs, sharing the module scopes with its page.
	pub fn with_shared(sample_rate: usize, state: &RackState, shared: Arc<RackShared>) -> Self {
		Self::with_scopes(
			sample_rate,
			state,
			Scopes {
				shared: Some(shared),
			},
		)
	}

	fn with_scopes(sample_rate: usize, state: &RackState, scopes: Scopes) -> Self {
		let mut engine = Self {
			lanes: [Lane::new(), Lane::new(), Lane::new()],
			parallel: Lane::new(),
			crossover: Crossover::new(sample_rate, state.split_low, state.split_high),
			modulation: Modulation::new(sample_rate),
			assignments: Vec::new(),
			macros: [0.0; 8],
			source_kinds: BTreeMap::new(),
			rate: sample_rate,
			scopes,
			note: -1,
			held: 0,
			note_tick: 0,
		};

		engine.apply(state);
		engine
	}

	/// Copies the editable state into the running bands.
	pub fn apply(&mut self, state: &RackState) {
		self.crossover.set_splits(state.split_low, state.split_high);

		let mut live = BTreeSet::new();
		for band in &state.bands {
			for module in band.slots.iter().flatten() {
				live.insert(module.id);
			}
		}
		for module in state.parallel.slots.iter().flatten() {
			live.insert(module.id);
		}
		self.scopes.retain(&live);

		for (index, band) in state.bands.iter().enumerate() {
			apply_lane(&mut self.lanes[index], band, self.rate, &self.scopes);
		}

		apply_lane(&mut self.parallel, &state.parallel, self.rate, &self.scopes);

		// A lane is as late as the active modules it runs in series. Give every
		// shorter lane the difference, so a compressor with lookahead in one
		// band does not comb-filter the others.
		let mut delays = [0usize; 4];

		for (index, lane) in self.lanes.iter().enumerate() {
			delays[index] = lane_delay(lane);
		}

		delays[3] = lane_delay(&self.parallel);

		let longest = delays.iter().copied().max().unwrap_or(0);

		for (index, lane) in self.lanes.iter_mut().enumerate() {
			lane.delay = delays[index];
			lane.compensation.set_length(longest - delays[index]);
		}

		self.parallel.delay = delays[3];
		self.parallel.compensation.set_length(longest - delays[3]);

		self.modulation.set_slots(&state.source_slots);
		self.source_kinds.clear();

		for slot in &state.source_slots {
			self.source_kinds.insert(slot.id, slot.kind);
		}

		self.assignments = state.assignments.clone();
		self.macros = state.macros;
		self.apply_automation(state);
	}

	/// Copy the host-written macro and automation values in, without rebuilding
	/// the modules: these move while a project plays.
	pub fn apply_macros(&mut self, state: &RackState) {
		self.macros = state.macros;
		self.apply_automation(state);
	}

	fn apply_automation(&mut self, state: &RackState) {
		for (slot, target) in state.automation.iter().enumerate() {
			let Some(target) = target else {
				continue;
			};
			let value = state.automation_values[slot].clamp(0.0, 1.0) as f64;

			match target {
				Target::Module { module, parameter } => {
					let Some(module) = self.module_mut(*module) else {
						continue;
					};
					let Some(control) = module.control(parameter).cloned() else {
						continue;
					};
					let mapped = denormalize(value, &control);
					module.set_base_target(parameter, mapped);
				}
				Target::BandGain(index) => {
					if let Some(lane) = self.lane_mut(*index) {
						lane.base_gain = value * 2.0;
					}
				}
				Target::BandPan(index) => {
					if let Some(lane) = self.lane_mut(*index) {
						lane.base_pan = value * 2.0 - 1.0;
					}
				}
			}
		}
	}

	/// Splits the frame, runs both slots of every band, and sums what is left.
	pub fn process(
		&mut self,
		samples: &mut [f32; 2],
		other: &[[f32; 2]],
		process_context: &mut Box<dyn ProcessContext>,
	) {
		let input = (samples[0] + samples[1]) * 0.5;
		// The products take the extra inputs as a slice of channel frames; the
		// compressor keys off `other[1]`, which is the rack's sidechain port.
		let sides: Vec<&[f32; 2]> = other.iter().collect();
		let sidechain = other
			.get(1)
			.map(|frame| (frame[0] + frame[1]) * 0.5)
			.unwrap_or(input);

		let events = process_context.events().to_vec();
		// The note the host is holding, so a card can show what its module hears.
		let mut note = self.note;
		let mut held = self.held;

		for event in &events {
			match *event {
				NoteEvent::NoteOn { note: pitch, .. } if pitch < 128 => {
					note = pitch as i32;
					held |= 1u128 << pitch;
				}
				NoteEvent::NoteOn { note: pitch, .. } => note = pitch as i32,
				NoteEvent::NoteOff { note: pitch, .. } if pitch < 128 => {
					held &= !(1u128 << pitch);

					if note == pitch as i32 {
						note = -1;
					}
				}
				NoteEvent::NoteOff { note: pitch, .. } if note == pitch as i32 => note = -1,
				NoteEvent::ImmediateStop => {
					note = -1;
					held = 0;
				}
				_ => {}
			}
		}
		let note_changed = note != self.note || held != self.held;
		self.note = note;
		self.held = held;

		if let Some(values) = self
			.modulation
			.advance(input, process_context.infos(), &events)
		{
			let coefficient = 1.0 - (-(64.0 / (0.01 * self.rate as f32))).exp();
			self.smooth(coefficient);

			self.apply_modulation(&values);
		}

		let full = *samples;
		let bands = self.crossover.split(*samples, process_context);
		// An empty parallel lane is not a dry copy: it only joins the sum once
		// it actually holds an active module.
		let parallel_active = self
			.parallel
			.slots
			.iter()
			.flatten()
			.any(|module| !module.bypass);
		let any_solo =
			self.lanes.iter().any(|lane| lane.solo) || (parallel_active && self.parallel.solo);
		let mut output = [0.0_f32; 2];

		for (index, lane) in self.lanes.iter_mut().enumerate() {
			if lane.mute || (any_solo && !lane.solo) {
				// Keep the alignment line moving, so unmuting does not replay
				// audio from before the mute.
				lane.compensation.process([0.0, 0.0]);
				continue;
			}

			let mut band = bands[index];

			for slot in &mut lane.slots {
				let Some(module) = slot else {
					continue;
				};

				if module.bypass {
					// A skipped module still shows the signal it passes on.
					module.scope.push_stereo(band, band, sidechain);
				} else {
					let entered = band;
					// The product modules read the sidechain as `other[1]`; the
					// rack's second input port arrives in the same order.
					module.paramed.process(&mut band, &sides, process_context);
					module.scope.push_stereo(entered, band, sidechain);
				}
			}

			let gain = lane.gain as f32;
			let pan = lane.pan as f32;
			let left = gain * if pan > 0.0 { 1.0 - pan } else { 1.0 };
			let right = gain * if pan < 0.0 { 1.0 + pan } else { 1.0 };

			let frame = lane.compensation.process([band[0] * left, band[1] * right]);
			output[0] += frame[0];
			output[1] += frame[1];
		}

		if !parallel_active {
			// Nothing to add.
		} else if self.parallel.mute || (any_solo && !self.parallel.solo) {
			self.parallel.compensation.process([0.0, 0.0]);
		} else {
			let mut frame = full;

			for slot in &mut self.parallel.slots {
				let Some(module) = slot else {
					continue;
				};

				if module.bypass {
					module.scope.push_stereo(frame, frame, sidechain);
				} else {
					let entered = frame;
					module.paramed.process(&mut frame, &sides, process_context);
					module.scope.push_stereo(entered, frame, sidechain);
				}
			}

			let gain = self.parallel.gain as f32;
			let pan = self.parallel.pan as f32;
			let left = gain * if pan > 0.0 { 1.0 - pan } else { 1.0 };
			let right = gain * if pan < 0.0 { 1.0 + pan } else { 1.0 };

			let frame = self
				.parallel
				.compensation
				.process([frame[0] * left, frame[1] * right]);
			output[0] += frame[0];
			output[1] += frame[1];
		}

		*samples = output;

		self.note_tick += 1;

		if note_changed || self.note_tick.is_multiple_of(64) {
			self.publish_notes();
			self.publish_meters();
		}
	}

	/// Publishes the band meter of every module whose card draws one. The OTT
	/// has three gain values, a vocoder a level per band; both read at control
	/// rate, because a card cannot draw faster than the page polls.
	fn publish_meters(&self) {
		let mut input = [0.0f32; MAX_METER_BANDS];
		let mut output = [0.0f32; MAX_METER_BANDS];

		for lane in self.lanes.iter().chain(std::iter::once(&self.parallel)) {
			for module in lane.slots.iter().flatten() {
				let Some(meters) = &module.meters else {
					continue;
				};

				if let Some(gains) = module.paramed.value.gain_reduction() {
					meters.publish_gains(gains);
					continue;
				}

				let bands = module.paramed.value.band_meters(&mut input, &mut output);

				if bands > 0 {
					meters.publish_bands(bands, &input[..bands], &output[..bands]);
				}
			}
		}
	}

	/// Publishes the held note, and the mask a note-following effect tunes to,
	/// to the scope its card reads. Only the modules whose card shows notes pay
	/// for it.
	fn publish_notes(&self) {
		for lane in self.lanes.iter().chain(std::iter::once(&self.parallel)) {
			for module in lane.slots.iter().flatten() {
				if !module.notes {
					continue;
				}

				module.scope.set_note(self.note);
				module.scope.set_notes(self.held);

				if let Some(mask) = module.paramed.value.allowed_notes() {
					module.scope.set_allowed_notes(mask as i32);
				}
			}
		}
	}

	/// Moves every base parameter and band control toward its target.
	fn smooth(&mut self, coefficient: f32) {
		let coefficient = coefficient as f64;
		let Self {
			lanes, parallel, ..
		} = self;

		for lane in lanes.iter_mut().chain(std::iter::once(parallel)) {
			let gain_difference = lane.base_gain - lane.smooth_gain;
			let pan_difference = lane.base_pan - lane.smooth_pan;

			lane.smooth_gain += gain_difference * coefficient;
			lane.smooth_pan += pan_difference * coefficient;

			if gain_difference.abs() < 1e-5 {
				lane.smooth_gain = lane.base_gain;
			}

			if pan_difference.abs() < 1e-5 {
				lane.smooth_pan = lane.base_pan;
			}

			lane.gain = lane.smooth_gain;
			lane.pan = lane.smooth_pan;

			for module in lane.slots.iter_mut().flatten() {
				module.smooth(coefficient as f32);
			}
		}
	}

	/// Applies the eight assignments once, at a control tick.
	fn apply_modulation(&mut self, values: &BTreeMap<u32, f32>) {
		let mut sums: BTreeMap<Target, f32> = BTreeMap::new();
		let macros = self.macros;

		for assignment in self.assignments.iter().flatten() {
			let value = if assignment.source >= MACRO_BASE {
				macros.get(assignment.source - MACRO_BASE)
			} else {
				values.get(&(assignment.source as u32))
			};
			let Some(value) = value else {
				continue;
			};
			let Some(target) = assignment.target.as_ref() else {
				continue;
			};

			let offset = match assignment.mode {
				AssignmentMode::Centered => *value * assignment.depth,
				AssignmentMode::FromBase => {
					let unipolar = assignment.source >= MACRO_BASE
						|| matches!(
							self.source_kinds.get(&(assignment.source as u32)),
							Some(SourceKind::Envelope | SourceKind::Pitch | SourceKind::Midi)
						);
					let unit = if unipolar {
						*value
					} else {
						(*value + 1.0) * 0.5
					};
					unit * assignment.depth
				}
			};

			*sums.entry(target.clone()).or_insert(0.0) += offset;
		}

		for (target, offset) in sums {
			match target {
				Target::Module { module, parameter } => {
					let Some(module) = self.module_mut(module) else {
						continue;
					};
					let Some(control) = module.control(&parameter).cloned() else {
						continue;
					};

					if control.kind != Kind::Float {
						continue;
					}

					let base = module.current_value(&parameter).unwrap_or(control.default);
					let normalized = normalize(base, &control) + offset as f64;
					module.set_value(
						&parameter,
						denormalize(normalized.clamp(0.0, 1.0), &control),
					);
				}
				Target::BandGain(index) => {
					let Some(lane) = self.lane_mut(index) else {
						continue;
					};
					let normalized = lane.smooth_gain / 2.0 + offset as f64;
					lane.gain = normalized.clamp(0.0, 1.0) * 2.0;
				}
				Target::BandPan(index) => {
					let Some(lane) = self.lane_mut(index) else {
						continue;
					};
					let normalized = (lane.smooth_pan + 1.0) / 2.0 + offset as f64;
					lane.pan = normalized.clamp(0.0, 1.0) * 2.0 - 1.0;
				}
			}
		}
	}

	/// The lane at an index, with the parallel full-range lane after the bands.
	fn lane_mut(&mut self, index: usize) -> Option<&mut Lane> {
		match index {
			0..=2 => self.lanes.get_mut(index),
			3 => Some(&mut self.parallel),
			_ => None,
		}
	}

	/// The lane at an index, with the parallel full-range lane after the bands.
	#[cfg(test)]
	fn lane(&self, index: usize) -> Option<&Lane> {
		match index {
			0..=2 => self.lanes.get(index),
			3 => Some(&self.parallel),
			_ => None,
		}
	}

	/// The module with this id, whichever lane and slot holds it.
	fn module_mut(&mut self, id: u32) -> Option<&mut Module> {
		if let Some(module) = self
			.lanes
			.iter_mut()
			.flat_map(|lane| lane.slots.iter_mut())
			.flatten()
			.find(|module| module.id == id)
		{
			return Some(module);
		}

		self.parallel
			.slots
			.iter_mut()
			.flatten()
			.find(|module| module.id == id)
	}

	/// The live source values and LFO/table phases.
	pub fn modulation_snapshot(&self) -> (Vec<(u32, f32, f32)>, u64) {
		self.modulation.snapshot()
	}

	/// The latency of the chain, in samples.
	///
	/// This is the slowest band; the other bands are padded to match it.
	pub fn delay(&self) -> usize {
		self.lanes
			.iter()
			.map(|lane| lane.delay)
			.chain(std::iter::once(self.parallel.delay))
			.max()
			.unwrap_or(0)
	}

	/// The latency of each band, for a test.
	#[cfg(test)]
	pub fn lane_delays(&self) -> [usize; 3] {
		[
			self.lanes[0].delay,
			self.lanes[1].delay,
			self.lanes[2].delay,
		]
	}

	/// The compensation delay of each band, for a test.
	#[cfg(test)]
	pub fn lane_compensations(&self) -> [usize; 3] {
		[
			self.lanes[0].compensation.len(),
			self.lanes[1].compensation.len(),
			self.lanes[2].compensation.len(),
		]
	}

	/// The macro values the engine is running with, for a test.
	#[cfg(test)]
	pub fn macro_values(&self) -> [f32; 8] {
		self.macros
	}

	/// One module parameter as the audio thread currently holds it, for a test.
	#[cfg(test)]
	pub fn module_parameter(&self, id: u32, parameter: &str) -> Option<f32> {
		let module = self
			.lanes
			.iter()
			.chain(std::iter::once(&self.parallel))
			.flat_map(|lane| lane.slots.iter())
			.flatten()
			.find(|module| module.id == id)?;

		match module.map.get(parameter)?.load(Ordering::SeqCst) {
			SetValue::Float(value) => Some(value),
			SetValue::Int(value) => Some(value as f32),
			SetValue::Bool(value) => Some(if value { 1.0 } else { 0.0 }),
			_ => None,
		}
	}

	/// A band's modulated gain, for a test.
	#[cfg(test)]
	pub fn band_gain(&self, index: usize) -> Option<f64> {
		self.lane(index).map(|lane| lane.gain)
	}

	/// The populated slots, in band order, for a test.
	#[cfg(test)]
	pub fn kinds(&self) -> Vec<ModuleKind> {
		self.lanes
			.iter()
			.chain(std::iter::once(&self.parallel))
			.flat_map(|lane| &lane.slots)
			.flatten()
			.map(|module| module.kind)
			.collect()
	}
}

/// Maps a parameter value into 0..1, respecting a logarithmic range.
pub(super) fn normalize(value: f64, control: &Control) -> f64 {
	let low = control.min;
	let high = control.max;

	if high <= low {
		return 0.0;
	}

	if control.log && low > 0.0 {
		((value.max(low).ln() - low.ln()) / (high.ln() - low.ln())).clamp(0.0, 1.0)
	} else {
		((value - low) / (high - low)).clamp(0.0, 1.0)
	}
}

/// Maps 0..1 back into a parameter's units.
pub(super) fn denormalize(position: f64, control: &Control) -> f64 {
	let low = control.min;
	let high = control.max;
	let position = position.clamp(0.0, 1.0);

	if high <= low {
		return low;
	}

	if control.log && low > 0.0 {
		low * (high / low).powf(position)
	} else {
		low + (high - low) * position
	}
}

/// Writes a numeric value into the map, converting to the type the parameter holds.
pub(super) fn store(map: &ParamMap, identifier: &str, value: f64) {
	let Some(atomic) = map.get(identifier) else {
		return;
	};

	// A read of the map can be skipped when nothing wrote it, so a write has to say
	// it happened even when it does not go through the map.
	map.touch();

	if atomic.is_int() {
		atomic.store(value.round() as i32, Ordering::SeqCst);
	} else if atomic.is_bool() {
		atomic.store(value != 0.0, Ordering::SeqCst);
	} else {
		atomic.store(value as f32, Ordering::SeqCst);
	}
}
