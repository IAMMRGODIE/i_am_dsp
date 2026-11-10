use super::*;

impl DrumInstrument {
	/// Imports a one-shot that every hit layers under it, from WAV bytes.
	pub fn set_click_sample(&mut self, bytes: &[u8]) -> Result<(), String> {
		self.click_sample = Some(decode_click(bytes)?);
		Ok(())
	}

	/// Shares a slot with the editor, so a file the page loads reaches the
	/// voices without the page being allowed to write serialized parameters.
	pub fn with_click_slot(mut self, slot: Arc<ClickSlot>) -> Self {
		if let Some(data) = slot.load() {
			self.click_sample = Some(data);
		}
		self.click_seen = slot.generation();
		self.click_slot = Some(slot);
		self
	}

	/// Shares the six one-shot slots with the editor.
	pub fn with_samples(mut self, slots: Arc<Samples>) -> Self {
		self.samples = Some(slots.clone());
		self.samples_seen = slots.generation();
		self.adopt_samples();

		self
	}

	/// Shares the snare's shapes with the editor.
	pub fn with_snare(mut self, shared: Arc<Mutex<SnareCurves>>) -> Self {
		self.snare_shared = shared;
		self
	}

	/// The shell's envelope.

	/// Start one hit, if the note is the kick's.
	pub fn start(&mut self, note: u8, velocity: f32) -> bool {
		self.adopt_samples();
		self.adopt_fit();

		let Some(drum) = drum_of(note) else {
			return false;
		};

		// A harder hit opens the drum up: the material and the FM index follow
		// the brightness, the body only gets louder.
		let velocity = velocity.clamp(0.0, 1.0);
		self.brightness = 1.0 + (velocity - 0.5) * 2.0 * self.velocity_brightness.clamp(0.0, 1.0);

		// A closed or pedal hat chokes the open one: that is the pair.
		if matches!(drum, Drum::HatClosed | Drum::HatPedal) {
			for voice in &mut self.playing {
				if voice.drum == Drum::HatOpen {
					voice.choking = true;
				}
			}
		}

		if self.playing.len() >= MAX_VOICES {
			self.playing.remove(0);
		}

		// A loaded one-shot replaces this drum's synthesis; the click is only
		// layered under a synthesised drum.
		let loaded = self.loaded[drum.index()].clone();
		let mut recipes: Vec<Recipe> = if let Some(data) = &loaded {
			vec![self.sample_recipe(data, self.sample_level(drum))]
		} else if drum == Drum::Snare {
			// The shell and the wires are separate voices, so their envelopes
			// and the shell's pitch are their own.
			vec![self.snare_base_recipe(), self.snare_noise_recipe()]
		} else {
			vec![match drum {
				Drum::Kick => self.kick_recipe(),
				Drum::Clap => self.clap_recipe(),
				Drum::HatClosed | Drum::HatPedal | Drum::HatOpen => self.hat_recipe(drum),
				Drum::Tom => self.tom_recipe(note),
				Drum::Crash | Drum::Ride => self.cymbal_recipe(drum),
				Drum::Snare => self.snare_base_recipe(),
			}]
		};

		if loaded.is_none()
			&& drum != Drum::Snare
			&& let Some(sample) = &self.click_sample
			&& self.click_sample_level > 0.0
		{
			recipes[0].layers.push(Layer {
				source: Source::Sample {
					data: Arc::new(sample.clone()),
					level: self.click_sample_level.clamp(0.0, 1.0),
				},
				keytrack: 0.0,
				velocity: 1.0,
				gain: 1.0,
			});
		}

		if matches!(drum, Drum::HatClosed | Drum::HatPedal | Drum::HatOpen) {
			// The hat's attack starts dry and hands over to the IR.
			self.hat_gate = 0.0;
		}

		// A drummer never hits twice the same: the humaniser moves the level and
		// the pitch of each hit, and zero leaves the kit exact.
		let humanize = self.humanize.clamp(0.0, 1.0);
		let level = (1.0 + self.rng.bipolar() * 0.3 * humanize) * self.level_for(drum);
		let cents = self.rng.bipolar() * 20.0 * humanize;
		let detune = 2f32.powf(cents / 1_200.0);
		let human = Humanized { level, detune };

		for recipe in &recipes {
			if self.playing.len() >= MAX_VOICES {
				self.playing.remove(0);
			}

			self.playing.push(Voice::start(
				recipe,
				drum,
				self.rate,
				velocity,
				human,
				{
					let pan = self.pan_for(drum).clamp(-1.0, 1.0);
					let angle = (pan + 1.0) * std::f32::consts::FRAC_PI_4;

					Stereo {
						pan_l: angle.cos(),
						pan_r: angle.sin(),
						width: self.width.clamp(0.0, 1.0),
					}
				},
				&mut self.rng,
			));
		}

		true
	}

	/// Renders one hit on its own, without a host.
	///
	/// The instrument is left with nothing playing, so this is for exporting a
	/// one-shot rather than for playing.
	pub fn render(&mut self, note: u8, velocity: f32, seconds: f32) -> Vec<f32> {
		self.playing.clear();
		let _ = self.start(note, velocity);
		let count = (seconds.max(0.01) * self.rate) as usize;

		(0..count).map(|_| self.next_sample()).collect()
	}

	/// Renders one hit as stereo frames, for a host that wants the width.
	pub fn render_stereo(&mut self, note: u8, velocity: f32, seconds: f32) -> Vec<[f32; 2]> {
		self.playing.clear();
		let _ = self.start(note, velocity);
		let count = (seconds.max(0.01) * self.rate) as usize;

		(0..count).map(|_| self.next_frame()).collect()
	}

	/// One sample of the summed voices, driven and levelled.
	///
	/// Public so an offline renderer can pull the instrument without a host.
	pub fn next_sample(&mut self) -> f32 {
		let frame = self.next_frame();

		(frame[0] + frame[1]) * 0.5
	}

	/// One stereo frame of the summed voices, driven and levelled.
	pub fn next_frame(&mut self) -> [f32; 2] {
		let rate = self.rate;
		let rng = &mut self.rng;
		let mut dry = [0.0f32; 2];
		let mut cymbal_send = [0.0f32; 2];
		let mut hat_send = [0.0f32; 2];

		for voice in &mut self.playing {
			let (frame, cymbal, hat) = voice.next(rng, rate);
			dry[0] += frame[0];
			dry[1] += frame[1];
			cymbal_send[0] += cymbal[0];
			cymbal_send[1] += cymbal[1];
			hat_send[0] += hat[0];
			hat_send[1] += hat[1];
		}
		self.playing.retain(|voice| !voice.finished());

		self.tail_send = (cymbal_send[0] + cymbal_send[1]) * 0.5;

		// The tails run while a voice of theirs is alive, and a little longer so
		// the IR's own ring can finish after the send stops.
		if self
			.playing
			.iter()
			.any(|voice| matches!(voice.drum, Drum::Crash | Drum::Ride))
		{
			self.tail_remaining = (rate * 0.3) as usize;
		}
		if self
			.playing
			.iter()
			.any(|voice| matches!(voice.drum, Drum::HatClosed | Drum::HatPedal | Drum::HatOpen))
		{
			self.hat_remaining = (rate * 0.05) as usize;
		}

		let direct = self.hat_direct.clamp(0.0, 10.0) * 0.001;

		if direct <= 0.0 {
			self.hat_gate = 1.0;
		} else if self.hat_gate < 1.0 {
			self.hat_gate = (self.hat_gate + 1.0 / (direct * rate)).min(1.0);
		}

		let mut out = dry;

		if self.tail_remaining > 0
			&& let Some(convolver) = &mut self.tail_convolver
		{
			self.tail_remaining -= 1;
			let wet = convolver.frame(cymbal_send);
			out[0] += wet[0];
			out[1] += wet[1];
		}

		if self.hat_remaining > 0
			&& let Some(convolver) = &mut self.hat_convolver
		{
			self.hat_remaining -= 1;
			let wet = convolver.frame(hat_send);
			out[0] += hat_send[0] * (1.0 - self.hat_gate) + wet[0] * self.hat_gate;
			out[1] += hat_send[1] * (1.0 - self.hat_gate) + wet[1] * self.hat_gate;
		} else if hat_send != [0.0, 0.0] {
			out[0] += hat_send[0];
			out[1] += hat_send[1];
		}

		let drive = self.drive.clamp(1.0, 8.0);
		let gain = self.gain.clamp(0.0, 1.0);

		[
			(out[0] * drive).tanh() * gain,
			(out[1] * drive).tanh() * gain,
		]
	}
}
