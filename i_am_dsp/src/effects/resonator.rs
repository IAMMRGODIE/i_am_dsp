//! A saw-excited, MIDI-tuned resonator.
//!
//! The input is mixed with a band-limited saw at the current note and fed into a
//! tuned comb with a damping low-pass in its feedback path. That is the small,
//! playable version of a string/pipe resonance: the note sets the delay length,
//! the decay sets the feedback and the damping darkens each round trip.
//!
//! One comb is one string, so a chord needs one string per note: the effect
//! keeps up to [MAX_VOICES] of them, hands a note to the one that is free (or
//! steals the oldest), and lets a released string ring out. The MIDI switch in the
//! header decides where the notes come from: on, the effect's note port, and no
//! note means no excitation; off, the Note knob, which is what an audio track with
//! no note source needs.

use i_am_dsp_derive::Parameters;

use crate::{Effect, ProcessContext, tools::interpolate::cubic_interpolate};

/// The lowest note the delay line has to hold (20 Hz).
const MIN_FREQUENCY: f32 = 20.0;

/// Feedback below this is flushed, where denormal arithmetic gets expensive.
const DENORMAL: f32 = 1e-25;

/// The most strings the effect can hold at once.
pub const MAX_VOICES: usize = 16;

/// How long a released string takes to fade out, so a note off cannot click.
const RELEASE_SECONDS: f32 = 0.02;

/// The frequency of a MIDI note, with A4 = 69 = 440 Hz.
fn note_frequency(note: f32) -> f32 {
	440.0 * 2.0_f32.powf((note - 69.0) / 12.0)
}

/// A polyBLEP correction for a naive saw at the given phase and step.
fn poly_blep(phase: f32, step: f32) -> f32 {
	if step <= 0.0 {
		return 0.0;
	}

	if phase < step {
		let t = phase / step;
		t + t - t * t - 1.0
	} else if phase > 1.0 - step {
		let t = (phase - 1.0) / step;
		t * t + t + t + 1.0
	} else {
		0.0
	}
}

/// One string: its own comb, damping state and excitation phase.
struct Voice<const CHANNELS: usize> {
	/// The MIDI note this string is tuned to.
	note: f32,
	/// The smoothed frequency, so a retune glides instead of clicking.
	frequency: f32,
	phase: f32,
	/// The comb delay line, one per channel.
	line: [Vec<f32>; CHANNELS],
	write: usize,
	damp_state: [f32; CHANNELS],
	/// The excitation level: the velocity while held, fading to zero after a
	/// note off.
	level: f32,
	/// Whether the level is falling.
	released: bool,
	/// When the string was handed its note, so the oldest one is stolen first.
	started: u64,
}

/// A tuned comb resonator driven by the input and a band-limited saw.
#[derive(Parameters)]
pub struct Resonator<const CHANNELS: usize = 2> {
	/// The note the resonator rings at when the note port is off.
	#[range(min = 0.0, max = 127.0)]
	#[unit(none)]
	#[default(60.0)]
	#[id(display = "Note (MIDI)")]
	#[layout(group = "Resonator", order = 0, widget = knob, size = large, hero = "resonator", step = 1.0)]
	pub note: f32,

	/// How long the resonance rings after the excitation stops.
	#[range(min = 0.05, max = 10.0)]
	#[logarithmic]
	#[unit(seconds)]
	#[default(1.5)]
	#[id(display = "Decay")]
	#[layout(group = "Resonator", order = 1, widget = knob)]
	pub decay: f32,

	/// The cutoff of the low-pass in the feedback path.
	#[range(min = 200.0, max = 18_000.0)]
	#[logarithmic]
	#[unit(hertz)]
	#[default(8_000.0)]
	#[id(display = "Damping")]
	#[layout(group = "Resonator", order = 2, widget = knob)]
	pub damping: f32,

	/// The level of the internal saw excitation.
	#[range(min = 0.0, max = 1.0)]
	#[unit(percent)]
	#[default(0.0)]
	#[id(display = "Saw")]
	#[layout(group = "Resonator", order = 3, widget = knob)]
	pub saw: f32,

	/// How much of the resonant signal is heard.
	#[range(min = 0.0, max = 1.0)]
	#[unit(percent)]
	#[default(0.5)]
	#[id(display = "Mix")]
	#[layout(group = "Resonator", order = 4, widget = knob)]
	pub mix: f32,

	/// How many notes can ring at once. A chord needs one string per note; a
	/// note that arrives when all of them are busy steals the oldest.
	#[range(min = 1, max = 16)]
	#[unit(none)]
	#[default(8)]
	#[id(display = "Voices")]
	#[layout(group = "Resonator", order = 5, widget = knob, step = 1.0)]
	pub voices: i32,

	/// Whether the note port is listened to.
	///
	/// Off, the Note knob is the only excitation and the effect needs no MIDI at
	/// all. On, only the notes played on the port excite the strings: with nothing
	/// held there is nothing to resonate, and the knob is not a fallback.
	#[default(false)]
	#[id(display = "MIDI")]
	#[layout(group = "Resonator", order = 6, widget = toggle)]
	pub midi: bool,

	#[skip]
	sample_rate: f32,
	/// The strings, at most [MAX_VOICES] of them.
	#[skip]
	strings: Vec<Voice<CHANNELS>>,
	/// Counts note hand-offs, so the oldest string is the one to steal.
	#[skip]
	started: u64,
	/// Whether the Note knob is currently exciting the first string.
	#[skip]
	knob_active: bool,
}

impl<const CHANNELS: usize> Resonator<CHANNELS> {
	/// A resonator tuned to middle C with a 1.5 s decay.
	pub fn new(sample_rate: usize) -> Self {
		assert!(CHANNELS > 0, "CHANNELS must be greater than 0");

		let mut resonator = Self {
			note: 60.0,
			decay: 1.5,
			damping: 8_000.0,
			saw: 0.0,
			mix: 0.5,
			voices: 8,
			midi: false,
			sample_rate: sample_rate as f32,
			strings: Vec::new(),
			started: 0,
			knob_active: false,
		};

		resonator.ensure_voices();
		resonator
	}

	/// How many strings are sounding right now.
	#[cfg(test)]
	fn sounding(&self) -> usize {
		self.strings
			.iter()
			.filter(|voice| voice.level > 0.0)
			.count()
	}

	/// The number of strings the parameters ask for, clamped to what exists.
	fn wanted(&self) -> usize {
		(self.voices.clamp(1, MAX_VOICES as i32)) as usize
	}

	/// Allocates the strings the parameter asks for. Growing allocates, which is
	/// why the voices control is a knob a person turns and not something the
	/// effect does per block.
	fn ensure_voices(&mut self) {
		let wanted = self.wanted();
		let capacity = (self.sample_rate / MIN_FREQUENCY).ceil() as usize + 8;

		while self.strings.len() < wanted {
			self.strings.push(Voice {
				note: self.note,
				frequency: note_frequency(self.note),
				phase: 0.0,
				line: core::array::from_fn(|_| vec![0.0; capacity]),
				write: 0,
				damp_state: [0.0; CHANNELS],
				level: 0.0,
				released: true,
				started: 0,
			});
		}
	}

	fn note_on(&mut self, note: usize, velocity: f32) {
		self.starting();

		let note = note as f32;
		let level = velocity.clamp(0.05, 1.0);
		self.started += 1;

		// Only the strings the Voices control asks for are in play; the rest of
		// the allocation is left alone until the control is raised.
		let wanted = self.wanted();
		let strings = &mut self.strings[..wanted];

		// A repeat of a note already ringing retriggers that string.
		if let Some(voice) = strings
			.iter_mut()
			.find(|voice| voice.note == note && !voice.released)
		{
			voice.level = level;
			voice.started = self.started;
			return;
		}

		let index = match strings.iter().position(|voice| voice.level <= 0.0) {
			Some(index) => index,
			// No free string: steal the oldest one. Its frequency glides to the
			// new note, so the hand-off is a very fast portamento, not a click.
			None => strings
				.iter()
				.enumerate()
				.min_by_key(|(_, voice)| voice.started)
				.map(|(index, _)| index)
				.unwrap_or(0),
		};

		let voice = &mut strings[index];
		voice.note = note;
		voice.level = level;
		voice.released = false;
		voice.started = self.started;
	}

	fn note_off(&mut self, note: usize) {
		let note = note as f32;

		for voice in &mut self.strings {
			if voice.note == note && !voice.released {
				voice.released = true;
			}
		}
	}

	fn stop_all(&mut self) {
		for voice in &mut self.strings {
			voice.released = true;
		}
	}

	/// Counts the hand-offs, and keeps the string list long enough for the
	/// current voice parameter.
	fn starting(&mut self) {
		if self.wanted() > self.strings.len() {
			self.ensure_voices();
		}
	}

	fn track_events(&mut self, process_context: &mut Box<dyn ProcessContext>) {
		for event in process_context.events() {
			match *event {
				crate::NoteEvent::NoteOn { note, velocity, .. } => self.note_on(note, velocity),
				crate::NoteEvent::NoteOff { note, .. } | crate::NoteEvent::Stop { note, .. } => {
					self.note_off(note);
				}
				crate::NoteEvent::ImmediateStop => self.stop_all(),
				_ => {}
			}
		}
	}

	/// Read the comb at a fractional delay from the write head.
	fn read_delayed(line: &[f32], write: usize, delay: f32) -> f32 {
		let capacity = line.len() as isize;
		let delay = delay.clamp(2.0, (capacity - 3) as f32);
		let whole = delay.floor() as isize;
		let fraction = delay.fract();
		let index =
			|offset: isize| ((write as isize + capacity - whole + offset) % capacity) as usize;

		cubic_interpolate(
			fraction,
			[
				line[index(-1)],
				line[index(0)],
				line[index(1)],
				line[index(2)],
			],
		)
	}

	fn damping_coefficient(&self) -> f32 {
		let cutoff = self.damping.clamp(20.0, self.sample_rate * 0.45);
		(1.0 - (-2.0 * core::f32::consts::PI * cutoff / self.sample_rate).exp()).clamp(0.0, 1.0)
	}
}

impl<const CHANNELS: usize> Effect<CHANNELS> for Resonator<CHANNELS> {
	fn delay(&self) -> usize {
		0
	}

	fn process(
		&mut self,
		samples: &mut [f32; CHANNELS],
		_other: &[&[f32; CHANNELS]],
		process_context: &mut Box<dyn ProcessContext>,
	) {
		if self.midi {
			self.track_events(process_context);
		}

		self.ensure_voices();

		let glide = 1.0 - (-1.0 / (0.005 * self.sample_rate)).exp();
		// A released string fades over RELEASE_SECONDS; the exponential keeps the
		// fade smooth whatever the block size.
		let release = (-1.0 / (RELEASE_SECONDS * self.sample_rate)).exp();
		let damping = self.damping_coefficient();
		let sample_rate = self.sample_rate;
		let decay = self.decay.max(0.05);
		let saw_gain = self.saw;
		let input = *samples;

		// With the note port off the Note knob is the excitation. Voice zero is that
		// string: it follows the knob as it turns. With the port on only its notes
		// excite the strings, so nothing is armed before one arrives.
		let knob_drives = !self.midi;

		if knob_drives {
			self.knob_active = true;

			for (index, voice) in self.strings.iter_mut().enumerate() {
				if index == 0 {
					voice.note = self.note;
					voice.level = 1.0;
					voice.released = false;
				} else if !voice.released {
					// A held MIDI note has to let go when the port is switched off.
					voice.released = true;
				}
			}
		} else if self.knob_active {
			// The port was switched on: let the knob string fade instead of
			// droning under the notes.
			self.knob_active = false;

			if let Some(voice) = self.strings.first_mut() {
				voice.released = true;
			}
		}

		let active = self.wanted();
		let mut wet = [0.0f32; CHANNELS];
		let mut sounding = 0.0f32;

		for voice in self.strings.iter_mut().take(active) {
			if voice.released && voice.level > 0.0 {
				voice.level *= release;

				if voice.level < 1e-4 {
					voice.level = 0.0;
				}
			}

			if voice.level <= 0.0 {
				continue;
			}

			sounding += 1.0;

			let target = note_frequency(voice.note).clamp(MIN_FREQUENCY, sample_rate * 0.45);
			voice.frequency += (target - voice.frequency) * glide;

			let step = (voice.frequency / sample_rate).clamp(0.0, 0.45);
			voice.phase += step;

			if voice.phase >= 1.0 {
				voice.phase -= 1.0;
			}

			let saw = 2.0 * voice.phase - 1.0 - poly_blep(voice.phase, step);
			let capacity = voice.line[0].len();
			let delay = (sample_rate / voice.frequency).clamp(2.0, (capacity - 3) as f32);
			let loop_seconds = delay / sample_rate;
			let feedback = 10.0_f32
				.powf(-3.0 * loop_seconds / decay)
				.clamp(0.0, 0.9995);
			let write = voice.write;

			for (channel, sample) in input.iter().enumerate() {
				let delayed = Self::read_delayed(&voice.line[channel], write, delay);
				voice.damp_state[channel] += damping * (delayed - voice.damp_state[channel]);
				let resonant = sample * voice.level
					+ saw * saw_gain * voice.level
					+ feedback * voice.damp_state[channel];

				if resonant.abs() <= DENORMAL {
					voice.line[channel][write] = 0.0;
				} else {
					voice.line[channel][write] = resonant;
				}

				wet[channel] += resonant;
			}

			voice.write = (write + 1) % capacity;
		}

		// A single note plays at the level it always did; a chord is scaled by
		// the square root of its size so it does not clip.
		let gain = if sounding > 0.0 {
			1.0 / sounding.sqrt()
		} else {
			0.0
		};

		for (channel, sample) in samples.iter_mut().enumerate() {
			*sample = input[channel] * (1.0 - self.mix) + wet[channel] * self.mix * gain;
		}
	}

	fn clear_tail(&mut self) {
		for voice in &mut self.strings {
			for line in &mut voice.line {
				line.fill(0.0);
			}

			voice.damp_state.fill(0.0);
			voice.level = 0.0;
			voice.released = true;
		}
	}
}
#[cfg(test)]
mod tests {
	use super::*;
	use crate::{NoteEvent, ProcessContext, ProcessInfos};

	struct TestContext {
		events: Vec<NoteEvent>,
	}

	impl ProcessContext for TestContext {
		fn infos(&self) -> &ProcessInfos {
			&crate::DEFAULT_PROCESS_INFO
		}

		fn next_event(&mut self) -> Option<NoteEvent> {
			if self.events.is_empty() {
				None
			} else {
				Some(self.events.remove(0))
			}
		}

		fn send_event(&mut self, event: NoteEvent) {
			self.events.push(event);
		}

		fn events(&self) -> &[NoteEvent] {
			&self.events
		}

		fn clear_events(&mut self) {
			self.events.clear();
		}
	}

	fn render(note: Option<usize>) -> Vec<f32> {
		let mut resonator = Resonator::<1>::new(48_000);
		resonator.midi = true;
		resonator.note = 60.0;
		resonator.decay = 2.0;
		resonator.saw = 0.0;
		resonator.mix = 1.0;
		let mut context: Box<dyn ProcessContext> = Box::new(TestContext {
			events: note
				.map(|note| {
					vec![NoteEvent::NoteOn {
						channel: 0,
						note,
						velocity: 1.0,
					}]
				})
				.unwrap_or_default(),
		});
		let mut output = Vec::new();

		for index in 0..4_800 {
			let mut frame = [if index % 2_400 == 0 { 0.8 } else { 0.0 }];
			resonator.process(&mut frame, &[], &mut context);
			output.push(frame[0]);
		}

		output
	}

	#[test]
	fn a_chord_rings_on_several_strings() {
		let mut resonator = Resonator::<1>::new(48_000);
		resonator.midi = true;
		resonator.voices = 8;
		resonator.decay = 2.0;
		resonator.mix = 1.0;
		let mut context: Box<dyn ProcessContext> = Box::new(TestContext {
			events: [48usize, 55, 64]
				.iter()
				.map(|note| NoteEvent::NoteOn {
					channel: 0,
					note: *note,
					velocity: 1.0,
				})
				.collect(),
		});

		for _ in 0..512 {
			let mut frame = [0.5];
			resonator.process(&mut frame, &[], &mut context);
		}

		assert_eq!(
			resonator.sounding(),
			3,
			"the chord did not get three strings"
		);
	}

	#[test]
	fn a_note_beyond_the_voice_count_steals_the_oldest() {
		let mut resonator = Resonator::<1>::new(48_000);
		resonator.midi = true;
		resonator.voices = 2;
		let events = [48usize, 52, 55, 60]
			.iter()
			.map(|note| NoteEvent::NoteOn {
				channel: 0,
				note: *note,
				velocity: 1.0,
			})
			.collect();
		let mut context: Box<dyn ProcessContext> = Box::new(TestContext { events });

		for _ in 0..64 {
			let mut frame = [0.5];
			resonator.process(&mut frame, &[], &mut context);
		}

		assert_eq!(resonator.sounding(), 2, "more strings rang than allowed");
		assert!(
			resonator
				.strings
				.iter()
				.any(|voice| voice.note == 60.0 && voice.level > 0.0),
			"the newest note did not replace the oldest"
		);
	}

	#[test]
	fn a_released_note_fades_its_string() {
		let mut resonator = Resonator::<1>::new(48_000);
		resonator.midi = true;
		resonator.voices = 8;
		resonator.decay = 0.2;
		let mut context: Box<dyn ProcessContext> = Box::new(TestContext {
			events: vec![NoteEvent::NoteOn {
				channel: 0,
				note: 60,
				velocity: 1.0,
			}],
		});

		for _ in 0..64 {
			let mut frame = [0.0];
			resonator.process(&mut frame, &[], &mut context);
		}

		assert_eq!(resonator.sounding(), 1);

		let mut stop: Box<dyn ProcessContext> = Box::new(TestContext {
			events: vec![NoteEvent::NoteOff {
				channel: 0,
				note: 60,
				velocity: 0.0,
			}],
		});

		for _ in 0..14_400 {
			let mut frame = [0.0];
			resonator.process(&mut frame, &[], &mut stop);
		}

		assert_eq!(resonator.sounding(), 0, "the string never released");
	}

	#[test]
	fn a_midi_note_retunes_the_resonator() {
		let low = render(Some(48));
		let high = render(Some(72));
		let energy = |frames: &[f32]| frames.iter().map(|sample| sample * sample).sum::<f32>();

		assert!(energy(&high) > 0.0);
		assert!(energy(&low) > 0.0);
		assert!(
			low.iter()
				.zip(high.iter())
				.any(|(a, b)| (a - b).abs() > 1e-4),
			"the note had no effect"
		);
	}

	#[test]
	fn the_saw_excites_a_note_without_an_input() {
		let mut resonator = Resonator::<1>::new(48_000);
		resonator.note = 69.0;
		resonator.saw = 0.8;
		resonator.mix = 1.0;
		let mut context: Box<dyn ProcessContext> = Box::new(TestContext { events: vec![] });
		let mut output = Vec::new();

		for _ in 0..4_800 {
			let mut frame = [0.0];
			resonator.process(&mut frame, &[], &mut context);
			output.push(frame[0]);
		}

		assert!(output.iter().any(|sample| sample.abs() > 1e-3));
		assert!(output.iter().all(|sample| sample.is_finite()));
	}

	#[test]
	fn the_note_knob_retunes_without_midi() {
		let mut resonator = Resonator::<1>::new(48_000);
		resonator.midi = false;
		resonator.saw = 0.8;
		resonator.mix = 1.0;
		let mut context: Box<dyn ProcessContext> = Box::new(TestContext { events: vec![] });

		for _ in 0..64 {
			let mut frame = [0.0];
			resonator.process(&mut frame, &[], &mut context);
		}

		resonator.note = 72.0;

		for _ in 0..4_800 {
			let mut frame = [0.0];
			resonator.process(&mut frame, &[], &mut context);
		}

		let voice = resonator.strings.first().expect("the knob string");
		let target = note_frequency(72.0);
		assert_eq!(voice.note, 72.0, "the knob did not retune the string");
		assert!(
			(voice.frequency - target).abs() < 0.5,
			"the string glided to {} instead of {target}",
			voice.frequency
		);
	}

	#[test]
	fn midi_on_does_not_fall_back_to_the_note_knob() {
		let mut resonator = Resonator::<1>::new(48_000);
		resonator.midi = true;
		resonator.voices = 8;
		let mut quiet: Box<dyn ProcessContext> = Box::new(TestContext { events: vec![] });

		for _ in 0..64 {
			let mut frame = [0.0];
			resonator.process(&mut frame, &[], &mut quiet);
		}

		assert_eq!(
			resonator.sounding(),
			0,
			"the knob excited a string while the port was on"
		);

		let mut notes: Box<dyn ProcessContext> = Box::new(TestContext {
			events: vec![NoteEvent::NoteOn {
				channel: 0,
				note: 72,
				velocity: 1.0,
			}],
		});

		for _ in 0..64 {
			let mut frame = [0.0];
			resonator.process(&mut frame, &[], &mut notes);
		}

		assert_eq!(resonator.sounding(), 1, "the note never got a string");
		assert!(
			resonator
				.strings
				.iter()
				.any(|voice| voice.note == 72.0 && voice.level > 0.0),
			"the note did not get a string"
		);
	}

	#[test]
	fn switching_the_port_on_releases_the_knob_string() {
		let mut resonator = Resonator::<1>::new(48_000);
		resonator.midi = false;
		let mut quiet: Box<dyn ProcessContext> = Box::new(TestContext { events: vec![] });

		for _ in 0..64 {
			let mut frame = [0.0];
			resonator.process(&mut frame, &[], &mut quiet);
		}

		assert!(resonator.strings[0].level > 0.0, "no knob string to start");

		resonator.midi = true;

		for _ in 0..64 {
			let mut frame = [0.0];
			resonator.process(&mut frame, &[], &mut quiet);
		}

		assert!(
			resonator.strings[0].released,
			"the knob string kept droning under the notes"
		);
	}

	#[test]
	fn a_switched_off_note_port_ignores_midi() {
		let mut resonator = Resonator::<1>::new(48_000);
		resonator.midi = false;
		resonator.voices = 8;
		let mut notes: Box<dyn ProcessContext> = Box::new(TestContext {
			events: vec![NoteEvent::NoteOn {
				channel: 0,
				note: 72,
				velocity: 1.0,
			}],
		});

		for _ in 0..64 {
			let mut frame = [0.0];
			resonator.process(&mut frame, &[], &mut notes);
		}

		assert!(
			resonator.strings.iter().all(|voice| voice.note != 72.0),
			"the note was used with the port switched off"
		);
		assert!(resonator.strings[0].level > 0.0, "the knob string stopped");
	}
}
