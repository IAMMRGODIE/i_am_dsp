//! A saw-excited, MIDI-tuned resonator.
//!
//! The input is mixed with a band-limited saw at the current note and fed into a
//! tuned comb with a damping low-pass in its feedback path. That is the small,
//! playable version of a string/pipe resonance: the note sets the delay length,
//! the decay sets the feedback and the damping darkens each round trip. MIDI
//! notes played on the effect's note port retune it; the Note knob is the
//! fallback for hosts or clips that do not send notes.

use i_am_dsp_derive::Parameters;

use crate::{
	Effect, ProcessContext,
	tools::interpolate::cubic_interpolate,
};

/// The lowest note the delay line has to hold (20 Hz).
const MIN_FREQUENCY: f32 = 20.0;

/// Feedback below this is flushed, where denormal arithmetic gets expensive.
const DENORMAL: f32 = 1e-25;

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

/// A tuned comb resonator driven by the input and a band-limited saw.
#[derive(Parameters)]
pub struct Resonator<const CHANNELS: usize = 2> {
	/// The note used when the MIDI port is silent.
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
	#[default(0.25)]
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

	#[skip]
	sample_rate: f32,
	/// The smoothed frequency, so a note change glides instead of clicking.
	#[skip]
	frequency: f32,
	#[skip]
	phase: f32,
	/// Notes currently held on the note port, newest last.
	#[skip]
	held_notes: Vec<usize>,
	/// The comb delay line, one per channel.
	#[skip]
	line: [Vec<f32>; CHANNELS],
	#[skip]
	write: usize,
	#[skip]
	damp_state: [f32; CHANNELS],
}

impl<const CHANNELS: usize> Resonator<CHANNELS> {
	/// A resonator tuned to middle C with a 1.5 s decay.
	pub fn new(sample_rate: usize) -> Self {
		assert!(CHANNELS > 0, "CHANNELS must be greater than 0");

		let capacity = (sample_rate as f32 / MIN_FREQUENCY).ceil() as usize + 8;
		let frequency = note_frequency(60.0);

		Self {
			note: 60.0,
			decay: 1.5,
			damping: 8_000.0,
			saw: 0.25,
			mix: 0.5,
			sample_rate: sample_rate as f32,
			frequency,
			phase: 0.0,
			held_notes: Vec::new(),
			line: core::array::from_fn(|_| vec![0.0; capacity]),
			write: 0,
			damp_state: [0.0; CHANNELS],
		}
	}

	fn target_note(&self) -> f32 {
		self.held_notes
			.last()
			.map(|note| *note as f32)
			.unwrap_or(self.note)
	}

	fn track_events(&mut self, process_context: &mut Box<dyn ProcessContext>) {
		for event in process_context.events() {
			match event {
				crate::NoteEvent::NoteOn { note, .. } => {
					if !self.held_notes.contains(note) {
						self.held_notes.push(*note);
					}
				}
				crate::NoteEvent::NoteOff { note, .. } | crate::NoteEvent::Stop { note, .. } => {
					self.held_notes.retain(|held| held != note);
				}
				crate::NoteEvent::ImmediateStop => self.held_notes.clear(),
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
		let index = |offset: isize| ((write as isize + capacity - whole + offset) % capacity) as usize;

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
		(1.0 - (-2.0 * core::f32::consts::PI * cutoff / self.sample_rate).exp())
			.clamp(0.0, 1.0)
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
		self.track_events(process_context);

		let target = note_frequency(self.target_note()).clamp(MIN_FREQUENCY, self.sample_rate * 0.45);
		let glide = 1.0 - (-1.0 / (0.005 * self.sample_rate)).exp();
		let damping = self.damping_coefficient();
		let capacity = self.line[0].len();
		let write = self.write;

		self.frequency += (target - self.frequency) * glide;
		let step = (self.frequency / self.sample_rate).clamp(0.0, 0.45);
		self.phase += step;
		if self.phase >= 1.0 {
			self.phase -= 1.0;
		}

		let saw = 2.0 * self.phase - 1.0 - poly_blep(self.phase, step);
		let delay = (self.sample_rate / self.frequency).clamp(2.0, (capacity - 3) as f32);
		let loop_seconds = delay / self.sample_rate;
		let feedback = 10.0_f32
			.powf(-3.0 * loop_seconds / self.decay.max(0.05))
			.clamp(0.0, 0.9995);
		let input = *samples;

		for (channel, sample) in samples.iter_mut().enumerate() {
			let delayed = Self::read_delayed(&self.line[channel], write, delay);
			self.damp_state[channel] += damping * (delayed - self.damp_state[channel]);
			let resonant = input[channel] + saw * self.saw + feedback * self.damp_state[channel];

			if resonant.abs() <= DENORMAL {
				self.line[channel][write] = 0.0;
			} else {
				self.line[channel][write] = resonant;
			}

			*sample = input[channel] * (1.0 - self.mix) + resonant * self.mix;
		}

		self.write = (write + 1) % capacity;
	}

	fn clear_tail(&mut self) {
		for line in &mut self.line {
			line.fill(0.0);
		}
		self.damp_state.fill(0.0);
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
			if self.events.is_empty() { None } else { Some(self.events.remove(0)) }
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
	fn a_midi_note_retunes_the_resonator() {
		let low = render(Some(48));
		let high = render(Some(72));
		let energy = |frames: &[f32]| frames.iter().map(|sample| sample * sample).sum::<f32>();

		assert!(energy(&high) > 0.0);
		assert!(energy(&low) > 0.0);
		assert!(
			low.iter().zip(high.iter()).any(|(a, b)| (a - b).abs() > 1e-4),
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
}
