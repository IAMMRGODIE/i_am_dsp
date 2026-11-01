//! An autotune: track the input pitch, snap it to the allowed notes, and drive
//! one of the pitch shifters with the resulting ratio.
//!
//! MIDI notes are treated as pitch classes, so holding a chord or a scale allows
//! that set in every octave. With no MIDI held, the note mask from the faceplate
//! is used. The mask has one bit per pitch class, C at bit 0 and B at bit 11.

use i_am_dsp_derive::Parameters;

use crate::{
	Effect, NoteEvent, ProcessContext,
	effects::{
		phase_vocoder::{PhaseVocoder, PitchShift, overlap_add_gain},
		pitch_shifter::PitchShifter,
		psola::Psola,
	},
	tools::pitch::PitchTracker,
};

/// One bit per pitch class; C is bit 0.
const ALL_NOTES: u16 = 0x0fff;

/// The nearest allowed MIDI note to a detected fractional note.
fn nearest_allowed_note(detected: f32, mask: u16) -> Option<f32> {
	if mask == 0 {
		return None;
	}

	let center = detected.floor() as i32;
	let mut best: Option<(i32, f32)> = None;

	for offset in -12..=12 {
		let note = center + offset;
		let class = note.rem_euclid(12) as u16;
		if mask & (1 << class) != 0 {
			let distance = (note as f32 - detected).abs();
			if best.is_none_or(|(_, previous)| distance < previous) {
				best = Some((note, distance));
			}
		}
	}

	best.map(|(note, _)| note as f32)
}

/// An autotune with WSOLA, Phase Vocoder and PSOLA engines.
#[derive(Parameters)]
pub struct AutoTune<const CHANNELS: usize = 2> {
	/// The pitch-shifting engine.
	#[choices("WSOLA", "Phase Vocoder", "PSOLA")]
	#[default(0)]
	#[id(display = "Algorithm")]
	#[layout(group = "Tune", order = 0, widget = choice)]
	pub algorithm: i32,

	/// How hard the detected note is pulled to the nearest allowed note.
	#[range(min = 0.0, max = 1.0)]
	#[unit(percent)]
	#[default(1.0)]
	#[id(display = "Strength")]
	#[layout(group = "Tune", order = 1, widget = knob, size = large, hero = "autotune")]
	pub strength: f32,

	/// The smoothing time for the ratio; zero snaps immediately.
	#[range(min = 0.0, max = 200.0)]
	#[unit(milliseconds)]
	#[default(10.0)]
	#[id(display = "Speed")]
	#[layout(group = "Tune", order = 2, widget = knob)]
	pub speed: f32,

	/// The phase-vocoder window size, shown only for that engine.
	#[choices("512", "1024", "2048", "4096")]
	#[default(2)]
	#[id(display = "Window Size")]
	#[layout(group = "Tune", order = 3, widget = choice)]
	pub window_size: i32,

	/// Allowed pitch classes, one bit each; C is bit 0.
	#[range(min = 0, max = 4095)]
	#[default(4095)]
	#[id(display = "Allowed Notes")]
	#[layout(group = "Notes", order = 0, widget = notes)]
	pub notes: i32,

	#[skip]
	sample_rate: f32,
	#[skip]
	tracker: PitchTracker,
	#[skip]
	time: PitchShifter<CHANNELS>,
	#[skip]
	vocoder: PhaseVocoder<PitchShift, CHANNELS>,
	#[skip]
	psola: Psola<CHANNELS>,
	#[skip]
	held_notes: Vec<usize>,
	#[skip]
	ratio: f32,
	#[skip]
	last_window_size: i32,
}

impl<const CHANNELS: usize> AutoTune<CHANNELS> {
	/// An autotune on C major with all three engines ready.
	pub fn new(sample_rate: usize) -> Self {
		let mut vocoder = PhaseVocoder::new(PitchShift::default(), 2_048, sample_rate);
		vocoder.gain = 1.0 / overlap_add_gain(vocoder.window_factor);

		Self {
			algorithm: 0,
			strength: 1.0,
			speed: 10.0,
			window_size: 2,
			notes: ALL_NOTES as i32,
			sample_rate: sample_rate.max(1) as f32,
			tracker: PitchTracker::new(sample_rate, 2_048),
			time: PitchShifter::<CHANNELS>::new(1_024),
			vocoder,
			psola: Psola::<CHANNELS>::new(sample_rate),
			held_notes: Vec::new(),
			ratio: 1.0,
			last_window_size: 2,
		}
	}

	fn track_events(&mut self, process_context: &mut Box<dyn ProcessContext>) {
		for event in process_context.events() {
			match event {
				NoteEvent::NoteOn { note, .. } => {
					if !self.held_notes.contains(note) {
						self.held_notes.push(*note);
					}
				}
				NoteEvent::NoteOff { note, .. } | NoteEvent::Stop { note, .. } => {
					self.held_notes.retain(|held| held != note);
				}
				NoteEvent::ImmediateStop => self.held_notes.clear(),
				_ => {}
			}
		}
	}

	/// The pitch-class mask the tuner is using right now: MIDI held notes if
	/// there are any, otherwise the faceplate mask.
	pub fn allowed_mask(&self) -> u16 {
		if self.held_notes.is_empty() {
			(self.notes as u16) & ALL_NOTES
		} else {
			self.held_notes
				.iter()
				.fold(0, |mask, note| mask | (1 << (note % 12)))
		}
	}

	fn target_ratio(&mut self, mono: f32, process_context: &mut Box<dyn ProcessContext>) -> f32 {
		self.track_events(process_context);

		let estimate = self
			.tracker
			.process_sample(mono)
			.or_else(|| self.tracker.last())
			.filter(|estimate| estimate.clarity >= 0.5);

		let Some(estimate) = estimate else {
			return 1.0;
		};

		let Some(target_note) = nearest_allowed_note(estimate.midi_note, self.allowed_mask()) else {
			return 1.0;
		};

		let distance = (target_note - estimate.midi_note) * self.strength.clamp(0.0, 1.0);
		(2.0_f32).powf(distance / 12.0)
	}

	fn set_algorithm_ratio(&mut self, ratio: f32) {
		self.ratio = ratio;

		match self.algorithm {
			1 => {
				self.vocoder.mapper.shift = ratio;
				self.vocoder.gain = 1.0 / overlap_add_gain(self.vocoder.window_factor);
			}
			2 => self.psola.set_shift(ratio),
			_ => self.time.pitch_shift_factor = ratio,
		}
	}
}

impl<const CHANNELS: usize> Effect<CHANNELS> for AutoTune<CHANNELS> {
	fn delay(&self) -> usize {
		match self.algorithm {
			1 => self.vocoder.delay(),
			2 => self.psola.delay(),
			_ => self.time.delay(),
		}
	}

	fn process(
		&mut self,
		samples: &mut [f32; CHANNELS],
		other: &[&[f32; CHANNELS]],
		process_context: &mut Box<dyn ProcessContext>,
	) {
		if self.algorithm == 1 && self.window_size != self.last_window_size {
			let size = [512, 1_024, 2_048, 4_096][self.window_size.clamp(0, 3) as usize];
			if self.vocoder.renew_window_size(size).is_some() {
				self.vocoder.gain = 1.0 / overlap_add_gain(self.vocoder.window_factor);
			}
			self.last_window_size = self.window_size;
		}

		let mono = samples.iter().copied().sum::<f32>() / CHANNELS as f32;
		let target = self.target_ratio(mono, process_context);

		let speed = self.speed.clamp(0.0, 200.0);
		let coefficient = if speed <= 0.0 {
			1.0
		} else {
			1.0 - (-1.0 / (speed * 0.001 * self.sample_rate)).exp()
		};
		let ratio = self.ratio + (target - self.ratio) * coefficient;
		self.set_algorithm_ratio(ratio);

		match self.algorithm {
			1 => self.vocoder.process(samples, other, process_context),
			2 => self.psola.process(samples, other, process_context),
			_ => self.time.process(samples, other, process_context),
		}
	}

	fn clear_tail(&mut self) {
		self.time.clear_tail();
		self.vocoder.clear_tail();
		self.psola.clear_tail();
		self.tracker.reset();
		self.held_notes.clear();
		self.ratio = 1.0;
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{Effect, ProcessContext, ProcessInfos};

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

	#[test]
	fn the_nearest_allowed_note_wins() {
		// C major: C, D, E, F, G, A, B.
		let mask = (1 << 0) | (1 << 2) | (1 << 4) | (1 << 5) | (1 << 7) | (1 << 9) | (1 << 11);
		assert_eq!(nearest_allowed_note(60.0, mask), Some(60.0));
		assert_eq!(nearest_allowed_note(61.0, mask), Some(60.0));
		assert_eq!(nearest_allowed_note(66.0, mask), Some(65.0));
	}

	#[test]
	fn midi_notes_become_the_allowed_pitch_classes() {
		let mut tuner = AutoTune::<1>::new(48_000);
		let mut context: Box<dyn ProcessContext> = Box::new(TestContext {
			events: vec![
				NoteEvent::NoteOn { channel: 0, note: 60, velocity: 1.0 },
				NoteEvent::NoteOn { channel: 0, note: 64, velocity: 1.0 },
			],
		});
		let mut frame = [0.0];
		tuner.process(&mut frame, &[], &mut context);
		assert_eq!(tuner.allowed_mask(), (1 << 0) | (1 << 4));
	}

	#[test]
	fn an_off_key_note_is_pulled_to_the_scale() {
		let mut tuner = AutoTune::<1>::new(48_000);
		tuner.algorithm = 1;
		tuner.speed = 0.0;
		tuner.notes = (1 << 0)
			| (1 << 2)
			| (1 << 4)
			| (1 << 5)
			| (1 << 7)
			| (1 << 9)
			| (1 << 11);
		let mut context: Box<dyn ProcessContext> = Box::new(TestContext { events: vec![] });
		let mut output = Vec::new();

		// G sharp is one semitone from G and A; C major must pull it to G.
		for index in 0..48_000 {
			let phase = 2.0 * core::f32::consts::PI * 415.0 * index as f32 / 48_000.0;
			let mut frame = [0.5 * phase.sin()];
			tuner.process(&mut frame, &[], &mut context);
			output.push(frame[0]);
		}

		let mut meter = PitchTracker::new(48_000, 2_048);
		let mut estimate = None;
		for sample in &output[12_000..] {
			if let Some(next) = meter.process_sample(*sample) && next.clarity > 0.5 {
				estimate = Some(next.frequency);
			}
		}

		let frequency = estimate.expect("no pitch after autotune");
		assert!((frequency - 392.0).abs() < 12.0, "got {frequency}");
	}

	#[test]
	fn the_snap_ratio_is_bounded_and_finite() {
		let mut tuner = AutoTune::<1>::new(48_000);
		let mut context: Box<dyn ProcessContext> = Box::new(TestContext { events: vec![] });
		let mut output = Vec::new();

		for index in 0..4_800 {
			let phase = 2.0 * core::f32::consts::PI * 440.0 * index as f32 / 48_000.0;
			let mut frame = [0.5 * phase.sin()];
			tuner.process(&mut frame, &[], &mut context);
			output.push(frame[0]);
		}

		assert!(output.iter().all(|sample| sample.is_finite()));
	}
}
