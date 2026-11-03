//! A phase-vocoder autotune that retunes each FFT bin independently.
//!
//! Every bin inside the selected band is moved to the nearest allowed pitch, so
//! the spectrum becomes a scale-quantized comb instead of a monophonic pitch
//! correction. Strength blends the move, Speed smooths it between frames, and the
//! low/high controls keep the operation away from the bass and the air band.
//!
//! The MIDI switch in the header decides where the allowed notes come from. On,
//! only the notes held on the port count: with nothing held, nothing is retuned,
//! and the faceplate mask is not a fallback. Off, the mask is used and the port is
//! ignored.

use i_am_dsp_derive::Parameters;

use crate::{
	Effect, NoteEvent, ProcessContext,
	effects::{
		autotune::nearest_allowed_note,
		phase_vocoder::{FrequencyMapper, PhaseVocoder, overlap_add_gain},
	},
};

/// One bit per pitch class; C is bit 0.
const ALL_NOTES: i32 = 0x0fff;

/// The frequency mapper behind SpectralRetuneEffect.
#[derive(Parameters)]
pub struct SpectralRetuneMapper {
	/// How far a bin is moved towards the nearest allowed note.
	#[range(min = 0.0, max = 1.0)]
	#[default(1.0)]
	pub strength: f32,
	/// The smoothing time for a bin's target frequency, in milliseconds.
	#[range(min = 0.0, max = 200.0)]
	#[default(10.0)]
	pub speed: f32,
	/// The lowest frequency that is retuned.
	#[range(min = 20.0, max = 20_000.0)]
	#[logarithmic]
	#[default(200.0)]
	pub low_frequency: f32,
	/// The highest frequency that is retuned.
	#[range(min = 20.0, max = 20_000.0)]
	#[logarithmic]
	#[default(8_000.0)]
	pub high_frequency: f32,
	/// Allowed pitch classes, one bit each; C is bit 0.
	#[range(min = 0, max = 4095)]
	#[default(4095)]
	pub notes: i32,
	#[skip]
	sample_rate: f32,
	#[skip]
	window_size: usize,
	/// The smoothed target for each bin, in hertz.
	#[skip]
	bins: Vec<f32>,
}

impl SpectralRetuneMapper {
	/// A mapper for the given sample rate and FFT size.
	pub fn new(sample_rate: usize, window_size: usize) -> Self {
		let mut mapper = Self {
			strength: 1.0,
			speed: 10.0,
			low_frequency: 200.0,
			high_frequency: 8_000.0,
			notes: ALL_NOTES,
			sample_rate: sample_rate.max(1) as f32,
			window_size: window_size.max(2),
			bins: Vec::new(),
		};
		mapper.resize(window_size);
		mapper
	}

	/// Rebuilds the per-bin smoothing state after a window-size change.
	pub fn resize(&mut self, window_size: usize) {
		self.window_size = window_size.max(2);
		let bins = self.window_size / 2 + 1;
		self.bins = (0..bins)
			.map(|index| index as f32 * self.sample_rate / self.window_size as f32)
			.collect();
	}

	fn coefficient(&self) -> f32 {
		if self.speed <= 0.0 {
			1.0
		} else {
			let frame_seconds = self.window_size as f32 * 0.25 / self.sample_rate;
			1.0 - (-frame_seconds / (self.speed * 0.001)).exp()
		}
	}
}

impl FrequencyMapper for SpectralRetuneMapper {
	fn map_frequency(&mut self, frequency: f32, amplitude: f32) -> (f32, f32) {
		let index = (frequency * self.window_size as f32 / self.sample_rate).round() as usize;
		if index >= self.bins.len() {
			return (frequency, amplitude);
		}

		if frequency < self.low_frequency || frequency > self.high_frequency {
			self.bins[index] = frequency;
			return (frequency, amplitude);
		}

		let midi = 69.0 + 12.0 * (frequency / 440.0).log2();
		let Some(target) = nearest_allowed_note(midi, (self.notes as u16) & 0x0fff) else {
			self.bins[index] = frequency;
			return (frequency, amplitude);
		};
		let target = 440.0 * 2.0_f32.powf((target - 69.0) / 12.0);

		let blended = frequency * (target / frequency).powf(self.strength.clamp(0.0, 1.0));
		self.bins[index] += (blended - self.bins[index]) * self.coefficient();

		(self.bins[index], amplitude)
	}
}

/// A spectral retune with a small product surface.
#[derive(Parameters)]
pub struct SpectralRetuneEffect {
	/// How far each bin is moved towards its nearest allowed note.
	#[range(min = 0.0, max = 1.0)]
	#[unit(percent)]
	#[default(1.0)]
	#[id(display = "Strength")]
	#[layout(group = "Retune", order = 0, widget = knob, size = large, hero = "autotune")]
	pub strength: f32,

	/// The smoothing time for the per-bin movement.
	#[range(min = 0.0, max = 200.0)]
	#[unit(milliseconds)]
	#[default(10.0)]
	#[id(display = "Speed")]
	#[layout(group = "Retune", order = 1, widget = knob)]
	pub speed: f32,

	/// The lowest frequency that is retuned.
	#[range(min = 20.0, max = 20_000.0)]
	#[logarithmic]
	#[unit(hertz)]
	#[default(200.0)]
	#[id(display = "Low")]
	#[layout(group = "Retune", order = 2, widget = knob)]
	pub low_frequency: f32,

	/// The highest frequency that is retuned.
	#[range(min = 20.0, max = 20_000.0)]
	#[logarithmic]
	#[unit(hertz)]
	#[default(8_000.0)]
	#[id(display = "High")]
	#[layout(group = "Retune", order = 3, widget = knob)]
	pub high_frequency: f32,

	/// The dry/wet balance.
	#[range(min = 0.0, max = 1.0)]
	#[unit(percent)]
	#[default(1.0)]
	#[id(display = "Mix")]
	#[layout(group = "Retune", order = 4, widget = knob)]
	pub mix: f32,

	/// Allowed pitch classes, one bit each; C is bit 0.
	#[range(min = 0, max = 4095)]
	#[default(4095)]
	#[id(display = "Allowed Notes")]
	#[layout(group = "Notes", order = 0, widget = notes)]
	pub notes: i32,

	/// Whether the note port sets the allowed notes.
	///
	/// On, only the notes held on the port count, and with nothing held nothing is
	/// retuned: the mask above is not a fallback. Off, the mask is the whole story
	/// and the port is ignored.
	#[default(false)]
	#[id(display = "MIDI")]
	#[layout(group = "Notes", order = 1, widget = toggle)]
	pub midi: bool,

	/// The FFT size, picked from the header.
	#[choices("512", "1024", "2048", "4096")]
	#[default(2)]
	#[id(display = "Window Size")]
	#[layout(group = "Retune", order = 3, widget = choice)]
	pub window_size: i32,

	#[skip]
	inner: PhaseVocoder<SpectralRetuneMapper, 2>,
	#[skip]
	last_window_size: i32,
	#[skip]
	held_notes: Vec<usize>,
}

impl SpectralRetuneEffect {
	/// A spectral retune with a 2048-sample FFT.
	pub fn new(sample_rate: usize) -> Self {
		let mapper = SpectralRetuneMapper::new(sample_rate, 2_048);
		let mut inner = PhaseVocoder::new(mapper, 2_048, sample_rate);
		inner.gain = 1.0 / overlap_add_gain(inner.window_factor);

		Self {
			strength: 1.0,
			speed: 10.0,
			low_frequency: 200.0,
			high_frequency: 8_000.0,
			mix: 1.0,
			notes: ALL_NOTES,
			midi: false,
			window_size: 2,
			inner,
			last_window_size: 2,
			held_notes: Vec::new(),
		}
	}

	fn track_events(&mut self, context: &mut Box<dyn ProcessContext>) {
		if !self.midi {
			self.held_notes.clear();
			return;
		}

		for event in context.events() {
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

	/// The pitch-class mask the retune is using right now.
	///
	/// With the note port on it is the held notes and nothing else: no notes held is
	/// a zero mask, which retunes nothing. With the port off it is the faceplate mask.
	pub fn allowed_mask(&self) -> u16 {
		if !self.midi {
			return (self.notes as u16) & 0x0fff;
		}

		self.held_notes
			.iter()
			.fold(0, |mask, note| mask | (1 << (note % 12)))
	}
}

impl Effect<2> for SpectralRetuneEffect {
	fn allowed_notes(&self) -> Option<u16> {
		Some(self.allowed_mask())
	}

	fn delay(&self) -> usize {
		self.inner.delay()
	}

	fn process(
		&mut self,
		samples: &mut [f32; 2],
		other: &[&[f32; 2]],
		context: &mut Box<dyn ProcessContext>,
	) {
		self.track_events(context);

		if self.window_size != self.last_window_size {
			let size = [512, 1_024, 2_048, 4_096][self.window_size.clamp(0, 3) as usize];
			if self.inner.renew_window_size(size).is_some() {
				self.inner.mapper.resize(size);
				self.inner.gain = 1.0 / overlap_add_gain(self.inner.window_factor);
			}
			self.last_window_size = self.window_size;
		}

		let dry = *samples;
		let notes = self.allowed_mask() as i32;
		let mapper = &mut self.inner.mapper;
		mapper.strength = self.strength;
		mapper.speed = self.speed;
		mapper.low_frequency = self.low_frequency;
		mapper.high_frequency = self.high_frequency;
		mapper.notes = notes;

		self.inner.process(samples, other, context);

		let mix = self.mix.clamp(0.0, 1.0);
		for (sample, dry) in samples.iter_mut().zip(dry) {
			*sample = dry * (1.0 - mix) + *sample * mix;
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	struct TestContext {
		events: Vec<NoteEvent>,
		infos: crate::ProcessInfos,
	}

	impl ProcessContext for TestContext {
		fn infos(&self) -> &crate::ProcessInfos {
			&self.infos
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

	fn mapper(mask: i32) -> SpectralRetuneMapper {
		let mut mapper = SpectralRetuneMapper::new(48_000, 2_048);
		mapper.strength = 1.0;
		mapper.speed = 0.0;
		mapper.low_frequency = 20.0;
		mapper.high_frequency = 20_000.0;
		mapper.notes = mask;
		mapper
	}

	#[test]
	fn a_bin_moves_to_the_nearest_allowed_note() {
		let mut mapper = mapper(1 << 0);
		let (frequency, _) = mapper.map_frequency(440.0, 1.0);
		assert!((frequency - 523.25).abs() < 1.0, "got {frequency}");
	}

	#[test]
	fn strength_zero_leaves_the_bin_alone() {
		let mut mapper = mapper(1 << 0);
		mapper.strength = 0.0;
		let (frequency, _) = mapper.map_frequency(440.0, 1.0);
		assert!((frequency - 440.0).abs() < 0.01);
	}

	#[test]
	fn midi_notes_override_the_faceplate_mask() {
		let mut effect = SpectralRetuneEffect::new(48_000);
		effect.midi = true;
		let mut context: Box<dyn ProcessContext> = Box::new(TestContext {
			events: vec![
				NoteEvent::NoteOn {
					channel: 0,
					note: 60,
					velocity: 1.0,
				},
				NoteEvent::NoteOn {
					channel: 0,
					note: 64,
					velocity: 1.0,
				},
			],
			infos: crate::ProcessInfos::default(),
		});
		let mut frame = [0.0_f32, 0.0];

		effect.process(&mut frame, &[], &mut context);

		assert_eq!(effect.allowed_mask(), (1 << 0) | (1 << 4));
	}

	#[test]
	fn the_faceplate_mask_is_not_a_fallback_when_midi_is_on() {
		let mut effect = SpectralRetuneEffect::new(48_000);
		effect.midi = true;
		effect.notes = (1 << 0) | (1 << 4);
		let mut context: Box<dyn ProcessContext> = Box::new(TestContext {
			events: vec![],
			infos: crate::ProcessInfos::default(),
		});
		let mut frame = [0.0_f32, 0.0];

		effect.process(&mut frame, &[], &mut context);

		assert_eq!(
			effect.allowed_mask(),
			0,
			"no notes held means nothing to retune"
		);
	}

	#[test]
	fn midi_off_ignores_the_port_and_uses_the_mask() {
		let mut effect = SpectralRetuneEffect::new(48_000);
		effect.midi = false;
		effect.notes = 1 << 0;
		let mut context: Box<dyn ProcessContext> = Box::new(TestContext {
			events: vec![NoteEvent::NoteOn {
				channel: 0,
				note: 64,
				velocity: 1.0,
			}],
			infos: crate::ProcessInfos::default(),
		});
		let mut frame = [0.0_f32, 0.0];

		effect.process(&mut frame, &[], &mut context);

		assert_eq!(effect.allowed_mask(), 1 << 0, "the port changed the mask");
	}

	#[test]
	fn mix_zero_is_the_dry_signal() {
		let mut effect = SpectralRetuneEffect::new(48_000);
		effect.mix = 0.0;
		let mut context: Box<dyn ProcessContext> = Box::new(TestContext {
			events: vec![],
			infos: crate::ProcessInfos::default(),
		});
		let mut frame = [0.5_f32, -0.25];

		effect.process(&mut frame, &[], &mut context);

		assert_eq!(frame, [0.5, -0.25]);
	}

	#[test]
	fn the_band_edges_are_not_moved() {
		let mut mapper = mapper(1 << 0);
		mapper.low_frequency = 1_000.0;
		mapper.high_frequency = 2_000.0;
		let (below, _) = mapper.map_frequency(440.0, 1.0);
		let (above, _) = mapper.map_frequency(4_000.0, 1.0);
		assert!((below - 440.0).abs() < 0.01);
		assert!((above - 4_000.0).abs() < 0.01);
	}
}
