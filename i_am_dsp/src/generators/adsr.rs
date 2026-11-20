//! ADSR (Attack, Decay, Sustain, Release) envelope generator

use std::collections::HashMap;

use i_am_dsp_derive::Parameters;

#[cfg(feature = "real_time_demo")]
use crate::tools::ring_buffer::RingBuffer;
use crate::{
	Generator, NoteEvent, ProcessInfos,
	prelude::{Oscillator, SyncedTime},
};

pub(crate) struct Note {
	// pub channel: u8,
	pub note: usize,
	pub velocity: f32,
}

mod tuning;
pub use tuning::*;

mod state;

use state::*;

/// How often a carried phase is rebuilt from the note's absolute time.
///
/// One block: carrying the phase is what saves a multiply per copy, and 64
/// samples of f32 accumulation is far below anything the ear can place.
const PHASE_RESYNC: usize = 64;

/// A warper to play an oscillator with ADSR envelope.
///
/// Adsr supports Midi Events, but cannot distinguish the note'channel.
///
/// Actually this is AHDSR.
///
/// The Bend function of Adsr is $\frac{\exp(bend_factor * time) - 1}{\exp(bend_factor) - 1}$
#[derive(Parameters)]
#[default_float_range(min = 0.0, max = 4000.0)]
pub struct Adsr<Osc: Oscillator<CHANNELS>, TuningSys: Tuning, const CHANNELS: usize> {
	#[sub_param]
	/// The oscillator to be played with ADSR envelope
	#[id(display = "")]
	#[layout(group = "Shape", order = 0)]
	pub oscillator: Osc,
	#[skip]
	tuning_sys: TuningSys,
	#[skip]
	note_playing: HashMap<usize, Vec<PlayingNote<CHANNELS>>>,
	/// The sample rate of the audio processing system
	///
	/// Saves in Hz
	#[skip]
	pub sample_rate: usize,
	/// The delay time, saves in milliseconds
	#[unit(milliseconds)]
	#[default(0.0)]
	#[id(display = "Delay")]
	#[layout(group = "Envelope", order = 0, widget = knob, hero = "envelope")]
	pub delay_time: f32,
	/// The same stage length in musical time: a sixteenth note of delay stays a sixteenth
	/// note whatever the host is playing at. See `Self::times` for how it is read.
	#[sub_param]
	#[id(display = "Delay")]
	#[layout(group = "Envelope", order = 1)]
	pub delay: SyncedTime,
	/// The attack time, saves in milliseconds
	#[unit(milliseconds)]
	#[default(10.0)]
	#[id(display = "Attack")]
	#[layout(group = "Envelope", order = 2, widget = knob)]
	pub attack_time: f32,
	/// The same stage length in musical time.
	#[sub_param]
	#[id(display = "Attack")]
	#[layout(group = "Envelope", order = 3)]
	pub attack: SyncedTime,
	/// The attack bend, 0.0 for no bend
	#[range(min = -10.0, max = 10.0)]
	#[default(0.0)]
	#[id(display = "Attack Bend")]
	#[layout(group = "Envelope", order = 4, widget = knob, bipolar)]
	pub attack_bend: f32,
	/// The hold time, saves in milliseconds
	#[unit(milliseconds)]
	#[default(100.0)]
	#[id(display = "Hold")]
	#[layout(group = "Envelope", order = 5, widget = knob)]
	pub hold_time: f32,
	/// The same stage length in musical time.
	#[sub_param]
	#[id(display = "Hold")]
	#[layout(group = "Envelope", order = 6)]
	pub hold: SyncedTime,
	/// The decay time, saves in milliseconds
	#[unit(milliseconds)]
	#[default(100.0)]
	#[id(display = "Decay")]
	#[layout(group = "Envelope", order = 7, widget = knob)]
	pub decay_time: f32,
	/// The same stage length in musical time.
	#[sub_param]
	#[id(display = "Decay")]
	#[layout(group = "Envelope", order = 8)]
	pub decay: SyncedTime,
	/// The decay bend, 0.0 for no bend
	#[range(min = -10.0, max = 10.0)]
	#[default(0.0)]
	#[id(display = "Decay Bend")]
	#[layout(group = "Envelope", order = 9, widget = knob, bipolar)]
	pub decay_bend: f32,
	/// The sustain level, saves in linear scale
	#[range(min = 0.0, max = 1.0)]
	#[default(1.0)]
	#[id(display = "Sustain")]
	#[layout(group = "Envelope", order = 10, widget = knob)]
	pub sustain_level: f32,
	/// The release time, saves in milliseconds
	#[unit(milliseconds)]
	#[default(100.0)]
	#[id(display = "Release")]
	#[layout(group = "Envelope", order = 11, widget = knob)]
	pub release_time: f32,
	/// The same stage length in musical time.
	#[sub_param]
	#[id(display = "Release")]
	#[layout(group = "Envelope", order = 12)]
	pub release: SyncedTime,
	/// The release bend, 0.0 for no bend
	#[range(min = -10.0, max = 10.0)]
	#[default(0.0)]
	#[id(display = "Release Bend")]
	#[layout(group = "Envelope", order = 13, widget = knob, bipolar)]
	pub release_bend: f32,
	/// Whether every stage follows the host's tempo instead of its own free length.
	///
	/// One switch for the whole envelope: with it on, every stage takes its length
	/// from the tempo, and the free lengths stand in when the host reports none.
	#[default(false)]
	#[id(display = "Sync")]
	#[layout(group = "Envelope", order = 14, widget = toggle)]
	pub sync: bool,
	/// saves in linear scale
	#[range(min = 0.01, max = 4.0)]
	#[logarithmic]
	#[default(0.8)]
	#[id(display = "Gain")]
	#[layout(group = "Shape", order = 1, widget = knob, size = large, hero = "shape")]
	pub gain: f32,
	/// The number of unison, 1 for no unison
	#[range(min = 1, max = 32)]
	#[default(1)]
	#[id(display = "Unison")]
	#[layout(group = "Voice", order = 0, widget = knob, hero = "unison")]
	pub unisons: usize,
	/// Maxium detune of each unison, the unit is note
	#[range(min = 0.0, max = 2.0)]
	#[default(2.0)]
	#[id(display = "Detune")]
	#[layout(group = "Voice", order = 1, widget = knob)]
	pub unison_detune: f32,
	/// Bend of each unison, 0.0 for no bend
	#[range(min = -10.0, max = 10.0)]
	#[default(0.0)]
	#[id(display = "Unison Bend")]
	#[layout(group = "Voice", order = 2, widget = knob, bipolar)]
	pub unison_bend: f32,
	/// The blend of each unison, 0.0 for no blend
	///
	/// This will lower the amplitude of "side" unison.
	#[range(min = 0.0, max = 1.0)]
	#[default(0.0)]
	#[id(display = "Blend")]
	#[layout(group = "Voice", order = 3, widget = knob)]
	pub unison_blend: f32,
	/// The random phase range of each unison, 0.0 for no random phase
	#[range(min = 0.0, max = 1.0)]
	#[default(1.0)]
	#[id(display = "Random Phase")]
	#[layout(group = "Voice", order = 4, widget = knob)]
	pub random_phase: f32,
	#[range(min = 0.25, max = 4.0)]
	#[logarithmic]
	#[default(1.0)]
	#[id(display = "Pitch")]
	#[layout(group = "Shape", order = 2, widget = knob)]
	/// The pitch factor of the oscillator, used for pitch shifting.
	pub pitch_factor: f32,
	/// Where the voice sits in the stereo field: -1.0 hard left, 0.0 centre, 1.0 hard right.
	///
	/// [`Self::random_pan`] spreads the unisons around this position, so the two do not
	/// fight: the position moves the whole voice, the spread widens it.
	#[range(min = -1.0, max = 1.0)]
	#[default(0.0)]
	#[id(display = "Pan")]
	#[layout(group = "Shape", order = 3, widget = knob, bipolar)]
	pub pan: f32,
	/// The random pan range of each unison, 0.0 for no random pan
	#[range(min = 0.0, max = 1.0)]
	#[default(1.0)]
	#[id(display = "Random Pan")]
	#[layout(group = "Voice", order = 5, widget = knob)]
	pub random_pan: f32,
	/// If true, the phase of each channel will also apply random phase.
	#[default(false)]
	#[id(display = "Phase per Channel")]
	#[layout(group = "Voice", order = 7, widget = toggle)]
	pub random_phase_by_channel: bool,
	/// The start phase postion of the oscillator, 0.0 for no phase shift.
	#[range(min = 0.0, max = 1.0)]
	#[default(0.0)]
	#[id(display = "Start Phase")]
	#[layout(group = "Voice", order = 6, widget = knob)]
	pub phase_start: f32,
	/// How long a new note slides from the pitch of the one before it.
	///
	/// Zero is off: a note starts at its own pitch. Anything else is portamento, and
	/// the slide is exponential in frequency — linear in pitch — which is what makes
	/// a long glide sound even. The slider is appended to the struct rather than
	/// inserted between the older parameters so a host that addresses them by
	/// position keeps its saved automation.
	#[range(min = 0.0, max = 2000.0)]
	#[logarithmic]
	#[unit(milliseconds)]
	#[default(0.0)]
	#[id(display = "Glide")]
	#[layout(group = "Voice", order = 8, widget = knob)]
	pub glide_time: f32,

	/// The stage lengths in milliseconds this block runs at: the free values, or what the
	/// tempo makes of them for the stages that are syncing.
	///
	/// The tempo lives in the process context, which the per sample envelope does not see, so
	/// it is read once a block and kept here.
	#[skip]
	times: EnvelopeTimes,
	/// The effect attach to a single note, if any.
	///
	/// Takes a mutable reference to the buffer, the playing note, and the sample rate.
	///
	/// Note: this field will **not** be saved in the preset file.
	#[skip]
	pub note_effect: Option<NoteProcessor<CHANNELS>>,

	/// The pitch each channel's last note started at, for the glide.
	///
	/// A frequency, not a note number: the tuning system decides what a note means,
	/// so the glide has to slide between the pitches the oscillator really played.
	#[skip]
	last_pitch: [f32; 16],
	/// The note a caller asked the next one to glide from, per channel.
	///
	/// A voice that is rebuilt for every note — the patcher does that — cannot rely
	/// on `last_pitch`, because its instance is new. It primes the one it is about to
	/// start instead.
	#[skip]
	pending_from: [Option<usize>; 16],

	#[cfg(feature = "real_time_demo")]
	#[skip]
	history: [RingBuffer<f32>; CHANNELS],
}

impl<Osc: Oscillator<CHANNELS>, TuningSys: Tuning, const CHANNELS: usize>
	Adsr<Osc, TuningSys, CHANNELS>
{
	/// Create a new Adsr with given oscillator and tuning system
	///
	/// # Panics
	///
	/// 1. If `sample_rate` is 0.
	/// 2. If `CHANNELS` is 0.
	pub fn new(oscillator: Osc, tuning_sys: TuningSys, sample_rate: usize) -> Self {
		assert!(CHANNELS > 0);
		assert!(sample_rate > 0);

		Self {
			oscillator,
			tuning_sys,
			note_playing: HashMap::new(),
			sample_rate,
			delay_time: 0.0,
			delay: SyncedTime::default(),
			attack_time: 10.0,
			attack: SyncedTime::default(),
			attack_bend: 0.0,
			hold_time: 100.0,
			hold: SyncedTime::default(),
			decay_time: 100.0,
			decay: SyncedTime::default(),
			decay_bend: 0.0,
			sustain_level: 1.0,
			release_time: 100.0,
			release: SyncedTime::default(),
			release_bend: 0.0,
			sync: false,
			gain: 0.8,
			unisons: 1,
			unison_detune: 2.0,
			unison_bend: 0.0,
			unison_blend: 0.0,
			random_phase: 1.0,
			pitch_factor: 1.0,
			pan: 0.0,
			random_pan: 1.0,
			random_phase_by_channel: false,
			phase_start: 0.0,
			glide_time: 0.0,
			times: EnvelopeTimes::DEFAULT,
			note_effect: None,
			last_pitch: [f32::NAN; 16],
			pending_from: [None; 16],

			#[cfg(feature = "real_time_demo")]
			history: core::array::from_fn(|_| RingBuffer::new(4096)),
		}
	}

	fn sample_count(&self, count: usize) -> f32 {
		let time = count as f32 / self.sample_rate as f32 * 1000.0;
		self.sample_time(time)
	}

	/// Reads the stage lengths for this block, in milliseconds.
	///
	/// A synced stage takes its length from the tempo instead of its own milliseconds. The
	/// tempo changes with the host, so this runs once a block rather than once a sample.
	fn update_times(&mut self, infos: &ProcessInfos) {
		// The envelope's one Sync switch overrides the per-stage synced values: a
		// stage's own switch is a host automation detail, the product only shows
		// the envelope-wide one.
		let sync_all = self.sync;
		let to_ms = |synced: &SyncedTime, free_ms: f32| {
			if sync_all {
				return synced
					.synced_seconds(infos)
					.map_or(free_ms, |seconds| seconds * 1000.0);
			}

			synced.seconds(infos, free_ms) * 1000.0
		};

		self.times = EnvelopeTimes {
			delay: to_ms(&self.delay, self.delay_time),
			attack: to_ms(&self.attack, self.attack_time),
			hold: to_ms(&self.hold, self.hold_time),
			decay: to_ms(&self.decay, self.decay_time),
			release: to_ms(&self.release, self.release_time),
		};
	}

	/// Fill in the per-unison table for a note.
	///
	/// The detune, blend and pan of a unison only change when one of the controls
	/// moves, so they are computed once here rather than once per unison per
	/// sample: `get_frequency` is a 2^x, and sixteen unisons would ask for it
	/// sixteen times a sample.
	fn rebuild_unisons(&self, playing_note: &mut PlayingNote<CHANNELS>) {
		let unisons = playing_note.phase_start.len();
		let mid_point = (unisons - 1) as f32 / 2.0;
		let mut voices = Vec::with_capacity(unisons);

		for i in 0..unisons {
			let index = if mid_point == 0.0 {
				0.0
			} else {
				(i as f32 - mid_point) / mid_point
			};
			let blend = 1.0 - index.abs() * self.unison_blend;
			let index = if index >= 0.0 {
				bend(index, self.unison_bend)
			} else {
				-bend(index.abs(), self.unison_bend)
			};
			let frequency = self
				.tuning_sys
				.get_frequency(playing_note.note.note as f32 + self.unison_detune * index);
			let mut gains = [0.0; CHANNELS];

			for (channel, gain) in gains.iter_mut().enumerate() {
				*gain = if CHANNELS == 1 {
					1.0
				} else {
					let angle_uniformed = channel as f32 / (CHANNELS - 1) as f32;
					let pan_factor = self.pan_position(playing_note.pan_factors[i]);
					let delta = (angle_uniformed - pan_factor).abs();
					2.0 - delta * 2.0
				};
			}

			let mut weight = [0.0; CHANNELS];

			for (channel, weight) in weight.iter_mut().enumerate() {
				*weight = blend * gains[channel];
			}

			voices.push(UnisonVoice { frequency, weight });
		}

		playing_note.unison_voices = voices;
		// The detune moved, so the copies' frequencies are stale.
		playing_note.pitch_last = f32::NAN;
		playing_note.pitch_applied.clear();
		playing_note.pitch_applied.resize(unisons, f32::NAN);
		playing_note.unison_controls = Some(UnisonControls {
			detune: self.unison_detune,
			bend: self.unison_bend,
			blend: self.unison_blend,
			pan: self.pan,
			random_pan: self.random_pan,
		});
	}

	/// Where a unison sits in the stereo field, 0.0 for hard left and 1.0 for hard right.
	///
	/// `unison_pan` is that unison's random position, which [`Self::random_pan`] scales and
	/// [`Self::pan`] moves.
	fn pan_position(&self, unison_pan: f32) -> f32 {
		(unison_pan / 2.0 * self.random_pan + 0.5 + self.pan / 2.0).clamp(0.0, 1.0)
	}

	/// Sample the ADSR envelope at given time in milliseconds
	pub fn sample_time(&self, mut time: f32) -> f32 {
		if time < self.times.delay {
			return 0.0;
		}
		time -= self.times.delay;

		if time < self.times.attack {
			let t = time / self.times.attack;
			// let a = bend(self.attack_time, self.attack_bend);
			return bend(t, self.attack_bend);
		}
		time -= self.times.attack;

		if time < self.times.hold {
			return 1.0;
		}
		time -= self.times.hold;

		if time < self.times.decay {
			let t = time / self.times.decay;
			let a = bend(t, self.decay_bend);
			let v = 1.0 + (self.sustain_level - 1.0) * a;
			return v;
		}

		self.sustain_level
	}

	/// Replaces the attached effect to a single note.
	pub fn replace_note_efftct<T>(&mut self, effect: T)
	where
		T: Fn(&mut [f32; CHANNELS], &PlayingNote<CHANNELS>, usize) + 'static + Send + Sync,
	{
		self.note_effect = Some(Box::new(effect));
	}

	/// Remove the attached effect to a single note.
	pub fn remove_note_effect(&mut self) {
		self.note_effect = None;
	}

	fn sample_release_time(&self, time: f32) -> Option<f32> {
		if time >= self.times.release {
			return None;
		}
		let t = time / self.times.release;
		let a = bend(t, self.release_bend);
		Some(1.0 - a)
	}
}

impl<
	Osc: Oscillator<CHANNELS> + Send + Sync,
	TuningSys: Tuning + Send + Sync,
	const CHANNELS: usize,
> Generator<CHANNELS> for Adsr<Osc, TuningSys, CHANNELS>
{
	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"ADSRed Generator"
	}

	fn glide_from(&mut self, channel: u8, note: usize) {
		let channel = (channel as usize).min(15);
		self.pending_from[channel] = Some(note);
	}

	fn generate(
		&mut self,
		process_context: &mut Box<dyn crate::ProcessContext>,
	) -> [f32; CHANNELS] {
		// The driver's other entry point renders the same oscillator, so it has
		// to be told to sync here too: a drawn table or canvas learns about an
		// edit only through it.
		self.oscillator.sync();

		for event in process_context.events() {
			match event {
				NoteEvent::NoteOn {
					note,
					velocity,
					channel,
				} => {
					let note = Note {
						// channel: *channel,
						note: *note,
						velocity: *velocity,
					};
					let phases = (0..self.unisons)
						.map(|_| {
							if self.random_phase != 0.0 {
								std::array::from_fn(|_| {
									(rand::random_range(0.0..=self.random_phase) + self.phase_start)
										% 1.0
								})
							} else {
								[self.phase_start; CHANNELS]
							}
						})
						.collect();
					let mut pan_factors = (0..self.unisons)
						.map(|_| rand::random_range(-1.0..=1.0))
						.collect::<Vec<_>>();

					let mean = pan_factors.iter().sum::<f32>() / pan_factors.len() as f32;
					pan_factors.iter_mut().for_each(|inner| *inner -= mean);
					let max = pan_factors
						.iter()
						.max_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap())
						.unwrap()
						.abs();
					pan_factors.iter_mut().for_each(|inner| {
						if max != 0.0 {
							*inner /= max;
						}
					});

					// The glide starts from the pitch of the previous note, in the same
					// channel. A caller that rebuilt the voice for this note primed the
					// pitch it should come from; otherwise the last note this instance
					// played is the one to slide away from.
					let channel_index = (*channel as usize).min(15);
					let target = self.tuning_sys.get_frequency(note.note as f32);
					let from = self.pending_from[channel_index]
						.take()
						.map(|note| self.tuning_sys.get_frequency(note as f32))
						.or_else(|| {
							let last = self.last_pitch[channel_index];

							last.is_finite().then_some(last)
						});
					let (pitch_factor, glide_step, glide_left) = match from {
						Some(from)
							if self.glide_time > 0.0
								&& from > 0.0
								&& target > 0.0
								&& (from - target).abs() > 0.01 =>
						{
							let samples = (self.glide_time / 1000.0 * self.sample_rate as f32)
								.round()
								.max(1.0) as u32;

							(
								from / target,
								(target / from).powf(1.0 / samples as f32),
								samples,
							)
						}
						_ => (1.0, 1.0, 0),
					};
					self.last_pitch[channel_index] = target;

					let playing_note = PlayingNote {
						note,
						count: 0,
						release: None,
						phase_start: phases,
						pan_factors,
						last_phase: vec![0.0; self.unisons],
						frequencies: Vec::new(),
						steps: Vec::new(),
						pitch_last: f32::NAN,
						pitch_applied: vec![f32::NAN; self.unisons],
						unison_voices: Vec::new(),
						unison_controls: None,
						last_time: 0.0,
						current_pitch_factor: pitch_factor,
						glide_step,
						glide_left,
						glide_finished: false,
						phase_bias: Vec::new(),
						channel: *channel,
					};
					self.note_playing
						.entry(playing_note.note.note)
						.or_default()
						.push(playing_note);
				}
				NoteEvent::NoteOff { note, .. } => {
					if let Some(playing_notes) = self.note_playing.remove(note) {
						for mut playing_note in playing_notes {
							let sample = self.sample_count(playing_note.count);
							playing_note.release = Some((sample, playing_note.count));
							self.note_playing
								.entry(*note)
								.or_default()
								.push(playing_note);
						}
					}
				}
				NoteEvent::Stop { note, .. } => {
					self.note_playing.remove(note);
				}
				NoteEvent::ImmediateStop => {
					self.note_playing.clear();
				}
				_ => {}
			}
		}

		let mut output = [0.0; CHANNELS];

		self.sample_rate = process_context.infos().sample_rate;
		self.update_times(process_context.infos());
		// An oscillator whose parameters were just changed can rebuild its
		// derived state here, once per sample rather than once per unison.
		self.oscillator.sync();
		let sample_rate = self.sample_rate as f32;
		let mut note_playing = std::mem::take(&mut self.note_playing);
		note_playing.retain(|_note, playing_notes| {
			playing_notes.retain_mut(|playing_note| {
				let unisons = playing_note.phase_start.len();

				let gain = if let Some((release, release_count)) = playing_note.release {
					let time = (playing_note.count - release_count) as f32 / sample_rate * 1000.0;
					let Some(gain) = self.sample_release_time(time) else {
						process_context.send_event(NoteEvent::NoteEnd {
							channel: playing_note.channel,
							note: playing_note.note(),
						});
						return false;
					};
					gain * playing_note.note.velocity * release
				} else {
					let time = playing_note.count as f32 / sample_rate * 1000.0;
					self.sample_time(time) * playing_note.note.velocity
				};

				let time = playing_note.count as f32 / sample_rate;
				let mut output_samples = [0.0; CHANNELS];
				let controls = UnisonControls {
					detune: self.unison_detune,
					bend: self.unison_bend,
					blend: self.unison_blend,
					pan: self.pan,
					random_pan: self.random_pan,
				};

				if playing_note.unison_voices.len() != unisons
					|| playing_note.unison_controls != Some(controls)
				{
					self.rebuild_unisons(playing_note);
				}

				// A glide walks the factor towards 1.0, one multiplication a sample: the
				// frequency moves by a constant ratio, which is a constant number of
				// cents, so the slide sounds even.
				if playing_note.glide_left > 0 {
					playing_note.glide_left -= 1;
					playing_note.current_pitch_factor = if playing_note.glide_left == 0 {
						playing_note.glide_finished = true;
						1.0
					} else {
						playing_note.current_pitch_factor * playing_note.glide_step
					};
				}

				let pitch = self.pitch_factor * playing_note.current_pitch_factor;
				let elapsed = time - playing_note.last_time;
				let unison_gain = gain / unisons as f32;
				let absolute = self.oscillator.absolute_time();

				// The pitch moves far less often than a sample: when it has not,
				// the copies' frequencies and steps are the ones the last sample
				// computed, and no copy has to be re-tuned.
				if playing_note.pitch_last != pitch {
					if playing_note.frequencies.len() != unisons {
						playing_note.frequencies.resize(unisons, 0.0);
						playing_note.steps.resize(unisons, 0.0);
					}

					let rate = self.sample_rate as f32;

					for i in 0..unisons {
						let frequency = playing_note.unison_voices[i].frequency * pitch;
						playing_note.frequencies[i] = frequency;
						playing_note.steps[i] = frequency / rate;

						if playing_note.pitch_applied[i] != frequency {
							self.oscillator.set_pitch(frequency, self.sample_rate);
							playing_note.pitch_applied[i] = frequency;
						}
					}

					playing_note.pitch_last = pitch;
				}

				// The glide just ended, so the pitch is the note's own from here on and
				// the absolute read is right again, give or take the phase the slide
				// travelled: remember that gap per copy and add it back at every resync.
				if playing_note.glide_finished {
					playing_note.glide_finished = false;
					playing_note.phase_bias.clear();
					// `last_phase` is the phase of the sample *before* this one, so it
					// is compared against the absolute read at that sample: anchoring
					// it one sample later would leave every resync a step short.
					let before =
						playing_note.count.saturating_sub(1) as f32 / self.sample_rate as f32;
					playing_note.phase_bias.extend((0..unisons).map(|i| {
						let absolute = crate::generators::wavetable::wrap01(
							playing_note.frequencies[i] * before,
						);

						crate::generators::wavetable::wrap01(playing_note.last_phase[i] - absolute)
					}));
				}

				// A phase carried forward is one add a copy, but f32 loses the
				// fraction over minutes; the absolute read puts it back periodically.
				let resync = playing_note.count % PHASE_RESYNC == 0;

				for i in 0..unisons {
					let voice = playing_note.unison_voices[i];
					let frequency = playing_note.frequencies[i];
					let t = if absolute {
						frequency * elapsed + playing_note.last_phase[i]
					} else if resync && playing_note.glide_left == 0 {
						// The absolute read assumes the pitch has been what it is now
						// since the note started. That is false while a glide runs, and
						// snapping to it would jump the phase — a click in the middle of
						// every slide. The carried phase is the right one, and a glide is
						// short enough that f32 drift cannot matter over it.
						crate::generators::wavetable::wrap01(
							frequency * time
								+ playing_note.phase_bias.get(i).copied().unwrap_or(0.0),
						)
					} else {
						let next = playing_note.last_phase[i] + playing_note.steps[i];

						if next >= 1.0 { next - 1.0 } else { next }
					};

					playing_note.last_phase[i] = t;

					let samples = self.oscillator.play_at(
						1.0,
						t,
						if self.random_phase_by_channel {
							playing_note.phase_start[i]
						} else {
							std::array::from_fn(|_| playing_note.phase_start[i][0])
						},
					);

					for (channel, output_samples) in output_samples.iter_mut().enumerate() {
						*output_samples += samples[channel] * unison_gain * voice.weight[channel];
					}
				}
				playing_note.last_time = time;
				if let Some(processor) = &self.note_effect {
					processor(&mut output_samples, playing_note, self.sample_rate)
				}
				for i in 0..CHANNELS {
					output[i] += output_samples[i] * gain;
				}

				playing_note.count += 1;

				true
			});
			!playing_notes.is_empty()
		});
		self.note_playing = note_playing;

		for (i, output) in output.iter_mut().enumerate() {
			*output *= self.gain;
			#[cfg(feature = "real_time_demo")]
			{
				self.history[i].push(*output);
			}
			let _ = i;
		}

		output
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		use crate::tools::ui_tools::draw_envelope;
		use crate::tools::ui_tools::{gain_ui, synced_time_ui};
		use egui::emath::RectTransform;
		use egui::*;

		CollapsingHeader::new("Oscillator Settings")
			.id_salt(format!("{}_oscillator_settings", id_prefix))
			.show(ui, |ui| {
				self.oscillator
					.demo_ui(ui, format!("{}_oscillator", id_prefix));
			});

		ui.horizontal(|ui| {
			Frame::canvas(ui.style()).show(ui, |ui| {
				const POINTS: usize = 128;
				let (_, rect) = ui.allocate_space(Vec2::splat(100.0));
				let total_time = self.times.total();
				let none_release_time = self.times.held();
				let to_screen =
					RectTransform::from_to(Rect::from_x_y_ranges(0.0..=1.0, -1.0..=0.0), rect);

				let mut positions = vec![Pos2::ZERO; POINTS];

				for (i, position) in positions.iter_mut().enumerate() {
					let time = i as f32 / POINTS as f32 * total_time;
					let sample = if time < none_release_time {
						self.sample_time(time)
					} else {
						self.sample_release_time(time - none_release_time)
							.unwrap_or(0.0)
							* self.sustain_level
					};
					*position = pos2(time / total_time, -sample);
					*position = to_screen * *position;
				}

				ui.painter()
					.extend([Shape::line(positions, (3.0, Color32::WHITE))]);
			});

			Frame::canvas(ui.style()).show(ui, |ui| {
				let (_, rect) = ui.allocate_space(Vec2::splat(100.0));
				let to_screen =
					RectTransform::from_to(Rect::from_x_y_ranges(-1.0..=1.0, -1.0..=0.0), rect);

				let mid_point = (self.unisons - 1) as f32 / 2.0;
				let shapes = (0..self.unisons).map(|i| {
					let index = if mid_point == 0.0 {
						0.0
					} else {
						(i as f32 - mid_point) / mid_point
					};
					let blend = 1.0 - index.abs() * self.unison_blend;
					let index = if index >= 0.0 {
						bend(index, self.unison_bend)
					} else {
						-bend(index.abs(), self.unison_bend)
					};
					let detune_factor = self.unison_detune * index / 2.0;
					let position_1 = to_screen * pos2(detune_factor, -blend);
					let position_2 = to_screen * pos2(detune_factor, 0.0);
					Shape::line(vec![position_1, position_2], (3.0, Color32::WHITE))
				});

				ui.painter().extend(shapes);
			});

			let history = self.history.iter().collect::<Vec<_>>();
			draw_envelope(ui, &history, true);
		});

		ui.separator();

		Grid::new(format!("{}_adsr_grid", id_prefix))
			.num_columns(6)
			.show(ui, |ui| {
				ui.label("Delay time");
				synced_time_ui(
					ui,
					"adsr_delay",
					&mut self.delay,
					&mut self.delay_time,
					0.0..=1000.0,
					"Delay (ms)",
				);
				ui.label("Attack time");
				synced_time_ui(
					ui,
					"adsr_attack",
					&mut self.attack,
					&mut self.attack_time,
					0.0..=1000.0,
					"Attack (ms)",
				);
				ui.label("Hold Time");
				synced_time_ui(
					ui,
					"adsr_hold",
					&mut self.hold,
					&mut self.hold_time,
					0.0..=1000.0,
					"Hold (ms)",
				);
				ui.end_row();

				ui.label("Decay Time");
				synced_time_ui(
					ui,
					"adsr_decay",
					&mut self.decay,
					&mut self.decay_time,
					0.0..=1000.0,
					"Decay (ms)",
				);
				ui.label("Sustain Level");
				gain_ui(ui, &mut self.sustain_level, Some("".to_string()), true);
				ui.label("Release Time");
				synced_time_ui(
					ui,
					"adsr_release",
					&mut self.release,
					&mut self.release_time,
					0.0..=1000.0,
					"Release (ms)",
				);
				ui.end_row();

				ui.label("Attack bend");
				ui.add(Slider::new(&mut self.attack_bend, -10.0..=10.0));
				ui.label("Decay bend");
				ui.add(Slider::new(&mut self.decay_bend, -10.0..=10.0));
				ui.label("Release bend");
				ui.add(Slider::new(&mut self.release_bend, -10.0..=10.0));
				ui.end_row();

				ui.label("Unison");
				ui.add(Slider::new(&mut self.unisons, 1..=32).text(""));
				ui.label("Unison detune");
				ui.add(Slider::new(&mut self.unison_detune, 0.0..=2.0).suffix("notes"));
				ui.label("Unison bend");
				ui.add(Slider::new(&mut self.unison_bend, -10.0..=10.0));
				ui.end_row();

				ui.label("Unison blend");
				ui.add(Slider::new(&mut self.unison_blend, 0.0..=1.0));
				ui.label("Random Phase");
				ui.add(Slider::new(&mut self.random_phase, 0.0..=1.0));
				ui.label("Phase Start");
				ui.add(Slider::new(&mut self.phase_start, 0.0..=1.0));
				ui.end_row();

				ui.label("Pan");
				ui.add(Slider::new(&mut self.pan, -1.0..=1.0));
				ui.label("Random Pan");
				ui.add(Slider::new(&mut self.random_pan, 0.0..=1.0));
				ui.label("Pitch Factor");
				ui.add(Slider::new(&mut self.pitch_factor, 0.25..=4.0).logarithmic(true));
				ui.label("Gain");
				gain_ui(ui, &mut self.gain, Some("".to_string()), false);
				ui.end_row();

				if ui
					.selectable_label(self.random_phase_by_channel, "Random Channel")
					.clicked()
				{
					self.random_phase_by_channel = !self.random_phase_by_channel
				}
			});
	}
}

pub(crate) fn bend(t: f32, bend: f32) -> f32 {
	// One copy of the maths, in the same place a page editor reads it from.
	crate::tools::page::segment_bend(t, bend)
}
#[cfg(test)]
mod tests {
	use super::*;
	use crate::ProcessContext;
	use crate::generators::wavetable::{SineWave, TableOsc};

	type TestAdsr = Adsr<TableOsc<SineWave>, EqualTemperament, 2>;

	fn adsr() -> TestAdsr {
		Adsr::new(TableOsc(SineWave), EqualTemperament, 48_000)
	}

	fn infos_with_tempo(tempo: f32) -> ProcessInfos {
		ProcessInfos {
			sample_rate: 48_000,
			trustable: true,
			playing: true,
			tempo: Some(tempo),
			..ProcessInfos::default()
		}
	}

	/// Without syncing, the stages keep the milliseconds they were given.
	#[test]
	fn an_unsynced_envelope_keeps_its_milliseconds() {
		let mut adsr = adsr();
		adsr.attack_time = 20.0;
		adsr.release_time = 300.0;
		adsr.update_times(&infos_with_tempo(120.0));

		assert_eq!(adsr.times.attack, 20.0);
		assert_eq!(adsr.times.release, 300.0);
		// Even at a tempo that would make a synced stage something else entirely.
		adsr.update_times(&infos_with_tempo(240.0));
		assert_eq!(adsr.times.attack, 20.0);
	}

	/// A synced stage takes its length from the tempo, and the envelope is sampled against
	/// that length rather than the free one.
	#[test]
	fn a_synced_stage_follows_the_tempo() {
		let mut adsr = adsr();
		adsr.attack_time = 10.0;
		adsr.attack = SyncedTime::synced(1.0);
		adsr.update_times(&infos_with_tempo(120.0));

		// A beat at 120 bpm is half a second.
		assert!((adsr.times.attack - 500.0).abs() < 1e-3);

		// 250 ms in is half way through the attack, not past it.
		assert!((adsr.sample_time(250.0) - 0.5).abs() < 1e-3);
		// The free value is over ten times shorter, so an unsynced envelope would be well
		// past its attack by now and sitting at the sustain level.
		adsr.attack = SyncedTime::default();
		adsr.update_times(&infos_with_tempo(120.0));
		assert_eq!(adsr.sample_time(250.0), adsr.sustain_level);

		// Half the tempo, twice the length.
		adsr.attack = SyncedTime::synced(1.0);
		adsr.update_times(&infos_with_tempo(60.0));
		assert!((adsr.times.attack - 1000.0).abs() < 1e-3);
		assert!((adsr.sample_time(250.0) - 0.25).abs() < 1e-3);
	}

	/// The envelope is read from the lengths, so a synced release ends the note where the
	/// tempo says rather than where the milliseconds did.
	#[test]
	fn a_synced_release_ends_the_note_on_the_beat() {
		let mut adsr = adsr();
		adsr.release_time = 100.0;
		adsr.release = SyncedTime::synced(0.5);
		adsr.update_times(&infos_with_tempo(120.0));

		// Half a beat at 120 bpm is 250 ms.
		assert!((adsr.times.release - 250.0).abs() < 1e-3);
		assert!(adsr.sample_release_time(249.0).is_some());
		assert!(adsr.sample_release_time(251.0).is_none());
	}

	/// The envelope can be built from a stopped transport too: the times are lengths, so the
	/// tempo still says how long a stage is even when the position is not moving.
	#[test]
	fn a_synced_stage_works_with_the_transport_stopped() {
		let mut adsr = adsr();
		adsr.attack = SyncedTime::synced(1.0);

		let infos = ProcessInfos {
			playing: false,
			..infos_with_tempo(120.0)
		};
		adsr.update_times(&infos);
		assert!((adsr.times.attack - 500.0).abs() < 1e-3);
	}

	/// The pan moves the voice, and the random spread widens it around wherever it is.
	#[test]
	fn pan_places_the_voice_and_random_pan_spreads_around_it() {
		let mut adsr = adsr();
		adsr.random_pan = 0.0;

		assert_eq!(adsr.pan_position(0.0), 0.5);
		adsr.pan = -1.0;
		assert_eq!(adsr.pan_position(0.0), 0.0, "hard left");
		adsr.pan = 1.0;
		assert_eq!(adsr.pan_position(0.0), 1.0, "hard right");

		// A spread of one reaches both edges from the centre...
		adsr.pan = 0.0;
		adsr.random_pan = 1.0;
		assert_eq!(adsr.pan_position(-1.0), 0.0);
		assert_eq!(adsr.pan_position(1.0), 1.0);

		// ...and from a side it stays inside the field instead of wrapping.
		adsr.pan = 1.0;
		assert_eq!(adsr.pan_position(1.0), 1.0);
		assert_eq!(adsr.pan_position(-1.0), 0.5);
	}
	/// A process context that hands the same event list to the one sample it drives.
	struct OneSample {
		infos: ProcessInfos,
		queue: Vec<crate::NoteEvent>,
	}

	impl ProcessContext for OneSample {
		fn infos(&self) -> &ProcessInfos {
			&self.infos
		}

		fn next_event(&mut self) -> Option<crate::NoteEvent> {
			None
		}

		fn send_event(&mut self, _: crate::NoteEvent) {}

		fn clear_events(&mut self) {
			self.queue.clear();
		}

		fn events(&self) -> &[crate::NoteEvent] {
			&self.queue
		}
	}

	fn note_on(note: usize) -> crate::NoteEvent {
		crate::NoteEvent::NoteOn {
			channel: 0,
			note,
			velocity: 1.0,
		}
	}

	/// Renders `samples` samples, delivering `events` on the first one.
	fn render(adsr: &mut TestAdsr, samples: usize, events: &[crate::NoteEvent]) -> Vec<f32> {
		let mut output = Vec::with_capacity(samples);

		for index in 0..samples {
			let mut context: Box<dyn ProcessContext> = Box::new(OneSample {
				infos: infos_with_tempo(120.0),
				queue: if index == 0 {
					events.to_vec()
				} else {
					Vec::new()
				},
			});

			output.push(adsr.generate(&mut context)[0]);
		}

		output
	}

	/// How sharp the sharpest corner of a signal is, relative to its peak.
	///
	/// A smooth sine of a few hundred hertz has a second difference of about
	/// (2 pi f / rate)^2 — a hundredth of the peak. A phase jump is a corner an
	/// order of magnitude sharper, so this one number separates the two.
	fn discontinuity(samples: &[f32]) -> f32 {
		let peak = samples
			.iter()
			.fold(0.0_f32, |peak, sample| peak.max(sample.abs()))
			.max(1e-6);
		let mut worst = 0.0_f32;

		for window in samples.windows(3) {
			let second = (window[2] - 2.0 * window[1] + window[0]).abs();
			worst = worst.max(second);
		}

		worst / peak
	}

	/// The frequency a window of a sine spends its zero crossings on.
	fn frequency_of(window: &[f32]) -> f32 {
		let crossings = window
			.windows(2)
			.filter(|pair| pair[0] <= 0.0 && pair[1] > 0.0)
			.count();

		crossings as f32 * 48_000.0 / window.len() as f32
	}

	/// Playing A4 and then A5, with a glide of 50 ms: the second note has to walk
	/// from 440 Hz to 880 Hz instead of jumping, and land exactly on it.
	#[test]
	fn a_glide_slides_from_the_previous_note() {
		let mut adsr = adsr();
		adsr.glide_time = 50.0;
		adsr.attack_time = 1.0;
		adsr.random_phase = 0.0;

		let first = render(&mut adsr, 3_000, &[note_on(57)]);
		// A 25 ms window is eleven cycles of A4, so the count is within a few Hz.
		assert!((frequency_of(&first[1_200..2_400]) - 440.0).abs() < 40.0);

		// The note before is stopped, so only the new one is in the mix; the glide
		// still starts from where it was.
		render(&mut adsr, 1, &[crate::NoteEvent::ImmediateStop]);

		let after = render(&mut adsr, 3_600, &[note_on(69)]);
		let early = frequency_of(&after[0..600]);
		let late = frequency_of(&after[2_400..3_600]);
		println!(
			"glide: early {early} mid {} late {late}",
			frequency_of(&after[600..1_200])
		);

		assert!(
			early > 440.0 && early < 600.0,
			"the first 12 ms are still near A4: {early}"
		);
		assert!((late - 880.0).abs() < 40.0, "50 ms in, it is A5: {late}");
	}

	/// Glide zero is the old behaviour: the second note starts at its own pitch.
	#[test]
	fn a_zero_glide_starts_at_the_new_note() {
		let mut adsr = adsr();
		adsr.glide_time = 0.0;
		adsr.attack_time = 1.0;
		adsr.random_phase = 0.0;

		render(&mut adsr, 200, &[note_on(57)]);
		render(&mut adsr, 1, &[crate::NoteEvent::ImmediateStop]);

		let after = render(&mut adsr, 4_800, &[note_on(69)]);

		// Two 50 ms windows: 44 cycles each, so the count lands within 25 Hz.
		assert!((frequency_of(&after[0..2_400]) - 880.0).abs() < 25.0);
		assert!((frequency_of(&after[2_400..4_800]) - 880.0).abs() < 25.0);
	}

	/// A glide must not jump the phase: the absolute read that corrects f32 drift
	/// assumes the pitch has been what it is now since the note started, which is
	/// false while the pitch is sliding.
	#[test]
	fn a_glide_keeps_the_phase_continuous() {
		let mut adsr = adsr();
		adsr.glide_time = 50.0;
		adsr.attack_time = 1.0;
		adsr.random_phase = 0.0;

		render(&mut adsr, 200, &[note_on(57)]);
		render(&mut adsr, 1, &[crate::NoteEvent::ImmediateStop]);

		let after = render(&mut adsr, 3_600, &[note_on(69)]);

		assert!(
			discontinuity(&after) < 0.05,
			"the slide has a corner in it: {}",
			discontinuity(&after)
		);
	}

	/// A short glide ends inside the drift-correction window. The resync that
	/// follows has to know about the phase the slide travelled, or it jumps.
	#[test]
	fn a_short_glide_keeps_the_phase_too() {
		for glide in [1.0_f32, 2.0, 3.0, 5.0, 8.0] {
			let mut adsr = adsr();
			adsr.glide_time = glide;
			adsr.attack_time = 1.0;
			adsr.random_phase = 0.0;

			render(&mut adsr, 200, &[note_on(57)]);
			render(&mut adsr, 1, &[crate::NoteEvent::ImmediateStop]);

			let after = render(&mut adsr, 1_500, &[note_on(69)]);

			assert!(
				discontinuity(&after) < 0.05,
				"a glide of {glide} ms has a corner in it: {}",
				discontinuity(&after)
			);
		}
	}

	/// A caller that builds a fresh voice per note — the patcher — primes it with
	/// the note the slide starts from, and gets the same glide.
	#[test]
	fn a_primed_voice_glides_from_the_note_it_was_told() {
		let mut adsr = adsr();
		adsr.glide_time = 50.0;
		adsr.attack_time = 1.0;
		adsr.random_phase = 0.0;

		adsr.glide_from(0, 57);

		let after = render(&mut adsr, 3_600, &[note_on(69)]);

		assert!(frequency_of(&after[0..600]) < 600.0);
		assert!((frequency_of(&after[2_400..3_600]) - 880.0).abs() < 40.0);
	}
}
