//! ADSR (Attack, Decay, Sustain, Release) envelope generator

use std::collections::HashMap;

use i_am_parameters_derive::Parameters;

#[cfg(feature = "real_time_demo")]
use crate::tools::ring_buffer::RingBuffer;
use crate::{generators::Note, prelude::Oscillator, Generator, NoteEvent};

/// A tuning system that converts MIDI note numbers to frequencies in Hz
pub trait Tuning {
	/// Returns frequency in Hz for given MIDI note number
	/// 
	/// Note: We allow note number to be float to allow for microtonal tuning
	fn get_frequency(&self, note: f32) -> f32;
}

/// The 12-tone equal temperament tuning system
pub struct EqualTemperament;

impl Tuning for EqualTemperament {
	fn get_frequency(&self, note: f32) -> f32 {
		const A4: f32 = 440.0;
		const A4_MIDI: f32 = 57.0;

		let semitones = (note - A4_MIDI) / 12.0;
		A4 * 2.0f32.powf(semitones)
	}
}

struct PlayingNote<const CHANNELS: usize> {
	note: Note,
	count: usize,
	release: Option<(f32, usize)>,
	phase_start: Vec<[f32; CHANNELS]>,
}

/// A warper to play an oscillator with ADSR envelope. 
/// 
/// Adsr supports Midi Events, but cannot distinguish the note'channel.
/// 
/// Actually this is AHDSR.
/// 
/// The Bend function of Adsr is $\frac{\exp(bend_factor * time) - 1}{\exp(bend_factor) - 1}$
#[derive(Parameters)]
#[default_float_range(min = 0.0, max = 4000.0)]
pub struct Adsr<
	Osc: Oscillator<CHANNELS>,
	TuningSys: Tuning,
	const CHANNELS: usize
> {
	#[sub_param]
	oscillator: Osc,
	#[skip]
	tuning_sys: TuningSys,
	#[skip]
	note_playing: HashMap<u8, PlayingNote<CHANNELS>>,
	/// The sample rate of the audio processing system
	/// 
	/// Saves in Hz
	#[skip]
	pub sample_rate: usize,
	/// The delay time, saves in milliseconds
	pub delay_time: f32,
	/// The attack time, saves in milliseconds
	pub attack_time: f32,
	/// The attack bend, 0.0 for no bend
	#[range(min = -10.0, max = 10.0)]
	pub attack_bend: f32,
	/// The hold time, saves in milliseconds
	pub hold_time: f32,
	/// The decay time, saves in milliseconds
	pub decay_time: f32,
	/// The decay bend, 0.0 for no bend
	#[range(min = -10.0, max = 10.0)]
	pub decay_bend: f32,
	/// The sustain level, saves in linear scale
	pub sustain_level: f32,
	/// The release time, saves in milliseconds
	pub release_time: f32,
	/// The release bend, 0.0 for no bend
	#[range(min = -10.0, max = 10.0)]
	pub release_bend: f32,
	/// saves in linear scale
	#[range(min = 0.01, max = 4.0)]
	#[logarithmic]
	pub gain: f32,
	/// The number of unison, 1 for no unison
	#[range(min = 1, max = 32)]
	pub unisons: usize,
	/// Maxium detune of each unison, the unit is note
	pub unison_detune: f32,
	/// Bend of each unison, 0.0 for no bend
	#[range(min = -10.0, max = 10.0)]
	pub unison_bend: f32,
	/// The blend of each unison, 0.0 for no blend
	/// 
	/// This will lower the amplitude of "side" unison.
	#[range(min = 0.0, max = 1.0)]
	pub unison_blend: f32,
	/// The random phase range of each unison, 0.0 for no random phase
	#[range(min = 0.0, max = 1.0)]
	pub random_phase: f32,
	#[range(min = 0.25, max = 4.0)]
	#[logarithmic]
	/// The pitch factor of the oscillator, used for pitch shifting.
	pub pitch_factor: f32,

	#[cfg(feature = "real_time_demo")]
	#[skip]
	history: [RingBuffer<f32>; CHANNELS],
}

impl<
	Osc: Oscillator<CHANNELS>,
	TuningSys: Tuning,
	const CHANNELS: usize
> Adsr<Osc, TuningSys, CHANNELS> {
	/// Create a new Adsr with given oscillator and tuning system
	/// 
	/// # Panics
	/// 
	/// 1. If `sample_rate` is 0.
	/// 2. If `CHANNELS` is 0.
	pub fn new(
		oscillator: Osc,
		tuning_sys: TuningSys,
		sample_rate: usize,
	) -> Self {
		assert!(CHANNELS > 0);
		assert!(sample_rate > 0);

		Self {
			oscillator,
			tuning_sys,
			note_playing: HashMap::new(),
			sample_rate,
			delay_time: 0.0,
			attack_time: 10.0,
			attack_bend: 0.0,
			hold_time: 100.0,
			decay_time: 100.0,
			decay_bend: 0.0,
			sustain_level: 0.5,
			release_time: 100.0,
			release_bend: 0.0,
			gain: 0.8,
			unisons: 1,
			unison_detune: 2.0,
			unison_bend: 0.0,
			unison_blend: 0.0,
			random_phase: 1.0,
			pitch_factor: 1.0,

			#[cfg(feature = "real_time_demo")]
			history: core::array::from_fn(|_| RingBuffer::new(4096)),
		}
	}

	fn sample_count(&self, count: usize) -> f32 {
		let time = count as f32 / self.sample_rate as f32 * 1000.0;
		self.sample_time(time)
	}

	fn sample_time(&self, mut time: f32) -> f32 {
		if time < self.delay_time {
			return 0.0;
		}
		time -= self.delay_time;

		if time < self.attack_time {
			let t = time / self.attack_time;
			// let a = bend(self.attack_time, self.attack_bend);
			return bend(t, self.attack_bend);
		}
		time -= self.attack_time;

		if time < self.hold_time {
			return 1.0;
		}
		time -= self.hold_time;

		if time < self.decay_time {
			let t = time / self.decay_time;
			let a = bend(t, self.decay_bend);
			let v = 1.0 + (self.sustain_level - 1.0) * a;
			return v;
		}
		
		self.sustain_level
	}

	fn sample_release_time(&self, time: f32) -> Option<f32> {
		if time >= self.release_time {
			return None;
		}
		let t = time / self.release_time;
		let a = bend(t, self.release_bend);
		Some(1.0 - a)
	}
}

impl<
	Osc: Oscillator<CHANNELS> + Send + Sync,
	TuningSys: Tuning + Send + Sync,
	const CHANNELS: usize
> Generator<CHANNELS> for Adsr<Osc, TuningSys, CHANNELS> {
	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"ADSRed Generator"
	}

	fn generate(&mut self, process_context: &mut Box<dyn crate::ProcessContext>) -> [f32; CHANNELS] {
		for event in process_context.events() {
			match event {
				NoteEvent::NoteOn {
					note,
					velocity,
					// channel,
					..
				} => {
					let note = Note { 
						// channel: *channel, 
						note: *note, 
						velocity: *velocity 
					};
					let phases = (0..self.unisons).map(|_| {
						if self.random_phase != 0.0 {
							let mut output = [0.0; CHANNELS];
							if self.random_phase.abs() != 0.0 {
								for output in &mut output {
									let random_phase = rand::random_range(0.0..=self.random_phase.abs());
									*output = random_phase;
								}
							}
							output
						}else {
							[0.0; CHANNELS]
						}
					}).collect();
					let playing_note = PlayingNote {
						note,
						count: 0,
						release: None,
						phase_start: phases
					};
					self.note_playing.insert(playing_note.note.note, playing_note);
				}
				NoteEvent::NoteOff {
					note,
					..
				} => {
					if let Some(mut playing_note) = self.note_playing.remove(note) {
						let sample = self.sample_count(playing_note.count);
						playing_note.release = Some((sample, playing_note.count));
						self.note_playing.insert(*note, playing_note);
					}
				}
				NoteEvent::Stop {
					note,
					..
				} => {
					self.note_playing.remove(note);
				}
				NoteEvent::ImmediateStop => {
					self.note_playing.clear();
				}
				_ => {},
			}
		}

		let mut output = [0.0; CHANNELS];

		let sample_rate = self.sample_rate as f32;
		let mut note_playing = std::mem::take(&mut self.note_playing);
		note_playing.retain(|note, playing_note| {
			let unisons = playing_note.phase_start.len();
			
			let gain = if let Some((release, release_count)) = playing_note.release {
				let time = (playing_note.count - release_count) as f32 / sample_rate * 1000.0;
				let Some(gain) = self.sample_release_time(time) else {
					return false;
				};
				gain * playing_note.note.velocity * release
			}else {
				let time = playing_note.count as f32 / sample_rate * 1000.0;
				self.sample_time(time) * playing_note.note.velocity
			};

			let time = playing_note.count as f32 / sample_rate;
			let mut output_samples = [0.0; CHANNELS];
			let mid_point = (unisons - 1) as f32 / 2.0;
			// let channels_f32 = CHANNELS as f32 - 1.0;
			for i in 0..unisons {
				let index = if mid_point == 0.0 {
					0.0
				}else {
					(i as f32 - mid_point) / mid_point
				};
				let blend = 1.0 - index.abs() * self.unison_blend;
				let index = if index >= 0.0 {
					bend(index, self.unison_bend)
				}else {
					- bend(index.abs(), self.unison_bend)
				};
				let detune_factor = self.unison_detune * index;
				let frequency = self.tuning_sys.get_frequency(*note as f32 + detune_factor) * self.pitch_factor;
				let samples = self.oscillator.play_at(
					frequency, 
					time,
					playing_note.phase_start[i],
				);
				for (i, output_samples) in output_samples.iter_mut().enumerate() {
					// let pan_gain = i as f32 / channels_f32 * index * 2.0 - index + 1.0;

					*output_samples += samples[i] * gain / unisons as f32 * blend;
				}
			}
			for i in 0..CHANNELS {
				output[i] += output_samples[i] * gain;
			}

			playing_note.count += 1;

			true
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
		use egui::*;
		use egui::emath::RectTransform;
		use crate::tools::ui_tools::draw_envelope;
		use crate::tools::ui_tools::gain_ui;

		CollapsingHeader::new("Oscillator Settings")
			.id_source(format!("{}_oscillator_settings", id_prefix))
			.show(ui, |ui| 
		{
			self.oscillator.demo_ui(ui, format!("{}_oscillator", id_prefix));
		});

		ui.horizontal(|ui| {
			Frame::canvas(ui.style()).show(ui, |ui| {
				const POINTS: usize = 128;
				let (_, rect) = ui.allocate_space(Vec2::splat(100.0));
				let total_time = self.delay_time + self.attack_time + self.hold_time + self.decay_time + self.release_time;
				let none_release_time = self.delay_time + self.attack_time + self.hold_time + self.decay_time;
				let to_screen = RectTransform::from_to(
					Rect::from_x_y_ranges(0.0..=1.0, -1.0..=0.0), 
					rect
				);

				let mut positions = vec![Pos2::ZERO; POINTS];

				for (i, position) in positions.iter_mut().enumerate() {
					let time = i as f32 / POINTS as f32 * total_time;
					let sample = if time < none_release_time {
						self.sample_time(time)
					}else {
						self.sample_release_time(time - none_release_time).unwrap_or(0.0) * self.sustain_level
					};
					*position = pos2(time / total_time, - sample);
					*position = to_screen * *position;
				}

				ui.painter().extend([
					Shape::line(positions, (3.0, Color32::WHITE))
				]);
			});

			Frame::canvas(ui.style()).show(ui, |ui| {				
				let (_, rect) = ui.allocate_space(Vec2::splat(100.0));
				let to_screen = RectTransform::from_to(
					Rect::from_x_y_ranges(-1.0..=1.0, -1.0..=0.0), 
					rect
				);

				let mid_point = (self.unisons - 1) as f32 / 2.0;
				let shapes = (0..self.unisons).map(|i| {
						let index = if mid_point == 0.0 {
						0.0
					}else {
						(i as f32 - mid_point) / mid_point
					};
					let blend = 1.0 - index.abs() * self.unison_blend;
					let index = if index >= 0.0 {
						bend(index, self.unison_bend)
					}else {
						- bend(index.abs(), self.unison_bend)
					};
					let detune_factor = self.unison_detune * index / 2.0;
					let position_1 = to_screen * pos2(detune_factor, - blend);
					let position_2 = to_screen * pos2(detune_factor, 0.0);
					Shape::line(vec![position_1, position_2], (3.0, Color32::WHITE))
				});

				ui.painter().extend(shapes);
			});

			
			let history = self.history.iter().collect::<Vec<_>>();
			draw_envelope(ui, &history, true);
		});

		ui.separator();

		Grid::new(format!("{}_adsr_grid", id_prefix)).num_columns(6).show(ui, |ui| {
			ui.label("Delay time");
			ui.add(Slider::new(&mut self.delay_time, 0.0..=1000.0).suffix("ms"));
			ui.label("Attack time");
			ui.add(Slider::new(&mut self.attack_time, 0.0..=1000.0).suffix("ms"));
			ui.label("Hold Time");
			ui.add(Slider::new(&mut self.hold_time, 0.0..=1000.0).suffix("ms"));
			ui.end_row();
			
			ui.label("Decay Time");
			ui.add(Slider::new(&mut self.decay_time, 0.0..=1000.0).suffix("ms"));
			ui.label("Sustain Level");
			gain_ui(ui, &mut self.sustain_level, Some("".to_string()), true);
			ui.label("Release Time");
			ui.add(Slider::new(&mut self.release_time, 0.0..=1000.0).suffix("ms"));
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
			ui.label("Gain");
			gain_ui(ui, &mut self.gain, Some("".to_string()), false);
			ui.end_row();

			ui.label("Pitch Factor");
			ui.add(Slider::new(&mut self.pitch_factor, 0.25..=4.0).logarithmic(true));
		});
	}
}

pub(crate) fn bend(t: f32, bend: f32) -> f32 {
	if bend == 0.0 {
		t
	}else {
		((t * bend).exp() - 1.0) / (bend.exp() - 1.0)
	}
}