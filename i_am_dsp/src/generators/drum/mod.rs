//! A drum voice, and the kick that proves it.
//!
//! M0 plays note 36. The body is a bank of decaying partials: a sine is just a
//! bank with one entry, and the inharmonic ratios below are membrane modes. The
//! click is a much shorter bank plus a band of noise, started with random phases
//! so it reads as material rather than a tone. Pitch and amplitude are the same
//! breakpoint curves the editor will draw.

pub mod curve;
pub mod sources;

use std::sync::{Arc, Mutex};

use i_am_dsp_derive::Parameters;

use crate::effects::convolver::FftConvolver;
use crate::generators::sampler_voice::SampleData;
use crate::{Generator, NoteEvent};
use curve::{Curve, Point};
use sources::{Band, NoiseBand, Partial, Rng, Running};

/// The note the kick answers to, in the host's numbering.
pub const KICK: u8 = 36;
/// The note the snare answers to.
pub const SNARE: u8 = 38;
/// The closed hi-hat's note.
pub const HAT_CLOSED: u8 = 42;
/// The pedal hi-hat's note.
pub const HAT_PEDAL: u8 = 44;
/// The open hi-hat's note.
pub const HAT_OPEN: u8 = 46;
/// The clap's note.
pub const CLAP: u8 = 39;
/// The lowest tom's note.
pub const TOM_LOW: u8 = 41;
/// The highest tom's note.
pub const TOM_HIGH: u8 = 48;
/// The crash's note.
pub const CRASH: u8 = 49;
/// The ride's note.
pub const RIDE: u8 = 51;

/// How many hits may ring at once.
const MAX_VOICES: usize = 16;

mod voice;
pub use voice::*;

/// A kick drum synthesised from partials, a click and noise.
#[derive(Parameters)]
pub struct DrumInstrument {
	#[skip]
	playing: Vec<Voice>,
	#[skip]
	rate: f32,
	#[skip]
	rng: Rng,
	/// The pitch curve, saved with the plug-in state. The voice reads the
	/// shared copy, which the editor keeps in step while the window is open.
	#[serde]
	#[id(display = "Pitch Curve")]
	pub pitch_curve: Curve,
	#[skip]
	shared: Arc<Mutex<Curve>>,
	/// The fundamental of the kick, in Hz.
	#[range(min = 30.0, max = 120.0)]
	#[default(52.0)]
	#[unit(hertz)]
	#[id(display = "Tune")]
	#[layout(group = "Kick", order = 0, widget = knob, hero = "kick")]
	pub tune: f32,
	/// How long the body rings, in seconds.
	#[id(display = "Decay")]
	#[serde]
	pub decay: f32,
	/// The click level.
	#[range(min = 0.0, max = 1.0)]
	#[default(0.5)]
	#[id(display = "Click")]
	#[layout(group = "Kick", order = 3, widget = knob)]
	pub click: f32,
	/// Where the click's material sits, in Hz.
	#[range(min = 300.0, max = 6000.0)]
	#[default(1800.0)]
	#[unit(hertz)]
	#[id(display = "Click Tone")]
	#[layout(group = "Kick", order = 4, widget = knob)]
	pub click_tone: f32,
	/// The drive into the output stage.
	#[range(min = 1.0, max = 8.0)]
	#[default(1.5)]
	#[id(display = "Drive")]
	#[layout(group = "Kit", order = 4, widget = knob)]
	pub drive: f32,
	/// The output level.
	#[range(min = 0.0, max = 1.0)]
	#[default(0.8)]
	#[id(display = "Gain")]
	#[layout(group = "Kit", order = 5, widget = knob)]
	pub gain: f32,
	/// The fundamental of the snare's shell, in Hz.
	#[range(min = 120.0, max = 400.0)]
	#[default(180.0)]
	#[unit(hertz)]
	#[id(display = "Snare Tune")]
	#[layout(group = "Snare", order = 0, widget = knob, hero = "snare")]
	pub snare_tune: f32,
	/// How long the shell rings, in seconds.
	#[id(display = "Snare Decay")]
	#[serde]
	pub snare_decay: f32,
	/// How much air and wire the hit throws off.
	#[range(min = 0.0, max = 1.0)]
	#[default(0.5)]
	#[id(display = "Wires")]
	#[layout(group = "Snare", order = 2, widget = knob)]
	pub snare_wires: f32,
	/// The snare's stick click.
	#[range(min = 0.0, max = 1.0)]
	#[default(0.4)]
	#[id(display = "Snare Click")]
	#[layout(group = "Snare", order = 3, widget = knob)]
	pub snare_click: f32,
	/// The metal cluster's tone, as a multiplier on the six frequencies.
	#[range(min = 0.4, max = 2.5)]
	#[default(1.0)]
	#[id(display = "Hat Tone")]
	#[layout(group = "Hat", order = 0, widget = knob, hero = "amp_hat")]
	pub hat_tune: f32,
	/// How long a closed hat rings, in seconds.
	#[id(display = "Closed")]
	#[serde]
	pub hat_decay: f32,
	/// How long an open hat rings, in seconds.
	#[range(min = 0.1, max = 2.0)]
	#[default(0.45)]
	#[id(display = "Open")]
	#[layout(group = "Hat", order = 2, widget = knob)]
	pub hat_open: f32,
	/// The highpass that makes the cluster metal rather than a chord.
	#[range(min = 1_500.0, max = 12_000.0)]
	#[default(6_500.0)]
	#[unit(hertz)]
	#[id(display = "Hat Cutoff")]
	#[layout(group = "Hat", order = 3, widget = knob)]
	pub hat_cutoff: f32,
	/// The hat's level.
	#[id(display = "Hat Level")]
	#[serde]
	pub hat_gain: f32,
	/// Where the clap's noise sits, in Hz.
	#[range(min = 600.0, max = 5_000.0)]
	#[default(1_600.0)]
	#[unit(hertz)]
	#[id(display = "Clap Tone")]
	#[layout(group = "Clap", order = 0, widget = knob, hero = "amp_clap")]
	pub clap_tone: f32,
	/// How long the clap's tail lasts, in seconds.
	#[id(display = "Clap Decay")]
	#[serde]
	pub clap_decay: f32,
	/// How long the toms ring, in seconds.
	#[id(display = "Tom Decay")]
	#[serde]
	pub tom_decay: f32,
	/// A multiplier on the toms' shell pitch.
	#[range(min = 0.5, max = 2.0)]
	#[default(1.0)]
	#[id(display = "Tom Tune")]
	#[layout(group = "Tom", order = 1, widget = knob, hero = "amp_tom")]
	pub tom_tune: f32,
	/// How long the cymbals ring, in seconds.
	#[id(display = "Cymbal Decay")]
	#[serde]
	pub cymbal_decay: f32,
	/// The cymbals' level.
	#[id(display = "Cymbal Level")]
	#[serde]
	pub cymbal_gain: f32,
	/// How much each hit differs from the last, in level and pitch.
	#[range(min = 0.0, max = 1.0)]
	#[default(0.15)]
	#[id(display = "Humanize")]
	#[layout(group = "Kit", order = 0, widget = knob, hero = "kit")]
	pub humanize: f32,
	/// A one-shot that is layered over every hit, saved with the state.
	#[serde]
	#[id(display = "Click Sample")]
	pub click_sample: Option<SampleData>,
	/// The imported click's level.
	#[range(min = 0.0, max = 1.0)]
	#[default(0.7)]
	#[id(display = "Click Level")]
	#[layout(group = "Kit", order = 3, widget = knob)]
	pub click_sample_level: f32,
	/// How much a harder hit opens the drum up instead of only getting louder.
	#[range(min = 0.0, max = 1.0)]
	#[default(0.35)]
	#[id(display = "Velocity Bright")]
	#[layout(group = "Kit", order = 1, widget = knob)]
	pub velocity_brightness: f32,
	/// The kick's level in the kit.
	#[range(min = 0.0, max = 2.0)]
	#[default(1.0)]
	#[id(display = "Kick Level")]
	#[layout(group = "Kick", order = 9, widget = knob)]
	pub level_kick: f32,
	/// The snare's level in the kit.
	#[range(min = 0.0, max = 2.0)]
	#[default(1.0)]
	#[id(display = "Snare Level")]
	#[layout(group = "Snare", order = 1, widget = knob)]
	pub level_snare: f32,
	/// The hat's level in the kit.
	#[range(min = 0.0, max = 2.0)]
	#[default(1.0)]
	#[id(display = "Hat Level")]
	#[layout(group = "Hat", order = 2, widget = knob)]
	pub level_hat: f32,
	/// The clap's level in the kit.
	#[range(min = 0.0, max = 2.0)]
	#[default(1.0)]
	#[id(display = "Clap Level")]
	#[layout(group = "Clap", order = 3, widget = knob)]
	pub level_clap: f32,
	/// The toms' level in the kit.
	#[range(min = 0.0, max = 2.0)]
	#[default(1.0)]
	#[id(display = "Tom Level")]
	#[layout(group = "Tom", order = 4, widget = knob)]
	pub level_tom: f32,
	/// The cymbals' level in the kit.
	#[range(min = 0.0, max = 2.0)]
	#[default(1.0)]
	#[id(display = "Cymbal Level")]
	#[layout(group = "Cymbal", order = 5, widget = knob)]
	pub level_cymbal: f32,
	/// The kick's one-shot level, when a sample replaces the synthesis.
	#[range(min = 0.0, max = 2.0)]
	#[default(1.0)]
	#[id(display = "Kick Shot")]
	#[layout(group = "Samples", order = 0, widget = knob, hero = "samples")]
	pub sample_level_kick: f32,
	/// The snare's one-shot level.
	#[range(min = 0.0, max = 2.0)]
	#[default(1.0)]
	#[id(display = "Snare Shot")]
	#[layout(group = "Samples", order = 1, widget = knob)]
	pub sample_level_snare: f32,
	/// The hat's one-shot level.
	#[range(min = 0.0, max = 2.0)]
	#[default(1.0)]
	#[id(display = "Hat Shot")]
	#[layout(group = "Samples", order = 2, widget = knob)]
	pub sample_level_hat: f32,
	/// The clap's one-shot level.
	#[range(min = 0.0, max = 2.0)]
	#[default(1.0)]
	#[id(display = "Clap Shot")]
	#[layout(group = "Samples", order = 3, widget = knob)]
	pub sample_level_clap: f32,
	/// The toms' one-shot level.
	#[range(min = 0.0, max = 2.0)]
	#[default(1.0)]
	#[id(display = "Tom Shot")]
	#[layout(group = "Samples", order = 4, widget = knob)]
	pub sample_level_tom: f32,
	/// The cymbals' one-shot level.
	#[range(min = 0.0, max = 2.0)]
	#[default(1.0)]
	#[id(display = "Cymbal Shot")]
	#[layout(group = "Samples", order = 5, widget = knob)]
	pub sample_level_cymbal: f32,
	/// Where the kick sits in the stereo field.
	#[range(min = -1.0, max = 1.0)]
	#[default(0.0)]
	#[id(display = "Kick Pan")]
	#[layout(group = "Kick", order = 10, widget = knob)]
	pub pan_kick: f32,
	/// Where the snare sits.
	#[range(min = -1.0, max = 1.0)]
	#[default(0.0)]
	#[id(display = "Snare Pan")]
	#[layout(group = "Snare", order = 10, widget = knob)]
	pub pan_snare: f32,
	/// Where the hats sit.
	#[range(min = -1.0, max = 1.0)]
	#[default(0.0)]
	#[id(display = "Hat Pan")]
	#[layout(group = "Hat", order = 11, widget = knob)]
	pub pan_hat: f32,
	/// Where the clap sits.
	#[range(min = -1.0, max = 1.0)]
	#[default(0.0)]
	#[id(display = "Clap Pan")]
	#[layout(group = "Clap", order = 10, widget = knob)]
	pub pan_clap: f32,
	/// Where the toms sit.
	#[range(min = -1.0, max = 1.0)]
	#[default(0.0)]
	#[id(display = "Tom Pan")]
	#[layout(group = "Tom", order = 10, widget = knob)]
	pub pan_tom: f32,
	/// Where the cymbals sit.
	#[range(min = -1.0, max = 1.0)]
	#[default(0.0)]
	#[id(display = "Cymbal Pan")]
	#[layout(group = "Cymbal", order = 10, widget = knob)]
	pub pan_cymbal: f32,
	/// How wide the decorrelated noise parts sit.
	#[range(min = 0.0, max = 1.0)]
	#[default(0.35)]
	#[id(display = "Width")]
	#[layout(group = "Kit", order = 6, widget = knob)]
	pub width: f32,
	/// How many milliseconds of a hat stay dry before the IR takes over.
	#[range(min = 0.0, max = 10.0)]
	#[default(4.0)]
	#[id(display = "Hat Direct")]
	#[layout(group = "Hat", order = 12, widget = knob)]
	pub hat_direct: f32,
	/// Which kit the instrument starts from. Choosing one writes the knobs.
	#[range(min = 0, max = 1)]
	#[choices("Studio", "808")]
	#[default(0)]
	#[id(display = "Kit")]
	#[layout(group = "Kit", order = 2, widget = choice)]
	pub kit: i32,
	/// The brightness this hit was started with: 1 is the nominal material.
	#[skip]
	brightness: f32,
	/// The cymbal tail, convolved with a real crash's room.
	#[skip]
	tail_convolver: Option<Box<FftConvolver<2>>>,
	/// The hat's colour, convolved with a real hat's short response.
	#[skip]
	hat_convolver: Option<Box<FftConvolver<2>>>,
	/// The send of the last sample into that tail.
	#[skip]
	tail_send: f32,
	/// How many samples the tail still has to process before it goes quiet.
	#[skip]
	tail_remaining: usize,
	/// How many samples the hat's tail still has.
	#[skip]
	hat_remaining: usize,
	/// 0 while a hat's dry attack is still in front, then 1 for the wet.
	#[skip]
	hat_gate: f32,
	/// The page's click slot, when one is shared.
	#[skip]
	click_slot: Option<Arc<ClickSlot>>,
	/// The slot generation the instrument last took.
	#[skip]
	click_seen: u64,
	/// The six one-shot slots, when the page shares them.
	#[skip]
	samples: Option<Arc<Samples>>,
	/// The generation last taken from those slots.
	#[skip]
	samples_seen: u64,
	/// The samples currently adopted, one per drum.
	#[skip]
	loaded: [Option<SampleData>; 6],
	/// The amplitude shapes, saved with the state and hidden from the face.
	#[serde]
	pub amp_curves: Vec<Curve>,
	/// The snare's shell, wires and pitch, saved and hidden.
	#[serde]
	pub snare_curves: SnareCurves,
	/// The snare's shapes the voices read.
	#[skip]
	snare_shared: Arc<Mutex<SnareCurves>>,
	/// The amplitude shapes the voices read.
	#[skip]
	amps: Arc<Mutex<[Curve; 6]>>,
	/// A fitted bank that replaces the cymbal's synthesis.
	#[skip]
	fitted: Option<DrumPreset>,
	/// The fit slot, when a host shares one.
	#[skip]
	fit_slot: Option<Arc<FitSlot>>,
	/// The fit generation last taken.
	#[skip]
	fit_seen: u64,
	/// How hard the cymbals' bank is phase modulated.
	#[range(min = 0.0, max = 1.0)]
	#[default(0.8)]
	#[id(display = "Cymbal FM")]
	#[layout(group = "Cymbal", order = 3, widget = knob, hero = "amp_cymbal")]
	pub cymbal_fm: f32,
}

/// A pitch curve that stays at one, for the drums whose note is their pitch.
fn flat_pitch() -> Curve {
	Curve::new(vec![
		Point {
			time: 0.0,
			value: 1.0,
			bend: 0.0,
		},
		Point {
			time: 4.0,
			value: 1.0,
			bend: 0.0,
		},
	])
}

/// The pitch fall a fresh kick starts with: a deep dive in the first 30 ms.
pub fn default_pitch() -> Curve {
	Curve::new(vec![
		Point {
			time: 0.0,
			value: 4.5,
			bend: -2.0,
		},
		Point {
			time: 0.03,
			value: 1.0,
			bend: 0.0,
		},
	])
}

impl Default for DrumInstrument {
	fn default() -> Self {
		Self {
			playing: Vec::new(),
			rate: 48_000.0,
			rng: Rng::default(),
			pitch_curve: default_pitch(),
			shared: Arc::new(Mutex::new(default_pitch())),
			tune: 52.0,
			decay: 0.45,
			click: 0.5,
			click_tone: 1800.0,
			drive: 1.5,
			gain: 0.8,
			snare_tune: 180.0,
			snare_decay: 0.18,
			snare_wires: 0.5,
			snare_click: 0.4,
			hat_tune: 1.0,
			hat_decay: 0.05,
			hat_open: 0.45,
			hat_cutoff: 6_500.0,
			hat_gain: 0.6,
			clap_tone: 1_600.0,
			clap_decay: 0.35,
			tom_decay: 0.5,
			tom_tune: 1.0,
			cymbal_decay: 2.5,
			cymbal_gain: 0.5,
			cymbal_fm: 0.8,
			humanize: 0.15,
			velocity_brightness: 0.35,
			kit: 0,
			sample_level_kick: 1.0,
			sample_level_snare: 1.0,
			sample_level_hat: 1.0,
			sample_level_clap: 1.0,
			sample_level_tom: 1.0,
			sample_level_cymbal: 1.0,
			click_slot: None,
			click_seen: 0,
			samples: None,
			samples_seen: 0,
			loaded: Default::default(),
			amp_curves: Vec::new(),
			amps: Arc::new(Mutex::new(default_amp_templates())),
			snare_curves: SnareCurves::default(),
			snare_shared: Arc::new(Mutex::new(SnareCurves::default())),
			fitted: None,
			fit_slot: None,
			fit_seen: 0,
			click_sample: None,
			click_sample_level: 0.7,
			level_kick: 1.0,
			level_snare: 1.0,
			level_hat: 1.0,
			level_clap: 1.0,
			level_tom: 1.0,
			level_cymbal: 1.0,
			pan_kick: 0.0,
			pan_snare: 0.0,
			pan_hat: 0.0,
			pan_clap: 0.0,
			pan_tom: 0.0,
			pan_cymbal: 0.0,
			width: 0.35,
			hat_direct: 4.0,
			brightness: 1.0,
			tail_convolver: None,
			hat_convolver: None,
			tail_send: 0.0,
			tail_remaining: 0,
			hat_remaining: 0,
			hat_gate: 1.0,
		}
	}
}

mod preset;
mod render;
pub use preset::*;

impl Generator<2> for DrumInstrument {
	fn generate(&mut self, context: &mut Box<dyn crate::ProcessContext>) -> [f32; 2] {
		let rate = context.infos().sample_rate as f32;
		if rate > 0.0 && (rate - self.rate).abs() > 0.5 {
			self.rate = rate;
			self.playing.clear();
		}

		self.adopt_samples();
		self.adopt_click();
		self.adopt_fit();

		let events = context.events().to_vec();
		context.clear_events();
		for event in &events {
			match event {
				NoteEvent::NoteOn { note, velocity, .. } => {
					self.start((*note).clamp(0, 127) as u8, *velocity);
				}
				NoteEvent::ImmediateStop => self.playing.clear(),
				_ => {}
			}
		}

		self.next_frame()
	}
}

#[cfg(test)]
mod tests;
