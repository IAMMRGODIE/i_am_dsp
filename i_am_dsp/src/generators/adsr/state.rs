use super::*;

/// The stage lengths an envelope runs at, in milliseconds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct EnvelopeTimes {
	pub(super) delay: f32,
	pub(super) attack: f32,
	pub(super) hold: f32,
	pub(super) decay: f32,
	pub(super) release: f32,
}

impl EnvelopeTimes {
	/// The lengths a fresh envelope starts with, matching the free parameters.
	pub(super) const DEFAULT: Self = Self {
		delay: 0.0,
		attack: 10.0,
		hold: 100.0,
		decay: 100.0,
		release: 100.0,
	};

	/// How long the note sounds before it is let go.
	#[cfg(feature = "real_time_demo")]
	pub(super) fn held(&self) -> f32 {
		self.delay + self.attack + self.hold + self.decay
	}

	/// How long the note sounds in total, release included.
	#[cfg(feature = "real_time_demo")]
	pub(super) fn total(&self) -> f32 {
		self.held() + self.release
	}
}

/// One unison's properties that the controls fix: the frequency its detune makes
/// of the note, how much of the unison blend it gets, and its pan gain per
/// channel.
#[derive(Clone, Copy)]
pub(crate) struct UnisonVoice<const CHANNELS: usize> {
	pub(super) frequency: f32,
	/// `blend * gains`: the unison blend and the pan do not move inside a block,
	/// so the mix reads one factor instead of multiplying three together.
	pub(super) weight: [f32; CHANNELS],
}

/// The control values the unison table was built from.
///
/// A unison's detune, blend and pan are the same on every sample until one of
/// these moves; comparing them is cheaper than recomputing a 2^x and a pan
/// curve per unison per sample.
#[derive(Clone, Copy, PartialEq)]
pub(crate) struct UnisonControls {
	pub(super) detune: f32,
	pub(super) bend: f32,
	pub(super) blend: f32,
	pub(super) pan: f32,
	pub(super) random_pan: f32,
}

/// A struct to hold the state of a playing note
pub struct PlayingNote<const CHANNELS: usize> {
	pub(crate) note: Note,
	pub(crate) channel: u8,
	pub(crate) count: usize,
	pub(crate) release: Option<(f32, usize)>,
	pub(crate) phase_start: Vec<[f32; CHANNELS]>,
	pub(crate) pan_factors: Vec<f32>,
	pub(crate) last_time: f32,
	pub(crate) last_phase: Vec<f32>,
	/// Each copy's frequency at the current pitch.
	pub(crate) frequencies: Vec<f32>,
	/// Each copy's phase step per sample, `frequency / rate`.
	pub(crate) steps: Vec<f32>,
	/// The pitch the copies were last computed for; NaN forces a recompute.
	pub(crate) pitch_last: f32,
	/// The frequency each unison was last tuned to, so a still pitch does not
	/// re-tune the oscillator every sample.
	pub(crate) pitch_applied: Vec<f32>,
	/// The per-unison table, rebuilt when any of the cached controls move.
	pub(crate) unison_voices: Vec<UnisonVoice<CHANNELS>>,
	pub(crate) unison_controls: Option<UnisonControls>,
	/// The current pitch factor of the oscillator, used for pitch shifting.
	pub current_pitch_factor: f32,
	/// How much the pitch factor moves per sample while a glide runs.
	pub(crate) glide_step: f32,
	/// Samples left in the glide; zero when the note sits at its own pitch.
	pub(crate) glide_left: u32,
}

pub(crate) type NoteProcessor<const CHANNELS: usize> =
	Box<dyn Fn(&mut [f32; CHANNELS], &PlayingNote<CHANNELS>, usize) + Send + Sync>;

impl<const CHANNELS: usize> PlayingNote<CHANNELS> {
	/// Get how long since the note has been playing in ms
	pub fn current_time(&self, sample_rate: usize) -> f32 {
		self.count as f32 / sample_rate as f32 * 1000.0
	}

	/// Get the note number of the playing note
	pub fn note(&self) -> usize {
		self.note.note
	}

	/// Get the velocity of the playing note
	pub fn velocity(&self) -> f32 {
		self.note.velocity
	}

	/// Get the release time of the playing note in ms
	///
	/// If the note is not in release phase, returns None.
	pub fn release_time(&self, sample_rate: usize) -> Option<f32> {
		self.release
			.as_ref()
			.map(|(_, count)| *count as f32 / sample_rate as f32 * 1000.0)
	}

	/// Get the release level of the playing note
	///
	/// If the note is not in release phase, returns None.
	pub fn release_level(&self) -> Option<f32> {
		self.release.as_ref().map(|(level, _)| *level)
	}
}
