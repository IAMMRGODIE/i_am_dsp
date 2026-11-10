use super::*;

/// The player controls shared by every preset.
#[derive(Clone, Debug, PartialEq)]
pub struct EngineParams {
	/// Which way the string itself is modelled.
	///
	/// This is not a quality setting. The waveguide is cheap and its low
	/// partials are exact; the modal bank is exact in tuning at any stiffness
	/// and costs a great deal more. Which one sounds like the instrument is a
	/// musical question, and the answer is different for a guitar and a piano.
	pub core: Core,
	/// Which contact law the engine uses.
	pub excitation: Excitation,
	/// The output gain.
	pub gain: f32,
	/// Scales the frequency independent decay rate. Larger damps faster.
	pub damping: f32,
	/// Scales the frequency dependent decay rate. Larger is darker.
	pub brightness: f32,
	/// Scales the inharmonicity coefficient `B`.
	pub inharmonicity: f32,
	/// The excitation position, as a fraction of the speaking length.
	pub position: f32,
	/// How many strings sound per note.
	pub strings: usize,
	/// The unison detune, in cents.
	pub detune_cents: f32,
	/// Hammer felt hardness, or plectrum hardness.
	pub hardness: f32,
	/// Bow force, normalised to 0..1.
	pub bow_force: f32,
	/// Bow speed, normalised to 0..1.
	pub bow_speed: f32,
	/// How much of the soundboard colouring is mixed in.
	pub body: f32,
	/// How much sympathetic resonance the undamped strings contribute.
	pub sympathetic: f32,
	/// How far the whole instrument is shifted, in semitones.
	///
	/// Negative is down. This is a transpose control rather than a property of
	/// the instrument, and it defaults to concert pitch so that
	/// [crate::instrument::note_to_frequency] keeps meaning what it says and a
	/// note number sounds the pitch that note number names. A patch that wants
	/// to sit lower sets it itself.
	///
	/// It exists because the model is only convincing over about two octaves in
	/// the middle of its range: below that a string has too few partials to have
	/// a character, and above it the excitation is short enough compared to the
	/// period that the contact model's approximations start to show. Shifting
	/// the instrument down is treating the symptom rather than the cause, but it
	/// is an honest symptom to treat, and it happens to make a piano sound like
	/// a nine foot concert grand rather than a bright upright.
	pub transpose: f32,
	/// How fast the player's left hand shakes the string, in hertz.
	pub vibrato_rate: f32,
	/// How far the pitch swings either side of the note, in cents.
	pub vibrato_depth: f32,
	/// How long a note sounds before the vibrato arrives, in seconds.
	///
	/// A player starts a note with a settled hand and adds the shake a moment
	/// later, which is a large part of why a real vibrato sounds expressive
	/// rather than like a modulation effect.
	pub vibrato_delay: f32,
	/// How long the vibrato takes to reach full depth, in seconds.
	pub vibrato_onset: f32,
}

impl Default for EngineParams {
	fn default() -> Self {
		Self {
			core: Core::default(),
			excitation: Excitation::Pluck,
			gain: 0.8,
			damping: 1.0,
			brightness: 1.0,
			inharmonicity: 1.0,
			position: 1.0 / 6.0,
			strings: 1,
			detune_cents: 0.0,
			hardness: 0.5,
			bow_force: 0.4,
			bow_speed: 0.35,
			body: 0.6,
			sympathetic: 0.6,
			transpose: 0.0,
			vibrato_rate: 5.4,
			vibrato_depth: 12.0,
			vibrato_delay: 0.12,
			vibrato_onset: 0.14,
		}
	}
}

impl EngineParams {
	/// Whether moving from `self` to `other` changes the string itself, and so
	/// invalidates a fitted dispersion filter.
	///
	/// The transpose belongs here even though it only moves the pitch, because
	/// the presets scale the string from *where the key is*, so a transposed
	/// instrument is a different string at every note.
	pub fn affects_string_design(&self, other: &Self) -> bool {
		self.core != other.core
			|| self.damping != other.damping
			|| self.brightness != other.brightness
			|| self.inharmonicity != other.inharmonicity
			|| self.transpose != other.transpose
	}
}
