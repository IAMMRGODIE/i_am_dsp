//! Instrument presets and the per-note string scaling.
//!
//! A preset is not a new engine - it is a pair of curves that turn a note
//! number into a [StringConfig], plus the defaults for the player controls.
//! Swapping Piano for Violin changes the string and the box it is bolted to,
//! and nothing else.
//!
//! # Any string, any excitation
//!
//! The excitation is deliberately *not* part of the preset. A preset says what
//! the string is made of, how long it is and what body it drives; the excitation
//! says what is touching it. A piano string struck by a felt hammer and the same
//! string plucked by a fingernail are two instruments made of the same parts,
//! and there is no reason for the model to refuse to play one of them. Some of
//! the combinations are silly - a piano string bowed by a rosined ribbon is not
//! a thing anyone wants - and some are useful: a guitar played with a hammer is
//! a piano, and a cello plucked is a pizzicato.
//!
//! What the preset does contribute to the excitation is *geometry*: how far
//! along the string it is applied. A bow goes near the bridge where a finger
//! goes near the middle, and that follows the excitation, not the instrument.

use crate::engine::EngineParams;
use crate::string::StringConfig;
use crate::voicing::Voicing;

/// Convert a MIDI note number to a frequency in hertz, with A4 = 69 = 440 Hz.
pub fn note_to_frequency(note: f32) -> f32 {
	440.0 * 2.0f32.powf((note - 69.0) / 12.0)
}

/// The three ways a string can be excited.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Excitation {
	/// A finger or plectrum holding and releasing the string.
	Pluck,
	/// A hammer with a non-linear felt.
	Hammer,
	/// A rosined bow sliding across the string.
	Bow,
}

impl Excitation {
	/// A short human readable name.
	pub fn name(&self) -> &'static str {
		match self {
			Excitation::Pluck => "Pluck",
			Excitation::Hammer => "Hammer",
			Excitation::Bow => "Bow",
		}
	}

	/// Everything, in menu order.
	pub const ALL: [Excitation; 3] = [Excitation::Pluck, Excitation::Hammer, Excitation::Bow];
}

/// Everything a voice needs in order to be built and excited.
#[derive(Clone, Debug)]
pub struct VoiceSpec {
	/// The string itself.
	pub string: StringConfig,
	/// How it is excited.
	pub excitation: Excitation,
	/// Where it is excited, as a fraction of the speaking length from the nut.
	pub position: f32,
	/// How many strings sound together for this note.
	pub strings: usize,
	/// How far the unison strings are detuned, in cents.
	pub detune_cents: f32,
	/// The pluck displacement as a fraction of the speaking length.
	pub pluck_amplitude: f32,
	/// How long the finger takes to slip off, in seconds; zero is instant.
	pub pluck_release: f32,
	/// The hammer mass, in kilograms.
	pub hammer_mass: f32,
	/// The hammer velocity at impact, in metres per second.
	pub hammer_velocity: f32,
	/// The felt exponent.
	pub hammer_exponent: f32,
	/// The felt stiffness.
	pub hammer_stiffness: f32,
	/// The bow force, in newtons.
	pub bow_force: f32,
	/// The bow speed, in metres per second.
	pub bow_velocity: f32,
	/// Sub-steps used to integrate the contact.
	pub contact_substeps: u32,
	/// The bridge force a full velocity note is expected to start with.
	pub reference_force: f32,
}

/// A family of instruments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Preset {
	/// A grand piano: felt hammers, three strings, strong inharmonicity.
	Piano,
	/// A steel string acoustic guitar.
	Guitar,
	/// A bowed violin.
	Violin,
	/// A bowed cello.
	Cello,
}

impl Preset {
	/// Every preset, for building a menu.
	pub const ALL: [Preset; 4] = [Preset::Piano, Preset::Guitar, Preset::Violin, Preset::Cello];

	/// A short human readable name.
	pub fn name(&self) -> &'static str {
		match self {
			Preset::Piano => "Piano",
			Preset::Guitar => "Guitar",
			Preset::Violin => "Violin",
			Preset::Cello => "Cello",
		}
	}

	/// The player controls this preset starts from.
	pub fn defaults(&self) -> EngineParams {
		let mut params = EngineParams::default();
		match self {
			Preset::Piano => {
				params.excitation = Excitation::Hammer;
				params.position = 1.0 / 8.0;
				params.strings = 3;
				params.detune_cents = 0.6;
				params.hardness = 0.5;
				params.damping = 0.9;
				params.brightness = 0.85;
				params.inharmonicity = 1.0;
				params.body = 0.85;
				params.gain = 0.8;
				params.sympathetic = 0.55;
				params.vibrato_depth = 4.0;
			}
			Preset::Guitar => {
				params.excitation = Excitation::Pluck;
				params.position = 1.0 / 5.0;
				params.strings = 1;
				params.hardness = 0.5;
				params.damping = 1.4;
				params.brightness = 1.2;
				params.inharmonicity = 1.0;
				params.body = 0.9;
				params.gain = 0.9;
				params.sympathetic = 0.4;
				params.vibrato_depth = 18.0;
			}
			Preset::Violin => {
				params.excitation = Excitation::Bow;
				params.position = 0.09;
				params.strings = 1;
				params.hardness = 0.5;
				params.damping = 1.0;
				params.brightness = 1.0;
				params.inharmonicity = 1.0;
				params.bow_force = 0.35;
				params.bow_speed = 0.3;
				params.body = 0.9;
				params.gain = 0.8;
				params.sympathetic = 0.35;
				// The vibrato measured on a solo violin recording: five shakes a
				// second and thirteen cents either way. The old 42 cents was a
				// caricature - wide enough to hear the modulation instead of the
				// note - and it began 90 ms in, before a player's hand has moved.
				params.vibrato_rate = 4.9;
				params.vibrato_depth = 19.0;
				params.vibrato_delay = 0.22;
				params.vibrato_onset = 0.3;
			}
			Preset::Cello => {
				params.excitation = Excitation::Bow;
				params.position = 0.10;
				params.strings = 1;
				params.hardness = 0.5;
				params.damping = 0.8;
				params.brightness = 0.90;
				params.inharmonicity = 1.0;
				// The bow was fitted to a recorded cello section rather than
				// picked by ear: force and speed were swept against the partial
				// levels of ten legato notes, and the settings here are the ones
				// whose whole spectrum sits closest to the recording's (4.9 dB of
				// band error, against 5.8 for the settings before). The speed is a
				// slow stroke, about 0.12 m/s, and the force about 1.2 N.
				params.bow_force = 0.40;
				params.bow_speed = 0.12;
				params.body = 0.9;
				params.gain = 0.8;
				params.sympathetic = 0.45;
				// A cellist's hand is heavier and slower to arrive than a
				// violinist's, and the measured width is about 24 cents.
				params.vibrato_rate = 6.0;
				params.vibrato_depth = 30.0;
				params.vibrato_delay = 0.2;
				params.vibrato_onset = 0.28;
			}
		}
		params
	}

	/// Build the voice specification for one note.
	///
	/// `velocity` is in 0..1. The preset supplies the string and its body; the
	/// excitation and the player's controls supply everything that touches it.
	pub fn spec(&self, note: u8, velocity: f32, params: &EngineParams) -> VoiceSpec {
		let note = note.clamp(0, 127);
		let t = ((note as f32) - 21.0) / 87.0;
		let velocity = velocity.clamp(0.0, 1.0);
		// The pitch that actually sounds. The transpose moves it without moving
		// the key, and deliberately does not move the string scaling below,
		// which stays tied to where the key is: a string tuned two octaves down
		// from its key is a longer, heavier string, which is the point.
		let f0 = note_to_frequency(note as f32 + params.transpose);

		let (base_b, tension, impedance, t60_low, t60_high, max_partials, sections) = match self {
			Preset::Piano => (
				// The stiff-string coefficient, from the reference piano: it rises
				// with the fundamental from the wound bass to the treble, which is
				// what stretches the partials by the amount the ear hears as a
				// piano. The old `1e-5 * 10^(3t)` was 0.6 of that at C4, and a
				// harmonic series that tidy reads as an organ, not a piano.
				7e-4 * (f0 / 440.0).powf(1.4) + 1e-4 * (55.0 / f0).clamp(0.0, 2.2),
				geom(700.0, 420.0, t),
				geom(14.0, 3.0, t),
				// The fundamental's T60. The old anchors, 30 s at A0 and 3 s at C8,
				// made E1 ring for 19 s where the sample set lets it go in 11, and
				// fell too fast through the middle: C4 decayed in 7 s against the
				// recording's 9. Measured envelope by envelope, this set of samples
				// wants a nearly flat 10-12 s from E1 to C4, so the anchors are 11 s
				// at A0 and 8 s at C8 - flatter than a geometric curve wants to be,
				// but the bridge's damping already makes the treble decay faster
				// than its anchor.
				// The first polarization is the prompt: 2.5 times faster than the
				// tail, so these anchors are the sample's 11-8 s divided by that.
				// The second polarization multiplies by 0.4 and lands on the tail
				// the samples asked for.
				geom(4.4, 3.2, t),
				geom(6.0, 0.45, t),
				40,
				8,
			),
			Preset::Guitar => (
				geom(1.5e-4, 4.0e-4, t),
				geom(75.0, 65.0, t),
				geom(1.7, 2.4, t),
				geom(7.0, 3.0, t),
				// The tenth partial's T60. It used to be 1.6 s at the bottom of
				// the range, which with the bridge's extra damping left the tenth
				// partial at 0.75 s against the 1.9 s of a real classical guitar,
				// so every partial above 700 Hz was gone before the note bloomed
				// and the sound had no air. Measured against the samples, 3 s puts
				// the tenth partial where the recording has it.
				geom(3.0, 0.45, t),
				32,
				8,
			),
			Preset::Violin => (1.4e-4, 60.0, 0.35, 0.9, 0.20, 32, 8),
			Preset::Cello => (9.0e-5, 110.0, 0.75, 1.2, 0.25, 24, 8),
		};

		let (mut sigma0, mut sigma1) = decay_from_t60(f0, t60_low, t60_high);
		sigma0 *= params.damping.max(0.01);
		sigma1 *= params.brightness.max(0.01);
		// A plucked note decays faster than a bowed one on the same string: the
		// bow keeps feeding the string, so the preset's T60 is really the decay
		// of a note after the bow lifts. Left alone, a pizzicato rang for 1.5
		// seconds where the recorded ones last about one.
		//
		// Only the bowed presets need it. Their T60 anchors describe a note
		// under the bow; a guitar's and a piano's describe the note itself,
		// and shortening those made a guitar's low E die in 5.6 s where a real
		// one rings for about 8.
		if params.excitation == Excitation::Pluck && matches!(self, Preset::Cello | Preset::Violin)
		{
			sigma0 *= 1.6;
		}

		// The colour layers. A bowed string is driven continuously, so the
		// two stage decay is masked by the bow and the bow locks the partials
		// together; a plucked or struck one has neither of those things.
		let plucked = matches!(params.excitation, Excitation::Pluck | Excitation::Hammer);
		let (late_decay, doublet, winding) = match self {
			// A piano's prompt decay is real but small. At 0.78 the late stage
			// doubled the T60 of F3 to 15 s against the 8 s of a recorded note,
			// and a piano whose every note has twice the tail it should is the
			// sound of the sustain pedal being held down.
			Preset::Piano => (1.0, 0.0, 0.25),
			Preset::Guitar => (0.70, 0.18, 0.55),
			Preset::Violin => (0.85, 0.07, 0.30),
			Preset::Cello => (0.85, 0.10, 0.45),
		};
		let mut voicing = Voicing {
			late_decay: if plucked { late_decay } else { 1.0 },
			late_delay: 0.35,
			late_ramp: 1.2,
			doublet: if plucked { doublet } else { 0.0 },
			doublet_detune: 0.003,
			// The winding only exists where the string is wound: the bottom
			// of the instrument's range. A plain treble string has no cover
			// wire to ring. A bowed note gets none of it either: the bow
			// drives a Helmholtz cycle that locks the partials together, and
			// a layer of off-grid ringers under it is heard as noise.
			winding: if plucked {
				winding * ((420.0 - f0) / 260.0).clamp(0.0, 1.0)
			} else {
				0.0
			},
			pluck_tilt: 0.0,
			// The two things a piano has that a guitar does not: a second
			// polarization that holds the aftersound, and the longitudinal sum
			// tones the hammer's blow rings out of the string.
			phantom: if matches!(self, Preset::Piano) {
				1.0
			} else {
				0.0
			},
			horizontal: if matches!(self, Preset::Piano) {
				1.0
			} else {
				0.0
			},
			horizontal_decay: 0.4,
		};
		if params.excitation == Excitation::Pluck {
			// A soft pluck is flesh and a hard one is a plectrum: the harder
			// the contact, the less it darkens the top of the attack.
			voicing.pluck_tilt = 0.035 * (1.0 - 0.7 * velocity);
		}

		let string = StringConfig {
			frequency: f0,
			inharmonicity: base_b * params.inharmonicity,
			// The board the string is bolted to. See crate::bridge for what it
			// does to the decay and the tuning of each partial.
			bridge: BRIDGE_COUPLING,
			sigma0,
			sigma1,
			tension,
			impedance,
			max_partials,
			dispersion_sections: sections,
			dispersion_delay_fraction: 1.0,
			voicing,
			dispersion_coefficients: None,
		};

		// Everything that touches the string comes from the excitation, the
		// note and the player. None of it comes from the preset, which is what
		// makes any preset playable any way.
		let (
			pluck_amplitude,
			hammer_mass,
			hammer_velocity,
			hammer_exponent,
			hammer_stiffness,
			bow_force,
			bow_velocity,
		) = match params.excitation {
			Excitation::Pluck => (
				0.003 + 0.013 * velocity.powf(1.4),
				0.0,
				0.0,
				0.0,
				0.0,
				0.0,
				0.0,
			),
			Excitation::Hammer => {
				let mass = geom(0.0105, 0.0055, t);
				let velocity_h = 0.45 + 4.6 * velocity.powf(1.7);
				(
					0.0,
					mass,
					velocity_h,
					2.4 + 0.8 * t,
					// The contact time only goes as stiffness to the power of
					// minus one over (p+1), which is about a quarter, so the
					// stiffness has to span two orders of magnitude for the
					// treble hammer to be short enough to excite a 2 kHz string.
					geom(4.0e8, 2.5e10, t) * 10f32.powf(1.2 * (params.hardness - 0.5)),
					0.0,
					0.0,
				)
			}
			Excitation::Bow => (
				0.0,
				0.0,
				0.0,
				0.0,
				0.0,
				saturate_force(0.15 + 2.6 * params.bow_force, impedance),
				0.05 + 0.55 * params.bow_speed,
			),
		};

		// A bow is held near the bridge and a finger or a hammer near the
		// middle, so the geometry follows the excitation.
		//
		// The force itself was limited above, before it reached the string.
		let position = if params.excitation == Excitation::Bow {
			1.0 - params.position.clamp(0.02, 0.5)
		} else if params.excitation == Excitation::Pluck {
			// A player plucks a good deal further along the string than the bow
			// sits - around a third of the speaking length from the nut, where
			// the fingerboard ends - so the pluck point follows the control but
			// sits well inside it. Plucking where the bow sits gave the attack a
			// centroid of 320 Hz on a cello's C3 against 160 in the recording.
			//
			// A guitar is the exception: the hand hangs over the soundhole,
			// three quarters of the way down the string, not over the end of
			// the fingerboard. Plucking a guitar at 0.4 put a null on its fifth
			// partial and left the second twenty decibels under the first,
			// which is exactly what a thin guitar sounds like.
			if matches!(self, Preset::Guitar) {
				(params.position + 0.52).clamp(0.60, 0.85)
			} else {
				(params.position + 0.2).clamp(0.15, 0.45)
			}
		} else {
			params.position.clamp(0.01, 0.5)
		};

		// What the bridge force is expected to be at full velocity. Getting this
		// roughly right is what keeps the level even across the keyboard instead
		// of the bass drowning the treble.
		let reference_force = match params.excitation {
			// A string plucked to a height h at a fraction beta of its length is
			// a triangle, and the tension pulls the bridge along its steeper
			// slope: T h / (L (1 - beta)). In units of the speaking length that
			// is T A / (1 - beta). It is exact rather than an estimate, because
			// the string is already in that state before it is let go.
			//
			// Normalised by the energy the triangle carries, not by the force at
			// the bridge corner. The corner's force goes as 1/(1 - beta), so a
			// pluck moved from the middle to the soundhole would be turned down
			// by the gain stage by four decibels even though the note carries
			// the same energy - and the position control would double as a
			// volume control, which is what made the soundhole pluck quiet.
			Excitation::Pluck => {
				let beta = position.clamp(0.02, 0.98);
				// 0.0125 rather than the full 0.016 amplitude keeps the rendered
				// level where the old corner-force normalisation had it, while
				// removing its dependence on where along the string the pluck is.
				tension * 0.0125 / (beta * (1.0 - beta)).sqrt()
			}
			Excitation::Hammer => {
				// The peak force of a felt hammer follows from the energy it
				// carries and the stiffness of the felt:
				//     F = K^(1/(p+1)) * ((p+1) E / 4)^(p/(p+1))
				//
				// This has to be evaluated at *full* velocity, not at the
				// velocity that was played. Normalising by the force the note
				// actually produces would cancel the dynamics out exactly, and a
				// piano that plays every note at the same loudness no matter how
				// hard the key is hit is not a piano.
				let full = 0.45 + 4.6;
				let energy = 0.5 * hammer_mass * full * full;
				let exponent = hammer_exponent;
				let peak = hammer_stiffness.powf(1.0 / (exponent + 1.0))
					* (((exponent + 1.0) * energy / 4.0).max(1e-12))
						.powf(exponent / (exponent + 1.0));
				// A light string yields under the felt instead of being driven
				// by it, so the hammer has to be scaled to what it is hitting or
				// a guitar struck by a piano's hammer is simply a click. The
				// contact time is what has to stay sensible, and it goes as the
				// square root of the mass over the stiffness, so the stiffness
				// follows the string's impedance.
				peak * (impedance / 7.0).max(0.05)
			}
			// A bowed string settles into a Helmholtz cycle whose corner carries
			// roughly twice the bow speed.
			Excitation::Bow => impedance * 2.0 * bow_velocity,
		};

		let contact_substeps = match params.excitation {
			Excitation::Pluck => 1,
			Excitation::Hammer => 4,
			Excitation::Bow => 2,
		};

		// The string does not let go in a single sample. A plectrum releases in
		// under a millisecond, a fingertip in three or four, and a soft pluck
		// takes longer than a hard one.
		let pluck_release = if params.excitation == Excitation::Pluck {
			0.0015 + 0.0025 * (1.0 - velocity)
		} else {
			0.0
		};

		VoiceSpec {
			string,
			excitation: params.excitation,
			position,
			strings: params.strings.clamp(1, 4),
			detune_cents: params.detune_cents.max(0.0),
			pluck_amplitude,
			pluck_release,
			hammer_mass,
			hammer_velocity,
			hammer_exponent,
			hammer_stiffness,
			bow_force,
			bow_velocity,
			contact_substeps,
			reference_force: reference_force.max(1e-4),
		}
	}
}

/// A geometric interpolation, the natural one for acoustics.
fn geom(a: f32, b: f32, t: f32) -> f32 {
	a * (b / a).powf(t.clamp(0.0, 1.0))
}

/// Turn two decay time anchors - the fundamental and its tenth partial - into
/// the `sigma0 + sigma1 f^2` law that the string loss expects.
/// How much normal force a string of this impedance takes before the rosin
/// stops behaving like a constant.
///
/// A bow pressed harder does not transmit proportionally more friction: the
/// hair and the rosin film saturate, and the string's stick-slip cycle starts to
/// skip slips. In the model a skipped slip is a period doubling, heard as a
/// subharmonic thump rather than as loudness. It arrives at about 2.2 N on a
/// cello string and about 1.1 N on a violin's thinner one, which is close to
/// two and a half times the string's impedance in each case -- the same property that
/// decides how much force the string takes to speak at all.
fn bow_force_knee(impedance: f32) -> f32 {
	2.3 * impedance.clamp(0.1, 6.0)
}

/// Past the knee the extra pressure only adds a little, and approaches
/// knee + a fifth of it.
fn saturate_force(force: f32, impedance: f32) -> f32 {
	let knee = bow_force_knee(impedance);
	let width = 0.2 * knee;
	if force <= knee {
		force
	} else {
		knee + width * (1.0 - (-(force - knee) / width).exp())
	}
}

/// How much of the bridge admittance reaches the string.
///
/// A real board is neither rigid nor infinitely soft: it drains a partial that
/// sits on one of its resonances several times faster than its neighbours, and
/// moves it off its harmonic position by a few cents. The measured T60 presets
/// already carry the average decay, so this only has to add the scatter.
const BRIDGE_COUPLING: f32 = 0.35;

/// Turn two decay time anchors - the fundamental and its tenth partial - into
/// the loss law that the string expects.
fn decay_from_t60(f0: f32, t60_low: f32, t60_high: f32) -> (f32, f32) {
	const LN1000: f32 = 6.907_755;
	let low = LN1000 / t60_low.max(1e-3);
	let high = LN1000 / t60_high.max(1e-3);
	let f_high = 10.0 * f0;
	let denominator = (f_high * f_high - f0 * f0).max(1.0);
	let sigma1 = ((high - low) / denominator).max(0.0);
	let sigma0 = (low - sigma1 * f0 * f0).max(0.02);
	(sigma0, sigma1)
}
