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
                // A violinist's hand shakes the string much harder and more
                // slowly than a guitarist's, and it starts almost at once.
                params.vibrato_rate = 5.6;
                params.vibrato_depth = 42.0;
                params.vibrato_delay = 0.09;
            }
            Preset::Cello => {
                params.excitation = Excitation::Bow;
                params.position = 0.10;
                params.strings = 1;
                params.hardness = 0.5;
                params.damping = 0.8;
                params.brightness = 0.9;
                params.inharmonicity = 1.0;
                params.bow_force = 0.45;
                params.bow_speed = 0.35;
                params.body = 0.9;
                params.gain = 0.8;
                params.sympathetic = 0.45;
                params.vibrato_rate = 5.0;
                params.vibrato_depth = 34.0;
                params.vibrato_delay = 0.11;
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
                1e-5 * 10f32.powf(3.0 * t),
                geom(700.0, 420.0, t),
                geom(14.0, 3.0, t),
                geom(30.0, 3.0, t),
                geom(6.0, 0.45, t),
                40,
                8,
            ),
            Preset::Guitar => (
                geom(1.5e-4, 4.0e-4, t),
                geom(75.0, 65.0, t),
                geom(1.7, 2.4, t),
                geom(7.0, 3.0, t),
                geom(1.6, 0.45, t),
                32,
                8,
            ),
            Preset::Violin => (1.4e-4, 60.0, 0.35, 0.9, 0.20, 32, 8),
            Preset::Cello => (9.0e-5, 110.0, 0.75, 1.2, 0.25, 24, 8),
        };

        let (mut sigma0, mut sigma1) = decay_from_t60(f0, t60_low, t60_high);
        sigma0 *= params.damping.max(0.01);
        sigma1 *= params.brightness.max(0.01);

        let string = StringConfig {
            frequency: f0,
            inharmonicity: base_b * params.inharmonicity,
            sigma0,
            sigma1,
            tension,
            impedance,
            max_partials,
            dispersion_sections: sections,
            dispersion_delay_fraction: 1.0,
            dispersion_coefficients: None,
        };

        // Everything that touches the string comes from the excitation, the
        // note and the player. None of it comes from the preset, which is what
        // makes any preset playable any way.
        let (pluck_amplitude, hammer_mass, hammer_velocity, hammer_exponent, hammer_stiffness,
             bow_force, bow_velocity) = match params.excitation {
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
                0.15 + 2.6 * params.bow_force,
                0.05 + 0.55 * params.bow_speed,
            ),
        };

        // A bow is held near the bridge and a finger or a hammer near the
        // middle, so the geometry follows the excitation.
        let position = if params.excitation == Excitation::Bow {
            1.0 - params.position.clamp(0.02, 0.5)
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
            Excitation::Pluck => tension * (0.003 + 0.013) / (1.0 - position),
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

        VoiceSpec {
            string,
            excitation: params.excitation,
            position,
            strings: params.strings.clamp(1, 4),
            detune_cents: params.detune_cents.max(0.0),
            pluck_amplitude,
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
