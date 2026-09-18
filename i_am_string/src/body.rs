//! The body: the resonances of the soundboard, the box or the belly.
//!
//! A vibrating string on its own sounds like a string. What makes it sound like
//! a piano or a violin is that everything it does is filtered by a body with a
//! few dozen resonances of its own. A violin body has its air resonance near
//! 280 Hz, the two main wood resonances at about 470 and 590 Hz, a forest of
//! higher modes, and the famous "bridge hill" around 2 to 3 kHz. Those peaks
//! and, just as importantly, the valleys between them, are what the ear uses to
//! name the instrument.
//!
//! A single lowpass and one peaking filter does not do that. It sounds like a
//! tone control, which is exactly the criticism a physical model gets when it is
//! described as sounding synthetic.

use crate::instrument::Preset;

/// A second order section.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl Biquad {
    fn normalise(b0: f32, b1: f32, b2: f32, a0: f32, a1: f32, a2: f32) -> Self {
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            ..Default::default()
        }
    }

    pub(crate) fn lowpass(frequency: f32, q: f32, sample_rate: f32) -> Self {
        let omega = std::f32::consts::TAU * (frequency / sample_rate).clamp(1e-5, 0.49);
        let (sine, cosine) = omega.sin_cos();
        let alpha = sine / (2.0 * q);
        let b1 = 1.0 - cosine;
        Self::normalise(b1 * 0.5, b1, b1 * 0.5, 1.0 + alpha, -2.0 * cosine, 1.0 - alpha)
    }

    pub(crate) fn highpass(frequency: f32, q: f32, sample_rate: f32) -> Self {
        let omega = std::f32::consts::TAU * (frequency / sample_rate).clamp(1e-5, 0.49);
        let (sine, cosine) = omega.sin_cos();
        let alpha = sine / (2.0 * q);
        let b0 = (1.0 + cosine) * 0.5;
        Self::normalise(b0, -(1.0 + cosine), b0, 1.0 + alpha, -2.0 * cosine, 1.0 - alpha)
    }

    /// A constant peak gain bandpass, which is what a body resonance is.
    pub(crate) fn bandpass(frequency: f32, q: f32, sample_rate: f32) -> Self {
        let omega = std::f32::consts::TAU * (frequency / sample_rate).clamp(1e-5, 0.49);
        let (sine, cosine) = omega.sin_cos();
        let alpha = sine / (2.0 * q.max(0.1));
        Self::normalise(alpha, 0.0, -alpha, 1.0 + alpha, -2.0 * cosine, 1.0 - alpha)
    }

    #[inline]
    pub(crate) fn process(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

/// One measured-ish resonance: frequency, sharpness, and how strongly it radiates.
type Mode = (f32, f32, f32);

/// The resonance set of a piano soundboard.
///
/// A soundboard is large and heavily damped, so its modes are broad and its
/// response is closer to smooth than a violin's. Most of a piano's character
/// comes from its strings; the board mostly has to not get in the way.
const PIANO: &[Mode] = &[
    (58.0, 7.0, 1.00),
    (94.0, 8.0, 0.90),
    (141.0, 9.0, 0.80),
    (203.0, 10.0, 0.70),
    (286.0, 11.0, 0.62),
    (398.0, 10.0, 0.53),
    (556.0, 9.0, 0.45),
    (772.0, 8.0, 0.38),
    (1063.0, 8.0, 0.31),
    (1450.0, 7.0, 0.25),
    (1980.0, 6.0, 0.19),
    (2700.0, 5.0, 0.14),
];

/// The box of a steel string guitar: a Helmholtz air mode, a top plate mode,
/// and the bridge hill.
const GUITAR: &[Mode] = &[
    (102.0, 14.0, 1.00),
    (204.0, 16.0, 0.85),
    (295.0, 13.0, 0.60),
    (405.0, 12.0, 0.50),
    (525.0, 11.0, 0.45),
    (690.0, 10.0, 0.42),
    (905.0, 9.0, 0.44),
    (1200.0, 9.0, 0.50),
    (1650.0, 8.0, 0.60),
    (2300.0, 7.0, 0.68),
    (3100.0, 6.0, 0.55),
    (4200.0, 5.0, 0.34),
];

/// A violin belly: the air mode, the two main wood resonances, the wood forest
/// and the bridge hill.
const VIOLIN: &[Mode] = &[
    (278.0, 28.0, 1.00),
    (335.0, 12.0, 0.30),
    (405.0, 12.0, 0.36),
    (472.0, 18.0, 0.78),
    (592.0, 17.0, 0.92),
    (700.0, 13.0, 0.48),
    (855.0, 11.0, 0.42),
    (1010.0, 10.0, 0.48),
    (1250.0, 9.0, 0.55),
    (1600.0, 8.0, 0.62),
    (2000.0, 7.0, 0.72),
    (2500.0, 6.0, 0.80),
    (3000.0, 5.0, 0.70),
    (3800.0, 4.0, 0.45),
];

/// A cello belly, which is a violin's scaled up by roughly a factor of two and a
/// half.
const CELLO: &[Mode] = &[
    (123.0, 28.0, 1.00),
    (148.0, 12.0, 0.30),
    (180.0, 12.0, 0.36),
    (196.0, 18.0, 0.78),
    (255.0, 17.0, 0.92),
    (305.0, 13.0, 0.48),
    (375.0, 11.0, 0.42),
    (450.0, 10.0, 0.48),
    (565.0, 9.0, 0.55),
    (720.0, 8.0, 0.62),
    (900.0, 7.0, 0.72),
    (1150.0, 6.0, 0.80),
    (1400.0, 5.0, 0.70),
    (1800.0, 4.0, 0.45),
];

/// The whole body: a set of resonances, a direct path, and the radiation
/// roll-off at the top.
#[derive(Clone, Debug)]
pub(crate) struct Body {
    modes: Vec<Biquad>,
    gains: Vec<f32>,
    direct: f32,
    highpass: Biquad,
    lowpass: Biquad,
    output: f32,
}

/// A deterministic value in `[-1, 1)` for one resonance on one side.
///
/// The two channels of a stereo body must not be two different instruments, so
/// they cannot simply be detuned by opposite constants: a constant shift moves
/// every resonance the same way, which the ear hears as one body transposed.
/// What is wanted is two views of the *same* body whose fine structure differs,
/// which is what independent per-resonance scatter gives. It has to be
/// deterministic, because a body that changed every time a note was played
/// would not be a body.
fn scatter(index: usize, side: f32) -> f32 {
    let mut x = (index as u32)
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add(0x85EB_CA6B);
    x ^= x >> 13;
    x = x.wrapping_mul(0xC2B2_AE35);
    x ^= x >> 16;
    // The side selects a different starting point in the sequence rather than a
    // different sign, so the two channels are uncorrelated instead of mirrored.
    let offset = if side < 0.0 { 0.0 } else { 0.5 };
    let unit = (x >> 8) as f32 / (1u32 << 24) as f32;
    ((unit + offset).fract()) * 2.0 - 1.0
}

impl Body {
    /// Build one side of the body of an instrument.
    ///
    /// `side` is `-1.0` or `1.0`, one per channel. Two bodies built this way
    /// agree on everything that names the instrument - where the air resonance
    /// is, where the bridge hill is, how bright the radiation is - and disagree
    /// on the several percent of scatter that no two listening positions ever
    /// share. That is the difference between an instrument and a mono source
    /// panned into the middle of the stereo field.
    pub(crate) fn for_preset(preset: Preset, sample_rate: f32, side: f32) -> Self {
        let (table, direct, highpass, lowpass): (&[Mode], f32, f32, f32) = match preset {
            Preset::Piano => (PIANO, 0.35, 32.0, 9000.0),
            Preset::Guitar => (GUITAR, 0.22, 72.0, 7500.0),
            Preset::Violin => (VIOLIN, 0.18, 210.0, 11000.0),
            Preset::Cello => (CELLO, 0.18, 78.0, 9000.0),
        };

        let modes: Vec<Biquad> = table
            .iter()
            .enumerate()
            .map(|(index, (frequency, q, _))| {
                // Seven percent one way or the other is around half a bandwidth
                // for the sharper resonances and rather less for the broad ones,
                // which is the amount of disagreement two points on a soundboard
                // actually see without either of them sounding wrong.
                let detune = 1.0 + 0.07 * scatter(index, side);
                Biquad::bandpass(frequency * detune, q / detune.sqrt(), sample_rate)
            })
            .collect();
        let gains: Vec<f32> = table
            .iter()
            .enumerate()
            .map(|(index, (_, _, gain))| gain * (1.0 + 0.30 * scatter(index + 11, side)))
            .collect();
        // Normalise so that the body neither buries the string nor disappears
        // behind it whichever note is played.
        let total = direct + gains.iter().sum::<f32>() * 0.5;
        Self {
            modes,
            gains,
            direct,
            // Radiation off a finite plate reaches two ears through different
            // path lengths, so one side is a little brighter and loses a little
            // more of its bottom. The two percent between them is the tilt, not
            // a tone control.
            highpass: Biquad::highpass(highpass * (1.0 + 0.10 * side), 0.7, sample_rate),
            lowpass: Biquad::lowpass(lowpass * (1.0 - 0.08 * side), 0.7, sample_rate),
            output: 1.0 / total.max(1e-3),
        }
    }

    #[inline]
    pub(crate) fn process(&mut self, x: f32) -> f32 {
        let mut wet = self.direct * x;
        for (mode, gain) in self.modes.iter_mut().zip(self.gains.iter()) {
            wet += *gain * mode.process(x);
        }
        let shaped = self.lowpass.process(wet);
        let shaped = self.highpass.process(shaped);
        shaped * self.output
    }
}
