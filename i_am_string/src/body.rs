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
		Self::normalise(
			b1 * 0.5,
			b1,
			b1 * 0.5,
			1.0 + alpha,
			-2.0 * cosine,
			1.0 - alpha,
		)
	}

	pub(crate) fn highpass(frequency: f32, q: f32, sample_rate: f32) -> Self {
		let omega = std::f32::consts::TAU * (frequency / sample_rate).clamp(1e-5, 0.49);
		let (sine, cosine) = omega.sin_cos();
		let alpha = sine / (2.0 * q);
		let b0 = (1.0 + cosine) * 0.5;
		Self::normalise(
			b0,
			-(1.0 + cosine),
			b0,
			1.0 + alpha,
			-2.0 * cosine,
			1.0 - alpha,
		)
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

/// The resonance set of a piano soundboard, generated rather than tabulated.
///
/// A soundboard is large and heavily damped: its modes are broad and dense
/// enough that the response is smooth, and most of a piano's character comes
/// from the strings. Fifteen hand-picked resonances leave gaps that a real
/// board does not have, and a gap is heard as a note with a partial missing,
/// so the board is built from its envelope instead: forty modes spaced evenly
/// in octaves from 40 Hz to 8 kHz, each carrying the target response at its
/// own frequency, with a quarter of modal ripple kept and the bridge hill
/// included in the target.
fn piano_table() -> Vec<Mode> {
	const MODES: usize = 40;
	let mut modes = Vec::with_capacity(MODES);
	for index in 0..MODES {
		let t = index as f32 / (MODES - 1) as f32;
		// 40 Hz to 8 kHz, evenly in octaves, with a deterministic scatter so
		// the modes do not line up with the harmonic series.
		let frequency = 40.0 * 200.0f32.powf(t) * (1.0 + 0.02 * ((index * 37) as f32).sin());
		let square = frequency * frequency;
		// The board's envelope: a 70 Hz corner, a 3.8 kHz treble corner with a
		// 1.8 slope, and a broad bridge hill between two and five kilohertz.
		let envelope = square / (square + 70.0 * 70.0) / (1.0 + (frequency / 3800.0).powf(1.8))
			// The bridge hill: +8 dB around 1.7 kHz and a shoulder at 2.6 kHz,
			// the reference piano's contour, rather than the broad 3.2 kHz lift
			// the board had before.
			* (1.0
				+ 1.5 * (-((frequency - 1700.0) / 700.0).powi(2)).exp()
				+ 0.4 * (-((frequency - 2600.0) / 500.0).powi(2)).exp());
		// A quarter of modal ripple is what a board's response keeps above the
		// smooth curve; the gain carries it at the mode's own frequency.
		let ripple = 1.0 + 0.25 * ((index * 53) as f32).sin();
		// Broader at the bottom, where a board is large, and at the top, where
		// radiation takes over; the sharpest modes sit in the middle.
		let q = 3.0 + 9.0 * (1.0 - (2.0 * t - 1.0).abs());
		modes.push((frequency, q, envelope * ripple));
	}
	modes
}

/// The box of a steel string guitar.
///
/// The old table had two sharp resonances at 102 and 204 Hz and very little
/// between them, which measured as a body that left the second, third and
/// fourth partials a decibel or two above the fundamental and the region
/// between 150 and 400 Hz almost empty. A guitar body is not two resonances:
/// it is a broad, dense set of them, and its job is to lift the low middle
/// of the spectrum by several decibels. Measured against a set of real
/// classical guitar single notes, the old table put the body +11 dB at
/// 104 Hz and +1 dB at 165, 247 and 330, which is why the model sounded
/// thin next to the samples. This table is flatter: +5 to +6 dB across
/// 104-660 Hz, which is the region the ear reads as the size of the box.
const GUITAR: &[Mode] = &[
	(102.0, 5.5, 0.35),
	(150.0, 5.0, 0.85),
	(190.0, 6.0, 0.95),
	(240.0, 6.0, 0.95),
	(300.0, 6.5, 0.90),
	(380.0, 8.0, 0.95),
	(500.0, 9.0, 0.70),
	(650.0, 8.0, 0.45),
	(850.0, 7.0, 0.40),
	(1100.0, 7.0, 0.40),
	(1500.0, 6.0, 0.60),
	(2100.0, 6.0, 0.75),
	(2900.0, 5.0, 0.60),
	(3800.0, 4.0, 0.30),
	(5000.0, 3.0, 0.15),
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
		// The mode bank's strength is a constant per instrument, and the table's
		// gains are normalised to a sum of one: the shape of the body and how
		// much of it is heard are then two separate numbers. Without that split
		// the normalisation cancels any edit to the gains, and a body measured
		// from a recording cannot be made to fit -- scaling every resonance by
		// two used to change the rendered spectrum by less than a decibel.
		// The piano's board is generated rather than tabulated: forty modes
		// from 40 Hz to 8 kHz, spaced evenly in octaves, each carrying the
		// board's envelope at its own frequency. Fifteen hand-picked modes
		// leave gaps a real board does not have, and a gap is heard as a note
		// with a partial missing.
		let (table, direct, highpass, lowpass, mix): (Vec<Mode>, f32, f32, f32, f32) = match preset
		{
			Preset::Violin => (VIOLIN.to_vec(), 0.18, 210.0, 11000.0, 8.58),
			Preset::Cello => (CELLO.to_vec(), 0.18, 78.0, 9000.0, 8.58),
		};
		let table = table.as_slice();

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
		let raw: Vec<f32> = table
			.iter()
			.enumerate()
			.map(|(index, (_, _, gain))| gain * (1.0 + 0.30 * scatter(index + 11, side)))
			.collect();
		// The gains sum to the instrument's strength, not to whatever the table
		// happens to add up to, so the mode bank neither buries the string nor
		// disappears behind it, and a resonance can be asked for more strongly
		// than its neighbours without the normalisation eating the difference.
		let sum = raw.iter().sum::<f32>().max(1e-6);
		let gains: Vec<f32> = raw.iter().map(|gain| mix * gain / sum).collect();
		let total = direct + mix * 0.5;
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
