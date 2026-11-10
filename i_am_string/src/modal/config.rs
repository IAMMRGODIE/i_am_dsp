use crate::voicing::Voicing;

/// The physical parameters of one string.
#[derive(Clone, Debug)]
pub struct StringConfig {
	/// The fundamental frequency in hertz.
	pub frequency: f32,
	/// The inharmonicity coefficient `B` of `f_n = n f_0 sqrt(1 + B n^2)`.
	pub inharmonicity: f32,
	/// The frequency independent part of the decay rate, in 1/s.
	pub sigma0: f32,
	/// The frequency dependent part of the decay rate, in seconds.
	pub sigma1: f32,
	/// The string tension, in newtons.
	pub tension: f32,
	/// The string's wave impedance `Z0 = sqrt(T mu)`, in N s/m.
	pub impedance: f32,
	/// How strongly the bridge admittance acts on the modes, 0 for a rigid
	/// termination.
	pub bridge: f32,
	/// How the note is coloured after it has been excited.
	///
	/// The harmonic series is only the skeleton: the pairs, the tail and the
	/// wound layer are what make it a string. See [crate::voicing::Voicing].
	pub voicing: Voicing,
}

impl Default for StringConfig {
	fn default() -> Self {
		Self {
			frequency: 220.0,
			inharmonicity: 1e-4,
			sigma0: 0.6,
			sigma1: 3e-6,
			tension: 80.0,
			impedance: 1.0,
			bridge: 0.0,
			voicing: Voicing::default(),
		}
	}
}

/// The largest number of modes one string may use.
///
/// A mode is a damped oscillator, and the ear reads a string's stiffness from
/// how far its *high* partials have been pushed sharp, so the partials near the
/// top of the band are not decoration. What sets the ceiling is cost: measured, a
/// mode costs a little under a nanosecond a sample, eight voices of a three
/// string piano is twenty four strings, and the arithmetic runs out long before
/// the physics does. At 27.5 Hz a hundred and sixty modes reach 5 kHz, and one
/// note further up the range every string is stopped by Nyquist instead; the
/// partials this gives up are the ones the loss law has already buried.
pub(crate) const MAX_MODES: usize = 160;

/// The highest partial the detuned pair is added to.
///
/// Above this the pair is a roughness rather than a bloom, and it is the
/// low partials where the few cents between two polarisations are heard as
/// a slow swell.
pub(crate) const DOUBLET_MODES: usize = 12;

/// The highest partial the winding layer reaches.
pub(crate) const WINDING_MODES: usize = 6;

/// The winding layer's share of the partial it sits beside, at the first
/// order, before the `1/sqrt(n)` taper.
pub(crate) const WINDING_LEVEL: f64 = 0.12;

/// The winding layer's decay rate, in 1/s. A T60 of 0.6 s.
pub(crate) const WINDING_SIGMA: f64 = 11.5;

/// The longest T60 the relaxed tail may reach, in seconds.
///
/// Without the cap a two-stage decay turns a piano's lowest string, already
/// thirty seconds, into a minute and a half of drone.
pub(crate) const LATE_T60_MAX: f64 = 30.0;

/// How far the tail relaxation has to move before the coefficients are
/// rebuilt.
pub(crate) const BLEND_RESOLUTION: f32 = 0.02;

/// How much of the board's mode-to-mode scatter applies at one frequency.
///
/// Below a couple of hundred hertz the board is a mass, not a set of modes, and
/// its admittance is smooth: a mode grid that fine is not something a real
/// soundboard resolves, and letting it modulate a bass string's partials pulled
/// whole notes off the decay law. The scatter fades in over the octave where a
/// board really does have modes.
pub(crate) fn bridge_weight(frequency: f32) -> f32 {
	let t = ((frequency - 150.0) / 300.0).clamp(0.0, 1.0);
	t * t * (3.0 - 2.0 * t)
}

/// The most a board mode may stretch one partial's decay either way.
///
/// A real board does not hand a string fifty seconds of sustain, and a partial
/// whose admittance happens to fall in a gap between board modes should not
/// outlive the fundamental by an order of magnitude. The clamp keeps the
/// scatter audible without letting it become the note.
pub(crate) const BRIDGE_DAMPING_MIN: f64 = 0.45;
/// See [BRIDGE_DAMPING_MIN].
pub(crate) const BRIDGE_DAMPING_MAX: f64 = 2.5;

/// The highest harmonic that can be a phantom's parent.
///
/// The sum tones of low partials are where the metallic attack lives; a
/// pair of high partials sums past the band the ear attaches to the
/// instrument.
pub(crate) const PHANTOM_ORDER: usize = 12;

/// How many neighbours each parent pairs with.
///
/// The coupling is strongest between adjacent modes and falls off as
/// `0.55^(j-i)`, so three neighbours carry all of it.
pub(crate) const PHANTOM_PAIRS: usize = 3;

/// The highest frequency a phantom is kept at, in hertz.
pub(crate) const PHANTOM_FMAX: f64 = 4200.0;

/// The phantom's share of its parents' geometric mean.
pub(crate) const PHANTOM_LEVEL: f32 = 0.22;

/// The highest harmonic that gets a second polarization.
pub(crate) const HORIZONTAL_MODES: usize = 16;

/// How far the damper must move before the coefficients are worth rebuilding.
///
/// The engine moves the damper every sample and it approaches its target
/// exponentially, so rebuilding on any change at all would put an `exp` and a
/// `sin_cos` per mode on the audio thread for every sample. Half a percent is
/// well below the resolution of the decay it controls.
pub(crate) const DAMPER_RESOLUTION: f32 = 0.005;

/// The fewest samples between two rebuilds of the mode coefficients.
///
/// The damper moves by four parts in a thousand of what is left of its journey
/// on every sample, so it covers a full travel in a few hundred samples and a
/// resolution gate alone would let it rebuild thirty times over that. Capping
/// the rate turns that into eight rebuilds of a hundred and sixty modes, which
/// is a fraction of a millisecond, and the staircase it leaves is far finer
/// than the ear can follow on a decay.
pub(crate) const REBUILD_INTERVAL: u32 = 32;

/// How far the pitch has to move before the coefficients are worth rebuilding.
///
/// A pitch is a multiplication on every mode's frequency, so unlike a damper it
/// cannot be applied to the state without recomputing the transition. One part
/// in a thousand is under two cents, and with the rebuild rate capped below it
/// leaves a staircase at well over a kilohertz, which is far above the five or
/// six hertz a hand shakes at.
pub(crate) const PITCH_RESOLUTION: f32 = 1e-3;

/// The force a mode has to be able to put on the bridge to be worth updating.
///
/// A mode that has decayed below this is skipped entirely. That is not only a
/// saving of twelve floating point operations: a geometric sequence that is left
/// running reaches the denormal range, where every one of those operations drops
/// out of the pipeline into microcode and costs a hundred times as much. A
/// measured mode bank with a live string costs a fifth of a microsecond per
/// sample per string, and the same bank left ringing for a second costs seven,
/// which is the difference between a playable instrument and a stutter.
///
/// A nanonewton is a hundred and twenty decibels below a note. Nothing that
/// sleeps here has ever been audible.
pub(crate) const QUIET_FORCE: f32 = 1e-9;

impl StringConfig {
	/// A convenient constructor for a plain, ideal string.
	pub fn new(frequency: f32) -> Self {
		Self {
			frequency,
			..Default::default()
		}
	}
}
