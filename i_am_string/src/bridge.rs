//! The driving-point admittance of the bridge, which is where a string's
//! energy actually leaves.
//!
//! A string does not end at a rigid wall. What terminates it is a soundboard
//! whose mechanical impedance falls with frequency in a rough, mode-dense way,
//! so the termination reflects some partials better than others: a partial that
//! happens to sit on a board resonance drains away several times faster than its
//! neighbours, and the reactive part of the admittance pulls it a little off its
//! harmonic position.
//!
//! This is the model the coopiano piano uses, reduced to what a string engine
//! needs: a deterministic modal sum with plate-like modal density, loss factor
//! 0.03, Rayleigh-distributed mode shapes and a mobility that falls above a few
//! hundred hertz as the board's mass takes over. The mean of the real part is
//! removed from the decay it injects, because the fitted T60 law already carries
//! the average.
//!
//! It is built once, at 4 Hz resolution, and interpolated.

use std::sync::OnceLock;

/// The resolution of the table, in hertz.
const STEP: usize = 4;
/// The highest frequency the table covers, in hertz.
pub const F_MAX: f32 = 14_000.0;
/// The board's mean mobility over 150-300 Hz, in seconds per kilogram.
const MEAN_MOBILITY: f32 = 1.5e-3;

struct Admittance {
	real: Box<[f32]>,
	imag: Box<[f32]>,
	/// The real part relative to its own smoothed trend: the mode-to-mode
	/// scatter with the average frequency dependence taken out, which the
	/// fitted T60 law already carries.
	ratio: Box<[f32]>,
}

/// A small deterministic generator, so the table is the same on every run.
struct Lcg(u64);

impl Lcg {
	fn next(&mut self) -> f32 {
		self.0 = self
			.0
			.wrapping_mul(6364136223846793005)
			.wrapping_add(1442695040888963407);
		((self.0 >> 33) as f32) / ((1u64 << 31) as f32)
	}
}

fn build() -> Admittance {
	let mut rng = Lcg(0x6272_6964_6765_5a01);
	// One mode per roughly 15 Hz, a little irregular, up to above the top of
	// the table.
	let mut modes: Vec<(f32, f32)> = Vec::new();
	let mut f = 28.0f32;
	while f < F_MAX * 1.2 {
		let shape = -(rng.next().max(1e-6)).ln(); // Rayleigh power
		modes.push((f, shape));
		f += 15.0 * (0.6 + 0.8 * rng.next());
	}

	let n = (F_MAX as usize) / STEP + 2;
	let mut real = vec![0.0f32; n];
	let mut imag = vec![0.0f32; n];
	for (index, value) in real.iter_mut().enumerate() {
		let hz = (index * STEP) as f32;
		if hz < 1.0 {
			continue;
		}
		let w = std::f32::consts::TAU * hz;
		let (mut re, mut im) = (0.0f32, 0.0f32);
		for &(fm, shape) in &modes {
			let wm = std::f32::consts::TAU * fm;
			let sigma = std::f32::consts::PI * 0.03 * fm;
			// j w shape / (wm^2 - w^2 + j 2 sigma w)
			let den_re = wm * wm - w * w;
			let den_im = 2.0 * sigma * w;
			let den = den_re * den_re + den_im * den_im;
			// multiply numerator (0, w*shape) by conj(den)
			let num_re = w * shape * den_im;
			let num_im = w * shape * den_re;
			re += num_re / den;
			im += num_im / den;
		}
		// The board stops moving with the string above a few hundred hertz and
		// its mass impedance takes over.
		let trend = 1.0 / (1.0 + hz / 400.0);
		*value = re * trend;
		imag[index] = im * trend;
	}

	// Normalise the mobility over the band the reference uses.
	let lo = 150 / STEP;
	let hi = 300 / STEP;
	let mean: f32 = real[lo..=hi].iter().sum::<f32>() / (hi - lo + 1) as f32;
	let scale = MEAN_MOBILITY / mean.max(1e-12);
	for value in real.iter_mut() {
		*value *= scale;
	}
	for value in imag.iter_mut() {
		*value *= scale;
	}

	// The scatter around a local trend. Smoothing over a couple of hundred
	// hertz leaves a couple of board modes in the average, which is what makes
	// the ratio describe a resonance rather than the board's overall tilt.
	let span = 200 / STEP;
	let mut ratio = vec![1.0f32; n];
	for index in 0..n {
		let lo = index.saturating_sub(span);
		let hi = (index + span).min(n - 1);
		let average: f32 = real[lo..=hi].iter().sum::<f32>() / (hi - lo + 1) as f32;
		ratio[index] = (real[index] / average.max(1e-12)).clamp(0.25, 4.0);
	}

	Admittance {
		real: real.into_boxed_slice(),
		imag: imag.into_boxed_slice(),
		ratio: ratio.into_boxed_slice(),
	}
}

fn table() -> &'static Admittance {
	static TABLE: OnceLock<Admittance> = OnceLock::new();
	TABLE.get_or_init(build)
}

/// The real and imaginary parts of the bridge mobility at one frequency.
pub fn admittance(frequency: f32) -> (f32, f32) {
	let table = table();
	let position = (frequency.max(0.0) / STEP as f32).min((table.real.len() - 2) as f32);
	let index = position as usize;
	let fraction = position - index as f32;
	let re = table.real[index] + (table.real[index + 1] - table.real[index]) * fraction;
	let im = table.imag[index] + (table.imag[index + 1] - table.imag[index]) * fraction;
	(re, im)
}

/// How much faster a partial at this frequency decays than its neighbours.
///
/// Around one for an average partial, up to four for one that sits on a board
/// resonance and down to a quarter for one in a gap between resonances. The
/// board's overall rise with frequency is not in here: the fitted T60 law
/// already carries it, so only the scatter is wanted.
pub fn damping_ratio(frequency: f32) -> f32 {
	let table = table();
	let position = (frequency.max(0.0) / STEP as f32).min((table.ratio.len() - 2) as f32);
	let index = position as usize;
	let fraction = position - index as f32;
	table.ratio[index] + (table.ratio[index + 1] - table.ratio[index]) * fraction
}

/// How far the reactive part of the bridge pulls a partial off its harmonic
/// position, in hertz.
///
/// The reflection at the bridge carries a phase of -2 Z Im(Y), which moves the
/// mode by f0 Z Im(Y) / pi.
pub fn detune(f0: f32, impedance: f32, partial: f32) -> f32 {
	let (_, im) = admittance(partial);
	f0 * impedance * im / std::f32::consts::PI
}
