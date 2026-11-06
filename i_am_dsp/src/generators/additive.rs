//! This is the implementation of AdditiveOsc, which is a type of oscillator that adds up multiple sine waves.

use std::{
	collections::HashMap,
	f32::consts::PI,
	sync::{Arc, Mutex},
};

use rustfft::{Fft, FftPlanner, num_complex::Complex};

use crate::{prelude::Parameters, tools::interpolate::cubic_interpolate};

/// A trait for generating frequency information for AdditiveOsc
pub trait GenFreqInfo: Send + Sync + Parameters {
	/// Returns the ratio and amplitude of the oscillator at the given index
	///
	/// the ratio of the frequency must greater than 1.0
	fn gen_info(&mut self, index: usize, total_amount: usize) -> FreqInfo;

	/// A value that changes whenever the generated partials would change.
	///
	/// The oscillator asks for this once per sample and regenerates its partial
	/// table when it moves. A generator that returns 0 keeps whatever it built at
	/// construction, which is right for one whose controls are not live.
	fn revision(&self) -> u64 {
		0
	}
	#[cfg(feature = "real_time_demo")]
	/// Returns true if ratio should be updated in real-time demo
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) -> bool;
}

/// Frequency information for AdditiveOsc
pub struct FreqInfo {
	/// The ratio relative to the base frequency
	pub ratio: f32,
	/// The amplitude of the oscillator
	pub amplitude: f32,
	/// The damping factor of current frequency
	///
	/// 0.0 means no damping.
	///
	/// The damping function we use is `exp(-t * damping)`
	pub damping: f32,
	/// the phase of the oscillator, in [0.0, 1.0]
	pub phase: f32,
}

impl Default for FreqInfo {
	fn default() -> Self {
		Self {
			ratio: 1.0,
			amplitude: 0.0,
			damping: 0.0,
			phase: 0.0,
		}
	}
}

/// How far a partial ratio may be from an integer and still be played by the wavetable fast path.
///
/// A ratio has to be an integer for the sum of all partials to repeat once per cycle of the
/// fundamental, which is what makes a single period of the sum enough to describe all of it.
const INTEGER_RATIO_TOLERANCE: f32 = 1e-4;

/// How many wavetable entries are spent per period of the highest partial.
const TABLE_OVERSAMPLING: f32 = 32.0;

/// The smallest wavetable the fast path uses, which still oversamples its highest partial.
const MIN_TABLE_SIZE: usize = 256;

/// The largest wavetable the fast path uses.
const MAX_TABLE_SIZE: usize = 16384;

/// An undamped partial is one whose per sample damping factor is exactly one.
const UNDAMPED_FACTOR: f32 = 1.0;

/// The inverse FFT plans the wavetable build uses, one per table size.
///
/// A plan is not cheap to build and every level of every wavetable needs one, so they are shared
/// for the lifetime of the process.
fn inverse_fft(size: usize) -> Arc<dyn Fft<f32>> {
	lazy_static::lazy_static! {
		static ref PLANS: Mutex<HashMap<usize, Arc<dyn Fft<f32>>>> = Mutex::new(HashMap::new());
	}

	let mut plans = PLANS.lock().unwrap_or_else(|error| error.into_inner());

	plans
		.entry(size)
		.or_insert_with(|| FftPlanner::new().plan_fft_inverse(size))
		.clone()
}

/// One band limited version of the sum of all partials.
///
/// A level keeps every partial up to a ratio that halves from level to level, which is what lets
/// the oscillator drop the partials that would alias at the pitch it is playing.
struct WavetableLevel {
	/// The number of entries per period, always a power of two.
	size: usize,
	/// One less than the size, used to wrap table indices around one period.
	mask: usize,
	/// The sum of every partial sine over one period of the fundamental.
	sine: Box<[f32]>,
	/// The same sum with cosines, which holds every partial a quarter period ahead.
	cosine: Box<[f32]>,
}

impl WavetableLevel {
	/// Builds the level that keeps every partial up to the given ratio.
	///
	/// The partials are transformed back from their own spectrum, one entry per partial, which is
	/// much cheaper than adding every entry of the table once per partial.
	pub(crate) fn build<const MAX_SINES: usize>(
		ratios: &[f32; MAX_SINES],
		amplitudes: &[f32; MAX_SINES],
		phases: &[f32; MAX_SINES],
		num_sines: usize,
		limit: f32,
		size: usize,
	) -> Self {
		let mut spectrum = vec![Complex::<f32>::ZERO; size];

		for j in 0..num_sines {
			let ratio = ratios[j].round();
			let bin = ratio as usize;

			// The ratios are integers here, so a partial lands exactly on one bin of the table.
			if ratio > limit || bin * 2 >= size {
				continue;
			}

			let angle = 2.0 * PI * phases[j];
			spectrum[bin] += Complex::new(amplitudes[j] * angle.cos(), amplitudes[j] * angle.sin());
		}

		inverse_fft(size).process(&mut spectrum);

		// The inverse transform of positive frequencies alone is complex: its imaginary part is the
		// sum of the sines and its real part the sum of the cosines.
		let (sine, cosine): (Vec<f32>, Vec<f32>) =
			spectrum.iter().map(|bin| (bin.im, bin.re)).unzip();

		Self {
			size,
			mask: size - 1,
			sine: sine.into_boxed_slice(),
			cosine: cosine.into_boxed_slice(),
		}
	}

	/// Reads one table at the given position measured in entries, with a cubic interpolation.
	#[inline]
	fn read(table: &[f32], position: f32, mask: usize) -> f32 {
		let index = position as usize;
		let before = table[index.wrapping_sub(1) & mask];
		let now = table[index & mask];
		let next = table[(index + 1) & mask];
		let after = table[(index + 2) & mask];

		cubic_interpolate(position - index as f32, [before, now, next, after])
	}

	/// Reads the sum of the partials this level keeps, at the given phase of the fundamental.
	#[inline]
	fn play(&self, fundamental_phase: f32) -> (f32, f32) {
		let position = (fundamental_phase - fundamental_phase.floor()) * self.size as f32;

		(
			Self::read(&self.sine, position, self.mask),
			Self::read(&self.cosine, position, self.mask),
		)
	}
}

/// One period of the sum of all partials, cut into band limited levels.
///
/// Both tables of a level are indexed by the phase of the fundamental, so playing a partial from
/// them is a single interpolated lookup instead of one sine and one powf per partial per sample.
/// The cosine table exists so that a per channel phase offset can be applied with the angle
/// addition rule instead of rebuilding the tables for every phase.
pub(crate) struct Wavetable {
	/// The levels, from the one that holds every partial to the one that holds the fundamental.
	levels: Box<[WavetableLevel]>,
	/// The number of partials the levels were built from.
	num_sines: usize,
	/// The largest partial ratio the first level holds.
	max_ratio: f32,
}

impl Wavetable {
	/// The number of entries a level with the given highest partial uses.
	fn size_of(limit: f32) -> usize {
		((limit * TABLE_OVERSAMPLING) as usize)
			.next_power_of_two()
			.clamp(MIN_TABLE_SIZE, MAX_TABLE_SIZE)
	}

	/// Builds one band limited level per octave, from the first num_sines partials.
	///
	/// Returns None when the partials cannot be described by a single period, which happens when a
	/// ratio is not an integer or when the highest partial does not fit in the largest table. The
	/// caller then has to fall back to summing sines directly.
	pub(crate) fn build<const MAX_SINES: usize>(
		ratios: &[f32; MAX_SINES],
		amplitudes: &[f32; MAX_SINES],
		phases: &[f32; MAX_SINES],
		num_sines: usize,
	) -> Option<Self> {
		if num_sines == 0 || num_sines > MAX_SINES {
			return None;
		}

		let mut max_ratio = 1.0f32;
		for ratio in ratios.iter().take(num_sines) {
			let nearest = ratio.round();
			if !ratio.is_finite()
				|| nearest < 1.0
				|| (ratio - nearest).abs() > INTEGER_RATIO_TOLERANCE * nearest
			{
				return None;
			}
			max_ratio = max_ratio.max(nearest);
		}

		// Partials above the Nyquist of the table would fold back onto the lower ones.
		if max_ratio * 2.0 > Self::size_of(max_ratio) as f32 {
			return None;
		}

		let mut levels = Vec::new();
		let mut limit = max_ratio;

		loop {
			levels.push(WavetableLevel::build(
				ratios,
				amplitudes,
				phases,
				num_sines,
				limit,
				Self::size_of(limit),
			));

			if limit < 2.0 {
				break;
			}
			limit = (limit / 2.0).max(1.0);
		}

		Some(Self {
			levels: levels.into_boxed_slice(),
			num_sines,
			max_ratio,
		})
	}

	/// Picks the pair of levels that keeps every partial below the given ratio, and how much of the
	/// coarser one to mix in.
	///
	/// Both levels of the pair are conservative: their highest partial is never above the limit, so
	/// the result never aliases. The mix starts out on the finer level and reaches the coarser one
	/// just as the limit drops to where the next level down becomes the finer one, so the
	/// transition between two levels stays continuous.
	#[inline]
	fn select(&self, max_harmonic: f32) -> (usize, usize, f32) {
		let last = self.levels.len() - 1;

		// Every partial fits below the limit, which also covers a limit that is not a number at all.
		if max_harmonic.is_nan() || max_harmonic >= self.max_ratio {
			return (0, 0, 0.0);
		}

		let bits = (self.max_ratio / max_harmonic).to_bits();
		let exponent = ((bits >> 23) & 0xff) as i32 - 127;
		let mantissa = bits & 0x007f_ffff;
		// The ceiling of the base two logarithm, exactly, from its exponent and its fraction.
		let level = (if mantissa == 0 {
			exponent
		} else {
			exponent + 1
		})
		.clamp(0, last as i32) as usize;

		if level >= last {
			return (last, last, 0.0);
		}

		// The mantissa is the linear part of the same logarithm, which is all the shape of the
		// crossfade needs and avoids a call into the math library. Cubing the fade keeps the level
		// at full brightness for most of its range and only hands over to the coarser one close to
		// the ratio where the finer one would start to alias.
		let log2 = f32::from_bits(mantissa | 0x3f80_0000) - 1.0 + exponent as f32;
		let fade = (1.0 + log2 - level as f32).clamp(0.0, 1.0);

		(level, level + 1, fade * fade * fade)
	}

	/// Reads the sum of the partials below the given ratio, at the given phase of the fundamental.
	#[inline]
	pub(crate) fn play(&self, fundamental_phase: f32, max_harmonic: f32) -> (f32, f32) {
		// A note above the Nyquist frequency has no partial left to play.
		if max_harmonic < 1.0 {
			return (0.0, 0.0);
		}

		let (level, coarser, blend) = self.select(max_harmonic);
		let (sine, cosine) = self.levels[level].play(fundamental_phase);

		if blend <= 0.0 {
			return (sine, cosine);
		}

		let (coarse_sine, coarse_cosine) = self.levels[coarser].play(fundamental_phase);

		(
			sine + (coarse_sine - sine) * blend,
			cosine + (coarse_cosine - cosine) * blend,
		)
	}
}

mod bended;
mod drawn;
mod osc;
mod spectral;

pub use bended::BendedSawGen;
pub use drawn::{
	DRAWN_PARTIALS, DrawnSpectrum, PageBank, SpectralFeed, SpectralFrame, SpectrumRenderer,
	TableBankOsc,
};
pub use osc::AdditiveOsc;
pub use spectral::SpectralGen;
