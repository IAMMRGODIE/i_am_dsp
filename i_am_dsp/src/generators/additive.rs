//! This is the implementation of AdditiveOsc, which is a type of oscillator that adds up multiple sine waves.

use std::{collections::HashMap, f32::consts::PI, sync::{Arc, Mutex}};

use i_am_dsp_derive::Parameters;
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use wide::f32x4;

use crate::{
	prelude::{bend, Oscillator, Parameters},
	tools::interpolate::cubic_interpolate,
};

/// A trait for generating frequency information for AdditiveOsc
pub trait GenFreqInfo: Send + Sync + Parameters {
	/// Returns the ratio and amplitude of the oscillator at the given index
	/// 
	/// the ratio of the frequency must greater than 1.0
	fn gen_info(&mut self, index: usize, total_amount: usize) -> FreqInfo;
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
	fn build<const MAX_SINES: usize>(
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
		let (sine, cosine): (Vec<f32>, Vec<f32>) = spectrum.iter().map(|bin| (bin.im, bin.re)).unzip();

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
struct Wavetable {
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
	fn build<const MAX_SINES: usize>(
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
		let level = (if mantissa == 0 { exponent } else { exponent + 1 }).clamp(0, last as i32) as usize;

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
	fn play(&self, fundamental_phase: f32, max_harmonic: f32) -> (f32, f32) {
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

/// A type of oscillator that adds up multiple sine waves
#[derive(Parameters)]
pub struct AdditiveOsc<
	Freq: GenFreqInfo,
	const MAX_SINES: usize = 64,
	const CHANNELS: usize = 2
> {
	#[sub_param]
	freq_gen: Freq,
	#[skip]
	calculated_ratios: [f32; MAX_SINES],
	#[skip]
	calculated_amplitudes: [f32; MAX_SINES],
	#[skip]
	calculated_dampings: [f32; MAX_SINES],
	#[skip]
	calculated_phases: [f32; MAX_SINES],
	#[skip]
	max_ratio: f32,
	#[skip]
	min_ratio: f32,
	#[skip]
	max_amplitude: f32,
	/// The highest partial ratio that still fits below the Nyquist frequency of the played note.
	///
	/// It stays infinite until a driver tells us the pitch through set_pitch, which leaves the
	/// oscillator unbandlimited for callers that only want the raw sum of the partials.
	#[skip]
	max_harmonic: f32,
	/// One period of the partials, used when every ratio is an integer and nothing is damped.
	///
	/// It is rebuilt by new, recaculate_ratio and rebuild_table, and ignored while num_sines does
	/// not match the number of partials it was built from.
	#[skip]
	table: Option<Wavetable>,
	/// Whether any partial decays over time, which rules out the wavetable and the `powf` shortcut.
	#[skip]
	has_damping: bool,
	/// The number of sine waves to add up
	pub num_sines: usize,
	#[range(min = 0.01, max = 4.0)]
	#[logarithmic]
	/// The gain of the oscillator, which saves in linear scale
	pub gain: f32,
}

impl<
	Freq: GenFreqInfo,
	const MAX_SINES: usize,
	const CHANNELS: usize
> AdditiveOsc<Freq, MAX_SINES, CHANNELS> {
	/// Creates a new AdditiveOsc with the given frequency generator and number of sine waves
	/// 
	/// # Panics
	/// 
	/// 1. If the number of sine waves is greater than `MAX_SINES`
	/// 2. If the number of sine waves is 0
	/// 3. If the number of channels is 0
	pub fn new(mut freq_gen: Freq, num_sines: usize) -> Self {
		assert!(num_sines <= MAX_SINES);
		assert!(MAX_SINES > 0);
		assert!(CHANNELS > 0);
		let index_0 = freq_gen.gen_info(0, MAX_SINES);
		let mut max_ratio = index_0.ratio;
		let mut min_ratio = max_ratio;
		let mut max_amplitude = index_0.amplitude;
		let calculated_ratios: [FreqInfo; MAX_SINES] = core::array::from_fn(|i| {
			let output = freq_gen.gen_info(i, MAX_SINES);
			max_ratio = max_ratio.max(output.ratio);
			min_ratio = min_ratio.min(output.ratio);
			max_amplitude = max_amplitude.max(output.amplitude);
			output
		});
		
		let calculated_amplitudes = core::array::from_fn(|i| {
			calculated_ratios[i].amplitude
		});

		let calculated_dampings = core::array::from_fn(|i| {
			(- calculated_ratios[i].damping).exp()
		});

		let calculated_phases = core::array::from_fn(|i| {
			calculated_ratios[i].phase
		});

		let calculated_ratios = core::array::from_fn(|i| {
			calculated_ratios[i].ratio
		});

		let mut output = Self {
			freq_gen,
			calculated_ratios,
			calculated_amplitudes,
			calculated_dampings,
			calculated_phases,
			max_amplitude,
			max_ratio,
			min_ratio: min_ratio.max(1.0),
			table: None,
			has_damping: false,
			max_harmonic: f32::INFINITY,
			num_sines,
			gain: 1.0
		};
		output.update_damping_cache();
		output.rebuild_table();
		output
	}

	/// Changes the frequency generator of the oscillator
	pub fn change_freq_gen(&mut self, freq_gen: Freq) {
		self.freq_gen = freq_gen;
		self.recaculate_ratio();
	}

	/// Recalculates the ratio and amplitude of the oscillator
	pub fn recaculate_ratio(&mut self) {
		let index_0 = self.freq_gen.gen_info(0, MAX_SINES);
		let mut max_ratio = index_0.ratio;
		let mut min_ratio = max_ratio;
		let mut max_amplitude = index_0.amplitude;

		let calculated_ratios: [FreqInfo; MAX_SINES] = core::array::from_fn(|i| {
			let output = self.freq_gen.gen_info(i, MAX_SINES);
			max_ratio = max_ratio.max(output.ratio);
			min_ratio = min_ratio.min(output.ratio);
			max_amplitude = max_amplitude.max(output.amplitude);
			output
		});
		
		let calculated_amplitudes = core::array::from_fn(|i| {
			calculated_ratios[i].amplitude
		});

		let calculated_dampings = core::array::from_fn(|i| {
			(- calculated_ratios[i].damping).exp()
		});

		let calculated_phases = core::array::from_fn(|i| {
			calculated_ratios[i].phase
		});

		let calculated_ratios = core::array::from_fn(|i| {
			calculated_ratios[i].ratio
		});
		
		self.max_ratio = max_ratio;
		self.min_ratio = min_ratio.max(1.0);
		self.calculated_phases = calculated_phases;
		self.calculated_dampings = calculated_dampings;
		self.calculated_ratios = calculated_ratios;
		self.calculated_amplitudes = calculated_amplitudes;
		self.max_amplitude = max_amplitude;
		self.update_damping_cache();
		self.rebuild_table();
	}

	/// Rebuilds the wavetable the fast path plays from.
	///
	/// Call it after changing num_sines directly. Until it is called the oscillator sums its
	/// partials one by one, which is correct but slower.
	pub fn rebuild_table(&mut self) {
		self.table = if self.has_damping {
			None
		}else {
			Wavetable::build(
				&self.calculated_ratios,
				&self.calculated_amplitudes,
				&self.calculated_phases,
				self.num_sines.min(MAX_SINES),
			)
		};
	}

	/// Reads a group of amplitudes, with the partials above the pitch limit set to zero.
	#[inline]
	fn band_limit(&self, amplitudes: &[f32], ratios: f32x4, max_harmonic: f32x4) -> f32x4 {
		ratios.simd_gt(max_harmonic).select(f32x4::ZERO, f32x4::from(amplitudes))
	}

	/// Caches whether any partial decays over time, which costs a `powf` per partial per sample.
	fn update_damping_cache(&mut self) {
		self.has_damping = self.calculated_dampings.iter().any(|damping| *damping != UNDAMPED_FACTOR);
	}

	/// Applies a per channel phase offset to the summed sine and cosine of all partials.
	#[inline]
	fn apply_channel_phase(&self, sine: f32, cosine: f32, phase: [f32; CHANNELS]) -> [f32; CHANNELS] {
		let mut output = [0.0; CHANNELS];

		for (output, phase) in output.iter_mut().zip(phase) {
			let angle = 2.0 * PI * phase;
			let (sin, cos) = angle.sin_cos();
			*output = (sine * cos + cosine * sin) * self.gain;
		}

		output
	}

	/// Plays every partial from the wavetable, which already holds their sum.
	#[inline]
	fn play_table(&self, table: &Wavetable, frequency: f32, time: f32, phase: [f32; CHANNELS]) -> [f32; CHANNELS] {
		let (sine, cosine) = table.play(time * frequency, self.max_harmonic);

		self.apply_channel_phase(sine, cosine, phase)
	}

	/// Sums the partials one sine at a time.
	///
	/// This is the fallback for partial sets that are not one period long. The sine and cosine of a
	/// partial are only computed once for all channels: a per channel phase is an angle offset, so
	/// it can be applied to the two sums afterwards.
	fn play_sines(&self, frequency: f32, time: f32, phase: [f32; CHANNELS]) -> [f32; CHANNELS] {
		let len = self.num_sines.min(MAX_SINES).min(self.calculated_ratios.len());
		let max_harmonic = f32x4::splat(self.max_harmonic);

		if self.has_damping {
			let mut output = [0.0; CHANNELS];

			for (i, output) in output.iter_mut().enumerate() {
				for j in (0..len).step_by(4) {
					let end = (j + 4).min(len);
					let ratio = f32x4::from(&self.calculated_ratios[j..end]);
					let freq = ratio * frequency;
					let phase_freq = f32x4::from(&self.calculated_phases[j..end]);
					let t = time * freq + phase[i] + phase_freq;
					let amp = self.band_limit(&self.calculated_amplitudes[j..end], ratio, max_harmonic) * self.gain;
					let damping = f32x4::from(&self.calculated_dampings[j..end]).powf_simd(t);

					*output += ((t * 2.0 * PI).sin() * amp * damping).reduce_add();
				}
			}

			return output;
		}

		let mut sum_sine = f32x4::ZERO;
		let mut sum_cosine = f32x4::ZERO;

		for j in (0..len).step_by(4) {
			let end = (j + 4).min(len);
			let ratio = f32x4::from(&self.calculated_ratios[j..end]);
			let freq = ratio * frequency;
			let phase_freq = f32x4::from(&self.calculated_phases[j..end]);
			let t = (time * freq + phase_freq) * f32x4::splat(2.0 * PI);
			let amp = self.band_limit(&self.calculated_amplitudes[j..end], ratio, max_harmonic);
			let (sin, cos) = t.sin_cos();

			sum_sine = sin.mul_add(amp, sum_sine);
			sum_cosine = cos.mul_add(amp, sum_cosine);
		}

		self.apply_channel_phase(sum_sine.reduce_add(), sum_cosine.reduce_add(), phase)
	}
}

impl<
	Freq: GenFreqInfo,
	const MAX_SINES: usize,
	const CHANNELS: usize
> Oscillator<CHANNELS> for AdditiveOsc<Freq, MAX_SINES, CHANNELS> {
	fn play_at(&self, frequency: f32, time: f32, phase: [f32; CHANNELS]) -> [f32; CHANNELS] {
		let num_sines = self.num_sines.min(MAX_SINES);

		if let Some(table) = self.table.as_ref()
			&& table.num_sines == num_sines
		{
			return self.play_table(table, frequency, time, phase);
		}

		self.play_sines(frequency, time, phase)
	}

	fn set_pitch(&mut self, frequency: f32, sample_rate: usize) {
		// Everything above this ratio folds back into the audible band at this pitch.
		self.max_harmonic = if frequency > 0.0 && sample_rate > 0 {
			sample_rate as f32 / (2.0 * frequency)
		}else {
			f32::INFINITY
		};
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		use egui::*;
		use crate::tools::ui_tools::gain_ui;
		use egui::emath::RectTransform;

		Resize::default()
			.max_width(ui.available_width())
			.min_width(ui.available_width())
			.id_salt(format!("{}_additive_osc_ratio", id_prefix))
			.show(ui, |ui| 
		{
			Frame::canvas(ui.style()).show(ui, |ui| {
				let (_, rect) = ui.allocate_space(ui.available_size());
				let to_screen = RectTransform::from_to(
					Rect::from_x_y_ranges(0.0..=1.0, -1.0..=1.0), 
					rect
				);

				if MAX_SINES == 1 {
					let y = self.calculated_amplitudes[0];
					let pos_1 = to_screen * pos2(0.5, y);
					let pos_2 = to_screen * pos2(0.5, 0.0);
					ui.painter().line_segment([pos_1, pos_2], (3.0, Color32::WHITE));
					return;
				}

				let mid_ratio = (self.max_ratio * self.min_ratio).sqrt();
				let min_ratio = self.min_ratio.min(mid_ratio / 2.0);
				let max_ratio = self.max_ratio.max(mid_ratio * 2.0);
				let max_amp = self.max_amplitude;

				ui.painter().extend(self.calculated_ratios
					.iter().zip(self.calculated_amplitudes.iter())
					.take(self.num_sines)
					.map(|(ratio, amplitude)| 
				{
					let lerped_ratio = (ratio.ln() - min_ratio.ln()) / (max_ratio.ln() - min_ratio.ln());
					let x = lerped_ratio * 0.99 + 0.005;
					let y = - amplitude / max_amp;
					Shape::line_segment([
						to_screen * pos2(x, y),
						to_screen * pos2(x, 0.0),
					], (3.0, Color32::WHITE))
				}));
			});
		});

		if self.freq_gen.demo_ui(ui, format!("{}_freq_gen", id_prefix)) {
			self.recaculate_ratio();
		}

		ui.horizontal(|ui| {
			ui.label(format!("Max Ratio: {:.2}", self.max_ratio));
			ui.label(format!("Min Ratio: {:.2}", self.min_ratio));
			let num_sines = self.num_sines;
			ui.add(Slider::new(&mut self.num_sines, 1..=MAX_SINES).text("Num Sines"));
			if self.num_sines != num_sines {
				self.rebuild_table();
			}
			gain_ui(ui, &mut self.gain, None, false);
		});
	}
}

/// A Simple Bend Frequency Generator
#[derive(Debug, Clone, PartialEq)]
#[derive(Parameters)]
pub struct BendedSawGen {
	/// How much the frequency should be bent
	/// 
	/// default to 0.0
	#[range(min = -10.0, max = 10.0)]
	pub bend_amount: f32,
	/// must be greater than 0.0
	/// 
	/// default to 1.0
	pub center_space: f32,
	/// How much the bend should be shifted
	/// 
	/// default to 0.0
	#[range(min = -10.0, max = 10.0)]
	pub total_bend_amount: f32,
	/// How much the frequency should be scaled
	/// 
	/// default to 1.0
	pub scale_amount: f32,
}

impl Default for BendedSawGen {
	fn default() -> Self {
		Self {
			bend_amount: 0.0,
			center_space: 1.0,
			total_bend_amount: 0.0,
			scale_amount: 1.0,
		}
	}
}

impl GenFreqInfo for BendedSawGen {
	fn gen_info(&mut self, index: usize, total_amount: usize) -> FreqInfo {
		if total_amount == 0 {
			return Default::default();
		}else if total_amount == 1 {
			return FreqInfo {
				ratio: 1.0,
				amplitude: 1.0,
				damping: 0.0,
				phase: 0.0,
			};
		}

		let index = index + 1;
		let index = index as f32;
		let ratio = index * self.scale_amount;
		let ratio_ln = ratio.ln();
		let total_ln = (total_amount as f32).ln();
		let bend_trunc = (ratio_ln / self.center_space).trunc();
		let bent_fract = (ratio_ln / self.center_space).fract();
		let bended = bend(bent_fract, self.bend_amount);
		let ratio = bend_trunc * self.center_space + bended;
		let t = ratio / total_ln;
		let t = bend(t, self.total_bend_amount);
		let ratio = (t * total_ln).exp();
		let amplitude = (2.0 / PI) * (- 1.0_f32).powf(index) / (index + 1.0);

		FreqInfo {
			ratio,
			amplitude,
			damping: 0.0,
			phase: 0.0,
		}
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) -> bool {
		use egui::*;

		let mut change = self.clone();

		Grid::new(format!("{}_bended_saw_gen", id_prefix))
			.num_columns(4)
			.show(ui, |ui| 
		{
			ui.label("Bend Amount");
			ui.add(Slider::new(&mut change.bend_amount, -10.0..=10.0));
			// ui.end_row();

			ui.label("Center Space");
			ui.add(Slider::new(&mut change.center_space, 0.125..=8.0).logarithmic(true));
			ui.end_row();

			ui.label("Total Bend Amount");
			ui.add(Slider::new(&mut change.total_bend_amount, -10.0..=10.0));
			// ui.end_row();

			ui.label("Scale Amount");
			ui.add(Slider::new(&mut change.scale_amount, 0.01..=1.0));
			ui.end_row();
		});

		if change != *self {
			*self = change;
			true
		}else {
			false
		}

	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::prelude::{Parameter, SetValue};

	const SINES: usize = 16;

	/// A frequency generator that can be asked for the partials a wavetable cannot hold.
	struct TestGen {
		ratio_scale: f32,
		damping: f32,
		phase_scale: f32,
	}

	impl TestGen {
		/// Integer ratios, no damping and no phase, which is what the fast path is for.
		fn clean() -> Self {
			Self {
				ratio_scale: 1.0,
				damping: 0.0,
				phase_scale: 0.0,
			}
		}
	}

	impl Parameters for TestGen {
		fn get_parameters(&self) -> Vec<Parameter> {
			Vec::new()
		}

		fn set_parameter(&mut self, _: &str, _: SetValue) -> bool {
			false
		}
	}

	impl GenFreqInfo for TestGen {
		fn gen_info(&mut self, index: usize, _: usize) -> FreqInfo {
			let index = (index + 1) as f32;

			FreqInfo {
				ratio: self.ratio_scale * index,
				amplitude: 1.0 / index,
				damping: self.damping,
				phase: self.phase_scale * index,
			}
		}

		#[cfg(feature = "real_time_demo")]
		fn demo_ui(&mut self, _: &mut egui::Ui, _: String) -> bool {
			false
		}
	}

	/// The exact sum of all partials in f64, so that an oscillator can be compared against it.
	fn reference<Freq: GenFreqInfo, const MAX_SINES: usize, const CHANNELS: usize>(
		osc: &AdditiveOsc<Freq, MAX_SINES, CHANNELS>,
		frequency: f32,
		time: f64,
		phase: [f32; CHANNELS],
		limit: f64,
	) -> [f64; CHANNELS] {
		let mut output = [0.0f64; CHANNELS];

		for (i, output) in output.iter_mut().enumerate() {
			for j in 0..osc.num_sines.min(MAX_SINES) {
				// The oscillator drops the partials above the Nyquist frequency of the played note.
				if (osc.calculated_ratios[j] as f64).round() > limit {
					continue;
				}

				let cycles = time * osc.calculated_ratios[j] as f64 * frequency as f64
					+ phase[i] as f64
					+ osc.calculated_phases[j] as f64;
				let damping = (osc.calculated_dampings[j] as f64).powf(cycles);

				*output += (2.0 * std::f64::consts::PI * cycles).sin()
					* osc.calculated_amplitudes[j] as f64
					* damping
					* osc.gain as f64;
			}
		}

		output
	}

	fn max_error<Freq: GenFreqInfo, const MAX_SINES: usize, const CHANNELS: usize>(
		osc: &AdditiveOsc<Freq, MAX_SINES, CHANNELS>,
		phase: [f32; CHANNELS],
	) -> f64 {
		let mut max_error = 0.0f64;

		for step in 0..=1024 {
			let time = step as f32 / 1024.0;
			let output = osc.play_at(1.0, time, phase);
			let reference = reference(osc, 1.0, time as f64, phase, osc.max_harmonic as f64);

			for (output, reference) in output.iter().zip(reference) {
				max_error = max_error.max((*output as f64 - reference).abs());
			}
		}

		max_error
	}

	#[test]
	fn the_selected_levels_never_alias() {
		let osc: AdditiveOsc<BendedSawGen, SINES, 2> = AdditiveOsc::new(BendedSawGen::default(), SINES);
		let table = osc.table.as_ref().unwrap();

		// Below a limit of one the oscillator plays nothing at all, so only the rest is a level
		for step in 50..=4000 {
			let max_harmonic = step as f32 * 0.02;
			let (fine, coarse, blend) = table.select(max_harmonic);
			let limit = |level: usize| table.max_ratio / 2.0f32.powi(level as i32);

			assert!(limit(fine) <= max_harmonic, "level {fine} keeps partials above {max_harmonic}");
			assert!(limit(coarse) <= max_harmonic, "level {coarse} keeps partials above {max_harmonic}");
			assert!(fine <= coarse);
			assert!((0.0..=1.0).contains(&blend));
		}
	}

	#[test]
	fn the_wavetable_is_band_limited_to_the_played_pitch() {
		let mut osc: AdditiveOsc<BendedSawGen, SINES, 2> = AdditiveOsc::new(BendedSawGen::default(), SINES);

		for pitch in [12000.0f32, 6000.0, 3000.0, 1500.0, 750.0, 375.0, 187.5, 93.75] {
			osc.set_pitch(pitch, 48000);

			let table = osc.table.as_ref().unwrap();
			let (fine, coarse, blend) = table.select(osc.max_harmonic);
			let limit = |level: usize| (table.max_ratio / 2.0f32.powi(level as i32)) as f64;

			for step in 0..=64 {
				let time = step as f32 / 64.0;
				let output = osc.play_at(1.0, time, [0.0, 0.25]);
				let fine_reference = reference(&osc, 1.0, time as f64, [0.0, 0.25], limit(fine));
				let coarse_reference = reference(&osc, 1.0, time as f64, [0.0, 0.25], limit(coarse));

				for (i, output) in output.iter().enumerate() {
					let expected = fine_reference[i] * (1.0 - blend as f64) + coarse_reference[i] * blend as f64;
					let error = (*output as f64 - expected).abs();

					assert!(error < 5e-5, "pitch {pitch} time {time} error {error:e}");
				}
			}
		}
	}

	#[test]
	fn a_pitch_above_nyquist_plays_nothing() {
		let mut osc: AdditiveOsc<BendedSawGen, SINES, 2> = AdditiveOsc::new(BendedSawGen::default(), SINES);
		osc.set_pitch(30000.0, 48000);

		assert!(osc.max_harmonic < 1.0);
		assert_eq!(osc.play_at(1.0, 0.31, [0.0, 0.25]), [0.0, 0.0]);
	}

	#[test]
	fn the_sine_fallback_drops_the_partials_above_nyquist() {
		let mut osc: AdditiveOsc<BendedSawGen, SINES, 2> = AdditiveOsc::new(BendedSawGen::default(), SINES);
		// a table built for another number of partials is ignored, which forces the sine path
		osc.num_sines = SINES / 2;
		osc.set_pitch(3000.0, 48000);

		assert_eq!(osc.max_harmonic, 8.0);
		assert!(max_error(&osc, [0.0, 0.25]) < 1e-5);
	}

	#[test]
	fn integer_partials_are_played_from_the_wavetable() {
		let osc: AdditiveOsc<BendedSawGen, SINES, 2> = AdditiveOsc::new(BendedSawGen::default(), SINES);

		assert!(osc.table.is_some(), "integer ratios should build a wavetable");
		assert!(!osc.has_damping);
	}

	#[test]
	fn wavetable_matches_the_exact_sum() {
		let osc: AdditiveOsc<BendedSawGen, SINES, 2> = AdditiveOsc::new(BendedSawGen::default(), SINES);

		assert!(max_error(&osc, [0.0, 0.25]) < 5e-5);
	}

	#[test]
	fn phased_partials_are_played_from_the_wavetable() {
		let osc: AdditiveOsc<TestGen, SINES, 2> = AdditiveOsc::new(
			TestGen { phase_scale: 0.13, ..TestGen::clean() },
			SINES
		);

		assert!(osc.table.is_some());
		assert!(max_error(&osc, [0.0, 0.25]) < 5e-5);
	}

	#[test]
	fn non_integer_partials_fall_back_to_sines() {
		let osc: AdditiveOsc<TestGen, SINES, 2> = AdditiveOsc::new(
			TestGen { ratio_scale: 1.37, ..TestGen::clean() },
			SINES
		);

		assert!(osc.table.is_none(), "non integer ratios cannot use a wavetable");
		assert!(max_error(&osc, [0.0, 0.25]) < 5e-5);
	}

	#[test]
	fn damped_partials_fall_back_to_sines() {
		let osc: AdditiveOsc<TestGen, SINES, 2> = AdditiveOsc::new(
			TestGen { damping: 0.5, ..TestGen::clean() },
			SINES
		);

		assert!(osc.table.is_none(), "damping is not periodic, so it cannot use a wavetable");
		assert!(osc.has_damping);
		assert!(max_error(&osc, [0.0, 0.0]) < 5e-5);
	}

	#[test]
	fn a_changed_num_sines_still_plays_every_partial() {
		let mut osc: AdditiveOsc<BendedSawGen, SINES, 2> = AdditiveOsc::new(BendedSawGen::default(), SINES);
		osc.num_sines = SINES / 2;

		assert!(max_error(&osc, [0.0, 0.5]) < 5e-5);

		osc.rebuild_table();
		assert_eq!(osc.table.as_ref().map(|table| table.num_sines), Some(SINES / 2));
		assert!(max_error(&osc, [0.0, 0.5]) < 5e-5);
	}

	#[test]
	fn rebuilding_the_table_after_the_ratios_change_keeps_it_usable() {
		let mut osc: AdditiveOsc<BendedSawGen, SINES, 2> = AdditiveOsc::new(BendedSawGen::default(), SINES);
		osc.freq_gen.scale_amount = 2.0;
		osc.recaculate_ratio();

		assert!(osc.table.is_some());
		assert!((osc.max_ratio - (SINES * 2) as f32).abs() < 1e-3);
		assert!(max_error(&osc, [0.0, 0.25]) < 5e-5);
	}
}
