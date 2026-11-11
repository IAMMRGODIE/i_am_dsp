//! Fitting one recorded note to the mode table.
//!
//! The shipped tables were made by `gentleman-modal/`, a numpy script that
//! peak-picks a zero padded spectrum, measures every partial's two decays by
//! demodulating the recording, and then places every partial's amplitude and
//! phase with one least squares over the clip. This is that pipeline in the
//! crate, so a sample can be fitted where it is loaded instead of in a batch
//! tool.
//!
//! # Why the least squares is affordable here
//!
//! A note of a few seconds is a hundred thousand samples, and forty eight
//! partials with a prompt and a tail each make a hundred and ninety two
//! columns. Built the obvious way, the normal equations cost one multiply per
//! sample and per pair of columns, which is billions of operations for a single
//! note.
//!
//! Both basis functions are damped sinusoids, so the product of any two of them
//! is a damped sinusoid as well:
//!
//! ```text
//! e^(-s1 t) e^(i w1 t) * e^(-s2 t) e^(i w2 t) = e^(-(s1+s2) t) e^(i (w1+w2) t)
//! ```
//!
//! Sampled at a fixed rate that is a geometric series, and a geometric series
//! has a closed form. The Gram matrix therefore costs nothing per sample, and
//! only the right hand side still has to touch the recording, once per column.
//! The fit is the same fit the script computes; it is just a few hundred times
//! cheaper, which is what makes it usable inside a plug-in.

use crate::fitted::Mode;

/// The highest partial the fitter keeps, in hertz.
const F_MAX: f64 = 9_000.0;
/// Peaks further than this below the loudest one are noise.
const PEAK_FLOOR: f64 = 75.0;
/// The most partials one note can keep.
const MAX_MODES: usize = 48;
/// Where the prompt decay stops and the tail begins, in seconds.
const PROMPT_END: f64 = 0.25;
/// The natural log of 1000, which turns a T60 into a decay rate.
const LN_1000: f64 = 6.907_755;
/// How many samples the envelope is averaged over before it is fitted.
const DECIMATION: usize = 8;

/// One note as the fitter found it.
pub struct FittedNote {
	/// The MIDI note, after snapping the fundamental to the keyboard.
	pub note: u8,
	/// The fundamental the fit used, in hertz.
	pub frequency: f32,
	/// The partials, sorted by frequency.
	pub modes: Vec<Mode>,
}

/// Fit one recorded note to a mode table.
///
/// `samples` is one channel of the recording, at `rate`. When `note` is
/// given the fit is centred on that note; when it is not, the fundamental is
/// guessed from the spectrum and then snapped to the nearest key. A sample
/// that is silent, too short or has nothing above its own noise floor has no
/// table to offer and gives `None`.
pub fn fit(samples: &[f32], rate: f32, note: Option<u8>) -> Option<FittedNote> {
	if rate <= 0.0 || rate.is_nan() {
		return None;
	}
	let rate = rate as f64;
	if samples.len() < (0.2 * rate) as usize {
		return None;
	}
	let signal = normalise(samples)?;
	let (frequency, note) = match note {
		Some(note) => (estimate_f0(&signal, rate, note_frequency(note)), note),
		None => {
			let guess = guess_f0(&signal, rate)?;
			let note = note_of_frequency(guess);
			(estimate_f0(&signal, rate, note_frequency(note)), note)
		}
	};
	let modes = detect(&signal, rate, frequency)?;
	if modes.is_empty() {
		return None;
	}
	Some(FittedNote {
		note,
		frequency: frequency as f32,
		modes,
	})
}

/// The frequency a MIDI note is tuned to.
fn note_frequency(note: u8) -> f64 {
	440.0 * 2f64.powf((note as f64 - 69.0) / 12.0)
}

/// The key nearest a frequency.
fn note_of_frequency(frequency: f64) -> u8 {
	let note = (69.0 + 12.0 * (frequency.max(1.0) / 440.0).log2()).round();
	note.clamp(0.0, 127.0) as u8
}

/// The recording, de-meaned and scaled to a unit peak.
fn normalise(samples: &[f32]) -> Option<Vec<f64>> {
	let mean = samples.iter().map(|sample| *sample as f64).sum::<f64>() / samples.len() as f64;
	let mut signal: Vec<f64> = samples.iter().map(|sample| *sample as f64 - mean).collect();
	let peak = signal
		.iter()
		.fold(0.0f64, |peak, sample| peak.max(sample.abs()));
	if peak < 1e-6 {
		return None;
	}
	for sample in &mut signal {
		*sample /= peak;
	}
	Some(signal)
}

/// A Hann window, which is what the spectra are read through.
fn hann(length: usize) -> Vec<f64> {
	if length <= 1 {
		return vec![1.0; length];
	}
	(0..length)
		.map(|index| {
			let sine = (std::f64::consts::PI * index as f64 / (length - 1) as f64).sin();
			sine * sine
		})
		.collect()
}

/// The smallest power of two that holds `length`.
fn next_power_of_two(length: usize) -> usize {
	let mut size = 1;
	while size < length {
		size <<= 1;
	}
	size
}

/// The magnitude spectrum of a windowed segment, zero padded to `size`.
fn spectrum(segment: &[f64], size: usize) -> Vec<f64> {
	let mut real = vec![0.0; size];
	let mut imaginary = vec![0.0; size];
	let window = hann(segment.len());
	for (index, sample) in segment.iter().enumerate() {
		real[index] = sample * window[index];
	}
	fft(&mut real, &mut imaginary);
	(0..size / 2 + 1)
		.map(|bin| (real[bin] * real[bin] + imaginary[bin] * imaginary[bin]).sqrt())
		.collect()
}

/// An in-place radix-2 complex FFT.
fn fft(real: &mut [f64], imaginary: &mut [f64]) {
	let length = real.len();
	let mut reversed = 0usize;
	for index in 1..length {
		let mut bit = length >> 1;
		while reversed & bit != 0 {
			reversed ^= bit;
			bit >>= 1;
		}
		reversed |= bit;
		if index < reversed {
			real.swap(index, reversed);
			imaginary.swap(index, reversed);
		}
	}
	let mut span = 2;
	while span <= length {
		let angle = -std::f64::consts::TAU / span as f64;
		let mut start = 0;
		while start < length {
			for step in 0..span / 2 {
				let (sine, cosine) = (angle * step as f64).sin_cos();
				let even = start + step;
				let odd = even + span / 2;
				let odd_real = real[odd] * cosine - imaginary[odd] * sine;
				let odd_imaginary = real[odd] * sine + imaginary[odd] * cosine;
				real[odd] = real[even] - odd_real;
				imaginary[odd] = imaginary[even] - odd_imaginary;
				real[even] += odd_real;
				imaginary[even] += odd_imaginary;
			}
			start += span;
		}
		span <<= 1;
	}
}

/// How much energy sits on the harmonic series of one candidate fundamental.
fn harmonic_score(spectrum: &[f64], size: usize, rate: f64, frequency: f64) -> f64 {
	let mut score = 0.0;
	for harmonic in 1..=6 {
		let bin = (harmonic as f64 * frequency * size as f64 / rate).round() as isize;
		if bin >= 1 && (bin as usize) + 1 < spectrum.len() {
			let bin = bin as usize;
			score += spectrum[bin - 1].max(spectrum[bin]).max(spectrum[bin + 1]);
		}
	}
	score
}

/// The fundamental of the note, refined around the key it is expected to be.
fn estimate_f0(signal: &[f64], rate: f64, expected: f64) -> f64 {
	let length = ((0.36 * rate) as usize).min(signal.len());
	if length < 256 {
		return expected;
	}
	let size = next_power_of_two(8 * length);
	let spectrum = spectrum(&signal[..length], size);
	let mut best = (0.0, expected);
	for cents in -250..=250 {
		if cents % 5 != 0 {
			continue;
		}
		let frequency = expected * 2f64.powf(cents as f64 / 1200.0);
		let score = harmonic_score(&spectrum, size, rate, frequency);
		if score > best.0 {
			best = (score, frequency);
		}
	}
	best.1
}

/// The fundamental of a sample whose note is not known.
///
/// Every strong peak is a candidate, and so is every subharmonic of one, because
/// a bass note's fundamental can be quieter than its second partial. The
/// candidate whose harmonic series holds the most energy wins, and the caller
/// snaps it to the keyboard before refining it.
fn guess_f0(signal: &[f64], rate: f64) -> Option<f64> {
	let length = ((0.36 * rate) as usize).min(signal.len());
	if length < 256 {
		return None;
	}
	let size = next_power_of_two(8 * length);
	let spectrum = spectrum(&signal[..length], size);
	let step = rate / size as f64;
	let db: Vec<f64> = spectrum
		.iter()
		.map(|magnitude| 20.0 * (magnitude + 1e-12).log10())
		.collect();
	let loudest = db
		.iter()
		.fold(f64::MIN, |loudest, value| loudest.max(*value));
	let found = peaks(&db, loudest - 60.0, 6.0, 4);
	let mut candidates: Vec<(f64, f64)> = Vec::new();
	for peak in found {
		let frequency = peak as f64 * step;
		if !(20.0..=5_000.0).contains(&frequency) {
			continue;
		}
		for divisor in 1..=4 {
			let candidate = frequency / divisor as f64;
			if candidate < 20.0 {
				continue;
			}
			let score = harmonic_score(&spectrum, size, rate, candidate);
			candidates.push((score, candidate));
		}
	}
	let best = candidates
		.iter()
		.fold(0.0f64, |best, (score, _)| best.max(*score));
	if best <= 0.0 {
		return None;
	}
	// A note whose fundamental is weak, or a sample with only two partials, has
	// several candidates that explain the spectrum equally well: every
	// subharmonic of the fundamental does. Taking the highest of the near ties
	// keeps the octave right, because a harmonic explains fewer partials than
	// the fundamental does and only ties when there is nothing to tell them
	// apart.
	let chosen = candidates
		.iter()
		.filter(|(score, _)| *score >= best * 0.999)
		.fold(0.0f64, |chosen, (_, candidate)| chosen.max(*candidate));
	(chosen > 0.0).then_some(chosen)
}

/// The local maxima of a curve, the way a peak picker sees them.
fn peaks(values: &[f64], height: f64, prominence: f64, distance: usize) -> Vec<usize> {
	if values.len() < 3 {
		return Vec::new();
	}
	let mut found = Vec::new();
	for index in 1..values.len() - 1 {
		if !(values[index] > values[index - 1] && values[index] > values[index + 1]) {
			continue;
		}
		let peak = values[index];
		if peak < height {
			continue;
		}
		let mut left = peak;
		let mut cursor = index;
		while cursor > 0 && values[cursor - 1] <= peak {
			cursor -= 1;
			left = left.min(values[cursor]);
		}
		let mut right = peak;
		let mut cursor = index;
		while cursor + 1 < values.len() && values[cursor + 1] <= peak {
			cursor += 1;
			right = right.min(values[cursor]);
		}
		if peak - left.max(right) < prominence {
			continue;
		}
		found.push(index);
	}
	if distance <= 1 {
		return found;
	}
	let mut order: Vec<usize> = (0..found.len()).collect();
	order.sort_by(|a, b| {
		values[found[*b]]
			.partial_cmp(&values[found[*a]])
			.unwrap_or(std::cmp::Ordering::Equal)
	});
	let mut kept: Vec<usize> = Vec::new();
	for index in order {
		let candidate = found[index];
		if kept
			.iter()
			.all(|other| candidate.abs_diff(*other) >= distance)
		{
			kept.push(candidate);
		}
	}
	kept.sort_unstable();
	kept
}

/// The signal demodulated at one frequency, averaged into blocks of
/// [DECIMATION] samples.
fn coarse_demod(signal: &[f64], rate: f64, frequency: f64) -> Vec<(f64, f64)> {
	let angle = -std::f64::consts::TAU * frequency / rate;
	let (sine, cosine) = angle.sin_cos();
	let mut phase = (1.0, 0.0);
	let mut accumulator = (0.0, 0.0);
	let mut out = Vec::with_capacity(signal.len() / DECIMATION + 1);
	for (index, sample) in signal.iter().enumerate() {
		accumulator.0 += sample * phase.0;
		accumulator.1 += sample * phase.1;
		if (index + 1) % DECIMATION == 0 {
			out.push((
				accumulator.0 / DECIMATION as f64,
				accumulator.1 / DECIMATION as f64,
			));
			accumulator = (0.0, 0.0);
		}
		phase = (
			phase.0 * cosine - phase.1 * sine,
			phase.0 * sine + phase.1 * cosine,
		);
	}
	out
}

/// A moving average of the demodulated signal, the way `same` convolution
/// centres it: the kernel is the Hann window the script uses.
fn smooth(signal: &[(f64, f64)], kernel: &[f64]) -> Vec<f64> {
	if kernel.is_empty() {
		return signal
			.iter()
			.map(|(real, imaginary)| real.hypot(*imaginary))
			.collect();
	}
	let offset = (kernel.len() - 1) / 2;
	let mut out = Vec::with_capacity(signal.len());
	for index in 0..signal.len() {
		let mut sum = (0.0, 0.0);
		for (tap, weight) in kernel.iter().enumerate() {
			let source = index as isize + offset as isize - tap as isize;
			if source >= 0 && (source as usize) < signal.len() {
				let value = signal[source as usize];
				sum.0 += value.0 * weight;
				sum.1 += value.1 * weight;
			}
		}
		out.push(sum.0.hypot(sum.1));
	}
	out
}

/// The quarter percentile of a set of values, interpolating like numpy does.
fn percentile(values: &[f64], percent: f64) -> f64 {
	if values.is_empty() {
		return 0.0;
	}
	let mut sorted = values.to_vec();
	sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
	let position = (percent / 100.0) * (sorted.len() - 1) as f64;
	let lower = position.floor() as usize;
	let upper = (lower + 1).min(sorted.len() - 1);
	let fraction = position - lower as f64;
	sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction
}

/// The T60 of a decay, fitted on a window of its envelope.
fn fit_slope(times: &[f64], envelope: &[f64], low: f64, high: f64) -> Option<f64> {
	let mut count = 0.0;
	let mut sum_x = 0.0;
	let mut sum_y = 0.0;
	let mut sum_xx = 0.0;
	let mut sum_xy = 0.0;
	for (time, value) in times.iter().zip(envelope) {
		if *time <= low || *time >= high {
			continue;
		}
		let x = *time;
		let y = 20.0 * (value + 1e-12).log10();
		count += 1.0;
		sum_x += x;
		sum_y += y;
		sum_xx += x * x;
		sum_xy += x * y;
	}
	if count < 12.0 {
		return None;
	}
	let denominator = count * sum_xx - sum_x * sum_x;
	if denominator.abs() < 1e-12 {
		return None;
	}
	let slope = (count * sum_xy - sum_x * sum_y) / denominator;
	if slope >= -0.5 {
		return None;
	}
	Some(60.0 / -slope)
}

/// One partial as the envelope stage measured it, before the least squares.
struct RawMode {
	frequency: f64,
	/// The T60 of the prompt decay, in seconds.
	prompt: f64,
	/// The T60 of the tail, in seconds.
	tail: f64,
	/// The harmonic order the table keys the velocity profile on.
	order: f64,
	/// The complex projection of the signal on the prompt and the tail basis.
	projection: [(f64, f64); 2],
}

/// Find the partials of one note, measure their decays, and place their
/// amplitudes and phases.
fn detect(signal: &[f64], rate: f64, f0: f64) -> Option<Vec<Mode>> {
	let start = (0.06 * rate) as usize;
	let length = ((0.9 * rate) as usize).min(signal.len().saturating_sub(start));
	if length < 256 || f0 <= 0.0 {
		return None;
	}
	let size = next_power_of_two(8 * length);
	let magnitude = spectrum(&signal[start..start + length], size);
	let step = rate / size as f64;
	let db: Vec<f64> = magnitude
		.iter()
		.map(|value| 20.0 * (value / length as f64 + 1e-12).log10())
		.collect();
	let loudest = db
		.iter()
		.fold(f64::MIN, |loudest, value| loudest.max(*value));
	let distance = (4).max((6.0f64.max(0.08 * f0) / step) as usize);
	let mut found = peaks(&db, loudest - PEAK_FLOOR, 6.0, distance);
	if found.is_empty() {
		return None;
	}
	found.sort_by(|a, b| {
		db[*b]
			.partial_cmp(&db[*a])
			.unwrap_or(std::cmp::Ordering::Equal)
	});
	found.truncate(MAX_MODES);
	found.sort_unstable();

	// The envelope is smooth on a millisecond scale, so it is measured on a
	// decimated signal; the kernel is decimated with it, because the
	// convolution has to stay on one timeline.
	let span = (rate * 0.5 / f0).max(3.0) as usize;
	let window = hann(span);
	let total: f64 = window.iter().sum();
	let kernel: Vec<f64> = window
		.iter()
		.step_by(DECIMATION)
		.map(|tap| tap / total)
		.collect();

	let mut raw: Vec<RawMode> = Vec::new();
	for index in found {
		let left = db[index - 1];
		let right = db[index + 1];
		let centre = db[index];
		let offset = 0.5 * (left - right) / (left - 2.0 * centre + right);
		let frequency = index as f64 * step + offset * step;
		if !(25.0 < frequency && frequency < F_MAX) {
			continue;
		}
		let demodulated = coarse_demod(signal, rate, frequency);
		let envelope = smooth(&demodulated, &kernel);
		let loudest = envelope
			.iter()
			.fold(0.0f64, |loudest, value| loudest.max(*value));
		if loudest < 1e-6 {
			continue;
		}
		let times: Vec<f64> = (0..envelope.len())
			.map(|sample| sample as f64 * DECIMATION as f64 / rate)
			.collect();
		let last = times.last().copied().unwrap_or(0.0);
		let late: Vec<f64> = times
			.iter()
			.zip(&envelope)
			.filter(|(time, _)| **time > 1.2)
			.map(|(_, value)| *value)
			.collect();
		let floor = percentile(&late, 25.0);

		// The tail is the part of the note that is not the prompt: it has to
		// be past the attack, still above the room, and not the last few
		// samples, where the recording has already faded.
		let mut tail_keep: Vec<usize> = (0..times.len())
			.filter(|index| {
				times[*index] > 0.35
					&& times[*index] < last - 0.05
					&& envelope[*index] > floor * 2.0
			})
			.collect();
		if tail_keep.len() < 12 {
			tail_keep = (0..times.len())
				.filter(|index| times[*index] > 0.12 && times[*index] < 0.6)
				.collect();
		}
		if tail_keep.len() < 12 {
			continue;
		}
		let tail_high = times[tail_keep[tail_keep.len() - 1]];
		let mut tail = fit_slope(&times, &envelope, 0.35, tail_high);
		if tail.is_none() {
			tail = fit_slope(&times, &envelope, 0.12, tail_high);
		}
		let mut prompt = fit_slope(&times, &envelope, 0.03, PROMPT_END);
		if prompt.is_none() {
			prompt = fit_slope(&times, &envelope, 0.006, 0.06);
		}
		if prompt.is_none() {
			match tail {
				Some(tail) => prompt = Some(tail.min(0.6)),
				None => continue,
			}
		}
		let prompt = prompt.unwrap_or(0.0).clamp(0.04, 0.6);
		let tail = tail.unwrap_or(prompt * 2.0).clamp(0.05, 30.0);
		let projection = [
			project(signal, rate, frequency, LN_1000 / prompt),
			project(signal, rate, frequency, LN_1000 / tail),
		];
		raw.push(RawMode {
			frequency,
			prompt,
			tail,
			order: (frequency / f0).round(),
			projection,
		});
	}

	// Nothing rings below the fundamental: what sits under it is the room, the
	// finger noise or a beating sideband, and it makes the fit sound octave low.
	raw.retain(|mode| mode.frequency > 0.92 * f0);
	if raw.is_empty() {
		return None;
	}
	raw.sort_by(|a, b| {
		a.frequency
			.partial_cmp(&b.frequency)
			.unwrap_or(std::cmp::Ordering::Equal)
	});

	let coefficients = solve(&raw, signal.len(), rate)?;
	let mut modes = Vec::with_capacity(raw.len());
	for (mode, [prompt, tail]) in raw.iter().zip(coefficients) {
		if prompt.0 + tail.0 <= 1e-6 {
			continue;
		}
		modes.push(Mode {
			order: mode.order as f32,
			frequency: mode.frequency as f32,
			prompt: prompt.0 as f32,
			tail: tail.0 as f32,
			prompt_t60: mode.prompt as f32,
			tail_t60: mode.tail as f32,
			prompt_phase: prompt.1 as f32,
			tail_phase: tail.1 as f32,
		});
	}
	Some(modes)
}

/// The complex projection of the signal on one damped sinusoid.
fn project(signal: &[f64], rate: f64, frequency: f64, sigma: f64) -> (f64, f64) {
	let angle = -std::f64::consts::TAU * frequency / rate;
	let (sine, cosine) = angle.sin_cos();
	let decay = (-sigma / rate).exp();
	let mut phase = (1.0, 0.0);
	let mut envelope = 1.0;
	let mut sum = (0.0, 0.0);
	for sample in signal {
		sum.0 += sample * phase.0 * envelope;
		sum.1 += sample * phase.1 * envelope;
		phase = (
			phase.0 * cosine - phase.1 * sine,
			phase.0 * sine + phase.1 * cosine,
		);
		envelope *= decay;
	}
	sum
}

/// One sum of a geometric series with a rotating phase.
///
/// `sum r^k e^(i theta k)` for `k` from zero to `count - 1`, which is what
/// the inner product of two damped sinusoids is at a fixed sample rate.
fn geometric(ratio: f64, angle: f64, count: usize) -> (f64, f64) {
	let (sine, cosine) = angle.sin_cos();
	let (zr, zi) = (ratio * cosine, ratio * sine);
	let (dr, di) = (1.0 - zr, -zi);
	let denominator = dr * dr + di * di;
	if denominator < 1e-24 {
		return (count as f64, 0.0);
	}
	// z^count by squaring, so a long note does not cost a loop per column.
	let (mut power_r, mut power_i) = (1.0f64, 0.0f64);
	let (mut base_r, mut base_i) = (zr, zi);
	let mut exponent = count;
	while exponent > 0 {
		if exponent & 1 == 1 {
			let real = power_r * base_r - power_i * base_i;
			power_i = power_r * base_i + power_i * base_r;
			power_r = real;
		}
		let real = base_r * base_r - base_i * base_i;
		base_i *= 2.0 * base_r;
		base_r = real;
		exponent >>= 1;
	}
	let (ur, ui) = (1.0 - power_r, -power_i);
	(
		(ur * dr + ui * di) / denominator,
		(ui * dr - ur * di) / denominator,
	)
}

/// Place every partial's amplitude and phase with one least squares.
fn solve(modes: &[RawMode], samples: usize, rate: f64) -> Option<Vec<[(f64, f64); 2]>> {
	let count = modes.len();
	let columns = 4 * count;
	let pairs = 2 * count;
	let mut gram = vec![0.0f64; columns * columns];
	let mut rhs = vec![0.0f64; columns];

	for (index, mode) in modes.iter().enumerate() {
		rhs[4 * index] = mode.projection[0].0;
		rhs[4 * index + 1] = -mode.projection[0].1;
		rhs[4 * index + 2] = mode.projection[1].0;
		rhs[4 * index + 3] = -mode.projection[1].1;
	}

	let basis = |index: usize| -> (f64, f64) {
		let mode = &modes[index / 2];
		let t60 = if index.is_multiple_of(2) {
			mode.prompt
		} else {
			mode.tail
		};
		(std::f64::consts::TAU * mode.frequency, LN_1000 / t60)
	};

	for first in 0..pairs {
		let (frequency_a, sigma_a) = basis(first);
		let column_a = 4 * (first / 2) + 2 * (first % 2);
		for second in first..pairs {
			let (frequency_b, sigma_b) = basis(second);
			let column_b = 4 * (second / 2) + 2 * (second % 2);
			let ratio = (-(sigma_a + sigma_b) / rate).exp();
			let (same_real, same_imaginary) =
				geometric(ratio, (frequency_a + frequency_b) / rate, samples);
			let (cross_real, cross_imaginary) =
				geometric(ratio, (frequency_b - frequency_a) / rate, samples);
			let block = [
				(same_real + cross_real) / 2.0,
				(same_imaginary + cross_imaginary) / 2.0,
				(same_imaginary - cross_imaginary) / 2.0,
				(cross_real - same_real) / 2.0,
			];
			for (offset, value) in block.iter().enumerate() {
				let (row, column) = (column_a + offset / 2, column_b + offset % 2);
				gram[row * columns + column] = *value;
				gram[column * columns + row] = *value;
			}
		}
	}

	// The two decays of a partial can be the same number, and then its prompt
	// and tail columns are the same column; the ridge is what keeps the system
	// solvable, and it is the same relative ridge the script uses.
	let mean = (0..columns)
		.map(|index| gram[index * columns + index])
		.sum::<f64>()
		/ columns as f64;
	for index in 0..columns {
		gram[index * columns + index] += 1e-9 * mean;
	}

	let coefficients = cholesky(gram, &rhs, columns)?;
	let mut out = Vec::with_capacity(count);
	for index in 0..count {
		let (a, b) = (coefficients[4 * index], coefficients[4 * index + 1]);
		let (c, d) = (coefficients[4 * index + 2], coefficients[4 * index + 3]);
		out.push([(a.hypot(b), -b.atan2(a)), (c.hypot(d), -d.atan2(c))]);
	}
	Some(out)
}

/// Solve a symmetric positive definite system by Cholesky factorisation.
fn cholesky(mut matrix: Vec<f64>, rhs: &[f64], count: usize) -> Option<Vec<f64>> {
	for row in 0..count {
		for column in 0..=row {
			let mut sum = matrix[row * count + column];
			for inner in 0..column {
				sum -= matrix[row * count + inner] * matrix[column * count + inner];
			}
			if row == column {
				if sum <= 0.0 {
					return None;
				}
				matrix[row * count + column] = sum.sqrt();
			} else {
				matrix[row * count + column] = sum / matrix[column * count + column];
			}
		}
	}
	let mut forward = vec![0.0; count];
	for row in 0..count {
		let mut sum = rhs[row];
		for inner in 0..row {
			sum -= matrix[row * count + inner] * forward[inner];
		}
		forward[row] = sum / matrix[row * count + row];
	}
	let mut out = vec![0.0; count];
	for row in (0..count).rev() {
		let mut sum = forward[row];
		for inner in row + 1..count {
			sum -= matrix[inner * count + row] * out[inner];
		}
		out[row] = sum / matrix[row * count + row];
	}
	Some(out)
}

#[cfg(test)]
mod tests;
