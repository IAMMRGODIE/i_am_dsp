//! Wave Similarity Overlap and Add (WSOLA) algorithm implementation in Rust.

use std::f32::consts::PI;

use crate::tools::ring_buffer::RingBuffer;

/// Wave Similarity Overlap and Add (WSOLA) algorithm implementation in Rust.
///
/// Stretches (or compresses) the signal currently held in `input` by
/// `stretch_factor` and returns the result. The output has a length of
/// `ceil(capacity * stretch_factor)` samples.
///
/// The algorithm splits the input into windowed frames of `2 * hop` samples
/// that are placed every `hop` samples in the output (50 % overlap) and
/// overlap-added together. Before a frame is written, the input is searched
/// near the corresponding analysis position for the segment that best matches
/// the signal that has *already been synthesized* at that splice point, which
/// keeps the phase of the output continuous even though the analysis positions
/// advance at a different rate than the synthesis positions.
///
/// If `last_output` is not empty, its trailing `hop` samples are cross-faded
/// into the very first frame of the result, so consecutive calls on a stream
/// stay seamless.
///
/// # Panics
///
/// 1. `similarity_measure` returns NaN,
/// 2. `stretch_factor` is not greater than 0.0,
/// 3. `max_offset` is not greater than 0,
/// 4. `ref_range` is not greater than 0,
/// 5. `hop` is not greater or equal than `ref_range`,
/// 6. input capacity is not greater or equal than `2 * hop`.
pub fn wsola(
	input: &RingBuffer<f32>,
	last_output: &[f32],
	stretch_factor: f32,
	max_offset: usize,
	hop: usize,
	ref_range: usize,
	similarity_measure: fn(&[f32], &[f32]) -> f32,
) -> Vec<f32> {
	const OVERLAP_FACTOR: usize = 2;
	let window_size = OVERLAP_FACTOR * hop;

	assert!(stretch_factor > 0.0, "stretch_factor must be greater than 0");
	assert!(max_offset > 0, "max_offset must be positive");
	assert!(hop > 0, "hop must be positive");
	assert!(ref_range > 0, "ref_range must be positive");
	assert!(hop >= ref_range, "hop must be greater or equal than ref_range");
	assert!(window_size <= input.capacity(), "input capacity must be greater or equal than 2 * hop");

	if input.capacity() == 0 {
		return vec![];
	}

	let input_len = input.capacity();
	// Analysis positions advance by `hop / stretch_factor` so that the output
	// covers `stretch_factor` samples per input sample. Round to the nearest
	// integer analysis hop, but never stall completely.
	let analysis_hop = ((hop as f32 / stretch_factor).round() as usize).max(1);
	let output_len = (input_len as f32 * stretch_factor).ceil() as usize;
	// The last input position at which a full window can still start.
	let max_fit_offset = input_len - window_size;

	let mut output = vec![0.0; output_len];

	// The tail of the previous output block. Used as the similarity reference
	// and as the cross-fade partner for the very first frame.
	let no_previous = last_output.is_empty();
	let mut prev_tail = vec![0.0; hop];
	if !no_previous {
		let copied = last_output.len().min(hop);
		prev_tail[hop - copied..].copy_from_slice(&last_output[last_output.len() - copied..]);
	}

	let mut analysis_pos = 0usize; // nominal analysis position in the input
	let mut output_pos = 0usize;   // synthesis position of the current frame
	let mut frame_idx = 0usize;

	while output_pos < output_len {
		// The analysis pointer may advance past the last full-window position
		// (e.g. when stretching heavily). Saturate it so the tail of the input
		// is reused instead of dropping out to silence.
		let center = analysis_pos.min(max_fit_offset);

		// The reference is the signal already synthesized directly before the
		// splice point. For the first frame that is the previous output block,
		// for later frames it is the region of *this* output that the previous
		// frame has already written (`ref_range <= hop` guarantees those
		// samples exist). Near the end of the output there is nothing to splice
		// against, so the frame is simply placed at the analysis center.
		let best_offset = if frame_idx == 0 && !no_previous {
			let reference = &prev_tail[prev_tail.len() - ref_range..];
			find_best_offset(input, center, max_offset, max_fit_offset, reference, similarity_measure).0
		}else if frame_idx > 0 && output_pos + ref_range <= output_len {
			let reference = &output[output_pos..output_pos + ref_range];
			find_best_offset(input, center, max_offset, max_fit_offset, reference, similarity_measure).0
		}else {
			center
		};

		let frame: Vec<f32> = input.range(best_offset..best_offset + window_size).cloned().collect();

		for (i, &sample) in frame.iter().enumerate() {
			let pos = output_pos + i;
			if pos >= output_len {
				break;
			}

			let weight = hann_window(i, window_size);
			if frame_idx == 0 && !no_previous && i < hop {
				// Cross-fade the tail of the previous block into the onset of
				// the new one. The two windows are complementary
				// (`w[i] + w[i + hop] = 1`), so the transition is gain-constant.
				output[pos] += sample * weight + prev_tail[i] * hann_window(i + hop, window_size);
			}else {
				output[pos] += sample * weight;
			}
		}

		output_pos += hop;
		analysis_pos += analysis_hop;
		frame_idx += 1;
	}

	output
}

/// Periodic Hann window (`w[0] = 0`, `w[N / 2] = 1`).
///
/// With 50 % overlap (`hop = N / 2`) the window is complementary to itself
/// shifted by one hop: `w[i] + w[i + hop] = 1`, which makes the overlap-add
/// in [`wsola`] exactly amplitude preserving.
#[inline(always)]
fn hann_window(i: usize, size: usize) -> f32 {
	0.5 - 0.5 * (2.0 * PI * i as f32 / size as f32).cos()
}

/// Searches the input for the segment that best matches `reference` and
/// returns its start position together with the similarity score.
///
/// The search is restricted to the range `[center - max_offset,
/// center + max_offset]` clamped to `[0, max_fit_offset]`, so the returned
/// position always admits a full window of `window_size = 2 * hop` samples.
fn find_best_offset(
	input: &RingBuffer<f32>,
	center: usize,
	max_offset: usize,
	max_fit_offset: usize,
	reference: &[f32],
	similarity_measure: fn(&[f32], &[f32]) -> f32,
) -> (usize, f32) {
	let lo = center.saturating_sub(max_offset).min(max_fit_offset);
	let hi = (center + max_offset).min(max_fit_offset);

	let mut best_offset = lo;
	let mut best_similarity = f32::NEG_INFINITY;

	for candidate_start in lo..=hi {
		let candidate: Vec<f32> = input
			.range(candidate_start..candidate_start + reference.len())
			.cloned()
			.collect();
		let similarity = similarity_measure(reference, &candidate);
		assert!(!similarity.is_nan(), "similarity_measure returned NaN");
		if similarity > best_similarity {
			best_similarity = similarity;
			best_offset = candidate_start;
		}
	}

	(best_offset, best_similarity)
}

/// Normalized Cross-Correlation (NCC) similarity measure implementation in Rust.
pub fn normalized_cross_correlation(x: &[f32], y: &[f32]) -> f32 {
	let x_avg = x.iter().sum::<f32>() / x.len() as f32;
	let y_avg = y.iter().sum::<f32>() / y.len() as f32;
	let mut numerator = 0.0;
	let mut denominator_x = 0.0;
	let mut denominator_y = 0.0;
	for i in 0..x.len().min(y.len()) {
		numerator += (x[i] - x_avg) * (y[i] - y_avg);
		denominator_x += (x[i] - x_avg) * (x[i] - x_avg);
		denominator_y += (y[i] - y_avg) * (y[i] - y_avg);
	}
	let denominator = (denominator_x * denominator_y).sqrt();
	if denominator == 0.0 {
		return 0.0;
	}
	numerator / denominator
}

/// Mean Square Error (MSE) similarity measure implementation in Rust.
///
/// Returns the negated MSE so that larger values mean *more* similar,
/// consistent with the other similarity measures.
pub fn negative_mean_square_error(x: &[f32], y: &[f32]) -> f32 {
	let mut error = 0.0;
	for i in 0..x.len() {
		error += (x[i] - y[i]).powi(2);
	}
	- error / x.len() as f32
}

/// Cosine similarity measure implementation in Rust.
pub fn cosine_similarity(x: &[f32], y: &[f32]) -> f32 {
	let dot_product = x.iter().zip(y).map(|(a, b)| a * b).sum::<f32>();
	let x_norm = x.iter().map(|a| a * a).sum::<f32>().sqrt();
	let y_norm = y.iter().map(|a| a * a).sum::<f32>().sqrt();
	if x_norm == 0.0 || y_norm == 0.0 {
		return 0.0;
	}
	dot_product / (x_norm * y_norm)
}

#[cfg(test)]
mod tests {
	use super::*;

	fn sine_buffer(capacity: usize, freq: f32, sample_rate: f32) -> RingBuffer<f32> {
		let mut buffer = RingBuffer::new(capacity);
		buffer.fill_with(|i| (2.0 * PI * freq * i as f32 / sample_rate).sin());
		buffer
	}

	fn sum_abs(samples: &[f32]) -> f32 {
		samples.iter().map(|s| s.abs()).sum()
	}

	/// With `stretch_factor = 1.0` and a periodic input, suitable analysis
	/// offsets are found exactly, so the overlap-add must reconstruct the
	/// input in the interior (both ends fade in/out by design).
	#[test]
	fn stretch_one_reconstructs_signal() {
		let capacity = 4096usize;
		let input = sine_buffer(capacity, 440.0, 48_000.0);
		let hop = capacity / 4;
		let ref_range = hop / 2;

		let output = wsola(
			&input,
			&[],
			1.0,
			hop / 32,
			hop,
			ref_range,
			negative_mean_square_error,
		);

		assert_eq!(output.len(), capacity);

		let start = 2 * hop;
		let end = output.len() - 2 * hop;
		let mut max_diff = 0.0f32;
		for i in start..end {
			let diff = (output[i] - input[i]).abs();
			max_diff = max_diff.max(diff);
		}
		assert!(max_diff < 0.01, "interior max difference was {max_diff}");
	}

	/// Stretching must produce the expected output length and a non-silent
	/// result for both stretch factors above and below 1.
	#[test]
	fn output_length_follows_stretch_factor() {
		let capacity = 2048usize;
		let input = sine_buffer(capacity, 330.0, 48_000.0);
		let hop = capacity / 4;
		let ref_range = hop / 2;

		let doubled = wsola(
			&input,
			&[],
			2.0,
			hop / 32,
			hop,
			ref_range,
			negative_mean_square_error,
		);
		assert_eq!(doubled.len(), (capacity as f32 * 2.0).ceil() as usize);
		assert!(sum_abs(&doubled) > 0.0, "stretched output should not be silent");

		let halved = wsola(
			&input,
			&[],
			0.5,
			hop / 32,
			hop,
			ref_range,
			negative_mean_square_error,
		);
		assert_eq!(halved.len(), (capacity as f32 * 0.5).ceil() as usize);
		assert!(sum_abs(&halved) > 0.0, "compressed output should not be silent");
	}

	/// Cross-fading a previous output block must not panic and must blend the
	/// previous tail into the first frame (bounded, non-zero onset), keeping
	/// consecutive blocks seamless.
	#[test]
	fn previous_output_is_crossfaded() {
		let capacity = 2048usize;
		let input = sine_buffer(capacity, 440.0, 48_000.0);
		let hop = capacity / 4;
		let ref_range = hop / 2;

		let previous = sine_buffer(capacity / 2, 440.0, 48_000.0);
		let previous: Vec<f32> = previous.range(0..previous.capacity()).cloned().collect();

		let output = wsola(
			&input,
			&previous,
			1.0,
			hop / 32,
			hop,
			ref_range,
			negative_mean_square_error,
		);

		assert_eq!(output.len(), capacity);
		// The onset region is a blend of the previous tail and new content:
		// it must not be pure silence nor a hard jump away from the previous
		// block's amplitude.
		let onset_energy = sum_abs(&output[..hop]);
		assert!(onset_energy > 0.0, "onset should not be silent");
		let prev_energy = sum_abs(&previous[previous.len() - hop..]);
		assert!(
			(onset_energy - prev_energy).abs() < prev_energy * 2.0,
			"onset energy diverged from previous tail energy"
		);
	}

	/// The similarity search must respect input bounds even when the window is
	/// as large as the whole buffer.
	#[test]
	fn full_size_window_does_not_panic() {
		let capacity = 1024usize;
		let input = sine_buffer(capacity, 220.0, 48_000.0);
		let hop = capacity / 2;

		let output = wsola(
			&input,
			&[],
			1.0,
			10,
			hop,
			hop / 2,
			negative_mean_square_error,
		);
		assert_eq!(output.len(), capacity);
	}
}
