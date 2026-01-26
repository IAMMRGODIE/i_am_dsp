//! Wave Similarity Overlap and Add (WSOLA) algorithm implementation in Rust.

use std::f32::consts::PI;

use crate::tools::ring_buffer::RingBuffer;

/// Wave Similarity Overlap and Add (WSOLA) algorithm implementation in Rust.
/// 
/// Panics if
/// 1. similarity_measure returns NaN,
/// 2. stretch_factor is not greater than 0.0,
/// 3. max_offset is not greater than 0,
/// 4. ref_range is not greater than 0,
/// 5. hop is not greater or equal than ref_range,
/// 6. input size is not greater than 2 * hop size,
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

	let no_last_output = last_output.is_empty();

	let last_output = if last_output.len() < hop {
		let mut added_zero = vec![0.0; hop - last_output.len()];
		added_zero.extend_from_slice(last_output);
		added_zero
	}else {
		last_output.to_vec()
	};
	
	assert!(max_offset > 0, "max_offset must be positive");
	assert!(hop > 0, "hop must be positive");
	assert!(ref_range > 0, "ref_range must be positive");

	let analysis_hop = (hop as f32 / stretch_factor).round() as usize;
	let output_len = (input.capacity() as f32 * stretch_factor).ceil() as usize;
	let mut output = vec![0.0; output_len];
	// println!("output size: {}", output.len());

	let mut current_input_pos = max_offset;
	let mut frame_idx = 0;
	while current_input_pos <= input.capacity() + max_offset {
		let reference: Vec<f32> = if frame_idx == 0 {
			last_output[last_output.len() - ref_range..].to_vec()
		}else {
			let ref_start = current_input_pos.saturating_sub(ref_range);
			let ref_end = (ref_start + ref_range).min(input.capacity());
			input.range(ref_start..ref_end).cloned().collect()
		};

		let (best_offset, _) = find_best_offset(
			input,
			current_input_pos,
			max_offset,
			window_size,
			&reference,
			similarity_measure,
		);

		let frame_start = best_offset;
		let frame_end = frame_start + window_size;
		if frame_end > input.capacity() {
			break;
		}

		let output_start = frame_idx * hop;
		for (i, sample) in input.range(frame_start..frame_end).enumerate() {
			if output_start + i >= output.len() {
				break;
			}

			if frame_idx == 0 && no_last_output {
				output[output_start + i] += sample;
			}else if i < hop && frame_idx == 0 {
				output[output_start + i] += sample * hamming_window(i, window_size) + last_output[last_output.len() - hop + i];
			}else if output_start + i < output.len() {
				output[output_start + i] += sample * hamming_window(i, window_size);
			}
		}

		current_input_pos += analysis_hop;
		frame_idx += 1;
	}

	output
}

#[inline(always)]
fn hamming_window(i: usize, size: usize) -> f32 {
	0.54 - 0.46 * (2.0 * PI * i as f32 / (size - 1) as f32).cos()
}

fn find_best_offset(
	input: &RingBuffer<f32>,
	center: usize,
	max_offset: usize,
	window_size: usize,
	reference: &[f32],
	similarity_measure: fn(&[f32], &[f32]) -> f32,
) -> (usize, f32) {
	let search_start = center.saturating_sub(max_offset);
	let search_end = (center + max_offset).min(input.capacity() - window_size);
	
	let mut best_offset = center;
	let mut best_similarity = f32::NEG_INFINITY;

	for candidate_start in search_start..=search_end {
		let candidate = input.range(candidate_start..candidate_start + reference.len()).cloned().collect::<Vec<f32>>();
		let similarity = similarity_measure(reference, &candidate);
		
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