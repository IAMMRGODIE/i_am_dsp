//! Frequency-domain helpers for imported tables.

use rustfft::FftPlanner;
use rustfft::num_complex::Complex;

/// Turn a time slice into a zero-phase kernel using only its magnitude.
///
/// This is the import path for "use the sample as a frequency curve": the
/// sample's phase is dropped, the magnitude becomes the filter response, and
/// the inverse transform returns a symmetric kernel.
pub fn magnitude_kernel(samples: &[f32], size: usize) -> Vec<f32> {
	let size = size.max(2);
	let mut buffer = vec![Complex::new(0.0, 0.0); size];

	for (index, sample) in samples.iter().take(size).enumerate() {
		buffer[index] = Complex::new(*sample, 0.0);
	}

	let mut planner = FftPlanner::<f32>::new();
	let forward = planner.plan_fft_forward(size);
	let inverse = planner.plan_fft_inverse(size);
	forward.process(&mut buffer);

	for value in &mut buffer {
		*value = Complex::new(value.norm(), 0.0);
	}
	inverse.process(&mut buffer);

	let mut kernel = vec![0.0; size];
	for (index, value) in buffer.iter().enumerate() {
		kernel[(index + size / 2) % size] = value.re / size as f32;
	}

	let energy = kernel.iter().map(|value| value * value).sum::<f32>().sqrt();
	if energy > 1e-6 {
		for value in &mut kernel {
			*value /= energy;
		}
	}

	kernel
}
