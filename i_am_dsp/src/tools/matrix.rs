//! Matrix helpers for the feedback delay network.

use rand_distr::{Distribution, StandardNormal};

pub(crate) type Mat<const N: usize> = [[f32; N]; N];

/// A random unit vector.
///
/// A unit vector describes the Householder reflection [I - 2 v v^T], which is
/// orthogonal, so it does not change the energy of the network, and mixes every
/// delay line into every other one, which is what a feedback delay network needs
/// from its feedback matrix.
pub(crate) fn random_householder_vector<const N: usize>() -> [f32; N] {
	assert!(N > 0);
	let normal = StandardNormal;
	let mut rng = rand::rng();
	loop {
		let vector: [f32; N] = core::array::from_fn(|_| normal.sample(&mut rng));
		let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
		if norm > 1e-6 {
			return core::array::from_fn(|i| vector[i] / norm);
		}
	}
}

/// Multiplies a row major matrix with a vector.
pub(crate) fn mat_mul_vec<const N: usize>(a: &Mat<N>, b: &[f32; N]) -> [f32; N] {
	core::array::from_fn(|i| {
		let row = &a[i];
		let mut sum = 0.0;
		for j in 0..N {
			sum += row[j] * b[j];
		}
		sum
	})
}
