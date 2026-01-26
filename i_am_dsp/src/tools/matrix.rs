use rand_distr::{Distribution, StandardNormal};
use wide::f32x4;

pub(crate) type Mat<const N: usize> = [[f32; N]; N];

pub(crate) const fn same<const N: usize>(inner: f32) -> [f32; N] {
	[inner; N]
}

pub(crate) fn scalar_multiply<const DELAY_LINES: usize>(v: &[f32], scalar: f32) -> [f32; DELAY_LINES] {
	let mut result = [0.0; DELAY_LINES];
	for i in 0..DELAY_LINES {
		result[i] = v[i] * scalar;
	}
	result
}

pub(crate) fn vector_add<const DELAY_LINES: usize>(v1: &[f32; DELAY_LINES], v2: &[f32; DELAY_LINES]) -> [f32; DELAY_LINES] {
	let mut result = [0.0; DELAY_LINES];
	for i in 0..DELAY_LINES {
		result[i] = v1[i] + v2[i];
	}
	result
}

fn dot_product(v1: &[f32], v2: &[f32]) -> f32 {
	let len = v1.len().min(v2.len());
	let mut result = 0.0;
	for i in (0..len).step_by(4) {
		let l = f32x4::from(&v1[i..(i + 4).min(len)]);
		let r = f32x4::from(&v2[i..(i + 4).min(len)]);
		result += (l * r).reduce_add()
	}

	result
}

fn vector_norm(v: &[f32]) -> f32 {
	dot_product(v, v).sqrt()
}

pub(crate) fn random_householder_matrix<const DELAY_LINES: usize>() -> Mat<DELAY_LINES> {
	assert!(DELAY_LINES > 0);
	let mut output = [[0.0; DELAY_LINES]; DELAY_LINES];
	let normal = StandardNormal;
	let mut rng = rand::rng();
	let (vector, norm): ([f32; DELAY_LINES], f32) = loop {
		let vector = core::array::from_fn(|_| normal.sample(&mut rng));
		let norm = vector_norm(&vector);
		if norm != 0.0 {
			break (vector, norm);
		}
	};

	for i in 0..DELAY_LINES {
		for j in 0..DELAY_LINES {
			output[i][j] = if i == j { 1.0 } else { 0.0 } - 2.0 * vector[i] * vector[j] / norm / norm;
		}
	}

	let (_, det) = mat_inverse_and_det(&output);
	if det < 0.0 {
		for output_inner in output.iter_mut().take(DELAY_LINES) {
			output_inner[0] = -output_inner[0];
		}
	}

	output
}

pub(crate) const fn identity<const N: usize>() -> Mat<N> {
	let mut result = [[0.0; N]; N];
	let mut i = 0;
	while i < N {
		result[i][i] = 1.0;
		i += 1;
	}
	result
}

pub(crate) fn mat_add<const N: usize>(a: &Mat<N>, b: &Mat<N>) -> Mat<N> {
	let mut result = [[0.0; N]; N];
	for i in 0..N {
		for j in 0..N {
			result[i][j] = a[i][j] + b[i][j];
		}
	}
	result
}

pub(crate) fn mat_scale<const N: usize>(a: &Mat<N>, s: f32) -> Mat<N> {
	let mut c = [[0.0; N]; N];
	for i in 0..N {
		for j in 0..N {
			c[i][j] = a[i][j] * s;
		}
	}
	c
}

fn mat_inverse_and_det<const N: usize>(a: &Mat<N>) -> (Option<Mat<N>>, f32) {
	let mut aug: [Vec<f32>; N] = core::array::from_fn(|_| vec![0.0f32; 2 * N]);
	for i in 0..N {
		for j in 0..N {
			aug[i][j] = a[i][j];
		}
		aug[i][N + i] = 1.0;
	}

	let mut det = 1.0f32;
	for col in 0..N {
		let mut piv = col;
		let mut maxv = aug[piv][col].abs();
		for (r, item) in aug.iter().enumerate().take(N).skip(col + 1) {
			let v = item[col].abs();
			if v > maxv {
				maxv = v;
				piv = r;
			}
		}
		if maxv < 1e-8 {
			return (None, 0.0);
		}
		if piv != col {
			aug.swap(piv, col);
			det = -det;
		}
		let pivot_val = aug[col][col];
		det *= pivot_val;
		for aug_inner in aug.iter_mut().take(2 * N).skip(col) {
			aug_inner[col] /= pivot_val;
		}
		for r in 0..N {
			if r == col { continue; }
			let factor = aug[r][col];
			if factor.abs() > 0.0 {
				for aug_inner in aug.iter_mut().take(2 * N).skip(col) {
					aug_inner[r] -= factor * aug_inner[col];
				}
			}
		}
	}

	let mut inv = [[0.0f32; N]; N];
	for i in 0..N {
		for j in 0..N {
			inv[i][j] = aug[i][N + j];
		}
	}
	(Some(inv), det)
}

pub(crate) fn mat_mul_vec<const N: usize>(a: &Mat<N>, b: &[f32; N]) -> [f32; N] {
	let mut c = [0.0; N];
	for i in 0..N {
		let mut sum = 0.0;
		for (k, item) in a.iter().enumerate() {
			sum += item[i] * b[k];
		}
		c[i] = sum;
	}
	c
}