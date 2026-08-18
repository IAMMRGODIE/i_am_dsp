//! Hilbert transform implementation using IIR all-pass filters.
//! 
//! This module provides a Hilbert transform that outputs complex analytic signals
//! (real + imaginary parts) using cascaded all-pass filters with pre-computed coefficients.

use crate::parameters::{Parameter, Parameters, SetValue};

/// Coefficients for a single all-pass section
#[derive(Debug, Clone)]
struct AllPassCoefficients {
	/// Numerator coefficients (b)
	#[allow(dead_code)]
	pub b: Vec<f64>,
	/// Denominator coefficients (a), excluding a0=1
	#[allow(dead_code)]
	pub a: Vec<f64>,
}

/// Hilbert transform filter bank for a specific order
#[derive(Debug, Clone)]
struct HilbertFilterBank {
	#[allow(dead_code)]
	pub a0_sections: Vec<AllPassCoefficients>,
	#[allow(dead_code)]
	pub a1_sections: Vec<AllPassCoefficients>,
}

impl HilbertFilterBank {
	/// Get filter coefficients for the specified order (2-12)
	fn from_order(order: usize) -> Option<Self> {
		match order {
			2 => Some(Self {
				a0_sections: vec![AllPassCoefficients {
					b: vec![-0.440684, -0.481803, 1.0],
					a: vec![-0.481803, -0.440684],
				}],
				a1_sections: vec![AllPassCoefficients {
					b: vec![-0.440684, 0.481803, 1.0],
					a: vec![0.481803, -0.440684],
				}],
			}),
			3 => Some(Self {
				a0_sections: vec![AllPassCoefficients {
					b: vec![0.251474, -0.71074, -0.501994, 1.0],
					a: vec![-0.501994, -0.71074, 0.251474],
				}],
				a1_sections: vec![AllPassCoefficients {
					b: vec![-0.25148, -0.710734, 0.502008, 1.0],
					a: vec![0.502008, -0.710734, -0.25148],
				}],
			}),
			4 => Some(Self {
				a0_sections: vec![AllPassCoefficients {
					b: vec![0.148243, 0.372223, -1.00272, -0.499732, 1.0],
					a: vec![-0.499732, -1.00272, 0.372223, 0.148243],
				}],
				a1_sections: vec![AllPassCoefficients {
					b: vec![0.148243, -0.372222, -1.00272, 0.499732, 1.0],
					a: vec![0.499732, -1.00272, -0.372222, 0.148243],
				}],
			}),
			5 => Some(Self {
				a0_sections: vec![AllPassCoefficients {
					b: vec![-0.0884212, 0.36949, 0.524555, -1.29744, -0.499987, 1.0],
					a: vec![-0.499987, -1.29744, 0.524555, 0.36949, -0.0884212],
				}],
				a1_sections: vec![AllPassCoefficients {
					b: vec![0.0884211, 0.36949, -0.524555, -1.29744, 0.499986, 1.0],
					a: vec![0.499986, -1.29744, -0.524555, 0.36949, 0.0884211],
				}],
			}),
			6 => Some(Self {
				a0_sections: vec![AllPassCoefficients {
					b: vec![-0.0530168, -0.20355, 0.682386, 0.67196, -1.59411, -0.499962, 1.0],
					a: vec![-0.499962, -1.59411, 0.67196, 0.682386, -0.20355, -0.0530168],
				}],
				a1_sections: vec![AllPassCoefficients {
					b: vec![-0.0530168, 0.20355, 0.682387, -0.671959, -1.59411, 0.499961, 1.0],
					a: vec![0.499961, -1.59411, -0.671959, 0.682387, 0.20355, -0.0530168],
				}],
			}),
			7 => Some(Self {
				a0_sections: vec![AllPassCoefficients {
					b: vec![0.0318762, -0.175188, -0.368242, 1.08401, 0.820913, -1.89171, -0.499977, 1.0],
					a: vec![-0.499977, -1.89171, 0.820913, 1.08401, -0.368242, -0.175188, 0.0318762],
				}],
				a1_sections: vec![AllPassCoefficients {
					b: vec![-0.0318757, -0.17519, 0.368235, 1.08401, -0.820898, -1.89172, 0.499968, 1.0],
					a: vec![0.499968, -1.89172, -0.820898, 1.08401, 0.368235, -0.17519, -0.0318757],
				}],
			}),
			8 => Some(Self {
				a0_sections: vec![AllPassCoefficients {
					b: vec![0.0191939, 0.0988579, -0.396236, -0.576294, 1.57513, 0.969921, -2.18985, -0.499972, 1.0],
					a: vec![-0.499972, -2.18985, 0.969921, 1.57513, -0.576294, -0.396236, 0.0988579, 0.0191939],
				}],
				a1_sections: vec![AllPassCoefficients {
					b: vec![0.0191936, -0.0988598, -0.396232, 0.576306, 1.57513, -0.969941, -2.18984, 0.499983, 1.0],
					a: vec![0.499983, -2.18984, -0.969941, 1.57513, 0.576306, -0.396232, -0.0988598, 0.0191936],
				}],
			}),
			9 => Some(Self {
				a0_sections: vec![AllPassCoefficients {
					b: vec![-0.0185115, 0.114128, 0.288976, -0.946864, -0.979066, 2.49793, 1.20744, -2.66373, -0.500211, 1.0],
					a: vec![-0.500211, -2.66373, 1.20744, 2.49793, -0.979066, -0.946864, 0.288976, 0.114128, -0.0185115],
				}],
				a1_sections: vec![AllPassCoefficients {
					b: vec![0.0184965, 0.114193, -0.288734, -0.947178, 0.978232, 2.4984, -1.2064, -2.66395, 0.499777, 1.0],
					a: vec![0.499777, -2.66395, -1.2064, 2.4984, 0.978232, -0.947178, -0.288734, 0.114193, 0.0184965],
				}],
			}),
			10 => Some(Self {
				a0_sections: vec![AllPassCoefficients {
					b: vec![-0.0118172, -0.067864, 0.299063, 0.525218, -1.57376, -1.32466, 3.27105, 1.36661, -2.98393, -0.499873, 1.0],
					a: vec![-0.499873, -2.98393, 1.36661, 3.27105, -1.32466, -1.57376, 0.525218, 0.299063, -0.067864, -0.0118172],
				}],
				a1_sections: vec![AllPassCoefficients {
					b: vec![-0.0118128, 0.0678916, 0.298993, -0.525448, -1.57352, 1.32527, 3.27076, -1.36725, -2.9838, 0.500117, 1.0],
					a: vec![0.500117, -2.9838, -1.36725, 3.27076, 1.32527, -1.57352, -0.525448, 0.298993, 0.0678916, -0.0118128],
				}],
			}),
			11 => Some(Self {
				a0_sections: vec![AllPassCoefficients {
					b: vec![0.0199294, -0.107216, -0.331088, 1.01663, 1.36759, -3.34595, -2.36392, 5.06862, 1.85181, -3.63207, -0.54434, 1.0],
					a: vec![-0.54434, -3.63207, 1.85181, 5.06862, -2.36392, -3.34595, 1.36759, 1.01663, -0.331088, -0.107216, 0.0199294],
				}],
				a1_sections: vec![AllPassCoefficients {
					b: vec![-0.0169911, -0.119998, 0.279881, 1.09243, -1.15163, -3.50323, 1.98552, 5.20722, -1.55243, -3.67641, 0.455659, 1.0],
					a: vec![0.455659, -3.67641, -1.55243, 5.20722, 1.98552, -3.50323, -1.15163, 1.09243, 0.279881, -0.119998, -0.0169911],
				}],
			}),
			12 => Some(Self {
				a0_sections: vec![AllPassCoefficients {
					b: vec![0.00171837, 0.0445413, -0.141178, -0.495237, 1.1661, 1.77257, -3.61443, -2.83541, 5.28473, 2.11315, -3.69693, -0.599625, 1.0],
					a: vec![-0.599625, -3.69693, 2.11315, 5.28473, -2.83541, -3.61443, 1.77257, 1.1661, -0.495237, -0.141178, 0.0445413, 0.00171837],
				}],
				a1_sections: vec![AllPassCoefficients {
					b: vec![0.00539638, -0.0217539, -0.202435, 0.283653, 1.41853, -1.0863, -4.04949, 1.8076, 5.62455, -1.38356, -3.79653, 0.40037, 1.0],
					a: vec![0.40037, -3.79653, -1.38356, 5.62455, 1.8076, -4.04949, -1.0863, 1.41853, 0.283653, -0.202435, -0.0217539, 0.00539638],
				}],
			}),
			_ => None,
		}
	}
}

/// State for a single all-pass filter section
#[derive(Debug, Clone)]
struct AllPassState {
	x_history: Vec<f64>,
	y_history: Vec<f64>,
}

impl AllPassState {
	/// Create a new all-pass state with the given filter order
	fn new(order: usize) -> Self {
		Self {
			x_history: vec![0.0; order],
			y_history: vec![0.0; order],
		}
	}

	/// Process one sample through this all-pass section
	fn process(&mut self, input: f64, coeffs: &AllPassCoefficients) -> f64 {
		let n = coeffs.b.len() - 1;
		
		let mut numerator = 0.0;
		for k in 0..=n {
			let x_val = if k == 0 {
				input
			} else if k - 1 < self.x_history.len() {
				self.x_history[k - 1]
			} else {
				0.0
			};
			numerator += coeffs.b[k] * x_val;
		}

		let mut feedback = 0.0;
		for k in 1..=n {
			let y_val = if k - 1 < self.y_history.len() {
				self.y_history[k - 1]
			} else {
				0.0
			};
			feedback += coeffs.a[k - 1] * y_val;
		}

		let output = numerator - feedback;

		if n > 0 {
			for i in (1..n).rev() {
				if i < self.x_history.len() {
					self.x_history[i] = self.x_history[i - 1];
				}
				if i < self.y_history.len() {
					self.y_history[i] = self.y_history[i - 1];
				}
			}
			self.x_history[0] = input;
			self.y_history[0] = output;
		}

		output
	}
}

/// Complex sample representation
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComplexSample {
	/// Real part of the complex number
	pub real: f32,
	/// Imaginary part of the complex number
	pub imag: f32,
}

impl ComplexSample {
	/// Create a new complex sample with the given real and imaginary parts.
	pub fn new(real: f32, imag: f32) -> Self {
		Self { real, imag }
	}

	/// Create a new complex sample with the given real part and zero imaginary part.
	pub fn from_real(real: f32) -> Self {
		Self { real, imag: 0.0 }
	}

	/// Create a new complex sample with the given imaginary part and zero real part.
	pub fn magnitude(&self) -> f32 {
		(self.real * self.real + self.imag * self.imag).sqrt()
	}

	/// Create a new complex sample with the given imaginary part and zero real part.
	pub fn phase(&self) -> f32 {
		self.imag.atan2(self.real)
	}
}

impl std::ops::Add for ComplexSample {
	type Output = Self;
	fn add(self, rhs: Self) -> Self {
		Self {
			real: self.real + rhs.real,
			imag: self.imag + rhs.imag,
		}
	}
}

impl std::ops::Sub for ComplexSample {
	type Output = Self;
	fn sub(self, rhs: Self) -> Self {
		Self {
			real: self.real - rhs.real,
			imag: self.imag - rhs.imag,
		}
	}
}

impl std::ops::Mul<f32> for ComplexSample {
	type Output = Self;
	fn mul(self, rhs: f32) -> Self {
		Self {
			real: self.real * rhs,
			imag: self.imag * rhs,
		}
	}
}

/// An IIR Hilbert transform that outputs complex analytic signals.
/// An IIR Hilbert transform that outputs complex analytic signals.
/// 
/// The Hilbert transform creates a 90-degree phase shift, producing an analytic signal
/// where the original is the real part and the transformed is the imaginary part.
pub struct HilbertTransform<const ORDER: usize, const CHANNELS: usize = 2> {
	filter_bank: HilbertFilterBank,
	a0_states: Vec<Vec<AllPassState>>,
	a1_states: Vec<Vec<AllPassState>>,
	#[allow(dead_code)]
	sample_rate: usize,
}

impl<const ORDER: usize, const CHANNELS: usize> HilbertTransform<ORDER, CHANNELS> {
	/// Create a new Hilbert transform with the given sample rate and filter order.
	pub fn new(sample_rate: usize) -> Self {
		assert!(CHANNELS > 0, "CHANNELS must be greater than 0");
		assert!(ORDER >= 2 && ORDER <= 12, "ORDER must be between 2 and 12");

		let filter_bank = HilbertFilterBank::from_order(ORDER)
			.expect("Invalid filter order");

		let a0_states = (0..CHANNELS)
			.map(|_| {
				filter_bank.a0_sections.iter()
					.map(|c| AllPassState::new(c.b.len() - 1))
					.collect()
			})
			.collect();

		let a1_states = (0..CHANNELS)
			.map(|_| {
				filter_bank.a1_sections.iter()
					.map(|c| AllPassState::new(c.b.len() - 1))
					.collect()
			})
			.collect();

		Self {
			filter_bank,
			a0_states,
			a1_states,
			sample_rate,
		}
	}

	/// Apply Hilbert transform to a single sample from one channel.
	/// 
	/// Returns the complex analytic signal where real part is from A0 path
	/// and imaginary part is from A1 path.
	pub fn apply_transform_single(&mut self, input: f32, channel: usize) -> ComplexSample {
		assert!(channel < CHANNELS, "Channel index out of bounds");

		let input_f64 = input as f64;

		let mut real_part = input_f64;
		for (section_idx, coeffs) in self.filter_bank.a0_sections.iter().enumerate() {
			real_part = self.a0_states[channel][section_idx].process(real_part, coeffs);
		}

		let mut imag_part = input_f64;
		for (section_idx, coeffs) in self.filter_bank.a1_sections.iter().enumerate() {
			imag_part = self.a1_states[channel][section_idx].process(imag_part, coeffs);
		}

		ComplexSample::new(real_part as f32, - imag_part as f32)
	}

	/// Apply Hilbert transform to all channels.
	/// 
	/// Returns an array of complex analytic signals, one per channel.
	pub fn apply_transform(&mut self, samples: &[f32; CHANNELS]) -> [ComplexSample; CHANNELS] {
		core::array::from_fn(|ch| self.apply_transform_single(samples[ch], ch))
	}

	/// Reset all internal filter state to zero.
	pub fn reset(&mut self) {
		for states in self.a0_states.iter_mut() {
			for state in states.iter_mut() {
				state.x_history.fill(0.0);
				state.y_history.fill(0.0);
			}
		}
		for states in self.a1_states.iter_mut() {
			for state in states.iter_mut() {
				state.x_history.fill(0.0);
				state.y_history.fill(0.0);
			}
		}
	}
}

impl<const ORDER: usize, const CHANNELS: usize> Parameters for HilbertTransform<ORDER, CHANNELS> {
	fn get_parameters(&self) -> Vec<Parameter> {
		vec![]
	}

	fn set_parameter(&mut self, _identifier: &str, _value: SetValue) -> bool {
		false
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_creation() {
		let _ht = HilbertTransform::<4, 2>::new(44100);
	}

	#[test]
	fn test_complex_sample() {
		let c = ComplexSample::new(3.0, 4.0);
		assert_eq!(c.magnitude(), 5.0);
	}
}
