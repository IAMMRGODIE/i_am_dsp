//! A spectrum drawn with a tilt, an odd/even balance and a formant.

use i_am_dsp_derive::Parameters;

use super::{FreqInfo, GenFreqInfo};

/// An additive spectrum whose controls are the spectrum itself.
///
/// Every ratio is an integer, so `AdditiveOsc` can hold the whole sum in a
/// band limited wavetable built by an inverse FFT: changing a control rebuilds
/// that table, and playback stays one lookup per sample.
#[derive(Debug, Clone, PartialEq, Parameters)]
pub struct SpectralGen {
	/// How fast the partials fall away: amplitude is `1 / n^tilt`.
	#[range(min = 0.0, max = 4.0)]
	#[default(1.0)]
	#[id(display = "Tilt")]
	pub tilt: f32,
	/// How much the even partials are pulled out, 1.0 for odd partials only.
	#[range(min = 0.0, max = 1.0)]
	#[default(0.0)]
	#[id(display = "Odd")]
	pub odd: f32,
	/// How much a formant lifts the partials around the centre.
	#[range(min = 0.0, max = 4.0)]
	#[default(0.0)]
	#[id(display = "Formant")]
	pub formant: f32,
	/// The formant's centre, as a partial ratio.
	#[range(min = 1.0, max = 16.0)]
	#[logarithmic]
	#[default(4.0)]
	#[id(display = "Centre")]
	pub formant_center: f32,
}

impl Default for SpectralGen {
	fn default() -> Self {
		Self {
			tilt: 1.0,
			odd: 0.0,
			formant: 0.0,
			formant_center: 4.0,
		}
	}
}

impl GenFreqInfo for SpectralGen {
	fn revision(&self) -> u64 {
		let mut hash = 0xcbf2_9ce4_8422_2325u64;

		for value in [self.tilt, self.odd, self.formant, self.formant_center] {
			hash = (hash ^ value.to_bits() as u64).wrapping_mul(0x0000_0100_0000_01b3);
		}

		hash
	}

	fn gen_info(&mut self, index: usize, total_amount: usize) -> FreqInfo {
		if total_amount == 0 {
			return Default::default();
		} else if total_amount == 1 {
			return FreqInfo {
				ratio: 1.0,
				amplitude: 1.0,
				damping: 0.0,
				phase: 0.0,
			};
		}

		let partial = (index + 1) as f32;
		let mut amplitude = partial.powf(-self.tilt.clamp(0.0, 8.0));

		// index 0 is the fundamental, so an odd index is an even partial.
		if index % 2 == 1 {
			amplitude *= 1.0 - self.odd.clamp(0.0, 1.0);
		}

		if self.formant > 0.0 {
			let distance = (partial / self.formant_center.max(1.0)).ln() / 0.5;
			amplitude *= 1.0 + self.formant * (-distance * distance).exp();
		}

		FreqInfo {
			ratio: partial,
			amplitude,
			damping: 0.0,
			phase: 0.0,
		}
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) -> bool {
		use egui::*;

		let mut change = self.clone();

		Grid::new(format!("{}_spectral_gen", id_prefix))
			.num_columns(2)
			.show(ui, |ui| {
				ui.label("Tilt");
				ui.add(Slider::new(&mut change.tilt, 0.0..=4.0));
				ui.end_row();

				ui.label("Odd");
				ui.add(Slider::new(&mut change.odd, 0.0..=1.0));
				ui.end_row();

				ui.label("Formant");
				ui.add(Slider::new(&mut change.formant, 0.0..=4.0));
				ui.end_row();

				ui.label("Centre");
				ui.add(Slider::new(&mut change.formant_center, 1.0..=16.0).logarithmic(true));
				ui.end_row();
			});

		if change != *self {
			*self = change;
			true
		} else {
			false
		}
	}
}
