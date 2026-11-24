//! A spectrum that bends a saw: the first generator the oscillator plays.

use std::f32::consts::PI;

use i_am_dsp_derive::Parameters;

use crate::prelude::bend;

use super::{FreqInfo, GenFreqInfo};

/// A Simple Bend Frequency Generator
#[derive(Debug, Clone, PartialEq, Parameters)]
pub struct BendedSawGen {
	/// How much the frequency should be bent
	///
	/// default to 0.0
	#[range(min = -10.0, max = 10.0)]
	#[default(0.0)]
	#[id(display = "Bend")]
	pub bend_amount: f32,
	/// must be greater than 0.0
	///
	/// default to 1.0
	#[range(min = 0.125, max = 8.0)]
	#[logarithmic]
	#[default(1.0)]
	#[id(display = "Center")]
	pub center_space: f32,
	/// How much the bend should be shifted
	///
	/// default to 0.0
	#[range(min = -10.0, max = 10.0)]
	#[default(0.0)]
	#[id(display = "Overall")]
	pub total_bend_amount: f32,
	/// How much the frequency should be scaled
	///
	/// default to 1.0
	#[range(min = 0.0, max = 1.0)]
	#[default(1.0)]
	#[id(display = "Scale")]
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
	fn revision(&self) -> u64 {
		// Four floats are all that change the partials; a small FNV-style hash is
		// enough to tell a host's edit from a repeated read.
		let mut hash = 0xcbf2_9ce4_8422_2325u64;

		for value in [
			self.bend_amount,
			self.center_space,
			self.total_bend_amount,
			self.scale_amount,
		] {
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
		let amplitude = (2.0 / PI) * (-1.0_f32).powf(index) / (index + 1.0);

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
			.show(ui, |ui| {
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
		} else {
			false
		}
	}
}
