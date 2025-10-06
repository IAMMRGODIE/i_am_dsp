//! This is the implementation of AdditiveOsc, which is a type of oscillator that adds up multiple sine waves.

use std::f32::consts::PI;

use i_am_parameters_derive::Parameters;

use crate::prelude::{bend, Oscillator, Parameters, SineWave, WaveTable};

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
}

impl Default for FreqInfo {
	fn default() -> Self {
		Self {
			ratio: 1.0,
			amplitude: 0.0,
		}
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
	caculated_ratios: [FreqInfo; MAX_SINES],
	max_ratio: f32,
	#[skip]
	min_ratio: f32,
	#[skip]
	max_amplitude: f32,
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
		let caculated_ratios = core::array::from_fn(|i| {
			let output = freq_gen.gen_info(i, MAX_SINES);
			max_ratio = max_ratio.max(output.ratio);
			min_ratio = min_ratio.min(output.ratio);
			max_amplitude = max_amplitude.max(output.amplitude);
			output
		});

		Self {
			freq_gen,
			caculated_ratios,
			max_amplitude,
			max_ratio,
			min_ratio: min_ratio.max(1.0),
			num_sines,
			gain: 1.0
		}
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

		let caculated_ratios = core::array::from_fn(|i| {
			let mut output = self.freq_gen.gen_info(i, MAX_SINES);
			output.ratio = output.ratio.max(1.0);
			max_ratio = max_ratio.max(output.ratio);
			min_ratio = min_ratio.min(output.ratio);
			max_amplitude = max_amplitude.max(output.amplitude);
			output
		});
		self.max_ratio = max_ratio;
		self.min_ratio = min_ratio.max(1.0);
		self.caculated_ratios = caculated_ratios;
		self.max_amplitude = max_amplitude;
	}
}

impl<
	Freq: GenFreqInfo,
	const MAX_SINES: usize,
	const CHANNELS: usize
> Oscillator<CHANNELS> for AdditiveOsc<Freq, MAX_SINES, CHANNELS> {
	fn play_at(&mut self, frequency: f32, time: f32, phase: [f32; CHANNELS]) -> [f32; CHANNELS] {
		let mut output = [0.0; CHANNELS];
		let sin = SineWave;

		for i in 0..CHANNELS {
			for j in 0..self.num_sines.min(MAX_SINES) {
				let freq = frequency * self.caculated_ratios[j].ratio;
				let t = (time * freq + phase[i]) % 1.0;
				output[i] += sin.sample(t, i) * self.caculated_ratios[j].amplitude * self.gain;
			}
		}

		output
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		use egui::*;
		use crate::tools::ui_tools::gain_ui;
		use egui::emath::RectTransform;

		Resize::default()
			.max_width(ui.available_width())
			.min_width(ui.available_width())
			.id_source(format!("{}_additive_osc_ratio", id_prefix))
			.show(ui, |ui| 
		{
			Frame::canvas(ui.style()).show(ui, |ui| {
				let (_, rect) = ui.allocate_space(ui.available_size());
				let to_screen = RectTransform::from_to(
					Rect::from_x_y_ranges(0.0..=1.0, -1.0..=1.0), 
					rect
				);

				if MAX_SINES == 1 {
					let y = self.caculated_ratios[0].amplitude;
					let pos_1 = to_screen * pos2(0.5, y);
					let pos_2 = to_screen * pos2(0.5, 0.0);
					ui.painter().line_segment([pos_1, pos_2], (3.0, Color32::WHITE));
					return;
				}

				let mid_ratio = (self.max_ratio * self.min_ratio).sqrt();
				let min_ratio = self.min_ratio.min(mid_ratio / 2.0);
				let max_ratio = self.max_ratio.max(mid_ratio * 2.0);
				let max_amp = self.max_amplitude;

				ui.painter().extend(self.caculated_ratios
					.iter()
					.take(self.num_sines)
					.map(|inner| 
				{
					let lerped_ratio = (inner.ratio.ln() - min_ratio.ln()) / (max_ratio.ln() - min_ratio.ln());
					let x = lerped_ratio * 0.99 + 0.005;
					let y = inner.amplitude / max_amp;
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
			ui.add(Slider::new(&mut self.num_sines, 1..=MAX_SINES).text("Num Sines"));
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