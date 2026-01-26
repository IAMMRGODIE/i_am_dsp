//! A reverb effect that combines multiple reverbs to create a more natural sounding effect.

use i_am_dsp_derive::Parameters;

use crate::{prelude::{AllpassFilter, CombsFilter, FftConvolver, Lowpass, RayTracing, Reverb}, Effect};

const IR_SIZE: usize = 6144;

/// A reverb effect that combines multiple reverbs to create a more natural sounding effect.
#[derive(Parameters)]
pub struct HybridReverb<
	const CHANNELS: usize, 
	const DELAY_LINES: usize
> {
	#[sub_param]
	/// The combs filters used to create early reflections.
	pub comb_filters: [CombsFilter<CHANNELS>; 4],
	#[sub_param]
	/// The all pass filters used to create late reflections.
	pub allpass_filters: [AllpassFilter<CHANNELS>; 4],
	#[sub_param]
	/// The physical simulate ir generator create the reverb behind the early reflections.
	pub ir_gen: RayTracing,
	#[sub_param]
	/// The convolver used to apply the reverb to the input signal.
	pub pre_convolver: FftConvolver<CHANNELS>,
	#[sub_param]
	/// The FDN reverb used to create tails.
	pub fdn_reverb: Reverb<Lowpass<CHANNELS>, DELAY_LINES, CHANNELS>,
	#[sub_param]
	/// The convolver used to apply the tails to the input signal.
	pub post_convolver: FftConvolver<CHANNELS>,
	#[range(min = 0.01, max = 1.0)]
	#[logarithmic]
	/// The wet gain of the effect.
	pub wet_gain: f32,
	#[range(min = 0.01, max = 1.0)]
	#[logarithmic]
	/// The dry gain of the effect.
	pub dry_gain: f32,
}

impl<const CHANNELS: usize, const DELAY_LINES: usize> HybridReverb<CHANNELS, DELAY_LINES> {
	fn gen_ir(ir_gen: &RayTracing) -> [Vec<f32>; CHANNELS] {
		let single_ir = ir_gen.generate_ir(Some(IR_SIZE));
		core::array::from_fn(|_| {
			single_ir.iter().map(|f| *f * rand::random_range(0.8..=1.0)).collect::<Vec<_>>()
		})
	}

	/// Creates a new `HybridReverb` effect.
	pub fn new(
		ir_gen: RayTracing,
		sample_rate: usize,
		dry_gain: f32,
		wet_gain: f32,
	) -> Self {
		let comb_filters = [
			CombsFilter::new(0.8, 31),
			CombsFilter::new(0.8, 37),
			CombsFilter::new(0.8, 41),
			CombsFilter::new(0.8, 53),
		];

		let allpass_filters = [
			AllpassFilter::new(0.7, 23),
			AllpassFilter::new(0.7, 31),
			AllpassFilter::new(0.7, 43),
			AllpassFilter::new(0.7, 59),
		];

		let ir_a = Self::gen_ir(&ir_gen);
		let ir_b = Self::gen_ir(&ir_gen);

		let mut pre_convolver = FftConvolver::new(ir_a, sample_rate);
		pre_convolver.dry_gain = 0.0;
		pre_convolver.wet_gain = 0.5;
		let mut post_convolver = FftConvolver::new(ir_b, sample_rate);
		post_convolver.dry_gain = 0.0;
		post_convolver.wet_gain = 0.5;

		let fdn_reverb = Reverb::new(
			Lowpass::new(sample_rate, 16000.0, 0.01),  
			sample_rate, 
			Default::default(), 
			20, 
			10.0
		);

		Self {
			comb_filters,
			allpass_filters,
			ir_gen,
			pre_convolver,
			fdn_reverb,
			post_convolver,
			wet_gain,
			dry_gain,
		}
	}

	/// Updates the impulse response of the reverb.
	/// 
	/// Should be called whenever the `ir_gen` parameter is changed.
	pub fn update_ir(&mut self) {
		self.pre_convolver.replace_ir(Self::gen_ir(&self.ir_gen));
		self.post_convolver.replace_ir(Self::gen_ir(&self.ir_gen));
	}
}

impl<const CHANNELS: usize, const DELAY_LINES: usize> Effect<CHANNELS> for HybridReverb<CHANNELS, DELAY_LINES> {
	fn delay(&self) -> usize {
		self.pre_convolver.delay() + self.post_convolver.delay() + self.fdn_reverb.delay()
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"Hybrid Reverb"
	}

	fn process(
		&mut self, 
		samples: &mut [f32; CHANNELS], 
		other: &[&[f32; CHANNELS]],
		process_context: &mut Box<dyn crate::ProcessContext>,
	) {
		let mut output = [0.0; CHANNELS];

		for combs in self.comb_filters.iter_mut() {
			let mut in_samples = *samples;
			combs.process(&mut in_samples, other, process_context);
			for (i, sample) in in_samples.into_iter().enumerate() {
				output[i] += sample / 4.0;
			}
		}

		for allpass in self.allpass_filters.iter_mut() {
			allpass.process(&mut output, other, process_context);
		}

		self.pre_convolver.process(&mut output, other, process_context);
		self.fdn_reverb.process(&mut output, other, process_context);
		self.post_convolver.process(&mut output, other, process_context);

		for (i, output) in output.into_iter().enumerate() {
			samples[i] = output * self.wet_gain + samples[i] * self.dry_gain;
		}
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		use egui::CollapsingHeader;
		use crate::{prelude::GenIr, tools::ui_tools::gain_ui};

		CollapsingHeader::new("Combs Filters")
			.id_salt(format!("{}_hybrid_reverb_comb", id_prefix))
			.show(ui, |ui| {
				for combs in self.comb_filters.iter_mut() {
					combs.demo_ui(ui, format!("{}_hybrid_reverb_comb", id_prefix));
				}
			});

		CollapsingHeader::new("Allpass Filters")
			.id_salt(format!("{}_hybrid_reverb_allpass", id_prefix))
			.show(ui, |ui| {
				for allpass in self.allpass_filters.iter_mut() {
					allpass.demo_ui(ui, format!("{}_hybrid_reverb_allpass", id_prefix));
				}
			});
		
		if self.ir_gen.demo_ui(ui, format!("{}_hybrid_reverb_ir_gen", id_prefix)) {
			self.update_ir();
		}

		self.fdn_reverb.demo_ui(ui, format!("{}_hybrid_reverb_fdn_reverb", id_prefix));

		gain_ui(ui, &mut self.dry_gain, Some("Dry gain".to_string()), false);
		gain_ui(ui, &mut self.wet_gain, Some("Wet gain".to_string()), false);
	}
}