//! A six-band parametric EQ: bell, shelf, cut and notch bands in series.
//!
//! The bands are minimum-phase RBJ biquads. The page draws the same transfer
//! function from the same coefficients, so the curve is the filter, not an
//! approximation of it.

use i_am_dsp_derive::Parameters;

use super::biquad_design as design;
use crate::effects::filter::Biquad;
use crate::{Effect, ProcessContext};

/// The number of bands.
pub const EQ_BANDS: usize = 6;

/// The most sections one band can use: a 48 dB/oct cut is four 12 dB/oct
/// biquads in series.
const MAX_STAGES: usize = 4;

/// One band's current settings, kept so the coefficients are only rebuilt when
/// a control actually moves.
#[derive(Clone, Copy, PartialEq)]
struct BandSettings {
	on: bool,
	kind: i32,
	frequency: f32,
	gain: f32,
	q: f32,
	slope: i32,
}

impl Default for BandSettings {
	fn default() -> Self {
		Self {
			on: false,
			kind: -1,
			frequency: 1_000.0,
			gain: 0.0,
			q: 0.707,
			slope: 0,
		}
	}
}

/// The RBJ coefficients for one band.
///
/// kind is the faceplate's order: 0 bell, 1 low shelf, 2 high shelf, 3 low cut,
/// 4 high cut, 5 notch. Low cut is a highpass and high cut a lowpass.
fn coefficients(
	kind: i32,
	frequency: f32,
	gain_db: f32,
	q: f32,
	sample_rate: f32,
) -> ([f32; 3], [f32; 2]) {
	let cutoff = frequency.clamp(10.0, sample_rate * 0.49);
	let w0 = design::w0(sample_rate, cutoff);
	let alpha = design::alpha_for_q(w0, q.clamp(0.05, 20.0));

	match kind {
		// Bell.
		0 => design::bell(w0, alpha, gain_db),
		// Low shelf.
		1 => design::low_shelf(w0, alpha, gain_db),
		// High shelf.
		2 => design::high_shelf(w0, alpha, gain_db),
		// Low cut (highpass).
		3 => design::high_pass(w0, alpha),
		// High cut (lowpass).
		4 => design::low_pass(w0, alpha),
		// Notch.
		_ => design::band_stop(w0, alpha),
	}
}

/// The number of 12 dB/oct sections a band uses. Only the cut types read the
/// slope; a bell, a shelf and a notch are always one section.
fn stage_count(settings: &BandSettings) -> usize {
	if settings.kind == 3 || settings.kind == 4 {
		(settings.slope.clamp(0, 3) + 1) as usize
	} else {
		1
	}
}

/// A six-band parametric EQ.
#[derive(Parameters)]
pub struct ParametricEq<const CHANNELS: usize = 2> {
	/// Which band the faceplate edits.
	#[range(min = 0, max = 5)]
	#[choices("1", "2", "3", "4", "5", "6")]
	#[default(0)]
	#[id(display = "Band")]
	#[layout(group = "EQ", order = 0, widget = choice, hero = "eq")]
	pub band: i32,

	/// Band 1 on/off.
	#[default(true)]
	#[id(display = "On")]
	#[layout(group = "EQ", order = 1, widget = toggle)]
	pub b1_on: bool,
	/// Band 1 type.
	#[range(min = 0, max = 5)]
	#[choices("Bell", "Low Shelf", "High Shelf", "Low Cut", "High Cut", "Notch")]
	#[default(3)]
	#[id(display = "Type")]
	#[layout(group = "EQ", order = 2, widget = choice)]
	pub b1_type: i32,
	/// Band 1 frequency.
	#[range(min = 20.0, max = 20_000.0)]
	#[logarithmic]
	#[unit(hertz)]
	#[default(20.0)]
	#[id(display = "Freq")]
	#[layout(group = "EQ", order = 3, widget = knob)]
	pub b1_freq: f32,
	/// Band 1 gain.
	#[range(min = -24.0, max = 24.0)]
	#[unit(decibels)]
	#[default(0.0)]
	#[id(display = "Gain")]
	#[layout(group = "EQ", order = 4, widget = knob)]
	pub b1_gain: f32,
	/// Band 1 Q.
	#[range(min = 0.1, max = 10.0)]
	#[logarithmic]
	#[default(0.707)]
	#[id(display = "Q")]
	#[layout(group = "EQ", order = 5, widget = knob)]
	pub b1_q: f32,

	/// Band 1 cutoff slope, for the cut types.
	#[range(min = 0, max = 3)]
	#[choices("12 dB/oct", "24 dB/oct", "36 dB/oct", "48 dB/oct")]
	#[default(0)]
	#[id(display = "Slope")]
	#[layout(group = "EQ", order = 6, widget = choice)]
	pub b1_slope: i32,

	/// Band 2 on/off.
	#[default(true)]
	#[id(display = "On")]
	#[layout(group = "EQ", order = 6, widget = toggle)]
	pub b2_on: bool,
	/// Band 2 type.
	#[range(min = 0, max = 5)]
	#[choices("Bell", "Low Shelf", "High Shelf", "Low Cut", "High Cut", "Notch")]
	#[default(0)]
	#[id(display = "Type")]
	#[layout(group = "EQ", order = 7, widget = choice)]
	pub b2_type: i32,
	/// Band 2 frequency.
	#[range(min = 20.0, max = 20_000.0)]
	#[logarithmic]
	#[unit(hertz)]
	#[default(100.0)]
	#[id(display = "Freq")]
	#[layout(group = "EQ", order = 8, widget = knob)]
	pub b2_freq: f32,
	/// Band 2 gain.
	#[range(min = -24.0, max = 24.0)]
	#[unit(decibels)]
	#[default(0.0)]
	#[id(display = "Gain")]
	#[layout(group = "EQ", order = 9, widget = knob)]
	pub b2_gain: f32,
	/// Band 2 Q.
	#[range(min = 0.1, max = 10.0)]
	#[logarithmic]
	#[default(0.707)]
	#[id(display = "Q")]
	#[layout(group = "EQ", order = 10, widget = knob)]
	pub b2_q: f32,

	/// Band 2 cutoff slope, for the cut types.
	#[range(min = 0, max = 3)]
	#[choices("12 dB/oct", "24 dB/oct", "36 dB/oct", "48 dB/oct")]
	#[default(0)]
	#[id(display = "Slope")]
	#[layout(group = "EQ", order = 11, widget = choice)]
	pub b2_slope: i32,

	/// Band 3 on/off.
	#[default(true)]
	#[id(display = "On")]
	#[layout(group = "EQ", order = 11, widget = toggle)]
	pub b3_on: bool,
	/// Band 3 type.
	#[range(min = 0, max = 5)]
	#[choices("Bell", "Low Shelf", "High Shelf", "Low Cut", "High Cut", "Notch")]
	#[default(0)]
	#[id(display = "Type")]
	#[layout(group = "EQ", order = 12, widget = choice)]
	pub b3_type: i32,
	/// Band 3 frequency.
	#[range(min = 20.0, max = 20_000.0)]
	#[logarithmic]
	#[unit(hertz)]
	#[default(400.0)]
	#[id(display = "Freq")]
	#[layout(group = "EQ", order = 13, widget = knob)]
	pub b3_freq: f32,
	/// Band 3 gain.
	#[range(min = -24.0, max = 24.0)]
	#[unit(decibels)]
	#[default(0.0)]
	#[id(display = "Gain")]
	#[layout(group = "EQ", order = 14, widget = knob)]
	pub b3_gain: f32,
	/// Band 3 Q.
	#[range(min = 0.1, max = 10.0)]
	#[logarithmic]
	#[default(1.0)]
	#[id(display = "Q")]
	#[layout(group = "EQ", order = 15, widget = knob)]
	pub b3_q: f32,

	/// Band 3 cutoff slope, for the cut types.
	#[range(min = 0, max = 3)]
	#[choices("12 dB/oct", "24 dB/oct", "36 dB/oct", "48 dB/oct")]
	#[default(0)]
	#[id(display = "Slope")]
	#[layout(group = "EQ", order = 16, widget = choice)]
	pub b3_slope: i32,

	/// Band 4 on/off.
	#[default(true)]
	#[id(display = "On")]
	#[layout(group = "EQ", order = 16, widget = toggle)]
	pub b4_on: bool,
	/// Band 4 type.
	#[range(min = 0, max = 5)]
	#[choices("Bell", "Low Shelf", "High Shelf", "Low Cut", "High Cut", "Notch")]
	#[default(0)]
	#[id(display = "Type")]
	#[layout(group = "EQ", order = 17, widget = choice)]
	pub b4_type: i32,
	/// Band 4 frequency.
	#[range(min = 20.0, max = 20_000.0)]
	#[logarithmic]
	#[unit(hertz)]
	#[default(2_000.0)]
	#[id(display = "Freq")]
	#[layout(group = "EQ", order = 18, widget = knob)]
	pub b4_freq: f32,
	/// Band 4 gain.
	#[range(min = -24.0, max = 24.0)]
	#[unit(decibels)]
	#[default(0.0)]
	#[id(display = "Gain")]
	#[layout(group = "EQ", order = 19, widget = knob)]
	pub b4_gain: f32,
	/// Band 4 Q.
	#[range(min = 0.1, max = 10.0)]
	#[logarithmic]
	#[default(1.0)]
	#[id(display = "Q")]
	#[layout(group = "EQ", order = 20, widget = knob)]
	pub b4_q: f32,

	/// Band 4 cutoff slope, for the cut types.
	#[range(min = 0, max = 3)]
	#[choices("12 dB/oct", "24 dB/oct", "36 dB/oct", "48 dB/oct")]
	#[default(0)]
	#[id(display = "Slope")]
	#[layout(group = "EQ", order = 21, widget = choice)]
	pub b4_slope: i32,

	/// Band 5 on/off.
	#[default(true)]
	#[id(display = "On")]
	#[layout(group = "EQ", order = 21, widget = toggle)]
	pub b5_on: bool,
	/// Band 5 type.
	#[range(min = 0, max = 5)]
	#[choices("Bell", "Low Shelf", "High Shelf", "Low Cut", "High Cut", "Notch")]
	#[default(2)]
	#[id(display = "Type")]
	#[layout(group = "EQ", order = 22, widget = choice)]
	pub b5_type: i32,
	/// Band 5 frequency.
	#[range(min = 20.0, max = 20_000.0)]
	#[logarithmic]
	#[unit(hertz)]
	#[default(8_000.0)]
	#[id(display = "Freq")]
	#[layout(group = "EQ", order = 23, widget = knob)]
	pub b5_freq: f32,
	/// Band 5 gain.
	#[range(min = -24.0, max = 24.0)]
	#[unit(decibels)]
	#[default(0.0)]
	#[id(display = "Gain")]
	#[layout(group = "EQ", order = 24, widget = knob)]
	pub b5_gain: f32,
	/// Band 5 Q.
	#[range(min = 0.1, max = 10.0)]
	#[logarithmic]
	#[default(0.707)]
	#[id(display = "Q")]
	#[layout(group = "EQ", order = 25, widget = knob)]
	pub b5_q: f32,

	/// Band 5 cutoff slope, for the cut types.
	#[range(min = 0, max = 3)]
	#[choices("12 dB/oct", "24 dB/oct", "36 dB/oct", "48 dB/oct")]
	#[default(0)]
	#[id(display = "Slope")]
	#[layout(group = "EQ", order = 26, widget = choice)]
	pub b5_slope: i32,

	/// Band 6 on/off.
	#[default(false)]
	#[id(display = "On")]
	#[layout(group = "EQ", order = 26, widget = toggle)]
	pub b6_on: bool,
	/// Band 6 type.
	#[range(min = 0, max = 5)]
	#[choices("Bell", "Low Shelf", "High Shelf", "Low Cut", "High Cut", "Notch")]
	#[default(4)]
	#[id(display = "Type")]
	#[layout(group = "EQ", order = 27, widget = choice)]
	pub b6_type: i32,
	/// Band 6 frequency.
	#[range(min = 20.0, max = 20_000.0)]
	#[logarithmic]
	#[unit(hertz)]
	#[default(20_000.0)]
	#[id(display = "Freq")]
	#[layout(group = "EQ", order = 28, widget = knob)]
	pub b6_freq: f32,
	/// Band 6 gain.
	#[range(min = -24.0, max = 24.0)]
	#[unit(decibels)]
	#[default(0.0)]
	#[id(display = "Gain")]
	#[layout(group = "EQ", order = 29, widget = knob)]
	pub b6_gain: f32,
	/// Band 6 Q.
	#[range(min = 0.1, max = 10.0)]
	#[logarithmic]
	#[default(0.707)]
	#[id(display = "Q")]
	#[layout(group = "EQ", order = 30, widget = knob)]
	pub b6_q: f32,

	/// Band 6 cutoff slope, for the cut types.
	#[range(min = 0, max = 3)]
	#[choices("12 dB/oct", "24 dB/oct", "36 dB/oct", "48 dB/oct")]
	#[default(0)]
	#[id(display = "Slope")]
	#[layout(group = "EQ", order = 31, widget = choice)]
	pub b6_slope: i32,

	/// Gain after the bands.
	#[range(min = -12.0, max = 12.0)]
	#[unit(decibels)]
	#[default(0.0)]
	#[id(display = "Output")]
	#[layout(group = "EQ", order = 31, widget = knob)]
	pub output: f32,

	#[skip]
	sample_rate: f32,
	#[skip]
	filters: [[Biquad<CHANNELS>; MAX_STAGES]; EQ_BANDS],
	#[skip]
	last: [BandSettings; EQ_BANDS],
}

impl<const CHANNELS: usize> ParametricEq<CHANNELS> {
	/// A flat six-band EQ: a 20 Hz low cut, four neutral bells/shelves and a
	/// high cut that starts off.
	pub fn new(sample_rate: usize) -> Self {
		let mut effect = Self {
			band: 0,
			b1_on: true,
			b1_type: 3,
			b1_freq: 20.0,
			b1_gain: 0.0,
			b1_q: 0.707,
			b1_slope: 0,
			b2_on: true,
			b2_type: 0,
			b2_freq: 100.0,
			b2_gain: 0.0,
			b2_q: 0.707,
			b2_slope: 0,
			b3_on: true,
			b3_type: 0,
			b3_freq: 400.0,
			b3_gain: 0.0,
			b3_q: 1.0,
			b3_slope: 0,
			b4_on: true,
			b4_type: 0,
			b4_freq: 2_000.0,
			b4_gain: 0.0,
			b4_q: 1.0,
			b4_slope: 0,
			b5_on: true,
			b5_type: 2,
			b5_freq: 8_000.0,
			b5_gain: 0.0,
			b5_q: 0.707,
			b5_slope: 0,
			b6_on: false,
			b6_type: 4,
			b6_freq: 20_000.0,
			b6_gain: 0.0,
			b6_q: 0.707,
			b6_slope: 0,
			output: 0.0,
			sample_rate: sample_rate as f32,
			filters: core::array::from_fn(|_| core::array::from_fn(|_| Biquad::new(sample_rate))),
			last: [BandSettings::default(); EQ_BANDS],
		};

		effect.update();
		effect
	}

	fn settings(&self, index: usize) -> BandSettings {
		match index {
			0 => BandSettings {
				on: self.b1_on,
				kind: self.b1_type,
				frequency: self.b1_freq,
				gain: self.b1_gain,
				q: self.b1_q,
				slope: self.b1_slope,
			},
			1 => BandSettings {
				on: self.b2_on,
				kind: self.b2_type,
				frequency: self.b2_freq,
				gain: self.b2_gain,
				q: self.b2_q,
				slope: self.b2_slope,
			},
			2 => BandSettings {
				on: self.b3_on,
				kind: self.b3_type,
				frequency: self.b3_freq,
				gain: self.b3_gain,
				q: self.b3_q,
				slope: self.b3_slope,
			},
			3 => BandSettings {
				on: self.b4_on,
				kind: self.b4_type,
				frequency: self.b4_freq,
				gain: self.b4_gain,
				q: self.b4_q,
				slope: self.b4_slope,
			},
			4 => BandSettings {
				on: self.b5_on,
				kind: self.b5_type,
				frequency: self.b5_freq,
				gain: self.b5_gain,
				q: self.b5_q,
				slope: self.b5_slope,
			},
			_ => BandSettings {
				on: self.b6_on,
				kind: self.b6_type,
				frequency: self.b6_freq,
				gain: self.b6_gain,
				q: self.b6_q,
				slope: self.b6_slope,
			},
		}
	}

	fn update(&mut self) {
		for index in 0..EQ_BANDS {
			let setting = self.settings(index);
			let was_on = self.last[index].on;

			if setting == self.last[index] {
				continue;
			}

			let previous_stages = stage_count(&self.last[index]);
			let stages = stage_count(&setting);

			self.last[index] = setting;

			if !setting.on {
				continue;
			}

			let filters = &mut self.filters[index];

			if !was_on {
				for filter in filters.iter_mut() {
					filter.clear_state();
				}
			} else if stages > previous_stages {
				// A new stage starts from rest; the older ones keep their state so
				// turning the slope up does not click.
				for filter in filters.iter_mut().take(stages).skip(previous_stages) {
					filter.clear_state();
				}
			}

			let (b, a) = coefficients(
				setting.kind,
				setting.frequency,
				setting.gain,
				setting.q,
				self.sample_rate,
			);

			for filter in filters.iter_mut().take(stages) {
				filter.b = b;
				filter.a = a;
			}
		}
	}
}

impl<const CHANNELS: usize> Effect<CHANNELS> for ParametricEq<CHANNELS> {
	fn delay(&self) -> usize {
		0
	}

	fn process(
		&mut self,
		samples: &mut [f32; CHANNELS],
		_other: &[&[f32; CHANNELS]],
		_: &mut Box<dyn ProcessContext>,
	) {
		self.update();
		let mut frame = *samples;

		for (index, filters) in self.filters.iter_mut().enumerate() {
			if !self.last[index].on {
				continue;
			}

			for filter in filters.iter_mut().take(stage_count(&self.last[index])) {
				frame = filter.process_sample(frame);
			}
		}

		let output = 10.0_f32.powf(self.output / 20.0);

		for (sample, value) in samples.iter_mut().zip(frame.iter()) {
			*sample = value * output;
		}
	}

	fn clear_tail(&mut self) {
		for filters in self.filters.iter_mut() {
			for filter in filters.iter_mut() {
				filter.clear_state();
			}
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn process(effect: &mut ParametricEq<2>, frames: usize) -> f32 {
		let mut context: Box<dyn ProcessContext> = Box::new(());
		let mut energy = 0.0;

		for index in 0..frames {
			let phase = 2.0 * std::f32::consts::PI * 1_000.0 * index as f32 / 48_000.0;
			let value = 0.5 * phase.sin();
			let mut samples = [value, value];
			effect.process(&mut samples, &[], &mut context);

			if index >= 4_096 {
				energy += samples[0] * samples[0];
			}
		}

		(energy / (frames - 4_096) as f32).sqrt()
	}

	/// A flat EQ is flat.
	#[test]
	fn a_flat_eq_passes_the_signal() {
		let mut effect = ParametricEq::<2>::new(48_000);
		let output = process(&mut effect, 8_192);

		assert!((output - 0.353_553).abs() < 0.005, "{output}");
	}

	/// A bell at the test tone's frequency boosts it by its gain.
	#[test]
	fn a_bell_boosts_its_centre() {
		let mut effect = ParametricEq::<2>::new(48_000);
		effect.b3_freq = 1_000.0;
		effect.b3_gain = 12.0;
		effect.b3_q = 1.0;
		let output = process(&mut effect, 8_192);

		assert!(output > 0.7, "{output}");
	}

	/// A low cut removes a low tone.
	#[test]
	fn a_low_cut_removes_the_low_end() {
		let mut effect = ParametricEq::<2>::new(48_000);
		effect.b1_freq = 500.0;
		let mut context: Box<dyn ProcessContext> = Box::new(());
		let mut energy = 0.0;

		for index in 0..8_192 {
			let phase = 2.0 * std::f32::consts::PI * 100.0 * index as f32 / 48_000.0;
			let mut samples = [0.5 * phase.sin(), 0.5 * phase.sin()];
			effect.process(&mut samples, &[], &mut context);

			if index >= 4_096 {
				energy += samples[0] * samples[0];
			}
		}

		let rms = (energy / 4_096.0).sqrt();
		assert!(rms < 0.03, "{rms}");
	}

	/// More sections make a cut steeper.
	#[test]
	fn the_slope_stacks_cut_sections() {
		let render = |slope: i32| {
			let mut effect = ParametricEq::<2>::new(48_000);
			effect.b1_freq = 500.0;
			effect.b1_slope = slope;
			let mut context: Box<dyn ProcessContext> = Box::new(());
			let mut energy = 0.0;

			for index in 0..8_192 {
				let phase = 2.0 * std::f32::consts::PI * 100.0 * index as f32 / 48_000.0;
				let mut samples = [0.5 * phase.sin(), 0.5 * phase.sin()];
				effect.process(&mut samples, &[], &mut context);

				if index >= 4_096 {
					energy += samples[0] * samples[0];
				}
			}

			(energy / 4_096.0).sqrt()
		};

		let shallow = render(0);
		let steep = render(3);
		assert!(
			steep < shallow * 0.05,
			"12 dB/oct {shallow}, 48 dB/oct {steep}"
		);
	}

	/// The band count and channel count are generic.
	#[test]
	fn the_eq_is_generic_over_channels() {
		let mut effect = ParametricEq::<1>::new(48_000);
		let mut context: Box<dyn ProcessContext> = Box::new(());

		for _ in 0..1_024 {
			let mut samples = [0.5];
			effect.process(&mut samples, &[], &mut context);
			assert!(samples[0].is_finite());
		}
	}
}
