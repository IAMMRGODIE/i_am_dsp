use super::super::{BendedSawGen, SpectralGen};
use super::*;
use crate::prelude::{Parameter, SetValue};

const SINES: usize = 16;

/// A frequency generator that can be asked for the partials a wavetable cannot hold.
struct TestGen {
	ratio_scale: f32,
	damping: f32,
	phase_scale: f32,
}

impl TestGen {
	/// Integer ratios, no damping and no phase, which is what the fast path is for.
	fn clean() -> Self {
		Self {
			ratio_scale: 1.0,
			damping: 0.0,
			phase_scale: 0.0,
		}
	}
}

impl Parameters for TestGen {
	fn get_parameters(&self) -> Vec<Parameter> {
		Vec::new()
	}

	fn set_parameter(&mut self, _: &str, _: SetValue) -> bool {
		false
	}
}

impl GenFreqInfo for TestGen {
	fn gen_info(&mut self, index: usize, _: usize) -> FreqInfo {
		let index = (index + 1) as f32;

		FreqInfo {
			ratio: self.ratio_scale * index,
			amplitude: 1.0 / index,
			damping: self.damping,
			phase: self.phase_scale * index,
		}
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, _: &mut egui::Ui, _: String) -> bool {
		false
	}
}

/// The exact sum of all partials in f64, so that an oscillator can be compared against it.
fn reference<Freq: GenFreqInfo, const MAX_SINES: usize, const CHANNELS: usize>(
	osc: &AdditiveOsc<Freq, MAX_SINES, CHANNELS>,
	frequency: f32,
	time: f64,
	phase: [f32; CHANNELS],
	limit: f64,
) -> [f64; CHANNELS] {
	let mut output = [0.0f64; CHANNELS];

	for (i, output) in output.iter_mut().enumerate() {
		for j in 0..osc.num_sines.min(MAX_SINES) {
			// The oscillator drops the partials above the Nyquist frequency of the played note.
			if (osc.calculated_ratios[j] as f64).round() > limit {
				continue;
			}

			let cycles = time * osc.calculated_ratios[j] as f64 * frequency as f64
				+ phase[i] as f64
				+ osc.calculated_phases[j] as f64;
			let damping = (osc.calculated_dampings[j] as f64).powf(cycles);

			*output += (2.0 * std::f64::consts::PI * cycles).sin()
				* osc.calculated_amplitudes[j] as f64
				* damping * osc.gain as f64;
		}
	}

	output
}

fn max_error<Freq: GenFreqInfo, const MAX_SINES: usize, const CHANNELS: usize>(
	osc: &AdditiveOsc<Freq, MAX_SINES, CHANNELS>,
	phase: [f32; CHANNELS],
) -> f64 {
	let mut max_error = 0.0f64;

	for step in 0..=1024 {
		let time = step as f32 / 1024.0;
		let output = osc.play_at(1.0, time, phase);
		let reference = reference(osc, 1.0, time as f64, phase, osc.max_harmonic as f64);

		for (output, reference) in output.iter().zip(reference) {
			max_error = max_error.max((*output as f64 - reference).abs());
		}
	}

	max_error
}

#[test]
fn the_selected_levels_never_alias() {
	let osc: AdditiveOsc<BendedSawGen, SINES, 2> = AdditiveOsc::new(BendedSawGen::default(), SINES);
	let table = osc.table.as_ref().unwrap();

	// Below a limit of one the oscillator plays nothing at all, so only the rest is a level
	for step in 50..=4000 {
		let max_harmonic = step as f32 * 0.02;
		let (fine, coarse, blend) = table.select(max_harmonic);
		let limit = |level: usize| table.max_ratio / 2.0f32.powi(level as i32);

		assert!(
			limit(fine) <= max_harmonic,
			"level {fine} keeps partials above {max_harmonic}"
		);
		assert!(
			limit(coarse) <= max_harmonic,
			"level {coarse} keeps partials above {max_harmonic}"
		);
		assert!(fine <= coarse);
		assert!((0.0..=1.0).contains(&blend));
	}
}

#[test]
fn the_wavetable_is_band_limited_to_the_played_pitch() {
	let mut osc: AdditiveOsc<BendedSawGen, SINES, 2> =
		AdditiveOsc::new(BendedSawGen::default(), SINES);

	for pitch in [
		12000.0f32, 6000.0, 3000.0, 1500.0, 750.0, 375.0, 187.5, 93.75,
	] {
		osc.set_pitch(pitch, 48000);

		let table = osc.table.as_ref().unwrap();
		let (fine, coarse, blend) = table.select(osc.max_harmonic);
		let limit = |level: usize| (table.max_ratio / 2.0f32.powi(level as i32)) as f64;

		for step in 0..=64 {
			let time = step as f32 / 64.0;
			let output = osc.play_at(1.0, time, [0.0, 0.25]);
			let fine_reference = reference(&osc, 1.0, time as f64, [0.0, 0.25], limit(fine));
			let coarse_reference = reference(&osc, 1.0, time as f64, [0.0, 0.25], limit(coarse));

			for (i, output) in output.iter().enumerate() {
				let expected =
					fine_reference[i] * (1.0 - blend as f64) + coarse_reference[i] * blend as f64;
				let error = (*output as f64 - expected).abs();

				assert!(error < 5e-5, "pitch {pitch} time {time} error {error:e}");
			}
		}
	}
}

#[test]
fn a_pitch_above_nyquist_plays_nothing() {
	let mut osc: AdditiveOsc<BendedSawGen, SINES, 2> =
		AdditiveOsc::new(BendedSawGen::default(), SINES);
	osc.set_pitch(30000.0, 48000);

	assert!(osc.max_harmonic < 1.0);
	assert_eq!(osc.play_at(1.0, 0.31, [0.0, 0.25]), [0.0, 0.0]);
}

#[test]
fn the_sine_fallback_drops_the_partials_above_nyquist() {
	let mut osc: AdditiveOsc<BendedSawGen, SINES, 2> =
		AdditiveOsc::new(BendedSawGen::default(), SINES);
	// a table built for another number of partials is ignored, which forces the sine path
	osc.num_sines = SINES / 2;
	osc.set_pitch(3000.0, 48000);

	assert_eq!(osc.max_harmonic, 8.0);
	assert!(max_error(&osc, [0.0, 0.25]) < 1e-5);
}

#[test]
fn integer_partials_are_played_from_the_wavetable() {
	let osc: AdditiveOsc<BendedSawGen, SINES, 2> = AdditiveOsc::new(BendedSawGen::default(), SINES);

	assert!(
		osc.table.is_some(),
		"integer ratios should build a wavetable"
	);
	assert!(!osc.has_damping);
}

#[test]
fn wavetable_matches_the_exact_sum() {
	let osc: AdditiveOsc<BendedSawGen, SINES, 2> = AdditiveOsc::new(BendedSawGen::default(), SINES);

	assert!(max_error(&osc, [0.0, 0.25]) < 5e-5);
}

#[test]
fn phased_partials_are_played_from_the_wavetable() {
	let osc: AdditiveOsc<TestGen, SINES, 2> = AdditiveOsc::new(
		TestGen {
			phase_scale: 0.13,
			..TestGen::clean()
		},
		SINES,
	);

	assert!(osc.table.is_some());
	assert!(max_error(&osc, [0.0, 0.25]) < 5e-5);
}

#[test]
fn non_integer_partials_fall_back_to_sines() {
	let osc: AdditiveOsc<TestGen, SINES, 2> = AdditiveOsc::new(
		TestGen {
			ratio_scale: 1.37,
			..TestGen::clean()
		},
		SINES,
	);

	assert!(
		osc.table.is_none(),
		"non integer ratios cannot use a wavetable"
	);
	assert!(max_error(&osc, [0.0, 0.25]) < 5e-5);
}

#[test]
fn damped_partials_fall_back_to_sines() {
	let osc: AdditiveOsc<TestGen, SINES, 2> = AdditiveOsc::new(
		TestGen {
			damping: 0.5,
			..TestGen::clean()
		},
		SINES,
	);

	assert!(
		osc.table.is_none(),
		"damping is not periodic, so it cannot use a wavetable"
	);
	assert!(osc.has_damping);
	assert!(max_error(&osc, [0.0, 0.0]) < 5e-5);
}

#[test]
fn a_changed_num_sines_still_plays_every_partial() {
	let mut osc: AdditiveOsc<BendedSawGen, SINES, 2> =
		AdditiveOsc::new(BendedSawGen::default(), SINES);
	osc.num_sines = SINES / 2;

	assert!(max_error(&osc, [0.0, 0.5]) < 5e-5);

	osc.rebuild_table();
	assert_eq!(
		osc.table.as_ref().map(|table| table.num_sines),
		Some(SINES / 2)
	);
	assert!(max_error(&osc, [0.0, 0.5]) < 5e-5);
}

#[test]
fn rebuilding_the_table_after_the_ratios_change_keeps_it_usable() {
	let mut osc: AdditiveOsc<BendedSawGen, SINES, 2> =
		AdditiveOsc::new(BendedSawGen::default(), SINES);
	osc.freq_gen.scale_amount = 2.0;
	osc.recaculate_ratio();

	assert!(osc.table.is_some());
	assert!((osc.max_ratio - (SINES * 2) as f32).abs() < 1e-3);
	assert!(max_error(&osc, [0.0, 0.25]) < 5e-5);
}

/// A host edit to the bend moves the partials through the sync hook: the
/// oscillator is otherwise free to keep the table it built.
#[test]
fn a_bend_edit_rebuilds_the_partials_through_sync() {
	let mut osc: AdditiveOsc<BendedSawGen, SINES, 2> =
		AdditiveOsc::new(BendedSawGen::default(), SINES);
	let before = osc.play_at(1.0, 0.37, [0.0; 2]);

	assert!(osc.set_parameter("freq_gen.bend_amount", SetValue::Float(3.0)));
	osc.sync();

	let after = osc.play_at(1.0, 0.37, [0.0; 2]);
	assert!(
		(before[0] - after[0]).abs() > 1e-4,
		"the bend did not reach the partials: {before:?} vs {after:?}"
	);
}

/// The spectrum generator's ratios are all integers, which is what lets the
/// oscillator put the whole sum in an inverse-FFT wavetable.
#[test]
fn the_spectral_generator_plays_from_an_fft_table() {
	let osc: AdditiveOsc<SpectralGen, SINES, 2> = AdditiveOsc::new(SpectralGen::default(), SINES);

	assert!(
		osc.table.is_some(),
		"integer ratios should use the inverse-FFT table"
	);
	assert!(max_error(&osc, [0.0, 0.5]) < 5e-5);
}
