//! Tests for the tuning / temperament systems in the DSP library.
//!
//! Note: this library uses the convention that A4 = MIDI 57 (see `A4_MIDI`), so
//! middle C (C4) is MIDI 48.

use i_am_dsp::prelude::{
	Tuning, EqualTemperament, NEdoTuning, JustIntonation, PythagoreanTuning, ScaleTuning,
	A4_FREQ, PENTATONIC,
};

const C4_MIDI: f32 = 48.0;

fn approx(a: f32, b: f32) -> bool {
	(a - b).abs() < 0.001
}

#[test]
fn equal_temperament_matches_12_edo() {
	assert_eq!(EqualTemperament.get_frequency(57.0), A4_FREQ); // A4
	// 12-EDO is exactly equal temperament.
	for n in 0..=127 {
		let n = n as f32;
		assert!(approx(
			NEdoTuning::<12>.get_frequency(n),
			EqualTemperament.get_frequency(n)
		), "note {n}");
	}
}

#[test]
fn n_edo_scales_properly() {
	// 12-EDO doubles the frequency every 12 notes (an octave).
	let base = NEdoTuning::<12>.get_frequency(C4_MIDI);
	assert!(approx(NEdoTuning::<12>.get_frequency(C4_MIDI + 12.0), base * 2.0));

	// 24-EDO: an octave is 24 steps.
	let base24 = NEdoTuning::<24>.get_frequency(C4_MIDI);
	assert!(approx(NEdoTuning::<24>.get_frequency(C4_MIDI + 24.0), base24 * 2.0));
	let eleven_tone_up = NEdoTuning::<24>.get_frequency(C4_MIDI + 11.0);
	assert!(approx(eleven_tone_up, base24 * 2.0f32.powf(11.0 / 24.0)));
}

#[test]
fn just_intonation_ratios() {
// In just tuning the fifth (G above C) is a clean 3:2, the major third a clean 5:4.
	let c4 = JustIntonation.get_frequency(C4_MIDI);
	let g4 = JustIntonation.get_frequency(C4_MIDI + 7.0);
	assert!(approx(g4 / c4, 1.5), "fifth ratio = {}", g4 / c4);

	let e4 = JustIntonation.get_frequency(C4_MIDI + 4.0);
	assert!(approx(e4 / c4, 1.25), "major third = {}", e4 / c4);

	// Octave stays pure.
	let c5 = JustIntonation.get_frequency(C4_MIDI + 12.0);
	assert!(approx(c5 / c4, 2.0), "octave = {}", c5 / c4);
}

#[test]
fn pythagorean_fifths_are_pure() {
	// The fifth is still a pure 3:2.
	let c = PythagoreanTuning.get_frequency(C4_MIDI);
	let g = PythagoreanTuning.get_frequency(C4_MIDI + 7.0);
	assert!(approx(g / c, 1.5), "fifth = {}", g / c);

	// The major third (E) is a sharp 81:64, NOT the just 5:4.
	let e = PythagoreanTuning.get_frequency(C4_MIDI + 4.0);
	assert!(approx(e / c, 81.0 / 64.0), "pythagorean third = {}", e / c);
}

#[test]
fn scale_tuning_custom_scale() {
	// A simple 4-note cycle: C, whole-tone, major third, fifth.
	let custom = ScaleTuning::<4>::new([1.0, 9.0 / 8.0, 5.0 / 4.0, 3.0 / 2.0]);
	let c4 = custom.get_frequency(C4_MIDI);
	assert!(approx(custom.get_frequency(C4_MIDI + 1.0) / c4, 9.0 / 8.0));

	// Wraps into the next octave after four steps.
	assert!(approx(custom.get_frequency(C4_MIDI + 4.0) / c4, 2.0));

	// The exported pentatonic preset is octave-consistent.
	let p0 = PENTATONIC.get_frequency(C4_MIDI);
	assert!(approx(PENTATONIC.get_frequency(C4_MIDI + 5.0) / p0, 2.0));
}
