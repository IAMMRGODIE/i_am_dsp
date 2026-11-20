//! A slide between two notes on one string.
//!
//! A portamento is not two notes overlapping: it is one string whose speaking
//! length changes under the hand. These tests ask the rendered sound where it
//! is ringing, rather than asking the model what it thinks it is doing.

use i_am_string::prelude::*;

const FS: f32 = 48_000.0;

/// The frequency a MIDI note number names, on this crate's numbering.
fn note_frequency(note: f32) -> f32 {
	440.0 * 2f32.powf((note - 69.0) / 12.0)
}

/// The magnitude of one frequency, by the Goertzel algorithm.
fn magnitude(samples: &[f32], frequency: f32) -> f32 {
	let n = samples.len() as f64;
	let omega = std::f64::consts::TAU * frequency as f64 / FS as f64;
	let coeff = 2.0 * omega.cos();
	let (mut s1, mut s2) = (0.0f64, 0.0f64);
	for &x in samples {
		let s0 = coeff * s1 - s2 + x as f64;
		s2 = s1;
		s1 = s0;
	}
	let real = s1 - s2 * omega.cos();
	let imaginary = s2 * omega.sin();
	((real * real + imaginary * imaginary).sqrt() / (n * 0.5)) as f32
}

/// The strongest frequency in a range, refined by a parabolic fit.
fn peak_in(samples: &[f32], low: f32, high: f32, step: f32) -> f32 {
	let mut best = low;
	let mut best_value = f32::NEG_INFINITY;
	let mut values = Vec::new();
	let mut f = low;
	while f <= high {
		let value = magnitude(samples, f);
		values.push((f, value));
		if value > best_value {
			best_value = value;
			best = f;
		}
		f += step;
	}
	let index = values.iter().position(|v| v.0 == best).unwrap_or(0);
	if index == 0 || index + 1 >= values.len() {
		return best;
	}
	let left = values[index - 1].1;
	let centre = values[index].1;
	let right = values[index + 1].1;
	let denominator = left - 2.0 * centre + right;
	if denominator.abs() < 1e-12 {
		return best;
	}
	best + step * (0.5 * (left - right) / denominator)
}

/// A cello whose vibrato is off, so a window of it rings at one pitch.
fn cello(glide: f32, legato: bool) -> StringEngine {
	let mut engine = StringEngine::new(FS);
	engine.set_preset(Preset::Cello);
	engine.params_mut().strings = 1;
	engine.params_mut().vibrato_depth = 0.0;
	// Measure the string, not the soundboard colouring over it.
	engine.params_mut().body = 0.0;
	engine.params_mut().sympathetic = 0.0;
	engine.params_mut().glide_time = glide;
	engine.params_mut().legato = legato;
	engine
}

fn render(engine: &mut StringEngine, seconds: f32) -> Vec<f32> {
	(0..(FS * seconds) as usize)
		.map(|_| engine.next_sample())
		.collect()
}

#[test]
fn a_plain_note_rings_at_its_pitch() {
	let mut engine = cello(0.3, true);
	engine.note_on(62, 0.8);
	render(&mut engine, 0.3);
	let audio = render(&mut engine, 0.25);
	let peak = peak_in(&audio, note_frequency(60.0), note_frequency(64.0), 0.5);
	println!("a plain 62 rings at {peak:.1} Hz");
	assert!(
		(peak - note_frequency(62.0)).abs() < 3.0,
		"a plain note is off: {peak:.1} Hz"
	);
}

#[test]
fn a_legato_note_takes_the_string_over_and_arrives() {
	let mut engine = cello(0.3, true);
	engine.note_on(60, 0.8);
	render(&mut engine, 0.4);

	engine.note_on(62, 0.8);
	assert_eq!(
		engine.active_voices(),
		1,
		"the legato note struck a string of its own"
	);

	// A third of a second of slide: the first window is still near the note it
	// came from, and a quarter second after that it has arrived.
	let early = render(&mut engine, 0.1);
	render(&mut engine, 0.3);
	let late = render(&mut engine, 0.25);

	let early_peak = peak_in(&early, note_frequency(59.0), note_frequency(63.0), 0.5);
	let late_peak = peak_in(&late, note_frequency(60.5), note_frequency(63.5), 0.5);
	println!("slide 60 -> 62: {early_peak:.1} Hz over the first 100 ms, {late_peak:.1} Hz after");

	// A window over the first hundred milliseconds averages the pitch the slide
	// crossed in it, so it sits between the two notes rather than on either.
	assert!(
		early_peak > note_frequency(60.0) && early_peak < note_frequency(61.5),
		"the slide did not start near the note before it: {early_peak:.1} Hz"
	);
	let wanted = note_frequency(62.0);
	assert!(
		(late_peak - wanted).abs() < 3.0,
		"the slide did not arrive: {late_peak:.1} Hz against {wanted:.1}"
	);
}

#[test]
fn a_slide_too_far_to_stretch_is_a_note_of_its_own() {
	let mut engine = cello(0.3, true);
	engine.note_on(60, 0.8);
	render(&mut engine, 0.4);

	// A fifth is far past what the model can retune a string by.
	engine.note_on(67, 0.8);
	assert_eq!(
		engine.active_voices(),
		2,
		"the far note took the string over"
	);
}

#[test]
fn without_legato_every_note_is_its_own_string() {
	let mut engine = cello(0.0, false);
	engine.note_on(60, 0.8);
	render(&mut engine, 0.1);
	engine.note_on(61, 0.8);
	assert_eq!(engine.active_voices(), 2);
}

#[test]
fn a_slide_after_a_release_is_a_note_of_its_own() {
	let mut engine = cello(0.3, true);
	engine.note_on(60, 0.8);
	render(&mut engine, 0.1);
	engine.note_off(60);

	engine.note_on(61, 0.8);
	assert_eq!(
		engine.active_voices(),
		2,
		"a note took over a string that was already letting go"
	);
}
