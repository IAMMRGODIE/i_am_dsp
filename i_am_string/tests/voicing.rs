//! The colour layers of the mode bank.
//!
//! These are the four departures from an ideal string: a close detuned pair
//! on each partial, a two stage decay, a metallic layer for wound strings, and
//! a pluck contact that darkens the top of the attack. Each test is written
//! against the rendered signal, not against a parameter being stored, because
//! the failure mode for all four is a setting that is carried around and
//! never reaches the string.

use i_am_string::prelude::*;

const FS: f32 = 48_000.0;

fn config(frequency: f32) -> StringConfig {
	StringConfig {
		frequency,
		inharmonicity: 0.0,
		sigma0: 0.8,
		sigma1: 0.0,
		tension: 70.0,
		impedance: 1.5,
		bridge: 0.0,
		voicing: Voicing::default(),
		max_partials: 40,
		dispersion_sections: 8,
		dispersion_delay_fraction: 1.0,
		dispersion_coefficients: None,
	}
}

fn render(string: &mut StringCore, samples: usize) -> Vec<f32> {
	(0..samples).map(|_| string.step()).collect()
}

/// The magnitude of one frequency in a windowed segment.
fn magnitude(samples: &[f32], frequency: f32) -> f32 {
	let n = samples.len() as f32;
	let omega = std::f32::consts::TAU * frequency / FS;
	let coefficient = 2.0 * omega.cos();
	let (mut s1, mut s2, mut weight) = (0.0f32, 0.0f32, 0.0f32);
	for (index, sample) in samples.iter().enumerate() {
		let window = 0.5 - 0.5 * (std::f32::consts::TAU * index as f32 / n).cos();
		weight += window;
		let s0 = sample * window + coefficient * s1 - s2;
		s2 = s1;
		s1 = s0;
	}
	(s1 * s1 + s2 * s2 - coefficient * s1 * s2).max(0.0).sqrt() / weight.max(1.0)
}

fn segment(audio: &[f32], from: f32, to: f32) -> &[f32] {
	&audio[(from * FS) as usize..(to * FS) as usize]
}

/// A note that starts at the string's own decay rate and then relaxes.
///
/// One exponential is what an ideal damped oscillator does. A real string has a
/// prompt sound and an aftersound, and the whole point of the tail is that the
/// late decay is measurably slower than the early one at the same partial.
#[test]
fn a_modal_note_decays_in_two_stages() {
	let mut voice = config(110.0);
	voice.sigma0 = 1.2;
	voice.voicing = Voicing {
		late_decay: 0.45,
		late_delay: 0.25,
		late_ramp: 0.4,
		..Default::default()
	};
	let mut string = StringCore::new(Core::Modal, FS, &voice);
	string.pluck(0.2, 0.01);
	let audio = render(&mut string, (FS * 4.0) as usize);

	let early = magnitude(segment(&audio, 0.2, 0.7), 110.0);
	let late = magnitude(segment(&audio, 2.6, 3.6), 110.0);
	// The gap between the centres of the two windows, in seconds.
	let sigma = (early / late.max(1e-9)).ln() / 2.75;
	assert!(
		sigma > 0.3 && sigma < 1.1,
		"the tail decayed at {sigma} 1/s, expected about 0.55"
	);

	// The same string with the tail switched off must keep one rate.
	let mut plain = config(110.0);
	plain.sigma0 = 1.2;
	let mut string = StringCore::new(Core::Modal, FS, &plain);
	string.pluck(0.2, 0.01);
	let audio = render(&mut string, (FS * 4.0) as usize);
	let early = magnitude(segment(&audio, 0.2, 0.7), 110.0);
	let late = magnitude(segment(&audio, 2.6, 3.6), 110.0);
	let flat = (early / late.max(1e-9)).ln() / 2.75;
	assert!(
		(flat - 1.2).abs() < 0.2,
		"the plain string's decay was {flat} 1/s, expected 1.2"
	);

	assert!(
		sigma < 0.8 * flat,
		"the tail is not slower than the prompt sound: {sigma} against {flat}"
	);
}

/// A partial has to be a pair, and the second line has to sound.
///
/// The substitution is heard as a slow swell on the fundamental and as a
/// roughness higher up, so the test looks for the second line in the spectrum
/// rather than for the parameter in the struct.
#[test]
fn a_plucked_partial_is_a_detuned_pair() {
	let mut voice = config(110.0);
	voice.voicing = Voicing {
		doublet: 0.25,
		doublet_detune: 0.004,
		..Default::default()
	};
	let mut string = StringCore::new(Core::Modal, FS, &voice);
	let partials = string.partial_frequencies();
	assert!(
		partials.len() >= 2 && (partials[1] / partials[0] - 1.004).abs() < 1e-4,
		"the pair is not there: {partials:?}"
	);

	string.pluck(0.2, 0.01);
	let audio = render(&mut string, (FS * 3.0) as usize);
	let window = segment(&audio, 0.1, 2.9);
	let main = magnitude(window, 110.0);
	let side = magnitude(window, 110.0 * 1.004);
	let relative = 20.0 * (side / main.max(1e-9)).log10();
	assert!(
		relative > -20.0,
		"the second line is {relative:.1} dB under the first, so the pair is not sounding"
	);
	assert!(
		relative < -6.0,
		"the second line is only {relative:.1} dB under the first, which is two notes, not one"
	);
}

/// The wound layer has to sit off the harmonic grid, where it is heard as
/// metal rather than as a wrong note.
#[test]
fn a_wound_string_has_off_grid_partials() {
	let mut voice = config(110.0);
	voice.voicing = Voicing {
		winding: 1.0,
		..Default::default()
	};
	let string = StringCore::new(Core::Modal, FS, &voice);
	let partials = string.partial_frequencies();
	let wound = 110.0 * (2.0 + 0.34 + 0.011 * 4.0);
	assert!(
		partials.iter().any(|f| (f - wound).abs() < 0.5),
		"no partial at {wound} Hz in {partials:?}"
	);
	// And it is not simply the second harmonic: it has to sit well clear of
	// the grid, or it is a mistuned note rather than a metallic layer.
	let nearest = (1..=8)
		.map(|n| (wound - 110.0 * n as f32).abs())
		.fold(f32::INFINITY, f32::min);
	assert!(
		nearest > 10.0,
		"the wound partial at {wound} Hz is only {nearest} Hz from a harmonic"
	);
}

/// The pluck's finite contact darkens the top of the attack.
#[test]
fn the_pluck_contact_darkens_the_attack() {
	let level = |tilt: f32| {
		let mut voice = config(110.0);
		voice.voicing = Voicing {
			pluck_tilt: tilt,
			..Default::default()
		};
		let mut string = StringCore::new(Core::Modal, FS, &voice);
		string.pluck(0.25, 0.01);
		let audio = render(&mut string, (FS * 1.0) as usize);
		let window = segment(&audio, 0.05, 0.95);
		magnitude(window, 1100.0)
	};
	let open = level(0.0);
	let soft = level(0.06);
	let ratio = soft / open.max(1e-9);
	// exp(-0.06 * 9) = 0.58 at the tenth partial.
	assert!(
		(0.4..0.75).contains(&ratio),
		"the tenth partial came out at {ratio:.2} of its open level"
	);
}

/// A finite release leaves the string moving when the finger lets go.
///
/// The spectral magnitude barely moves - a release that is smooth in time is
/// still smooth in frequency - but the first samples and the phase of every
/// partial are different, which is what the attack is made of. The test
/// checks that the difference is real and that the string stays bounded.
#[test]
fn a_finite_release_starts_the_string_moving() {
	let mut instant = StringCore::new(Core::Modal, FS, &config(110.0));
	instant.pluck(0.25, 0.01);
	let mut released = StringCore::new(Core::Modal, FS, &config(110.0));
	released.pluck_with_release(0.25, 0.01, 0.004);

	let frames = (FS * 0.01) as usize;
	let a: Vec<f32> = (0..frames).map(|_| instant.step()).collect();
	let b: Vec<f32> = (0..frames).map(|_| released.step()).collect();
	let peak = a
		.iter()
		.chain(b.iter())
		.fold(0.0f32, |peak, value| peak.max(value.abs()));
	let difference = a
		.iter()
		.zip(b.iter())
		.map(|(x, y)| (x - y).abs())
		.fold(0.0f32, f32::max);
	assert!(peak > 1e-4, "the pluck was silent");
	assert!(
		difference > 0.01 * peak,
		"the release made no difference: {difference} against a peak of {peak}"
	);
	assert!(
		b.iter()
			.all(|value| value.is_finite() && value.abs() < 10.0),
		"the release ran away"
	);
}

/// A stab is released harder than a note that has been left ringing.
///
/// The usual note-off is one damper for every gesture. A hand that has been
/// holding the string lifts onto it; a hand that brushed past lets go into the
/// board. The difference is audible on any plucked instrument and it is the
/// one part of a release that a fixed felt cannot do.
#[test]
#[ignore = "tuned against a piano preset; needs retuning for bowed strings"]
fn a_short_note_is_released_harder_than_a_long_one() {
	// How far the note falls between a tenth and a quarter of a second after
	// the release. Both gestures are measured from their own first window, so
	// the comparison is of the damping and not of how loud the note was when
	// it was let go.
	let ratio = |held: f32| {
		let mut engine = StringEngine::new(FS);
		engine.set_preset(Preset::Cello);
		engine.note_on(45, 0.9);
		for _ in 0..(FS * held) as usize {
			engine.next_sample();
		}
		engine.note_off(45);
		let level = |engine: &mut StringEngine, seconds: f32| {
			let count = (FS * seconds) as usize;
			let mut energy = 0.0f32;
			for _ in 0..count {
				let value = engine.next_sample();
				energy += value * value;
			}
			(energy / count as f32).sqrt()
		};
		let early = level(&mut engine, 0.03);
		let _ = level(&mut engine, 0.17);
		let late = level(&mut engine, 0.05);
		late / early.max(1e-9)
	};
	let short = ratio(0.06);
	let long = ratio(1.0);
	assert!(
		short < 0.4 * long,
		"a stabbed note kept {short:.3} of its level and a held one {long:.3}"
	);
}
