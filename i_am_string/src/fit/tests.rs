use super::*;

/// A note made of decaying partials, which is what the fitter is meant to
/// recover.
fn synthetic(partials: &[(f64, f64, f64)], rate: f64, seconds: f64) -> Vec<f32> {
	let count = (seconds * rate) as usize;
	let mut out = vec![0.0f32; count];
	for (frequency, amplitude, t60) in partials {
		let sigma = LN_1000 / t60;
		for (index, sample) in out.iter_mut().enumerate() {
			let time = index as f64 / rate;
			*sample += (amplitude
				* (-sigma * time).exp()
				* (std::f64::consts::TAU * frequency * time).sin()) as f32;
		}
	}
	out
}

#[test]
fn a_pure_tone_lands_in_one_bin() {
	let rate = 48_000.0;
	let length = 4_096;
	let signal: Vec<f64> = (0..length)
		.map(|index| (std::f64::consts::TAU * 1_000.0 * index as f64 / rate).sin())
		.collect();
	let size = next_power_of_two(8 * length);
	let spectrum = spectrum(&signal, size);
	let bin = (1_000.0 * size as f64 / rate).round() as usize;
	let loudest = (0..spectrum.len())
		.max_by(|a, b| {
			spectrum[*a]
				.partial_cmp(&spectrum[*b])
				.unwrap_or(std::cmp::Ordering::Equal)
		})
		.unwrap_or(0);
	assert!(
		loudest.abs_diff(bin) <= 2,
		"the tone landed at {loudest}, expected {bin}"
	);
}

#[test]
fn a_geometric_sum_matches_the_loop() {
	let ratio = (-3.0f64 / 48_000.0).exp();
	let angle = std::f64::consts::TAU * 440.0 / 48_000.0;
	let count = 12_000;
	let (real, imaginary) = geometric(ratio, angle, count);
	let mut expected = (0.0, 0.0);
	let mut power = (1.0f64, 0.0f64);
	let (sine, cosine) = angle.sin_cos();
	for _ in 0..count {
		expected.0 += power.0;
		expected.1 += power.1;
		power = (
			power.0 * cosine - power.1 * sine,
			power.0 * sine + power.1 * cosine,
		);
		power = (power.0 * ratio, power.1 * ratio);
	}
	assert!((real - expected.0).abs() < 1e-6, "{real} vs {}", expected.0);
	assert!(
		(imaginary - expected.1).abs() < 1e-6,
		"{imaginary} vs {}",
		expected.1
	);
}

#[test]
fn the_fitter_recovers_the_partials_it_was_given() {
	let rate = 44_100.0;
	// Far apart, so one partial's demodulated sidebands do not land on the
	// other's envelope: the fitter measures each partial on its own.
	let partials = [(130.81, 0.5, 2.0), (700.0, 0.25, 1.0), (1_500.0, 0.15, 0.6)];
	let samples = synthetic(&partials, rate, 1.5);
	let fitted = fit(&samples, rate as f32, Some(48)).expect("a fitted note");
	assert_eq!(fitted.note, 48);
	assert!(
		(fitted.frequency - 130.81).abs() < 0.5,
		"fundamental {}",
		fitted.frequency
	);
	assert!(
		fitted.modes.len() >= 3,
		"only {} partials",
		fitted.modes.len()
	);
	for (frequency, _, _) in &partials {
		let found = fitted
			.modes
			.iter()
			.any(|mode| (mode.frequency as f64 - frequency).abs() < 1.0);
		assert!(
			found,
			"no partial at {frequency}: {:?}",
			fitted
				.modes
				.iter()
				.map(|mode| mode.frequency)
				.collect::<Vec<_>>()
		);
	}
	let first = &fitted.modes[0];
	assert!(
		(first.tail_t60 as f64 - 2.0).abs() < 0.4,
		"tail {}",
		first.tail_t60
	);
	assert!(
		(0.04..=0.6).contains(&first.prompt_t60),
		"prompt {}",
		first.prompt_t60
	);
}

#[test]
fn the_note_is_found_without_being_told() {
	let rate = 44_100.0;
	let samples = synthetic(&[(130.81, 0.6, 1.5), (261.63, 0.3, 1.0)], rate, 1.5);
	let fitted = fit(&samples, rate as f32, None).expect("a fitted note");
	assert_eq!(
		fitted.note, 48,
		"the fundamental was read as {}",
		fitted.frequency
	);
}

#[test]
fn a_silent_recording_has_no_table() {
	let rate = 44_100.0;
	assert!(fit(&vec![0.0; rate as usize], rate as f32, None).is_none());
	assert!(fit(&[], rate as f32, None).is_none());
}

#[test]
fn a_note_the_fitter_cannot_use_is_refused() {
	let rate = 44_100.0;
	// A tenth of a second is shorter than any window the fitter fits on.
	let samples = synthetic(&[(130.81, 0.6, 1.5)], rate, 0.1);
	assert!(fit(&samples, rate as f32, Some(48)).is_none());
}
