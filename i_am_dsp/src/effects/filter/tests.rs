use super::*;

fn impulse_response<E: Effect<2>>(effect: &mut E, length: usize) -> Vec<f32> {
	let mut context: Box<dyn ProcessContext> = Box::new(());
	let mut output = Vec::with_capacity(length);

	for index in 0..length {
		let mut frame = [if index == 0 { 1.0 } else { 0.0 }; 2];
		effect.process(&mut frame, &[], &mut context);
		output.push(frame[0]);
	}

	output
}

/// A feedback comb repeats the impulse every D samples with gain g, g^2, ...
#[test]
fn the_comb_is_a_feedback_delay() {
	let mut comb = CombsFilter::<2>::new(0.5, 4);
	let response = impulse_response(&mut comb, 13);

	assert!((response[0] - 1.0).abs() < 1e-6, "{response:?}");
	assert!((response[4] - 0.5).abs() < 1e-6, "{response:?}");
	assert!((response[8] - 0.25).abs() < 1e-6, "{response:?}");
	assert!(response[1].abs() < 1e-6, "{response:?}");
	assert!(response[5].abs() < 1e-6, "{response:?}");
}

/// The standard Schroeder allpass: y[n] = -g x[n] + x[n-D] + g y[n-D].
/// Its impulse response has unit energy, unlike a plain feedback delay.
#[test]
fn the_allpass_has_unit_energy() {
	let mut allpass = AllpassFilter::<2>::new(0.5, 4);
	let response = impulse_response(&mut allpass, 256);
	let energy: f32 = response.iter().map(|sample| sample * sample).sum();

	assert!((energy - 1.0).abs() < 1e-3, "energy {energy}");
	assert!((response[0] + 0.5).abs() < 1e-6, "{response:?}");
	assert!((response[4] - 0.75).abs() < 1e-6, "{response:?}");
}
