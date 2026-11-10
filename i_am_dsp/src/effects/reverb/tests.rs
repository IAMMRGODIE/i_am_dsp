use super::*;
use crate::prelude::Biquad;

/// Renders the impulse response of one channel.
fn impulse_response(reverb: &mut Reverb<Biquad<2>, 8, 2>, samples: usize) -> Vec<f32> {
	let mut ctx: Box<dyn ProcessContext> = Box::new(());
	let mut output = Vec::with_capacity(samples);
	for i in 0..samples {
		let mut frame = [if i == 0 { 1.0 } else { 0.0 }; 2];
		reverb.process(&mut frame, &[], &mut ctx);
		output.push(frame[0]);
	}
	output
}

fn energy(response: &[f32], from: usize, to: usize) -> f64 {
	response[from..to.min(response.len())]
		.iter()
		.map(|x| (*x as f64) * (*x as f64))
		.sum()
}

fn default_reverb() -> Reverb<Biquad<2>, 8, 2> {
	Reverb::new(
		Biquad::new(48_000),
		48_000,
		FeedbackMatrixGen::default(),
		10,
		10.0,
	)
}

/// The impulse response has to be finite, start loud and die away.
#[test]
fn impulse_response_decays() {
	for decay_time in [200.0f32, 1000.0, 5000.0] {
		let mut reverb = default_reverb();
		reverb.decay_time = decay_time;
		let seconds = (decay_time / 1000.0) * 3.0;
		let response = impulse_response(&mut reverb, (48_000.0 * seconds) as usize);

		assert!(
			response.iter().all(|x| x.is_finite()),
			"response is not finite"
		);
		let early = energy(&response, 0, 4_800);
		let late = energy(&response, response.len() - 4_800, response.len());
		assert!(early > 0.0, "the reverb is silent");
		assert!(
			late < early * 1e-6,
			"the tail did not decay: {early:e} -> {late:e}"
		);
	}
}

/// A longer decay time has to leave more energy in the tail.
#[test]
fn longer_decay_keeps_more_energy() {
	let mut short = default_reverb();
	short.decay_time = 500.0;
	let mut long = default_reverb();
	long.decay_time = 5000.0;

	let short_response = impulse_response(&mut short, 48_000);
	let long_response = impulse_response(&mut long, 48_000);

	let short_tail = energy(&short_response, 24_000, 48_000);
	let long_tail = energy(&long_response, 24_000, 48_000);
	assert!(
		long_tail > short_tail * 100.0,
		"decay time has little effect: {short_tail:e} vs {long_tail:e}"
	);
}

/// Every delay line has to be able to reach the output on its own: the
/// output weights are per line, not per channel.
#[test]
fn output_factors_weight_each_line() {
	let mut only_first = default_reverb();
	only_first.feedback_matrix = core::array::from_fn(|_| FeedbackMatrix::identity());
	only_first.output_factors = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

	let mut only_second = default_reverb();
	only_second.feedback_matrix = core::array::from_fn(|_| FeedbackMatrix::identity());
	only_second.output_factors = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

	let first = impulse_response(&mut only_first, 4_800);
	let second = impulse_response(&mut only_second, 4_800);
	assert!(energy(&first, 0, 4_800) > 1e-12, "the first line is silent");
	assert!(
		energy(&second, 0, 4_800) > 1e-12,
		"the second line is silent"
	);
}

/// The diffusion amount has to change what comes out.
#[test]
fn diffusion_changes_the_early_response() {
	let mut dry = default_reverb();
	dry.feedback_matrix = core::array::from_fn(|_| FeedbackMatrix::identity());
	dry.diffusion = 0.0;
	let mut diffused = default_reverb();
	diffused.feedback_matrix = core::array::from_fn(|_| FeedbackMatrix::identity());
	diffused.diffusion = 1.0;

	let dry_response = impulse_response(&mut dry, 2_400);
	let diffused_response = impulse_response(&mut diffused, 2_400);
	assert!(
		energy(&dry_response, 0, 2_400) != energy(&diffused_response, 0, 2_400),
		"diffusion has no effect"
	);
}

/// Degenerate delay times must not divide by zero or panic.
#[test]
fn empty_delay_lines_do_not_panic() {
	let mut reverb = default_reverb();
	reverb.set_delay_time([0; 8]);
	reverb.set_center_delay(0.0);
	let response = impulse_response(&mut reverb, 480);
	assert!(response.iter().all(|x| x.is_finite()));
}

/// The reported latency is the pre delay in front of the network.
#[test]
fn latency_is_the_pre_delay() {
	let mut reverb = default_reverb();
	assert_eq!(reverb.delay(), reverb.pure_delay.delay());
	assert!(
		reverb.delay() > 0,
		"the default pre delay should be reported"
	);
	reverb.pure_delay.delay_time = 25.0;
	assert_eq!(reverb.delay(), 1_200);
}

/// The diffused output must not start with the dry signal itself.
#[test]
fn the_direct_signal_is_not_passed_through() {
	let mut reverb = default_reverb();
	let response = impulse_response(&mut reverb, 48_000);
	assert!(
		response[0].abs() < 0.5,
		"the direct signal leaks: {}",
		response[0]
	);
}

/// Renders a stereo impulse response with the impulse only on the left.
fn stereo_impulse_response(
	reverb: &mut Reverb<Biquad<2>, 8, 2>,
	samples: usize,
) -> (Vec<f32>, Vec<f32>) {
	let mut ctx: Box<dyn ProcessContext> = Box::new(());
	let mut left = Vec::with_capacity(samples);
	let mut right = Vec::with_capacity(samples);
	for i in 0..samples {
		let mut frame = [if i == 0 { 1.0 } else { 0.0 }, 0.0];
		reverb.process(&mut frame, &[], &mut ctx);
		left.push(frame[0]);
		right.push(frame[1]);
	}
	(left, right)
}

/// Sharing the feedback has to carry the left channel into the right one.
/// With independent channels an impulse that only enters the left one can
/// never reach the right one.
#[test]
fn spread_shares_the_channels() {
	let mut independent = default_reverb();
	independent.feedback_matrix = core::array::from_fn(|_| FeedbackMatrix::identity());
	independent.spread = 0.0;

	let mut shared = default_reverb();
	shared.feedback_matrix = core::array::from_fn(|_| FeedbackMatrix::identity());
	shared.spread = 1.0;

	let (_, right) = stereo_impulse_response(&mut independent, 9_600);
	assert!(
		right.iter().all(|x| *x == 0.0),
		"an independent channel leaked"
	);

	let (_, right) = stereo_impulse_response(&mut shared, 9_600);
	assert!(
		energy(&right, 0, 9_600) > 1e-12,
		"the shared channel stayed silent"
	);
}

/// The rotation that shares the channels is orthogonal, so it must not change
/// how the network decays.
#[test]
fn spread_keeps_the_network_stable() {
	for spread in [0.0f32, 0.5, 1.0] {
		let mut reverb = default_reverb();
		reverb.spread = spread;
		reverb.decay_time = 5000.0;
		let response = impulse_response(&mut reverb, 48_000 * 5);
		assert!(
			response.iter().all(|x| x.is_finite()),
			"spread {spread} blew up"
		);
		let early = energy(&response, 0, 4_800);
		let late = energy(&response, response.len() - 4_800, response.len());
		assert!(late < early, "spread {spread} did not decay");
	}
}
