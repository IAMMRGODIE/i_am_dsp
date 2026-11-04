use super::*;

fn compressor_with_lookahead(lookahead: usize) -> Compressor<RectifyEnvelope<2>, 2> {
	let mut compressor = Compressor::new(
		RectifyEnvelope::<2>::new(48_000, 48_000.0),
		48_000,
		1.0,
		100.0,
		0.0,
		1.0,
	);
	compressor.lookahead = lookahead;
	compressor
}

fn wrapper_of(effect: impl Effect<2> + 'static) -> EffectWarpper {
	EffectWarpper {
		path: (String::new(), String::new()),
		effect: Box::new(effect),
		mix: 1.0,
		dry_delay: std::array::from_fn(|_| RingBuffer::new(0)),
	}
}

/// The dry signal must be delayed by exactly the effect latency so that it
/// lines up with the (latency-delayed) wet signal when mixed.
#[test]
fn align_dry_delays_by_effect_latency() {
	let mut wrapper = wrapper_of(compressor_with_lookahead(16));
	assert_eq!(wrapper.effect.delay(), 16);

	for i in 0..40usize {
		let delayed = wrapper.align_dry([i as f32, i as f32 * 2.0]);
		let expected = i as isize - 16;
		if expected >= 0 {
			assert!(
				(delayed[0] - expected as f32).abs() < 1e-6,
				"ch0 at {i}: {} vs {expected}",
				delayed[0]
			);
			assert!(
				(delayed[1] - expected as f32 * 2.0).abs() < 1e-6,
				"ch1 at {i}"
			);
		} else {
			assert_eq!(
				delayed[0], 0.0,
				"ch0 should be silence before latency, at {i}"
			);
			assert_eq!(delayed[1], 0.0);
		}
	}
}

/// Changing the effect latency at runtime must resize the dry delay line.
/// The retained history may show up briefly, but once the line has been
/// flushed the delay must equal the new latency exactly.
#[test]
fn align_dry_resizes_when_latency_changes() {
	let mut wrapper = wrapper_of(compressor_with_lookahead(4));
	for i in 0..12usize {
		wrapper.align_dry([i as f32, 0.0]);
	}

	// change latency via a new effect with different latency
	wrapper.effect = Box::new(compressor_with_lookahead(8));
	assert_eq!(wrapper.effect.delay(), 8);
	for i in 0..24usize {
		let delayed = wrapper.align_dry([i as f32, 0.0]);
		// once the buffer has been fully overwritten with post-change
		// inputs, the delay must be the new latency
		if i >= 8 {
			let expected = i - 8;
			assert!(
				(delayed[0] - expected as f32).abs() < 1e-6,
				"ch0 at {i}: {} vs {expected}",
				delayed[0]
			);
		}
	}
}

/// Graph latency: parallel output tracks count once at their maximum, and
/// the master bus latency is added on top.
#[test]
fn delay_is_max_track_plus_master() {
	let mut shared = SharedData {
		generators: vec![],
		effects: std::array::from_fn(|_| vec![]),
		track_delay: std::array::from_fn(|_| std::array::from_fn(|_| RingBuffer::new(0))),
		#[cfg(feature = "standalone")]
		ctx: None,
		lfos: vec![],
		sample_rate: 48_000,
		current_phase: 0.0,
	};

	// track 0 has 64 samples of latency, track 1 has 32
	shared.effects[0].push(wrapper_of(compressor_with_lookahead(64)));
	shared.effects[1].push(wrapper_of(compressor_with_lookahead(32)));
	// master bus adds 8 more
	shared.effects[MAX_OUTPUT_TRACKS].push(wrapper_of(compressor_with_lookahead(8)));

	assert_eq!(shared.delay(), 64 + 8);
}
