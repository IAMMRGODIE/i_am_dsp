use super::*;
use crate::ProcessInfos;

/// Renders the impulse response of the delay.
fn impulse_response(delay: &mut PureDelay<1>, samples: usize) -> Vec<f32> {
	let mut ctx: Box<dyn ProcessContext> = Box::new(());
	let mut output = Vec::with_capacity(samples);
	for i in 0..samples {
		let mut frame = [if i == 0 { 1.0 } else { 0.0 }];
		delay.process(&mut frame, &[], &mut ctx);
		output.push(frame[0]);
	}
	output
}

/// The energy of the impulse has to end up where the delay time asks for,
/// including between two samples.
#[test]
fn impulse_lands_at_the_delay_time() {
	for delay_time in [0.0f32, 5.0, 10.0, 10.1, 100.0, 380.0] {
		let mut delay = PureDelay::<1>::new(48_000, delay_time, 48_000);
		let response = impulse_response(&mut delay, 24_000);

		let total: f32 = response.iter().map(|x| x.abs()).sum();
		let centroid: f32 = response
			.iter()
			.enumerate()
			.map(|(i, x)| i as f32 * x.abs())
			.sum::<f32>()
			/ total;
		let expected = delay_time / 1000.0 * 48_000.0;

		assert!(
			(total - 1.0).abs() < 0.2,
			"delay {delay_time} ms lost energy: {total}"
		);
		assert!(
			(centroid - expected).abs() < 1.5,
			"delay {delay_time} ms landed at {centroid}, expected {expected}"
		);
	}
}

/// The reported latency has to be the delay the effect actually applies.
#[test]
fn latency_matches_the_delay() {
	let mut delay = PureDelay::<1>::new(48_000, 10.0, 48_000);
	assert_eq!(delay.delay(), 480);
	delay.delay_time = 10.5;
	assert_eq!(delay.delay(), 504);
}

/// The clear-tail action drops a feedback tail immediately.
#[test]
fn clear_tail_silences_the_delay() {
	let mut delay = Delay::<(), 2>::new((), 48_000, 10.0, 48_000);
	delay.dry_gain = 0.0;
	delay.wet_gain = 1.0;
	let mut context: Box<dyn ProcessContext> = Box::new(());

	for index in 0..4_000 {
		let input = if index == 0 { 1.0 } else { 0.0 };
		let mut frame = [input, input];
		delay.process(&mut frame, &[], &mut context);
	}

	delay.clear_tail();

	for _ in 0..1_000 {
		let mut frame = [0.0; 2];
		delay.process(&mut frame, &[], &mut context);
		assert!(frame.iter().all(|sample| sample.abs() < 1e-6), "{frame:?}");
	}
}

/// Resizing and clearing must not panic, and a cleared line is silent.
#[test]
fn clear_and_resize() {
	let mut delay = PureDelay::<1>::new(48_000, 10.0, 48_000);
	let _ = impulse_response(&mut delay, 1_000);
	delay.clear_history();
	let response = impulse_response(&mut delay, 10);
	assert!(response[1..].iter().all(|x| *x == 0.0));
	delay.resize_history(128);
	assert_eq!(delay.history_len(), 128);
	delay.resize_history(0);
	let response = impulse_response(&mut delay, 10);
	assert!(response.iter().all(|x| x.is_finite()));
}
/// A process context that reports fixed infos, for the synced cases.
struct FixedInfos(ProcessInfos);

impl ProcessContext for FixedInfos {
	fn infos(&self) -> &ProcessInfos {
		&self.0
	}

	fn next_event(&mut self) -> Option<crate::NoteEvent> {
		None
	}

	fn send_event(&mut self, _: crate::NoteEvent) {}

	fn clear_events(&mut self) {}

	fn events(&self) -> &[crate::NoteEvent] {
		&[]
	}
}

fn infos_with_tempo(tempo: f32) -> ProcessInfos {
	ProcessInfos {
		sample_rate: 48_000,
		trustable: true,
		playing: true,
		tempo: Some(tempo),
		..ProcessInfos::default()
	}
}

/// A process context with a position as well as a tempo.
fn infos_with_position(tempo: f32, beats: f64, playing: bool) -> ProcessInfos {
	let mut infos = infos_with_tempo(tempo);
	infos.position_beats = Some(beats);
	infos.playing = playing;
	infos
}

/// The phase a flanger's LFO has reached after a while, which is what its rate decides.
fn flanger_phase(rate: (bool, f32), infos: ProcessInfos, samples: usize) -> f64 {
	use crate::generators::wavetable::SineWave;

	let (sync, beats) = rate;
	let mut flanger = Flanger::new(1 << 12, 48_000, SineWave);
	flanger.lfo_frequency = beats;
	flanger.tempo = SyncedTime { sync, beats };

	let mut ctx: Box<dyn ProcessContext> = Box::new(FixedInfos(infos));
	let mut frame = [0.0; 2];
	for _ in 0..samples {
		flanger.process(&mut frame, &[], &mut ctx);
	}

	flanger.phase.phase()
}

/// A synced LFO takes its rate from the tempo: a cycle per beat at 120 bpm is two a
/// second, which is twice what the free value asks for.
#[test]
fn a_synced_lfo_follows_the_tempo() {
	// A quarter of a second is 12_000 samples, and a 1 Hz LFO covers a quarter of a
	// cycle in it.
	let free = flanger_phase((false, 1.0), infos_with_tempo(120.0), 12_000);
	assert!((free - 0.25).abs() < 1e-6, "free phase {free}");

	// Synced to a beat, the same LFO runs at 2 Hz and covers half a cycle.
	let synced = flanger_phase((true, 1.0), infos_with_tempo(120.0), 12_000);
	assert!((synced - 0.5).abs() < 1e-6, "synced phase {synced}");

	// Half the tempo, half the rate.
	let slow = flanger_phase((true, 1.0), infos_with_tempo(60.0), 12_000);
	assert!((slow - 0.25).abs() < 1e-6, "slow phase {slow}");
}

/// While the transport runs, a synced LFO takes its phase from the song position, so a
/// seek moves the sweep with the music instead of leaving it wherever it happened to be.
#[test]
fn a_synced_lfo_locks_to_the_grid() {
	// Beat 4.5 with a cycle per beat is half way through a cycle, whatever came before.
	let phase = flanger_phase((true, 1.0), infos_with_position(120.0, 4.5, true), 1);
	assert!((phase - 0.5).abs() < 1e-6, "phase {phase}");

	let phase = flanger_phase((true, 1.0), infos_with_position(120.0, 2.75, true), 1);
	assert!((phase - 0.75).abs() < 1e-6, "phase {phase}");
}

/// A stopped transport leaves the position where it was, and a synced LFO that followed it
/// would freeze. It keeps sweeping at the synced rate instead, which is what makes it
/// usable while the DAW is stopped.
#[test]
fn a_synced_lfo_keeps_sweeping_while_the_transport_is_stopped() {
	let stopped = infos_with_position(120.0, 4.5, false);

	// One beat per cycle at 120 bpm is 24_000 samples, so half of it is 12_000.
	let phase = flanger_phase((true, 1.0), stopped, 12_000);
	assert!((phase - 0.5).abs() < 1e-6, "phase {phase}");

	// ...and it is not the frozen grid phase: beat 4.5 would be half a cycle, and the
	// sweep is still at the start of its own.
	let first = flanger_phase((true, 1.0), infos_with_position(120.0, 4.5, false), 1);
	assert!(first < 0.01, "first phase {first}");
}

/// The samples the echoes of an impulse land on, per channel.
///
/// `input` is what the impulse is put on: `[1.0, 0.0]` for a signal on one side only,
/// `[1.0, 1.0]` for the same signal on both.
fn echo_samples_with(delay: &mut Delay<(), 2>, samples: usize, input: [f32; 2]) -> [Vec<usize>; 2] {
	let mut ctx: Box<dyn ProcessContext> = Box::new(());
	let mut echoes = [Vec::new(), Vec::new()];

	for i in 0..samples {
		let mut frame = if i == 0 { input } else { [0.0, 0.0] };
		delay.process(&mut frame, &[], &mut ctx);

		// The dry input passes through the effect, so the first samples carry the
		// impulse itself; only what comes later is an echo.
		if i < 100 {
			continue;
		}
		for (channel, echo) in echoes.iter_mut().enumerate() {
			if frame[channel].abs() > 1e-4 {
				echo.push(i);
			}
		}
	}

	echoes
}

/// 100 ms at 48 kHz is 4_800 samples, and that is where the first repeat belongs — the
/// ping-pong read used to stretch it across the channels and land at twice the time.
#[test]
fn ping_pong_alternates_at_the_delay_time() {
	let mut delay = Delay::<(), 2>::new((), 1 << 16, 100.0, 48_000);
	delay.decay_factor = 0.5;
	delay.pingpong = true;

	let echoes = echo_samples_with(&mut delay, 48_000 / 2, [1.0, 0.0]);

	assert_eq!(
		echoes[0].first(),
		Some(&4_800),
		"left repeats: {:?}",
		echoes[0]
	);
	assert_eq!(
		echoes[1].first(),
		Some(&9_600),
		"right repeats: {:?}",
		echoes[1]
	);
	assert_eq!(
		echoes[0].get(1),
		Some(&14_400),
		"left repeats: {:?}",
		echoes[0]
	);
}

/// The alternation has to survive a source that is on *both* sides. Feeding each line its
/// own input would make the two sides repeat together, and the effect would be a stereo
/// offset rather than a ping-pong.
#[test]
fn ping_pong_alternates_with_a_stereo_input() {
	let mut delay = Delay::<(), 2>::new((), 1 << 16, 100.0, 48_000);
	delay.decay_factor = 0.5;
	delay.pingpong = true;

	let echoes = echo_samples_with(&mut delay, 48_000 / 2, [1.0, 1.0]);

	// One side at a time, in turn.
	assert_eq!(
		echoes[0].first(),
		Some(&4_800),
		"left repeats: {:?}",
		echoes[0]
	);
	assert_eq!(
		echoes[1].first(),
		Some(&9_600),
		"right repeats: {:?}",
		echoes[1]
	);
	assert_eq!(
		echoes[0].get(1),
		Some(&14_400),
		"left repeats: {:?}",
		echoes[0]
	);
	assert_eq!(
		echoes[1].get(1),
		Some(&19_200),
		"right repeats: {:?}",
		echoes[1]
	);
}

/// Without ping-pong both sides repeat into themselves, at the same distance.
#[test]
fn a_plain_delay_repeats_on_the_same_side() {
	let mut delay = Delay::<(), 2>::new((), 1 << 16, 100.0, 48_000);
	delay.decay_factor = 0.5;

	let echoes = echo_samples_with(&mut delay, 48_000 / 2, [1.0, 0.0]);

	assert_eq!(echoes[0].first(), Some(&4_800));
	assert_eq!(echoes[0].get(1), Some(&9_600));
	assert!(
		echoes[1].is_empty(),
		"the other side is silent: {:?}",
		echoes[1]
	);
}

/// Moving a delay time has to move the read head, and a read head that teleports is a
/// click; the glide is what turns it into a slide.
#[test]
fn a_moved_delay_time_glides_instead_of_jumping() {
	let mut delay = PureDelay::<1>::new(1 << 16, 10.0, 48_000);
	let mut ctx: Box<dyn ProcessContext> = Box::new(());
	let mut frame = [0.0];

	// 10 ms is 480 samples, and a fresh delay starts there rather than gliding to it.
	delay.process(&mut frame, &[], &mut ctx);
	assert!((delay.smoothed_delay - 480.0).abs() < 1e-3);

	// Asked for 100 ms, the read head has to travel: one sample of a 20 ms glide moves it
	// by about 1/960 of the way, not 4_320 samples.
	delay.delay_time = 100.0;
	delay.process(&mut frame, &[], &mut ctx);
	assert!(
		delay.smoothed_delay > 480.0 && delay.smoothed_delay < 490.0,
		"the read head jumped to {}",
		delay.smoothed_delay
	);

	// It does get there.
	for _ in 0..48_000 / 5 {
		delay.process(&mut frame, &[], &mut ctx);
	}
	assert!(
		(delay.smoothed_delay - 4_800.0).abs() < 1.0,
		"settled at {}",
		delay.smoothed_delay
	);
}

/// A synced delay takes its time from the tempo, and reports it to whoever asks for the
/// latency even though only the process context knows the tempo.
#[test]
fn a_synced_delay_follows_the_tempo() {
	let mut delay = PureDelay::<1>::new(1 << 16, 10.0, 48_000);
	delay.tempo = SyncedTime::synced(0.5);

	let mut ctx: Box<dyn ProcessContext> = Box::new(FixedInfos(infos_with_tempo(120.0)));
	let mut frame = [0.0];
	for _ in 0..48_000 / 5 {
		delay.process(&mut frame, &[], &mut ctx);
	}

	// An eighth note at 120 bpm is 250 ms, which is 12_000 samples — not the 480 the free
	// value asks for.
	assert!(
		(delay.smoothed_delay - 12_000.0).abs() < 1.0,
		"read {}",
		delay.smoothed_delay
	);
	assert_eq!(delay.delay(), 12_000);
}

/// The pan decides where the first ping-pong repeat lands: -1 left, +1 right,
/// 0 centred at equal power.
#[test]
fn ping_pong_pan_moves_the_first_repeat() {
	fn first_repeat(pan: f32) -> [f32; 2] {
		let mut delay = Delay::<(), 2>::new((), 2048, 10.0, 48_000);
		delay.pingpong = true;
		delay.pingpong_pan = pan;
		delay.decay_factor = 0.5;
		delay.wet_gain = 1.0;

		let mut ctx: Box<dyn ProcessContext> = Box::new(());
		let mut at = [0.0; 2];

		for i in 0..600 {
			let mut frame = if i == 0 { [1.0, 1.0] } else { [0.0, 0.0] };
			delay.process(&mut frame, &[], &mut ctx);
			if i == 480 {
				at = frame;
			}
		}

		at
	}

	let left = first_repeat(-1.0);
	assert!(left[0] > 0.4 && left[1] < 0.01, "left repeat: {left:?}");

	let right = first_repeat(1.0);
	assert!(right[0] < 0.01 && right[1] > 0.4, "right repeat: {right:?}");

	let centre = first_repeat(0.0);
	assert!(
		(centre[0] - centre[1]).abs() < 1e-4,
		"centre repeat: {centre:?}"
	);
	assert!(centre[0] > 0.3, "centre repeat: {centre:?}");
}

/// A fully dry chorus has to be exactly the input; the old code divided the dry
/// path by the number of chorus lines as well.
#[test]
fn a_fully_dry_chorus_is_the_input() {
	use crate::generators::wavetable::SineWave;

	let mut chorus = Chorus::<SineWave, 2>::new(1 << 12, 4, 48_000, SineWave);
	chorus.dry_gain = 1.0;
	chorus.wet_gain = 0.0;

	let mut ctx: Box<dyn ProcessContext> = Box::new(());

	for i in 0..512 {
		let input = (core::f32::consts::TAU * 440.0 * i as f32 / 48_000.0).sin() * 0.5;
		let mut frame = [input, input];
		chorus.process(&mut frame, &[], &mut ctx);
		assert!(
			(frame[0] - input).abs() < 1e-6,
			"dry sample {i}: {}",
			frame[0]
		);
	}
}

/// The flanger's feedback line has to be filled, or the feedback knob does nothing.
#[test]
fn the_flanger_feedback_repeats() {
	use crate::generators::wavetable::SineWave;

	let mut flanger = Flanger::<SineWave, 2>::new(1 << 12, 48_000, SineWave);
	flanger.center_delay_time = 2.0;
	flanger.lfo_amplitude = 0.0;
	flanger.feed_back_gain = 0.8;
	flanger.dry_gain = 0.0;
	flanger.wet_gain = 1.0;

	let mut ctx: Box<dyn ProcessContext> = Box::new(());
	let mut at = [0.0; 3];

	for i in 0..400 {
		let mut frame = if i == 0 { [1.0, 1.0] } else { [0.0, 0.0] };
		flanger.process(&mut frame, &[], &mut ctx);
		if i == 96 {
			at[0] = frame[0];
		}
		if i == 192 {
			at[1] = frame[0];
		}
		if i == 288 {
			at[2] = frame[0];
		}
	}

	assert!(at[0] > 0.5, "first repeat: {at:?}");
	assert!((at[1] - 0.8).abs() < 0.1, "second repeat: {at:?}");
	assert!((at[2] - 0.64).abs() < 0.1, "third repeat: {at:?}");
}
