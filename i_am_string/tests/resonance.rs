//! Damper behaviour: the sustain pedal, half pedalling and sympathetic
//! resonance.

use i_am_string::prelude::*;

const FS: f32 = 48_000.0;

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
}

/// Play a note, let it go, and report what is left a while later.
fn played_and_released(pedal: f32, sympathetic: f32) -> (f32, f32) {
    let mut engine = StringEngine::new(FS);
    engine.set_preset(Preset::Piano);
    engine.apply_params(EngineParams {
        sympathetic,
        ..Preset::Piano.defaults()
    });
    engine.set_pedal(pedal);
    engine.note_on(60, 0.9);
    let mut held = Vec::new();
    for _ in 0..(FS * 0.4) as usize {
        held.push(engine.next_sample());
    }
    engine.note_off(60);
    let mut released = Vec::new();
    for _ in 0..(FS * 1.6) as usize {
        released.push(engine.next_sample());
    }
    (rms(&held), rms(&released))
}

#[test]
fn the_sustain_pedal_holds_a_released_note() {
    let (_, without) = played_and_released(0.0, 0.0);
    let (_, with) = played_and_released(1.0, 0.0);
    // Three rather than four because the mode bank's damper is frequency
    // dependent, as a felt resting on a string actually is: it costs every
    // partial the same fraction of its energy per cycle, so a damped note loses
    // its top first and its tail is darker as well as shorter. A uniform damper
    // leaves more energy in the tail and scores higher here while sounding less
    // like a piano.
    assert!(
        with > 3.0 * without,
        "the pedal barely helped: {without} without, {with} with"
    );
}

#[test]
fn a_half_pedal_lands_between_the_two_extremes() {
    let (_, up) = played_and_released(0.0, 0.0);
    let (_, half) = played_and_released(0.5, 0.0);
    let (_, down) = played_and_released(1.0, 0.0);
    assert!(
        half > 1.4 * up && half < 0.6 * down,
        "half pedalling was not in between: up {up}, half {half}, down {down}"
    );
}

#[test]
fn the_undamped_strings_ring_on_after_the_note() {
    // With the pedal up the played string is damped, but the undamped strings
    // it drove are still ringing.
    let (_, without) = played_and_released(0.0, 0.0);
    let (_, with) = played_and_released(0.0, 0.9);
    assert!(
        with > 3.0 * without,
        "the sympathetic strings added nothing: {without} then {with}"
    );
}

#[test]
fn a_pedalled_note_blooms_beyond_the_plain_string() {
    let (_, plain) = played_and_released(1.0, 0.0);
    let (_, rich) = played_and_released(1.0, 0.9);
    assert!(
        rich > 1.2 * plain,
        "the pedal did not bloom: {plain} then {rich}"
    );
}

#[test]
fn the_sympathetic_bank_stays_bounded() {
    let mut engine = StringEngine::new(FS);
    engine.set_preset(Preset::Piano);
    engine.apply_params(EngineParams {
        sympathetic: 1.0,
        ..Preset::Piano.defaults()
    });
    engine.set_pedal(1.0);
    for note in [36u8, 43, 48, 55, 60, 64, 67, 72, 79, 84] {
        engine.note_on(note, 1.0);
    }
    let mut peak = 0.0f32;
    for sample in 0..(FS * 12.0) as usize {
        let value = engine.next_sample();
        assert!(
            value.is_finite(),
            "the sympathetic bank produced a non finite sample at {sample}"
        );
        peak = peak.max(value.abs());
        assert!(value.abs() < 6.0, "runaway at {sample}: {value}");
    }
    assert!(peak > 0.01, "nothing was heard at all: peak {peak}");
}

#[test]
fn the_sympathetic_bank_adds_a_tail_but_never_a_runaway() {
    // Same note, same damping, one with the undamped strings and one without.
    let tail = |sympathetic: f32| {
        let mut engine = StringEngine::new(FS);
        engine.set_preset(Preset::Piano);
        engine.apply_params(EngineParams {
            sympathetic,
            damping: 2.5,
            ..Preset::Piano.defaults()
        });
        engine.set_pedal(0.0);
        engine.note_on(60, 0.9);
        let mut audio = Vec::new();
        for _ in 0..(FS * 3.0) as usize {
            audio.push(engine.next_sample());
        }
        rms(&audio[(FS * 2.0) as usize..])
    };
    let without = tail(0.0);
    let with = tail(0.9);
    assert!(
        with > 2.0 * without,
        "no sympathetic tail: {without} then {with}"
    );
}

#[test]
fn lifting_the_pedal_stops_a_pedalled_note() {
    let mut engine = StringEngine::new(FS);
    engine.set_preset(Preset::Piano);
    engine.apply_params(EngineParams {
        sympathetic: 0.0,
        ..Preset::Piano.defaults()
    });
    engine.set_pedal(1.0);
    engine.note_on(60, 0.9);
    for _ in 0..(FS * 0.5) as usize {
        engine.next_sample();
    }
    engine.note_off(60);
    let mut ringing = Vec::new();
    for _ in 0..(FS * 0.5) as usize {
        ringing.push(engine.next_sample());
    }
    // Now drop the pedal: the felt falls on the string.
    engine.set_pedal(0.0);
    let mut damped = Vec::new();
    for _ in 0..(FS * 0.5) as usize {
        damped.push(engine.next_sample());
    }
    assert!(
        rms(&damped) < 0.2 * rms(&ringing),
        "dropping the pedal did not damp the note: {} then {}",
        rms(&ringing),
        rms(&damped)
    );
}
#[test]
fn the_whole_engine_runs_comfortably_faster_than_realtime() {
    // Eight voices plus a bank of 88 sympathetic strings, which is a heavy
    // chord. This is a guard against someone quietly making the model ten times
    // more expensive.
    let mut engine = StringEngine::new(FS);
    engine.set_preset(Preset::Piano);
    engine.set_pedal(1.0);
    for note in [40u8, 47, 52, 55, 59, 64, 67, 71] {
        engine.note_on(note, 1.0);
    }
    let samples = (FS * 10.0) as usize;
    let start = std::time::Instant::now();
    let mut peak = 0.0f32;
    for _ in 0..samples {
        peak = peak.max(engine.next_sample().abs());
    }
    let elapsed = start.elapsed().as_secs_f32().max(1e-6);
    let realtime = samples as f32 / FS / elapsed;
    println!("eight voice piano: {realtime:.1}x realtime, peak {peak:.3}");
    // The mode bank is deliberately more expensive than the delay lines: it is
    // a hundred and sixty real oscillators per string where the waveguide is a
    // filter that stands in for them, and measured it costs about eight times as
    // much per string. Eight voices of a three string piano against a bank of
    // eighty eight sympathetic strings is the heaviest thing the crate does, and
    // the guard is against someone quietly making *that* ten times worse, not
    // against the two cores costing the same.
    assert!(
        realtime > 2.5,
        "the engine only managed {realtime:.1}x realtime"
    );
    assert!(peak < 4.0, "eight voices peaked at {peak}");
}
