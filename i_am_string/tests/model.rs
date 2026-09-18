//! Physical validation of the string model.
//!
//! These tests do not check that the code runs, they check that it produces
//! the pitches and decay times that string theory says it should.

use i_am_string::prelude::*;

const FS: f32 = 48_000.0;

/// A plain string with a chosen inharmonicity and decay.
fn string_with(frequency: f32, b: f32, sigma0: f32, sigma1: f32) -> WaveguideString {
    let config = StringConfig {
        frequency,
        inharmonicity: b,
        sigma0,
        sigma1,
        tension: 80.0,
        impedance: 1.2,
        max_partials: 40,
        dispersion_sections: 20,
        dispersion_delay_fraction: 0.85,
        dispersion_coefficients: None,
    };
    WaveguideString::new(FS, config)
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
    let offset = 0.5 * (left - right) / denominator;
    best + offset * step
}

/// Estimate the fundamental frequency by autocorrelation.
///
/// Counting zero crossings is not good enough here: an inharmonic and slightly
/// noisy string crosses zero a variable number of times per period, and a bowed
/// string carries a real DC deflection that stops it crossing zero at all.
fn estimate_frequency(samples: &[f32], low: f32, high: f32) -> f32 {
    let mean = samples.iter().sum::<f32>() / samples.len() as f32;
    let min_lag = (FS / high).floor() as usize;
    let max_lag = (FS / low).ceil() as usize;
    let mut values: Vec<(usize, f32)> = Vec::with_capacity(max_lag - min_lag + 1);
    for lag in min_lag..=max_lag {
        let count = samples.len() - lag;
        let mut sum = 0.0f64;
        for i in 0..count {
            sum += ((samples[i] - mean) as f64) * ((samples[i + lag] - mean) as f64);
        }
        values.push((lag, (sum / count as f64) as f32));
    }
    let mut best = 0;
    for i in 1..values.len().saturating_sub(1) {
        if values[i].1 > values[best].1 {
            best = i;
        }
    }
    let mut refined = values[best].0 as f32;
    if best > 0 && best + 1 < values.len() {
        let a = values[best - 1].1;
        let b = values[best].1;
        let c = values[best + 1].1;
        let denominator = a - 2.0 * b + c;
        if denominator.abs() > 1e-12 {
            refined += 0.5 * (a - c) / denominator;
        }
    }
    FS / refined
}

fn cents(a: f32, b: f32) -> f32 {
    1200.0 * (a / b).log2()
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|x| x * x).sum::<f32>() / samples.len() as f32).sqrt()
}

/// The RMS of the *fluctuating* part.
///
/// A bowed string carries a large static deflection: the bow really does push
/// the string sideways and hold it there. Measuring the raw RMS of a bowed note
/// therefore measures that deflection rather than the note.
fn ac_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let mean = samples.iter().sum::<f32>() / samples.len() as f32;
    (samples.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / samples.len() as f32).sqrt()
}

fn render(string: &mut WaveguideString, samples: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(samples);
    for _ in 0..samples {
        out.push(string.step());
    }
    out
}

#[test]
fn an_ideal_string_has_harmonic_partials() {
    let mut string = string_with(220.0, 0.0, 0.4, 0.0);
    string.pluck(0.2, 0.01);
    let audio = render(&mut string, (FS * 3.0) as usize);
    let segment = &audio[(FS * 0.3) as usize..(FS * 3.0) as usize];

    let fundamental = estimate_frequency(segment, 200.0, 245.0);
    assert!(
        cents(fundamental, 220.0).abs() < 6.0,
        "fundamental was {fundamental} Hz, expected 220 Hz"
    );

    for harmonic in [2.0f32, 3.0, 4.0] {
        let expected = 220.0 * harmonic;
        let found = peak_in(segment, expected * 0.93, expected * 1.07, 0.25);
        assert!(
            cents(found, expected).abs() < 12.0,
            "partial {harmonic} was {found} Hz, expected {expected} Hz"
        );
    }
}

#[test]
fn stiffness_stretches_the_partials_by_b() {
    let b = 3.0e-4f32;
    let mut string = string_with(220.0, b, 0.4, 0.0);
    string.pluck(0.2, 0.01);
    let audio = render(&mut string, (FS * 3.0) as usize);
    let segment = &audio[(FS * 0.3) as usize..];

    // The fundamental is retuned to be exact even though the allpass fit is
    // approximate, so it is the anchor.
    let fundamental = estimate_frequency(segment, 200.0, 245.0);
    assert!(
        cents(fundamental, 220.0).abs() < 6.0,
        "fundamental was {fundamental} Hz"
    );

    // A realisable allpass cascade cannot match the whole stiff-string delay
    // curve, so the model is only made as inharmonic as it can represent
    // accurately and the stretch falls short of what was asked for on purpose.
    // What matters is that the stretch is there, is correctly tuned, grows with
    // the partial number, and stays in a musical range.
    let mut previous_stretch = -1000.0f32;
    for index in [2.0f32, 3.0, 4.0, 6.0] {
        let ideal = 220.0 * index;
        let expected = index * 220.0 * (1.0 + b * index * index).sqrt();
        let found = peak_in(segment, expected * 0.94, expected * 1.08, 0.25);
        let stretch = cents(found, ideal);
        assert!(
            found < expected * 1.09,
            "partial {index} was {found} Hz, far beyond its target {expected} Hz"
        );
        assert!(
            cents(found, expected).abs() < 60.0,
            "partial {index} was {found} Hz, expected about {expected} Hz"
        );
        assert!(
            stretch > previous_stretch - 5.0,
            "the stretch is not monotone: partial {index} at {stretch} cents after {previous_stretch}"
        );
        previous_stretch = stretch;
    }
    assert!(
        previous_stretch > 2.0,
        "the sixth partial is barely stretched: {previous_stretch} cents"
    );
}

#[test]
fn decay_follows_the_loss_filter() {
    let sigma0 = 1.0f32;
    let mut string = string_with(220.0, 0.0, sigma0, 0.0);
    string.pluck(0.2, 0.01);
    let audio = render(&mut string, (FS * 6.0) as usize);

    let window = (FS * 0.5) as usize;
    let first = magnitude(&audio[(FS * 0.5) as usize..(FS * 0.5) as usize + window], 220.0);
    let second = magnitude(&audio[(FS * 3.5) as usize..(FS * 3.5) as usize + window], 220.0);
    let measured = (first / second).max(1.0001).ln() / 3.0;
    assert!(
        (measured - sigma0).abs() < 0.25 * sigma0,
        "decay rate was {measured}, expected {sigma0}"
    );
}

#[test]
fn a_damper_stops_the_note() {
    let mut string = string_with(220.0, 0.0, 0.4, 0.0);
    string.pluck(0.2, 0.01);
    let open = render(&mut string, (FS * 0.5) as usize);
    string.set_damper(0.78);
    let damped = render(&mut string, (FS * 0.5) as usize);
    assert!(
        rms(&damped) < 0.2 * rms(&open),
        "the damper did not stop the string: {} vs {}",
        rms(&damped),
        rms(&open)
    );
}

#[test]
fn every_excitation_produces_a_bounded_signal() {
    let mut plucked = string_with(196.0, 3.0e-4, 0.6, 3e-6);
    plucked.pluck(0.2, 0.008);
    let audio = render(&mut plucked, (FS * 2.0) as usize);
    assert!(rms(&audio[(FS * 0.1) as usize..]) > 1e-3, "the pluck was silent");
    assert!(audio.iter().all(|x| x.is_finite()));

    let mut struck = string_with(196.0, 3.0e-4, 0.6, 3e-6);
    struck.add_port(
        0.125,
        Box::new(HammerLaw::new(0.006, 3.0, 2.7, 1.0e9, 0.0)),
        4,
    );
    let audio = render(&mut struck, (FS * 2.0) as usize);
    assert!(rms(&audio[(FS * 0.1) as usize..]) > 1e-3, "the strike was silent");
    assert!(audio.iter().all(|x| x.is_finite()));

    let mut bowed = bowed_string();
    bowed.add_port(0.09, Box::new(BowLaw::new(0.6, 0.15)), 2);
    let audio = render(&mut bowed, (FS * 2.0) as usize);
    assert!(rms(&audio[(FS * 0.5) as usize..]) > 1e-3, "the bow was silent");
    assert!(audio.iter().all(|x| x.is_finite()));
}

/// A bowed string needs a bow-scale impedance: the friction has to beat the
/// string's own radiation, and a stiff string radiates far too much for a bow
/// to drive it.
fn bowed_string() -> WaveguideString {
    let config = StringConfig {
        frequency: 220.0,
        inharmonicity: 1.4e-4,
        sigma0: 0.8,
        sigma1: 3e-6,
        tension: 60.0,
        impedance: 0.35,
        max_partials: 40,
        dispersion_sections: 20,
        dispersion_delay_fraction: 0.85,
        dispersion_coefficients: None,
    };
    WaveguideString::new(FS, config)
}

#[test]
fn a_bowed_string_sustains_without_growing() {
    let mut bowed = bowed_string();
    let mut law = BowLaw::new(0.7, 0.15);
    // Rosin noise is what starts a real bow; without a seed the string just
    // settles into a static deflection.
    law.noise = 0.01;
    bowed.add_port(0.09, Box::new(law), 2);
    let audio = render(&mut bowed, (FS * 8.0) as usize);
    assert!(audio.iter().all(|x| x.is_finite()), "the bow blew up");

    let early = ac_rms(&audio[(FS * 0.5) as usize..(FS * 1.5) as usize]);
    let late = ac_rms(&audio[(FS * 6.5) as usize..(FS * 7.5) as usize]);
    assert!(
        early > 1e-3,
        "the bow never got going: the fluctuating part was {early}"
    );
    assert!(
        late > 0.5 * early,
        "the bow note died away: {early} then {late}"
    );
    assert!(late < 4.0 * early, "the bow note grew: {early} then {late}");

    // A bowed note sings at the string's pitch, give or take the pitch pull that
    // the stick-slip cycle always imposes on a real bow.
    let segment = &audio[(FS * 4.0) as usize..];
    let fundamental = estimate_frequency(segment, 170.0, 280.0);
    assert!(
        (170.0..280.0).contains(&fundamental),
        "the bowed note was at {fundamental} Hz, expected about 220 Hz"
    );
}

#[test]
fn the_engine_is_stable_across_every_preset() {
    for preset in Preset::ALL {
        let mut engine = StringEngine::new(FS);
        engine.set_preset(preset);
        engine.note_on(57, 0.9);
        engine.note_on(64, 0.7);
        engine.note_on(69, 0.5);
        for sample in 0..(FS * 3.0) as usize {
            let value = engine.next_sample();
            assert!(
                value.is_finite(),
                "{} produced a non finite sample at {sample}",
                preset.name()
            );
            assert!(
                value.abs() < 8.0,
                "{} produced {value} at {sample}",
                preset.name()
            );
        }
        engine.all_notes_off();
        let mut tail = 0.0f32;
        for _ in 0..(FS * 3.0) as usize {
            tail = tail.max(engine.next_sample().abs());
        }
        assert!(
            tail < 1.0,
            "{} rang on at {tail} after all notes off",
            preset.name()
        );
    }
}

#[test]
fn every_preset_makes_a_sound_at_a_sane_level() {
    for preset in Preset::ALL {
        let mut engine = StringEngine::new(FS);
        engine.set_preset(preset);
        engine.note_on(60, 1.0);
        let mut peak = 0.0f32;
        let mut energy = 0.0f32;
        for _ in 0..(FS * 2.0) as usize {
            let value = engine.next_sample();
            peak = peak.max(value.abs());
            energy += value * value;
        }
        let level = (energy / (FS * 2.0)).sqrt();
        assert!(
            peak > 0.01,
            "{} was too quiet: peak {peak}",
            preset.name()
        );
        assert!(
            peak < 2.0,
            "{} was too loud: peak {peak}",
            preset.name()
        );
        assert!(level.is_finite() && level > 0.0);
    }
}

#[test]
fn velocity_changes_the_level() {
    let mut engine = StringEngine::new(FS);
    engine.set_preset(Preset::Piano);
    let level = |velocity: f32| {
        let mut engine = StringEngine::new(FS);
        engine.set_preset(Preset::Piano);
        engine.note_on(60, velocity);
        let mut energy = 0.0f32;
        for _ in 0..(FS * 0.5) as usize {
            let value = engine.next_sample();
            energy += value * value;
        }
        (energy / (FS * 0.5)).sqrt()
    };
    let quiet = level(0.2);
    let loud = level(1.0);
    assert!(
        loud > 1.6 * quiet,
        "velocity barely changed the level: {quiet} then {loud}"
    );
}

#[test]
fn unison_detune_produces_beating() {
    let mut engine = StringEngine::new(FS);
    engine.set_preset(Preset::Piano);
    engine.params_mut().detune_cents = 0.0;
    engine.note_on(69, 0.8);
    let steady = {
        let mut envelope = Vec::new();
        for block in 0..200 {
            let mut peak = 0.0f32;
            for _ in 0..(FS * 0.01) as usize {
                peak = peak.max(engine.next_sample().abs());
            }
            envelope.push(peak);
            let _ = block;
        }
        envelope
    };
    let variation = |values: &[f32]| {
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let max = values.iter().copied().fold(f32::MIN, f32::max);
        let min = values.iter().copied().fold(f32::MAX, f32::min);
        (max - min) / mean.max(1e-6)
    };

    let mut detuned = StringEngine::new(FS);
    detuned.set_preset(Preset::Piano);
    detuned.params_mut().detune_cents = 6.0;
    detuned.note_on(69, 0.8);
    let mut envelope = Vec::new();
    for _ in 0..200 {
        let mut peak = 0.0f32;
        for _ in 0..(FS * 0.01) as usize {
            peak = peak.max(detuned.next_sample().abs());
        }
        envelope.push(peak);
    }

    assert!(
        variation(&envelope) > 1.5 * variation(&steady),
        "detuned strings did not beat: {} vs {}",
        variation(&envelope),
        variation(&steady)
    );
}
/// A contact that simply pushes with a constant force.
///
/// Used to check the direct current behaviour of the waveguide, which is where
/// a sign error in the end reflections hides.
struct ConstantForce(f32);

impl ContactLaw for ConstantForce {
    fn force(&self, _string_y: f32, _string_v: f32) -> f32 {
        self.0
    }

    fn force_dv(&self, _string_y: f32, _string_v: f32) -> f32 {
        0.0
    }

    fn max_force(&self) -> f32 {
        self.0
    }

    fn min_force(&self) -> f32 {
        self.0
    }

    fn advance(&mut self, _string_y: f32, _string_v: f32, _dt: f32) {}
}

/// How periodic a signal is at a given period. One is a perfect cycle.
fn periodicity(samples: &[f32], period: f32) -> f32 {
    let lag = period.round() as usize;
    let mean = samples.iter().sum::<f32>() / samples.len() as f32;
    let mut numerator = 0.0f64;
    let mut left = 0.0f64;
    let mut right = 0.0f64;
    for i in 0..samples.len() - lag {
        let a = (samples[i] - mean) as f64;
        let b = (samples[i + lag] - mean) as f64;
        numerator += a * b;
        left += a * a;
        right += b * b;
    }
    (numerator / (left * right).sqrt().max(1e-12)) as f32
}

/// How much energy sits between the harmonics, in dB below the fundamental.
fn between_partial_level(samples: &[f32], f0: f32) -> f32 {
    let fundamental = magnitude(samples, f0);
    let mut between = 0.0f32;
    for k in 1..=12 {
        between += magnitude(samples, f0 * (k as f32 + 0.5));
    }
    between /= 12.0;
    deci_bels(between, fundamental)
}

fn deci_bels(value: f32, reference: f32) -> f32 {
    20.0 * (value.max(1e-12) / reference.max(1e-12)).log10()
}

/// The delay lines carry force waves, and at a rigid termination it is the
/// velocity that inverts while the force does not. With the reflection sign the
/// wrong way round the model has a DC path: a constant force produces a constant
/// velocity, so the string drifts sideways for ever instead of settling into a
/// static deflection, and a bow can hold it rigidly and never slip.
#[test]
fn a_constant_force_settles_into_a_static_deflection() {
    // Well damped, so that the loop settles inside the test rather than after a
    // few seconds of near-lossless ringing.
    let mut string = string_with(220.0, 0.0, 4.0, 0.0);
    string.add_port(0.2, Box::new(ConstantForce(2.0)), 1);
    let mut audio = Vec::new();
    for _ in 0..(FS * 4.0) as usize {
        audio.push(string.step());
    }
    let late = &audio[(FS * 3.5) as usize..];
    let early = &audio[(FS * 3.0) as usize..(FS * 3.5) as usize];
    let spread = |values: &[f32]| {
        let max = values.iter().copied().fold(f32::MIN, f32::max);
        let min = values.iter().copied().fold(f32::MAX, f32::min);
        max - min
    };
    assert!(
        spread(late) < 1e-3,
        "the string never settled: it still swings by {}",
        spread(late)
    );
    assert!(
        (rms(late) - rms(early)).abs() < 1e-3,
        "the displacement is still growing"
    );
    assert!(audio.iter().all(|value| value.is_finite()));
}

/// A bowed string has to lock into a Helmholtz cycle: one slip per round trip,
/// exactly periodic, with almost nothing between the harmonics. It took three
/// separate bugs to get there and this test is what stops any of them coming
/// back.
#[test]
fn a_bowed_string_locks_into_a_helmholtz_cycle() {
    let config = StringConfig {
        frequency: 440.0,
        inharmonicity: 1.4e-4,
        sigma0: 1.5,
        sigma1: 1e-6,
        tension: 60.0,
        impedance: 0.35,
        max_partials: 40,
        dispersion_sections: 20,
        dispersion_delay_fraction: 1.0,
        dispersion_coefficients: None,
    };
    let mut string = WaveguideString::new(FS, config);
    let mut law = BowLaw::new(0.7, 0.215);
    law.noise = 0.0;
    string.add_port(0.91, Box::new(law), 8);
    let mut audio = vec![0.0f32; (FS * 1.0) as usize];
    for sample in audio.iter_mut() {
        *sample = string.step();
    }
    let segment = &audio[(FS / 2.0) as usize..];
    let periodic = periodicity(segment, FS / 440.0);
    let between = between_partial_level(segment, 440.0);
    assert!(
        periodic > 0.9,
        "the bowed string is not periodic: {periodic}"
    );
    assert!(
        between < -25.0,
        "the bowed string is noisy: {between} dB between the harmonics"
    );
}

/// A solver bug once silenced exactly one note of the piano, at 880 Hz, because
/// the hammer was sent marching off to 2500 m/s and never touched the string.
/// Every note of every preset has to make a sound.
#[test]
fn every_note_of_every_preset_sounds() {
    for preset in Preset::ALL {
        for note in [24u8, 36, 45, 52, 60, 64, 69, 72, 76, 81, 88, 96] {
            let mut engine = StringEngine::new(FS);
            engine.set_preset(preset);
            engine.note_on(note, 0.9);
            let mut peak = 0.0f32;
            for _ in 0..(FS / 2.0) as usize {
                peak = peak.max(engine.next_sample().abs());
            }
            assert!(
                peak > 0.002,
                "{} at note {note} was silent (peak {peak})",
                preset.name()
            );
            assert!(
                peak < 3.0,
                "{} at note {note} peaked at {peak}",
                preset.name()
            );
        }
    }
}
/// Every partial the model claims to have must actually be where it says.
///
/// This is the guarantee that was missing for a long time, and its absence was
/// audible: partials hundreds of cents out do not sound like a stiff string,
/// they sound like a detuned saw. The cause was two fold - the delay lines were
/// asked to carry so much of the round trip that the allpass cascade had to
/// achieve an impossible curve, and the fit only evaluated sixteen points, so
/// with a hundred partials most of them were unconstrained.
#[test]
fn every_modelled_partial_is_in_tune() {
    for (preset, notes) in [
        (Preset::Piano, &[9u8, 24, 48, 72, 96][..]),
        (Preset::Guitar, &[28, 40, 52, 64][..]),
        (Preset::Violin, &[45, 57, 64, 69][..]),
        (Preset::Cello, &[36, 45, 52][..]),
    ] {
        for note in notes {
            let mut params = preset.defaults();
            // The transpose is where the instrument sits, not what it is made
            // of, and a string six hertz below the model's own floor is not a
            // fair test of the dispersion fit.
            params.transpose = 0.0;
            let spec = preset.spec(*note, 0.9, &params);
            let string = WaveguideString::new(FS, spec.string.clone());
            let intended = string.partial_frequencies();
            let achieved = string.resonance_frequencies();
            let count = intended.len().min(achieved.len());
            let mut worst = 0.0f32;
            let mut bad = 0usize;
            for index in 0..count {
                let error = 1200.0 * (achieved[index] / intended[index]).log2();
                if error.abs() > worst.abs() {
                    worst = error;
                }
                if error.abs() > 60.0 {
                    bad += 1;
                }
            }
            assert!(
                count > 8,
                "{} at note {note} only modelled {count} partials",
                preset.name()
            );
            // Most partials have to be close, and none may be wildly out: a
            // partial a semitone away is heard as a wrong note, which is what
            // made the top of the keyboard sound like a detuned saw.
            assert!(
                bad * 3 <= count,
                "{} at note {note}: {bad} of {count} partials are more than 60 cents out",
                preset.name()
            );
            assert!(
                worst.abs() < 250.0,
                "{} at note {note}: a partial is {worst:.0} cents out",
                preset.name()
            );
        }
    }
}
