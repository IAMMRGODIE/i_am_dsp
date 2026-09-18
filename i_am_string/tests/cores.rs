//! The two string cores against each other.
//!
//! The crate keeps two ways of modelling the same string. These are the
//! promises that make the choice a control rather than a fork: both are stable,
//! both are in the same place in level, and the mode bank is in tune at
//! stiffnesses where the delay lines are not.

use i_am_string::prelude::*;

const FS: f32 = 48_000.0;

/// The energy of `samples` at `frequency`, by the Goertzel recurrence.
fn goertzel(samples: &[f32], frequency: f32) -> f32 {
    let omega = std::f32::consts::TAU * frequency / FS;
    let coefficient = 2.0 * omega.cos();
    let (mut s1, mut s2) = (0.0f32, 0.0f32);
    for sample in samples {
        let s0 = sample + coefficient * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    (s1 * s1 + s2 * s2 - coefficient * s1 * s2).max(0.0)
}

fn rms(samples: &[f32]) -> f32 {
    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len().max(1) as f32).sqrt()
}

fn pluck_and_render(
    core: Core,
    config: &StringConfig,
    position: f32,
    amplitude: f32,
    samples: usize,
) -> Vec<f32> {
    let mut string = StringCore::new(core, FS, config);
    string.pluck(position, amplitude);
    let mut buffer: Vec<f32> = (0..samples).map(|_| string.step()).collect();
    // The impulse response is still ringing at the end of the window, and a
    // rectangular cut spreads every peak into a sinc that is wider than the
    // error being measured.
    let last = (buffer.len() - 1) as f32;
    for (index, sample) in buffer.iter_mut().enumerate() {
        let t = index as f32 / last;
        *sample *= 0.5 - 0.5 * (std::f32::consts::TAU * t).cos();
    }
    buffer
}

/// Where a string actually rings, read off its own rendered output.
///
/// Asking a model for its partial frequencies reports what it was built to do.
/// The ear hears what came out, so the predicted partial is hunted for in the
/// spectrum over a window wide enough to find it however far the two cores
/// disagree.
fn measure_partial(samples: &[f32], wanted: f32) -> (f32, f32) {
    let mut best = wanted;
    let mut best_energy = 0.0f32;
    let steps = 160;
    for step in 0..steps {
        let ratio = 0.90 * (1.20f32).powf(step as f32 / (steps - 1) as f32);
        let probe = wanted * ratio;
        let energy = goertzel(samples, probe);
        if energy > best_energy {
            best_energy = energy;
            best = probe;
        }
    }
    (best, best_energy)
}

fn level(core: Core, preset: Preset, note: u8) -> f32 {
    let mut engine = StringEngine::new(FS);
    engine.set_preset(preset);
    engine.params_mut().core = core;
    engine.note_on(note, 1.0);
    let frames = (FS * 1.5) as usize;
    let mut peak = 0.0f32;
    for _ in 0..frames {
        peak = peak.max(engine.next_sample().abs());
    }
    peak
}

/// The whole point of the architecture, stated as a test.
///
/// A stiff string resonates at `n f0 sqrt(1 + B n^2)`, not at `n f0`. The mode
/// bank has that law in it, so it puts its energy in the stretched series. The
/// delay lines have a filter that has to *fit* that law, and once the string is
/// stiff enough the fit is what gives out. So: at a stiffness a piano's treble
/// would recognise, does the string ring where the physics says it should?
#[test]
fn the_mode_bank_is_in_tune_where_the_waveguide_is_not() {
    // C7 on a grand piano, where the stretch is hundreds of cents at the top of
    // the modelled series and a mistuned partial is plainly audible.
    let engine = StringEngine::new(FS);
    let spec = Preset::Piano.spec(96, 1.0, engine.params());
    let config = &spec.string;
    let b = config.inharmonicity;
    assert!(b > 2e-3, "this test needs a stiff string, got B = {b}");

    let predicate = |n: usize| {
        let order = n as f32;
        config.frequency * order * (1.0 + b * order * order).sqrt()
    };

    let mut report = Vec::new();
    for core in Core::ALL {
        // Off-centre, so the even partials are excited as well as the odd ones.
        let samples = pluck_and_render(core, config, 0.29, 6e-4, 1 << 14);
        assert!(rms(&samples) > 1e-12, "{} was silent", core.name());
        let mut worst = 0.0f32;
        let mut found = 0;
        for n in 1..=8 {
            let wanted = predicate(n);
            let (measured, energy) = measure_partial(&samples, wanted);
            // Only count a partial that is really there.
            if energy.sqrt() < 1e-6 {
                continue;
            }
            found += 1;
            let cents = (1200.0 * (measured / wanted).log2()).abs();
            worst = worst.max(cents);
        }
        println!("{}: {found} of 8 partials found, worst {worst:.1} cents", core.name());
        report.push((core, found, worst));
    }

    let (_, modal_found, modal_worst) = report
        .iter()
        .find(|(core, _, _)| *core == Core::Modal)
        .copied()
        .unwrap();
    assert!(
        modal_found >= 6,
        "the mode bank only produced {modal_found} measurable partials"
    );
    assert!(
        modal_worst < 5.0,
        "the mode bank's partials are up to {modal_worst:.1} cents out, and it is \
         supposed to be exact by construction"
    );

    let (_, _, waveguide_worst) = report
        .iter()
        .find(|(core, _, _)| *core == Core::Waveguide)
        .copied()
        .unwrap();
    assert!(
        waveguide_worst > 10.0,
        "the waveguide's partials are only {waveguide_worst:.1} cents out, so this \
         test is no longer measuring the difference it claims to"
    );
}

/// Switching cores must not be a level change, or nobody can hear the timbre.
#[test]
fn the_two_cores_play_at_the_same_level() {
    for preset in Preset::ALL {
        for note in [48u8, 60, 72] {
            let waveguide = level(Core::Waveguide, preset, note);
            let modal = level(Core::Modal, preset, note);
            let ratio = waveguide / modal.max(1e-9);
            println!(
                "{} at {note}: waveguide {waveguide:.4}, modal {modal:.4}, ratio {ratio:.2}",
                preset.name()
            );
            assert!(
                (0.55..1.8).contains(&ratio),
                "{} at note {note} is {ratio:.2} times louder on the waveguide",
                preset.name()
            );
        }
    }
}

/// A mode that has fallen below the noise floor must actually stop working.
///
/// This is a performance contract, not a cosmetic one. A geometric sequence left
/// running reaches the denormal range, where every operation on it drops out of
/// the pipeline into microcode and costs a hundred times as much while sounding
/// exactly the same. Measured, a live mode bank costs a sixth of a microsecond a
/// sample and the same bank left ringing for a second costs seven.
#[test]
fn a_settled_string_stops_computing() {
    let mut engine = StringEngine::new(FS);
    engine.set_preset(Preset::Guitar);
    let spec = Preset::Guitar.spec(60, 1.0, engine.params());
    let mut string = StringCore::new(Core::Modal, FS, &spec.string);
    string.pluck(spec.position, spec.pluck_amplitude);
    for _ in 0..(FS as usize * 8) {
        string.step();
    }
    assert!(
        string.is_quiet(),
        "the string never settled, envelope {}",
        string.envelope()
    );
    // Every mode is now below its own inaudibility threshold, so the sum is down
    // in the noise and no longer moving.
    let mut peak = 0.0f32;
    for _ in 0..(FS as usize * 4) {
        peak = peak.max(string.step().abs());
    }
    assert!(
        peak < 1e-5,
        "a settled string was still moving at {peak}, which is no longer a floor"
    );
}

/// Both cores, every preset, every register: bounded, finite and audible.
#[test]
fn every_preset_is_stable_on_both_cores() {
    for core in Core::ALL {
        for preset in Preset::ALL {
            for note in [30u8, 48, 60, 72, 96, 108] {
                let mut engine = StringEngine::new(FS);
                engine.set_preset(preset);
                engine.params_mut().core = core;
                engine.note_on(note, 1.0);
                let mut peak = 0.0f32;
                for _ in 0..(FS as usize / 2) {
                    let value = engine.next_sample();
                    assert!(value.is_finite(), "{} at {note} produced {value}", preset.name());
                    peak = peak.max(value.abs());
                }
                assert!(
                    peak < 4.0,
                    "{} on the {} core at note {note} peaked at {peak}",
                    preset.name(),
                    core.name()
                );
                assert!(
                    peak > 1e-4,
                    "{} on the {} core at note {note} never sounded",
                    preset.name(),
                    core.name()
                );
            }
        }
    }
}
