//! Any string, any excitation, and the player's left hand.

use i_am_string::prelude::*;

const FS: f32 = 48_000.0;

/// A preset describes a string and a body. What touches the string is a
/// separate choice, and every one of them has to make a sound on every one.
///
/// This is not a hypothetical: the parameters of a hammer used to be filled in
/// only for the piano, so switching a guitar to a hammer handed the solver a
/// felt of zero stiffness and the note simply never happened.
#[test]
fn every_preset_plays_every_way() {
    for preset in Preset::ALL {
        for excitation in Excitation::ALL {
            for note in [45u8, 60, 76] {
                let mut params = preset.defaults();
                params.excitation = excitation;
                if excitation == Excitation::Hammer {
                    // A guitar is not built to be beaten, so it needs a lighter
                    // hammer than a piano's before it sounds like anything.
                    params.hardness = 0.45;
                }
                let mut engine = StringEngine::new(FS);
                engine.set_preset(preset);
                engine.apply_params(params);
                engine.note_on(note, 1.0);
                let mut peak = 0.0f32;
                for _ in 0..(FS * 1.5) as usize {
                    let value = engine.next_sample();
                    assert!(value.is_finite(), "{} produced {value}", preset.name());
                    peak = peak.max(value.abs());
                }
                println!(
                    "{} {} note {note}: peak {peak:.4}",
                    preset.name(),
                    excitation.name()
                );
                assert!(
                    peak > 1e-3,
                    "{} played with a {} at note {note} peaked at {peak}, which is \
                     silence",
                    preset.name(),
                    excitation.name()
                );
                // A three string note is three strings. The reference force
                // each preset is scaled against is the force *one* string
                // makes, so a piano plucked - which is three strings released
                // together and staying in step for the length of the note - is
                // legitimately about three times a guitar. This bound is a
                // runaway guard, not a mixing decision.
                assert!(
                    peak < 8.0,
                    "{} played with a {} at note {note} peaked at {peak}, which is \
                     running away",
                    preset.name(),
                    excitation.name()
                );
            }
        }
    }
}

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

/// Where the string is actually ringing, near `wanted`.
fn find_peak(samples: &[f32], wanted: f32) -> f32 {
    let mut best = wanted;
    let mut best_energy = 0.0f32;
    let steps = 200;
    // Plus or minus two percent, which is three times the widest bend here.
    for step in 0..steps {
        let ratio = 0.98 * (1.04f32).powf(step as f32 / (steps - 1) as f32);
        let probe = wanted * ratio;
        let energy = goertzel(samples, probe);
        if energy > best_energy {
            best_energy = energy;
            best = probe;
        }
    }
    best
}

/// A bend has to move the pitch by as much as it says, on both cores.
///
/// This is the reason a bend is not a chorus or a pitch shifter: it is the
/// string itself, so every partial moves together and the ratios between them
/// are untouched. The test asks the rendered sound where it is ringing rather
/// than asking the model what it thinks it is doing.
#[test]
fn a_bend_moves_the_pitch() {
    for core in Core::ALL {
        for preset in [Preset::Guitar, Preset::Violin] {
            let params = preset.defaults();
            let spec = preset.spec(60, 1.0, &params);
            let f0 = spec.string.frequency;
            for cents in [-40.0f32, 0.0, 50.0] {
                let ratio = 2f32.powf(cents / 1200.0);
                let mut string = StringCore::new(core, FS, &spec.string);
                string.set_pitch_ratio(ratio);
                if preset == Preset::Violin {
                    let mut law = BowLaw::new(spec.bow_force, spec.bow_velocity);
                    law.noise = 0.0015;
                    string.add_port(spec.position, Box::new(law), spec.contact_substeps);
                } else {
                    string.pluck(0.29, 6e-4);
                }
                // Two seconds, so that fifty cents at this pitch is several
                // transform bins wide rather than a fraction of one.
                let mut buffer: Vec<f32> = (0..(FS * 2.0) as usize)
                    .map(|_| {
                        string.set_pitch_ratio(ratio);
                        string.step()
                    })
                    .collect();
                let last = (buffer.len() - 1) as f32;
                for (index, sample) in buffer.iter_mut().enumerate() {
                    let t = index as f32 / last;
                    *sample *= 0.5 - 0.5 * (std::f32::consts::TAU * t).cos();
                }
                let wanted = f0 * ratio;
                let found = find_peak(&buffer, wanted);
                let error = 1200.0 * (found / wanted).log2();
                println!(
                    "{} on the {} core at {cents:+.0} cents: ringing at {found:.2} Hz, \
                     wanted {wanted:.2}, {error:+.1} cents out",
                    preset.name(),
                    core.name()
                );
                assert!(
                    error.abs() < 25.0,
                    "{} on the {} core was asked to bend {cents:+.0} cents and rang \
                     {error:+.1} cents out",
                    preset.name(),
                    core.name()
                );
            }
        }
    }
}
