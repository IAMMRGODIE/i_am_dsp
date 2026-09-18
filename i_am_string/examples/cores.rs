//! A measurement harness for the two string cores.
//!
//! Run with `cargo run --release --example cores`.

use i_am_string::prelude::*;
use std::time::Instant;

const FS: f32 = 48_000.0;

fn level(engine: &mut StringEngine, seconds: f32) -> (f32, f32) {
    let frames = (FS * seconds) as usize;
    let mut peak = 0.0f32;
    let mut energy = 0.0f32;
    for _ in 0..frames {
        let value = engine.next_sample();
        peak = peak.max(value.abs());
        energy += value * value;
    }
    (peak, (energy / frames as f32).sqrt())
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

/// Where a single string actually resonates, read off its own impulse response.
///
/// Asking the model for its partial frequencies only reports what it was asked
/// to build. The point of the comparison is what came out, so each predicted
/// partial is hunted for in the rendered spectrum, over a window wide enough to
/// find it however far the two cores disagree.
fn measured_partials(core: Core, config: &StringConfig, count: usize) -> Vec<f32> {
    let mut string = StringCore::new(core, FS, config);
    string.pluck(0.5, 1e-4);
    let mut buffer = vec![0.0f32; 1 << 16];
    for sample in buffer.iter_mut() {
        *sample = string.step();
    }
    // A gentle taper: the impulse response is still ringing at the end of the
    // window, and a rectangular cut spreads every peak into a sinc.
    let last = (buffer.len() - 1) as f32;
    for (index, sample) in buffer.iter_mut().enumerate() {
        let t = index as f32 / last;
        *sample *= 0.5 - 0.5 * (std::f32::consts::TAU * t).cos();
    }

    let mut found = Vec::new();
    let mut rms = 0.0f32;
    for sample in &buffer {
        rms += sample * sample;
    }
    rms = (rms / buffer.len() as f32).sqrt().max(1e-30);
    for index in 1..=count {
        let order = index as f32;
        let wanted =
            config.frequency * order * (1.0 + config.inharmonicity * order * order).sqrt();
        if wanted >= FS * 0.45 {
            break;
        }
        // Search plus or minus six percent on a grid fine enough that the peak
        // is located to a couple of cents.
        let mut best = wanted;
        let mut best_energy = 0.0f32;
        let steps = 240;
        for step in 0..steps {
            let ratio = 0.94 * (1.12f32).powf(step as f32 / (steps - 1) as f32);
            let probe = wanted * ratio;
            let energy = goertzel(&buffer, probe);
            if energy > best_energy {
                best_energy = energy;
                best = probe;
            }
        }
        // Only report it if there is really a resonance there.
        if best_energy.sqrt() / (buffer.len() as f32 * rms) > 1e-3 {
            found.push(best);
        } else {
            found.push(f32::NAN);
        }
    }
    found
}

fn main() {
    println!("{:>12} {:>10} {:>8} {:>8}", "preset", "core", "peak", "rms");
    for core in Core::ALL {
        for preset in Preset::ALL {
            let mut engine = StringEngine::new(FS);
            engine.set_preset(preset);
            engine.params_mut().core = core;
            engine.note_on(60, 1.0);
            let (peak, rms) = level(&mut engine, 2.0);
            println!(
                "{:>12} {:>10} {:>8.4} {:>8.4}",
                preset.name(),
                core.name(),
                peak,
                rms
            );
        }
    }

    // The architectural claim: the modal bank is in tune at any stiffness, and
    // the waveguide is only in tune while the stiffness is small.
    println!();
    println!(
        "{:>12} {:>5} {:>7} {:>10} {:>10} {:>10}",
        "core", "note", "B", "worst ct", "median ct", "on pitch"
    );
    for preset in [Preset::Piano, Preset::Guitar] {
        for note in [36u8, 60, 84, 96] {
            let spec = preset.spec(note, 1.0, &preset.defaults());
            for core in Core::ALL {
                let found = measured_partials(core, &spec.string, 24);
                let mut worst = 0.0f32;
                let mut errors = Vec::new();
                for (index, frequency) in found.iter().enumerate() {
                    if !frequency.is_finite() {
                        continue;
                    }
                    let order = (index + 1) as f32;
                    let b = spec.string.inharmonicity;
                    let wanted =
                        spec.string.frequency * order * (1.0 + b * order * order).sqrt();
                    let cents = (1200.0 * (frequency / wanted).log2()).abs();
                    errors.push(cents);
                    worst = worst.max(cents);
                }
                errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median = errors.get(errors.len() / 2).copied().unwrap_or(f32::NAN);
                let on_pitch = errors.iter().filter(|cents| **cents < 25.0).count();
                println!(
                    "{:>12} {:>5} {:>7.1e} {:>10.1} {:>10.1} {:>6}/{:<3}",
                    core.name(),
                    note,
                    spec.string.inharmonicity,
                    worst,
                    median,
                    on_pitch,
                    errors.len()
                );
            }
        }
    }

    // Raw cost of one string, without the body or the sympathetic bank.
    println!();
    println!("{:>12} {:>10} {:>10}", "preset", "core", "us/sample");
    for core in Core::ALL {
        for preset in Preset::ALL {
            let spec = preset.spec(48, 1.0, &preset.defaults());
            let mut string = StringCore::new(core, FS, &spec.string);
            string.pluck(spec.position, spec.pluck_amplitude);
            let frames = 48_000;
            let start = Instant::now();
            let mut sink = 0.0f32;
            for _ in 0..frames {
                sink += string.step();
            }
            let elapsed = start.elapsed().as_secs_f64();
            println!(
                "{:>12} {:>10} {:>10.3}   (sink {:.4})",
                preset.name(),
                core.name(),
                elapsed * 1e6 / frames as f64,
                sink
            );
        }
    }
}