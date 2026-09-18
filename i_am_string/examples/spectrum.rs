//! Where the two cores put their energy.
//!
//! Run with `cargo run --release --example spectrum`.

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

/// A third of an octave of energy around `centre`, in decibels.
fn band(samples: &[f32], centre: f32) -> f32 {
    let ratio = 2f32.powf(1.0 / 6.0);
    let mut energy = 0.0;
    let steps = 5;
    for step in 0..steps {
        let t = step as f32 / (steps - 1) as f32;
        let probe = centre * ratio.powf(-1.0 + 2.0 * t);
        energy += goertzel(samples, probe);
    }
    let reference = samples.len() as f32;
    10.0 * (energy / reference / reference).max(1e-30).log10()
}

fn render(core: Core, preset: Preset, note: u8, seconds: f32) -> Vec<f32> {
    let mut engine = StringEngine::new(FS);
    engine.set_preset(preset);
    engine.params_mut().core = core;
    engine.note_on(note, 1.0);
    let mut samples: Vec<f32> = (0..(FS * seconds) as usize)
        .map(|_| engine.next_sample())
        .collect();
    let last = (samples.len() - 1) as f32;
    for (index, sample) in samples.iter_mut().enumerate() {
        let t = index as f32 / last;
        *sample *= 0.5 - 0.5 * (std::f32::consts::TAU * t).cos();
    }
    samples
}

fn main() {
    let centres: Vec<f32> = (0..26)
        .map(|index| 62.5 * 2f32.powf(index as f32 / 3.0))
        .collect();
    for preset in [Preset::Piano, Preset::Violin, Preset::Guitar] {
        for note in [72u8, 96] {
            let waveguide = render(Core::Waveguide, preset, note, 1.0);
            let modal = render(Core::Modal, preset, note, 1.0);
            // Normalise each to the same total energy, so what is compared is
            // the shape of the spectrum rather than the level.
            let norm = |samples: &Vec<f32>| {
                (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32)
                    .sqrt()
                    .max(1e-30)
            };
            let (wr, mr) = (norm(&waveguide), norm(&modal));
            println!(
                "\n{} note {note}: rms {wr:.5} vs {mr:.5} (modal {:+.1} dB)",
                preset.name(),
                20.0 * (mr / wr).log10()
            );
            println!("{:>9} {:>10} {:>10} {:>10}", "Hz", "waveguide", "modal", "modal-wave");
            let mut wave_centroid = 0.0;
            let mut modal_centroid = 0.0;
            let mut wave_total = 0.0;
            let mut modal_total = 0.0;
            for centre in &centres {
                let w = band(&waveguide, *centre) - 20.0 * wr.log10();
                let m = band(&modal, *centre) - 20.0 * mr.log10();
                let wave_linear = 10f32.powf(w / 10.0);
                let modal_linear = 10f32.powf(m / 10.0);
                wave_centroid += centre * wave_linear;
                modal_centroid += centre * modal_linear;
                wave_total += wave_linear;
                modal_total += modal_linear;
                println!("{centre:>9.0} {w:>10.1} {m:>10.1} {:>10.1}", m - w);
            }
            println!(
                "centroid: waveguide {:.0} Hz, modal {:.0} Hz",
                wave_centroid / wave_total,
                modal_centroid / modal_total
            );
        }
    }
}
