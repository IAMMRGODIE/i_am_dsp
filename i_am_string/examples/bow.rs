//! What a bowed string actually does, and how the two cores differ.
//!
//! Run with `cargo run --release --example bow`.

use i_am_string::prelude::*;

const FS: f32 = 48_000.0;

/// One pole DC blocker, the same one the engine uses.
struct DirectCurrent {
    x: f32,
    y: f32,
}

impl DirectCurrent {
    fn process(&mut self, input: f32) -> f32 {
        let output = input - self.x + 0.9985 * self.y;
        self.x = input;
        self.y = output;
        output
    }
}

fn measure(preset: Preset, note: u8, core: Core, engine: &StringEngine) {
    let spec = preset.spec(note, 1.0, engine.params());
    let mut string = StringCore::new(core, FS, &spec.string);
    let mut law = BowLaw::new(spec.bow_force, spec.bow_velocity);
    law.noise = 0.0015;
    string.add_port(spec.position, Box::new(law), spec.contact_substeps);
    for _ in 0..(FS * 0.4) as usize {
        string.step();
    }
    let period = (FS / spec.string.frequency) as usize;
    let mut blocker = DirectCurrent { x: 0.0, y: 0.0 };
    let (mut total, mut alternating, mut sticking, mut flyback) = (0.0f32, 0.0f32, 0, 0.0f32);
    let frames = (FS * 0.5) as usize;
    for _ in 0..frames {
        let sample = string.step();
        let blocked = blocker.process(sample);
        total += sample * sample;
        alternating += blocked * blocked;
        let (_, velocity) = string.port_states()[0];
        if (spec.bow_velocity - velocity).abs() < 0.012 {
            sticking += 1;
        }
        flyback = flyback.max(-velocity);
    }
    let frames = frames as f32;
    println!(
        "{:>10} ac rms {:.4} of {:.4} ({:.0}%), stick {:>3.0}%, flyback {:>6.2}, \
         {} modes",
        core.name(),
        (alternating / frames).sqrt(),
        (total / frames).sqrt(),
        100.0 * (alternating / total.max(1e-12)).sqrt(),
        100.0 * sticking as f32 / frames,
        flyback,
        string.partial_frequencies().len(),
    );
    let _ = period;
}

fn main() {
    let mut engine = StringEngine::new(FS);
    for preset in [Preset::Violin, Preset::Cello] {
        engine.set_preset(preset);
        for note in [48u8, 60, 72, 84, 96] {
            println!("\n{} note {note} ({:.0} Hz)", preset.name(),
                note_to_frequency(note as f32));
            for core in Core::ALL {
                measure(preset, note, core, &engine);
            }
        }
    }
}
