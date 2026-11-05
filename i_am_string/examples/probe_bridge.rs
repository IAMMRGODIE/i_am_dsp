//! Prints the damping each partial gets from the bridge admittance.

use i_am_string::prelude::*;

const MIN: f64 = 0.45;
const MAX: f64 = 2.5;

fn weight(frequency: f32) -> f64 {
	let t = ((frequency - 150.0) / 300.0).clamp(0.0, 1.0);
	(t * t * (3.0 - 2.0 * t)) as f64
}

fn main() {
	for note in [28u8, 36, 48, 52, 60, 72, 84] {
		let f0 = 440.0f32 * 2.0f32.powf((note as f32 - 69.0) / 12.0);
		let b = 7e-4 * (f0 / 440.0).powf(1.4) + 1e-4 * (55.0 / f0).clamp(0.0, 2.2);
		println!("note {note} f0 {f0:.1}");
		let mut line = String::new();
		for k in 1..=12 {
			let frequency = k as f32 * f0 * (1.0 + b * k as f32 * k as f32).sqrt();
			let ratio = damping_ratio(frequency) as f64;
			let damping =
				(1.0 + weight(frequency) * (ratio - 1.0)).clamp(MIN, MAX);
			line.push_str(&format!("{k}:{ratio:.2}/{damping:.2} "));
		}
		println!("  {line}");
	}
}