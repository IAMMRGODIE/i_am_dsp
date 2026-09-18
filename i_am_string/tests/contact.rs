//! The contact solver's contract with the string.
//!
//! A port is an implicit equation: the string's velocity at the excitation
//! point is an affine function of the force applied there, so the contact law
//! can be solved for the force that satisfies it. What that affine function has
//! to be is easy to get subtly wrong, and the failure is not a small error in a
//! corner of the model - it makes a bowed string slide along under the bow for
//! ever without ever slipping back, which is a string that makes no sound.

use i_am_string::modal::{ModalString, StringConfig as ModalConfig};
use i_am_string::prelude::*;

const FS: f32 = 48_000.0;

/// A force that is simply held on the string.
struct Constant(f32);

impl ContactLaw for Constant {
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

fn modal(frequency: f32) -> ModalString {
    ModalString::new(
        FS,
        ModalConfig {
            frequency,
            inharmonicity: 1e-4,
            sigma0: 0.6,
            sigma1: 3e-6,
            tension: 80.0,
            impedance: 1.0,
        },
    )
}

/// The same string, but heavily damped, so a transient is over in a moment.
fn damped(frequency: f32) -> ModalString {
    ModalString::new(
        FS,
        ModalConfig {
            frequency,
            inharmonicity: 1e-4,
            sigma0: 40.0,
            sigma1: 1e-4,
            tension: 80.0,
            impedance: 1.0,
        },
    )
}

/// A force held on a string pulls it into a triangle and then it stops.
///
/// This is the test that catches the port equation using the string's velocity
/// before the sample rather than after it. The static deflection is the same
/// either way, because a wrong equation still has *some* fixed point; what
/// differs is whether the string is moving when it gets there.
#[test]
fn a_held_force_settles_and_stops() {
    let mut string = damped(220.0);
    let length = string.speaking_length() as f64;
    string.add_port(0.5, Box::new(Constant(4.0)), 1);
    for _ in 0..(FS as usize / 2) {
        string.step();
    }
    let (displacement, velocity) = string.port_states()[0];
    assert!(
        velocity.abs() < 1e-3,
        "the string never settled: it is still moving at {velocity} m/s"
    );
    // A point load on a taut string pulls it into a triangle, and the height of
    // that triangle is F L beta (1 - beta) / T.
    let wanted = (4.0 * length * 0.25 / 80.0) as f32;
    assert!(
        (displacement - wanted).abs() < 0.1 * wanted,
        "the static deflection is {displacement} m, wanted about {wanted}"
    );
}

/// A bowed string has to stick and then let go, once per period.
#[test]
fn a_bowed_string_actually_slips_once_a_period() {
    for frequency in [130.8f32, 261.6, 523.3] {
        let mut string = modal(frequency);
        let mut law = BowLaw::new(0.8, 0.18);
        law.noise = 0.0015;
        string.add_port(0.9, Box::new(law), 2);
        // Let the stroke settle into whatever cycle it is going to find.
        for _ in 0..(FS * 0.5) as usize {
            string.step();
        }
        let mut sticking = 0;
        let mut flyback = 0.0f32;
        let frames = (FS * 0.25) as usize;
        for _ in 0..frames {
            string.step();
            let (_, velocity) = string.port_states()[0];
            if (0.18 - velocity).abs() < 0.012 {
                sticking += 1;
            }
            flyback = flyback.max(-velocity);
        }
        let fraction = sticking as f32 / frames as f32;
        // The flyback is the assertion that matters. A port equation that uses
        // the string's velocity before the sample instead of after it still
        // grips and still looks plausible, but it never lets go: measured, the
        // string ends up pinned at the bow speed with a bridge force that is
        // eighty six percent static.
        assert!(
            flyback > 2.0 * 0.18,
            "at {frequency} Hz the string only reached {flyback} m/s backwards, which \
             is not a flyback"
        );
        assert!(
            fraction > 0.2 && fraction < 0.9,
            "at {frequency} Hz the bow stuck for {fraction:.3} of the period, which \
             is neither gripping nor releasing"
        );
    }
}
