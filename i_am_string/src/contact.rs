//! Contact laws: what the player does to the string at the excitation point.
//!
//! All three excitations reduce to an external force `F` applied at a point,
//! which the port solver turns into a string velocity. Only the constitutive
//! law differs, and that is the whole of the [ContactLaw] trait.
//!
//! All quantities are SI: metres, newtons, seconds.

use crate::Rng;

/// The constitutive law of an excitation port.
///
/// Implementations hold the state of whatever is touching the string (a
/// finger, a hammer, a bow) and answer a single question: given the string's
/// displacement and velocity at the contact point, what force is the player
/// applying right now?
pub trait ContactLaw: Send + Sync {
    /// The force applied to the string, positive along the string's +y axis.
    fn force(&self, string_y: f32, string_v: f32) -> f32;

    /// The derivative of [ContactLaw::force] with respect to the string
    /// velocity. Used to speed up the port solver; it may be approximate.
    fn force_dv(&self, string_y: f32, string_v: f32) -> f32;

    /// A safe upper bound on the force for the current state.
    ///
    /// The port solver brackets the root of the implicit contact equation with
    /// this, so it must never underestimate.
    fn max_force(&self) -> f32;

    /// A safe lower bound on the force for the current state.
    ///
    /// A hammer and a finger can only push, so this is zero. Friction is
    /// different: it opposes the slide, so it reverses sign, and a solver that
    /// assumes otherwise simply clips the whole slip phase and the bow produces
    /// hash instead of a note.
    fn min_force(&self) -> f32 {
        0.0
    }

    /// Advance the law's own state by `dt` seconds.
    fn advance(&mut self, string_y: f32, string_v: f32, dt: f32);

    /// Let go of the string over the next `dt` seconds.
    ///
    /// A hammer leaves on its own; a bow has to be lifted. Implementations
    /// that do not need it keep the default no-op.
    fn release(&mut self, dt: f32) {
        let _ = dt;
    }

    /// Whether the contact can still affect the string.
    fn is_active(&self) -> bool {
        true
    }
}

/// A finger or plectrum modelled as a unilateral penalty constraint.
///
/// The finger holds the string at `finger_y`; the force is a stiff spring plus
/// heavy damping, which is the standard penalty formulation of the kinematic
/// constraint `y(x_p) = y_finger`. Releasing the string is modelled by moving
/// the finger away, after which the force falls to zero on its own.
#[derive(Clone, Debug)]
pub struct PluckLaw {
    /// The displacement of the finger, in metres.
    pub finger_y: f32,
    /// The velocity of the finger, in metres per second.
    pub finger_v: f32,
    /// Penalty stiffness, in newtons per metre.
    pub stiffness: f32,
    /// Contact damping, in newton seconds per metre.
    pub damping: f32,
}

impl PluckLaw {
    /// A constraint stiff enough to be treated as rigid over an audio sample.
    pub fn new(finger_y: f32, impedance: f32, sample_rate: f32) -> Self {
        // The penalty spring has to dominate the string impedance over one
        // sample, otherwise the constraint leaks and the pluck sounds mushy.
        let stiffness = impedance * sample_rate * 40.0;
        Self {
            finger_y,
            finger_v: 0.0,
            stiffness,
            damping: impedance * 2.0,
        }
    }

    /// Move the finger away from the string, releasing it.
    pub fn release(&mut self, velocity: f32) {
        self.finger_v = velocity;
    }
}

impl ContactLaw for PluckLaw {
    fn force(&self, string_y: f32, string_v: f32) -> f32 {
        let compression = (self.finger_y - string_y).max(0.0);
        let approach = (self.finger_v - string_v).max(0.0);
        self.stiffness * compression + self.damping * approach
    }

    fn force_dv(&self, _string_y: f32, string_v: f32) -> f32 {
        if string_v < self.finger_v { -self.damping } else { 0.0 }
    }

    fn max_force(&self) -> f32 {
        self.stiffness * self.finger_y.abs() + self.damping * self.finger_v.abs() + 1.0
    }

    fn advance(&mut self, _string_y: f32, _string_v: f32, _dt: f32) {}
}

/// A piano hammer: a mass with a non-linear felt spring.
///
/// `F = K * delta^p * (1 + 1.5 * lambda * d(delta)/dt)`, the Hunt-Crossley
/// form of the felt law, where `delta` is the felt compression. The exponent
/// controls how the timbre hardens with dynamics, and the hysteresis term
/// removes the energy that real felt dissipates.
#[derive(Clone, Debug)]
pub struct HammerLaw {
    /// Hammer mass, in kilograms.
    pub mass: f32,
    /// Felt stiffness.
    pub stiffness: f32,
    /// Felt exponent, typically between 2.2 and 3.5.
    pub exponent: f32,
    /// Hysteresis factor of the Hunt-Crossley model.
    pub hysteresis: f32,
    /// Hammer displacement, in metres.
    pub y: f32,
    /// Hammer velocity, in metres per second.
    pub v: f32,
    /// Set once the hammer has bounced off and is no longer in contact.
    pub separated: bool,
}

impl HammerLaw {
    /// A hammer released towards the string at `velocity` metres per second.
    pub fn new(mass: f32, velocity: f32, exponent: f32, stiffness: f32, string_y: f32) -> Self {
        Self {
            mass,
            stiffness,
            exponent,
            hysteresis: 0.05,
            y: string_y,
            v: velocity,
            separated: false,
        }
    }

    /// The felt compression, in metres.
    pub fn compression(&self, string_y: f32) -> f32 {
        (self.y - string_y).max(0.0)
    }
}

impl ContactLaw for HammerLaw {
    fn force(&self, string_y: f32, string_v: f32) -> f32 {
        let delta = self.y - string_y;
        if delta <= 0.0 {
            return 0.0;
        }
        let rate = self.v - string_v;
        let force = self.stiffness
            * delta.powf(self.exponent)
            * (1.0 + 1.5 * self.hysteresis * rate);
        force.max(0.0)
    }

    fn force_dv(&self, string_y: f32, string_v: f32) -> f32 {
        let delta = self.y - string_y;
        if delta <= 0.0 {
            return 0.0;
        }
        let rate = self.v - string_v;
        let value = delta.powf(self.exponent);
        let slope = self.exponent * delta.powf(self.exponent - 1.0);
        let factor = 1.0 + 1.5 * self.hysteresis * rate;
        if self.stiffness * value * factor <= 0.0 {
            // The Hunt-Crossley factor has gone negative and the force has been
            // clamped to zero, so its slope is zero too. Reporting the unclamped
            // slope here sends the solver marching off to infinity, because
            // above the clamp the force is constant while the derivative says it
            // is falling.
            return 0.0;
        }
        -self.stiffness * (slope * factor + value * 1.5 * self.hysteresis)
    }

    fn max_force(&self) -> f32 {
        let reach = self.y.abs() + (self.v.abs() * 0.01) + 1e-3;
        self.stiffness * reach.powf(self.exponent) * (1.0 + 3.0 * self.hysteresis) + 1.0
    }

    fn is_active(&self) -> bool {
        !self.separated
    }

    fn advance(&mut self, string_y: f32, string_v: f32, dt: f32) {
        let delta = self.y - string_y;
        let force = self.force(string_y, string_v);
        if force > 0.0 {
            self.separated = false;
        } else if delta <= 0.0 && self.v <= string_v {
            // The felt has come off the string and the hammer is leaving. It is
            // done: integrating a free hammer any further would let its position
            // run away and the felt law would follow it.
            self.separated = true;
        }
        if self.separated {
            return;
        }
        self.v -= force / self.mass * dt;
        self.y += self.v * dt;
        if self.y < string_y && self.v < string_v {
            self.separated = true;
        }
    }
}

/// A rosin friction curve: Stribeck plus Coulomb plus viscous.
///
/// The negative slope around zero relative velocity is what makes a bowed
/// string sing. Linearised, it subtracts from the port impedance and turns the
/// string into a negative resistance oscillator.
#[derive(Clone, Copy, Debug)]
pub struct FrictionCurve {
    /// The friction coefficient at zero relative velocity.
    pub mu_static: f32,
    /// The limiting (kinetic) friction coefficient.
    pub mu_dynamic: f32,
    /// The relative velocity at which the static term has decayed by 1/e.
    pub stribeck_velocity: f32,
    /// The viscous (lubricated) slope at high relative velocity.
    pub viscous: f32,
    /// The relative velocity over which Coulomb friction changes sign.
    ///
    /// This is the single most delicate number in the whole model. It has to be
    /// wider than the velocity the string gains in one step, or the sign of the
    /// friction flips back and forth every sample and the bow produces broadband
    /// hash instead of a tone.
    pub stiction_band: f32,
}

impl Default for FrictionCurve {
    fn default() -> Self {
        Self {
            mu_static: 0.60,
            mu_dynamic: 0.20,
            stribeck_velocity: 0.03,
            viscous: 0.05,
            stiction_band: 0.012,
        }
    }
}

impl FrictionCurve {
    /// The *signed* friction coefficient at a relative velocity `rel`.
    ///
    /// Friction opposes the slide, so this is an odd function of `rel`. Getting
    /// the sign wrong is not a subtle error: a friction force that never
    /// reverses pumps energy into the string on every sample and the model runs
    /// away to infinity.
    #[inline]
    pub fn mu(&self, rel: f32) -> f32 {
        let speed = rel.abs();
        let coulomb = self.mu_dynamic
            + (self.mu_static - self.mu_dynamic) * (-speed / self.stribeck_velocity).exp();
        coulomb * (rel / self.stiction_band.max(1e-5)).tanh() + self.viscous * rel
    }

    /// The slope of the signed friction curve at `rel`.
    ///
    /// This is the exact derivative of [FrictionCurve::mu], including the
    /// stiction band. That band contributes a very large positive slope right at
    /// zero, which is exactly the sticking equilibrium a real bow has, and
    /// leaving it out makes the solver's Newton step wrong by two orders of
    /// magnitude just where it matters most.
    #[inline]
    pub fn slope(&self, rel: f32) -> f32 {
        let speed = rel.abs();
        let band = self.stiction_band.max(1e-5);
        let decay = (-speed / self.stribeck_velocity).exp();
        let coulomb = self.mu_dynamic + (self.mu_static - self.mu_dynamic) * decay;
        let tanh = (speed / band).tanh();
        let stribeck = -(self.mu_static - self.mu_dynamic) / self.stribeck_velocity * decay;
        stribeck * tanh + coulomb * (1.0 - tanh * tanh) / band + self.viscous
    }

    /// The largest coefficient the curve can reach for a sane relative speed.
    pub fn max_mu(&self) -> f32 {
        self.mu_static + self.viscous * 20.0
    }
}

/// A bow: a normal force and a sliding speed acting through a friction curve.
#[derive(Clone, Debug)]
pub struct BowLaw {
    /// The bow force pressing the hair onto the string, in newtons.
    pub normal_force: f32,
    /// The bow speed, in metres per second.
    pub bow_velocity: f32,
    /// The friction curve of the rosined hair.
    pub curve: FrictionCurve,
    /// Amplitude of the surface noise added to the bow speed.
    pub noise: f32,
    /// A multiplier on the bow force while the stroke settles.
    ///
    /// A real bow bites: the hair digs into the string as the stroke starts and
    /// the force settles back over the first few tens of milliseconds. It is a
    /// large part of what makes a bowed note sound started rather than switched
    /// on.
    pub onset: f32,
    /// How fast the bite relaxes, in 1/s.
    pub onset_rate: f32,
    noise_state: f32,
    rng: Rng,
}

impl BowLaw {
    /// A bow pressing with `normal_force` and moving at `bow_velocity`.
    pub fn new(normal_force: f32, bow_velocity: f32) -> Self {
        Self {
            normal_force,
            bow_velocity,
            curve: FrictionCurve::default(),
            noise: 0.0,
            onset: 1.8,
            onset_rate: 26.0,
            noise_state: 0.0,
            rng: Rng::new(0x51ED_2701),
        }
    }

    /// The relative velocity between the hair and the string.
    #[inline]
    pub fn relative(&self, string_v: f32) -> f32 {
        self.bow_velocity + self.noise_state - string_v
    }
}

impl ContactLaw for BowLaw {
    fn force(&self, _string_y: f32, string_v: f32) -> f32 {
        self.normal_force * self.onset * self.curve.mu(self.relative(string_v))
    }

    fn force_dv(&self, _string_y: f32, string_v: f32) -> f32 {
        -self.normal_force * self.curve.slope(self.relative(string_v))
    }

    fn max_force(&self) -> f32 {
        self.normal_force * self.curve.max_mu() + 1.0
    }

    fn min_force(&self) -> f32 {
        -(self.normal_force * self.curve.max_mu() + 1.0)
    }

    fn release(&mut self, dt: f32) {
        // A real bow is lifted over tens of milliseconds, and an instant drop
        // of the friction force clicks.
        self.normal_force *= (1.0 - 14.0 * dt).clamp(0.0, 1.0);
    }

    fn is_active(&self) -> bool {
        self.normal_force > 1e-4
    }

    fn advance(&mut self, _string_y: f32, _string_v: f32, dt: f32) {
        self.onset += (1.0 - self.onset) * (dt * self.onset_rate).min(1.0);
        if self.noise > 0.0 {
            // A slow random walk keeps the rosin noise correlated, which sounds
            // like surface noise rather than white hiss.
            let target = self.rng.bipolar() * self.noise;
            self.noise_state += (target - self.noise_state) * (dt * 400.0).min(1.0);
        }
    }
}
