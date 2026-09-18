//! One way to model a string: one damped oscillator per partial.
//!
//! # Why this exists alongside the waveguide
//!
//! The obvious way to model a string is two delay lines with a dispersion filter
//! in the loop, and that is what [crate::string] does. It has a hard limit this
//! string does not: the dispersion filter has to realise the stiff string's loop
//! delay, which falls with frequency in a way a cascade of first order allpass
//! sections cannot follow once the stretch gets large. The failure is not
//! graceful - measured on the rendered output of C7 on a grand piano, the
//! waveguide's partials come out up to forty five cents from where the physics
//! puts them - and a string of mistuned partials does not sound stiff, it sounds
//! like a detuned saw.
//!
//! Here the partials *are* the model:
//!
//! ```text
//! f_n = n f_0 sqrt(1 + B n^2)
//! ```
//!
//! so the tuning is exact by construction - the same note measured the same way
//! comes out within a fifth of a cent - the inharmonicity control means what it
//! says, and there is no filter to design, no fit to cache and no cold start.
//! What it costs is roughly eight times the arithmetic per string, because it is
//! a hundred and sixty real oscillators where the waveguide is a filter standing
//! in for them.
//!
//! # How a contact still works
//!
//! Each mode obeys
//!
//! ```text
//! q'' + 2 sigma q' + omega^2 q = phi_n(x_k) F(t) / M_n
//! ```
//!
//! and the displacement and velocity at an excitation point are the mode sums
//! `y = sum phi_n(x_k) q_n` and `v = sum phi_n(x_k) q_n'`. Discretising a mode
//! exactly over one sample gives a linear map plus a term proportional to the
//! force *at the current sample*, so summing over modes leaves
//!
//! ```text
//! v = G F + v_free
//! ```
//!
//! with a single scalar `G` for the whole string. The port is therefore a pure
//! conductance, exactly as the waveguide's was a pure resistance, and the same
//! bracketed Newton solve handles the contact. The rest of the engine does not
//! know which of the two it is talking to.

use crate::contact::ContactLaw;

/// The physical parameters of one string.
#[derive(Clone, Debug)]
pub struct StringConfig {
    /// The fundamental frequency in hertz.
    pub frequency: f32,
    /// The inharmonicity coefficient `B` of `f_n = n f_0 sqrt(1 + B n^2)`.
    pub inharmonicity: f32,
    /// The frequency independent part of the decay rate, in 1/s.
    pub sigma0: f32,
    /// The frequency dependent part of the decay rate, in seconds.
    pub sigma1: f32,
    /// The string tension, in newtons.
    pub tension: f32,
    /// The string's wave impedance `Z0 = sqrt(T mu)`, in N s/m.
    pub impedance: f32,
}

impl Default for StringConfig {
    fn default() -> Self {
        Self {
            frequency: 220.0,
            inharmonicity: 1e-4,
            sigma0: 0.6,
            sigma1: 3e-6,
            tension: 80.0,
            impedance: 1.0,
        }
    }
}

/// The largest number of modes one string may use.
///
/// A mode is a damped oscillator, and the ear reads a string's stiffness from
/// how far its *high* partials have been pushed sharp, so the partials near the
/// top of the band are not decoration. What sets the ceiling is cost: measured, a
/// mode costs a little under a nanosecond a sample, eight voices of a three
/// string piano is twenty four strings, and the arithmetic runs out long before
/// the physics does. At 27.5 Hz a hundred and sixty modes reach 5 kHz, and one
/// note further up the range every string is stopped by Nyquist instead; the
/// partials this gives up are the ones the loss law has already buried.
const MAX_MODES: usize = 160;

/// How far the damper must move before the coefficients are worth rebuilding.
///
/// The engine moves the damper every sample and it approaches its target
/// exponentially, so rebuilding on any change at all would put an `exp` and a
/// `sin_cos` per mode on the audio thread for every sample. Half a percent is
/// well below the resolution of the decay it controls.
const DAMPER_RESOLUTION: f32 = 0.005;

/// The fewest samples between two rebuilds of the mode coefficients.
///
/// The damper moves by four parts in a thousand of what is left of its journey
/// on every sample, so it covers a full travel in a few hundred samples and a
/// resolution gate alone would let it rebuild thirty times over that. Capping
/// the rate turns that into eight rebuilds of a hundred and sixty modes, which
/// is a fraction of a millisecond, and the staircase it leaves is far finer
/// than the ear can follow on a decay.
const REBUILD_INTERVAL: u32 = 32;

/// How far the pitch has to move before the coefficients are worth rebuilding.
///
/// A pitch is a multiplication on every mode's frequency, so unlike a damper it
/// cannot be applied to the state without recomputing the transition. One part
/// in a thousand is under two cents, and with the rebuild rate capped below it
/// leaves a staircase at well over a kilohertz, which is far above the five or
/// six hertz a hand shakes at.
const PITCH_RESOLUTION: f32 = 1e-3;

/// The force a mode has to be able to put on the bridge to be worth updating.
///
/// A mode that has decayed below this is skipped entirely. That is not only a
/// saving of twelve floating point operations: a geometric sequence that is left
/// running reaches the denormal range, where every one of those operations drops
/// out of the pipeline into microcode and costs a hundred times as much. A
/// measured mode bank with a live string costs a fifth of a microsecond per
/// sample per string, and the same bank left ringing for a second costs seven,
/// which is the difference between a playable instrument and a stutter.
///
/// A nanonewton is a hundred and twenty decibels below a note. Nothing that
/// sleeps here has ever been audible.
const QUIET_FORCE: f32 = 1e-9;

impl StringConfig {
    /// A convenient constructor for a plain, ideal string.
    pub fn new(frequency: f32) -> Self {
        Self {
            frequency,
            ..Default::default()
        }
    }
}

/// One mode of the string.
///
/// The update is the exact solution of the damped oscillator over one sample
/// with the force held constant, which is what keeps the decay accurate when the
/// decay rate is six orders of magnitude below the sample rate. Writing it as a
/// direct form recurrence instead would lose the decay entirely in `f32`.
///
/// The modes live in one array of these rather than in a set of parallel arrays,
/// which is not the layout the shape of the arithmetic suggests. It was tried
/// the other way and measured: the bank is the hottest loop in the crate, eight
/// voices of a three string piano being six thousand damped oscillators updated
/// forty eight thousand times a second, and splitting it into thirteen separate
/// allocations made it twice as slow. The loop is a straight line over one
/// buffer here, and the buffer is small enough to stay in cache; thirteen
/// pointers competing for registers cost more than the stride ever did.
#[derive(Clone, Copy, Debug, Default)]
struct Mode {
    /// The modal coordinate and its rate of change.
    q: f32,
    v: f32,
    /// The state transition, already scaled by `exp(-sigma T)`.
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    /// The contribution of a force held over one sample.
    e: f32,
    f: f32,
    /// What this mode contributes to the bridge force.
    bridge: f32,
    /// The decay rate before the damper is taken into account, in 1/s.
    sigma: f32,
    /// The frequency, in hertz.
    frequency: f32,
    /// The displacement below which this mode is inaudible at the bridge.
    quiet_q: f32,
    /// The matching velocity.
    quiet_v: f32,
    /// Whether the mode is still loud enough to be worth updating.
    awake: bool,
}

/// One excitation point on the string.
struct Port {
    law: Box<dyn ContactLaw>,
    /// The mode shape at this point, one entry per mode.
    weights: Vec<f32>,
    /// The instantaneous velocity per unit force at this point, in m/(s N).
    conductance: f32,
    y: f32,
    v: f32,
    substeps: u32,
    active: bool,
}

impl Port {
    /// The residual of the implicit contact equation.
    #[inline]
    fn residual(&self, v: f32, free: f32, conductance: f32, y: f32) -> f32 {
        v - conductance * self.law.force(y, v) - free
    }

    /// Solve for the string velocity at this port over one sample.
    fn solve(&mut self, free: f32, conductance: f32, dt: f32) -> f32 {
        let substeps = self.substeps.max(1);
        let sub_dt = dt / substeps as f32;
        let mut y = self.y;
        let mut v = self.v;
        let mut force = 0.0;
        for _ in 0..substeps {
            let solved = self.solve_once(free, conductance, y, v);
            v = solved.0;
            force = solved.1;
            self.law.advance(y, v, sub_dt);
            y += v * sub_dt;
        }
        self.y = y;
        self.v = v;
        force
    }

    /// A bracketed Newton solve.
    ///
    /// The bracket is exact. With no contact force the velocity is `free`, and
    /// with the largest possible force it is `G F_max` above that. Bisection
    /// therefore always converges and Newton only accelerates it. Friction
    /// makes the equation non-monotone - it has a negative differential
    /// resistance over a wide band - so the solve is warm started and step
    /// limited to follow the branch the string is already on, with the bracket
    /// as the guaranteed fallback.
    fn solve_once(&self, free: f32, conductance: f32, y: f32, warm: f32) -> (f32, f32) {
        let lo = free + conductance * self.law.min_force();
        let mut hi = free + conductance * self.law.max_force();
        if !hi.is_finite() || hi <= lo {
            hi = lo + 1e-9;
        }
        let mut guard = 0;
        while self.residual(hi, free, conductance, y) < 0.0 && guard < 8 {
            hi = lo + (hi - lo) * 4.0;
            guard += 1;
        }

        let limit = (hi - lo) * 0.05;
        // A velocity tolerance, and a loose one on purpose. The contact law is
        // solved to a nanometre per second, which is six orders below anything a
        // string does and four below the waveguide's own solver; asking for more
        // does not make the answer better, it makes the solver fall through to
        // the bracketed phase, which costs sixty four iterations of a felt law.
        let tolerance = 1e-9 * (1.0 + free.abs());

        let mut v = warm.clamp(lo, hi);
        let mut best = v;
        let mut best_residual = f32::INFINITY;
        for _ in 0..24 {
            let residual = self.residual(v, free, conductance, y);
            if residual.abs() <= tolerance {
                return (v, self.law.force(y, v));
            }
            if residual.abs() < best_residual {
                best_residual = residual.abs();
                best = v;
            }
            let slope = 1.0 - conductance * self.law.force_dv(y, v);
            let step = if slope.abs() > 1e-12 {
                (residual / slope).clamp(-limit, limit)
            } else {
                residual.signum() * limit
            };
            let next = (v - step).clamp(lo, hi);
            if (next - v).abs() <= f32::EPSILON * (1.0 + next.abs()) {
                break;
            }
            v = next;
        }

        // Guaranteed convergence, from a bracket grown outwards from the best
        // iterate rather than across the whole reachable force range.
        if best_residual > tolerance {
            let centre = self.residual(best, free, conductance, y);
            let mut bracket = None;
            let mut width = limit.max(1e-9);
            for _ in 0..48 {
                let upper = (best + width).min(hi);
                if self.residual(upper, free, conductance, y) * centre <= 0.0 {
                    bracket = Some((best, upper, centre));
                    break;
                }
                let lower = (best - width).max(lo);
                let value = self.residual(lower, free, conductance, y);
                if value * centre <= 0.0 {
                    bracket = Some((lower, best, value));
                    break;
                }
                if upper >= hi && lower <= lo {
                    break;
                }
                width *= 1.7;
            }
            if let Some((mut lower, mut upper, mut lower_value)) = bracket {
                for _ in 0..64 {
                    let mid = 0.5 * (lower + upper);
                    // The width has to be reachable in `f32`, whose epsilon is
                    // around 1.2e-7. A target tighter than that is never met, so
                    // the loop always runs all sixty four iterations and each one
                    // evaluates the contact law twice.
                    if upper - lower <= 1e-7 * (1.0 + mid.abs()) {
                        break;
                    }
                    let value = self.residual(mid, free, conductance, y);
                    if value * lower_value <= 0.0 {
                        upper = mid;
                    } else {
                        lower = mid;
                        lower_value = value;
                    }
                }
                let solved = 0.5 * (lower + upper);
                return (solved, self.law.force(y, solved));
            }
        }
        (best, self.law.force(y, best))
    }
}

/// A single physical string, as a bank of modes with any number of ports.
pub struct ModalString {
    modes: Vec<Mode>,
    ports: Vec<Port>,
    dt: f32,
    /// The mass the mode shapes are normalised against, in kilograms.
    modal_mass: f64,
    /// The speaking length, in metres.
    length: f32,
    envelope: f32,
    damper: f32,
    /// The damper the coefficients were last built for.
    damper_applied: f32,
    /// Scratch space: the force seen by each mode, in mode shape units.
    shaped_force: Vec<f32>,
    /// Samples since the coefficients were last rebuilt, to cap the rebuild rate.
    since_rebuild: u32,
    /// How far the pitch is currently shifted, as a frequency multiplier.
    pitch_ratio: f32,
    /// The pitch the coefficients were last built for.
    pitch_applied: f32,
    config: StringConfig,
}

impl ModalString {
    /// Build a string from its physical description.
    pub fn new(sample_rate: f32, config: StringConfig) -> Self {
        let sample_rate = if sample_rate > 0.0 { sample_rate } else { 48_000.0 };
        let fs = sample_rate as f64;
        let nyquist = fs * 0.5 * 0.94;
        let f0 = (config.frequency as f64).clamp(4.0, nyquist * 0.5);
        let b = config.inharmonicity.max(0.0) as f64;

        // The string's material constants follow from the tension, the wave
        // impedance and the fundamental: mu = Z0^2 / T, c = T / Z0, L = c/(2f0).
        let tension = config.tension.max(1.0) as f64;
        let impedance = config.impedance.max(1e-4) as f64;
        let mu = impedance * impedance / tension;
        let length = tension / impedance / (2.0 * f0);
        let modal_mass = mu * length * 0.5;


        let mut modes = Vec::new();
        for index in 1..=MAX_MODES {
            let n = index as f64;
            let frequency = n * f0 * (1.0 + b * n * n).sqrt();
            if frequency >= nyquist {
                break;
            }
            let sigma = config.sigma0 as f64
                + config.sigma1 as f64 * frequency * frequency;
            let wave_number = n * std::f64::consts::PI / length;
            // `T dy/dx` at the bridge is the force the string pulls the bridge
            // with, and sin(n pi x / L) has slope n pi / L times cos(n pi). The
            // further halving is the convention the waveguide uses and the
            // engine is written against: what a string hands over is the force
            // wave *arriving* at the bridge, which the rigid termination then
            // doubles. Both cores return the same quantity so they can be
            // compared without a level jump, and the reference force in
            // [crate::instrument] is scaled to match.
            let bridge =
                0.5 * tension * wave_number * if index % 2 == 0 { -1.0 } else { 1.0 };
            // A mode falls silent when the force it puts on the bridge drops
            // under the floor. As a displacement that is the floor divided by
            // the mode's own coupling, which is why a high mode can be pinned
            // much sooner than a low one.
            let quiet_q = QUIET_FORCE / bridge.abs().max(1e-30) as f32;
            modes.push(Mode {
                quiet_q,
                quiet_v: quiet_q * std::f32::consts::TAU * frequency as f32,
                sigma: sigma as f32,
                frequency: frequency as f32,
                bridge: bridge as f32,
                awake: true,
                ..Default::default()
            });
        }
        if modes.is_empty() {
            let bridge = 0.5 * tension;
            modes.push(Mode {
                sigma: config.sigma0,
                frequency: f0 as f32,
                bridge: bridge as f32,
                quiet_q: QUIET_FORCE / bridge.max(1e-30) as f32,
                quiet_v: QUIET_FORCE,
                awake: true,
                ..Default::default()
            });
        }

        let mut string = Self {
            modes,
            ports: Vec::new(),
            dt: 1.0 / sample_rate,
            modal_mass,
            length: length as f32,
            envelope: 0.0,
            damper: 1.0,
            damper_applied: f32::NAN,
            shaped_force: Vec::new(),
            since_rebuild: 0,
            pitch_ratio: 1.0,
            pitch_applied: f32::NAN,
            config,
        };
        string.rebuild_modes();
        string
    }

    /// The configuration this string was built from.
    pub fn config(&self) -> &StringConfig {
        &self.config
    }

    /// The frequencies the string resonates at.
    ///
    /// For a modal string these are exact rather than fitted, which is the whole
    /// point of the architecture.
    pub fn partial_frequencies(&self) -> Vec<f32> {
        self.modes.iter().map(|mode| mode.frequency).collect()
    }

    /// The same list: the model resonates where it says it does.
    pub fn resonance_frequencies(&self) -> Vec<f32> {
        self.partial_frequencies()
    }

    /// Remove every excitation port and silence the string.
    pub fn reset(&mut self) {
        for mode in &mut self.modes {
            mode.q = 0.0;
            mode.v = 0.0;
            mode.awake = false;
        }
        self.ports.clear();
        self.envelope = 0.0;
    }

    /// Add an excitation port at `position`, measured from the nut.
    pub fn add_port(&mut self, position: f32, law: Box<dyn ContactLaw>, substeps: u32) {
        let beta = position.clamp(0.004, 0.996) as f64;
        let weights: Vec<f32> = (1..=self.modes.len())
            .map(|index| (index as f64 * std::f64::consts::PI * beta).sin() as f32)
            .collect();
        let conductance = weights
            .iter()
            .zip(self.modes.iter())
            .map(|(weight, mode)| weight * weight * mode.f)
            .sum::<f32>()
            .max(1e-12);
        // A string that is about to be driven has to be listening, even if it
        // has been silent for long enough that every mode had gone to sleep.
        for mode in &mut self.modes {
            mode.awake = true;
        }
        self.ports.push(Port {
            law,
            weights,
            conductance,
            y: 0.0,
            v: 0.0,
            substeps: substeps.max(1),
            active: true,
        });
    }

    /// Release the string from a triangular displacement.
    ///
    /// `amplitude` is the peak displacement as a fraction of the speaking
    /// length. The mode amplitudes of a triangle are known in closed form.
    pub fn pluck(&mut self, position: f32, amplitude: f32) {
        let beta = position.clamp(0.01, 0.5) as f64;
        // The amplitude is a fraction of the speaking length and the mode
        // coordinates are displacements, so it has to become a length before the
        // mode sum means anything. Leaving it as a fraction and reading metres
        // out of the sum is a scale error of the whole string length, which on a
        // scaled string is more than a factor of ten.
        let height = amplitude as f64 * self.length as f64;
        self.ports.clear();
        let pi = std::f64::consts::PI;
        for (index, mode) in self.modes.iter_mut().enumerate() {
            let n = (index + 1) as f64;
            let shape = (n * pi * beta).sin();
            mode.q = (2.0 * height / (n * n * pi * pi * beta * (1.0 - beta)) * shape) as f32;
            mode.v = 0.0;
            // A plucked mode is awake whatever its amplitude, because the pluck
            // is what put it there.
            mode.awake = true;
        }
        self.envelope = 0.0;
    }

    /// Tell every port to let go of the string.
    pub fn release_ports(&mut self, dt: f32) {
        for port in &mut self.ports {
            port.law.release(dt);
        }
    }

    /// The current displacement and velocity at each excitation port.
    pub fn port_states(&self) -> Vec<(f32, f32)> {
        self.ports.iter().map(|port| (port.y, port.v)).collect()
    }

    /// Advance the string by one sample and return the force at the bridge.
    pub fn step(&mut self) -> f32 {
        self.since_rebuild += 1;
        if self.since_rebuild >= REBUILD_INTERVAL && self.coefficients_stale() {
            self.rebuild_modes();
        }

        // Pass one: what the string would be doing at each contact point at the
        // end of this sample if no force were applied there.
        //
        // It has to be the *evolved* velocity, `c q + d v`, and not the velocity
        // the string has now. Solving `v = sum(w v) + G F` instead looks
        // equivalent and is not: with a constant force on the string it has no
        // solution at rest. The true static state has every mode at
        // `q = w F / (M omega^2)`, where the free evolution contributes
        // `-t sum(w omega^2 q) = -t sum(w^2 F / M) = -G F` and cancels the
        // force term exactly. Drop that term and the solver has to invent a
        // steady velocity to balance the force, which is a string that slides
        // along under the bow for ever without a flyback and makes no sound -
        // measured, a bowed violin stuck for ninety four percent of the period
        // with a bridge force eighty six percent static.
        let mut forces = Vec::with_capacity(self.ports.len());
        for port in &self.ports {
            let mut free = 0.0;
            for (mode, weight) in self.modes.iter().zip(port.weights.iter()) {
                free += weight * (mode.c * mode.q + mode.d * mode.v);
            }
            forces.push((free, port.conductance));
        }

        // Pass two: solve each contact. Ports share the modes but not the
        // solver, exactly as they shared the delay lines before.
        let dt = self.dt;
        self.shaped_force.clear();
        self.shaped_force.resize(self.modes.len(), 0.0);
        for (port, (free, conductance)) in self.ports.iter_mut().zip(forces) {
            if !port.active {
                continue;
            }
            let force = port.solve(free, conductance, dt);
            for (shaped, weight) in self.shaped_force.iter_mut().zip(port.weights.iter()) {
                *shaped += weight * force;
            }
            if !port.law.is_active() {
                port.active = false;
            }
        }

        // Pass three: advance every mode with the force it saw, and read the
        // bridge force off the mode displacements.
        let mut bridge = 0.0;
        for (mode, shaped) in self.modes.iter_mut().zip(self.shaped_force.iter()) {
            // A sleeping mode still has to wake if something is pushing on it.
            // With no force and nothing left to say, it is skipped whole, which
            // is both faster than computing it and the reason it never reaches
            // the denormal range.
            if !mode.awake && *shaped == 0.0 {
                continue;
            }
            let q = mode.q;
            let v = mode.v;
            mode.q = mode.a * q + mode.b * v + mode.e * shaped;
            mode.v = mode.c * q + mode.d * v + mode.f * shaped;
            bridge += mode.bridge * mode.q;
            mode.awake = mode.q.abs() > mode.quiet_q || mode.v.abs() > mode.quiet_v;
        }
        self.envelope = (self.envelope * 0.9997).max(bridge.abs());
        bridge
    }

    /// Set the extra damping of a damper felt.
    ///
    /// `1.0` is an open string and `0.5` is felt resting on the string, matching
    /// the meaning it had when the string was a waveguide.
    pub fn set_damper(&mut self, gain: f32) {
        self.damper = gain.clamp(0.0, 1.0);
    }

    /// Whether anything that needs new coefficients has moved enough.
    #[inline]
    fn coefficients_stale(&self) -> bool {
        (self.damper - self.damper_applied).abs() > DAMPER_RESOLUTION
            || (self.pitch_ratio - self.pitch_applied).abs() > PITCH_RESOLUTION
    }

    /// Bend the whole string, as a player's hand does.
    ///
    /// `ratio` is a frequency multiplier: `1.0` is the note as tuned, and
    /// `2^(cents/1200)` bends it by that many cents. Both directions are allowed,
    /// because a player pulls a string sharp and lets it back, and neither is a
    /// special case here.
    ///
    /// What changes is every mode's frequency; what deliberately does not change
    /// is its damping and its coupling to the bridge. A real hand raises the
    /// tension, which does alter both, but by a few parts in a hundred at the
    /// depths anyone plays at, and modelling that properly would mean rebuilding
    /// the mode shapes and the string's whole geometry on every sample of a
    /// shake.
    pub fn set_pitch_ratio(&mut self, ratio: f32) {
        self.pitch_ratio = ratio.clamp(0.25, 4.0);
    }

    /// How far the pitch is currently bent.
    pub fn pitch_ratio(&self) -> f32 {
        self.pitch_ratio
    }

    /// The speaking length, in metres.
    ///
    /// It follows from the tension, the impedance and the fundamental rather
    /// than being set directly, because a string whose tension, mass per unit
    /// length and pitch are all given has no length left to choose.
    pub fn speaking_length(&self) -> f32 {
        self.length
    }

    /// The string's wave impedance, in newton seconds per metre.
    pub fn impedance(&self) -> f32 {
        self.config.impedance
    }

    /// The current damper setting.
    pub fn damper(&self) -> f32 {
        self.damper
    }

    /// A running estimate of the output magnitude.
    pub fn envelope(&self) -> f32 {
        self.envelope
    }

    /// Whether the string has decayed into inaudibility.
    pub fn is_quiet(&self) -> bool {
        self.envelope < 1e-6
    }

    /// Build the per-sample coefficients for every mode.
    fn rebuild_modes(&mut self) {
        let t = self.dt as f64;
        let modal_mass = self.modal_mass;
        // A felt resting on the string is a loss that is proportional to how
        // fast the string is moving under it, so it costs every partial the same
        // fraction of its energy per cycle and the high partials die first.
        // Scaling the added decay with the fundamental instead would leave a
        // damped note with a long ringing top, which is the opposite of what a
        // damper does.
        let felt: f64 = if self.damper < 1.0 {
            -(self.damper.max(1e-6).ln()) as f64
        } else {
            0.0
        };

        let bend = self.pitch_ratio as f64;
        for mode in self.modes.iter_mut() {
            // The bend scales every partial together, which is what a change of
            // tension does to a string: the tuning stretches but the ratios
            // between the partials hold.
            let frequency = mode.frequency as f64 * bend;
            let omega = std::f64::consts::TAU * frequency;
            // The exact solution below is the underdamped one, and its state
            // transition is only guaranteed to be a contraction while the decay
            // rate stays under the frequency. A felt can easily want more than
            // that near the top of the band, and a mode that rings forever is a
            // far worse outcome than one that stops in three samples.
            let sigma = (mode.sigma as f64 + felt * frequency).min(0.9 * omega);
            let damped = (omega * omega - sigma * sigma).max(1e-12).sqrt();
            let decay = (-sigma * t).exp();
            let (sine, cosine) = (damped * t).sin_cos();
            let sigma_over_damped = sigma / damped;

            mode.a = (decay * (cosine + sigma_over_damped * sine)) as f32;
            mode.b = (decay * sine / damped) as f32;
            mode.c = (decay * -omega * omega * sine / damped) as f32;
            mode.d = (decay * (cosine - sigma_over_damped * sine)) as f32;

            // A force held over the sample moves the mode by this much. It is
            // deliberately free of the excitation position: the mode shape is
            // applied per port instead, because two ports see different shapes.
            let i1 = (damped - decay * (damped * cosine + sigma * sine)) / (omega * omega);
            let i2 = (sigma - decay * (sigma * cosine - damped * sine)) / (omega * omega);
            let gain = 1.0 / modal_mass;
            mode.e = (gain * i1 / damped) as f32;
            mode.f = (gain * (i2 - sigma_over_damped * i1)) as f32;
        }

        // The conductance depends on the mode input gains, so it has to follow
        // the coefficients, and the ports hold weights that were built when the
        // mode list was.
        for port in &mut self.ports {
            let mut total = 0.0;
            for (weight, mode) in port.weights.iter().zip(self.modes.iter()) {
                total += weight * weight * mode.f;
            }
            port.conductance = total.max(1e-12);
        }
        self.damper_applied = self.damper;
        self.pitch_applied = self.pitch_ratio;
        self.since_rebuild = 0;
    }
}
