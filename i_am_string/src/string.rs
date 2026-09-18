//! The waveguide string core.
//!
//! # Representation
//!
//! The string is two delay lines of N samples each, carrying the right- and
//! left-going *force* waves. A wave launched at the nut takes N samples to
//! reach the bridge, and one more N to come back, so the fundamental sits at
//! fs / (2N) before dispersion.
//!
//! Using force waves rather than velocity waves pays off twice. The initial
//! condition of a pluck is just F = -/+ (T/2) dy0/dx, with the string tension
//! as the only material constant, and the bridge output is already the force
//! that drives the soundboard, with no differentiation needed.
//!
//! # Ports
//!
//! An excitation is a point where an external force is applied. Writing A for
//! the incident force wave from the nut side and B for the incident force wave
//! from the bridge side, a massless junction gives
//!
//! ```text
//! v = (A - B + F/2) / Z0
//! S = A + F/2      departing towards the bridge
//! Q = B - F/2      departing towards the nut
//! ```
//!
//! With F supplied by a ContactLaw this is one scalar implicit equation in v,
//! which Port::solve brackets and iterates.

use crate::contact::ContactLaw;
use crate::delay::DelayLine;
use crate::dispersion::{DispersionFilter, Target};

/// The physical parameters of one string.
#[derive(Clone, Debug)]
pub struct StringConfig {
    /// The fundamental frequency in hertz.
    pub frequency: f32,
    /// The inharmonicity coefficient B of f_n = n f_0 sqrt(1 + B n^2).
    pub inharmonicity: f32,
    /// The frequency independent part of the decay rate, in 1/s.
    pub sigma0: f32,
    /// The frequency dependent part of the decay rate, in seconds.
    pub sigma1: f32,
    /// The string tension, in newtons. Scales the pluck excitation.
    pub tension: f32,
    /// The string's wave impedance Z0 = sqrt(T mu), in N s/m.
    pub impedance: f32,
    /// The highest partial that is modelled.
    pub max_partials: usize,
    /// The number of allpass sections used for the dispersion fit.
    pub dispersion_sections: usize,
    /// How much of the shortest modelled round trip the delay lines carry.
    ///
    /// The remainder is supplied by the dispersion allpass. Leaving the whole
    /// round trip to the delay lines forces the allpass to realise a delay curve
    /// that falls far too steeply; giving the allpass more of the work makes the
    /// curve it has to fit flatter and much easier to realise.
    pub dispersion_delay_fraction: f32,
    /// A pre-fitted set of allpass coefficients.
    ///
    /// Fitting the dispersion filter takes several milliseconds, which is far
    /// too long to do on the audio thread, so an engine fits it once per note
    /// and passes it back in here. `None` designs a fresh filter.
    pub dispersion_coefficients: Option<Vec<f32>>,
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
            max_partials: 40,
            dispersion_sections: 8,
            dispersion_delay_fraction: 1.0,
            dispersion_coefficients: None,
        }
    }
}

impl StringConfig {
    /// A convenient constructor for a plain, ideal string.
    pub fn new(frequency: f32) -> Self {
        Self {
            frequency,
            ..Default::default()
        }
    }
}

/// How much total stretch (`B k^2` at the top partial) the dispersion filter
/// can be trusted to realise without the fit falling apart.
///
/// Measured, not guessed: above roughly this the worst partial error goes from
/// tens of cents to hundreds, which is the difference between an instrument and
/// a detuned saw.
const AFFORDABLE_STRETCH: f64 = 0.30;

/// A one pole lowpass, used as the frequency dependent loop loss.
#[derive(Clone, Copy, Debug, Default)]
struct OnePole {
    a: f32,
    state: f32,
}

impl OnePole {
    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        self.state = (1.0 - self.a) * x + self.a * self.state;
        self.state
    }

    /// The magnitude response at omega radians per sample.
    fn magnitude(a: f64, omega: f64) -> f64 {
        let denominator = 1.0 - 2.0 * a * omega.cos() + a * a;
        (1.0 - a) / denominator.max(1e-12).sqrt()
    }

    /// The phase delay at omega radians per sample, in samples.
    fn phase_delay(a: f64, omega: f64) -> f64 {
        if omega <= 1e-12 {
            return 0.0;
        }
        (a * omega.sin()).atan2(1.0 - a * omega.cos()) / omega
    }
}

/// One excitation point on the string.
struct Port {
    law: Box<dyn ContactLaw>,
    delay_from_nut: f32,
    delay_from_bridge: f32,
    /// The nominal travel time from the port to each end, which is what the
    /// bend scales. The lines themselves are allocated longer than this so that
    /// bending *down* has somewhere to read from, and reading them at their own
    /// length would put the reflected wave back on the string fifteen percent
    /// late - which is not a subtle detune, it is a bowed string that grows
    /// instead of singing.
    bridge_travel: f32,
    nut_travel: f32,
    to_bridge: DelayLine,
    to_nut: DelayLine,
    y: f32,
    v: f32,
    substeps: u32,
    active: bool,
}

impl Port {
    /// Solve the contact for one sample and return the force that was applied.
    fn solve(&mut self, incident: f32, impedance: f32, dt: f32) -> f32 {
        let substeps = self.substeps.max(1);
        let sub_dt = dt / substeps as f32;
        let mut y = self.y;
        let mut v = self.v;
        let mut force = 0.0;
        for _ in 0..substeps {
            let solved = self.solve_once(incident, impedance, y, v);
            v = solved.0;
            force = solved.1;
            self.law.advance(y, v, sub_dt);
            y += v * sub_dt;
        }
        self.y = y;
        self.v = v;
        force
    }

    /// The residual of the implicit contact equation.
    #[inline]
    fn residual(&self, v: f32, incident: f32, impedance: f32, y: f32) -> f32 {
        impedance * v - 0.5 * self.law.force(y, v) - incident
    }

    /// Solve the implicit contact equation for the string velocity.
    ///
    /// The bracket is exact: with no contact force the root is incident / Z0,
    /// and with the largest possible force it is at most F_max / (2 Z0) above
    /// that.
    ///
    /// Inside that bracket the equation is *not* always monotone. A bow has a
    /// negative differential resistance over a wide band of relative velocities,
    /// which gives the residual two roots: one on the sticking branch and one on
    /// the slipping branch. A plain bracketed solve jumps between them from
    /// sample to sample, and the result is broadband hash instead of a tone. So
    /// the solve runs in two phases: a warm started, step limited Newton that
    /// follows the branch the string is already on, and only if that fails a
    /// bracketed fallback that is guaranteed to converge.
    fn solve_once(&self, incident: f32, impedance: f32, y: f32, warm: f32) -> (f32, f32) {
        // v = (incident + F/2) / Z0, so the bracket is the whole reachable force
        // range. For anything that only pushes, the lower end is incident / Z0.
        let lo = (incident + 0.5 * self.law.min_force()) / impedance;
        let mut hi = (incident + 0.5 * self.law.max_force()) / impedance;
        if !hi.is_finite() || hi <= lo {
            hi = lo + 1e-6;
        }
        let mut guard = 0;
        while self.residual(hi, incident, impedance, y) < 0.0 && guard < 8 {
            hi = lo + (hi - lo) * 4.0;
            guard += 1;
        }

        let limit = (hi - lo) * 0.05;
        let tolerance = 1e-9 * (1.0 + incident.abs());

        // Phase one: warm started, step limited Newton. The step limit is what
        // keeps it on the branch the string is already on.
        let mut v = warm.clamp(lo, hi);
        let mut best = v;
        let mut best_residual = f32::INFINITY;
        for _ in 0..24 {
            let residual = self.residual(v, incident, impedance, y);
            if residual.abs() <= tolerance {
                return (v, self.law.force(y, v));
            }
            if residual.abs() < best_residual {
                best_residual = residual.abs();
                best = v;
            }
            let slope = impedance - 0.5 * self.law.force_dv(y, v);
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

        // Phase two, only when Newton failed. A bracket is grown outwards a step
        // at a time from the best iterate, never taken as the whole reachable
        // force range: that range can be thousands of metres per second wide,
        // and a bisection across it would land nowhere near a physical root.
        if best_residual > tolerance {
            let centre = self.residual(best, incident, impedance, y);
            let mut bracket = None;
            let mut width = limit.max(1e-6);
            for _ in 0..48 {
                let upper = (best + width).min(hi);
                if self.residual(upper, incident, impedance, y) * centre <= 0.0 {
                    bracket = Some((best, upper, centre));
                    break;
                }
                let lower = (best - width).max(lo);
                let value = self.residual(lower, incident, impedance, y);
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
                    if upper - lower <= 1e-12 * (1.0 + mid.abs()) {
                        break;
                    }
                    let value = self.residual(mid, incident, impedance, y);
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

/// How much longer than nominal the delay lines are allocated.
///
/// A bend is applied by reading the loop at a scaled distance rather than by
/// resizing the lines, because resizing is what would reallocate, and a line
/// that reallocates mid-note loses the string. Bending down reads *further* back
/// than nominal, so the lines have to already be that long.
const BEND_HEADROOM: f32 = 1.15;

/// A single physical string, dispersive and lossy, with any number of
/// excitation ports.
pub struct WaveguideString {
    right: DelayLine,
    left: DelayLine,
    /// The half length of the string at its tuned pitch, in samples.
    nominal: f32,
    /// How far the pitch is currently bent, as a frequency multiplier.
    pitch_ratio: f32,
    dispersion: DispersionFilter,
    loss: OnePole,
    loop_gain: f32,
    damper: f32,
    impedance: f32,
    tension: f32,
    dt: f32,
    ports: Vec<Port>,
    envelope: f32,
    config: StringConfig,
}

impl WaveguideString {
    /// Build a string from its physical description.
    pub fn new(sample_rate: f32, mut config: StringConfig) -> Self {
        let sample_rate = if sample_rate > 0.0 { sample_rate } else { 48_000.0 };
        let fs = sample_rate as f64;
        let nyquist = fs * 0.5 * 0.94;
        let f0 = (config.frequency as f64).clamp(8.0, nyquist * 0.5);
        let requested = config.inharmonicity.max(0.0) as f64;

        let build = |b: f64| {
            let mut partials: Vec<(f64, f64)> = Vec::new();
            for k in 1..=config.max_partials.max(1) {
                let index = k as f64;
                let f = index * f0 * (1.0 + b * index * index).sqrt();
                if f >= nyquist {
                    break;
                }
                partials.push((index, f));
            }
            if partials.is_empty() {
                partials.push((1.0, f0));
            }
            partials
        };

        // A cascade of first order allpass sections can only realise a
        // stiff-string delay curve over a limited stretch. Push past it and the
        // fit does not degrade gracefully: the upper partials end up hundreds of
        // cents out, and a string of mistuned partials sounds like a detuned saw
        // rather than like an instrument. So the model is only made as
        // inharmonic as it can be *accurately*, and the stretch is deliberately
        // left short of what was asked for when the two disagree.
        let mut partials = build(requested);
        let reach = (partials.len() as f64).powi(2);
        let affordable = AFFORDABLE_STRETCH / reach.max(1.0);
        let b = if requested > affordable {
            partials = build(affordable);
            affordable
        } else {
            requested
        };
        // The configuration has to report what was actually modelled, or the
        // tuning display and the tests would compare against a stretch the
        // string never had.
        config.inharmonicity = b as f32;

        let omega_of = |f: f64| std::f64::consts::TAU * f / fs;
        let (loss_a, loop_gain) =
            design_loss(&partials, config.sigma0 as f64, config.sigma1 as f64, fs);
        let loss_phase = |f: f64| OnePole::phase_delay(loss_a as f64, omega_of(f));

        // The delay lines supply an integer round trip; the allpass cascade has
        // to make up the rest, which is largest at the fundamental and shrinks
        // to nothing at the top partial.
        let round_trips: Vec<f64> = partials
            .iter()
            .map(|&(k, f)| k * fs / f - loss_phase(f))
            .collect();
        // The delay lines carry the *integer* part of the shortfall, but their
        // length is left fractional: the read is interpolated, and linear
        // interpolation is exactly linear phase, so the fundamental lands where
        // it should instead of a few cents off.
        let shortest = round_trips.iter().copied().fold(f64::INFINITY, f64::min);
        // How much of the round trip the delay lines carry decides how hard the
        // allpass cascade has to work. Give them all of it and the cascade has
        // to pull its delay down to zero at the top partial, which it cannot do:
        // a first order allpass has a phase floor of pi/omega, so any section
        // that is contributing at all still contributes about pi/omega up there.
        // Leave the cascade that much to do instead, and it only has to shape a
        // gentle curve rather than an impossible one.
        let sections_wanted = config.dispersion_sections.max(1) as f64;
        let floor = 0.5 * sections_wanted * std::f64::consts::PI / omega_of(partials[partials.len() - 1].1);
        let fraction = config.dispersion_delay_fraction.clamp(0.05, 1.0) as f64;
        let n0 = if shortest.is_finite() && shortest > 2.0 {
            ((shortest - floor).max(shortest * 0.4) * 0.5).min(shortest * 0.5 * fraction)
        } else {
            1.0
        };
        let delay = 2.0 * n0;

        let targets: Vec<Target> = partials
            .iter()
            .enumerate()
            .map(|(i, &(_, f))| {
                let total = round_trips[i] + loss_phase(f);
                let want = (total - delay).max(0.0);
                // Weighting by 1 / D^2 equalises the *relative* delay error, which
                // is exactly the tuning error of that partial.
                let weight = 1.0 / (total * total).max(1.0);
                Target {
                    omega: omega_of(f),
                    delay: want,
                    weight,
                }
            })
            .collect();

        let dispersion = match &config.dispersion_coefficients {
            Some(coefficients) => DispersionFilter::from_coefficients(coefficients),
            None => DispersionFilter::design(&targets, config.dispersion_sections),
        };

        // The allpass fit is never exact, and a few cents of error at the
        // fundamental is far more audible than a slightly wrong stretch higher
        // up. The delay lines are fractional, so once the filter is fixed the
        // fundamental can be tuned exactly by solving
        //     (2N + tau_allpass + tau_loss) * omega_1 = 2 pi
        // for N.
        let f1 = partials[0].1;
        let omega1 = omega_of(f1);
        let required = fs / f1 - dispersion.phase_delay(omega1) - loss_phase(f1);
        let n0 = if required.is_finite() && required > 2.0 {
            (0.5 * required).clamp(1.0, n0 * 2.0)
        } else {
            n0
        };

        Self {
            right: DelayLine::new(n0 as f32 * BEND_HEADROOM),
            left: DelayLine::new(n0 as f32 * BEND_HEADROOM),
            // (both lines have the same length)
            nominal: n0 as f32,
            pitch_ratio: 1.0,
            dispersion,
            loss: OnePole {
                a: loss_a,
                state: 0.0,
            },
            loop_gain,
            damper: 1.0,
            impedance: config.impedance.max(1e-4),
            tension: config.tension.max(1.0),
            dt: 1.0 / sample_rate,
            ports: Vec::new(),
            envelope: 0.0,
            config,
        }
    }

    /// The configuration this string was built from.
    pub fn config(&self) -> &StringConfig {
        &self.config
    }

    /// The number of allpass sections the dispersion fit produced.
    pub fn dispersion_sections(&self) -> usize {
        self.dispersion.sections()
    }

    /// The fitted allpass coefficients, for inspection and for tests.
    pub fn dispersion_coefficients(&self) -> Vec<f32> {
        self.dispersion.coefficients()
    }

    /// The modelled partial frequencies, for tests and for tuning displays.
    pub fn partial_frequencies(&self) -> Vec<f32> {
        let fs = 1.0 / self.dt as f64;
        let f0 = self.config.frequency as f64;
        let b = self.config.inharmonicity.max(0.0) as f64;
        let nyquist = fs * 0.5 * 0.94;
        let mut out = Vec::new();
        for k in 1..=self.config.max_partials.max(1) {
            let index = k as f64;
            let f = index * f0 * (1.0 + b * index * index).sqrt();
            if f >= nyquist {
                break;
            }
            out.push(f as f32);
        }
        out
    }

    /// The frequencies the loop actually resonates at, found by scanning the
    /// loop phase. Comparing these against [WaveguideString::partial_frequencies]
    /// is how the dispersion fit is judged.
    pub fn resonance_frequencies(&self) -> Vec<f32> {
        // The nominal half length, not the line's own length: the lines are
        // allocated with headroom for a bend, and the string does not get longer
        // just because there is somewhere to put it.
        let n = self.nominal as f64;
        let pi = std::f64::consts::PI;
        // Two delay lines, the dispersion filter and the loss filter, all once
        // per round trip.
        let loss_a = self.loss.a as f64;
        let phase_delay = |omega: f64| {
            2.0 * n + self.dispersion.phase_delay(omega) + OnePole::phase_delay(loss_a, omega)
        };
        let mut out = Vec::new();
        for index in 1..=self.config.max_partials.max(1) {
            let target = std::f64::consts::TAU * index as f64;
            let mut lo = 1e-9;
            let mut hi = pi;
            if phase_delay(hi) * hi < target {
                break;
            }
            for _ in 0..80 {
                let mid = 0.5 * (lo + hi);
                if phase_delay(mid) * mid < target {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let omega = 0.5 * (lo + hi);
            let frequency = omega * (1.0 / self.dt as f64) / std::f64::consts::TAU;
            out.push(frequency as f32);
        }
        out
    }

    /// Remove every excitation port and reset the travelling wave state.
    pub fn reset(&mut self) {
        self.right.clear();
        self.left.clear();
        self.dispersion.clear();
        self.loss.state = 0.0;
        self.ports.clear();
        self.envelope = 0.0;
    }

    /// Add an excitation port at `position`, measured from the nut.
    pub fn add_port(&mut self, position: f32, law: Box<dyn ContactLaw>, substeps: u32) {
        let beta = position.clamp(0.004, 0.996);
        let n = self.nominal;
        self.ports.push(Port {
            law,
            delay_from_nut: beta * n,
            delay_from_bridge: (1.0 - beta) * n,
            bridge_travel: (1.0 - beta) * n,
            nut_travel: beta * n,
            to_bridge: DelayLine::new((1.0 - beta) * n * BEND_HEADROOM),
            to_nut: DelayLine::new(beta * n * BEND_HEADROOM),
            y: 0.0,
            v: 0.0,
            substeps: substeps.max(1),
            active: true,
        });
    }

    /// Release the string from a triangular displacement.
    ///
    /// `amplitude` is the peak displacement as a fraction of the speaking
    /// length. The initial shape is split into the two travelling waves exactly,
    /// which is the whole of a pluck: F = -/+ (T/2) dy0/dx.
    pub fn pluck(&mut self, position: f32, amplitude: f32) {
        let beta = position.clamp(0.01, 0.5);
        let n = self.nominal.max(1.0);
        let half_tension = 0.5 * self.tension;
        let left_slope = amplitude / beta;
        let right_slope = amplitude / (1.0 - beta);
        self.ports.clear();
        self.dispersion.clear();
        self.loss.state = 0.0;
        // A string released from rest has equal travelling displacement on both
        // sides, y+ = y- = y0/2, which in force waves is the *same* sign in both
        // lines. Opposite signs would be the mirror image, which is only correct
        // if the ends reflect force with a sign flip - they do not.
        self.right.load(|d| {
            let s = (d / n).clamp(0.0, 1.0);
            if s < beta {
                -half_tension * left_slope
            } else {
                half_tension * right_slope
            }
        });
        self.left.load(|d| {
            let s = (1.0 - d / n).clamp(0.0, 1.0);
            if s < beta {
                -half_tension * left_slope
            } else {
                half_tension * right_slope
            }
        });
    }

    /// The current displacement and velocity at each excitation port.
    ///
    /// A bowed string is supposed to alternate between sticking to the bow and
    /// slipping past it; reading these is how you tell whether it actually is.
    pub fn port_states(&self) -> Vec<(f32, f32)> {
        self.ports.iter().map(|port| (port.y, port.v)).collect()
    }

    /// Tell every port to let go of the string.
    pub fn release_ports(&mut self, dt: f32) {
        for port in &mut self.ports {
            port.law.release(dt);
        }
    }

    /// Advance the string by one sample and return the force at the bridge.
    pub fn step(&mut self) -> f32 {
        let dt = self.dt;
        let impedance = self.impedance;
        let loop_gain = self.loop_gain * self.damper;
        // Bending up shortens the round trip, so the reads come *closer*.
        let scale = 1.0 / self.pitch_ratio;
        let nominal = self.nominal;

        let bridge_in;
        {
            let Self {
                right,
                left,
                ports,
                dispersion,
                loss,
                ..
            } = self;

            let mut bridge_sum = 0.0;
            let mut nut_sum = 0.0;
            for port in ports.iter_mut() {
                if !port.active {
                    // A released contact still has waves in flight towards both
                    // ends of the string, so its delay lines have to keep
                    // running or that energy would be trapped forever.
                    port.to_bridge.push(0.0);
                    port.to_nut.push(0.0);
                    bridge_sum += port.to_bridge.tap(port.bridge_travel * scale);
                    nut_sum += port.to_nut.tap(port.nut_travel * scale);
                    continue;
                }
                let from_nut = right.arrival(port.delay_from_nut * scale);
                let from_bridge = left.arrival(port.delay_from_bridge * scale);
                let incident = from_nut - from_bridge;
                let force = port.solve(incident, impedance, dt);
                // Only the scattered part is injected: the incident waves are
                // already travelling down the two delay lines and will reach the
                // ends on their own. Adding them here as well would double every
                // arrival, and with a bow on the string that feedback is enough
                // to make the model explode.
                let half = 0.5 * force;
                port.to_bridge.push(half);
                port.to_nut.push(-half);
                bridge_sum += port.to_bridge.tap(port.bridge_travel * scale);
                nut_sum += port.to_nut.tap(port.nut_travel * scale);
                if !port.law.is_active() {
                    port.active = false;
                }
            }

            bridge_in = right.arrival(nominal * scale) + bridge_sum;
            let nut_in = left.arrival(nominal * scale) + nut_sum;

            // One pass around the loop: dispersion, then loss, then the rigid
            // bridge reflection.
            //
            // The reflection coefficient is *plus* one, not minus. These delay
            // lines carry force waves, and at a rigid termination it is the
            // velocity that inverts while the force does not. Getting this
            // backwards leaves the model with a DC path - a constant force then
            // produces a constant velocity instead of a static deflection - and
            // a bow simply holds the string rigidly forever instead of ever
            // slipping, so it never sings.
            let dispersed = dispersion.process(bridge_in);
            let filtered = loss.process(dispersed);
            left.push(filtered * loop_gain);
            right.push(nut_in);
        }

        self.envelope = (self.envelope * 0.9997).max(bridge_in.abs());
        bridge_in
    }

    /// Bend the whole string, as a player's hand does.
    ///
    /// `ratio` is a frequency multiplier. The bend is applied by reading both
    /// delay lines, and every port, at a scaled distance, which changes how long
    /// a wave takes to go round without touching the storage. What it does not
    /// do is change the dispersion filter's curve, so a large bend detunes the
    /// upper partials slightly against the lower ones - a semitone or so is
    /// audible as a slight dulling, which is why the headroom is set for a
    /// vibrato rather than for a pitch bend.
    pub fn set_pitch_ratio(&mut self, ratio: f32) {
        self.pitch_ratio = ratio.clamp(1.0 / BEND_HEADROOM, BEND_HEADROOM);
    }

    /// How far the pitch is currently bent.
    pub fn pitch_ratio(&self) -> f32 {
        self.pitch_ratio
    }

    /// Set the extra per-round-trip damping of a damper felt.
    ///
    /// 1.0 is an open string, 0.5 is a damper resting on the string.
    pub fn set_damper(&mut self, gain: f32) {
        self.damper = gain.clamp(0.0, 1.0);
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
}

/// Fit one scalar gain plus a one pole lowpass to the requested per-round-trip
/// decay of every partial.
fn design_loss(partials: &[(f64, f64)], sigma0: f64, sigma1: f64, fs: f64) -> (f32, f32) {
    let gains: Vec<f64> = partials
        .iter()
        .map(|&(k, f)| {
            let sigma = sigma0 + sigma1 * f * f;
            (-sigma * k / f).exp()
        })
        .collect();
    let omegas: Vec<f64> = partials
        .iter()
        .map(|&(_, f)| std::f64::consts::TAU * f / fs)
        .collect();
    let logs: Vec<f64> = gains.iter().map(|g| g.max(1e-12).ln()).collect();

    let objective = |a: f64| {
        let residuals: Vec<f64> = (0..logs.len())
            .map(|i| logs[i] - OnePole::magnitude(a, omegas[i]).max(1e-12).ln())
            .collect();
        let mean = residuals.iter().sum::<f64>() / residuals.len() as f64;
        residuals.iter().map(|r| (r - mean) * (r - mean)).sum()
    };

    let a = golden_section(&objective, 0.0, 0.995, 26);
    let mean = (0..logs.len())
        .map(|i| logs[i] - OnePole::magnitude(a, omegas[i]).max(1e-12).ln())
        .sum::<f64>()
        / logs.len() as f64;
    let gain = mean.exp().min(1.0) as f32;
    (a as f32, gain)
}

fn golden_section(f: &impl Fn(f64) -> f64, mut lo: f64, mut hi: f64, iterations: usize) -> f64 {
    const INV_PHI: f64 = 0.618_033_988_749_894_9;
    let mut c = hi - INV_PHI * (hi - lo);
    let mut d = lo + INV_PHI * (hi - lo);
    let mut fc = f(c);
    let mut fd = f(d);
    for _ in 0..iterations {
        if fc < fd {
            hi = d;
            d = c;
            fd = fc;
            c = hi - INV_PHI * (hi - lo);
            fc = f(c);
        } else {
            lo = c;
            c = d;
            fc = fd;
            d = lo + INV_PHI * (hi - lo);
            fd = f(d);
        }
    }
    0.5 * (lo + hi)
}
