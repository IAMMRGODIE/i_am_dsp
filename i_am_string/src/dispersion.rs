//! String stiffness, implemented as a cascade of first order allpass sections.
//!
//! A stiff string is dispersive: the n-th partial sits at
//! `f_n = n f_0 sqrt(1 + B n^2)`, so the loop phase delay has to fall from
//! `D_1 = fs / f_1` at the fundamental to `D_1 / sqrt(1 + B K^2)` at the
//! highest modelled partial. The integer part of that is the delay line; the
//! residual, frequency dependent part is what these allpass sections supply.

/// A first order allpass, `H(z) = (a + z^-1) / (1 + a z^-1)`.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Allpass {
    a: f32,
    state: f32,
}

impl Allpass {
    pub(crate) fn phase_delay(a: f64, omega: f64) -> f64 {
        if omega <= 1e-12 {
            return (1.0 - a) / (1.0 + a);
        }
        let num = (a * omega.sin()).atan2(1.0 + a * omega.cos());
        let den = omega.sin().atan2(a + omega.cos());
        -(num - den) / omega
    }

    #[inline]
    pub(crate) fn process(&mut self, x: f32) -> f32 {
        let y = self.a * x + self.state;
        self.state = x - self.a * y;
        y
    }

    pub(crate) fn clear(&mut self) {
        self.state = 0.0;
    }
}

/// One point of the dispersion design target curve.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Target {
    /// Angular frequency in radians per sample.
    pub(crate) omega: f64,
    /// The allpass phase delay wanted at that frequency, in samples.
    pub(crate) delay: f64,
    /// Relative importance of this point.
    pub(crate) weight: f64,
}

/// A cascade of first order allpass sections used for string dispersion.
#[derive(Clone, Debug, Default)]
pub(crate) struct DispersionFilter {
    sections: Vec<Allpass>,
}

impl DispersionFilter {
    /// The largest number of frequencies the fit is evaluated at.
    ///
    /// This used to be sixteen, on the reasoning that the target delay curve is
    /// smooth. It is, but the *achieved* one is not: leaving eighty of ninety
    /// partials unconstrained lets them land hundreds of cents away, which is
    /// precisely the metallic, detuned-saw quality the model was criticised
    /// for. Every partial that is modelled has to be a target.
    const MAX_TARGETS: usize = 64;

    /// Fit `sections` allpass coefficients to the requested delay curve.
    pub(crate) fn design(targets: &[Target], sections: usize) -> Self {
        let sections = sections.max(1);
        let reduced;
        let targets = if targets.len() > Self::MAX_TARGETS {
            let last = targets.len() - 1;
            reduced = (0..Self::MAX_TARGETS)
                .map(|i| targets[i * last / (Self::MAX_TARGETS - 1)])
                .collect::<Vec<_>>();
            reduced.as_slice()
        } else {
            targets
        };
        if targets.is_empty() {
            return Self {
                sections: vec![Allpass::default(); sections],
            };
        }

        // The objective is not convex: a cascade of first order sections has to
        // trade off a delay that is nearly flat over the first two octaves
        // against one that collapses at the top, and where the steep section
        // sits is a discrete choice. Coordinate descent from a single start
        // therefore gets stuck, so several very different starts are tried and
        // the best is kept.
        let mut best: Option<(f64, Vec<f64>)> = None;
        let mut consider = |coeffs: Vec<f64>| {
            let error = total_error(&coeffs, targets);
            if best.as_ref().is_none_or(|(e, _)| error < *e) {
                best = Some((error, coeffs));
            }
        };

        let common = best_common_coefficient(sections, targets);
        let mut starts: Vec<Vec<f64>> = vec![
            vec![common; sections],
            // Section coefficients that are individually calibrated to land on
            // the target at the fundamental, and again at the middle partial.
            calibrated(sections, targets, 0),
            calibrated(sections, targets, targets.len() / 2),
            spread(sections, 0.30),
        ];
        if sections > 2 {
            starts.push(spread(sections, 0.0));
        }

        for start in starts {
            let mut coeffs = start;
            for _ in 0..3 {
                for k in 0..sections {
                    // Only one coefficient moves at a time, so the contribution
                    // of the other sections is computed once per sweep. Without
                    // this the search is sections times more expensive.
                    let others: Vec<f64> = targets
                        .iter()
                        .map(|target| {
                            coeffs
                                .iter()
                                .enumerate()
                                .filter(|(i, _)| *i != k)
                                .map(|(_, &a)| Allpass::phase_delay(a, target.omega))
                                .sum::<f64>()
                        })
                        .collect();
                    let f = |x: f64| {
                        targets
                            .iter()
                            .enumerate()
                            .map(|(i, target)| {
                                let error =
                                    others[i] + Allpass::phase_delay(x, target.omega) - target.delay;
                                target.weight * error * error
                            })
                            .sum::<f64>()
                    };
                    coeffs[k] = golden_section(&f, -0.9995, 0.9995, 15);
                }
            }
            consider(coeffs);
        }

        let (_, coeffs) = best.unwrap_or_else(|| (0.0, vec![0.0; sections]));
        Self {
            sections: coeffs
                .into_iter()
                .map(|a| Allpass {
                    a: a as f32,
                    state: 0.0,
                })
                .collect(),
        }
    }

    /// Build the cascade from coefficients that were fitted earlier.
    pub(crate) fn from_coefficients(coefficients: &[f32]) -> Self {
        Self {
            sections: coefficients
                .iter()
                .map(|&a| Allpass {
                    a: a.clamp(-0.9995, 0.9995),
                    state: 0.0,
                })
                .collect(),
        }
    }

    /// The number of allpass sections.
    pub(crate) fn sections(&self) -> usize {
        self.sections.len()
    }

    /// The fitted coefficients.
    pub(crate) fn coefficients(&self) -> Vec<f32> {
        self.sections.iter().map(|s| s.a).collect()
    }

    /// The phase delay the cascade currently provides, in samples.
    pub(crate) fn phase_delay(&self, omega: f64) -> f64 {
        self.sections
            .iter()
            .map(|s| Allpass::phase_delay(s.a as f64, omega))
            .sum()
    }

    pub(crate) fn clear(&mut self) {
        for section in &mut self.sections {
            section.clear();
        }
    }

    #[inline]
    pub(crate) fn process(&mut self, x: f32) -> f32 {
        let mut y = x;
        for section in &mut self.sections {
            y = section.process(y);
        }
        y
    }
}

fn total_error(coeffs: &[f64], targets: &[Target]) -> f64 {
    let mut total = 0.0;
    for target in targets {
        let got: f64 = coeffs
            .iter()
            .map(|&a| Allpass::phase_delay(a, target.omega))
            .sum();
        let error = got - target.delay;
        total += target.weight * error * error;
    }
    total
}

fn best_common_coefficient(sections: usize, targets: &[Target]) -> f64 {
    let mut best_a = 0.0f64;
    let mut best = f64::INFINITY;
    const SCAN: usize = 240;
    for i in 0..=SCAN {
        let a = -0.999 + 1.998 * (i as f64) / (SCAN as f64);
        let mut total = 0.0;
        for target in targets {
            let got = Allpass::phase_delay(a, target.omega) * sections as f64;
            let error = got - target.delay;
            total += target.weight * error * error;
        }
        if total < best {
            best = total;
            best_a = a;
        }
    }
    best_a
}

/// A start where every section is given the same coefficient, chosen so that
/// the cascade lands exactly on the target at one of the partials.
fn calibrated(sections: usize, targets: &[Target], index: usize) -> Vec<f64> {
    let target = &targets[index.min(targets.len() - 1)];
    let want = target.delay / sections as f64;
    let mut lo = -0.9995f64;
    let mut hi = 0.9995f64;
    // The phase delay falls as the coefficient rises, so this is a plain
    // bisection.
    if Allpass::phase_delay(lo, target.omega) < want {
        return vec![lo; sections];
    }
    if Allpass::phase_delay(hi, target.omega) > want {
        return vec![hi; sections];
    }
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if Allpass::phase_delay(mid, target.omega) > want {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    vec![0.5 * (lo + hi); sections]
}

/// A start that walks the coefficients from a mild to a steep section.
fn spread(sections: usize, floor_amount: f64) -> Vec<f64> {
    (0..sections)
        .map(|i| {
            let t = if sections <= 1 {
                0.0
            } else {
                i as f64 / (sections - 1) as f64
            };
            -floor_amount - (0.995 - floor_amount) * t
        })
        .collect()
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
