use super::*;

/// One mode of the string.
///
/// The update is the exact solution of the damped oscillator over one sample
/// with the force held constant, which is what keeps the decay accurate when the
/// decay rate is six orders of magnitude below the sample rate. Writing it as a
/// direct form recurrence instead would lose the decay entirely in `f32`.
///
/// The bank is stepped four modes at a time: the lanes are gathered out of this
/// one array, which keeps the layout below and still vectorises the update.
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
pub(super) struct Mode {
	/// The modal coordinate and its rate of change.
	pub(super) q: f32,
	pub(super) v: f32,
	/// The state transition, already scaled by `exp(-sigma T)`.
	pub(super) a: f32,
	pub(super) b: f32,
	pub(super) c: f32,
	pub(super) d: f32,
	/// The contribution of a force held over one sample.
	pub(super) e: f32,
	pub(super) f: f32,
	/// What this mode contributes to the bridge force.
	pub(super) bridge: f32,
	/// The decay rate before the damper is taken into account, in 1/s.
	pub(super) sigma: f32,
	/// The frequency, in hertz.
	pub(super) frequency: f32,
	/// The harmonic order the mode's shape belongs to.
	///
	/// Integral for a harmonic or a polarisation pair, fractional for the
	/// winding layer, which rings off the grid but is driven through, and
	/// radiates as, the shape of the partial it sits beside.
	pub(super) order: f32,
	/// The mode's share of that shape, 1 for a harmonic.
	pub(super) level: f32,
	/// The fraction of `sigma` the decay rate relaxes to at the end of the
	/// note. 1 is a single exponential.
	pub(super) tail: f32,
	/// Whether this is the second polarization of a partial.
	pub(super) horizontal: bool,
	/// The displacement below which this mode is inaudible at the bridge.
	pub(super) quiet_q: f32,
	/// The matching velocity.
	pub(super) quiet_v: f32,
	/// Whether the mode is still loud enough to be worth updating.
	pub(super) awake: bool,
}

/// A pair of transverse partials whose product drives one phantom.
pub(super) struct Pair {
	/// The two parent modes in the mode list.
	pub(super) first: usize,
	pub(super) second: usize,
	/// The coupling of this pair, already including the neighbour falloff.
	pub(super) gain: f32,
}

/// One excitation point on the string.
pub(super) struct Port {
	pub(super) law: Box<dyn ContactLaw>,
	/// The mode shape at this point, one entry per mode.
	pub(super) weights: Vec<f32>,
	/// The instantaneous velocity per unit force at this point, in m/(s N).
	pub(super) conductance: f32,
	pub(super) y: f32,
	pub(super) v: f32,
	pub(super) substeps: u32,
	pub(super) active: bool,
}

impl Port {
	/// The residual of the implicit contact equation.
	#[inline]
	pub(super) fn residual(&self, v: f32, free: f32, conductance: f32, y: f32) -> f32 {
		v - conductance * self.law.force(y, v) - free
	}

	/// Solve for the string velocity at this port over one sample.
	pub(super) fn solve(&mut self, free: f32, conductance: f32, dt: f32) -> f32 {
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
	pub(super) fn solve_once(&self, free: f32, conductance: f32, y: f32, warm: f32) -> (f32, f32) {
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
