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

use crate::bridge::{damping_ratio, detune};
use crate::contact::ContactLaw;

mod config;

pub use config::StringConfig;
use config::*;

mod modes;

use modes::*;
use wide::f32x4;

/// The exact state transition of a damped oscillator over one sample.
fn transition(frequency: f64, sigma: f64, t: f64) -> (f32, f32, f32, f32) {
	let omega = std::f64::consts::TAU * frequency;
	let sigma = sigma.min(0.9 * omega);
	let damped = (omega * omega - sigma * sigma).max(1e-12).sqrt();
	let decay = (-sigma * t).exp();
	let (sine, cosine) = (damped * t).sin_cos();
	let ratio = sigma / damped;
	(
		(decay * (cosine + ratio * sine)) as f32,
		(decay * sine / damped) as f32,
		(decay * -omega * omega * sine / damped) as f32,
		(decay * (cosine - ratio * sine)) as f32,
	)
}

/// `dst += src * gain`, four lanes at a time.
///
/// The mode shapes and the force they carry are both flat arrays, so the sum a
/// contact adds is a contiguous multiply-add rather than a walk over the bank.
fn saxpy(dst: &mut [f32], src: &[f32], gain: f32) {
	let lanes = f32x4::splat(gain);
	let mut index = 0;

	while index + 4 <= dst.len() {
		let sum = f32x4::from(&dst[index..index + 4]) + f32x4::from(&src[index..index + 4]) * lanes;
		dst[index..index + 4].copy_from_slice(&sum.to_array());
		index += 4;
	}

	while index < dst.len() {
		dst[index] += src[index] * gain;
		index += 1;
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
	/// Scratch space: each contact's free velocity this sample.
	free: Vec<f32>,
	/// Samples since the coefficients were last rebuilt, to cap the rebuild rate.
	since_rebuild: u32,
	/// How far the pitch is currently shifted, as a frequency multiplier.
	pitch_ratio: f32,
	/// The pitch the coefficients were last built for.
	pitch_applied: f32,
	/// How long the string has been sounding, in seconds.
	age: f64,
	/// How far the decay has relaxed towards its tail, 0..1.
	blend: f32,
	/// The blend the coefficients were last built for.
	blend_applied: f32,
	/// The transverse pairs whose products drive phantoms.
	pairs: Vec<Pair>,
	/// The phantom modes, created once the strike has excited the parents.
	phantoms: Vec<Mode>,
	/// Whether the phantoms have been seeded from this note's parents.
	seeded: bool,
	config: StringConfig,
}

impl ModalString {
	/// Build a string from its physical description.
	pub fn new(sample_rate: f32, config: StringConfig) -> Self {
		let sample_rate = if sample_rate > 0.0 {
			sample_rate
		} else {
			48_000.0
		};
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

		let voicing = config.voicing;
		// What every mode is: its harmonic order, where it rings, how much of
		// its shape it carries, how fast it decays and how far that decay
		// relaxes after the prompt sound. The harmonic series comes first; the
		// pair and the winding layer are added beside the partials they colour.
		let mut built: Vec<(f64, f64, f64, f64, f32, bool)> = Vec::new();
		for index in 1..=MAX_MODES {
			let n = index as f64;
			let mut frequency = n * f0 * (1.0 + b * n * n).sqrt();
			if frequency >= nyquist {
				break;
			}
			let mut sigma = config.sigma0 as f64 + config.sigma1 as f64 * frequency * frequency;
			if config.bridge > 0.0 && (frequency as f32) < crate::bridge::F_MAX {
				// The board drains a partial that sits on one of its resonances
				// faster than its neighbours, and its reactance pulls the
				// partial off the harmonic grid. The table is already centred on
				// one, so the blend leaves the fitted decay law where it is.
				let weight = bridge_weight(frequency as f32) as f64;
				let ratio = damping_ratio(frequency as f32) as f64;
				let damping = (1.0 + config.bridge as f64 * weight * (ratio - 1.0))
					.clamp(BRIDGE_DAMPING_MIN, BRIDGE_DAMPING_MAX);
				sigma *= damping;
				frequency += detune(f0 as f32, impedance as f32, frequency as f32) as f64
					* config.bridge as f64;
			}
			built.push((n, frequency, 1.0, sigma, voicing.late_decay, false));

			// The pair. One partial is two polarisations that are not quite the
			// same length, and the few cents between them are what make a note
			// bloom and beat rather than stand still. It carries a few percent
			// of the partial, so it is colour and not a second note.
			if voicing.doublet > 0.0 && index <= DOUBLET_MODES {
				let side = frequency * (1.0 + voicing.doublet_detune as f64);
				if side < nyquist {
					built.push((
						n,
						side,
						voicing.doublet as f64,
						sigma * 1.1,
						voicing.late_decay,
						false,
					));
				}
			}

			// The winding. A wound string's cover wire rings on its own, off
			// the harmonic grid, and dies inside a second, so at these levels it
			// is heard as a metallic edge on the attack and not as a pitch.
			if voicing.winding > 0.0 && (2..=WINDING_MODES).contains(&index) {
				let wound = f0 * (n + 0.34 + 0.011 * n * n);
				if wound < nyquist {
					let level = voicing.winding as f64 * WINDING_LEVEL / n.sqrt();
					built.push((n, wound, level, WINDING_SIGMA, 1.0, false));
				}
			}

			// The second polarization. Same shape, the slow decay: the hammer
			// hits both planes, the bridge drains one, and the other is left
			// holding the note after the prompt.
			if voicing.horizontal > 0.0 && index <= HORIZONTAL_MODES {
				let level = voicing.horizontal as f64;
				built.push((
					n,
					frequency * (1.0 + 2.5e-6),
					level,
					sigma * voicing.horizontal_decay as f64,
					1.0,
					true,
				));
			}
		}
		if built.is_empty() {
			built.push((1.0, f0, 1.0, config.sigma0 as f64, 1.0, false));
		}

		let mut modes = Vec::with_capacity(built.len());
		for (order, frequency, level, sigma, tail, horizontal) in built {
			let wave_number = order * std::f64::consts::PI / length;
			// `T dy/dx` at the bridge is the force the string pulls the bridge
			// with, and sin(n pi x / L) has slope n pi / L times cos(n pi). The
			// further halving is the convention the waveguide uses and the
			// engine is written against: what a string hands over is the force
			// wave *arriving* at the bridge, which the rigid termination then
			// doubles. Both cores return the same quantity so they can be
			// compared without a level jump, and the reference force in
			// [crate::instrument] is scaled to match.
			// The sign of the slope at the bridge is `cos(n pi)`: an exact plus
			// or minus one for a harmonic, and something in between only for the
			// winding layer's fractional orders. Keeping the harmonic case
			// exact is what lets a string without colour layers render the same
			// samples it did before those layers existed.
			let whole = order.round();
			let sign = if (order - whole).abs() < 1e-9 {
				if whole as i64 % 2 == 0 { 1.0 } else { -1.0 }
			} else {
				(order * std::f64::consts::PI).cos()
			};
			let bridge = 0.5 * tension * wave_number * sign;
			// A mode falls silent when the force it puts on the bridge drops
			// under the floor. As a displacement that is the floor divided by
			// the mode's own coupling, which is why a high mode can be pinned
			// much sooner than a low one.
			let quiet_q = QUIET_FORCE / bridge.abs().max(1e-30) as f32;
			modes.push(Mode {
				order: order as f32,
				level: level as f32,
				tail,
				horizontal,
				quiet_q,
				quiet_v: quiet_q * std::f32::consts::TAU * frequency as f32,
				sigma: sigma as f32,
				frequency: frequency as f32,
				bridge: bridge as f32,
				awake: true,
				..Default::default()
			});
		}

		// The pairs whose products make phantoms. Only harmonics are parents:
		// a detuned pair or a winding mode is a colour, not a place the
		// longitudinal coupling is heard from.
		let mut pairs = Vec::new();
		if voicing.phantom > 0.0 {
			let parents: Vec<usize> = modes
				.iter()
				.enumerate()
				.filter(|(_, mode)| {
					!mode.horizontal
						&& (mode.level - 1.0).abs() < 1e-6
						&& mode.order <= PHANTOM_ORDER as f32
						&& (mode.order - mode.order.round()).abs() < 1e-3
				})
				.map(|(index, _)| index)
				.collect();
			for (start, &first) in parents.iter().enumerate() {
				for distance in 0..=PHANTOM_PAIRS {
					let Some(&second) = parents.get(start + distance) else {
						break;
					};
					let sum = modes[first].frequency as f64 + modes[second].frequency as f64;
					if sum > PHANTOM_FMAX || sum >= nyquist {
						break;
					}
					pairs.push(Pair {
						first,
						second,
						gain: voicing.phantom * PHANTOM_LEVEL * 0.55_f32.powi(distance as i32),
					});
				}
			}
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
			free: Vec::new(),
			since_rebuild: 0,
			pitch_ratio: 1.0,
			pitch_applied: f32::NAN,
			age: 0.0,
			blend: 0.0,
			blend_applied: f32::NAN,
			pairs,
			phantoms: Vec::new(),
			seeded: false,
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
	/// point of the architecture. The list includes the colour layers - the
	/// detuned pairs and the winding partials - because those are modes too.
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
		// A silent string starts its life over, tail and all, so that the next
		// pluck on the same object is not coloured by the last one's age.
		self.age = 0.0;
		self.blend = 0.0;
		self.blend_applied = f32::NAN;
		self.phantoms.clear();
		self.seeded = false;
		self.rebuild_modes();
	}

	/// Add an excitation port at `position`, measured from the nut.
	pub fn add_port(&mut self, position: f32, law: Box<dyn ContactLaw>, substeps: u32) {
		let beta = position.clamp(0.004, 0.996) as f64;
		let weights: Vec<f32> = self
			.modes
			.iter()
			.map(|mode| {
				(mode.level as f64 * (mode.order as f64 * std::f64::consts::PI * beta).sin()) as f32
			})
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

	/// Start a calibrated bank from its own levels, all in phase.
	///
	/// A measured table already says how loud each partial is, so the contact
	/// solver is skipped and the modes are given their state directly. The
	/// amplitude is a scale factor the caller picks; the table only knows the
	/// ratios.
	pub fn launch(&mut self, amplitude: f32) {
		self.ports.clear();
		self.age = 0.0;
		self.blend = 0.0;
		self.blend_applied = f32::NAN;
		self.phantoms.clear();
		self.seeded = false;
		for mode in &mut self.modes {
			mode.q = 0.0;
			mode.v = amplitude * mode.level * std::f32::consts::TAU * mode.frequency;
			mode.awake = true;
		}
		self.rebuild_modes();
	}

	/// Release the string from a triangular displacement.
	///
	/// `amplitude` is the peak displacement as a fraction of the speaking
	/// length. The mode amplitudes of a triangle are known in closed form.
	pub fn pluck(&mut self, position: f32, amplitude: f32) {
		// The pluck point is given from the nut, and the triangle is not
		// symmetric about the middle: 0.75 is the soundhole of a guitar and
		// 0.25 is the same point seen from the other end.
		let beta = position.clamp(0.01, 0.95) as f64;
		// The amplitude is a fraction of the speaking length and the mode
		// coordinates are displacements, so it has to become a length before the
		// mode sum means anything. Leaving it as a fraction and reading metres
		// out of the sum is a scale error of the whole string length, which on a
		// scaled string is more than a factor of ten.
		let height = amplitude as f64 * self.length as f64;
		self.ports.clear();
		// A fresh pluck starts the note over, tail included, so that the age of
		// the last note cannot colour this one.
		self.age = 0.0;
		self.blend = 0.0;
		self.blend_applied = f32::NAN;
		self.phantoms.clear();
		self.seeded = false;
		let pi = std::f64::consts::PI;
		let tilt = self.config.voicing.pluck_tilt.max(0.0) as f64;
		for mode in self.modes.iter_mut() {
			let n = mode.order as f64;
			let shape = (n * pi * beta).sin();
			// The finite width of the finger, plectrum or hammer felt: the top of
			// a triangle's spectrum falls as 1/n^2, which is brighter than any
			// real pluck. This is the extra tilt the contact imposes, and it is
			// written as the exponential the top of a struck spectrum actually
			// follows rather than as a sinc, because the contact is soft.
			let contact = (-tilt * (n - 1.0).max(0.0)).exp();
			mode.q = (mode.level as f64 * 2.0 * height / (n * n * pi * pi * beta * (1.0 - beta))
				* shape
				* contact) as f32;
			mode.v = 0.0;
			// A plucked mode is awake whatever its amplitude, because the pluck
			// is what put it there.
			mode.awake = true;
		}
		self.envelope = 0.0;
		self.rebuild_modes();
	}

	/// Release the string from a triangular displacement over a finite time.
	///
	/// `seconds` is how long the finger takes to slip off. The string follows
	/// it while it is still touching, so it is already moving when contact is
	/// lost: the displacement is the same triangle, but every mode starts
	/// with a velocity as well. The partials are unchanged; what changes is
	/// the first few milliseconds, which are no longer a step.
	pub fn pluck_with_release(&mut self, position: f32, amplitude: f32, seconds: f32) {
		self.pluck(position, amplitude);
		if seconds <= 0.0 {
			return;
		}
		let beta = position.clamp(0.01, 0.95) as f64;
		// The finger is a few millimetres wide, and the string does not keep
		// up with the whole stroke: it slips before the finger has gone far.
		// A quarter of the finger's speed is what the string keeps.
		let width = 0.006;
		let height = amplitude as f64 * self.length as f64;
		let velocity = -0.25 * height / seconds.max(1e-4) as f64;
		let pi = std::f64::consts::PI;
		for mode in &mut self.modes {
			let n = mode.order as f64;
			let aperture = n * pi * width * 0.5;
			let aperture = if aperture.abs() < 1e-9 {
				1.0
			} else {
				aperture.sin() / aperture
			};
			mode.v = (4.0 * velocity / (n * pi) * (n * pi * beta).sin() * aperture) as f32;
		}
	}

	/// Create the phantom modes from the partials the strike has excited.
	///
	/// The coupling is quadratic, so a soft note has almost none and a hard
	/// one has a lot. Each phantom is bounded by a fraction of the geometric
	/// mean of its parents, so a fortissimo sum tone can never outgrow what
	/// drives it.
	fn seed_phantoms(&mut self) {
		self.seeded = true;
		if self.pairs.is_empty() {
			return;
		}
		let peak = self
			.pairs
			.iter()
			.flat_map(|pair| [pair.first, pair.second])
			.map(|index| self.modes[index].q.abs())
			.fold(0.0f32, f32::max);
		if peak <= 1e-12 {
			return;
		}
		let t = self.dt as f64;
		let tension = self.config.tension.max(1.0) as f64;
		let mut phantoms = Vec::with_capacity(self.pairs.len());
		for pair in &self.pairs {
			let first = &self.modes[pair.first];
			let second = &self.modes[pair.second];
			let a = first.q as f64;
			let b = second.q as f64;
			let mut amplitude = pair.gain as f64 * a * b / peak as f64;
			let limit = 0.65 * (a.abs() * b.abs()).sqrt();
			amplitude = amplitude.clamp(-limit, limit);
			if amplitude.abs() < 1e-10 {
				continue;
			}
			let frequency = first.frequency as f64 + second.frequency as f64;
			let sigma = first.sigma as f64 + second.sigma as f64;
			let order = frequency / self.config.frequency.max(1.0) as f64;
			let wave_number = order * std::f64::consts::PI / self.length as f64;
			let sign = (order * std::f64::consts::PI).cos();
			let bridge = 0.5 * tension * wave_number * sign;
			let quiet_q = QUIET_FORCE / bridge.abs().max(1e-30) as f32;
			let (a11, b1, c1, d1) = transition(frequency, sigma, t);
			phantoms.push(Mode {
				order: order as f32,
				level: 1.0,
				tail: 1.0,
				quiet_q,
				quiet_v: quiet_q * std::f32::consts::TAU * frequency as f32,
				sigma: sigma as f32,
				frequency: frequency as f32,
				bridge: bridge as f32,
				awake: true,
				q: amplitude as f32,
				a: a11,
				b: b1,
				c: c1,
				d: d1,
				..Default::default()
			});
		}
		self.phantoms = phantoms;
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
		// The note ages even when nothing is touching it: the tail is a
		// property of the string, not of the contact.
		self.age += self.dt as f64;
		// The strike has had time to excite the parents; the phantoms can be
		// read off them now. Waiting past the contact also keeps the hammer's
		// own force out of the sum tones, which is the model's job only.
		if !self.seeded && self.age >= 0.0035 {
			self.seed_phantoms();
		}
		let voicing = &self.config.voicing;
		self.blend = if voicing.late_decay < 1.0 {
			(((self.age - voicing.late_delay.max(0.0) as f64) / voicing.late_ramp.max(1e-3) as f64)
				.clamp(0.0, 1.0)) as f32
		} else {
			0.0
		};

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
		// The scratch vector is reused: a string steps once per sample, and an
		// allocation here would be one per sample per string.
		self.free.clear();
		for port in &self.ports {
			let mut free = 0.0;
			for (mode, weight) in self.modes.iter().zip(port.weights.iter()) {
				free += weight * (mode.c * mode.q + mode.d * mode.v);
			}
			self.free.push(free);
		}

		// Pass two: solve each contact. Ports share the modes but not the
		// solver, exactly as they shared the delay lines before.
		let dt = self.dt;

		if self.shaped_force.len() != self.modes.len() {
			self.shaped_force.resize(self.modes.len(), 0.0);
		}

		self.shaped_force.fill(0.0);

		for (port, free) in self.ports.iter_mut().zip(self.free.iter().copied()) {
			if !port.active {
				continue;
			}
			let force = port.solve(free, port.conductance, dt);
			saxpy(&mut self.shaped_force, &port.weights, force);
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
		// The phantoms have no external force; they ring on what they were
		// given and put their own force on the bridge.
		for mode in &mut self.phantoms {
			if !mode.awake {
				continue;
			}
			let q = mode.q;
			let v = mode.v;
			mode.q = mode.a * q + mode.b * v;
			mode.v = mode.c * q + mode.d * v;
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
			|| (self.blend - self.blend_applied).abs() > BLEND_RESOLUTION
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
		let blend = self.blend as f64;
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
			// A real partial does not keep one decay rate for the whole note.
			// The prompt sound loads the board and the aftersound is what the
			// board gives back, and the two stages are heard as a note that
			// settles rather than one that fades at a single rate. The tail is
			// capped so that a long note cannot relax into a drone.
			let late = if mode.tail >= 1.0 {
				mode.sigma as f64
			} else {
				(mode.sigma as f64 * mode.tail as f64).max(6.907_755 / LATE_T60_MAX)
			};
			let relaxed = mode.sigma as f64 + (late - mode.sigma as f64) * blend;
			let sigma = (relaxed + felt * frequency).min(0.9 * omega);
			let (a, b, c, d) = transition(frequency, sigma, t);
			mode.a = a;
			mode.b = b;
			mode.c = c;
			mode.d = d;
			let damped = (omega * omega - sigma * sigma).max(1e-12).sqrt();
			let decay = (-sigma * t).exp();
			let (sine, cosine) = (damped * t).sin_cos();
			let sigma_over_damped = sigma / damped;

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
		self.blend_applied = self.blend;
		self.since_rebuild = 0;
	}
}
