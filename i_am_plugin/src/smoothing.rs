//! Sample to sample parameter smoothing.
//!
//! A VST3 host driving a plug-in through `clap-wrapper` reports parameter automation
//! once per block: the shim keeps the last automation point of a block and drops the
//! rest. A ramp a user drew in the host's automation lane therefore arrives at the DSP
//! as a staircase, one step per block, and a step in a gain or a cutoff is audible.
//!
//! The wrapper can undo that without asking anything of the DSP. It remembers the value
//! it last handed over, and when the target moves it walks the value the DSP reads over
//! the time the plug-in declares, instead of jumping. The walk is per sample, after the
//! block's parameter events have been applied, because that is the only moment that
//! sees the host's newest target before the DSP does. The DSP still reads a plain
//! float from the same atomic; nothing above this module changes.
//!
//! The cost is a delay on that parameter: what the DSP sees is always a little behind
//! what the host says, by at most the ramp time. That is why a plug-in names the
//! parameters itself (`Plugin::SMOOTHED_PARAMETERS`) instead of smoothing everything:
//! a waveform choice must not lag behind the click that made it.
//!
//! The ramp writes the float inside `AtomicValue` directly rather than through
//! `AtomicValue::store`, so the changed flag stays exactly as the host and the editor
//! left it: the host must not be told about the intermediate values of its own
//! automation, or every ramp would echo back as a burst of parameter changes.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use i_am_dsp::prelude::{AtomicValue, ParamMap};

use crate::plugin::SmoothedParameter;

/// The parameters a plug-in declared, with the state each ramp needs to walk.
///
/// Built once when the plug-in is activated, on the main thread: the audio thread only
/// ever walks the list, and it holds the atomics directly so a tick costs no lookup.
pub(crate) struct Smoothing {
	ramps: Vec<Ramp>,
	/// The map's write counter.
	///
	/// A ramp writes the float the DSP reads without going through the map, so this is
	/// how the parameter sync learns there is something new to read. Bumping it on a
	/// sample that moved is what keeps the sync from scanning every parameter to find
	/// out.
	writes: Arc<AtomicU64>,
}

/// One parameter's ramp.
struct Ramp {
	/// The atomic the DSP reads, held directly: a hash lookup per sample would cost
	/// more than the smoothing saves.
	atomic: Arc<AtomicValue>,
	/// How long a jump takes, in samples.
	samples: u32,
	/// The value this ramp last handed to the DSP.
	value: f32,
	/// The value the parameter is moving towards, as the host or the editor wrote it.
	target: f32,
	/// How much to add per sample while a ramp runs.
	step: f32,
	/// Samples left in the ramp; zero when the value is at its target.
	remaining: u32,
}

impl Smoothing {
	/// Build the ramps for the parameters the plug-in named.
	///
	/// Names that the parameter map does not have, parameters that are not floats, and
	/// ramps shorter than one sample are dropped with a note rather than failing the
	/// activation: a typo in a table should cost a smoothed parameter, not the plug-in.
	pub(crate) fn new(
		map: &ParamMap,
		sample_rate: usize,
		smoothed: &'static [SmoothedParameter],
	) -> Self {
		let writes = map.write_stamp();
		let mut ramps = Vec::with_capacity(smoothed.len());

		for parameter in smoothed {
			let Some(atomic) = map.get(parameter.identifier) else {
				crate::log::note(&format!(
					"smoothing: the parameter map has no `{}`",
					parameter.identifier
				));
				continue;
			};

			let Some(value) = read(&atomic) else {
				crate::log::note(&format!(
					"smoothing: `{}` is not a float",
					parameter.identifier
				));
				continue;
			};

			let samples = (parameter.time_ms / 1000.0 * sample_rate as f32).round();

			if !samples.is_finite() || samples < 1.0 {
				continue;
			}

			ramps.push(Ramp {
				atomic,
				samples: samples as u32,
				value,
				target: value,
				step: 0.0,
				remaining: 0,
			});
		}

		Self { ramps, writes }
	}

	/// Move every ramp one sample forward.
	///
	/// Called after the block's events for this sample have been applied and before the
	/// DSP is allowed to read the parameters.
	pub(crate) fn tick(&mut self) {
		let mut moved = false;

		for ramp in &mut self.ramps {
			moved |= ramp.tick();
		}

		if moved {
			self.writes.fetch_add(1, Ordering::Relaxed);
		}
	}

	/// Adopt whatever the parameters hold now, dropping any ramp in flight.
	///
	/// A reset is a discontinuity by definition: the host starts a new playback, and
	/// gliding from where the last session left a parameter would be wrong.
	pub(crate) fn resync(&mut self) {
		for ramp in &mut self.ramps {
			ramp.resync();
		}

		self.writes.fetch_add(1, Ordering::Relaxed);
	}
}

impl Ramp {
	fn tick(&mut self) -> bool {
		let Some(current) = read(&self.atomic) else {
			return false;
		};

		// A value that is not the one this ramp last handed out was written by
		// somebody else -- the host, or the editor -- and becomes the new target.
		// Comparing for equality is right here: the ramp wrote these exact bits, so
		// anything else is a write, including the host repeating the same automation
		// point on the next block.
		if current != self.value {
			if current.is_nan() {
				return false;
			}

			self.target = current;
			self.remaining = self.samples;
			self.step = (self.target - self.value) / self.samples as f32;
		}

		if self.remaining == 0 {
			return false;
		}

		self.remaining -= 1;
		self.value = if self.remaining == 0 {
			self.target
		} else {
			self.value + self.step
		};

		write(&self.atomic, self.value);

		true
	}

	fn resync(&mut self) {
		let Some(current) = read(&self.atomic) else {
			return;
		};

		// The atomic holds whatever this ramp last wrote, not what the host asked for:
		// the target lives here. Only a write that landed after the last ramp step is
		// worth following, and either way the DSP starts from the target rather than
		// from the middle of a ramp that no longer exists.
		self.target = if current != self.value {
			current
		} else {
			self.target
		};
		self.value = self.target;
		self.step = 0.0;
		self.remaining = 0;

		if current != self.target {
			write(&self.atomic, self.target);
		}
	}
}

fn read(atomic: &AtomicValue) -> Option<f32> {
	match atomic {
		AtomicValue::Float { value, .. } => Some(value.load(Ordering::Relaxed)),
		_ => None,
	}
}

/// Write the number without touching the changed flag.
///
/// `AtomicValue::store` marks a change, which the wrapper reports to the host every
/// block: the host would then hear its own automation echoed back once per sample.
fn write(atomic: &AtomicValue, value: f32) {
	if let AtomicValue::Float { value: target, .. } = atomic {
		target.store(value, Ordering::Relaxed);
	}
}
#[cfg(test)]
mod tests {
	use std::sync::atomic::Ordering;

	use i_am_dsp::prelude::{ParamMap, Paramed, SetValue};
	use i_am_dsp_derive::Parameters;

	use super::{Smoothing, read};
	use crate::plugin::SmoothedParameter;

	#[derive(Parameters)]
	struct Gains {
		#[range(min = 0.0, max = 1.0)]
		gain: f32,
		#[range(min = 0, max = 3)]
		mode: i32,
	}

	fn map() -> ParamMap {
		Paramed::new(Gains { gain: 0.0, mode: 0 }).param_map()
	}

	/// The gain as the DSP would read it.
	fn value(atomic: &i_am_dsp::prelude::AtomicValue) -> f32 {
		read(atomic).expect("the gain is a float")
	}

	const GAIN: &[SmoothedParameter] = &[SmoothedParameter::new("gain", 20.0)];

	/// Twenty milliseconds at 48 kHz is 960 samples, and the jump is spread over every
	/// one of them: the DSP never sees the host's value appear in one step.
	#[test]
	fn a_jump_is_spread_over_the_ramp() {
		let map = map();
		let mut smoothing = Smoothing::new(&map, 48_000, GAIN);
		let atomic = map.get("gain").expect("the gain parameter");

		map.set("gain", SetValue::Float(1.0), Ordering::SeqCst);

		let mut seen = Vec::new();

		for _ in 0..1000 {
			smoothing.tick();
			seen.push(value(&atomic));
		}

		let step = 1.0 / 960.0;
		assert!(
			(seen[0] - step).abs() < 1e-6,
			"the first sample moves one step, not the whole jump: {}",
			seen[0]
		);
		assert_eq!(seen[959], 1.0, "the 960th sample lands on the target");
		assert_eq!(seen[999], 1.0, "and it stays there");

		// Every step is the same size, which is what makes it inaudible. The
		// tolerance is a whole ulp's worth of drift: 960 additions in f32 accumulate
		// about 1e-4 over the ramp, and that is still a step per sample, not a jump.
		for pair in seen[..960].windows(2) {
			assert!((pair[1] - pair[0] - step).abs() < 2e-4);
		}
	}

	/// A ramp writes the float the DSP reads without going through the map, so it has
	/// to say the map moved: otherwise the sync skips the only write of the block.
	#[test]
	fn a_ramp_announces_the_value_it_wrote() {
		let map = map();
		let mut smoothing = Smoothing::new(&map, 48_000, GAIN);

		map.set("gain", SetValue::Float(1.0), Ordering::SeqCst);
		let written = map.writes();

		smoothing.tick();

		assert!(
			map.writes() > written,
			"a ramp step has to be announced to the sync"
		);
	}

	/// A new target in the middle of a ramp starts from where the ramp got to.
	#[test]
	fn a_second_target_continues_from_the_ramp() {
		let map = map();
		let mut smoothing = Smoothing::new(&map, 48_000, GAIN);
		let atomic = map.get("gain").expect("the gain parameter");

		map.set("gain", SetValue::Float(1.0), Ordering::SeqCst);
		for _ in 0..480 {
			smoothing.tick();
		}

		let halfway = value(&atomic);
		assert!(
			(halfway - 0.5).abs() < 1e-3,
			"half a ramp is half the value: {halfway}"
		);

		map.set("gain", SetValue::Float(0.0), Ordering::SeqCst);
		smoothing.tick();

		// One step down from halfway, not from one.
		let after = value(&atomic);
		assert!(after < halfway && after > halfway - 1.0 / 960.0 - 1e-6);
	}

	/// Writing the value must not raise the changed flag, or the host hears its own
	/// automation echoed back once per sample.
	#[test]
	fn the_ramp_does_not_report_itself_as_a_change() {
		let map = map();
		let mut smoothing = Smoothing::new(&map, 48_000, GAIN);
		let atomic = map.get("gain").expect("the gain parameter");

		map.set("gain", SetValue::Float(1.0), Ordering::SeqCst);
		assert!(atomic.take_changed(Ordering::SeqCst));

		for _ in 0..960 {
			smoothing.tick();
			assert!(
				!atomic.take_changed(Ordering::SeqCst),
				"a ramp step is not a parameter change"
			);
		}
	}

	/// Only floats are ramped: a switch that lagged behind the click that made it would
	/// be a bug, and a table entry that names one is dropped instead.
	#[test]
	fn a_discrete_parameter_is_left_alone() {
		let map = map();
		const ENTRIES: &[SmoothedParameter] = &[
			SmoothedParameter::new("mode", 20.0),
			SmoothedParameter::new("missing", 20.0),
		];
		let mut smoothing = Smoothing::new(&map, 48_000, ENTRIES);

		map.set("mode", SetValue::Int(2), Ordering::SeqCst);
		smoothing.tick();

		assert_eq!(
			map.get("mode")
				.expect("the mode parameter")
				.load(Ordering::Relaxed),
			SetValue::Int(2)
		);
	}

	/// A reset drops the ramp and follows the parameters where they are.
	#[test]
	fn a_reset_takes_the_value_it_finds() {
		let map = map();
		let mut smoothing = Smoothing::new(&map, 48_000, GAIN);
		let atomic = map.get("gain").expect("the gain parameter");

		map.set("gain", SetValue::Float(1.0), Ordering::SeqCst);
		smoothing.tick();
		assert!(value(&atomic) < 1.0);

		smoothing.resync();
		assert_eq!(value(&atomic), 1.0);

		// And no stale ramp runs afterwards.
		smoothing.tick();
		assert_eq!(value(&atomic), 1.0);
	}

	/// A ramp that is not a whole sample long is no ramp at all.
	#[test]
	fn a_ramp_shorter_than_a_sample_is_dropped() {
		let map = map();
		const ENTRIES: &[SmoothedParameter] = &[SmoothedParameter::new("gain", 0.0)];
		let mut smoothing = Smoothing::new(&map, 48_000, ENTRIES);
		let atomic = map.get("gain").expect("the gain parameter");

		map.set("gain", SetValue::Float(1.0), Ordering::SeqCst);
		smoothing.tick();

		assert_eq!(value(&atomic), 1.0, "the value is taken as it comes");
	}
}
