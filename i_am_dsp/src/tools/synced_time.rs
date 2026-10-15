//! A time parameter that can follow the host's tempo instead of a fixed number of milliseconds.

use crate::ProcessInfos;

/// The musical half of a time parameter: a switch and a length in beats.
///
/// The free value — the milliseconds or hertz the effect used before — deliberately stays a
/// field of the effect, so each effect keeps its own range and its own identifier. Adopt this
/// next to it:
///
/// ```
/// use i_am_dsp::prelude::{Parameters, SyncedTime};
/// use i_am_dsp_derive::Parameters;
///
/// #[derive(Parameters)]
/// struct Delay {
///     /// `delay.sync` and `delay.beats`.
///     #[sub_param]
///     delay: SyncedTime,
///     /// `delay_time`, unchanged: a project that automated it keeps working.
///     #[range(min = 0.0, max = 4000.0)]
///     delay_time: f32,
/// }
/// ```
///
/// and then ask for the length instead of reading `delay_time` directly:
///
/// ```
/// # use i_am_dsp::prelude::{ProcessInfos, SyncedTime};
/// # let (delay, delay_time, sample_rate) = (SyncedTime::default(), 250.0, 48_000);
/// # let infos = &ProcessInfos::default();
/// let samples = delay.samples(infos, delay_time, sample_rate);
/// ```
///
/// # When the range is wrong for an effect
///
/// `beats` carries its range with it, so every effect that embeds it shares that range. For a
/// musical length that is not much of a limit — a thirty-second note to thirty-two beats
/// covers anything anyone asks a delay, an envelope or an LFO for — but an effect that does
/// want its own limits does not have to embed it: the conversion is a method on public fields,
/// so the parameters can be declared flat and a `SyncedTime` built from them.
///
/// ```
/// # use i_am_dsp::prelude::{Parameters, SyncedTime};
/// # use i_am_dsp_derive::Parameters;
/// #[derive(Parameters)]
/// struct Delay {
///     sync: bool,
///     // The effect's own range, not the shared one.
///     #[range(min = 0.03125, max = 4.0)]
///     #[logarithmic]
///     beats: f32,
///     #[range(min = 0.0, max = 4000.0)]
///     delay_time: f32,
/// }
///
/// # let delay = Delay { sync: true, beats: 0.5, delay_time: 999.0 };
/// # let mut infos = i_am_dsp::ProcessInfos::default();
/// # infos.tempo = Some(120.0);
/// let time = SyncedTime { sync: delay.sync, beats: delay.beats };
/// // 12_000 samples: an eighth note at 120 bpm, not the 999 ms free value.
/// assert_eq!(time.samples(&infos, delay.delay_time, 48_000), 12_000);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, i_am_dsp_derive::Parameters)]
pub struct SyncedTime {
	/// Whether to follow the host's tempo instead of the effect's own free value.
	#[default(false)]
	pub sync: bool,
	/// The length in quarter-note beats: `0.5` is an eighth note, `1.0` a quarter, `4.0` a
	/// bar in four-four. Dotted values are multiplied by `1.5` and triplets by `2/3`.
	#[range(min = 0.03125, max = 32.0)]
	#[logarithmic]
	#[default(1.0)]
	#[unit(beats)]
	pub beats: f32,
}

impl Default for SyncedTime {
	/// Off, with a length of one beat for when it is switched on.
	fn default() -> Self {
		Self { sync: false, beats: 1.0 }
	}
}

impl SyncedTime {
	/// A time that is already syncing, with the given length in beats.
	pub fn synced(beats: f32) -> Self {
		Self { sync: true, beats }
	}

	/// The musical value, when there is one to use: `(beats, tempo)`.
	///
	/// This is the single place the decision is made. Everything else derives from it, so a
	/// caller cannot end up with a `seconds` that syncs and a `rate_hz` that does not.
	fn musical(&self, infos: &ProcessInfos) -> Option<(f64, f64)> {
		if !self.sync {
			return None;
		}

		let tempo = infos.tempo.filter(|tempo| *tempo > 0.0)?;
		Some((self.beats as f64, tempo as f64))
	}

	/// Whether the musical value applies: syncing is on *and* the host reports a real tempo.
	///
	/// A host with no tempo (or a tempo of zero) is not something to sync to, so this is
	/// [`false`] and the free value is used instead — never a guessed 120 bpm.
	pub fn is_synced(&self, infos: &ProcessInfos) -> bool {
		self.musical(infos).is_some()
	}

	/// The length, in seconds.
	///
	/// Depends on the tempo only, never on the song position: an eighth note at 120 bpm is
	/// 250 ms whether the transport is rolling or stopped, so a stopped host does not make
	/// this stale — it makes it exact.
	///
	/// `free_ms` is the effect's own value, used when syncing is off or there is no tempo.
	pub fn seconds(&self, infos: &ProcessInfos, free_ms: f32) -> f32 {
		match self.musical(infos) {
			Some((beats, tempo)) => (beats * 60.0 / tempo) as f32,
			None => free_ms / 1000.0,
		}
	}

	/// The length, in samples, rounded to the nearest whole sample.
	pub fn samples(&self, infos: &ProcessInfos, free_ms: f32, sample_rate: usize) -> usize {
		(self.seconds(infos, free_ms) * sample_rate as f32).round().max(0.0) as usize
	}

	/// The rate, in cycles per second, for something that is measured as a frequency rather
	/// than a length — an LFO, a tremolo, a modulation depth.
	///
	/// `free_hz` is the effect's own value, used on the same terms as `free_ms` above.
	pub fn rate_hz(&self, infos: &ProcessInfos, free_hz: f32) -> f32 {
		match self.musical(infos) {
			Some((beats, tempo)) => (tempo / (60.0 * beats)) as f32,
			None => free_hz,
		}
	}

	/// How far the musical grid has moved, counted in whole cycles of this parameter.
	///
	/// This is for an effect that has to be *in phase* with the grid rather than merely run at
	/// the right speed: take the fraction of it for a phase, or the whole number for a
	/// quantized step. It reads [`ProcessInfos::position_beats`], which the host is the
	/// authority on, and it follows a seek or a loop because that moves the position.
	///
	/// [`None`] means there is no grid to be in phase with: syncing is off, or the host
	/// reports no beat timeline. Note that a *stopped* transport still reports a position, so
	/// the grid keeps its value and this keeps returning the same cycle — an effect that wants
	/// to stay alive while stopped wants [`SyncedPhase`], which free-runs in that case.
	pub fn cycles(&self, infos: &ProcessInfos) -> Option<f64> {
		if !self.sync {
			return None;
		}

		let beats_per_cycle = self.beats as f64;
		if beats_per_cycle <= 0.0 {
			return None;
		}

		infos.position_beats.map(|position| position / beats_per_cycle)
	}
}

/// A phase that stays on the host's grid and keeps running when the host stops.
///
/// An effect that only needs a rate calls [`SyncedTime::seconds`] or [`SyncedTime::rate_hz`]
/// and keeps no state. An effect that has to be in phase with the grid needs a phase, and that
/// is where the host alone is not enough: while the transport is stopped the grid does not
/// move, so following it would freeze the effect exactly when a musician is still playing with
/// it. This keeps a phase of its own for that case:
///
/// - transport running and the host reporting a position → the phase is the host's, so it
///   follows a seek, a loop and the tempo;
/// - transport stopped, or no position, or syncing off → the phase free-runs one sample at a
///   time at the tempo's rate.
///
/// It is a separate component so the choice is the caller's: an effect that would rather lock
/// up while stopped simply does not keep one.
///
/// The phase *jumps* when the host takes over again (the transport starting, a seek, a loop
/// wrap). That is what being in phase means, but if the jump is audible on the parameter it
/// drives, smoothing it is the effect's business.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SyncedPhase {
	phase: f64,
}

impl SyncedPhase {
	/// A phase at the start of a cycle.
	pub const fn new() -> Self {
		Self { phase: 0.0 }
	}

	/// The phase, in `0.0..1.0`.
	pub const fn phase(&self) -> f64 {
		self.phase
	}

	/// Back to the start of a cycle, for a transport restart or a panic button.
	pub fn reset(&mut self) {
		self.phase = 0.0;
	}

	/// Returns the phase of the sample being rendered, in `0.0..1.0`, and moves the phase on
	/// for the next one.
	///
	/// Call it once per sample, like the effect it belongs to. `free_ms` is the effect's own
	/// value, used on the same terms as [`SyncedTime::seconds`].
	pub fn advance(
		&mut self,
		time: &SyncedTime,
		infos: &ProcessInfos,
		free_ms: f32,
		sample_rate: usize,
	) -> f64 {
		if let Some(grid_phase) = self.grid_phase(time, infos) {
			self.phase = grid_phase;
			return self.phase;
		}

		// Free-running: this sample keeps the phase it has, and the next one moves on. The
		// tempo still sets the rate, so a stopped transport keeps the effect moving at the
		// right musical speed rather than at some guessed one.
		let phase = self.phase;
		let period = time.seconds(infos, free_ms) as f64;
		if period > 0.0 && sample_rate > 0 {
			self.phase = (self.phase + 1.0 / (period * sample_rate as f64)).rem_euclid(1.0);
		}

		phase
	}

	/// Where the host says this cycle is, while the transport is actually moving.
	///
	/// The `playing` check is the point: a stopped host still reports the position it stopped
	/// at, and locking to that would freeze the phase.
	fn grid_phase(&self, time: &SyncedTime, infos: &ProcessInfos) -> Option<f64> {
		if !infos.playing {
			return None;
		}

		time.cycles(infos).map(|cycles| cycles.rem_euclid(1.0))
	}
}
#[cfg(test)]
mod tests {
	use super::*;
	use crate::prelude::{Parameters, SetValue, Unit, Value};
	use i_am_dsp_derive::Parameters;

	/// A process with the pieces a synced effect reads.
	fn infos(tempo: Option<f32>, position: Option<f64>, playing: bool) -> ProcessInfos {
		ProcessInfos {
			sample_rate: 48_000,
			trustable: true,
			tempo,
			position_beats: position,
			playing,
			..ProcessInfos::default()
		}
	}

	#[test]
	fn a_free_time_ignores_the_tempo() {
		let time = SyncedTime { sync: false, beats: 0.5 };
		let host = infos(Some(120.0), Some(8.0), true);

		assert!(!time.is_synced(&host));
		assert_eq!(time.seconds(&host, 250.0), 0.25);
		assert_eq!(time.rate_hz(&host, 3.0), 3.0);
	}

	#[test]
	fn a_synced_time_follows_the_tempo() {
		let time = SyncedTime { sync: true, beats: 0.5 }; // an eighth note

		// At 120 bpm a beat is half a second, so an eighth note is 250 ms.
		assert_eq!(time.seconds(&infos(Some(120.0), None, true), 999.0), 0.25);
		// Half the tempo, twice the length — and the free value is never consulted.
		assert_eq!(time.seconds(&infos(Some(60.0), None, true), 999.0), 0.5);
		assert_eq!(time.samples(&infos(Some(120.0), None, true), 999.0, 48_000), 12_000);
	}

	#[test]
	fn a_synced_time_without_a_tempo_falls_back_to_the_free_value() {
		let time = SyncedTime { sync: true, beats: 0.5 };

		assert!(!time.is_synced(&infos(None, Some(8.0), true)));
		assert_eq!(time.seconds(&infos(None, Some(8.0), true), 250.0), 0.25);
		// A tempo of zero is not a tempo either; nothing falls back to a guessed 120.
		assert_eq!(time.seconds(&infos(Some(0.0), Some(8.0), true), 250.0), 0.25);
	}

	#[test]
	fn a_synced_time_reports_a_rate() {
		// One cycle every two beats: at 120 bpm that is one cycle per second.
		let time = SyncedTime { sync: true, beats: 2.0 };
		assert_eq!(time.rate_hz(&infos(Some(120.0), None, true), 7.0), 1.0);
		// No tempo: the effect's own rate stands.
		assert_eq!(time.rate_hz(&infos(None, None, true), 7.0), 7.0);
	}

	#[test]
	fn cycles_need_syncing_and_a_position() {
		let time = SyncedTime { sync: true, beats: 2.0 };

		assert_eq!(time.cycles(&infos(Some(120.0), Some(5.0), true)), Some(2.5));
		assert_eq!(time.cycles(&infos(Some(120.0), None, true)), None);

		// Sync off means the parameter is not musical, so there is no grid to be in phase
		// with even when the host reports everything.
		let free = SyncedTime { sync: false, beats: 2.0 };
		assert_eq!(free.cycles(&infos(Some(120.0), Some(5.0), true)), None);
	}

	#[test]
	fn the_phase_follows_the_host_grid_while_the_transport_runs() {
		let time = SyncedTime { sync: true, beats: 1.0 };
		let mut phase = SyncedPhase::new();

		// Beat 4.25 is a quarter of the way through the cycle that starts at beat 4.
		assert_eq!(phase.advance(&time, &infos(Some(120.0), Some(4.25), true), 500.0, 48_000), 0.25);
		// A seek moves the grid, and the phase with it.
		assert_eq!(phase.advance(&time, &infos(Some(120.0), Some(2.75), true), 500.0, 48_000), 0.75);
	}

	#[test]
	fn the_phase_keeps_running_while_the_transport_is_stopped() {
		// A cycle per eighth note: 250 ms at 120 bpm, which is 12_000 samples at 48 kHz.
		let time = SyncedTime { sync: true, beats: 0.5 };
		let stopped = infos(Some(120.0), Some(4.25), false);
		let mut phase = SyncedPhase::new();

		assert_eq!(phase.advance(&time, &stopped, 500.0, 48_000), 0.0);
		// 6_000 samples in total is half of the 12_000 the cycle takes.
		for _ in 0..5_999 {
			phase.advance(&time, &stopped, 500.0, 48_000);
		}

		// Half of the cycle, and not the frozen quarter the host is still reporting.
		assert!((phase.phase() - 0.5).abs() < 1e-9, "phase was {}", phase.phase());
	}

	#[test]
	fn the_phase_free_runs_without_a_position_or_without_syncing() {
		let time = SyncedTime { sync: true, beats: 1.0 };
		let mut phase = SyncedPhase::new();

		// Playing, but the host reports no beat timeline: 500 ms per cycle is 24_000 samples.
		let no_position = infos(Some(120.0), None, true);
		for _ in 0..12_000 {
			phase.advance(&time, &no_position, 500.0, 48_000);
		}
		assert!((phase.phase() - 0.5).abs() < 1e-9, "phase was {}", phase.phase());

		// Syncing off: the effect's own length drives it, 250 ms means 12_000 samples.
		let free = SyncedTime { sync: false, beats: 1.0 };
		let mut phase = SyncedPhase::new();
		for _ in 0..6_000 {
			phase.advance(&free, &infos(Some(120.0), Some(4.25), true), 250.0, 48_000);
		}
		assert!((phase.phase() - 0.5).abs() < 1e-9, "phase was {}", phase.phase());
	}

	#[test]
	fn resetting_the_phase_starts_the_cycle_again() {
		let time = SyncedTime { sync: true, beats: 0.5 };
		let stopped = infos(Some(120.0), Some(0.0), false);
		let mut phase = SyncedPhase::new();

		phase.advance(&time, &stopped, 500.0, 48_000);
		phase.reset();
		assert_eq!(phase.phase(), 0.0);
	}

	/// A time parameter sits next to the effect's own free value, and the identifiers do not
	/// collide: the effect keeps `delay_time`, which is what a project may already automate.
	#[derive(Parameters, Default)]
	struct Delay {
		#[sub_param]
		delay: SyncedTime,
		#[range(min = 0.0, max = 4000.0)]
		delay_time: f32,
	}

	/// The conversion is a method on public fields, so an effect that wants its own ranges can
	/// declare the parameters itself and still reuse it.
	#[derive(Parameters, Default)]
	struct FlatDelay {
		sync: bool,
		#[range(min = 0.03125, max = 4.0)]
		#[logarithmic]
		beats: f32,
		#[range(min = 0.0, max = 4000.0)]
		delay_time: f32,
	}

	#[test]
	fn an_effect_can_keep_its_own_ranges_and_identifiers() {
		let mut delay = FlatDelay::default();
		assert!(delay.set_parameter("sync", SetValue::Bool(true)));
		assert!(delay.set_parameter("beats", SetValue::Float(0.5)));

		let parameters = delay.get_parameters();
		let identifiers = parameters
			.iter()
			.map(|parameter| parameter.identifier.as_str())
			.collect::<Vec<_>>();
		assert_eq!(identifiers, ["sync", "beats", "delay_time"]);

		// Four beats, not the thirty-two the shared struct allows.
		match &parameters[1].value {
			Value::Float { range, .. } => assert_eq!((*range.start(), *range.end()), (0.03125, 4.0)),
			other => panic!("expected a float parameter, got {other:?}"),
		}

		let time = SyncedTime { sync: delay.sync, beats: delay.beats };
		assert_eq!(time.samples(&infos(Some(120.0), None, true), delay.delay_time, 48_000), 12_000);
	}

	#[test]
	fn a_synced_time_nests_beside_the_free_value() {
		let mut delay = Delay::default();
		let identifiers = delay
			.get_parameters()
			.into_iter()
			.map(|parameter| parameter.identifier)
			.collect::<Vec<_>>();
		assert_eq!(identifiers, ["delay.sync", "delay.beats", "delay_time"]);

		// The metadata a host presents them by survives the nesting, which is what turns the
		// beats slider into `1/8` and its starting value into one beat rather than a 128th.
		let parameters = delay.get_parameters();
		let beats = parameters
			.iter()
			.find(|parameter| parameter.identifier == "delay.beats")
			.expect("the nested beats parameter exists");
		assert_eq!(beats.meta.default, Some(1.0));
		assert_eq!(beats.meta.unit, Unit::Beats);

		assert!(delay.set_parameter("delay.sync", SetValue::Bool(true)));
		assert!(delay.set_parameter("delay.beats", SetValue::Float(0.5)));
		assert!(delay.delay.sync);
		assert_eq!(delay.delay.beats, 0.5);
	}
}