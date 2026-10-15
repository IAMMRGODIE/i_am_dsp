//! A sample-accurate musical clock, anchored to the transport a host reports.

/// Where the timeline was pinned, and how fast it runs from there.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Anchor {
	/// The absolute sample the host's position was read at.
	sample: u64,
	/// The song position, in quarter-note beats, at that sample.
	beats: f64,
	/// The tempo at that sample, in beats per minute.
	tempo: f64,
	/// How much the tempo changes per sample, in beats per minute.
	tempo_inc: f64,
}

/// The musical timeline, advanced sample by sample.
///
/// The host owns the timeline, so the clock never free-runs on its own: every transport
/// report re-anchors it, and between reports it interpolates from the anchor. That is what
/// makes the position sample-accurate *and* what keeps a seek, a loop wrap or a drift in the
/// host's block size from accumulating error — the next report corrects it.
///
/// Callers pass an absolute sample count, not an offset within the block, so a host that
/// reports the transport once and then stops repeating it keeps a smooth position instead of
/// restarting from zero every block.
///
/// ```
/// use i_am_dsp::prelude::BeatClock;
///
/// let mut clock = BeatClock::new(48_000);
/// // The host says: beat 8, 120 bpm, rolling, at absolute sample 0.
/// clock.anchor(0, 8.0, 120.0, 0.0);
/// clock.set_running(true);
/// assert_eq!(clock.advance_to(0), Some(8.0));
/// // At 120 bpm a beat is half a second, so 24_000 samples later it is beat 9.
/// assert_eq!(clock.advance_to(24_000), Some(9.0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeatClock {
	sample_rate: usize,
	/// Whether the transport is rolling. A stopped transport holds the position still.
	running: bool,
	anchor: Option<Anchor>,
	position: f64,
	/// The last sample [`Self::advance_to`] was asked about.
	///
	/// Starting the transport again measures from here, so the samples that went by while
	/// it was stopped are not counted as music.
	last_sample: u64,
}

impl BeatClock {
	/// Creates a clock with no timeline. It reports no position until it is anchored.
	pub fn new(sample_rate: usize) -> Self {
		Self {
			sample_rate: sample_rate.max(1),
			running: false,
			anchor: None,
			position: 0.0,
			last_sample: 0,
		}
	}

	/// The sample rate the clock converts samples with.
	pub fn sample_rate(&self) -> usize {
		self.sample_rate
	}

	/// Re-anchors the timeline to the position the host reports.
	///
	/// `sample` is an absolute sample count, `beats` is the song position there in
	/// quarter-note beats, `tempo` is in beats per minute, and `tempo_inc` is how much the
	/// tempo changes per sample (zero for a steady tempo).
	///
	/// Any previous anchor is dropped: the host's report always wins, even when it moves
	/// backwards, because that is exactly what a seek or a loop wrap looks like.
	pub fn anchor(&mut self, sample: u64, beats: f64, tempo: f64, tempo_inc: f64) {
		self.position = beats;
		self.anchor = Some(Anchor { sample, beats, tempo, tempo_inc });
	}

	/// Sets whether the transport is rolling.
	///
	/// A stopped transport holds the position where it is rather than gliding forward: the
	/// host may keep reporting a position while the user is stopped, and a sync'd effect
	/// should sit on that position, not run away from it.
	///
	/// No music passes while stopped, so starting again continues from where the position was
	/// left: without that, a two minute pause would be counted as two minutes of beats and the
	/// position would leap forward the moment the host hits play.
	pub fn set_running(&mut self, running: bool) {
		if running && !self.running && let Some(anchor) = self.anchor {
			self.anchor = Some(Anchor {
				sample: self.last_sample,
				beats: self.position,
				tempo: anchor.tempo,
				tempo_inc: anchor.tempo_inc,
			});
		}

		self.running = running;
	}

	/// Whether the transport is rolling.
	pub fn running(&self) -> bool {
		self.running
	}

	/// Forgets the timeline, so the clock reports no position until it is anchored again.
	///
	/// Used when a host reports a tempo or a transport state but no beat timeline at all:
	/// there is then nothing to be in phase with.
	pub fn clear(&mut self) {
		self.anchor = None;
	}

	/// Returns the position at `sample` (an absolute sample count) and remembers it.
	///
	/// Returns [`None`] while no anchor has been seen — a context with no musical timeline
	/// has nothing to sync to. While the transport is stopped the last position is returned
	/// unchanged.
	pub fn advance_to(&mut self, sample: u64) -> Option<f64> {
		self.last_sample = sample;

		let anchor = self.anchor?;

		if self.running {
			let elapsed = sample.saturating_sub(anchor.sample) as f64;
			// Over `elapsed` samples the tempo walks from `tempo` to `tempo + elapsed *
			// tempo_inc`, so the beats it covered are the sum of those rates:
			// `elapsed * tempo + tempo_inc * elapsed * (elapsed - 1) / 2`.
			let beats = anchor.beats
				+ (anchor.tempo * elapsed + anchor.tempo_inc * elapsed * (elapsed - 1.0) / 2.0)
					/ (60.0 * self.sample_rate as f64);
			self.position = beats;
		}

		Some(self.position)
	}

	/// The position at the last [`Self::advance_to`], or [`None`] with no anchor.
	pub fn position(&self) -> Option<f64> {
		self.anchor.is_some().then_some(self.position)
	}

	/// Advances a timeline that has no host behind it, and returns the position at `sample`.
	///
	/// This is the free-running counterpart to [`Self::anchor`]: with nobody to re-anchor
	/// from, the clock is its own authority. It keeps its position and only re-anchors when
	/// the rate it is driven with changes, at the sample the change is noticed — which is
	/// what makes a local grid one call per sample instead of a hand-rolled accumulator.
	///
	/// The first call sets the origin: with no history to measure from, the position is zero
	/// at that sample and counts from there. Everything after it is plain interpolation from
	/// the absolute sample count.
	///
	/// A local timeline is always running, so this leaves the clock running: stopping it is
	/// [`Self::clear`], which forgets the timeline entirely. Drive a host-owned timeline with
	/// [`Self::anchor`] and [`Self::advance_to`] instead.
	pub fn advance_local(&mut self, sample: u64, tempo: f64) -> f64 {
		match self.anchor {
			Some(anchor) if anchor.tempo == tempo && anchor.tempo_inc == 0.0 => {},
			_ => {
				// Catch up to `sample` at the rate that has held so far, then change the rate
				// there. Re-anchoring at the last sample anyone asked about would move the
				// change backwards in time.
				let position = self.advance_to(sample).unwrap_or(0.0);
				self.anchor(sample, position, tempo, 0.0);
			},
		}

		self.running = true;
		self.advance_to(sample).unwrap_or(0.0)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// 120 bpm at 48 kHz: one beat is 24_000 samples.
	fn rolling_clock() -> BeatClock {
		let mut clock = BeatClock::new(48_000);
		clock.anchor(0, 8.0, 120.0, 0.0);
		clock.set_running(true);
		clock
	}

	#[test]
	fn a_steady_tempo_advances_one_beat_per_beat() {
		let mut clock = rolling_clock();

		assert_eq!(clock.advance_to(0), Some(8.0));
		assert_eq!(clock.advance_to(24_000), Some(9.0));
		assert_eq!(clock.advance_to(12_000), Some(8.5));
		// Long sessions keep their precision: this is `f64` beats, not `f32`.
		assert_eq!(clock.advance_to(24_000 * 100_000), Some(100_008.0));
	}

	#[test]
	fn a_stopped_transport_holds_the_position() {
		let mut clock = rolling_clock();
		assert_eq!(clock.advance_to(0), Some(8.0));

		clock.set_running(false);
		assert_eq!(clock.advance_to(48_000), Some(8.0));
		assert_eq!(clock.advance_to(96_000), Some(8.0));

		// Starting again continues from where the position was left: the samples that went by
		// while stopped were not music, so one beat after the last frozen sample.
		clock.set_running(true);
		assert_eq!(clock.advance_to(120_000), Some(9.0));
	}

	#[test]
	fn a_tempo_ramp_accumulates_the_rates_it_passed_through() {
		let mut clock = BeatClock::new(48_000);
		// 120 bpm, gaining 1 bpm per sample: after 48_000 samples the tempo is 48_120 bpm,
		// which is absurd but makes the arithmetic easy to check against a hand sum.
		clock.anchor(0, 0.0, 120.0, 1.0);
		clock.set_running(true);

		let expected = (0..1000)
			.map(|k| (120.0 + k as f64) / (60.0 * 48_000.0))
			.sum::<f64>();
		let position = clock.advance_to(1_000).unwrap();
		assert!((position - expected).abs() < 1e-12, "{position} != {expected}");
	}

	#[test]
	fn an_unanchored_clock_reports_nothing() {
		let mut clock = BeatClock::new(48_000);
		assert_eq!(clock.advance_to(0), None);
		assert_eq!(clock.position(), None);

		clock.set_running(true);
		assert_eq!(clock.advance_to(48_000), None);
	}

	#[test]
	fn clearing_forgets_the_timeline() {
		let mut clock = rolling_clock();
		assert_eq!(clock.advance_to(0), Some(8.0));

		clock.clear();
		assert_eq!(clock.advance_to(24_000), None);
		assert_eq!(clock.position(), None);
	}

	/// A local clock starts at zero, runs at the rate it is driven with, and keeps its
	/// position across a change of rate instead of restarting.
	#[test]
	fn a_local_clock_runs_on_its_own_tempo() {
		let mut clock = BeatClock::new(48_000);

		// 120 bpm from sample 0: a beat is 24_000 samples.
		assert_eq!(clock.advance_local(0, 120.0), 0.0);
		assert_eq!(clock.advance_local(24_000, 120.0), 1.0);

		// 240 bpm from here: a beat is 12_000 samples, and the position does not jump.
		assert_eq!(clock.advance_local(24_000, 240.0), 1.0);
		assert_eq!(clock.advance_local(36_000, 240.0), 2.0);

		// It runs even though nothing ever said the transport started.
		assert!(clock.running());
	}

	#[test]
	fn a_seek_moves_the_position_backwards() {
		let mut clock = rolling_clock();
		assert_eq!(clock.advance_to(24_000), Some(9.0));

		// The host jumps to beat 2 and says so at sample 30_000.
		clock.anchor(30_000, 2.0, 120.0, 0.0);
		assert_eq!(clock.advance_to(30_000), Some(2.0));
		assert_eq!(clock.advance_to(54_000), Some(3.0));
	}
}