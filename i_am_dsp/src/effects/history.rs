//! A curve that replays the recent past at a mapped speed: the y axis is the
//! playback position in the history window, the x axis is where the current
//! moment sits inside that window.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use i_am_dsp_derive::Parameters;

use crate::prelude::SyncedTime;
use crate::{Effect, ProcessContext, ProcessInfos};

/// One breakpoint of a history page.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HistoryPoint {
	/// Position in the history window, 0 is the oldest sample, 1 the newest.
	pub time: f32,
	/// Playback position at that point: 1 is now, 0 the far edge of the window.
	pub value: f32,
	/// Bend of the segment that starts here.
	pub bend: f32,
}

/// One page of the time map.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HistoryPage {
	/// The page's breakpoints, sorted by `time`.
	pub points: Vec<HistoryPoint>,
}

impl HistoryPage {
	/// A page that plays the live input unchanged.
	pub fn flat() -> Self {
		Self {
			points: vec![
				HistoryPoint {
					time: 0.0,
					value: 1.0,
					bend: 0.0,
				},
				HistoryPoint {
					time: 1.0,
					value: 1.0,
					bend: 0.0,
				},
			],
		}
	}
}

impl Default for HistoryPage {
	fn default() -> Self {
		Self::flat()
	}
}

/// The pages the page edits and the effect reads, shared by the node's voices.
pub struct HistoryFeed {
	generation: AtomicU64,
	pages: Mutex<Arc<Vec<HistoryPage>>>,
}

impl HistoryFeed {
	/// A feed with one flat page.
	pub fn new() -> Arc<Self> {
		Arc::new(Self {
			generation: AtomicU64::new(0),
			pages: Mutex::new(Arc::new(vec![HistoryPage::flat()])),
		})
	}

	/// The current pages.
	pub fn pages(&self) -> Arc<Vec<HistoryPage>> {
		self.pages
			.lock()
			.unwrap_or_else(|error| error.into_inner())
			.clone()
	}

	/// Replaces the pages and moves the generation.
	pub fn set_pages(&self, pages: Vec<HistoryPage>) {
		let pages = if pages.is_empty() {
			vec![HistoryPage::flat()]
		} else {
			pages
		};
		*self.pages.lock().unwrap_or_else(|error| error.into_inner()) = Arc::new(pages);
		self.generation.fetch_add(1, Ordering::Release);
	}

	/// Moves whenever the pages change.
	pub fn generation(&self) -> u64 {
		self.generation.load(Ordering::Acquire)
	}
}

impl Default for HistoryFeed {
	fn default() -> Self {
		Self {
			generation: AtomicU64::new(0),
			pages: Mutex::new(Arc::new(vec![HistoryPage::flat()])),
		}
	}
}

/// The curve resolution the effect carries.
const TABLE: usize = 1024;

/// How long a history the ring keeps, in seconds. A beat length longer than this
/// is clamped rather than growing the allocation without bound.
const MAX_HISTORY_SECONDS: f32 = 8.0;

/// A time remap over a ring of recent input.
#[derive(Parameters)]
pub struct HistoryEffect {
	/// The length of the history window, in beats or its own milliseconds.
	#[sub_param]
	#[id(display = "History")]
	#[layout(group = "History", order = 0)]
	pub time: SyncedTime,
	/// The free length in milliseconds, used when Sync is off or no tempo is known.
	#[range(min = 10.0, max = 8000.0)]
	#[unit(milliseconds)]
	#[default(1000.0)]
	#[id(display = "History")]
	#[layout(group = "History", order = 1, widget = knob)]
	pub history_time: f32,
	/// Which page the curve plays, interpolated between pages.
	#[range(min = 0.0, max = 1.0)]
	#[default(0.0)]
	#[id(display = "Morph")]
	#[layout(group = "History", order = 2, widget = knob, hero = "history")]
	pub morph: f32,
	/// Dry/wet blend.
	#[range(min = 0.0, max = 1.0)]
	#[default(1.0)]
	#[id(display = "Mix")]
	#[layout(group = "History", order = 3, widget = knob)]
	pub mix: f32,
	#[skip]
	feed: Arc<HistoryFeed>,
	#[skip]
	sample_rate: f32,
	#[skip]
	ring: [Vec<f32>; 2],
	#[skip]
	write: usize,
	#[skip]
	phase: f64,
	#[skip]
	table: [f32; TABLE],
	#[skip]
	seen_pages: u64,
	#[skip]
	seen_morph: f32,
	#[skip]
	smoothed_delay: f64,
	#[skip]
	delay_ready: bool,
}

impl HistoryEffect {
	/// A history effect over a feed the page owns.
	pub fn with_feed(sample_rate: usize, feed: Arc<HistoryFeed>) -> Self {
		let length = ((sample_rate as f32 * MAX_HISTORY_SECONDS).ceil() as usize + 2).max(3);
		Self {
			time: SyncedTime::default(),
			history_time: 1000.0,
			morph: 0.0,
			mix: 1.0,
			feed,
			sample_rate: sample_rate as f32,
			ring: core::array::from_fn(|_| vec![0.0; length]),
			write: 0,
			phase: 0.0,
			table: [1.0; TABLE],
			seen_pages: u64::MAX,
			seen_morph: f32::NAN,
			smoothed_delay: 0.0,
			delay_ready: false,
		}
	}

	/// The pages this effect reads.
	pub fn feed(&self) -> Arc<HistoryFeed> {
		self.feed.clone()
	}

	fn history_samples(&self, infos: &ProcessInfos) -> f64 {
		let seconds = self
			.time
			.seconds(infos, self.history_time)
			.clamp(0.01, MAX_HISTORY_SECONDS);
		(seconds as f64 * self.sample_rate as f64)
			.min(self.ring[0].len() as f64 - 2.0)
			.max(2.0)
	}

	fn sync_table(&mut self) {
		let generation = self.feed.generation();

		if generation == self.seen_pages && self.morph == self.seen_morph {
			return;
		}

		self.seen_pages = generation;
		self.seen_morph = self.morph;

		let pages = self.feed.pages();
		let last = pages.len().saturating_sub(1);
		let position = self.morph.clamp(0.0, 1.0) * last as f32;
		let from = position.floor() as usize;
		let to = (from + 1).min(last);
		let blend = position - from as f32;

		for index in 0..TABLE {
			let x = index as f32 / (TABLE - 1) as f32;
			let a = page_value(&pages[from].points, x);
			let b = page_value(&pages[to].points, x);
			self.table[index] = a + (b - a) * blend;
		}
	}

	fn curve_at(&self, position: f64) -> f32 {
		let x = position.rem_euclid(1.0) as f32;
		let at = x * (TABLE - 1) as f32;
		let index = at.floor() as usize;
		let fraction = at - index as f32;
		let first = self.table[index.min(TABLE - 1)];
		let second = self.table[(index + 1).min(TABLE - 1)];
		first + (second - first) * fraction
	}

	fn read(&self, channel: usize, delay: f64) -> f32 {
		let length = self.ring[channel].len() as f64;
		let position = (self.write as f64 - delay).rem_euclid(length);
		let index = position.floor() as usize;
		let next = (index + 1) % self.ring[channel].len();
		let fraction = (position - index as f64) as f32;
		let first = self.ring[channel][index];
		let second = self.ring[channel][next];
		first + (second - first) * fraction
	}
}

/// The value of one page at `x`, with the page's segment bends.
fn page_value(points: &[HistoryPoint], x: f32) -> f32 {
	if points.len() < 2 {
		return 1.0;
	}

	let mut sorted: Vec<HistoryPoint> = points.to_vec();
	sorted.sort_by(|a, b| {
		a.time
			.partial_cmp(&b.time)
			.unwrap_or(core::cmp::Ordering::Equal)
	});
	let x = x.clamp(0.0, 1.0);

	if x <= sorted[0].time {
		return sorted[0].value;
	}

	for pair in sorted.windows(2) {
		let left = pair[0];
		let right = pair[1];

		if x > right.time {
			continue;
		}

		let span = (right.time - left.time).max(1e-6);
		let unit = ((x - left.time) / span).clamp(0.0, 1.0);
		let bend = left.bend;
		let shaped = if bend.abs() < 1e-4 {
			unit
		} else {
			((unit * bend).exp() - 1.0) / (bend.exp() - 1.0)
		};

		return left.value + (right.value - left.value) * shaped;
	}

	sorted.last().map(|point| point.value).unwrap_or(1.0)
}

impl Effect<2> for HistoryEffect {
	fn delay(&self) -> usize {
		0
	}

	fn process(
		&mut self,
		samples: &mut [f32; 2],
		_other: &[&[f32; 2]],
		process_context: &mut Box<dyn ProcessContext>,
	) {
		let infos = process_context.infos().clone();
		self.sync_table();
		let history = self.history_samples(&infos);
		let length = self.ring[0].len();
		let step = 1.0 / history;
		let host_phase = if infos.playing {
			if self.time.is_synced(&infos) {
				infos.position_beats.map(|beats| {
					let period = (self.time.beats as f64).max(1e-6);
					(beats / period).rem_euclid(1.0)
				})
			} else if let (Some(beats), Some(tempo)) = (
				infos.position_beats,
				infos.tempo.filter(|tempo| *tempo > 0.0),
			) {
				// Beat position is sample accurate; current_time can sit on a block
				// boundary and make the read head jump once per buffer.
				Some((beats * 60.0 / tempo as f64) * 1000.0 / self.history_time.max(0.01) as f64)
					.into_iter()
					.map(|phase| phase.rem_euclid(1.0))
					.next()
			} else {
				Some(
					(infos.current_time as f64 * 1000.0 / self.history_time.max(0.01) as f64)
						.rem_euclid(1.0),
				)
			}
		} else {
			None
		};

		for (channel, sample) in samples.iter().enumerate() {
			self.ring[channel][self.write] = *sample;
		}

		if let Some(phase) = host_phase {
			self.phase = phase;
		} else {
			self.phase = (self.phase + step).rem_euclid(1.0);
		}

		let value = self.curve_at(self.phase);
		let target = (1.0 - value as f64) * history;
		// A curve edit, a morph move or a host seek asks for a new read head at
		// once. Slewing it over a few milliseconds turns that step into a short
		// pitch slide instead of a click.
		let coefficient = 1.0 - (-1.0 / (0.005 * self.sample_rate as f64)).exp();
		if !self.delay_ready {
			self.delay_ready = true;
			self.smoothed_delay = target;
		} else {
			self.smoothed_delay += (target - self.smoothed_delay) * coefficient;
		}
		let delay = self.smoothed_delay.clamp(0.0, history - 1.0);
		let mix = self.mix.clamp(0.0, 1.0);

		for (channel, sample) in samples.iter_mut().enumerate() {
			let wet = self.read(channel, delay);
			*sample = *sample * (1.0 - mix) + wet * mix;
		}

		self.write = (self.write + 1) % length;
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::prelude::SyncedTime;
	use crate::{ProcessContext, ProcessInfos};

	struct TestContext(ProcessInfos);

	impl ProcessContext for TestContext {
		fn infos(&self) -> &ProcessInfos {
			&self.0
		}
		fn next_event(&mut self) -> Option<crate::NoteEvent> {
			None
		}
		fn send_event(&mut self, _: crate::NoteEvent) {}
		fn events(&self) -> &[crate::NoteEvent] {
			&[]
		}
		fn clear_events(&mut self) {}
	}

	#[test]
	fn the_flat_page_passes_the_signal_through() {
		let feed = HistoryFeed::new();
		let mut effect = HistoryEffect::with_feed(48_000, feed);
		effect.time = SyncedTime::default();
		effect.history_time = 100.0;
		let mut context: Box<dyn ProcessContext> = Box::new(TestContext(ProcessInfos {
			sample_rate: 48_000,
			playing: false,
			..ProcessInfos::default()
		}));
		let mut frame = [0.25f32, -0.5f32];

		effect.process(&mut frame, &[], &mut context);

		assert!((frame[0] - 0.25).abs() < 1e-4, "{frame:?}");
		assert!((frame[1] + 0.5).abs() < 1e-4, "{frame:?}");
	}

	#[test]
	fn a_zero_page_plays_the_far_edge_of_the_history() {
		let feed = HistoryFeed::new();
		feed.set_pages(vec![HistoryPage {
			points: vec![
				HistoryPoint {
					time: 0.0,
					value: 0.0,
					bend: 0.0,
				},
				HistoryPoint {
					time: 1.0,
					value: 0.0,
					bend: 0.0,
				},
			],
		}]);
		let mut effect = HistoryEffect::with_feed(48_000, feed);
		effect.time = SyncedTime::default();
		effect.history_time = 10.0;
		effect.mix = 1.0;
		let mut context: Box<dyn ProcessContext> = Box::new(TestContext(ProcessInfos {
			sample_rate: 48_000,
			playing: false,
			..ProcessInfos::default()
		}));
		let mut output = [0.0f32; 2];

		let mut frame = [1.0f32, 1.0f32];
		effect.process(&mut frame, &[], &mut context);

		for _ in 0..400 {
			let mut silent = [0.0f32; 2];
			effect.process(&mut silent, &[], &mut context);
			output = silent;
		}

		assert!(
			output[0].abs() < 1e-4,
			"the first samples were still live: {output:?}"
		);
	}
}
