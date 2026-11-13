//! The control-rate modulation sources.
//!
//! The engine asks for a value every 64 samples. Each source slot keeps its own
//! phase, envelope and random state, so the page can add several LFOs or
//! envelopes without one slot stepping on another.

use std::collections::BTreeMap;

use i_am_dsp::prelude::{
	EditableLfo, NoteEvent, PitchTracker, ProcessInfos, SyncedTime, WaveTable,
};

use crate::source_dsp;
use crate::sources::{SourceKind, SourceSettings, SourceSlot};

/// How many samples one modulation update covers.
pub const CONTROL_INTERVAL: usize = 64;

/// One running source slot.
pub(crate) struct SourceRuntime {
	id: u32,
	kind: SourceKind,
	bypass: bool,
	settings: SourceSettings,
	lfo: EditableLfo,
	lfo_phase: f32,
	table_phase: f32,
	envelope: f32,
	random_phase: f32,
	random_target: f32,
	random_value: f32,
	seed: u32,
	attack_coeff: f32,
	release_coeff: f32,
	random_coeff: f32,
	pitch_tracker: Option<PitchTracker>,
	pitch_value: f32,
	midi_note: f32,
	midi_velocity: f32,
	/// The patch's MIDI mappings, which carry a bend; the rack's settings
	/// curves are flat pairs and leave these empty.
	midi_note_points: Vec<(f32, f32, f32)>,
	midi_velocity_points: Vec<(f32, f32, f32)>,
}

impl SourceRuntime {
	/// A runtime for one source, with no slot around it.
	pub(crate) fn from_settings(
		kind: SourceKind,
		bypass: bool,
		settings: SourceSettings,
		rate: usize,
		id: u32,
	) -> Self {
		let mut runtime = Self {
			id,
			kind,
			bypass,
			settings: settings.clone(),
			lfo: editable_lfo(&settings.lfo_points),
			lfo_phase: 0.0,
			table_phase: 0.0,
			envelope: 0.0,
			random_phase: 0.0,
			random_target: 0.0,
			random_value: 0.0,
			seed: 0x1234_5678 ^ id.wrapping_mul(2_654_435_761),
			attack_coeff: 0.0,
			release_coeff: 0.0,
			random_coeff: 0.0,
			pitch_tracker: None,
			pitch_value: 0.0,
			midi_note: 0.0,
			midi_velocity: 0.0,
			midi_note_points: Vec::new(),
			midi_velocity_points: Vec::new(),
		};

		runtime.update_settings(kind, bypass, settings, rate);
		runtime
	}

	fn new(slot: &SourceSlot, rate: usize) -> Self {
		Self::from_settings(slot.kind, slot.bypass, slot.settings.clone(), rate, slot.id)
	}

	fn update(&mut self, slot: &SourceSlot, rate: usize) {
		self.update_settings(slot.kind, slot.bypass, slot.settings.clone(), rate);
	}

	pub(crate) fn update_settings(
		&mut self,
		kind: SourceKind,
		bypass: bool,
		settings: SourceSettings,
		rate: usize,
	) {
		self.kind = kind;
		self.bypass = bypass;
		self.settings = settings;
		self.lfo = editable_lfo(&self.settings.lfo_points);
		self.attack_coeff = source_dsp::time_coefficient(self.settings.env_attack, rate as f32);
		self.release_coeff = source_dsp::time_coefficient(self.settings.env_release, rate as f32);
		self.random_coeff = source_dsp::time_coefficient(self.settings.random_smooth, rate as f32);

		if self.kind == SourceKind::Pitch {
			self.pitch_tracker
				.get_or_insert_with(|| PitchTracker::new(rate, 2_048));
		} else {
			self.pitch_tracker = None;
		}
	}

	pub(crate) fn advance(
		&mut self,
		input: f32,
		infos: &ProcessInfos,
		rate: usize,
		events: &[NoteEvent],
	) {
		match self.kind {
			SourceKind::Lfo => {
				let frequency = source_rate(
					self.settings.lfo_rate,
					self.settings.lfo_sync,
					self.settings.lfo_beats,
					infos,
				);
				let next = self.lfo_phase + frequency / rate as f32;

				// One-shot stops at the end of the cycle instead of wrapping, so
				// the shape's last point is what it holds: the playhead parks at
				// the right edge and the modulation stays where the sweep left it.
				//
				// It stops a hair short of 1.0 rather than on it: the curve
				// sampler wraps a time of exactly 1.0 back to 0.0, which would
				// hold the value the sweep started from instead of the one it
				// ended on.
				self.lfo_phase = if self.settings.lfo_one_shot {
					next.min(1.0 - 1e-6)
				} else {
					next.fract()
				};
			}
			SourceKind::Envelope => {
				let target = input.abs();
				let coefficient = if target > self.envelope {
					self.attack_coeff
				} else {
					self.release_coeff
				};
				self.envelope += (target - self.envelope) * coefficient;
			}
			SourceKind::Random => {
				let frequency = source_rate(
					self.settings.random_rate,
					self.settings.random_sync,
					self.settings.random_beats,
					infos,
				);
				self.random_phase += frequency / rate as f32;

				if self.random_phase >= 1.0 {
					self.random_phase -= 1.0;
					self.seed = self
						.seed
						.wrapping_mul(1_664_525)
						.wrapping_add(1_013_904_223);
					self.random_target = (self.seed >> 8) as f32 / 16_777_216.0 * 2.0 - 1.0;
				}

				self.random_value += (self.random_target - self.random_value) * self.random_coeff;
			}
			SourceKind::Table => {
				let frequency = source_rate(
					self.settings.table_rate,
					self.settings.table_sync,
					self.settings.table_beats,
					infos,
				);
				self.table_phase = (self.table_phase + frequency / rate as f32).fract();
			}
			SourceKind::Pitch => {
				let clarity = self.settings.pitch_clarity.clamp(0.0, 1.0);
				let coefficient =
					source_dsp::time_coefficient(self.settings.pitch_glide, rate as f32);

				if let Some(tracker) = &mut self.pitch_tracker {
					tracker.process_sample(input);

					if let Some(estimate) = tracker.last()
						&& estimate.clarity >= clarity
					{
						let target = (estimate.midi_note / 127.0).clamp(0.0, 1.0);
						self.pitch_value += (target - self.pitch_value) * coefficient;
					}
				}
			}
			SourceKind::Midi => {
				for event in events {
					match event {
						NoteEvent::NoteOn { note, velocity, .. } => {
							self.midi_note = *note as f32;
							self.midi_velocity = velocity.clamp(0.0, 1.0);
						}
						NoteEvent::NoteOff { .. }
						| NoteEvent::Stop { .. }
						| NoteEvent::ImmediateStop => {
							self.midi_velocity = 0.0;
						}
						_ => {}
					}
				}
			}
		}
	}

	pub(crate) fn value(&self) -> f32 {
		if self.bypass {
			return 0.0;
		}

		match self.kind {
			SourceKind::Lfo => self.lfo.sample(self.lfo_phase, 0) * 2.0 - 1.0,
			SourceKind::Envelope => self.envelope.clamp(0.0, 1.0),
			SourceKind::Random => self.random_value.clamp(-1.0, 1.0),
			SourceKind::Table => source_dsp::morph(
				self.settings.table_shape,
				self.settings.table_morph,
				self.table_phase,
			),
			SourceKind::Pitch => self.pitch_value.clamp(0.0, 1.0),
			SourceKind::Midi => {
				if self.settings.midi_mode == 1 {
					let x = self.midi_velocity.clamp(0.0, 1.0);

					if self.midi_velocity_points.is_empty() {
						curve_value(&self.settings.midi_velocity_curve, x)
					} else {
						curve_value_bend(&self.midi_velocity_points, x)
					}
				} else {
					let note = self.midi_note
						+ self.settings.midi_transpose as f32
						+ self.settings.midi_fine / 100.0;
					let x = (note / 127.0).clamp(0.0, 1.0);

					if self.midi_note_points.is_empty() {
						curve_value(&self.settings.midi_note_curve, x)
					} else {
						curve_value_bend(&self.midi_note_points, x)
					}
				}
			}
		}
	}

	pub(crate) fn phase(&self) -> f32 {
		match self.kind {
			SourceKind::Lfo => self.lfo_phase,
			SourceKind::Table => self.table_phase,
			_ => 0.0,
		}
	}

	/// Keeps the patch's note mapping, with its bends.
	pub(crate) fn set_midi_note_points(&mut self, points: Vec<(f32, f32, f32)>) {
		self.midi_note_points = points;
	}

	/// Keeps the patch's velocity mapping, with its bends.
	pub(crate) fn set_midi_velocity_points(&mut self, points: Vec<(f32, f32, f32)>) {
		self.midi_velocity_points = points;
	}

	/// Where a MIDI source's input sits, 0..1, for its mapping picture.
	///
	/// Note mode reports the transposed note over 127; velocity mode reports
	/// the velocity itself. Every other source has no input to place.
	pub(crate) fn midi_position(&self) -> Option<f32> {
		if self.kind != SourceKind::Midi {
			return None;
		}

		if self.settings.midi_mode == 1 {
			return Some(self.midi_velocity.clamp(0.0, 1.0));
		}

		let note =
			self.midi_note + self.settings.midi_transpose as f32 + self.settings.midi_fine / 100.0;

		Some((note / 127.0).clamp(0.0, 1.0))
	}
}

/// The live source set.
pub struct Modulation {
	rate: usize,
	counter: usize,
	sources: Vec<SourceRuntime>,
	values: BTreeMap<u32, f32>,
	phases: BTreeMap<u32, f32>,
	generation: u64,
}

impl Modulation {
	/// An empty source set for a sample rate.
	pub fn new(rate: usize) -> Self {
		Self {
			rate,
			counter: 0,
			sources: Vec::new(),
			values: BTreeMap::new(),
			phases: BTreeMap::new(),
			generation: 0,
		}
	}

	/// Reconciles the runtimes with the page's slots, preserving the state of a
	/// slot whose kind did not change.
	pub fn set_slots(&mut self, slots: &[SourceSlot]) {
		let mut next = Vec::with_capacity(slots.len());

		for slot in slots {
			let position = self.sources.iter().position(|source| source.id == slot.id);

			if let Some(position) = position
				&& self.sources[position].kind == slot.kind
			{
				self.sources[position].update(slot, self.rate);
				next.push(self.sources.remove(position));
				continue;
			}

			next.push(SourceRuntime::new(slot, self.rate));
		}

		self.sources = next;
	}

	/// Advances one sample. Returns the source values on a control tick.
	pub fn advance(
		&mut self,
		input: f32,
		infos: &ProcessInfos,
		events: &[NoteEvent],
	) -> Option<BTreeMap<u32, f32>> {
		for source in &mut self.sources {
			source.advance(input, infos, self.rate, events);
		}

		self.counter += 1;

		if self.counter < CONTROL_INTERVAL {
			return None;
		}

		self.counter = 0;
		self.values.clear();
		self.phases.clear();
		self.generation = self.generation.wrapping_add(1);

		for source in &self.sources {
			self.values.insert(source.id, source.value());
			self.phases.insert(source.id, source.phase());
		}

		Some(self.values.clone())
	}

	/// The last values/phases and a generation counter for the live view.
	pub fn snapshot(&self) -> (Vec<(u32, f32, f32)>, u64) {
		let sources = self
			.sources
			.iter()
			.map(|source| (source.id, source.value(), source.phase()))
			.collect();
		(sources, self.generation)
	}
}

/// Piecewise-linear map through sorted x/y points.
fn curve_value(points: &[(f32, f32)], x: f32) -> f32 {
	if points.is_empty() {
		return x.clamp(0.0, 1.0);
	}

	let x = x.clamp(0.0, 1.0);

	if x <= points[0].0 {
		return points[0].1.clamp(0.0, 1.0);
	}

	for pair in points.windows(2) {
		let (x0, y0) = pair[0];
		let (x1, y1) = pair[1];

		if x <= x1 {
			let span = (x1 - x0).max(1e-6);
			let unit = (x - x0) / span;
			return (y0 + (y1 - y0) * unit).clamp(0.0, 1.0);
		}
	}

	points.last().map_or(x, |point| point.1.clamp(0.0, 1.0))
}

/// Piecewise map through sorted x/y/bend points.
fn curve_value_bend(points: &[(f32, f32, f32)], x: f32) -> f32 {
	if points.is_empty() {
		return x.clamp(0.0, 1.0);
	}

	let x = x.clamp(0.0, 1.0);

	if x <= points[0].0 {
		return points[0].1.clamp(0.0, 1.0);
	}

	for pair in points.windows(2) {
		let (x0, y0, bend) = pair[0];
		let (x1, y1, _) = pair[1];

		if x <= x1 {
			let span = (x1 - x0).max(1e-6);
			let unit = ((x - x0) / span).clamp(0.0, 1.0);

			return (y0 + (y1 - y0) * bend_unit(unit, bend)).clamp(0.0, 1.0);
		}
	}

	points.last().map_or(x, |point| point.1.clamp(0.0, 1.0))
}

/// The curve's unit position inside one segment, bent toward either end.
fn bend_unit(t: f32, bend: f32) -> f32 {
	if bend.abs() < 1e-4 {
		t
	} else {
		((t * bend).exp() - 1.0) / (bend.exp() - 1.0)
	}
}

/// A source rate in hertz, optionally read as a musical period by [`SyncedTime`].
fn source_rate(rate: f32, sync: bool, beats: f32, infos: &ProcessInfos) -> f32 {
	SyncedTime { sync, beats }
		.rate_hz(infos, rate.clamp(0.01, 20.0))
		.clamp(0.01, 100.0)
}

/// One table sample, morphing from the selected shape to the next one.
/// Builds an EditableLfo from the points the page last sent.
fn editable_lfo(points: &[(f32, f32, f32)]) -> EditableLfo {
	let mut lfo = EditableLfo::new();
	lfo.lfo_frequency = 1.0;
	lfo.one_shot = false;

	for (time, value, bend) in points {
		lfo.add_point(*time, *value, *bend);
	}

	lfo
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn a_midi_bend_pulls_the_curve_between_its_points() {
		let points = [(0.0, 0.0, 4.0), (1.0, 1.0, 0.0)];
		let value = curve_value_bend(&points, 0.5);

		assert!(value > 0.0 && value < 0.5, "{value}");
	}

	#[test]
	fn a_flat_bend_is_the_straight_line() {
		let points = [(0.0, 0.0, 0.0), (1.0, 1.0, 0.0)];

		assert!((curve_value_bend(&points, 0.25) - 0.25).abs() < 1e-6);
	}
}
