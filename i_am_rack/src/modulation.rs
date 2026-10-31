//! The four control-rate modulation sources: LFO, envelope, random and table.
//!
//! The engine asks for a value every 64 samples. The LFO and random phases
//! advance per sample so their timing does not depend on the control block, and
//! the envelope follows the input per sample so it can react to transients.

use std::f32::consts::TAU;

use i_am_dsp::prelude::{EditableLfo, WaveTable};

use crate::state::SourceSettings;

/// How many samples one modulation update covers.
pub const CONTROL_INTERVAL: usize = 64;

/// The runtime state of the three sources.
pub(crate) struct Modulation {
	rate: usize,
	counter: usize,
	lfo: EditableLfo,
	lfo_phase: f32,
	table_phase: f32,
	envelope: f32,
	random_phase: f32,
	random_target: f32,
	random_value: f32,
	seed: u32,
	settings: SourceSettings,
	attack_coeff: f32,
	release_coeff: f32,
	random_coeff: f32,
}

impl Modulation {
	/// A source set at rest for a sample rate.
	pub fn new(rate: usize) -> Self {
		let settings = SourceSettings::default();
		let mut modulation = Self {
			rate,
			counter: 0,
			lfo: EditableLfo::new(),
			lfo_phase: 0.0,
			table_phase: 0.0,
			envelope: 0.0,
			random_phase: 0.0,
			random_target: 0.0,
			random_value: 0.0,
			seed: 0x1234_5678,
			settings,
			attack_coeff: 0.0,
			release_coeff: 0.0,
			random_coeff: 0.0,
		};

		modulation.set_settings(settings);
		modulation
	}

	/// Copies the page's source settings into the running sources.
	pub fn set_settings(&mut self, settings: SourceSettings) {
		self.settings = settings;
		self.lfo = editable_lfo(settings.lfo_shape);
		self.attack_coeff = time_coefficient(settings.env_attack, self.rate);
		self.release_coeff = time_coefficient(settings.env_release, self.rate);
		self.random_coeff = time_coefficient(settings.random_smooth, self.rate);
	}

	/// Advances one sample. Returns the four source values on a control tick.
	pub fn advance(&mut self, input: f32) -> Option<[f32; 4]> {
		let lfo_rate = self.settings.lfo_rate.clamp(0.01, 20.0);
		self.lfo_phase = (self.lfo_phase + lfo_rate / self.rate as f32).fract();

		let table_rate = self.settings.table_rate.clamp(0.01, 20.0);
		self.table_phase = (self.table_phase + table_rate / self.rate as f32).fract();

		let target = input.abs();
		let coeff = if target > self.envelope {
			self.attack_coeff
		} else {
			self.release_coeff
		};
		self.envelope += (target - self.envelope) * coeff;

		let random_rate = self.settings.random_rate.clamp(0.01, 20.0);
		self.random_phase += random_rate / self.rate as f32;

		if self.random_phase >= 1.0 {
			self.random_phase -= 1.0;
			self.seed = self.seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
			self.random_target = (self.seed >> 8) as f32 / 16_777_216.0 * 2.0 - 1.0;
		}

		self.random_value += (self.random_target - self.random_value) * self.random_coeff;

		self.counter += 1;

		if self.counter < CONTROL_INTERVAL {
			return None;
		}

		self.counter = 0;

		Some([
			self.lfo.sample(self.lfo_phase, 0) * 2.0 - 1.0,
			self.envelope.clamp(0.0, 1.0),
			self.random_value.clamp(-1.0, 1.0),
			table(
				self.settings.table_shape,
				self.settings.table_morph,
				self.table_phase,
			),
		])
	}
}

/// One table sample, morphing from the selected shape to the next one.
fn table(shape: i32, morph: f32, phase: f32) -> f32 {
	let first = lfo(shape, phase);
	let second = lfo(shape + 1, phase);
	first + (second - first) * morph.clamp(0.0, 1.0)
}

/// Builds the editable LFO points for one of the four preset shapes.
fn editable_lfo(shape: i32) -> EditableLfo {
	let points: &[(f32, f32)] = match shape.rem_euclid(4) {
		1 => &[(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)],
		2 => &[(0.0, 0.0), (1.0, 1.0)],
		3 => &[(0.0, 1.0), (0.5, 1.0), (0.5001, 0.0), (1.0, 0.0)],
		_ => &[
			(0.0, 0.5),
			(0.25, 1.0),
			(0.5, 0.5),
			(0.75, 0.0),
			(1.0, 0.5),
		],
	};

	let mut lfo = EditableLfo::new();
	lfo.lfo_frequency = 1.0;
	lfo.one_shot = false;

	for (time, value) in points {
		lfo.add_point(*time, *value, 0.5);
	}

	lfo
}

/// One LFO sample, in -1..1.
fn lfo(shape: i32, phase: f32) -> f32 {
	match shape.rem_euclid(4) {
		1 => 1.0 - 4.0 * (phase - 0.5).abs(),
		2 => 2.0 * phase - 1.0,
		3 => {
			if phase < 0.5 {
				1.0
			} else {
				-1.0
			}
		},
		_ => (phase * TAU).sin(),
	}
}

/// The one-pole coefficient for a time constant in milliseconds.
fn time_coefficient(milliseconds: f32, sample_rate: usize) -> f32 {
	if milliseconds <= 0.0 {
		1.0
	} else {
		1.0 - (-1.0 / (milliseconds * sample_rate as f32 / 1000.0)).exp()
	}
}
