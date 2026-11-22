//! What a patch plays, as opposed to what it connects.
//!
//! The text form carries the graph and the small states a page owns -- a script, a
//! drawing, a curve, a history. A table, a sample or an impulse response is a
//! payload: it travels here, in one serialized blob the host saves beside the text,
//! so a project keeps its sound without turning the readable patch into a wall of
//! numbers. Every node's payload is keyed by its id, so a load puts each one back
//! where the graph expects it.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::Slots;
use i_am_dsp::prelude::{ImportedTable, ImportedWave, SampleData, WaveBank};
use i_am_effects::{IrData, IrFades};

/// Everything the slots of one patch hold.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SlotState {
	/// The wavetable a Shape node plays.
	#[serde(default)]
	pub waves: BTreeMap<u32, ImportedWave>,
	/// The fitted table a Fitted node plays.
	#[serde(default)]
	pub fitted: BTreeMap<u32, ImportedTable>,
	/// The sample a Sampler node plays, with its slices.
	#[serde(default)]
	pub samples: BTreeMap<u32, SampleState>,
	/// The response a Convolver node convolves with, and the fades it plays with.
	#[serde(default)]
	pub irs: BTreeMap<u32, IrState>,
	/// The click and the six one-shots a Drummer node plays.
	#[serde(default)]
	pub drummers: BTreeMap<u32, DrummerState>,
}

/// One sampler's sample and the points it is sliced at.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SampleState {
	/// The sample itself, once something was imported.
	#[serde(default)]
	pub sample: Option<SampleData>,
	/// Where each slice starts, as a fraction of the sample.
	#[serde(default)]
	pub slices: Vec<f32>,
}

/// One convolver's response and the fades it is played with.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IrState {
	/// The response the slot holds.
	pub ir: IrData,
	/// Its fade in and out, as fractions of its length.
	pub fades: IrFades,
}

/// One drummer's click and its six one-shots.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DrummerState {
	/// The click sample, when one was imported.
	#[serde(default)]
	pub click: Option<SampleData>,
	/// The one-shots, in the order the page lists them.
	#[serde(default)]
	pub shots: [Option<SampleData>; 6],
}

impl Slots {
	/// Everything the slots hold now, for the blob a host saves.
	pub fn capture(&self) -> SlotState {
		let mut state = SlotState::default();

		for (node, slot) in self.waves.lock().expect("the wave slots").iter() {
			if let Some(bank) = &slot.lock().expect("one wave slot").bank {
				state.waves.insert(*node, bank.imported());
			}
		}

		for (node, slot) in self.fitted.lock().expect("the fitted slots").iter() {
			if let Some(table) = slot.current() {
				state.fitted.insert(*node, (*table).clone());
			}
		}

		for (node, slot) in self.samples.lock().expect("the sample slots").iter() {
			let slot = slot.lock().expect("one sample slot");

			if slot.sample.is_some() || !slot.slices.is_empty() {
				state.samples.insert(
					*node,
					SampleState {
						sample: slot.sample.as_ref().map(|sample| (**sample).clone()),
						slices: slot.slices.clone(),
					},
				);
			}
		}

		for (node, slot) in self.irs.lock().expect("the ir slots").iter() {
			if let Some(ir) = slot.load() {
				state.irs.insert(
					*node,
					IrState {
						ir: (*ir).clone(),
						fades: slot.fades(),
					},
				);
			}
		}

		for (node, handle) in self.drummers.lock().expect("the drummers").iter() {
			let click = handle.click_sample();
			let shots = handle.one_shots();

			if click.is_some() || shots.iter().any(Option::is_some) {
				state.drummers.insert(*node, DrummerState { click, shots });
			}
		}

		state
	}

	/// Puts a saved blob back, before the graph is rebuilt around it.
	pub fn restore(&self, state: &SlotState) {
		for (node, wave) in &state.waves {
			let name = (!wave.name.is_empty()).then(|| wave.name.clone());
			let bank = WaveBank::new(name, wave.frames.clone());

			self.waves(*node).lock().expect("one wave slot").bank = Some(bank);
		}

		for (node, table) in &state.fitted {
			self.fitted(*node).set(table.clone());
		}

		for (node, sample) in &state.samples {
			let slot = self.sample(*node);
			let mut slot = slot.lock().expect("one sample slot");

			slot.sample = sample.sample.clone().map(Arc::new);
			slot.slices = sample.slices.clone();
		}

		for (node, ir) in &state.irs {
			let slot = self.ir(*node);

			slot.set(Arc::new(ir.ir.clone()));
			slot.set_fades(ir.fades);
		}

		for (node, kit) in &state.drummers {
			self.drummer(*node)
				.restore_kit(kit.click.clone(), kit.shots.clone());
		}
	}
}
