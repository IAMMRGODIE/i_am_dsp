//! Edits that reach a running engine without rebuilding the graph.
//!
//! A knob, a switch and a modulation's depth change no node and no wire, so
//! they are handed to the engine that is already playing instead of asking for
//! a rebuild: a rebuild re-creates every node and cuts the notes in flight,
//! which is heard as the patch going quiet under the hand.

use crate::graph::{ModulationMode, NodeId};

/// One edit that does not change the shape of the graph.
#[derive(Debug, Clone, PartialEq)]
pub enum Edit {
	/// One parameter of one node.
	Param {
		/// The node the control belongs to.
		node: NodeId,
		/// The parameter's name.
		key: String,
		/// Its new value, in the parameter's own units.
		value: f32,
	},
	/// One modulation's depth.
	Depth {
		/// Which modulation, in the graph's order.
		index: usize,
		/// The new depth: a share of the parameter's own range.
		depth: f32,
	},
	/// One modulation's mode.
	Mode {
		/// Which modulation, in the graph's order.
		index: usize,
		/// Around the base, or only up from it.
		mode: ModulationMode,
	},
}

impl Edit {
	/// Applies this edit to a running engine, in the outer graph or a voice.
	pub fn apply(&self, engine: &mut crate::engine::Engine) {
		match self {
			Self::Param { node, key, value } => {
				engine.set_param(*node, key, *value);
			}
			Self::Depth { index, depth } => {
				engine.set_modulation_depth(*index, *depth);
			}
			Self::Mode { index, mode } => {
				engine.set_modulation_mode(*index, *mode);
			}
		}
	}

	/// Applies this edit to every graph a polyphonic patch runs.
	pub fn apply_poly(&self, poly: &mut crate::poly::Poly) {
		match self {
			Self::Param { node, key, value } => poly.set_param(*node, key, *value),
			Self::Depth { index, depth } => poly.set_modulation_depth(*index, *depth),
			Self::Mode { index, mode } => poly.set_modulation_mode(*index, *mode),
		}
	}
}
