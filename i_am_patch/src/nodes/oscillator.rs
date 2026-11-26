//! The oscillators the constructors build, and the tables that feed them.

use super::*;
use super::table::build_table;

/// The oscillator a modulation node builds: two tables and a mode.
pub(crate) struct VoiceOsc {
	kind: ModKind,
	carrier: Box<dyn WaveTable + Send + Sync>,
	modulator: Box<dyn WaveTable + Send + Sync>,
	depth: Arc<AtomicU32>,
	ratio: Arc<AtomicU32>,
	integral: FmIntegral,
}

impl VoiceOsc {
	/// A modulation over the two tables, reading its controls from the atomics.
	pub(crate) fn new(
		kind: ModKind,
		carrier: Box<dyn WaveTable + Send + Sync>,
		modulator: Box<dyn WaveTable + Send + Sync>,
		depth: Arc<AtomicU32>,
		ratio: Arc<AtomicU32>,
	) -> Self {
		let integral = FmIntegral::new(modulator.as_ref());

		Self {
			kind,
			carrier,
			modulator,
			depth,
			ratio,
			integral,
		}
	}

	pub(super) fn depth(&self) -> f32 {
		f32::from_bits(self.depth.load(Ordering::Relaxed))
	}

	pub(super) fn ratio(&self) -> f32 {
		f32::from_bits(self.ratio.load(Ordering::Relaxed))
	}
}

impl Parameters for VoiceOsc {
	fn get_parameters(&self) -> Vec<Parameter> {
		Vec::new()
	}

	fn set_parameter(&mut self, _: &str, _: SetValue) -> bool {
		false
	}
}

impl Oscillator<2> for VoiceOsc {
	fn play_at(&self, frequency: f32, time: f32, phase: [f32; 2]) -> [f32; 2] {
		let depth = self.depth();
		// The ratio is continuous: a modulator slower than the carrier is a
		// valid setting, it only has to stay above a divide-by-zero guard.
		let ratio = self.ratio().max(0.01);
		let mut output = [0.0f32; 2];

		for (channel, value) in output.iter_mut().enumerate() {
			let t = (time * frequency + phase[channel]).rem_euclid(1.0);
			let u = (time * frequency * ratio + phase[channel]).rem_euclid(1.0);
			let modulator = self.modulator.sample(u, channel);
			let carrier = self.carrier.sample(t, channel);

			*value = match self.kind {
				ModKind::Am => carrier * (1.0 + depth * modulator),
				ModKind::Rm => carrier * (1.0 - depth + depth * modulator),
				ModKind::Pm => self
					.carrier
					.sample((t + depth * modulator).rem_euclid(1.0), channel),
				ModKind::Fm => self
					.carrier
					.sample((t + depth * self.integral.at(u)).rem_euclid(1.0), channel),
			};
		}

		output
	}

	fn sync(&mut self) {
		// Both tables are asked: whichever one is a product table follows its node
		// through this call, and in AM, RM and PM there is no other path to it.
		self.carrier.sync();
		self.modulator.sync();

		if matches!(self.kind, ModKind::Fm) {
			self.integral.sync(self.modulator.as_ref());
		}
	}
}

/// The table a shape node plays, with its choice shared with the card.
///
/// The table is cloned into the oscillator tree, so a later shape change has to
/// reach every copy: the choice lives in an atomic the node and every table read.
pub(super) struct ShapeTable {
	pub(super) shape: Arc<AtomicU32>,
}

impl Parameters for ShapeTable {
	fn get_parameters(&self) -> Vec<Parameter> {
		Vec::new()
	}

	fn set_parameter(&mut self, _: &str, _: SetValue) -> bool {
		false
	}
}

impl WaveTable for ShapeTable {
	fn sample(&self, t: f32, channel: usize) -> f32 {
		use i_am_dsp::prelude::{NoiseWave, SawWave, SineWave, SquareWave, TriangleWave};

		match self.shape.load(Ordering::Relaxed) {
			1 => TriangleWave.sample(t, channel),
			2 => SawWave.sample(t, channel),
			3 => SquareWave.sample(t, channel),
			4 => NoiseWave.sample(t, channel),
			_ => SineWave.sample(t, channel),
		}
	}
}

/// A table node: no audio of its own, only the shape downstream reads.
pub(super) struct TableShapeNode {
	shape: Arc<AtomicU32>,
}

impl TableShapeNode {
	pub(super) fn new(params: &BTreeMap<&'static str, f32>) -> Self {
		Self {
			shape: Arc::new(AtomicU32::new(param(params, "shape", 0.0) as u32)),
		}
	}
}

impl BlockNode for TableShapeNode {
	fn process(&mut self, _: &mut BlockContext, _: &[Buffer], _: &mut [Buffer]) {}

	fn set_param(&mut self, name: &str, value: f32) {
		if name == "shape" {
			self.shape.store(value as u32, Ordering::Relaxed);
		}
	}

	fn constructor_spec(&self) -> Option<ConstructorSpec> {
		Some(ConstructorSpec::Table(Box::new(ShapeTable {
			shape: self.shape.clone(),
		})))
	}
}

/// An oscillator node: the same shape, played at the note's pitch.
pub(super) struct OscShapeNode {
	shape: Arc<AtomicU32>,
}

impl OscShapeNode {
	pub(super) fn new(params: &BTreeMap<&'static str, f32>) -> Self {
		Self {
			shape: Arc::new(AtomicU32::new(param(params, "shape", 0.0) as u32)),
		}
	}
}

impl BlockNode for OscShapeNode {
	fn process(&mut self, _: &mut BlockContext, _: &[Buffer], _: &mut [Buffer]) {}

	fn set_param(&mut self, name: &str, value: f32) {
		if name == "shape" {
			self.shape.store(value as u32, Ordering::Relaxed);
		}
	}

	fn constructor_spec(&self) -> Option<ConstructorSpec> {
		Some(ConstructorSpec::Osc(Box::new(TableOsc(ShapeTable {
			shape: self.shape.clone(),
		}))))
	}
}

/// Builds the oscillator tree an envelope asked for.
///
/// A table is wrapped into an oscillator; a modulation takes the two tables its
/// input edges name and becomes a [`VoiceOsc`]. The tree is built once, at
/// graph build time, and its controls stay live through the atomics.
pub(crate) fn build_oscillator(
	nodes: &[Box<dyn BlockNode>],
	steps: &BTreeMap<crate::graph::NodeId, usize>,
	graph: &crate::graph::Graph,
	id: crate::graph::NodeId,
) -> Option<Box<dyn Oscillator<2> + Send + Sync>> {
	let step = *steps.get(&id)?;

	match nodes.get(step)?.constructor_spec()? {
		ConstructorSpec::Osc(oscillator) => Some(oscillator),
		ConstructorSpec::Table(table) => Some(Box::new(TableOsc(table))),
		ConstructorSpec::Mod { kind, depth, ratio } => {
			let carrier = build_table(nodes, steps, graph, id, 0)?;
			let modulator = build_table(nodes, steps, graph, id, 1)?;

			Some(Box::new(VoiceOsc::new(
				kind, carrier, modulator, depth, ratio,
			)))
		}
		// A warp produces a table; the envelope reads it through the same
		// table oscillator a bare table would get.
		ConstructorSpec::Warp { .. } => Some(Box::new(TableOsc(build_table_node(
			nodes, steps, graph, id,
		)?))),
	}
}
