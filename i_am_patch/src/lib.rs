//! The patch graph's reusable building blocks.
//!
//! The patcher and the rack share these: the LR4 crossover, the control-rate
//! modulation sources and their runtime. Nothing here knows about a graph, a
//! page or a host yet; those live in the products.

pub mod block;
pub mod catalog;
pub mod compile;
pub mod crossover;
pub mod edit;
pub mod engine;
pub mod graph;
pub mod modulation;
pub mod nodes;
pub mod poly;
pub mod scope;
mod source_dsp;
pub mod sources;
pub mod synths;
pub mod text;

/// How a node takes a file: the products' own decoders, re-exported so the crate
/// that hosts them does not need a dependency of its own.
pub use i_am_synths::{
	edit_slices, edit_waves, generate_slices, import_sample, import_waves, sample_preview,
	waves_preview,
};

#[cfg(test)]
/// A graph with one note voice, used by more than one module's tests.
pub(crate) fn voice_patch() -> crate::graph::Graph {
	use crate::graph::Graph;

	let mut graph = Graph::new();
	let midi = graph.add_node("input:midi", (0.0, 0.0)).expect("midi");
	let voice_in = graph.add_node("voice:in", (50.0, 0.0)).expect("voice in");
	let generator = graph.add_node("synth:additive", (100.0, 0.0)).expect("gen");
	let filter = graph.add_node("mod:filter", (200.0, 0.0)).expect("filter");
	let lfo = graph.add_node("lfo", (200.0, 100.0)).expect("lfo");
	let voice_out = graph
		.add_node("voice:out", (300.0, 0.0))
		.expect("voice out");
	let out = graph.add_node("out:audio", (400.0, 0.0)).expect("out");
	graph.connect((midi, 0), (voice_in, 0));
	graph.connect((voice_in, 0), (generator, 0));
	graph.connect((generator, 0), (filter, 0));
	graph.connect((filter, 0), (voice_out, 0));
	graph.connect((voice_out, 0), (out, 0));
	graph.modulate((lfo, 0), (filter, "cutoff"), 0.33);

	graph
}

/// One cycle of a constructor node's table, for the page's picture.
///
/// The page cannot sample a product's table itself: it asks the same code the
/// engine uses to build one, so the picture is the DSP's own output.
pub fn sample_constructor(
	graph: &crate::graph::Graph,
	slots: &crate::Slots,
	rate: usize,
	id: crate::graph::NodeId,
	points: usize,
) -> Option<Vec<f32>> {
	let mut nodes: Vec<Box<dyn crate::nodes::BlockNode>> = Vec::new();
	let mut steps = std::collections::BTreeMap::new();

	for instance in &graph.nodes {
		let Some(node) = crate::nodes::make_with(
			instance.kind.id,
			instance.id.0,
			rate,
			&instance.params,
			&instance.points,
			&instance.velocity_points,
			slots,
		) else {
			continue;
		};

		steps.insert(instance.id, nodes.len());
		nodes.push(node);
	}

	let points = points.clamp(8, 2048);

	if let Some(table) = crate::nodes::build_table_node(&nodes, &steps, graph, id) {
		return Some(
			(0..points)
				.map(|index| table.sample(index as f32 / points as f32, 0))
				.collect(),
		);
	}

	let oscillator = crate::nodes::build_oscillator(&nodes, &steps, graph, id)?;

	Some(
		(0..points)
			.map(|index| oscillator.play_at(1.0, index as f32 / points as f32, [0.0; 2])[0])
			.collect(),
	)
}

/// Where a page reads the scripts the running nodes own, by node id.
pub type Scripts = std::sync::Arc<
	std::sync::Mutex<
		std::collections::HashMap<u32, std::sync::Arc<dyn i_am_effects::ScriptSource>>,
	>,
>;

/// How many macro controls a patcher offers its host.
///
/// A plug-in's parameter table is fixed when it is created and nothing may add a
/// parameter later, so the pad is a fixed size: a graph uses as many of them as it
/// likes, and the unused ones cost a slot in the host's automation list and nothing
/// else. Sixty-four is what a page can show in a column and a host can list without
/// becoming a modal dialog.
pub const MACRO_COUNT: usize = 64;

/// The macro values a host automates, shared with the nodes that read them.
///
/// The values live beside the graph rather than in it: a graph parameter is what a
/// page draws and what a voice copies, and a macro is neither. The product pumps the
/// host's parameter table into this pad each block, and the `macro` node reads it as
/// a control signal — so the whole modulation path is the one a page knob already
/// uses, and a voice copy reads the same bank without being told.
#[derive(Clone, Debug)]
pub struct Macros(std::sync::Arc<[portable_atomic::AtomicF32; MACRO_COUNT]>);

impl Macros {
	/// An all-zero pad.
	pub fn new() -> Self {
		Self(std::sync::Arc::new(std::array::from_fn(|_| {
			portable_atomic::AtomicF32::new(0.0)
		})))
	}

	/// One macro's value, 0..1; a missing index reads as zero.
	pub fn get(&self, index: usize) -> f32 {
		self.0
			.get(index)
			.map(|slot| slot.load(std::sync::atomic::Ordering::Relaxed))
			.unwrap_or(0.0)
	}

	/// Writes one macro, clamped to the range a control signal carries.
	pub fn set(&self, index: usize, value: f32) {
		if let Some(slot) = self.0.get(index) {
			slot.store(value.clamp(0.0, 1.0), std::sync::atomic::Ordering::Relaxed);
		}
	}
}

impl Default for Macros {
	fn default() -> Self {
		Self::new()
	}
}

/// What a page hands to the nodes: the things a graph parameter cannot carry.
///
/// A fitted table, a sample, a wavetable and an impulse response are all
/// serialized values, which a graph parameter cannot hold, so the page's upload
/// routes write them into slots here. The node is built with the slot the page
/// already holds, so an import lands on the next block and the graph does not
/// have to be rebuilt for it.
#[derive(Clone, Default)]
pub struct Slots {
	fitted: std::sync::Arc<
		std::sync::Mutex<std::collections::BTreeMap<u32, i_am_dsp::prelude::TableSlot>>,
	>,
	samples: std::sync::Arc<
		std::sync::Mutex<
			std::collections::BTreeMap<
				u32,
				std::sync::Arc<std::sync::Mutex<i_am_dsp::prelude::SampleSlot>>,
			>,
		>,
	>,
	waves: std::sync::Arc<
		std::sync::Mutex<
			std::collections::BTreeMap<
				u32,
				std::sync::Arc<std::sync::Mutex<i_am_dsp::prelude::WaveSlot>>,
			>,
		>,
	>,
	irs: std::sync::Arc<
		std::sync::Mutex<std::collections::BTreeMap<u32, std::sync::Arc<i_am_effects::IrSlot>>>,
	>,
	canvases: std::sync::Arc<
		std::sync::Mutex<std::collections::BTreeMap<u32, i_am_synths::CanvasHandle>>,
	>,
	spectra: std::sync::Arc<
		std::sync::Mutex<std::collections::BTreeMap<u32, i_am_synths::SpectrumHandle>>,
	>,
	drummers: std::sync::Arc<
		std::sync::Mutex<std::collections::BTreeMap<u32, i_am_synths::DrummerHandle>>,
	>,
	filter_tables: std::sync::Arc<
		std::sync::Mutex<std::collections::BTreeMap<u32, i_am_effects::FilterTableHandle>>,
	>,
	histories: std::sync::Arc<
		std::sync::Mutex<std::collections::BTreeMap<u32, i_am_effects::HistoryHandle>>,
	>,
	scripts: std::sync::Arc<
		std::sync::Mutex<std::collections::BTreeMap<u32, i_am_synths::ScriptHandle>>,
	>,
	/// The host's macro values, read by the `macro` nodes.
	macros: Macros,
}

impl Slots {
	/// The host-driven macro values.
	pub fn macros(&self) -> &Macros {
		&self.macros
	}

	/// The fitted table of one node, created the first time it is asked for.
	pub fn fitted(&self, node: u32) -> i_am_dsp::prelude::TableSlot {
		self.fitted
			.lock()
			.unwrap_or_else(|error| error.into_inner())
			.entry(node)
			.or_default()
			.clone()
	}

	/// The sample slot of one node, created the first time it is asked for.
	pub fn sample(
		&self,
		node: u32,
	) -> std::sync::Arc<std::sync::Mutex<i_am_dsp::prelude::SampleSlot>> {
		self.samples
			.lock()
			.unwrap_or_else(|error| error.into_inner())
			.entry(node)
			.or_default()
			.clone()
	}

	/// The wavetable slot of one node, created the first time it is asked for.
	pub fn waves(
		&self,
		node: u32,
	) -> std::sync::Arc<std::sync::Mutex<i_am_dsp::prelude::WaveSlot>> {
		self.waves
			.lock()
			.unwrap_or_else(|error| error.into_inner())
			.entry(node)
			.or_default()
			.clone()
	}

	/// The impulse response slot of one node, created the first time it is
	/// asked for.
	pub fn ir(&self, node: u32) -> std::sync::Arc<i_am_effects::IrSlot> {
		self.irs
			.lock()
			.unwrap_or_else(|error| error.into_inner())
			.entry(node)
			.or_default()
			.clone()
	}

	/// The Canvas drawing of one node, created the first time it is asked for.
	pub fn canvas(&self, node: u32) -> i_am_synths::CanvasHandle {
		self.canvases
			.lock()
			.unwrap_or_else(|error| error.into_inner())
			.entry(node)
			.or_default()
			.clone()
	}
	/// The Spectrum pages of one node, created the first time it is asked for.
	pub fn spectrum(&self, node: u32) -> i_am_synths::SpectrumHandle {
		self.spectra
			.lock()
			.unwrap_or_else(|error| error.into_inner())
			.entry(node)
			.or_default()
			.clone()
	}
	/// The Drummer click and slots of one node, created the first time it is asked for.
	pub fn drummer(&self, node: u32) -> i_am_synths::DrummerHandle {
		self.drummers
			.lock()
			.unwrap_or_else(|error| error.into_inner())
			.entry(node)
			.or_default()
			.clone()
	}
	/// The Filter Table frame of one node, created the first time it is asked for.
	pub fn filter_table(&self, node: u32) -> i_am_effects::FilterTableHandle {
		self.filter_tables
			.lock()
			.unwrap_or_else(|error| error.into_inner())
			.entry(node)
			.or_default()
			.clone()
	}

	/// The History pages of one node, created the first time it is asked for.
	pub fn history(&self, node: u32) -> i_am_effects::HistoryHandle {
		self.histories
			.lock()
			.unwrap_or_else(|error| error.into_inner())
			.entry(node)
			.or_default()
			.clone()
	}

	/// The script feed of one node, created the first time it is asked for.
	pub fn script(&self, node: u32) -> i_am_synths::ScriptHandle {
		self.scripts
			.lock()
			.unwrap_or_else(|error| error.into_inner())
			.entry(node)
			.or_default()
			.clone()
	}
}
