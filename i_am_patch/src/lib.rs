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
pub mod state;
pub mod synths;
pub mod text;

/// How a node takes a file: the products' own decoders, re-exported so the crate
/// that hosts them does not need a dependency of its own.
pub use i_am_synths::{
	edit_slices, edit_waves, generate_slices, import_sample, import_waves, sample_preview,
	waves_preview,
};

/// The patch page's script, served at `/patcher.js`.
///
/// One IIFE split over the files in `src/page/patch` so that a picture lives
/// next to the pictures it belongs to: no piece stands alone, and the page
/// loads the concatenation as one script. The order is the page's own — a
/// `const` is not hoisted the way a function is, and the last file is what
/// starts the page up.
pub const PAGE_JS: &str = concat!(
	include_str!("page/patch/state.js"),
	include_str!("page/patch/sources.js"),
	include_str!("page/patch/heroes.js"),
	include_str!("page/patch/cards.js"),
	include_str!("page/patch/editors.js"),
	include_str!("page/patch/products.js"),
	include_str!("page/patch/canvas.js"),
	include_str!("page/patch/menus.js"),
	include_str!("page/patch/live.js"),
	include_str!("page/patch/drag.js"),
	include_str!("page/patch/view.js"),
	include_str!("page/patch/save.js"),
);

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
	pub(crate) fitted: std::sync::Arc<
		std::sync::Mutex<std::collections::BTreeMap<u32, i_am_dsp::prelude::TableSlot>>,
	>,
	pub(crate) samples: std::sync::Arc<
		std::sync::Mutex<
			std::collections::BTreeMap<
				u32,
				std::sync::Arc<std::sync::Mutex<i_am_dsp::prelude::SampleSlot>>,
			>,
		>,
	>,
	pub(crate) waves: std::sync::Arc<
		std::sync::Mutex<
			std::collections::BTreeMap<
				u32,
				std::sync::Arc<std::sync::Mutex<i_am_dsp::prelude::WaveSlot>>,
			>,
		>,
	>,
	pub(crate) irs: std::sync::Arc<
		std::sync::Mutex<std::collections::BTreeMap<u32, std::sync::Arc<i_am_effects::IrSlot>>>,
	>,
	canvases: std::sync::Arc<
		std::sync::Mutex<std::collections::BTreeMap<u32, i_am_synths::CanvasHandle>>,
	>,
	spectra: std::sync::Arc<
		std::sync::Mutex<std::collections::BTreeMap<u32, i_am_synths::SpectrumHandle>>,
	>,
	pub(crate) drummers: std::sync::Arc<
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
	/// The band meters the OTT and the vocoder publish, one node each.
	bands: std::sync::Arc<std::sync::Mutex<std::collections::BTreeMap<u32, BandMeter>>>,
	/// The scripted products' workers, one node each.
	effect_scripts: std::sync::Arc<std::sync::Mutex<std::collections::BTreeMap<u32, ScriptSlot>>>,
	/// The spectral dynamics curves, one node each.
	spectral: std::sync::Arc<
		std::sync::Mutex<
			std::collections::BTreeMap<u32, i_am_effects::SpectralDynamicsHandle>,
		>,
	>,
	/// The host's macro values, read by the `macro` nodes.
	macros: Macros,
}

/// The band meters one card draws, for the two products whose hero is a meter.
///
/// The OTT's three bands and the vocoder's bank are written by the effect the node
/// builds, so the page cannot read its own copy of them: it has to read the object
/// the node was built with. That is what went missing on a patch card — the meter
/// was made inside the build and dropped, so the hero had nothing to draw but an
/// empty grid.
#[derive(Clone)]
pub enum BandMeter {
	/// The OTT's three bands, in dB of net gain.
	Ott(std::sync::Arc<i_am_dsp::prelude::OttMeters>),
	/// The vocoder's modulator and output band levels.
	Vocoder(std::sync::Arc<i_am_dsp::prelude::VocoderMeters>),
}

impl BandMeter {
	/// The JSON a card's hero reads, the same shape the product's own route sends.
	pub fn json(&self) -> String {
		use std::fmt::Write;

		let mut json = String::new();

		match self {
			Self::Ott(meters) => {
				let bands = meters.get();

				let _ = write!(
					json,
					"{{\"bands\":[{:.2},{:.2},{:.2}]}}",
					bands[0], bands[1], bands[2]
				);
			}
			Self::Vocoder(meters) => {
				let (bands, input, output) = meters.snapshot();

				let _ = write!(json, "{{\"bands\":{bands},\"input\":[");

				for (index, value) in input.iter().enumerate() {
					if index > 0 {
						json.push(',');
					}

					let _ = write!(json, "{value:.4}");
				}

				json.push_str("],\"output\":[");

				for (index, value) in output.iter().enumerate() {
					if index > 0 {
						json.push(',');
					}

					let _ = write!(json, "{value:.4}");
				}

				json.push_str("]}");
			}
		}

		json
	}
}

/// The script one node runs, for the products whose page edits it.
///
/// A scripted node is rebuilt on every structural change, so the worker it runs
/// cannot live inside the node: the page asks for a node's script by id, and this
/// holds the same object the node was built with. The workers are different types
/// — a generator's feed, a mapper's table, a distortion's curve — and the page
/// only needs the source, the error and the points, so the slot keeps whichever
/// one the node is.
#[derive(Clone)]
pub enum ScriptSlot {
	/// A scripted generator or table: the synth products' feed.
	Synth(i_am_synths::ScriptHandle),
	/// A scripted mapper: its worker turns the source into a table.
	Mapper(std::sync::Arc<i_am_effects::ScriptWorker<i_am_effects::MapperTable>>),
	/// A scripted distortion: the same worker, over its transfer curve.
	Distortion(std::sync::Arc<i_am_effects::ScriptWorker<i_am_effects::DistortionTable>>),
}

impl ScriptSlot {
	/// The source the node is running.
	pub fn source(&self) -> String {
		match self {
			Self::Synth(handle) => handle.source_text(),
			Self::Mapper(worker) => worker.source(),
			Self::Distortion(worker) => worker.source(),
		}
	}

	/// Replaces the source, to be compiled again on the worker's own thread.
	pub fn set_source(&self, source: String) {
		match self {
			Self::Synth(handle) => {
				handle.apply(source);
			}
			Self::Mapper(worker) => {
				worker.set_source(source);
			}
			Self::Distortion(worker) => {
				worker.set_source(source);
			}
		}
	}

	/// The page's view of it: the source, the last error, and the points the
	/// script draws. The generator answers with its spectrum instead, which is
	/// what its own hero reads.
	pub fn state_json(&self) -> String {
		match self {
			Self::Synth(handle) => handle.state_json(),
			Self::Mapper(worker) => {
				let status = worker.status();
				script_json(worker.source(), status.error, status.applied, &worker.curve())
			}
			Self::Distortion(worker) => {
				let status = worker.status();
				script_json(worker.source(), status.error, status.applied, &worker.curve())
			}
		}
	}
}

/// One scripted product's editor state, in the shape a page's hero reads.
fn script_json(source: String, error: Option<String>, applied: u64, curve: &[[f32; 3]]) -> String {
	serde_json::json!({
		"source": source,
		"error": error,
		"applied": applied,
		"curve": curve
			.iter()
			.map(|point| [point[0], point[1], point[2]])
			.collect::<Vec<_>>(),
	})
	.to_string()
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

	/// The spectral dynamics curves of one node, created the first time it is
	/// asked for: the effect and the card edit the same two page lists.
	pub fn spectral(&self, node: u32) -> i_am_effects::SpectralDynamicsHandle {
		self.spectral
			.lock()
			.unwrap_or_else(|error| error.into_inner())
			.entry(node)
			.or_default()
			.clone()
	}

	/// The band meters one node published, if it is a metering product.
	pub fn bands(&self, node: u32) -> Option<BandMeter> {
		self.bands
			.lock()
			.unwrap_or_else(|error| error.into_inner())
			.get(&node)
			.cloned()
	}

	/// The OTT meters of one node: the handle its effect is built with.
	///
	/// A node keeps its id when its kind changes, so a handle of the other shape is
	/// replaced rather than answered for a node that no longer has one.
	pub fn ott(&self, node: u32) -> std::sync::Arc<i_am_dsp::prelude::OttMeters> {
		let mut bands = self
			.bands
			.lock()
			.unwrap_or_else(|error| error.into_inner());

		if let Some(BandMeter::Ott(meters)) = bands.get(&node) {
			return meters.clone();
		}

		let meters = std::sync::Arc::new(i_am_dsp::prelude::OttMeters::default());
		bands.insert(node, BandMeter::Ott(meters.clone()));
		meters
	}

	/// The script one scripted node runs, created the first time it is asked for.
	///
	/// The kind says which worker to start: the two scripted effects have one
	/// each, and the synth products use the handle this slot already keeps.
	pub fn script_slot(&self, node: u32, kind: &str) -> Option<ScriptSlot> {
		let mut scripts = self
			.effect_scripts
			.lock()
			.unwrap_or_else(|error| error.into_inner());

		if let Some(slot) = scripts.get(&node) {
			return Some(slot.clone());
		}

		let slot = match kind {
			"mod:script_mapper" => ScriptSlot::Mapper(i_am_effects::mapper_worker()),
			"mod:script_distortion" => ScriptSlot::Distortion(i_am_effects::distortion_worker()),
			_ => return None,
		};

		scripts.insert(node, slot.clone());

		Some(slot)
	}

	/// The vocoder meters of one node: the handle its effect is built with.
	pub fn vocoder(&self, node: u32) -> std::sync::Arc<i_am_dsp::prelude::VocoderMeters> {
		let mut bands = self
			.bands
			.lock()
			.unwrap_or_else(|error| error.into_inner());

		if let Some(BandMeter::Vocoder(meters)) = bands.get(&node) {
			return meters.clone();
		}

		let meters = std::sync::Arc::new(i_am_dsp::prelude::VocoderMeters::default());
		bands.insert(node, BandMeter::Vocoder(meters.clone()));
		meters
	}
}

#[cfg(test)]
mod band_tests {
	use super::*;

	/// The vocoder's hero reads its own product's shape: the band count and the two
	/// arrays. A meter that answered anything else would draw nothing.
	#[test]
	fn a_band_meter_answers_in_the_shape_the_hero_reads() {
		let slots = Slots::default();
		let json = BandMeter::Vocoder(slots.vocoder(4)).json();

		assert!(json.starts_with("{\"bands\":24,\"input\":["), "{json}");
		assert!(json.ends_with("]}"), "{json}");
		assert_eq!(json.matches("0.0000").count(), 48, "{json}");
	}

	/// One handle per node, not one per build: the effect a rebuild creates has to
	/// write into the meter the card is already reading.
	#[test]
	fn a_node_keeps_its_meter_across_builds() {
		let slots = Slots::default();
		let first = slots.ott(3);
		let second = slots.ott(3);

		assert!(std::sync::Arc::ptr_eq(&first, &second));
		assert!(matches!(slots.bands(3), Some(BandMeter::Ott(_))));

		// A node keeps its id when its kind changes, so the handle has to follow the
		// kind rather than hand back a meter of the other shape.
		slots.vocoder(3);
		assert!(matches!(slots.bands(3), Some(BandMeter::Vocoder(_))));
	}
}
