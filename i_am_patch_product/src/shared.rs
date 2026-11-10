//! What the audio thread and the page share.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use i_am_dsp::prelude::NoteEvent;
use i_am_patch::block::BLOCK;
use i_am_patch::compile::voice_group;
use i_am_patch::engine::Engine;
use i_am_patch::graph::Graph;
use i_am_patch::poly::Poly;
use i_am_patch::scope::Scope;

/// The graph, its generation and the rate, shared between the page and the
/// audio thread. A page edit bumps the generation; the audio thread rebuilds
/// the engine at the next block boundary.
pub struct PatcherShared {
	pub graph: Mutex<Graph>,
	pub generation: AtomicU64,
	pub rate: AtomicUsize,
	/// What the graph delays its output, plus the block buffer.
	pub latency: AtomicUsize,
	/// The nodes whose output the page is watching, one ring each.
	pub taps: Mutex<Vec<Tap>>,
	/// The phase each watched source is at, for the card's playhead.
	pub phases: Mutex<Vec<(u32, f32)>>,
	/// The last compile error, for the page's status bar.
	pub error: Mutex<Option<String>>,
}

/// One watched node and the ring its output goes into.
pub struct Tap {
	/// The node id.
	pub node: u32,
	/// The frames of its output.
	pub scope: Arc<Scope>,
}

impl Default for PatcherShared {
	fn default() -> Self {
		Self {
			graph: Mutex::new(demo()),
			generation: AtomicU64::new(1),
			rate: AtomicUsize::new(48_000),
			latency: AtomicUsize::new(0),
			taps: Mutex::new(Vec::new()),
			phases: Mutex::new(Vec::new()),
			error: Mutex::new(None),
		}
	}
}

impl PatcherShared {
	/// The current graph.
	pub fn graph(&self) -> Graph {
		self.graph.lock().expect("the graph lock").clone()
	}

	/// The graph as the line patch text, which is what the host state saves.
	pub fn text(&self) -> String {
		i_am_patch::text::to_text(&self.graph.lock().expect("the graph lock"))
	}

	/// Replaces the graph from text; false when the text does not parse.
	pub fn set_text(&self, text: &str) -> bool {
		let Ok(graph) = i_am_patch::text::from_text(text) else {
			return false;
		};

		*self.graph.lock().expect("the graph lock") = graph;
		self.bump();
		true
	}

	/// The ring for a watched node, made on first use.
	pub fn tap_for(&self, node: u32) -> Arc<Scope> {
		let mut taps = self.taps.lock().expect("the tap lock");

		if let Some(tap) = taps.iter().find(|tap| tap.node == node) {
			return tap.scope.clone();
		}

		let scope = Arc::new(Scope::new());
		taps.push(Tap {
			node,
			scope: scope.clone(),
		});
		scope
	}

	/// Records where a source's cycle is.
	pub fn set_phase(&self, node: u32, phase: f32) {
		let mut phases = self.phases.lock().expect("the phase lock");

		if let Some(entry) = phases.iter_mut().find(|(id, _)| *id == node) {
			entry.1 = phase;
		} else {
			phases.push((node, phase));
		}
	}

	/// Where a source's cycle is, if the page is watching it.
	pub fn phase_for(&self, node: u32) -> Option<f32> {
		self.phases
			.lock()
			.expect("the phase lock")
			.iter()
			.find(|(id, _)| *id == node)
			.map(|(_, phase)| *phase)
	}

	/// The watched nodes, for the audio thread's copy.
	pub fn tap_snapshot(&self) -> Vec<(u32, Arc<Scope>)> {
		self.taps
			.lock()
			.expect("the tap lock")
			.iter()
			.map(|tap| (tap.node, tap.scope.clone()))
			.collect()
	}

	/// The latency a host should compensate for.
	pub fn latency(&self) -> usize {
		self.latency.load(Ordering::Relaxed)
	}

	/// The last compile error, if the graph does not build.
	pub fn error(&self) -> Option<String> {
		self.error.lock().expect("the error lock").clone()
	}

	/// Records or clears the last compile error.
	pub fn set_error(&self, error: Option<String>) {
		*self.error.lock().expect("the error lock") = error;
	}

	/// Tells the audio thread the graph changed.
	pub fn bump(&self) {
		self.generation.fetch_add(1, Ordering::Relaxed);
	}
}

/// The graph a fresh instance starts with: the pluck, one voice, one LFO.
pub fn demo() -> Graph {
	let mut graph = Graph::new();
	let midi = graph.add_node("input:midi", (0.0, 120.0)).expect("midi");
	let voice_in = graph
		.add_node("voice:in", (340.0, 120.0))
		.expect("voice in");
	let generator = graph
		.add_node("synth:additive", (680.0, 120.0))
		.expect("gen");
	let filter = graph
		.add_node("mod:filter", (1020.0, 120.0))
		.expect("filter");
	let lfo = graph.add_node("lfo", (1020.0, 580.0)).expect("lfo");
	let voice_out = graph
		.add_node("voice:out", (1620.0, 120.0))
		.expect("voice out");
	let out = graph.add_node("out:audio", (1960.0, 120.0)).expect("out");
	graph.connect((midi, 0), (voice_in, 0));
	graph.connect((voice_in, 0), (generator, 0));
	graph.connect((generator, 0), (filter, 0));
	graph.connect((filter, 0), (voice_out, 0));
	graph.connect((voice_out, 0), (out, 0));
	graph.modulate((lfo, 0), (filter, "cutoff"), 800.0);
	graph.set_param(filter, "cutoff", 1_200.0);

	graph
}

/// The block buffering both products share: samples in, samples out, one
/// block of latency, and a rebuild when the page edits the graph.
pub struct Body {
	shared: Arc<PatcherShared>,
	engine: Option<Engine>,
	poly: Option<Poly>,
	built: u64,
	input: Vec<[f32; 2]>,
	output: Vec<[f32; 2]>,
	/// The host's sidechain frames for this block.
	sidechain: Vec<[f32; 2]>,
	index: usize,
	events: Vec<NoteEvent>,
	tempo: Option<f32>,
	position: Option<f32>,
}

impl Body {
	/// A body over the shared graph.
	pub fn new(shared: Arc<PatcherShared>) -> Self {
		let mut body = Self {
			shared,
			engine: None,
			poly: None,
			built: 0,
			input: vec![[0.0; 2]; BLOCK],
			output: vec![[0.0; 2]; BLOCK],
			sidechain: vec![[0.0; 2]; BLOCK],
			index: 0,
			events: Vec::new(),
			tempo: None,
			position: None,
		};
		// The host asks for the latency before the first block.
		body.rebuild();

		body
	}

	fn rebuild(&mut self) {
		let graph = self.shared.graph();
		let rate = self.shared.rate.load(Ordering::Relaxed).max(1);

		if voice_group(&graph).is_empty() {
			match Engine::new(&graph, rate) {
				Ok(engine) => {
					self.engine = Some(engine);
					self.poly = None;
					self.shared.set_error(None);
				}
				Err(error) => {
					self.engine = None;
					self.poly = None;
					self.shared.set_error(Some(format!("{error:?}")));
				}
			}
		} else {
			match Poly::new(&graph, rate) {
				Ok(poly) => {
					self.poly = Some(poly);
					self.engine = None;
					self.shared.set_error(None);
				}
				Err(error) => {
					self.poly = None;
					self.engine = None;
					self.shared.set_error(Some(format!("{error:?}")));
				}
			}
		}

		let latency = self
			.poly
			.as_ref()
			.map(|poly| poly.latency())
			.or_else(|| self.engine.as_ref().map(|engine| engine.latency()))
			.unwrap_or(0);

		self.shared
			.latency
			.store(latency + BLOCK, Ordering::Relaxed);
		self.built = self.shared.generation.load(Ordering::Relaxed);
	}

	/// The host's tempo and song position.
	pub fn set_transport(&mut self, tempo: Option<f32>, position: Option<f32>) {
		self.tempo = tempo;
		self.position = position;
	}

	/// Queues one note event for the next block.
	pub fn push_event(&mut self, event: &NoteEvent) {
		self.events.push(event.clone());
	}

	/// One sample: the output belongs to the block that ended one block ago.
	pub fn process(&mut self, input: [f32; 2], sidechain: [f32; 2]) -> [f32; 2] {
		if self.built != self.shared.generation.load(Ordering::Relaxed) {
			self.rebuild();
		}

		let out = self.output[self.index];
		self.input[self.index] = input;
		self.sidechain[self.index] = sidechain;
		self.index += 1;

		if self.index >= BLOCK {
			self.index = 0;

			if let Some(poly) = &mut self.poly {
				poly.set_transport(self.tempo, self.position);
				poly.process_with_sidechain(
					&self.input,
					&self.sidechain,
					&self.events,
					&mut self.output,
				);
			} else if let Some(engine) = &mut self.engine {
				engine.set_transport(self.tempo, self.position);
				engine.process_with_sidechain(
					&self.input,
					&self.sidechain,
					&self.events,
					&mut self.output,
				);
			}

			self.events.clear();

			// Every watched node's output is copied for the page.
			for (node, scope) in self.shared.tap_snapshot() {
				let id = i_am_patch::graph::NodeId(node);
				let block = self
					.poly
					.as_ref()
					.and_then(|poly| {
						poly.step_of(id)
							.and_then(|step| poly.tap_block(step))
							.or_else(|| poly.tap_voice_block(id))
					})
					.or_else(|| {
						self.engine.as_ref().and_then(|engine| {
							engine.step_of(id).and_then(|step| engine.tap_block(step))
						})
					});

				if let Some(block) = block {
					scope.write(&block);
				}

				let phase = self
					.poly
					.as_ref()
					.and_then(|poly| poly.control_phase(id))
					.or_else(|| {
						self.engine.as_ref().and_then(|engine| {
							engine
								.step_of(id)
								.and_then(|step| engine.control_phase(step))
						})
					});

				if let Some(phase) = phase {
					self.shared.set_phase(node, phase);
				}
			}
		}

		out
	}
}

/// The bypass and scope the page and the products share.
pub struct UiState {
	pub bypass: Arc<AtomicBool>,
	pub scope: Arc<i_am_web_gui::Scope>,
}

impl Default for UiState {
	fn default() -> Self {
		Self {
			bypass: Arc::new(AtomicBool::new(false)),
			scope: Arc::new(i_am_web_gui::Scope::default()),
		}
	}
}
