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
	/// Where each watched MIDI source's input sits, for its mapping point.
	pub midi_inputs: Mutex<Vec<(u32, f32)>>,
	/// The last compile error, for the page's status bar.
	pub error: Mutex<Option<String>>,
	/// The scripts the running nodes own, by node id, for the page's editor.
	pub scripts: i_am_patch::Scripts,
	/// What the page hands the nodes: fitted tables, samples, responses.
	pub slots: i_am_patch::Slots,
	/// The nodes the page asked to clear their tail, drained by the audio thread.
	pub clear_requests: Mutex<Vec<u32>>,
	/// The edits that move a control without changing the graph, drained by the
	/// audio thread at the next block so the notes in flight keep playing.
	pub edits: Mutex<Vec<i_am_patch::edit::Edit>>,
}

/// One watched node and the rings its signal goes into.
pub struct Tap {
	/// The node id.
	pub node: u32,
	/// The frames of its output.
	pub scope: Arc<Scope>,
	/// The frames arriving at it, so a card can draw before and after.
	pub input: Arc<Scope>,
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
			midi_inputs: Mutex::new(Vec::new()),
			error: Mutex::new(None),
			scripts: Default::default(),
			slots: Default::default(),
			clear_requests: Mutex::new(Vec::new()),
			edits: Mutex::new(Vec::new()),
		}
	}
}

impl PatcherShared {
	/// The current graph.
	pub fn graph(&self) -> Graph {
		self.graph.lock().expect("the graph lock").clone()
	}

	/// Queues an edit that moves a control, for the running engine.
	pub fn edit(&self, edit: i_am_patch::edit::Edit) {
		self.edits.lock().expect("the edit lock").push(edit);
	}

	/// The graph as the line patch text, which is what the host state saves.
	pub fn text(&self) -> String {
		i_am_patch::text::to_text_with_slots(
			&self.graph.lock().expect("the graph lock"),
			&self.slots,
		)
	}

	/// Replaces the graph from text; false when the text does not parse.
	pub fn set_text(&self, text: &str) -> bool {
		let Ok(graph) = i_am_patch::text::from_text_with_slots(text, &self.slots) else {
			return false;
		};

		*self.graph.lock().expect("the graph lock") = graph;
		self.bump();
		true
	}

	/// The rings for a watched node, made on first use: output and input.
	pub fn taps_for(&self, node: u32) -> (Arc<Scope>, Arc<Scope>) {
		let mut taps = self.taps.lock().expect("the tap lock");

		if let Some(tap) = taps.iter().find(|tap| tap.node == node) {
			return (tap.scope.clone(), tap.input.clone());
		}

		let scope = Arc::new(Scope::new());
		let input = Arc::new(Scope::new());
		taps.push(Tap {
			node,
			scope: scope.clone(),
			input: input.clone(),
		});
		(scope, input)
	}

	/// Records where a source's cycle is.
	pub fn set_phase(&self, node: u32, phase: f32) {
		// The audio thread must not wait for the page: a skipped phase only
		// costs one frame of the playhead.
		let Ok(mut phases) = self.phases.try_lock() else {
			return;
		};

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

	/// Records where a MIDI source's input sits.
	pub fn set_midi_input(&self, node: u32, position: f32) {
		// The audio thread must not wait for the page, like a phase.
		let Ok(mut inputs) = self.midi_inputs.try_lock() else {
			return;
		};

		if let Some(entry) = inputs.iter_mut().find(|(id, _)| *id == node) {
			entry.1 = position;
		} else {
			inputs.push((node, position));
		}
	}

	/// Where a MIDI source's input sits, if the page is watching it.
	pub fn midi_input_for(&self, node: u32) -> Option<f32> {
		self.midi_inputs
			.lock()
			.expect("the midi input lock")
			.iter()
			.find(|(id, _)| *id == node)
			.map(|(_, position)| *position)
	}

	/// The watched nodes, copied into the output list without waiting.
	pub fn tap_snapshot_into(&self, out: &mut Vec<(u32, Arc<Scope>, Arc<Scope>)>) {
		out.clear();

		if let Ok(taps) = self.taps.try_lock() {
			out.extend(
				taps.iter()
					.map(|tap| (tap.node, tap.scope.clone(), tap.input.clone())),
			);
		}
	}

	/// The watched nodes, for callers outside the audio thread.
	pub fn tap_snapshot(&self) -> Vec<(u32, Arc<Scope>, Arc<Scope>)> {
		let mut out = Vec::new();
		self.tap_snapshot_into(&mut out);
		out
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
	// A depth is a share of the cutoff's own range: a third of it is a sweep of
	// about two octaves either way, which is what the demo is for.
	graph.modulate((lfo, 0), (filter, "cutoff"), 0.33);
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
	/// The watched nodes, reused each block.
	taps: Vec<(u32, Arc<Scope>, Arc<Scope>)>,
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
			taps: Vec::new(),
		};
		// The host asks for the latency before the first block.
		body.rebuild();

		body
	}

	/// Drops the tail the node with this id is holding.
	pub fn clear_tail(&mut self, node: u32) {
		if let Some(engine) = &mut self.engine {
			engine.clear_tail(node);
		}

		if let Some(poly) = &mut self.poly {
			poly.clear_tail(node);
		}
	}

	/// Applies the edits the page made since the last block.
	///
	/// The lock is tried, not taken: a page holding it for a moment costs one
	/// block, and the edit stays queued for the next one.
	fn apply_edits(&mut self) {
		let Ok(mut edits) = self.shared.edits.try_lock() else {
			return;
		};

		if edits.is_empty() {
			return;
		}

		for edit in edits.drain(..) {
			if let Some(poly) = &mut self.poly {
				edit.apply_poly(poly);
			} else if let Some(engine) = &mut self.engine {
				edit.apply(engine);
			}
		}
	}

	fn rebuild(&mut self) {
		// The new engine is built from the graph, which already holds these, so
		// the queue is dropped rather than replayed onto a fresh engine.
		if let Ok(mut edits) = self.shared.edits.try_lock() {
			edits.clear();
		}

		let graph = self.shared.graph();
		let rate = self.shared.rate.load(Ordering::Relaxed).max(1);

		if voice_group(&graph).is_empty() {
			match Engine::new_with(&graph, rate, &self.shared.slots) {
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
			match Poly::new_with(&graph, rate, &self.shared.slots) {
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

		// The page edits a scripted node through this map; a rebuild replaces it,
		// because the nodes that owned the old workers are gone.
		{
			let mut scripts = self.shared.scripts.lock().expect("the script lock");
			scripts.clear();

			if let Some(engine) = &self.engine {
				for (node, script) in engine.script_sources() {
					scripts.insert(node, script);
				}
			}
		}

		if let Some(poly) = &mut self.poly {
			poly.set_scripts(self.shared.scripts.clone());
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
			self.apply_edits();

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
			self.shared.tap_snapshot_into(&mut self.taps);

			for (node, scope, input) in &self.taps {
				let id = i_am_patch::graph::NodeId(*node);
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

				// The other side of the node, so a card draws what arrived as
				// well as what left.
				let arriving = self
					.poly
					.as_ref()
					.and_then(|poly| poly.step_of(id).and_then(|step| poly.input_block(step)))
					.or_else(|| {
						self.engine.as_ref().and_then(|engine| {
							engine.step_of(id).and_then(|step| engine.input_block(step))
						})
					});

				if let Some(block) = arriving {
					input.write(&block);
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
					self.shared.set_phase(*node, phase);
				}

				let midi = self
					.poly
					.as_ref()
					.and_then(|poly| poly.midi_position(id))
					.or_else(|| {
						self.engine.as_ref().and_then(|engine| {
							engine
								.step_of(id)
								.and_then(|step| engine.midi_position(step))
						})
					});

				if let Some(position) = midi {
					self.shared.set_midi_input(*node, position);
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

#[cfg(test)]
mod tests {
	use super::*;
	use i_am_dsp::prelude::NoteEvent;

	/// The peak of the next blocks of output.
	fn peak(body: &mut Body, blocks: usize) -> f32 {
		let mut peak = 0.0_f32;

		for _ in 0..blocks {
			for _ in 0..BLOCK {
				let frame = body.process([0.0; 2], [0.0; 2]);
				peak = peak.max(frame[0].abs());
			}
		}

		peak
	}

	/// A control that moves reaches the engine that is already playing: the
	/// page does not have to make the graph look different for the sound to
	/// follow.
	#[test]
	fn an_edit_moves_the_running_engine() {
		let shared = Arc::new(PatcherShared::default());
		let mut body = Body::new(shared.clone());
		let out = shared
			.graph()
			.nodes
			.iter()
			.find(|node| node.kind.id == "out:audio")
			.map(|node| node.id)
			.expect("the demo graph has an output");

		body.push_event(&NoteEvent::NoteOn {
			channel: 0,
			note: 60,
			velocity: 1.0,
		});

		// The note's attack takes longer than a block, so the level is measured
		// once it has settled, over a window short enough that the envelope
		// moves little between the two readings.
		peak(&mut body, 40);

		let before = peak(&mut body, 4);

		assert!(before > 0.01, "the demo graph did not sound: {before}");

		shared.edit(i_am_patch::edit::Edit::Param {
			node: out,
			key: "gain".to_string(),
			value: 0.25,
		});

		// The body hands back the block that ended before the edit landed, so the
		// first window after it is still the old level.
		peak(&mut body, 4);

		let after = peak(&mut body, 4);

		assert!(
			after < before * 0.5,
			"{after} did not follow {before}: the edit never reached the engine"
		);
	}
}
