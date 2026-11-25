//! What the audio thread and the page share.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use i_am_dsp::prelude::{NoteEvent, ParamMap, SetValue, Value, from_binary, to_binary};
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
	/// What each modulated node's parameters are at, published every block: a
	/// card draws the value the engine is running, not the base the page posted.
	pub values: Mutex<NodeValues>,
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
	/// The host-facing macro parameters: the same sixty-four every instance.
	///
	/// A host cannot add a parameter later, so the table is built once and every
	/// graph draws from it. The page writes it, the host writes it, and the audio
	/// thread copies whatever it finds into the pad the `macro` nodes read.
	pub params: ParamMap,
	/// Where the page's Save button has got to.
	pub save: crate::save::SaveSlot,
	/// The last encoded payload blob, with the stamp it was made at: a save that
	/// changes nothing copies bytes instead of encoding them again.
	pub slot_cache: Mutex<(Option<i_am_patch::state::SlotStamp>, Vec<u8>)>,
}

/// The host's macro controls: sixty-four floats, zero to one, always present.
fn host_macro_map() -> ParamMap {
	let entries = (1..=i_am_patch::MACRO_COUNT).map(|index| {
		let value = Value::Float {
			value: 0.0,
			range: 0.0..=1.0,
			logarithmic: false,
		};

		(format!("macro{index}"), Arc::new(value.to_atomic_value()))
	});

	ParamMap::new_with(entries.collect::<Vec<_>>())
}

/// The value each parameter of a node is at, by node id, in the node's own
/// units.
///
/// This is the pair a card draws from: what the engine is running, rather than
/// the base the page posted.
pub type NodeValues = Vec<(u32, Vec<(&'static str, f32)>)>;

/// One watched node and the rings its signal goes into.
pub struct Tap {
	/// The node id.
	pub node: u32,
	/// The frames of its output.
	pub scope: Arc<Scope>,
	/// The frames arriving at it, so a card can draw before and after.
	pub input: Arc<Scope>,
}

impl PatcherShared {
	/// The shared state over a graph the product chose.
	///
	/// The two products are not the same instrument: an effect has to pass the track's
	/// audio through, an instrument makes its own. Each opens on a graph that shows
	/// what it is for.
	pub fn with_graph(graph: Graph) -> Self {
		Self::build(graph)
	}

	fn build(graph: Graph) -> Self {
		Self {
			graph: Mutex::new(graph),
			generation: AtomicU64::new(1),
			rate: AtomicUsize::new(48_000),
			latency: AtomicUsize::new(0),
			taps: Mutex::new(Vec::new()),
			phases: Mutex::new(Vec::new()),
			midi_inputs: Mutex::new(Vec::new()),
			values: Mutex::new(Vec::new()),
			error: Mutex::new(None),
			scripts: Default::default(),
			slots: Default::default(),
			clear_requests: Mutex::new(Vec::new()),
			edits: Mutex::new(Vec::new()),
			params: host_macro_map(),
			save: crate::save::SaveSlot::default(),
			slot_cache: Mutex::new((None, Vec::new())),
		}
	}
}

impl Default for PatcherShared {
	fn default() -> Self {
		Self::build(demo())
	}
}

impl PatcherShared {
	/// Copies the host's macro values into the pad the nodes read.
	///
	/// A host and a page both write the same atomics, so this is the one place the
	/// two sides meet: sixty-four loads per block, and a store only for a value that
	/// actually moved. The names are addressed by index because formatting a name
	/// here would allocate on the audio thread.
	pub fn pull_macros(&self) {
		let macros = self.slots.macros();

		for index in 0..i_am_patch::MACRO_COUNT {
			let Some(param) = self.params.get_ref_by_index(index) else {
				continue;
			};

			if let SetValue::Float(value) = param.load(Ordering::Relaxed)
				&& macros.get(index) != value
			{
				macros.set(index, value);
			}
		}
	}

	/// Writes one macro from its identifier, for a host restoring a project.
	pub fn set_macro(&self, identifier: &str, value: SetValue) -> bool {
		self.params.set(identifier, value, Ordering::SeqCst)
	}

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

	/// Every payload the nodes play, as the blob a host saves beside the text.
	///
	/// A table, a sample or a response is not a number, so it cannot ride in the
	/// text: the host stores this one serialized parameter next to it, and a load
	/// puts each payload back into the slot its node reads.
	pub fn slot_state_bytes(&self) -> Vec<u8> {
		let stamp = self.slots.stamp();

		// A project saves far more often than a table or a sample changes, and the
		// blob is only a copy of what the slots already hold: keep the encoded one
		// until the slots move.
		if let Ok(cache) = self.slot_cache.lock()
			&& cache.0.as_ref() == Some(&stamp)
		{
			return cache.1.clone();
		}

		let bytes = to_binary(&self.slots.capture()).unwrap_or_default();

		if let Ok(mut cache) = self.slot_cache.lock() {
			cache.0 = Some(stamp);
			cache.1 = bytes.clone();
		}

		bytes
	}

	/// Puts a saved blob back into the slots; false when it does not decode.
	pub fn set_slot_state(&self, bytes: &[u8]) -> bool {
		let Ok(state) = from_binary::<i_am_patch::state::SlotState>(bytes.to_vec()) else {
			return false;
		};

		self.slots.restore(&state);
		self.bump();
		true
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

	/// Records what each modulated node ended this block at.
	///
	/// The audio thread tries the lock: a page reading the list costs one block
	/// of the picture, where waiting for it would cost the sound. The published
	/// lists are kept between blocks, so a steady patch allocates nothing here.
	pub fn set_values(&self, values: &NodeValues) {
		let Ok(mut published) = self.values.try_lock() else {
			return;
		};

		published.truncate(values.len());

		while published.len() < values.len() {
			published.push((0, Vec::new()));
		}

		for (slot, (node, list)) in published.iter_mut().zip(values) {
			slot.0 = *node;
			slot.1.clear();
			slot.1.extend_from_slice(list);
		}
	}

	/// What the audio thread last published, for the page's pictures.
	pub fn values(&self) -> NodeValues {
		self.values.lock().expect("the values lock").clone()
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

/// The graph a fresh effect starts with: the input, a filter the LFO sweeps, the
/// output.
///
/// An effect is inserted on a track and has to pass that track's audio, so its default
/// is a working signal path, not an instrument: the demo graph would open an effect on
/// a synth that ignores its input and plays nothing the host ever sends it.
pub fn demo_effect() -> Graph {
	let mut graph = Graph::new();
	let input = graph.add_node("input:audio", (0.0, 120.0)).expect("input");
	let filter = graph
		.add_node("mod:filter", (340.0, 120.0))
		.expect("filter");
	let lfo = graph.add_node("lfo", (340.0, 580.0)).expect("lfo");
	let out = graph.add_node("out:audio", (680.0, 120.0)).expect("out");

	graph.connect((input, 0), (filter, 0));
	graph.connect((filter, 0), (out, 0));
	// The same third-of-the-range sweep the instrument's demo uses: about two octaves
	// either way, which is what a modulation looks like on the card.
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
	/// The nodes a modulation moves, for the values the page draws.
	modulated: Vec<u32>,
	/// Those values, reused each block.
	values: NodeValues,
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
			modulated: Vec::new(),
			values: Vec::new(),
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

		// The cards whose parameters a modulation moves are the ones whose values
		// the page keeps drawing, and the set moves with the graph.
		self.modulated.clear();

		for link in &graph.modulations {
			let node = link.target.0.0;

			if !self.modulated.contains(&node) {
				self.modulated.push(node);
			}
		}

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

	/// Copies the values each modulated node ended the block at, for the page.
	///
	/// A voice node only has a value while a voice plays it: with no voice there
	/// is no value to draw, and the card keeps the base picture.
	fn publish_values(&mut self) {
		if self.modulated.is_empty() {
			return;
		}

		let mut values = std::mem::take(&mut self.values);

		values.truncate(self.modulated.len());

		for (index, node) in self.modulated.iter().copied().enumerate() {
			if values.len() == index {
				values.push((node, Vec::new()));
			}

			values[index].0 = node;

			let id = i_am_patch::graph::NodeId(node);
			let list = &mut values[index].1;

			let known = match (&self.poly, &self.engine) {
				(Some(poly), _) => poly.values_into(id, list),
				(None, Some(engine)) => engine.values_into(id, list),
				(None, None) => false,
			};

			if !known {
				list.clear();
			}
		}

		self.shared.set_values(&values);
		self.values = values;
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

			// The values a modulation left each parameter at, before the page asks.
			self.publish_values();

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

	/// A host cannot add a parameter later, so the table is the same sixty-four
	/// whatever the graph holds.
	#[test]
	fn a_host_sees_sixty_four_macros() {
		let shared = PatcherShared::default();

		assert_eq!(shared.params.len(), i_am_patch::MACRO_COUNT);
		assert_eq!(shared.params.query_param_id(0), Some("macro1"));
		assert_eq!(
			shared.params.query_param_id(i_am_patch::MACRO_COUNT - 1),
			Some("macro64")
		);
	}

	/// The host writes an atomic and the audio thread copies it into the pad the
	/// macro nodes read; that copy is the whole of the host path.
	#[test]
	fn a_host_value_reaches_the_pad() {
		let shared = PatcherShared::default();
		assert_eq!(shared.slots.macros().get(7), 0.0);

		shared
			.params
			.set("macro8", SetValue::Float(0.75), Ordering::SeqCst);
		shared.pull_macros();

		assert_eq!(shared.slots.macros().get(7), 0.75);
	}

	/// The page takes the same path a host does, which is what makes a drag on the
	/// page look like automation to the host.
	#[test]
	fn the_page_writes_a_macro_through_the_map() {
		use i_am_plugin::wry::ProtocolRequest;

		let shared = PatcherShared::default();
		let request = ProtocolRequest {
			method: "POST",
			path: "/macro",
			query: Some("index=3&value=0.5"),
			body: &[],
		};

		let response = crate::web::route(&request, &shared).expect("the route answers");
		assert_eq!(response.status, 200);

		shared.pull_macros();
		assert_eq!(shared.slots.macros().get(2), 0.5);

		let read = ProtocolRequest {
			method: "GET",
			path: "/macros",
			query: None,
			body: &[],
		};
		let response = crate::web::route(&read, &shared).expect("the route answers");
		let json: serde_json::Value = serde_json::from_slice(&response.body).expect("json");
		assert_eq!(json["values"][2], 0.5);
	}

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

	/// The card asks what the engine is running, not only what its knob was left
	/// at: the effect demo sweeps the filter's cutoff with an LFO, and the value
	/// the page draws has to follow that sweep.
	#[test]
	fn the_page_reads_the_value_a_modulation_is_running() {
		let shared = Arc::new(PatcherShared::with_graph(demo_effect()));
		let mut body = Body::new(shared.clone());

		let filter = shared
			.graph()
			.nodes
			.iter()
			.find(|node| node.kind.id == "mod:filter")
			.map(|node| node.id.0)
			.expect("the effect demo has a filter");

		let cutoff = || -> Option<f32> {
			shared.values().into_iter().find_map(|(node, list)| {
				if node != filter {
					return None;
				}

				list.iter()
					.find(|(name, _)| *name == "cutoff")
					.map(|(_, value)| *value)
			})
		};

		// A third of the cutoff's range either way is a sweep of octaves, so a
		// window of a couple of hundred blocks has to show the value moving; a
		// reading of the base would leave it exactly on 1 200 Hz.
		let mut low = f32::MAX;
		let mut high = f32::MIN;

		for _ in 0..200 {
			for _ in 0..BLOCK {
				body.process([0.0; 2], [0.0; 2]);
			}

			if let Some(value) = cutoff() {
				low = low.min(value);
				high = high.max(value);
			}
		}

		assert!(
			high - low > 1.0,
			"the cutoff did not follow its modulation: {low} to {high}"
		);
		assert!(
			(low - 1_200.0).abs() > 0.5 || (high - 1_200.0).abs() > 0.5,
			"the page was handed the base value, not the one the engine runs"
		);
	}
}
