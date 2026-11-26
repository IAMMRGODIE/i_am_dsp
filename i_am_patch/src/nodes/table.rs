//! The tables the constructors hand over, and the parameters that reach them.

use super::*;

/// A table read through a warp, with its strength shared with the card.
pub(super) struct WarpedTable {
	pub(super) wave: Box<dyn WaveTable + Send + Sync>,
	pub(super) mode: WarpMode,
	pub(super) amount: Arc<AtomicU32>,
	pub(super) curve: Option<Box<dyn WaveTable + Send + Sync>>,
}

impl Parameters for WarpedTable {
	fn get_parameters(&self) -> Vec<Parameter> {
		Vec::new()
	}

	fn set_parameter(&mut self, _: &str, _: SetValue) -> bool {
		false
	}
}

impl WaveTable for WarpedTable {
	/// The table under the warp is asked too: a product table follows the node that
	/// built it, and a warp between them must not stop the message.
	fn sync(&mut self) {
		self.wave.sync();

		if let Some(curve) = &mut self.curve {
			curve.sync();
		}
	}

	fn sample(&self, t: f32, channel: usize) -> f32 {
		let amount = f32::from_bits(self.amount.load(Ordering::Relaxed));

		let position = match self.mode {
			WarpMode::Bend => bend_phase(amount, t),
			WarpMode::Mirror => mirror_phase(amount, t),
			WarpMode::Sync => sync_phase(amount, t),
			WarpMode::Quantize => quantize_phase(amount, t),
			WarpMode::Remap => match &self.curve {
				Some(curve) => t + (curve.sample(t, channel) - t) * amount,
				None => t,
			},
		};

		self.wave.sample(position.rem_euclid(1.0), channel)
	}
}

/// The values a product table was built from, shared with the table itself.
///
/// A product's parameters are plain fields, so a table built from them is a snapshot:
/// it cannot see a knob move. These atomics are what the table follows instead, and the
/// version says whether anything moved since the last look.
pub(super) struct LiveParams {
	entries: Vec<(&'static str, AtomicU32)>,
	version: AtomicU64,
}

impl LiveParams {
	pub(super) fn new(params: &BTreeMap<&'static str, f32>) -> Self {
		Self {
			entries: params
				.iter()
				.map(|(name, value)| (*name, AtomicU32::new(value.to_bits())))
				.collect(),
			version: AtomicU64::new(1),
		}
	}

	pub(super) fn write(&self, name: &str, value: f32) -> bool {
		for (key, cell) in &self.entries {
			if *key == name {
				cell.store(value.to_bits(), Ordering::Relaxed);
				self.version.fetch_add(1, Ordering::Release);

				return true;
			}
		}

		false
	}
}

/// A product's table, following the node that built it.
///
/// The engine asks every table to sync before it plays a sample, so the check is one
/// atomic load; when the version moved, the values the node holds are written back into
/// the table. That is what makes a knob move audible on the note that is already
/// sounding, instead of only on the next one.
pub(super) struct LiveTable {
	value: Box<dyn WaveTable + Send + Sync>,
	params: Arc<LiveParams>,
	seen: u64,
}

impl Parameters for LiveTable {
	fn get_parameters(&self) -> Vec<Parameter> {
		self.value.get_parameters()
	}

	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		self.value.set_parameter(identifier, value)
	}
}

impl WaveTable for LiveTable {
	fn sample(&self, t: f32, channel: usize) -> f32 {
		self.value.sample(t, channel)
	}

	fn sync(&mut self) {
		let version = self.params.version.load(Ordering::Acquire);

		if version != self.seen {
			self.seen = version;

			for (name, cell) in &self.params.entries {
				apply_param(
					self.value.as_mut(),
					name,
					f32::from_bits(cell.load(Ordering::Relaxed)),
				);
			}
		}

		self.value.sync();
	}
}

/// A product's table, handed to the node that reads it.
pub(super) struct ProductTableNode {
	factory: Box<dyn Fn() -> Box<dyn WaveTable + Send + Sync> + Send + Sync>,
	params: BTreeMap<&'static str, f32>,
	live: Arc<LiveParams>,
}

impl ProductTableNode {
	pub(super) fn new(
		factory: impl Fn() -> Box<dyn WaveTable + Send + Sync> + Send + Sync + 'static,
		params: &BTreeMap<&'static str, f32>,
	) -> Self {
		Self {
			factory: Box::new(factory),
			params: params.clone(),
			live: Arc::new(LiveParams::new(params)),
		}
	}
}

impl BlockNode for ProductTableNode {
	fn process(&mut self, _: &mut BlockContext, _: &[Buffer], _: &mut [Buffer]) {}

	fn set_param(&mut self, name: &str, value: f32) {
		for (key, stored) in &mut self.params {
			if *key == name {
				*stored = value;

				break;
			}
		}

		self.live.write(name, value);
	}

	fn constructor_spec(&self) -> Option<ConstructorSpec> {
		let mut value = (self.factory)();

		apply_params(value.as_mut(), &self.params);

		Some(ConstructorSpec::Table(Box::new(LiveTable {
			value,
			params: self.live.clone(),
			seen: self.live.version.load(Ordering::Acquire),
		})))
	}
}

/// Pushes the card's values into a freshly built table, whatever type each
/// parameter declares.
///
/// A product's parameter ids carry the path from its processor (`oscillator.`
/// for the voice's oscillator); the table is that oscillator alone, so the id is
/// tried with the path stripped too.
pub(super) fn apply_params(table: &mut dyn Parameters, params: &BTreeMap<&'static str, f32>) {
	for (name, value) in params {
		apply_param(table, name, *value);
	}
}

/// Writes one of the card's values into a table, trying the same id shapes.
pub(super) fn apply_param(table: &mut dyn Parameters, name: &str, value: f32) {
	let oscillator = name.strip_prefix("oscillator.").unwrap_or(name);
	let voice = name.strip_prefix("voice.").unwrap_or(name);

	for id in [name, oscillator, voice] {
		if write_param(table, id, value) {
			break;
		}
	}
}

/// Writes one parameter, trying the value types the DSP might declare.
pub(super) fn write_param(table: &mut dyn Parameters, id: &str, value: f32) -> bool {
	table.set_parameter(id, SetValue::Float(value))
		|| table.set_parameter(id, SetValue::Int(value as i32))
		|| table.set_parameter(id, SetValue::Bool(value > 0.5))
}

/// A warp node: no audio, only the strength and the mode.
pub(super) struct WarpNode {
	mode: WarpMode,
	param: &'static str,
	amount: Arc<AtomicU32>,
}

impl WarpNode {
	pub(super) fn new(mode: WarpMode, name: &'static str, params: &BTreeMap<&'static str, f32>) -> Self {
		Self {
			mode,
			param: name,
			amount: Arc::new(AtomicU32::new(param(params, name, 0.0).to_bits())),
		}
	}
}

impl BlockNode for WarpNode {
	fn process(&mut self, _: &mut BlockContext, _: &[Buffer], _: &mut [Buffer]) {}

	fn set_param(&mut self, name: &str, value: f32) {
		if name == self.param {
			self.amount.store(value.to_bits(), Ordering::Relaxed);
		}
	}

	fn constructor_spec(&self) -> Option<ConstructorSpec> {
		Some(ConstructorSpec::Warp {
			mode: self.mode,
			amount: self.amount.clone(),
		})
	}
}

/// The table a modulation's input port names.
pub(super) fn build_table(
	nodes: &[Box<dyn BlockNode>],
	steps: &BTreeMap<crate::graph::NodeId, usize>,
	graph: &crate::graph::Graph,
	id: crate::graph::NodeId,
	port: usize,
) -> Option<Box<dyn WaveTable + Send + Sync>> {
	build_table_node(nodes, steps, graph, table_source(graph, id, port)?)
}

/// The node an input port of `id` reads from.
pub(super) fn table_source(
	graph: &crate::graph::Graph,
	id: crate::graph::NodeId,
	port: usize,
) -> Option<crate::graph::NodeId> {
	graph
		.edges
		.iter()
		.find(|edge| edge.to.0 == id && edge.to.1 == port)
		.map(|edge| edge.from.0)
}

/// Resolves a table node, including the warps that read another table.
pub(crate) fn build_table_node(
	nodes: &[Box<dyn BlockNode>],
	steps: &BTreeMap<crate::graph::NodeId, usize>,
	graph: &crate::graph::Graph,
	id: crate::graph::NodeId,
) -> Option<Box<dyn WaveTable + Send + Sync>> {
	let step = *steps.get(&id)?;

	match nodes.get(step)?.constructor_spec()? {
		ConstructorSpec::Table(table) => Some(table),
		// An oscillator port feeding a table port is read through the inverse
		// adapter: the table is the oscillator's cycle at 1 Hz.
		ConstructorSpec::Osc(oscillator) => Some(Box::new(OscTable::<2, _>(oscillator))),
		ConstructorSpec::Mod { .. } => {
			let oscillator = build_oscillator(nodes, steps, graph, id)?;

			Some(Box::new(OscTable::<2, _>(oscillator)))
		}
		ConstructorSpec::Warp { mode, amount } => {
			let wave = build_table_node(nodes, steps, graph, table_source(graph, id, 0)?)?;
			let curve = if matches!(mode, WarpMode::Remap) {
				table_source(graph, id, 1)
					.and_then(|source| build_table_node(nodes, steps, graph, source))
			} else {
				None
			};

			Some(Box::new(WarpedTable {
				wave,
				mode,
				amount,
				curve,
			}))
		}
	}
}
