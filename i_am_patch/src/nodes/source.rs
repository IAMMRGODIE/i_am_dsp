//! The control sources, the macro and the map: what a modulation reads.

use super::*;

/// The zero-based macro a node follows, from its one-based `index` parameter.
pub(super) fn macro_index(params: &BTreeMap<&'static str, f32>) -> usize {
	let index = param(params, "index", 1.0).round();
	(index.clamp(1.0, crate::MACRO_COUNT as f32) as usize).saturating_sub(1)
}

/// One of the host's macros, as a control source.
///
/// The value is not a node parameter: the host owns it, the product pumps it into
/// the pad, and a voice copy of this node reads the same pad. That is what keeps a
/// macro out of the "copy the graph for the next voice" rules entirely.
pub(super) struct MacroNode {
	/// The pad the product fills from the host's parameter table.
	pub(super) macros: crate::Macros,
	/// Zero-based, from the node's `index` parameter.
	pub(super) index: usize,
}

impl BlockNode for MacroNode {
	fn set_param(&mut self, name: &str, value: f32) {
		if name == "index" {
			let index = value.round() as i64;
			self.index = index.clamp(1, crate::MACRO_COUNT as i64) as usize - 1;
		}
	}

	fn process(&mut self, _: &mut BlockContext, _: &[Buffer], outputs: &mut [Buffer]) {
		if let Some(Buffer::Control(out)) = outputs.first_mut() {
			*out = self.macros.get(self.index);
		}
	}
}

/// A curve from one control to another.
///
/// The node has no parameters: the curve is the whole of it, and the output
/// is a pure function of the input. That is what makes it a mapping rather
/// than a modulation - a modulation adds to a parameter, a map says how.
pub(super) struct MapNode {
	pub(super) points: Vec<crate::graph::Point>,
}

impl BlockNode for MapNode {
	fn process(&mut self, _: &mut BlockContext, inputs: &[Buffer], outputs: &mut [Buffer]) {
		let (Some(Buffer::Control(input)), Some(Buffer::Control(output))) =
			(inputs.first(), outputs.first_mut())
		else {
			return;
		};

		*output = crate::graph::Point::curve(&self.points, *input);
	}
}

/// A control source played as a graph node.
///
/// The graph wants a node with a port; the rack wants a slot. Both run the same
/// [SourceRuntime] now, so an LFO, envelope, random walk or table set to the
/// same numbers behaves the same on either page. The node keeps the catalog's
/// parameter names and maps them onto [SourceSettings]; the runtime does the
/// rest, including the tempo sync, the random stream and the smoothing.
pub(super) struct SourceNode {
	kind: SourceKind,
	pub(super) settings: SourceSettings,
	runtime: SourceRuntime,
	dirty: bool,
	/// Whether the LFO plays pasted points instead of its preset shape.
	custom: bool,
	/// The base shape parameter, which the engine re-applies every block.
	base_shape: i32,
}

impl SourceNode {
	pub(super) fn new(
		kind: SourceKind,
		rate: usize,
		params: &BTreeMap<&'static str, f32>,
		points: &[crate::graph::Point],
	) -> Self {
		let settings = SourceSettings::default();
		let mut node = Self {
			kind,
			settings: settings.clone(),
			runtime: SourceRuntime::from_settings(kind, false, settings, rate, 0),
			dirty: true,
			custom: false,
			base_shape: 0,
		};

		for (name, value) in params {
			node.set_param(name, *value);
		}

		// A curve pasted on a MIDI node is its note mapping: the note goes in on
		// the x axis and the control value comes out on the y, the way the rack's
		// MIDI source reads its own curve. The points keep their bend, which the
		// patch's curve evaluation applies.
		if kind == SourceKind::Midi && points.len() >= 2 {
			node.settings.midi_note_curve = points
				.iter()
				.map(|point| (point.time.clamp(0.0, 1.0), point.value.clamp(0.0, 1.0)))
				.collect();
			node.runtime.set_midi_note_points(
				points
					.iter()
					.map(|point| (point.time, point.value, point.bend))
					.collect(),
			);
			node.dirty = true;
		}

		// Points pasted on the node override the preset shape, exactly as the
		// rack's custom LFO does.
		if kind == SourceKind::Lfo && !points.is_empty() {
			node.settings.lfo_points = points
				.iter()
				.map(|point| (point.time, point.value, point.bend))
				.collect();
			node.base_shape = node.settings.lfo_shape;
			node.settings.lfo_shape = 4;
			node.custom = true;
			node.dirty = true;
		}

		node
	}

	/// Sets the velocity mapping, which the page keeps apart from the note one.
	pub(super) fn set_velocity_points(&mut self, points: &[crate::graph::Point]) {
		if self.kind != SourceKind::Midi || points.len() < 2 {
			return;
		}

		self.settings.midi_velocity_curve = points
			.iter()
			.map(|point| (point.time.clamp(0.0, 1.0), point.value.clamp(0.0, 1.0)))
			.collect();
		self.runtime.set_midi_velocity_points(
			points
				.iter()
				.map(|point| (point.time, point.value, point.bend))
				.collect(),
		);
		self.dirty = true;
	}
}

impl BlockNode for SourceNode {
	fn set_param(&mut self, name: &str, value: f32) {
		let mut changed = false;

		// The engine re-applies the base parameters every block. A pasted curve
		// outranks the base shape until the shape control is actually moved,
		// otherwise the preset would overwrite the points on the next block.
		if self.kind == SourceKind::Lfo && name == "shape" && self.custom {
			let next = (value.round() as i32).clamp(0, 4);

			if next == self.base_shape {
				return;
			}

			self.custom = false;
		}

		{
			let settings = &mut self.settings;

			match (self.kind, name) {
				(SourceKind::Lfo, "rate") => {
					let next = value.clamp(0.01, 20.0);
					if settings.lfo_rate != next {
						settings.lfo_rate = next;
						changed = true;
					}
				}
				(SourceKind::Lfo, "shape") => {
					let next = (value.round() as i32).clamp(0, 4);
					if settings.lfo_shape != next {
						settings.lfo_shape = next;

						if next <= 3 {
							settings.lfo_points = lfo_preset(next);
						}

						changed = true;
					}
				}
				(SourceKind::Lfo, "one_shot") => {
					let next = value != 0.0;
					if settings.lfo_one_shot != next {
						settings.lfo_one_shot = next;
						changed = true;
					}
				}
				(SourceKind::Lfo, "sync") => {
					let next = value != 0.0;
					if settings.lfo_sync != next {
						settings.lfo_sync = next;
						changed = true;
					}
				}
				(SourceKind::Lfo, "beats") => {
					let next = value.clamp(0.03125, 32.0);
					if settings.lfo_beats != next {
						settings.lfo_beats = next;
						changed = true;
					}
				}
				(SourceKind::Envelope, "attack") => {
					let next = value.clamp(1.0, 500.0);
					if settings.env_attack != next {
						settings.env_attack = next;
						changed = true;
					}
				}
				(SourceKind::Envelope, "release") => {
					let next = value.clamp(1.0, 500.0);
					if settings.env_release != next {
						settings.env_release = next;
						changed = true;
					}
				}
				(SourceKind::Random, "rate") => {
					let next = value.clamp(0.01, 20.0);
					if settings.random_rate != next {
						settings.random_rate = next;
						changed = true;
					}
				}
				(SourceKind::Random, "smooth") => {
					let next = value.clamp(0.0, 500.0);
					if settings.random_smooth != next {
						settings.random_smooth = next;
						changed = true;
					}
				}
				(SourceKind::Random, "sync") => {
					let next = value != 0.0;
					if settings.random_sync != next {
						settings.random_sync = next;
						changed = true;
					}
				}
				(SourceKind::Random, "beats") => {
					let next = value.clamp(0.03125, 32.0);
					if settings.random_beats != next {
						settings.random_beats = next;
						changed = true;
					}
				}
				(SourceKind::Table, "rate") => {
					let next = value.clamp(0.01, 20.0);
					if settings.table_rate != next {
						settings.table_rate = next;
						changed = true;
					}
				}
				(SourceKind::Table, "shape") => {
					let next = (value.round() as i32).clamp(0, 3);
					if settings.table_shape != next {
						settings.table_shape = next;
						changed = true;
					}
				}
				(SourceKind::Table, "morph") => {
					let next = value.clamp(0.0, 1.0);
					if settings.table_morph != next {
						settings.table_morph = next;
						changed = true;
					}
				}
				(SourceKind::Table, "sync") => {
					let next = value != 0.0;
					if settings.table_sync != next {
						settings.table_sync = next;
						changed = true;
					}
				}
				(SourceKind::Table, "beats") => {
					let next = value.clamp(0.03125, 32.0);
					if settings.table_beats != next {
						settings.table_beats = next;
						changed = true;
					}
				}
				(SourceKind::Pitch, "clarity") => {
					let next = value.clamp(0.0, 1.0);
					if settings.pitch_clarity != next {
						settings.pitch_clarity = next;
						changed = true;
					}
				}
				(SourceKind::Pitch, "glide") => {
					let next = value.clamp(0.0, 1000.0);
					if settings.pitch_glide != next {
						settings.pitch_glide = next;
						changed = true;
					}
				}
				(SourceKind::Midi, "mode") => {
					let next = (value.round() as i32).clamp(0, 1);
					if settings.midi_mode != next {
						settings.midi_mode = next;
						changed = true;
					}
				}
				(SourceKind::Midi, "transpose") => {
					let next = (value.round() as i32).clamp(-24, 24);
					if settings.midi_transpose != next {
						settings.midi_transpose = next;
						changed = true;
					}
				}
				(SourceKind::Midi, "fine") => {
					let next = value.clamp(-100.0, 100.0);
					if settings.midi_fine != next {
						settings.midi_fine = next;
						changed = true;
					}
				}
				_ => {}
			}
		}

		if changed {
			self.dirty = true;
		}
	}

	fn process(&mut self, context: &mut BlockContext, inputs: &[Buffer], outputs: &mut [Buffer]) {
		if self.dirty {
			self.runtime
				.update_settings(self.kind, false, self.settings.clone(), context.rate);
			self.dirty = false;
		}

		let frames = context.frames.min(BLOCK);
		let input = audio(inputs, 0);
		let infos = context.context.infos();

		for frame in 0..frames {
			let mono = input.channels[0][frame]
				.abs()
				.max(input.channels[1][frame].abs());
			// The note events belong to the block, not to a sample: hand them to
			// the runtime once.
			let events: &[NoteEvent] = if frame == 0 { &context.events } else { &[] };
			self.runtime.advance(mono, infos, context.rate, events);
		}

		if let Some(Buffer::Control(value)) = outputs.first_mut() {
			*value = self.runtime.value();
		}
	}

	fn phase(&self) -> Option<f32> {
		Some(self.runtime.phase())
	}

	fn midi_position(&self) -> Option<f32> {
		self.runtime.midi_position()
	}
}

/// A modulation node: no audio, only the two atomics and the mode.
pub(super) struct ModNode {
	kind: ModKind,
	depth: Arc<AtomicU32>,
	ratio: Arc<AtomicU32>,
}

impl ModNode {
	pub(super) fn new(kind: ModKind, params: &BTreeMap<&'static str, f32>) -> Self {
		Self {
			kind,
			depth: Arc::new(AtomicU32::new(param(params, "depth", 0.5).to_bits())),
			ratio: Arc::new(AtomicU32::new(param(params, "ratio", 1.0).to_bits())),
		}
	}
}

impl BlockNode for ModNode {
	fn process(&mut self, _: &mut BlockContext, _: &[Buffer], _: &mut [Buffer]) {}

	fn set_param(&mut self, name: &str, value: f32) {
		match name {
			"depth" => self.depth.store(value.to_bits(), Ordering::Relaxed),
			"ratio" => self.ratio.store(value.to_bits(), Ordering::Relaxed),
			_ => {}
		}
	}

	fn constructor_spec(&self) -> Option<ConstructorSpec> {
		Some(ConstructorSpec::Mod {
			kind: self.kind,
			depth: self.depth.clone(),
			ratio: self.ratio.clone(),
		})
	}
}
