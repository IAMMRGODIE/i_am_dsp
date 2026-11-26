//! The stand-in audio thread, so the page's pictures have something to draw.

use super::*;

/// Renders the demo graph on a stand-in clock, so the page's scope has audio.
pub(super) fn spawn_audio(shared: Shared) {
	std::thread::spawn(move || {
		let mut poly: Option<Poly> = None;
		let mut engine: Option<Engine> = None;
		let mut built = 0u64;
		let mut blocks = 0u32;
		let mut taps: Vec<(u32, std::sync::Arc<Scope>, std::sync::Arc<Scope>)> = Vec::new();

		loop {
			let generation = shared.generation.load(Ordering::Relaxed);

			if generation != built {
				// The new engine is built from the graph, which already holds the
				// queued edits, so they are dropped rather than replayed.
				if let Ok(mut edits) = shared.edits.try_lock() {
					edits.clear();
				}

				let graph = shared.graph.lock().expect("graph").clone();

				// The product plays a graph with a voice group through Poly and
				// one without through the plain engine; the preview does the
				// same, so a patch that never used a voice group still has audio.
				if voice_group(&graph).is_empty() {
					poly = None;
					engine = match Engine::new_with(&graph, shared.rate, &shared.slots) {
						Ok(engine) => {
							*shared.error.lock().expect("error") = None;
							Some(engine)
						}
						Err(error) => {
							eprintln!("the demo graph does not play: {error}");
							*shared.error.lock().expect("error") = Some(error.to_string());
							None
						}
					};
				} else {
					engine = None;
					poly = match Poly::new_with(&graph, shared.rate, &shared.slots) {
						Ok(poly) => {
							*shared.error.lock().expect("error") = None;
							Some(poly)
						}
						Err(error) => {
							eprintln!("the demo graph does not play: {error}");
							*shared.error.lock().expect("error") = Some(error.to_string());
							None
						}
					};
				}
				built = generation;
				blocks = 0;
			}

			// The edits a hand made since the last block reach the engine that
			// is already playing, so a knob does not cut the notes.
			if let Ok(mut edits) = shared.edits.try_lock() {
				for edit in edits.drain(..) {
					if let Some(poly) = &mut poly {
						edit.apply_poly(poly);
					} else if let Some(engine) = &mut engine {
						edit.apply(engine);
					}
				}
			}

			let mut events = Vec::new();

			// A note a second, so the voice group has something to play.
			if blocks.is_multiple_of(750) {
				events.push(NoteEvent::NoteOn {
					channel: 0,
					note: 60,
					velocity: 0.8,
				});
			}

			if blocks % 750 == 500 {
				events.push(NoteEvent::NoteOff {
					channel: 0,
					note: 60,
					velocity: 0.0,
				});
			}

			let input = [[0.0f32; 2]; i_am_patch::block::BLOCK];
			let mut output = [[0.0f32; 2]; i_am_patch::block::BLOCK];

			if let Some(poly) = &mut poly {
				poly.set_transport(
					Some(120.0),
					Some(
						blocks as f32 * i_am_patch::block::BLOCK as f32 / shared.rate as f32 * 2.0,
					),
				);
				poly.process(&input, &events, &mut output);

				shared.tap_snapshot_into(&mut taps);

				for (node, scope, input) in &taps {
					if let Some(block) = poly
						.step_of(NodeId(*node))
						.and_then(|step| poly.tap_block(step))
						.or_else(|| poly.tap_voice_block(NodeId(*node)))
					{
						scope.write(&block);
					}

					if let Some(block) = poly
						.step_of(NodeId(*node))
						.and_then(|step| poly.input_block(step))
					{
						input.write(&block);
					}

					if let Some(phase) = poly.control_phase(NodeId(*node)) {
						shared.set_phase(*node, phase);
					}

					if let Some(position) = poly.midi_position(NodeId(*node)) {
						shared.set_midi_input(*node, position);
					}
				}
			} else if let Some(engine) = &mut engine {
				engine.set_transport(
					Some(120.0),
					Some(
						blocks as f32 * i_am_patch::block::BLOCK as f32 / shared.rate as f32 * 2.0,
					),
				);
				engine.process(&input, &events, &mut output);

				shared.tap_snapshot_into(&mut taps);

				for (node, scope, input) in &taps {
					let id = NodeId(*node);
					let block: Option<AudioBlock> =
						engine.step_of(id).and_then(|step| engine.tap_block(step));

					if let Some(block) = block {
						scope.write(&block);
					}

					if let Some(block) =
						engine.step_of(id).and_then(|step| engine.input_block(step))
					{
						input.write(&block);
					}

					if let Some(phase) = engine
						.step_of(id)
						.and_then(|step| engine.control_phase(step))
					{
						shared.set_phase(*node, phase);
					}

					if let Some(position) = engine
						.step_of(id)
						.and_then(|step| engine.midi_position(step))
					{
						shared.set_midi_input(*node, position);
					}
				}
			}

			blocks = blocks.wrapping_add(1);
			std::thread::sleep(std::time::Duration::from_micros(1_300));
		}
	});
}
