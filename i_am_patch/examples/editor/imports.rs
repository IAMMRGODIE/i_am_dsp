//! The file routes: a sample, a table, a canvas or a response.

use super::*;

/// Fits one recorded note into a Modal node, the way the product's route does.
/// The three files a node can take besides a fitted note.
#[derive(Clone, Copy)]
pub(super) enum Import {
	/// The sampler's sample.
	Sample,
	/// The shapes' wavetable.
	Waves,
	/// The convolver's impulse response.
	Ir,
}

/// Answers the sampler, the shapes and the convolver: a GET describes what the
/// slot holds, a POST loads a file into it.
pub(super) fn import_route(shared: &Shared, method: &str, path: &str, body: &[u8], which: Import) -> String {
	let query = path.split_once('?').map(|(_, query)| query).unwrap_or("");
	let node = query
		.split('&')
		.find_map(|pair| pair.strip_prefix("node="))
		.and_then(|value| value.parse::<u32>().ok())
		.unwrap_or(0);
	let name = query
		.split('&')
		.find_map(|pair| pair.strip_prefix("name="))
		.map(i_am_web_gui::percent_decode)
		.filter(|name| !name.is_empty())
		.unwrap_or_else(|| "file".to_string());

	let action = query
		.split('&')
		.find_map(|pair| pair.strip_prefix("action="))
		.filter(|action| !action.is_empty());

	match which {
		Import::Sample => {
			let slot = shared.slots.sample(node);

			if method == "GET" {
				return i_am_patch::sample_preview(&slot);
			}

			if let Some(action) = action {
				let number = |key: &str| {
					query
						.split('&')
						.find_map(|pair| pair.strip_prefix(key)?.strip_prefix('='))
						.and_then(|value| value.parse::<f32>().ok())
				};
				let at = number("at").unwrap_or(0.0);
				let to = number("to").unwrap_or(at);

				return match action {
					"slices" => match i_am_patch::generate_slices(&slot, "grid") {
						Ok(count) => json!({ "ok": true, "slices": count }).to_string(),
						Err(error) => json!({ "ok": false, "error": error }).to_string(),
					},
					_ => match i_am_patch::edit_slices(&slot, action, at, to) {
						Ok(count) => json!({ "ok": true, "slices": count }).to_string(),
						Err(error) => json!({ "ok": false, "error": error }).to_string(),
					},
				};
			}

			match i_am_patch::import_sample(&slot, body, &name) {
				Ok((name, samples, seconds)) => json!({
					"ok": true,
					"name": name,
					"samples": samples,
					"seconds": seconds,
				})
				.to_string(),
				Err(error) => json!({ "ok": false, "error": error }).to_string(),
			}
		}
		Import::Waves => {
			let slot = shared.slots.waves(node);

			if method == "GET" {
				return i_am_patch::waves_preview(&slot);
			}

			if let Some(action) = action {
				let frame = query
					.split('&')
					.find_map(|pair| pair.strip_prefix("frame="))
					.and_then(|frame| frame.parse::<usize>().ok());

				return match i_am_patch::edit_waves(&slot, action, frame) {
					Ok(count) => json!({ "ok": true, "count": count }).to_string(),
					Err(error) => json!({ "ok": false, "error": error }).to_string(),
				};
			}

			let mode = query
				.split('&')
				.find_map(|pair| pair.strip_prefix("mode="))
				.filter(|mode| !mode.is_empty());

			match i_am_patch::import_waves(&slot, body, &name, mode) {
				Ok((name, frames)) => {
					json!({ "ok": true, "name": name, "frames": frames }).to_string()
				}
				Err(error) => json!({ "ok": false, "error": error }).to_string(),
			}
		}
		Import::Ir => {
			let slot = shared.slots.ir(node);

			if method == "GET" {
				return slot.json();
			}

			if let (Some(fade_in), Some(fade_out)) = (
				query
					.split('&')
					.find_map(|pair| pair.strip_prefix("fade_in="))
					.and_then(|value| value.parse::<f32>().ok()),
				query
					.split('&')
					.find_map(|pair| pair.strip_prefix("fade_out="))
					.and_then(|value| value.parse::<f32>().ok()),
			) {
				slot.set_fades(i_am_effects::IrFades {
					fade_in: fade_in.clamp(0.0, 1.0),
					fade_out: fade_out.clamp(0.0, 1.0),
				});

				return json!({ "ok": true }).to_string();
			}

			match i_am_effects::decode_ir(body, slot.host_rate(), name) {
				Ok(ir) => {
					slot.set(std::sync::Arc::new(ir));
					json!({ "ok": true }).to_string()
				}
				Err(error) => {
					slot.fail(error.clone());
					json!({ "ok": false, "error": error }).to_string()
				}
			}
		}
	}
}

pub(super) fn sample_route(shared: &Shared, path: &str, body: &[u8]) -> Value {
	let query = path.split_once('?').map(|(_, query)| query).unwrap_or("");
	let node = query
		.split('&')
		.find_map(|pair| pair.strip_prefix("node="))
		.and_then(|value| value.parse::<u32>().ok())
		.unwrap_or(0);
	let name = query
		.split('&')
		.find_map(|pair| pair.strip_prefix("name="))
		.unwrap_or("sample");

	match i_am_dsp::prelude::fit_sample(body, i_am_dsp::prelude::note_of_name(name)) {
		Ok(fitted) => {
			shared
				.slots
				.fitted(node)
				.set(i_am_dsp::prelude::ImportedTable {
					name: name.to_string(),
					bytes: fitted.bytes,
				});

			if let Ok(mut graph) = shared.graph.lock() {
				let id = i_am_patch::graph::NodeId(node);

				graph.set_param(id, "instrument", 5.0);
				graph.set_param(id, "derive", 1.0);
				graph.set_param(id, "transpose", 0.0);
			}

			shared.generation.fetch_add(1, Ordering::Relaxed);

			json!({
				"ok": true,
				"note": fitted.note,
				"frequency": fitted.frequency,
				"modes": fitted.modes,
			})
		}
		Err(error) => json!({ "ok": false, "error": error }),
	}
}
