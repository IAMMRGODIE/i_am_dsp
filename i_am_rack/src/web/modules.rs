//! The per-module routes the rack cards poll.

use crate::shared::RackShared;

/// The module id a page asked about, from the query string.
pub(super) fn module_id(request: &i_am_web_gui::ProtocolRequest<'_>) -> Option<u32> {
	request
		.query?
		.split('&')
		.find_map(|pair| pair.strip_prefix("id="))
		.and_then(|value| value.parse::<u32>().ok())
}

/// The note a module last heard, and the mask a pitch-tracking effect is tuning
/// to, when it publishes one.
pub(super) fn module_note(
	request: &i_am_web_gui::ProtocolRequest<'_>,
	shared: &RackShared,
) -> Option<i_am_web_gui::ProtocolResponse> {
	let id = module_id(request)?;

	let (note, held, allowed) = match shared.existing_scope(id) {
		Some(scope) => (scope.note(), scope.notes(), scope.allowed_notes()),
		None => (-1, 0, -1),
	};

	// A resonator can hold a chord, so the page draws every note, not the last.
	let notes = (0..128)
		.filter(|pitch| held & (1u128 << pitch) != 0)
		.map(|pitch| pitch.to_string())
		.collect::<Vec<_>>()
		.join(",");

	Some(i_am_web_gui::ProtocolResponse::json(format!(
		"{{\"note\":{note},\"notes\":[{notes}],\"allowed\":{}}}",
		if allowed < 0 {
			"null".to_string()
		} else {
			allowed.to_string()
		}
	)))
}

/// The audio one module saw and produced, in the same shape as the rack's own
/// scope documents: three rings for the mono one, four for the stereo pair.
/// The band meter a module's card draws: the OTT's three gain reductions, or a
/// vocoder's input and output level per band.
pub(super) fn module_bands(
	request: &i_am_web_gui::ProtocolRequest<'_>,
	shared: &RackShared,
) -> Option<i_am_web_gui::ProtocolResponse> {
	let id = module_id(request)?;
	let (bands, gains, input, output) = match shared.existing_meters(id) {
		Some(meters) => meters.snapshot(),
		None => (0, false, Vec::new(), Vec::new()),
	};
	let row = |values: &[f32]| {
		values
			.iter()
			.map(|value| format!("{value:.4}"))
			.collect::<Vec<_>>()
			.join(",")
	};

	Some(i_am_web_gui::ProtocolResponse::json(format!(
		"{{\"bands\":{bands},\"gains\":{gains},\"input\":[{}],\"output\":[{}]}}",
		row(&input),
		row(&output)
	)))
}

/// The impulse response a convolver module holds: a GET describes it, a POST
/// loads the bytes the page picked as a file.
pub(super) fn module_ir(
	request: &i_am_web_gui::ProtocolRequest<'_>,
	shared: &RackShared,
) -> Option<i_am_web_gui::ProtocolResponse> {
	let id = module_id(request)?;
	let slot = shared.existing_ir(id)?;

	match request.method {
		"GET" => Some(i_am_web_gui::ProtocolResponse::json(slot.json())),
		"POST" => {
			let name = request
				.query
				.and_then(|query| {
					query
						.split('&')
						.find_map(|pair| pair.strip_prefix("name="))
						.map(percent_decode)
				})
				.filter(|name| !name.is_empty())
				.unwrap_or_else(|| "impulse".to_string());

			match i_am_effects::decode_ir(request.body, slot.host_rate(), name) {
				Ok(ir) => {
					// The preset has to carry the room with it, so the state
					// keeps a copy of what the slot now plays.
					if let Some(module) = shared.lock().module_mut(id) {
						module.ir = Some(ir.clone());
					}

					slot.set(std::sync::Arc::new(ir));
					Some(i_am_web_gui::ProtocolResponse::json("{\"ok\":true}"))
				}
				Err(error) => {
					slot.fail(error.clone());
					Some(i_am_web_gui::ProtocolResponse::json(format!(
						"{{\"error\":{}}}",
						serde_json::to_string(&error)
							.unwrap_or_else(|_| "\"load failed\"".to_string())
					)))
				}
			}
		}
		_ => None,
	}
}

/// Decodes the percent escapes a page sends in a query value.
pub(super) use i_am_web_gui::percent_decode;

/// The script a scripted module runs: a GET describes it, a POST replaces it
/// and asks the worker to render for the parameters the module currently has.
pub(super) fn module_script(
	request: &i_am_web_gui::ProtocolRequest<'_>,
	shared: &RackShared,
) -> Option<i_am_web_gui::ProtocolResponse> {
	let id = module_id(request)?;
	let script = shared.existing_script(id)?;

	match request.method {
		"GET" => Some(i_am_web_gui::ProtocolResponse::json(script.json())),
		"POST" => {
			let source = String::from_utf8_lossy(request.body).to_string();
			script.set_source(source.clone());

			let rate = shared.rate.load(std::sync::atomic::Ordering::Relaxed) as f32;
			let (params, extra) = {
				let mut state = shared.lock();
				let module = state.module_mut(id)?;
				// The preset keeps the script, not only the worker.
				module.script = Some(source);
				let value = |name: &str| module.params.get(name).copied().unwrap_or(0.0) as f32;
				let extra = match module.kind {
					crate::module::ModuleKind::ScriptMapper => rate,
					_ => 10.0_f32.powf(value("in_gain") / 20.0),
				};

				([value("p1"), value("p2"), value("p3"), value("p4")], extra)
			};

			script.request(params, extra);
			Some(i_am_web_gui::ProtocolResponse::json("{\"ok\":true}"))
		}
		_ => None,
	}
}

/// The curve a script currently renders, for the card's picture.
pub(super) fn module_curve(
	request: &i_am_web_gui::ProtocolRequest<'_>,
	shared: &RackShared,
) -> Option<i_am_web_gui::ProtocolResponse> {
	let id = module_id(request)?;
	let script = shared.existing_script(id)?;

	Some(i_am_web_gui::ProtocolResponse::json(script.curve_json()))
}

pub(super) fn module_document(
	request: &i_am_web_gui::ProtocolRequest<'_>,
	shared: &RackShared,
	stereo: bool,
) -> Option<i_am_web_gui::ProtocolResponse> {
	let id = module_id(request)?;
	let channels = if stereo {
		i_am_web_gui::Channels::Stereo
	} else {
		i_am_web_gui::Channels::Mono
	};
	let scope = shared.scope(id, channels);

	Some(i_am_web_gui::ProtocolResponse {
		status: 200,
		content_type: "application/octet-stream",
		body: if stereo {
			scope.stereo_snapshot()
		} else {
			scope.snapshot()
		},
	})
}
