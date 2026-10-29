//! The Crusher's WebView2 GUI.
//!
//! The page is served over a custom `crusher://` protocol, which gives it its own
//! origin: `/state` returns a JSON snapshot of the shared parameter map (so host
//! automation shows up), and `window.ipc.postMessage("id=value")` writes a control
//! back into that map.
//!
//! [editor](crate::gui::editor) is generic over the plug-in, so the same page would
//! drive another plug-in with the same control ids.

use std::sync::atomic::Ordering;

use i_am_dsp::prelude::{ParamMap, Parameter, Parameters, SetValue, Value};
use i_am_plugin::{Editor, Plugin};
use i_am_plugin_wry::{ProtocolResponse, WryEditor};

/// The page, with `__PARAMS__` replaced by the control table at open time.
const PAGE: &str = include_str!("gui.html");

/// A control the page draws, derived from the effect's parameter.
#[derive(Clone)]
struct Control {
	id: &'static str,
	label: &'static str,
	unit: &'static str,
	min: f32,
	max: f32,
	log: bool,
	default: f32,
}

/// The controls, in the order the page lays them out: (id, label, unit).
const CONTROLS: [(&str, &str, &str); 6] = [
	("amount", "Amount", "percent"),
	("frequency", "LFO Freq", "hz"),
	("am_amount", "AM Amount", "percent"),
	("am_frequency", "AM Freq", "hz"),
	("input_gain.gain", "Input", "db"),
	("output_gain.gain", "Output", "db"),
];

/// Read the ranges, units and defaults the effect declares for those controls.
fn controls(parameters: &[Parameter]) -> Vec<Control> {
	let mut controls = Vec::with_capacity(CONTROLS.len());

	for (id, label, unit) in CONTROLS {
		let Some(parameter) = parameters.iter().find(|parameter| parameter.identifier == id) else {
			return Vec::new();
		};

		let Value::Float {
			value,
			range,
			logarithmic,
		} = &parameter.value
		else {
			return Vec::new();
		};

		controls.push(Control {
			id,
			label,
			unit,
			min: *range.start(),
			max: *range.end(),
			log: *logarithmic,
			default: *value,
		});
	}

	controls
}

/// The control table as the JavaScript the page runs sees it.
fn params_json(controls: &[Control]) -> String {
	let items: Vec<String> = controls
		.iter()
		.map(|control| {
			format!(
				"{{\"id\":\"{}\",\"label\":\"{}\",\"unit\":\"{}\",\"min\":{},\"max\":{},\"log\":{},\"default\":{}}}",
				control.id, control.label, control.unit, control.min, control.max, control.log, control.default,
			)
		})
		.collect();

	format!("[{}]", items.join(","))
}

/// The current values, as the page's `/state` route.
fn state_json(map: &ParamMap, controls: &[Control]) -> String {
	let mut items = Vec::with_capacity(controls.len());

	for control in controls {
		let Some(value) = map.get(control.id) else {
			continue;
		};

		if let SetValue::Float(value) = value.load(Ordering::Relaxed) {
			items.push(format!("\"{}\":{}", control.id, value));
		}
	}

	format!("{{{}}}", items.join(","))
}

/// Write one `id=value` command from the page into the shared parameter map.
///
/// `Paramed` copies the map into the effect on the next sample, so a control is heard
/// as soon as the page posts it. Malformed commands are ignored: the page is the only
/// writer, and a broken string must not take the audio thread with it.
fn apply_ipc(map: &ParamMap, body: &str) {
	if let Some((id, value)) = body.split_once('=')
		&& let (Some(atomic), Ok(value)) = (map.get(id), value.parse::<f32>())
	{
		atomic.store(value, Ordering::Relaxed);
	}
}

/// Build the web editor for a plug-in with the Crusher's controls.
pub fn editor<P: Plugin>(processor: &P) -> Option<Box<dyn Editor<P>>> {
	let parameters = Parameters::get_parameters(processor);
	let controls = controls(&parameters);

	// A missing control means this page and the effect have drifted apart; refusing
	// the editor is clearer than showing half of one.
	if controls.len() != CONTROLS.len() {
		return None;
	}

	let map = Plugin::param_map(processor);
	let html = PAGE.replace("__PARAMS__", &params_json(&controls));

	let ipc_map = map.clone();
	let state_map = map.clone();
	let state_controls = controls.clone();
	let page = html.clone();

	Some(Box::new(
		WryEditor::<P>::new(html, move |body| {
			apply_ipc(&ipc_map, body);
			None
		})
		.with_custom_protocol("crusher", move |path| match path {
			"/" | "/index.html" => ProtocolResponse::html(page.clone()),
			"/state" => ProtocolResponse::json(state_json(&state_map, &state_controls)),
			_ => ProtocolResponse::not_found(),
		})
		.with_url("crusher://localhost/"),
	))
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::CrusherWeb;

	fn map() -> ParamMap {
		Plugin::param_map(&CrusherWeb::new())
	}

	/// The page's only writer: a command has to reach the map the audio thread reads.
	#[test]
	fn an_ipc_command_writes_the_parameter() {
		let map = map();
		apply_ipc(&map, "amount=0.42");

		assert_eq!(
			map.get("amount").unwrap().load(Ordering::Relaxed).float(),
			Some(0.42)
		);
	}

	/// Nothing a page can post may panic or write past a bad parse.
	#[test]
	fn a_malformed_command_is_ignored() {
		let map = map();

		apply_ipc(&map, "nonsense");
		apply_ipc(&map, "amount=not-a-number");
		apply_ipc(&map, "missing.parameter=1.0");

		assert_eq!(
			map.get("amount").unwrap().load(Ordering::Relaxed).float(),
			Some(0.0)
		);
	}

	/// The page fetches this every poll; every control has to be in it.
	#[test]
	fn the_state_route_has_every_control() {
		let crusher = CrusherWeb::new();
		let controls = controls(&Parameters::get_parameters(&crusher));
		let state = state_json(&Plugin::param_map(&crusher), &controls);

		for control in CONTROLS {
			assert!(state.contains(control.0), "{state} is missing {}", control.0);
		}
	}
}
