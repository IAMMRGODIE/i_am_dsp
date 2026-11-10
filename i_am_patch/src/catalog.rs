//! The node catalog the patch page draws its menu and cards from.
//!
//! The product pages describe a product once, in its faceplate and its
//! parameter metadata; this module is the other reader of that description, so
//! a product's sections, pages and heroes reach the patch card through the same
//! code the product's own page uses. Both the editor example and the plug-in
//! serve this one catalog, so the two cannot drift.

use serde_json::{Value, json};

use crate::graph::{NodeDesc, PortDesc, SignalKind, catalog};

/// Every node kind, as the page reads it.
pub fn kinds_json() -> Vec<Value> {
	catalog().iter().map(kind_json).collect()
}

/// The product a kind is built from, when it is one.
///
/// The bare `filter` and `gain` names are the rack's, not the graph's; the
/// graph's products are `mod:` and `synth:`.
pub fn product_module(kind: &str) -> Option<&str> {
	match kind {
		"filter" => Some("filter"),
		"gain" => Some("gain"),
		_ => kind.strip_prefix("mod:"),
	}
}

fn kind_json(kind: &NodeDesc) -> Value {
	let port = |port: &PortDesc| {
		json!({
			"name": port.name,
			"kind": match port.kind {
				SignalKind::Midi => "midi",
				SignalKind::Control => "control",
				SignalKind::Audio(_) => "audio",
			},
			"optional": port.optional,
		})
	};

	let mut controls: Vec<Value> = Vec::new();
	let mut hero: Option<String> = None;
	let mut mode: Option<&'static str> = None;
	let mut header: &'static [&'static str] = &[];
	let mut visible: Option<Vec<(i32, Vec<&'static str>)>> = None;
	let mut switches: Vec<Value> = Vec::new();
	let mut accent = "#e07a2f";
	let mut face: Option<(f32, f32)> = None;
	let mut pages: Vec<Value> = Vec::new();
	let mut grouped: Vec<Value> = Vec::new();
	let mut title: Option<String> = None;

	if let Some(module) = product_module(kind.id)
		&& let Some(descriptor) = i_am_effects::catalog::get(module)
	{
		let sections = i_am_web_gui::sections(&descriptor.parameters());
		hero = sections.iter().find_map(|section| section.hero.clone());
		mode = descriptor.mode;
		header = descriptor.header;
		visible = descriptor.visible.map(|rows| {
			rows.iter()
				.map(|(mode, names)| (*mode, names.to_vec()))
				.collect()
		});
		accent = descriptor.accent;
		face = i_am_effects::catalog::face(descriptor.id);
		switches = descriptor
			.switches
			.iter()
			.map(|row| json!({ "control": row.control, "on": row.on, "off": row.off }))
			.collect::<Vec<Value>>();

		for section in &sections {
			for control in &section.controls {
				controls.push(serde_json::to_value(control).unwrap_or(Value::Null));
			}
		}

		grouped = sections_json(&sections, &[]);
		pages = pages_json(descriptor_pages(module), &sections);
	}

	// A synth product takes its card from its own faceplate and schema.
	if let Some(synth) = kind.id.strip_prefix("synth:")
		&& let Some(descriptor) = crate::synths::synth(synth)
	{
		let faceplate = (descriptor.faceplate)();
		let sections = i_am_web_gui::sections(&(descriptor.parameters)());
		hero = sections.iter().find_map(|section| section.hero.clone());
		accent = faceplate.accent;
		face = Some(faceplate.size);
		title = Some(faceplate.title.to_string());
		mode = faceplate.mode;
		header = faceplate.header.unwrap_or(&[]);
		visible = faceplate.visible.map(|rows| {
			rows.iter()
				.map(|(mode, names)| (*mode, names.to_vec()))
				.collect()
		});
		switches = faceplate
			.switches
			.unwrap_or(&[])
			.iter()
			.map(|row| json!({ "control": row.control, "on": row.on, "off": row.off }))
			.collect::<Vec<Value>>();
		pages = pages_json(faceplate.pages.unwrap_or(&[]), &sections);
		let hidden = faceplate.hidden.unwrap_or(&[]);
		grouped = sections_json(&sections, hidden);

		for section in &sections {
			for control in &section.controls {
				if hidden.contains(&control.id.as_str()) {
					continue;
				}

				controls.push(serde_json::to_value(control).unwrap_or(Value::Null));
			}
		}
	}

	// A source with a Sync switch trades its free rate for beats.
	switches = match kind.id {
		"lfo" | "random" | "table" => vec![json!({
			"control": "sync",
			"on": ["beats"],
			"off": ["rate"],
		})],
		_ => switches,
	};

	// The control sources keep the rack's colours.
	accent = match kind.id {
		"lfo" => "#4c8dff",
		"envelope" => "#35c77a",
		"random" => "#ff9a3d",
		"table" => "#c8e04a",
		"pitch" => "#b08cff",
		"midi" => "#ff6f91",
		_ => accent,
	};

	json!({
		"id": kind.id,
		"name": kind.name,
		"inputs": kind.inputs.iter().map(port).collect::<Vec<_>>(),
		"outputs": kind.outputs.iter().map(port).collect::<Vec<_>>(),
		"params": kind.params.iter().map(|p| json!([p.name, p.default])).collect::<Vec<_>>(),
		"param_meta": kind.params.iter().map(|p| json!({
			"name": p.name,
			"min": p.min,
			"max": p.max,
			"log": p.logarithmic,
			"step": p.step,
			"choices": p.choices,
			"boolean": p.boolean,
		})).collect::<Vec<_>>(),
		"controls": controls,
		"hero": hero,
		"mode": mode,
		"header": header,
		"visible": visible,
		"switches": switches,
		"accent": accent,
		"face": face.map(|(width, height)| [width, height]),
		"title": title,
		"pages": pages,
		"sections": grouped,
	})
}

/// The sections a product's page draws, with the hidden controls left out.
fn sections_json(sections: &[i_am_web_gui::Section], hidden: &[&str]) -> Vec<Value> {
	sections
		.iter()
		.map(|section| {
			json!({
				"name": section.name.clone().unwrap_or_default(),
				"hero": section.hero.clone(),
				"controls": section
					.controls
					.iter()
					.filter(|control| !hidden.contains(&control.id.as_str()))
					.map(|control| serde_json::to_value(control).unwrap_or(Value::Null))
					.collect::<Vec<Value>>(),
			})
		})
		.collect()
}

/// The pages a product splits into, each naming the sections it shows.
fn pages_json(pages: &[i_am_web_gui::Page], sections: &[i_am_web_gui::Section]) -> Vec<Value> {
	pages
		.iter()
		.map(|page| {
			let hero = page.sections.iter().find_map(|name| {
				sections
					.iter()
					.find(|section| section.name.as_deref() == Some(*name))
					.and_then(|section| section.hero.clone())
			});

			json!({ "name": page.name, "sections": page.sections, "hero": hero })
		})
		.collect()
}

/// The pages a product registers with its faceplate.
fn descriptor_pages(module: &str) -> &'static [i_am_web_gui::Page] {
	match module {
		"compressor" => i_am_effects::COMPRESSOR_PAGES,
		_ => &[],
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// Every page names sections the kind has, and every section names controls
	/// the kind has: a page pointing at a section that is not there draws an
	/// empty card, which is the drift this catalog exists to prevent.
	#[test]
	fn the_catalog_is_internally_consistent() {
		for kind in kinds_json() {
			let id = kind["id"].as_str().unwrap_or_default();
			let sections = kind["sections"].as_array().cloned().unwrap_or_default();
			let controls = kind["controls"].as_array().cloned().unwrap_or_default();
			let control_ids = controls
				.iter()
				.filter_map(|control| control["id"].as_str().map(str::to_string))
				.collect::<Vec<_>>();

			for section in &sections {
				let name = section["name"].as_str().unwrap_or_default();

				for control in section["controls"].as_array().cloned().unwrap_or_default() {
					let control_id = control["id"].as_str().unwrap_or_default();
					assert!(
						control_ids.iter().any(|id| id == control_id),
						"{id}: section {name} names control {control_id}, which the kind does not have"
					);
				}
			}

			for page in kind["pages"].as_array().cloned().unwrap_or_default() {
				for section in page["sections"].as_array().cloned().unwrap_or_default() {
					let section = section.as_str().unwrap_or_default();
					assert!(
						sections
							.iter()
							.any(|candidate| candidate["name"].as_str() == Some(section)),
						"{id}: page names section {section}, which the kind does not have"
					);
				}
			}
		}
	}

	/// The compressor's two pages reach the patch card, which is where the
	/// product page and the patch card used to disagree.
	#[test]
	fn the_compressor_keeps_its_pages() {
		let compressor = kinds_json()
			.into_iter()
			.find(|kind| kind["id"] == "mod:compressor")
			.expect("the compressor is in the catalog");
		let pages = compressor["pages"].as_array().expect("pages");
		let names = pages
			.iter()
			.map(|page| page["name"].as_str().unwrap_or_default())
			.collect::<Vec<_>>();

		assert_eq!(names, ["Compressor", "Detector"]);
	}
}
