//! Turning parameters into the JSON a web faceplate runs on.
//!
//! The page never guesses where a control goes: the parameter metadata already
//! says which section it belongs to, whether it is a knob or a slider, how big it
//! wants to be and whether its section has a picture. This module is the one place
//! that reads that metadata and writes it down for JavaScript.

use std::fmt::Write as _;
use std::sync::atomic::Ordering;

use i_am_dsp::prelude::{ParamMap, Parameter, SetValue, Unit, Value};

/// What a web faceplate knows about the plug-in it belongs to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Faceplate {
	/// The name in the window's own header.
	pub title: &'static str,
	/// The identity colour, as #rrggbb.
	pub accent: &'static str,
	/// The logical size the product asks the host for. The page also uses it to
	/// draw itself at real size when it is opened in a browser for a look.
	pub size: (f32, f32),
	/// A parameter the page draws as a mode switch in the header instead of a
	/// control in the body. Filter and Drive have several algorithms and only one
	/// of them is active at a time, which is a switch, not a knob.
	pub mode: Option<&'static str>,
}

impl Faceplate {
	/// A faceplate with a title, an identity colour and a size.
	pub const fn new(title: &'static str, accent: &'static str, size: (f32, f32)) -> Self {
		Self {
			title,
			accent,
			size,
			mode: None,
		}
	}

	/// Draw the named parameter as a mode switch in the header.
	pub const fn with_mode(mut self, mode: &'static str) -> Self {
		self.mode = Some(mode);
		self
	}
}

/// What a control holds, which decides how a value is read and written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
	/// A floating-point parameter.
	Float,
	/// A whole-number parameter.
	Int,
	/// An on/off parameter.
	Bool,
}

/// One control, as the page draws it.
#[derive(Debug, Clone)]
pub struct Control {
	/// The parameter identifier, which is also the key in /state.
	pub id: String,
	/// The name a person reads.
	pub label: String,
	/// What the value is.
	pub kind: Kind,
	/// How the value is shown: none, ms, hz, beats, s, samples, db or percent.
	pub unit: &'static str,
	/// The bottom of the range.
	pub min: f64,
	/// The top of the range.
	pub max: f64,
	/// Where a double click puts it.
	pub default: f64,
	/// Whether dragging is logarithmic, which is what a frequency or a gain wants.
	pub log: bool,
	/// Whether the middle of the range is the neutral point.
	pub bipolar: bool,
	/// The interval a release snaps to, if any.
	pub step: Option<f64>,
	/// knob, slider, toggle or choice.
	pub widget: &'static str,
	/// small, medium or large.
	pub size: &'static str,
	/// Where it sits in its section, when it said.
	pub order: Option<u16>,
	/// The names of a choice parameter's values.
	pub choices: Option<Vec<String>>,
}

/// A group of controls, and the picture that group shows above them.
#[derive(Debug, Clone)]
pub struct Section {
	/// The section's title, or nothing for a faceplate that only has one group.
	pub name: Option<String>,
	/// The hero this section asked for, if any.
	pub hero: Option<String>,
	/// The controls, in the order they are drawn.
	pub controls: Vec<Control>,
}

/// The sections an effect's parameters describe.
///
/// Sections appear in the order their first parameter was declared, and the
/// members of a section keep the order they declared; members that declared none
/// come last. This is the same arrangement the iced faceplate uses, so the two
/// front ends agree on what a faceplate looks like.
pub fn sections(parameters: &[Parameter]) -> Vec<Section> {
	let mut sections: Vec<Section> = Vec::new();

	for parameter in parameters {
		let Some(control) = control(parameter) else {
			continue;
		};

		let name = parameter.meta.layout.group.as_ref().map(|group| group.to_string());
		let hero = parameter.meta.layout.hero.as_ref().map(|hero| hero.to_string());

		let index = match sections
			.iter()
			.position(|section| section.name.as_deref() == name.as_deref())
		{
			Some(index) => index,
			None => {
				sections.push(Section {
					name,
					hero: hero.clone(),
					controls: Vec::new(),
				});
				sections.len() - 1
			},
		};

		// The first member that names a hero decides for the section: a hero
		// belongs to a section, and naming it twice would be a contradiction.
		if sections[index].hero.is_none() {
			sections[index].hero = hero;
		}

		sections[index].controls.push(control);
	}

	for section in &mut sections {
		// A stable sort, so everything without a declared order keeps the order it
		// was declared in.
		section.controls.sort_by_key(|control| control.order.unwrap_or(u16::MAX));
	}

	sections
}

/// One parameter as a control, or nothing when it is not one a person can move.
fn control(parameter: &Parameter) -> Option<Control> {
	let meta = &parameter.meta;
	let layout = &meta.layout;

	let (kind, min, max, log, current) = match &parameter.value {
		Value::Float {
			value,
			range,
			logarithmic,
		} => (
			Kind::Float,
			*range.start() as f64,
			*range.end() as f64,
			*logarithmic,
			*value as f64,
		),
		Value::Int {
			value,
			range,
			logarithmic,
		} => (
			Kind::Int,
			*range.start() as f64,
			*range.end() as f64,
			*logarithmic,
			*value as f64,
		),
		Value::Bool(value) => (Kind::Bool, 0.0, 1.0, false, if *value { 1.0 } else { 0.0 }),
		// Serialized state is not a control, and a placeholder is not one either.
		_ => return None,
	};

	let widget = match layout.widget {
		i_am_dsp::prelude::ParamWidget::Slider => "slider",
		i_am_dsp::prelude::ParamWidget::Toggle => "toggle",
		i_am_dsp::prelude::ParamWidget::Choice => "choice",
		i_am_dsp::prelude::ParamWidget::Knob => "knob",
		// Auto, and anything a newer build knows about.
		_ => match kind {
			Kind::Bool => "toggle",
			Kind::Int if meta.choices.is_some() => "choice",
			_ => "knob",
		},
	};

	let size = match layout.size {
		i_am_dsp::prelude::ParamSize::Small => "small",
		i_am_dsp::prelude::ParamSize::Large => "large",
		_ => "medium",
	};

	let label = meta
		.display_name
		.as_ref()
		.map(|name| name.to_string())
		.unwrap_or_else(|| parameter.identifier.clone());

	let default = meta.default.unwrap_or(current).clamp(min, max);

	Some(Control {
		id: parameter.identifier.clone(),
		label,
		kind,
		unit: unit_name(meta.unit),
		min,
		max,
		default,
		log,
		bipolar: layout.bipolar,
		step: layout.step,
		widget,
		size,
		order: layout.order,
		choices: meta
			.choices
			.map(|choices| choices.iter().map(|choice| (*choice).to_string()).collect()),
	})
}

/// The name the page knows a unit by.
fn unit_name(unit: Unit) -> &'static str {
	match unit {
		Unit::Milliseconds => "ms",
		Unit::Hertz => "hz",
		Unit::Beats => "beats",
		Unit::Seconds => "s",
		Unit::Samples => "samples",
		Unit::Decibels => "db",
		Unit::Percent => "percent",
		// None and anything a newer build adds.
		_ => "none",
	}
}

/// The whole faceplate as the page sees it.
pub fn face_json(face: &Faceplate, sections: &[Section]) -> String {
	face_json_with_rate(face, sections, None, false)
}

/// The faceplate, with the rate the plug-in is running at.
///
/// The page needs the rate to place a spectrum on the same frequency axis as the
/// response curve; a faceplate looked at outside a host has none, and the page
/// assumes 48 kHz for a preview.
pub fn face_json_with_rate(
	face: &Faceplate,
	sections: &[Section],
	rate: Option<usize>,
	hosted: bool,
) -> String {
	let mut out = String::new();

	out.push_str("{\"title\":");
	string(&mut out, face.title);
	out.push_str(",\"accent\":");
	string(&mut out, face.accent);
	let _ = write!(
		out,
		",\"size\":[{},{}]",
		number(face.size.0 as f64, Kind::Float),
		number(face.size.1 as f64, Kind::Float)
	);
	out.push_str(",\"mode\":");
	match face.mode {
		Some(mode) => string(&mut out, mode),
		None => out.push_str("null"),
	}
	out.push_str(",\"rate\":");
	match rate {
		Some(rate) => {
			let _ = write!(out, "{rate}");
		},
		None => out.push_str("null"),
	}
	let _ = write!(out, ",\"hosted\":{hosted}");
	out.push_str(",\"sections\":[");

	for (index, section) in sections.iter().enumerate() {
		if index > 0 {
			out.push(',');
		}

		out.push_str("{\"name\":");
		match &section.name {
			Some(name) => string(&mut out, name),
			None => out.push_str("null"),
		}

		out.push_str(",\"hero\":");
		match &section.hero {
			Some(hero) => string(&mut out, hero),
			None => out.push_str("null"),
		}

		out.push_str(",\"params\":[");
		for (position, control) in section.controls.iter().enumerate() {
			if position > 0 {
				out.push(',');
			}
			control_json(&mut out, control);
		}
		out.push_str("]}");
	}

	out.push_str("]}");
	out
}

/// One control as JSON.
fn control_json(out: &mut String, control: &Control) {
	out.push_str("{\"id\":");
	string(out, &control.id);
	out.push_str(",\"label\":");
	string(out, &control.label);
	out.push_str(",\"kind\":");
	string(
		out,
		match control.kind {
			Kind::Float => "float",
			Kind::Int => "int",
			Kind::Bool => "bool",
		},
	);
	out.push_str(",\"unit\":");
	string(out, control.unit);
	let _ = write!(
		out,
		",\"min\":{},\"max\":{},\"default\":{}",
		number(control.min, control.kind),
		number(control.max, control.kind),
		number(control.default, control.kind)
	);
	let _ = write!(
		out,
		",\"log\":{},\"bipolar\":{},\"step\":{}",
		control.log,
		control.bipolar,
		match control.step {
			Some(step) => number(step, Kind::Float),
			None => "null".to_string(),
		}
	);
	out.push_str(",\"widget\":");
	string(out, control.widget);
	out.push_str(",\"size\":");
	string(out, control.size);
	match control.order {
		Some(order) => {
			let _ = write!(out, ",\"order\":{order}");
		},
		None => out.push_str(",\"order\":null"),
	}

	out.push_str(",\"choices\":");
	match &control.choices {
		Some(choices) => {
			out.push('[');
			for (index, choice) in choices.iter().enumerate() {
				if index > 0 {
					out.push(',');
				}
				string(out, choice);
			}
			out.push(']');
		},
		None => out.push_str("null"),
	};
	out.push('}');
}

/// The current values, as the page's /state route.
pub fn state_json(map: &ParamMap, sections: &[Section]) -> String {
	let mut out = String::from("{");
	let mut first = true;

	for control in sections.iter().flat_map(|section| &section.controls) {
		let Some(atomic) = map.get(&control.id) else {
			continue;
		};

		let value = match atomic.load(Ordering::Relaxed) {
			SetValue::Float(value) => format!("{}", value),
			SetValue::Int(value) => format!("{value}"),
			SetValue::Bool(value) => format!("{value}"),
			// Nothing and serialized state have no number to show.
			_ => continue,
		};

		if !first {
			out.push(',');
		}
		first = false;

		string(&mut out, &control.id);
		out.push(':');
		out.push_str(&value);
	}

	out.push('}');
	out
}

/// The state, plus whether the plug-in is bypassed.
pub fn state_json_with_bypass(map: &ParamMap, sections: &[Section], bypass: bool) -> String {
	let mut out = state_json(map, sections);
	let value = if bypass { "true" } else { "false" };

	if out == "{}" {
		return format!("{{\"__bypass\":{value}}}");
	}

	out.truncate(out.len() - 1);
	out.push_str(",\"__bypass\":");
	out.push_str(value);
	out.push('}');
	out
}

/// Applies a bypass command from the page.
pub fn apply_bypass(bypass: &std::sync::atomic::AtomicBool, body: &str) -> bool {
	let Some((id, raw)) = body.split_once('=') else {
		return false;
	};

	if id != "__bypass" {
		return false;
	}

	match raw.trim() {
		"1" | "true" | "on" => {
			bypass.store(true, Ordering::Relaxed);
			true
		},
		"0" | "false" | "off" => {
			bypass.store(false, Ordering::Relaxed);
			true
		},
		_ => false,
	}
}

/// Writes one id=value command from the page into the shared parameter map.
///
/// The page is the only writer, so a malformed string is ignored rather than
/// allowed to take the audio thread with it. The type comes from the metadata, not
/// from the string: 1 is a bool for a switch and a float for a cutoff.
pub fn apply_command(map: &ParamMap, sections: &[Section], body: &str) -> bool {
	let Some((id, raw)) = body.split_once('=') else {
		return false;
	};

	let Some(control) = sections
		.iter()
		.flat_map(|section| &section.controls)
		.find(|control| control.id == id)
	else {
		return false;
	};

	let Some(atomic) = map.get(id) else {
		return false;
	};

	match control.kind {
		Kind::Float => raw
			.trim()
			.parse::<f32>()
			.map(|value| atomic.store(value, Ordering::Relaxed))
			.unwrap_or(false),
		Kind::Int => raw
			.trim()
			.parse::<i32>()
			.map(|value| atomic.store(value, Ordering::Relaxed))
			.unwrap_or(false),
		Kind::Bool => match raw.trim() {
			"1" | "true" | "on" => atomic.store(true, Ordering::Relaxed),
			"0" | "false" | "off" => atomic.store(false, Ordering::Relaxed),
			_ => false,
		},
	}
}

/// A number the page can read, without the binary noise of a widened f32.
fn number(value: f64, kind: Kind) -> String {
	if !value.is_finite() {
		return "0".to_string();
	}

	match kind {
		Kind::Float => format!("{}", value as f32),
		Kind::Int => format!("{}", value.round() as i64),
		Kind::Bool => format!("{}", if value != 0.0 { 1 } else { 0 }),
	}
}

/// A JSON string, escaped.
fn string(out: &mut String, value: &str) {
	out.push('"');
	for character in value.chars() {
		match character {
			'"' => out.push_str("\\\""),
			'\\' => out.push_str("\\\\"),
			'\n' => out.push_str("\\n"),
			'\r' => out.push_str("\\r"),
			'\t' => out.push_str("\\t"),
			character if (character as u32) < 0x20 => {
				let _ = write!(out, "\\u{:04x}", character as u32);
			},
			character => out.push(character),
		}
	}
	out.push('"');
}

#[cfg(test)]
mod tests {
	use super::*;
	use i_am_dsp::prelude::{ParamLayout, ParameterMeta};

	/// A float parameter with the metadata a faceplate reads.
	fn parameter(
		id: &str,
		value: f32,
		range: std::ops::RangeInclusive<f32>,
		layout: ParamLayout,
	) -> Parameter {
		let start = *range.start() as f64;

		Parameter {
			identifier: id.to_string(),
			value: Value::Float {
				value,
				range,
				logarithmic: true,
			},
			meta: ParameterMeta {
				default: Some(start),
				layout,
				..Default::default()
			},
		}
	}

	/// The section keeps declaration order, and members that declared one sort by it.
	#[test]
	fn sections_follow_the_metadata() {
		let parameters = vec![
			parameter("b", 0.5, 0.0..=1.0, ParamLayout::new().group("Second").order(1)),
			parameter("a", 0.5, 0.0..=1.0, ParamLayout::new().group("First").order(0)),
			parameter("c", 0.5, 0.0..=1.0, ParamLayout::new().group("First").order(1)),
		];

		let sections = sections(&parameters);

		assert_eq!(sections.len(), 2);
		assert_eq!(sections[0].name.as_deref(), Some("Second"));
		assert_eq!(sections[1].name.as_deref(), Some("First"));
		assert_eq!(
			sections[1]
				.controls
				.iter()
				.map(|control| control.id.as_str())
				.collect::<Vec<_>>(),
			["a", "c"]
		);
	}

	/// A parameter that cannot be moved is not drawn.
	#[test]
	fn serialized_state_is_not_a_control() {
		let parameter = Parameter {
			identifier: "coefficients".to_string(),
			value: Value::Serialized(vec![1, 2, 3]),
			meta: ParameterMeta::default(),
		};

		assert!(sections(&[parameter]).is_empty());
	}

	/// Labels are the only strings a page gets, so they have to survive a quote.
	#[test]
	fn a_label_is_escaped() {
		let mut out = String::new();
		string(&mut out, "a \"quoted\" name\n");

		assert_eq!(out, "\"a \\\"quoted\\\" name\\n\"");
	}

	/// A f32 widened to f64 formats as binary noise, which a knob would show.
	#[test]
	fn numbers_are_written_as_the_parameter_holds_them() {
		assert_eq!(number(0.01_f32 as f64, Kind::Float), "0.01");
		assert_eq!(number(4.0_f32 as f64, Kind::Int), "4");
	}

	/// The page puts this control in the header rather than in the body.
	#[test]
	fn the_mode_switch_is_named_in_the_faceplate() {
		let face = Faceplate::new("I Am Filter", "#4c8dff", (520.0, 330.0)).with_mode("mode");
		let json = face_json(&face, &[]);

		assert!(json.contains("\"mode\":\"mode\""), "{json}");
	}

	/// The switch is not a parameter, so it has its own command and its own field.
	#[test]
	fn a_bypass_command_sets_the_flag() {
		let bypass = std::sync::atomic::AtomicBool::new(false);

		assert!(apply_bypass(&bypass, "__bypass=1"));
		assert!(bypass.load(Ordering::Relaxed));
		assert!(apply_bypass(&bypass, "__bypass=0"));
		assert!(!bypass.load(Ordering::Relaxed));

		assert!(!apply_bypass(&bypass, "gain=1"));
		assert!(!apply_bypass(&bypass, "__bypass=maybe"));
	}

	/// Even with no controls, the page still has a bypass switch to read.
	#[test]
	fn the_state_carries_the_bypass_flag() {
		let map = ParamMap::new();

		assert_eq!(
			state_json_with_bypass(&map, &[], true),
			"{\"__bypass\":true}"
		);
		assert_eq!(
			state_json_with_bypass(&map, &[], false),
			"{\"__bypass\":false}"
		);
	}

	/// The page needs the rate to place a spectrum on the response's axis.
	#[test]
	fn the_rate_is_written_when_the_host_has_one() {
		let face = Faceplate::new("I Am Filter", "#4c8dff", (520.0, 330.0));
		let hosted = face_json_with_rate(&face, &[], Some(48_000), true);
		let preview = face_json(&face, &[]);

		assert!(hosted.contains("\"rate\":48000"), "{hosted}");
		assert!(hosted.contains("\"hosted\":true"), "{hosted}");
		assert!(preview.contains("\"rate\":null"), "{preview}");
		assert!(preview.contains("\"hosted\":false"), "{preview}");
	}
}
