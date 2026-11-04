//! The metadata a parameter carries: its unit, its layout and its name.

use super::*;

/// How a parameter wants its value shown to a person.
///
/// A host that only knows a number can do little better than printing it, which is how a
/// synced time ends up reading `0.00` where a musician expects `1/8`. A parameter declares its
/// unit and the host can ask for text; see [`ParameterMeta::default`] for the other half, the
/// value a parameter starts at.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum Unit {
	/// A plain number.
	#[default]
	None,
	/// A length in milliseconds.
	Milliseconds,
	/// A frequency in hertz.
	Hertz,
	/// A length in quarter-note beats: `0.5` is an eighth note, `0.75` a dotted eighth.
	Beats,
	/// A length in seconds.
	Seconds,
	/// A count of audio samples.
	Samples,
	/// A level in decibels.
	Decibels,
	/// A fraction of one, shown as a percentage: `0.5` reads as `50.0 %`.
	Percent,
}

impl Unit {
	/// Renders a value the way this unit wants to be read.
	pub fn format(self, value: f64) -> String {
		match self {
			Unit::None => format!("{value:.2}"),
			Unit::Milliseconds => format!("{value:.2} ms"),
			Unit::Hertz => format!("{value:.2} Hz"),
			Unit::Beats => format_beats(value),
			Unit::Seconds => format!("{value:.3} s"),
			Unit::Samples => format!("{value:.0} samples"),
			Unit::Decibels => format!("{value:.2} dB"),
			// The value is a fraction, so half of it reads as fifty percent.
			Unit::Percent => format!("{:.1} %", value * 100.0),
		}
	}

	/// Reads back what [`Self::format`] produced, for a host that lets the text be edited.
	///
	/// Returns [`None`] when the text is not this unit's business, so a caller can fall back
	/// to reading a plain number.
	pub fn parse(self, text: &str) -> Option<f64> {
		let text = text.trim();

		match self {
			Unit::None => text.parse().ok(),
			Unit::Milliseconds => Some(text.trim_end_matches("ms").trim().parse().ok()?),
			Unit::Hertz => Some(text.trim_end_matches("Hz").trim().parse().ok()?),
			Unit::Beats => parse_beats(text),
			Unit::Seconds => Some(text.trim_end_matches('s').trim().parse().ok()?),
			Unit::Samples => Some(text.trim_end_matches("samples").trim().parse().ok()?),
			Unit::Decibels => Some(text.trim_end_matches("dB").trim().parse().ok()?),
			Unit::Percent => Some(text.trim_end_matches('%').trim().parse::<f64>().ok()? / 100.0),
		}
	}
}

/// Renders a length in quarter-note beats as note values, which is how a musician reads one.
///
/// Anything that is not a note value (or a dotted or triplet one) falls back to a beat count,
/// because `3.7 beats` is more use than a wrong fraction.
fn format_beats(beats: f64) -> String {
	// A whole note is four quarter-note beats, so a note value is `4 / beats`.
	for (multiplier, suffix) in [(1.0, ""), (1.5, " dotted"), (2.0 / 3.0, " triplet")] {
		let note = beats / multiplier;
		if note > 0.0 {
			let denominator = (4.0 / note).round();
			// A power-of-two denominator is a note value; 3, 5, 7 and the rest are not.
			if (1.0..=128.0).contains(&denominator) {
				let denominator = denominator as u32;
				if denominator.is_power_of_two() && (4.0 / denominator as f64 - note).abs() < 1e-4 {
					return format!("1/{}{}", denominator, suffix);
				}
			}
		}
	}

	format!("{beats:.2} beats")
}

/// Reads the note values [`format_beats`] writes, or a plain beat count.
fn parse_beats(text: &str) -> Option<f64> {
	let (text, multiplier) = if let Some(text) = text.strip_suffix("dotted") {
		(text, 1.5)
	} else if let Some(text) = text.strip_suffix("triplet") {
		(text, 2.0 / 3.0)
	} else {
		(text, 1.0)
	};
	let text = text.trim().trim_end_matches("beats").trim();

	if let Some((numerator, denominator)) = text.split_once('/') {
		let numerator: f64 = numerator.trim().parse().ok()?;
		let denominator: f64 = denominator.trim().parse().ok()?;
		if denominator == 0.0 {
			return None;
		}
		// `1/8` is an eighth note, which is half a quarter-note beat.
		return Some(4.0 * numerator / denominator * multiplier);
	}

	text.parse::<f64>().ok().map(|beats| beats * multiplier)
}

#[cfg(feature = "layout")]
/// How a parameter wants to be drawn.
///
/// A hint, not a rule: a faceplate with its own idea of a parameter is free to
/// ignore it, and one with no idea at all can lay a whole effect out from these.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum ParamWidget {
	/// Whatever a faceplate draws a plain number with.
	#[default]
	Auto,
	/// A knob.
	Knob,
	/// A horizontal slider.
	Slider,
	/// An on/off switch.
	Toggle,
	/// One of a few named choices.
	Choice,
	/// A multi-select note mask, one bit per pitch class.
	Notes,
}

#[cfg(feature = "layout")]
/// How much room a parameter wants in a faceplate.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum ParamSize {
	/// A control for a parameter that is set once and left alone.
	Small,
	/// The default.
	#[default]
	Medium,
	/// The control the window is about.
	Large,
}

#[cfg(feature = "layout")]
/// Where a parameter wants to sit in a faceplate, and how it wants to be drawn.
///
/// A parameter already carries its range, its unit and the value it starts at.
/// What it does not carry is how a person is meant to meet it: which of the
/// other controls it belongs beside, whether it wants a knob or a slider,
/// whether it is a trim that deserves a small control or the one control the
/// whole window is about.
///
/// Every field here is a hint. A faceplate that reads them can lay an effect out
/// without being told anything else; one that ignores them still works, and a
/// parameter that declares none of them is laid out by the order it was
/// declared in.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ParamLayout {
	/// The section of a faceplate this parameter belongs to.
	pub group: Option<std::borrow::Cow<'static, str>>,
	/// Where it sits inside that group. Ties keep declaration order.
	pub order: Option<u16>,
	/// How it wants to be drawn.
	pub widget: ParamWidget,
	/// How much room it wants.
	pub size: ParamSize,
	/// Whether the middle of its range is its neutral point, so a control fills
	/// outward from the centre: a pan, a gain in decibels, a pitch offset.
	pub bipolar: bool,
	/// The interval its values snap to, when it has one: a whole voice, a
	/// semitone, an eighth note. A control that knows this can show its ticks.
	pub step: Option<f64>,
	/// The picture this parameter's section shows, if the host can draw it.
	///
	/// A name rather than a widget: the metadata says what a section is *about*
	/// ("transfer", "spectrum", "eq") and the host decides what that looks like,
	/// because the host is the one that knows what it can draw. A section whose
	/// parameters name none is drawn as controls and nothing else, and one that
	/// names something the host cannot draw is too.
	pub hero: Option<std::borrow::Cow<'static, str>>,
}

#[cfg(feature = "layout")]
impl ParamLayout {
	/// A layout with nothing declared, which is what a parameter has by default.
	pub fn new() -> Self {
		Self::default()
	}

	/// The section of a faceplate this parameter belongs to.
	pub fn group(mut self, group: impl Into<std::borrow::Cow<'static, str>>) -> Self {
		self.group = Some(group.into());
		self
	}

	/// Where it sits inside that group.
	pub fn order(mut self, order: u16) -> Self {
		self.order = Some(order);
		self
	}

	/// How it wants to be drawn.
	pub fn widget(mut self, widget: ParamWidget) -> Self {
		self.widget = widget;
		self
	}

	/// How much room it wants.
	pub fn size(mut self, size: ParamSize) -> Self {
		self.size = size;
		self
	}

	/// Whether the middle of its range is its neutral point.
	pub fn bipolar(mut self) -> Self {
		self.bipolar = true;
		self
	}

	/// The interval its values snap to.
	pub fn step(mut self, step: f64) -> Self {
		self.step = Some(step);
		self
	}

	/// Asks for a picture at the top of this parameter's section.
	///
	/// The name is the host's to interpret: an effect says its drive section is
	/// about a "transfer" curve, and whichever host draws it decides how.
	pub fn hero(mut self, hero: &'static str) -> Self {
		self.hero = Some(std::borrow::Cow::Borrowed(hero));
		self
	}

	/// Whether this says anything at all.
	pub fn is_empty(&self) -> bool {
		self == &Self::default()
	}
}

/// What a host needs to present a parameter beyond its value and its range.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ParameterMeta {
	/// The value the parameter starts at, for a host that resets parameters to a default.
	///
	/// **Declare one wherever there is an answer to give.** A host that is not told falls
	/// back to the bottom of the range, which is rarely where an effect starts: a synced time
	/// would come back as a thirty-second note rather than a beat, and a gain at its quietest
	/// rather than at unity. It is only right to leave it as [`None`] when there is no single
	/// answer — a parameter whose starting value is an argument to the constructor, like a
	/// delay time or a sample rate. See the `#[default(..)]` attribute for declaring one.
	pub default: Option<f64>,
	/// How to render the value as text.
	pub unit: Unit,
	/// What a host should call the parameter, when the identifier is not what a person should
	/// read: `#[id(display = "BPM Sync")]`.
	///
	/// A nested parameter is named by joining the names of the levels that contain it with a
	/// space, so a `Beats` field under a `Tempo` sub-parameter shows to a host as
	/// `Tempo Beats`. A level that declares nothing contributes its field name.
	pub display_name: Option<std::borrow::Cow<'static, str>>,
	/// What the numbers of an integer parameter mean, when it is a choice between
	/// named modes: `#[choices("Low", "Band", "High")]`.
	///
	/// A range says a parameter goes from nothing to two; this says what those are.
	/// A host that only shows the number can ignore it, and one that shows a list of
	/// names has nowhere else to get them from.
	pub choices: Option<&'static [&'static str]>,
	/// Where it wants to sit in a faceplate, and how it wants to be drawn.
	///
	/// Nothing reads this yet except a faceplate that chooses to; a host that only
	/// wants the value, the range and the unit can ignore all of it.
	#[cfg(feature = "layout")]
	pub layout: ParamLayout,
}

/// A helper struct for report parameters.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Parameter {
	/// An unique identifier for the parameter.
	pub identifier: String,
	/// The parameter value.
	pub value: Value,
	/// How to present it. Skipped when saving state: it comes from the code, not from the
	/// project, so a project saved by an older build simply takes the current one.
	#[serde(skip)]
	pub meta: ParameterMeta,
}

impl Parameter {
	/// A parameter with no metadata, for a hand-written [`Parameters`] implementation.
	pub fn new(identifier: impl Into<String>, value: Value) -> Self {
		Self {
			identifier: identifier.into(),
			value,
			meta: ParameterMeta::default(),
		}
	}

	/// The same parameter, with the metadata a host presents it by.
	pub fn with_meta(mut self, meta: ParameterMeta) -> Self {
		self.meta = meta;
		self
	}
}
