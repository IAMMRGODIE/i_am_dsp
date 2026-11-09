//! The first drum voice: a kick, on note 36.
//!
//! The DSP lives in `i_am_dsp::prelude::DrumInstrument`; this module is the
//! window, the note port, the shared instrument body, and the curve editor's
//! route.

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use i_am_dsp::prelude::{
	ClickSlot, Curve, DrumInstrument, DrumPreset, FitSlot, GeneratorEffect, Parameter, Parameters,
	Samples, SetValue, SnareCurves, decode_click, default_amp_templates, from_binary, slot_drum,
	to_binary,
};
use i_am_plugin::processor::EffectProcessor;
use i_am_plugin::{
	AspectRatioStrategy, Descriptor, GuiResizeHints, MidiPort, Plugin, Tag, WindowOptions,
};
use i_am_web_gui::{Faceplate, Page, ProtocolRequest, ProtocolResponse, Scope, editor_with};

const SIZE: (f32, f32) = (620.0, 320.0);
const TITLE: &str = "I Am Drummer";
const ACCENT: &str = "#c8793d";

const PAGES: &[Page] = &[
	Page::new("Kit", &["Kit"]),
	Page::new("Samples", &["Samples"]),
	Page::new("Kick", &["Kick"]),
	Page::new("Snare", &["Snare"]),
	Page::new("Hat", &["Hat"]),
	Page::new("Clap", &["Clap"]),
	Page::new("Tom", &["Tom"]),
	Page::new("Cymbal", &["Cymbal"]),
];

/// The faceplate of the first drum.
pub fn faceplate() -> Faceplate {
	Faceplate::new(TITLE, ACCENT, SIZE)
		.with_pages(PAGES)
		.with_hidden(&["click_sample"])
}

/// A kick drum, with a web faceplate and an editable pitch fall.
pub struct DrummerWeb {
	processor: EffectProcessor<GeneratorEffect<DrumInstrument>>,
	curve: Arc<Mutex<Curve>>,
	click: Arc<ClickSlot>,
	samples: Arc<Samples>,
	fit: Arc<FitSlot>,
	amps: Arc<Mutex<[Curve; 6]>>,
	snare: Arc<Mutex<SnareCurves>>,
	bypass: Arc<AtomicBool>,
	scope: Arc<Scope>,
}

impl Default for DrummerWeb {
	fn default() -> Self {
		let curve = Arc::new(Mutex::new(DrumInstrument::default().pitch_curve.clone()));
		let factory_curve = curve.clone();
		let click = ClickSlot::new();
		let factory_click = click.clone();
		let samples = Samples::new();
		let factory_samples = samples.clone();
		let fit = FitSlot::new();
		let factory_fit = fit.clone();
		let amps = Arc::new(Mutex::new(default_amp_templates()));
		let factory_amps = amps.clone();
		let snare = Arc::new(Mutex::new(SnareCurves::default()));
		let factory_snare = snare.clone();

		Self {
			processor: EffectProcessor::with_build(move |rate: usize| {
				GeneratorEffect::new(
					DrumInstrument::new(rate)
						.with_shared(factory_curve.clone())
						.with_click_slot(factory_click.clone())
						.with_samples(factory_samples.clone())
						.with_fit_slot(factory_fit.clone())
						.with_amps(factory_amps.clone())
						.with_snare(factory_snare.clone()),
				)
			}),
			curve,
			click,
			samples,
			fit,
			amps,
			snare,
			bypass: Arc::new(AtomicBool::new(false)),
			scope: Arc::new(Scope::new()),
		}
	}
}

impl Parameters for DrummerWeb {
	fn get_parameters(&self) -> Vec<Parameter> {
		self.processor.parameters()
	}

	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		// A host restoring the state writes the saved curve; the voice has to
		// hear it too, not only the parameter the next save will read.
		if identifier == "pitch_curve"
			&& let SetValue::Serialized(bytes) = &value
			&& let Ok(curve) = from_binary::<Curve>(bytes.clone())
			&& let Ok(mut shared) = self.curve.lock()
		{
			*shared = curve;
		}

		if identifier == "snare_curves"
			&& let SetValue::Serialized(bytes) = &value
			&& let Ok(curves) = from_binary::<SnareCurves>(bytes.clone())
			&& let Ok(mut shared) = self.snare.lock()
		{
			*shared = curves;
		}

		if identifier == "amp_curves"
			&& let SetValue::Serialized(bytes) = &value
			&& let Ok(curves) = from_binary::<Vec<Curve>>(bytes.clone())
			&& curves.len() == 6
			&& let Ok(mut shared) = self.amps.lock()
		{
			for (index, curve) in curves.into_iter().enumerate() {
				shared[index] = curve;
			}
		}

		// Choosing a kit writes the knobs it describes, so the page and the host
		// see the preset rather than only the choice.
		if identifier == "kit"
			&& let SetValue::Int(index) = value
		{
			let applied = self.processor.set_parameter("kit", SetValue::Int(index));

			for (identifier, value) in DrumInstrument::kit_preset(index) {
				self.processor
					.set_parameter(identifier, SetValue::Float(*value));
			}

			return applied;
		}

		self.processor.set_parameter(identifier, value)
	}
}

synth_processor!(DrummerWeb);

impl Plugin for DrummerWeb {
	const DESCRIPTOR: Descriptor = Descriptor::new("iamdsp.synths.drummer.web", TITLE)
		.with_vendor("iamplugins")
		.with_version(env!("CARGO_PKG_VERSION"))
		.with_description("A drum kit synthesised from partials, clicks and noise. M1: the kick.")
		.with_tags(&[
			Tag::Instrument,
			Tag::Synthesizer,
			Tag::Drum,
			Tag::DrumMachine,
		]);

	const INPUT_PORTS: &'static [i_am_plugin::AudioPort] = &[];
	const INPUT_MIDI_PORTS: &'static [MidiPort] = &[MidiPort { name: "Notes" }];
	const HAS_GUI: bool = true;

	fn new() -> Self {
		Self::default()
	}

	fn window_options() -> WindowOptions {
		const FACE_FRACTION: f64 = 0.85;

		WindowOptions::new()
			.with_size((SIZE.0 as f64 / FACE_FRACTION, SIZE.1 as f64 / FACE_FRACTION))
			.with_title(TITLE)
			.with_resize_hints(GuiResizeHints {
				can_resize_horizontally: true,
				can_resize_vertically: true,
				strategy: AspectRatioStrategy::Disregard,
			})
	}

	fn param_map(&self) -> i_am_dsp::prelude::ParamMap {
		self.processor.param_map()
	}

	fn sync_params(&mut self) {
		// The host asks for the parameters before it saves them; this is where
		// the curve the editor changed becomes the one the state carries.
		if let Ok(curve) = self.curve.lock()
			&& let Ok(bytes) = to_binary(&*curve)
		{
			self.processor
				.set_parameter("pitch_curve", SetValue::Serialized(bytes));
		}

		if let Ok(amps) = self.amps.lock()
			&& let Ok(bytes) = to_binary(&amps.to_vec())
		{
			self.processor
				.set_parameter("amp_curves", SetValue::Serialized(bytes));
		}

		if let Ok(snare) = self.snare.lock()
			&& let Ok(bytes) = to_binary(&*snare)
		{
			self.processor
				.set_parameter("snare_curves", SetValue::Serialized(bytes));
		}

		self.processor.sync_params();
	}

	fn on_activate(&mut self, sample_rate: usize) {
		self.processor.activate(sample_rate);
	}

	fn editor(&self) -> Option<Box<dyn i_am_plugin::Editor<Self>>> {
		let rate = self.processor.sample_rate();
		let curve = self.curve.clone();
		let click = self.click.clone();
		let samples = self.samples.clone();
		let fit = self.fit.clone();
		let amps = self.amps.clone();
		let snare = self.snare.clone();

		editor_with(
			self,
			faceplate(),
			self.bypass.clone(),
			self.scope.clone(),
			rate,
			move |request: &ProtocolRequest<'_>| {
				if request.path == "/click" {
					return Some(match request.method {
						"GET" => ProtocolResponse::json(click.json()),
						"POST" => match decode_click(request.body) {
							Ok(data) => {
								click.set(data);
								ProtocolResponse::json("{\"ok\":true}")
							}
							Err(error) => ProtocolResponse::json(format!(
								"{{\"error\":\"{}\"}}",
								error.replace('"', "")
							)),
						},
						_ => ProtocolResponse::json("{\"error\":\"method\"}"),
					});
				}

				if request.path == "/amp" {
					return Some(match request.method {
						"GET" => ProtocolResponse::json(amp_json(&amps, &snare)),
						"POST" => match std::str::from_utf8(request.body) {
							Ok(text) => {
								let drum = query_drum(request.query);

								match parse_amp(text) {
									Some(curve) => {
										if drum < 6 {
											if let Ok(mut shared) = amps.lock() {
												shared[drum] = curve;
											}
										} else if let Ok(mut shared) = snare.lock() {
											match drum {
												6 => shared.base = curve,
												7 => shared.noise = curve,
												8 => shared.pitch = curve,
												_ => {}
											}
										}
										ProtocolResponse::json("{\"ok\":true}")
									}
									None => ProtocolResponse::json("{\"error\":\"points\"}"),
								}
							}
							Err(_) => ProtocolResponse::json("{\"error\":\"utf8\"}"),
						},
						_ => ProtocolResponse::json("{\"error\":\"method\"}"),
					});
				}

				if request.path == "/preset" {
					return Some(match request.method {
						"GET" => ProtocolResponse::json(fit.json()),
						"POST" => match serde_json::from_slice::<DrumPreset>(request.body) {
							Ok(preset) => {
								fit.set(preset);
								ProtocolResponse::json("{\"ok\":true}")
							}
							Err(error) => ProtocolResponse::json(format!(
								"{{\"error\":\"{}\"}}",
								error.to_string().replace('"', "")
							)),
						},
						_ => ProtocolResponse::json("{\"error\":\"method\"}"),
					});
				}

				if request.path == "/samples" {
					return Some(match request.method {
						"GET" => ProtocolResponse::json(samples.json()),
						"POST" => {
							let query = request.query.unwrap_or("");
							let slot = query
								.split('&')
								.find_map(|part| part.strip_prefix("slot="))
								.unwrap_or("");
							let clear = query.split('&').any(|part| part == "clear=1");
							let Some(drum) = slot_drum(slot) else {
								return Some(ProtocolResponse::json("{\"error\":\"slot\"}"));
							};

							if clear {
								samples.set(drum, None);
								return Some(ProtocolResponse::json("{\"ok\":true}"));
							}

							match decode_click(request.body) {
								Ok(data) => {
									samples.set(drum, Some(data));
									ProtocolResponse::json("{\"ok\":true}")
								}
								Err(error) => ProtocolResponse::json(format!(
									"{{\"error\":\"{}\"}}",
									error.replace('"', "")
								)),
							}
						}
						_ => ProtocolResponse::json("{\"error\":\"method\"}"),
					});
				}

				if request.path != "/pitch" {
					return None;
				}

				match request.method {
					"GET" => Some(ProtocolResponse::json(pitch_json(&curve))),
					"POST" => {
						let Some(text) = std::str::from_utf8(request.body).ok() else {
							return Some(ProtocolResponse::json("{\"error\":\"utf8\"}"));
						};
						let Some(points) = parse_pitch(text) else {
							return Some(ProtocolResponse::json("{\"error\":\"points\"}"));
						};
						if let Ok(mut shared) = curve.lock() {
							*shared = points;
						}
						Some(ProtocolResponse::json("{\"ok\":true}"))
					}
					_ => Some(ProtocolResponse::json("{\"ok\":false}")),
				}
			},
			|_: &str| false,
		)
	}
}

/// The curve as the page draws it: `{"points":[[time, ratio, bend], ...]}`.
fn query_drum(query: Option<&str>) -> usize {
	query
		.unwrap_or("")
		.split('&')
		.find_map(|part| part.strip_prefix("drum="))
		.and_then(|value| value.parse().ok())
		.unwrap_or(0)
}

fn curve_json(curve: &Curve, out: &mut String) {
	out.push('[');

	for (index, point) in curve.points().iter().enumerate() {
		if index > 0 {
			out.push(',');
		}
		out.push_str(&format!("[{},{},{}]", point.time, point.value, point.bend));
	}

	out.push(']');
}

fn amp_json(amps: &Arc<Mutex<[Curve; 6]>>, snare: &Arc<Mutex<SnareCurves>>) -> String {
	let Ok(guard) = amps.lock() else {
		return "{\"curves\":[]}".to_string();
	};
	let Ok(snare) = snare.lock() else {
		return "{\"curves\":[]}".to_string();
	};
	let mut out = String::from("{\"curves\":[");

	for (drum, curve) in guard.iter().enumerate() {
		if drum > 0 {
			out.push(',');
		}
		curve_json(curve, &mut out);
	}

	// The snare's shell, wires and pitch follow the six drums.
	for curve in [&snare.base, &snare.noise, &snare.pitch] {
		out.push(',');
		curve_json(curve, &mut out);
	}

	out.push_str("]}");
	out
}

fn parse_amp(text: &str) -> Option<Curve> {
	let mut triples = Vec::new();

	for pair in text.trim().split(';') {
		if pair.trim().is_empty() {
			continue;
		}

		let mut parts = pair.split(':');
		let time: f32 = parts.next()?.trim().parse().ok()?;
		let value: f32 = parts.next()?.trim().parse().ok()?;
		let bend: f32 = parts.next().unwrap_or("0").trim().parse().unwrap_or(0.0);

		if !time.is_finite() || !value.is_finite() {
			return None;
		}

		triples.push((
			time.clamp(0.0, 10.0),
			value.clamp(0.0, 10.0),
			bend.clamp(-10.0, 10.0),
		));
	}

	if triples.len() < 2 {
		return None;
	}

	Some(Curve::from_triples(triples))
}

fn pitch_json(curve: &Arc<Mutex<Curve>>) -> String {
	let Ok(guard) = curve.lock() else {
		return "{\"points\":[]}".to_string();
	};
	let mut out = String::from("{\"points\":[");

	for (index, point) in guard.points().iter().enumerate() {
		if index > 0 {
			out.push(',');
		}
		out.push_str(&format!("[{},{},{}]", point.time, point.value, point.bend));
	}

	out.push_str("]}");
	out
}

/// `time:ratio:bend;...`, the same wire format the rack's curve editor uses.
fn parse_pitch(text: &str) -> Option<Curve> {
	let mut triples = Vec::new();

	for pair in text.trim().split(';') {
		if pair.trim().is_empty() {
			continue;
		}

		let mut parts = pair.split(':');
		let time: f32 = parts.next()?.trim().parse().ok()?;
		let value: f32 = parts.next()?.trim().parse().ok()?;
		let bend: f32 = parts.next().unwrap_or("0").trim().parse().unwrap_or(0.0);

		if !time.is_finite() || !value.is_finite() {
			return None;
		}

		// The editor goes to ten octaves; the wire carries the same range.
		triples.push((
			time.clamp(0.0, 1.0),
			value.clamp(0.05, 1_024.0),
			bend.clamp(-10.0, 10.0),
		));
	}

	if triples.len() < 2 {
		return None;
	}

	Some(Curve::from_triples(triples))
}

#[cfg(test)]
mod route_tests {
	use super::parse_pitch;

	#[test]
	fn a_ten_octave_point_survives_the_wire() {
		let curve = parse_pitch("0:1024:-2.5;0.03:1:0").expect("the wire parses");

		assert!(
			(curve.points()[0].value - 1_024.0).abs() < 1e-3,
			"the wire clamped the top of the curve: {}",
			curve.points()[0].value
		);
	}
}
