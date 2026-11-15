//! The Spectral Dynamics product: a compressor and a downward expander on
//! every FFT bin, with the threshold and ratio curves the page edits.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use i_am_dsp::prelude::{
	CurvePoint, DynamicsCurves, Effect, ParamMap, Parameter, Parameters, PhaseVocoder, SetValue,
	SpectralDynamics, Value, from_binary, to_binary,
};
use i_am_dsp::{ProcessContext, Processor};
use i_am_plugin::processor::{EffectMessage, EffectProcessor};
use i_am_plugin::{Descriptor, Plugin, WindowOptions};
use i_am_web_gui::{Faceplate, Page, ProtocolRequest, ProtocolResponse, Scope};

const SIZE: (f32, f32) = (760.0, 430.0);
const TITLE: &str = "I Am Spectral Dynamics";
const ACCENT: &str = "#8fd44a";

/// The analysis window the product runs.
const WINDOW: usize = 2_048;

/// The pages the page and the effect share.
#[derive(Clone)]
pub struct SpectralDynamicsHandle(Arc<DynamicsCurves>);

impl SpectralDynamicsHandle {
	/// A handle whose curves are flat: no tilt and the full master ratio.
	pub fn new() -> Self {
		Self(DynamicsCurves::new())
	}

	/// The curves this handle owns.
	pub fn curves(&self) -> Arc<DynamicsCurves> {
		self.0.clone()
	}

	/// Both page lists as JSON arrays of points.
	pub fn state_json(&self) -> String {
		let (threshold, ratio) = self.0.pages();
		let body = serde_json::json!({
			"threshold": pages_to_json(&threshold),
			"ratio": pages_to_json(&ratio),
		});

		body.to_string()
	}

	/// Replaces both page lists from the JSON the page posts.
	pub fn set_json(&self, body: &str) -> String {
		let Ok(body) = serde_json::from_str::<CurvesBody>(body) else {
			return "{\"error\":\"body\"}".to_string();
		};

		self.0.set_threshold(pages_from_json(body.threshold));
		self.0.set_ratio(pages_from_json(body.ratio));

		self.state_json()
	}
}

impl Default for SpectralDynamicsHandle {
	fn default() -> Self {
		Self::new()
	}
}

#[derive(serde::Deserialize)]
struct CurvesBody {
	threshold: Vec<Vec<Vec<f32>>>,
	ratio: Vec<Vec<Vec<f32>>>,
}

fn pages_to_json(pages: &[Vec<CurvePoint>]) -> Vec<Vec<[f32; 3]>> {
	pages
		.iter()
		.map(|page| {
			page.iter()
				.map(|point| [point.time, point.value, point.bend])
				.collect::<Vec<_>>()
		})
		.collect()
}

fn pages_from_json(pages: Vec<Vec<Vec<f32>>>) -> Vec<Vec<CurvePoint>> {
	pages
		.into_iter()
		.map(|page| {
			page.into_iter()
				.map(|point| {
					// A row is time, value and bend; a shorter row is a point on
					// the straight line, which is what a page that has not been
					// rebuilt yet still sends.
					CurvePoint {
						time: point.first().copied().unwrap_or(0.0),
						value: point.get(1).copied().unwrap_or(0.5),
						bend: point.get(2).copied().unwrap_or(0.0),
					}
				})
				.collect::<Vec<_>>()
		})
		.collect()
}

fn curve_route(
	request: &ProtocolRequest<'_>,
	handle: &SpectralDynamicsHandle,
) -> Option<ProtocolResponse> {
	if request.path != "/spectral-dynamics" {
		return None;
	}

	match request.method {
		"GET" => Some(ProtocolResponse::json(handle.state_json())),
		"POST" => Some(ProtocolResponse::json(
			handle.set_json(&String::from_utf8_lossy(request.body)),
		)),
		_ => Some(ProtocolResponse::json("{\"error\":\"method\"}")),
	}
}

/// The effect a rack module and a test build without a page use.
pub fn spectral_dynamics(sample_rate: usize) -> Box<dyn Effect<2>> {
	Box::new(PhaseVocoder::new(
		SpectralDynamics::new(DynamicsCurves::new(), sample_rate),
		WINDOW,
		sample_rate,
	))
}

/// The product.
pub struct SpectralDynamicsWeb {
	processor: EffectProcessor<PhaseVocoder<SpectralDynamics, 2>>,
	bypass: Arc<AtomicBool>,
	scope: Arc<Scope>,
	handle: SpectralDynamicsHandle,
}

impl Default for SpectralDynamicsWeb {
	fn default() -> Self {
		Self::with_handle(SpectralDynamicsHandle::new())
	}
}

impl SpectralDynamicsWeb {
	/// A product over the curves `handle` owns.
	pub fn with_handle(handle: SpectralDynamicsHandle) -> Self {
		let curves = handle.curves();

		Self {
			processor: EffectProcessor::with_build(move |rate| {
				PhaseVocoder::new(SpectralDynamics::new(curves.clone(), rate), WINDOW, rate)
			}),
			bypass: Arc::new(AtomicBool::new(false)),
			scope: Arc::new(Scope::new()),
			handle,
		}
	}

	/// The curves this product edits.
	pub fn handle(&self) -> SpectralDynamicsHandle {
		self.handle.clone()
	}

	/// The faceplate size this product declares to a host.
	pub const fn size() -> (f32, f32) {
		SIZE
	}

	/// The rate the effect inside is currently built for.
	pub fn sample_rate(&self) -> usize {
		self.processor.sample_rate()
	}
}

/// The face reads a short name, not the mapper nesting the DSP happens to have.
fn short_label(identifier: &str) -> Option<&'static str> {
	Some(match identifier.rsplit('.').next()? {
		"compress_threshold" => "Compress",
		"expand_threshold" => "Expand",
		"compress_ratio" => "Ratio",
		"expand_ratio" => "Expand Ratio",
		"range" => "Range",
		"makeup" => "Makeup",
		"attack" => "Attack",
		"release" => "Release",
		"mix" => "Mix",
		"morph" => "Morph",
		"curve_depth" => "Curve Depth",
		_ => return None,
	})
}

impl Parameters for SpectralDynamicsWeb {
	fn get_parameters(&self) -> Vec<Parameter> {
		let mut parameters = self.processor.parameters();

		for parameter in &mut parameters {
			if let Some(name) = short_label(&parameter.identifier) {
				parameter.meta.display_name = Some(name.into());
			}
		}

		let (threshold, ratio) = self.handle.curves().pages();
		let pages = to_binary(&(threshold, ratio)).expect("the curves serialize");

		parameters.push(Parameter::new("spectral_curves", Value::Serialized(pages)));
		parameters
	}

	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		if identifier == "spectral_curves"
			&& let SetValue::Serialized(bytes) = value
		{
			if let Ok((threshold, ratio)) =
				from_binary::<(Vec<Vec<CurvePoint>>, Vec<Vec<CurvePoint>>)>(bytes)
			{
				self.handle.curves().set_threshold(threshold);
				self.handle.curves().set_ratio(ratio);
				return true;
			}

			return false;
		}

		self.processor.set_parameter(identifier, value)
	}
}

impl Processor for SpectralDynamicsWeb {
	type Message = EffectMessage;

	fn delay(&self) -> usize {
		i_am_dsp::Processor::delay(&self.processor)
	}

	fn process(
		&mut self,
		samples: &mut [f32; 2],
		other: &[[f32; 2]],
		context: &mut Box<dyn ProcessContext>,
	) {
		let input = *samples;
		if self.bypass.load(Ordering::Relaxed) {
			self.scope.push_stereo(input, input, 0.0);
			return;
		}

		i_am_dsp::Processor::process(&mut self.processor, samples, other, context);
		self.scope.push_stereo(input, *samples, 0.0);
	}

	fn on_message(&self, message: Self::Message) {
		i_am_dsp::Processor::on_message(&self.processor, message)
	}
}

/// The vocoder's own controls, which are plumbing rather than a decision the
/// player makes: the product fixes the window, the offset and the make-up gain.
const HIDDEN: &[&str] = &["window_factor", "window_offset", "gain", "window_size"];

/// The faceplate's tabs: one section each.
///
/// Seven dynamics knobs, two ratio knobs and two threshold knobs stacked in one
/// column do not fit a window this size, and the two curve pictures need room
/// under their own controls. A tab per section gives each its own page; the
/// tabs name the sections, so the page does not draw the captions twice.
pub const SPECTRAL_DYNAMICS_PAGES: &[Page] = &[
	Page::new("Dynamics", &["Dynamics"]),
	Page::new("Curve", &["Curve"]),
];

/// The faceplate: the dynamics controls, then each curve with its own picture.
pub fn faceplate() -> Faceplate {
	Faceplate::new(TITLE, ACCENT, SIZE)
		.with_hidden(HIDDEN)
		.with_pages(SPECTRAL_DYNAMICS_PAGES)
}

impl Plugin for SpectralDynamicsWeb {
	const DESCRIPTOR: Descriptor = Descriptor::new("iamdsp.effects.spectral_dynamics.web", TITLE)
		.with_vendor("iamplugins")
		.with_version(env!("CARGO_PKG_VERSION"));

	const HAS_GUI: bool = true;

	fn new() -> Self {
		Self::default()
	}

	fn window_options() -> WindowOptions {
		i_am_web_gui::window_options(SIZE, TITLE)
	}

	fn param_map(&self) -> ParamMap {
		self.processor.param_map()
	}

	fn sync_params(&mut self) {
		self.processor.sync_params()
	}

	fn on_activate(&mut self, sample_rate: usize) {
		self.processor.activate(sample_rate)
	}

	fn editor(&self) -> Option<Box<dyn i_am_plugin::Editor<Self>>> {
		let handle = self.handle.clone();

		i_am_web_gui::editor_with(
			self,
			faceplate(),
			self.bypass.clone(),
			self.scope.clone(),
			self.processor.sample_rate(),
			move |request: &ProtocolRequest<'_>| curve_route(request, &handle),
			|_: &str| false,
		)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// The page posts arrays and gets the same arrays back.
	#[test]
	fn the_curve_route_round_trips_the_pages() {
		let handle = SpectralDynamicsHandle::new();
		let body = r#"{"threshold":[[[0.0,0.2,0.0],[1.0,0.8,0.6]]],"ratio":[[[0.0,0.0,0.0],[1.0,1.0,0.0]]]}"#;

		let answer = handle.set_json(body);

		assert!(answer.contains("0.2"), "answer {answer}");
		assert!(answer.contains("0.8"), "answer {answer}");

		let (threshold, ratio) = handle.curves().pages();

		assert_eq!(threshold[0].len(), 2);
		assert!((threshold[0][1].value - 0.8).abs() < 1e-6);
		assert!(
			(threshold[0][1].bend - 0.6).abs() < 1e-6,
			"the bend has to survive the route"
		);
		assert_eq!(ratio[0].len(), 2);
	}

	/// A page edit is state, so it has to survive the host saving and loading it.
	#[test]
	fn the_curves_survive_a_state_round_trip() {
		let product = SpectralDynamicsWeb::default();
		product.handle.curves().set_threshold(vec![vec![
			CurvePoint::new(0.0, 0.1),
			CurvePoint::new(1.0, 0.9),
		]]);

		let saved = product
			.get_parameters()
			.into_iter()
			.find(|parameter| parameter.identifier == "spectral_curves")
			.expect("the state parameter");

		let Value::Serialized(bytes) = saved.value else {
			panic!("the state parameter is serialized");
		};

		let mut restored = SpectralDynamicsWeb::default();
		assert!(restored.set_parameter("spectral_curves", SetValue::Serialized(bytes)));

		let (threshold, _) = restored.handle.curves().pages();

		assert!((threshold[0][0].value - 0.1).abs() < 1e-6);
	}
}
