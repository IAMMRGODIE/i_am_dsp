//! The script a scripted module runs, and the worker that renders it.
//!
//! Both scripted products keep their source in a worker that re-renders a table
//! off the audio thread; the effect swaps the table in when it changes. The rack
//! keeps one worker per module, next to the scopes, meters and impulse slots.

use std::sync::Arc;

use i_am_effects::{
	DistortionTable, MapperTable, ScriptWorker, distortion_worker, mapper_worker,
};

/// The worker of one scripted module.
pub(crate) enum ScriptSlot {
	/// The scripted distortion's transfer table.
	Distortion(Arc<ScriptWorker<DistortionTable>>),
	/// The scripted mapper's frequency table.
	Mapper(Arc<ScriptWorker<MapperTable>>),
}

impl ScriptSlot {
	/// A distortion worker, running the product's default script.
	pub fn distortion() -> Self {
		Self::Distortion(distortion_worker())
	}

	/// A mapper worker, running the product's default script.
	pub fn mapper() -> Self {
		Self::Mapper(mapper_worker())
	}

	/// Replaces the source, which the worker renders on its thread.
	pub fn set_source(&self, source: String) {
		match self {
			Self::Distortion(worker) => {
				worker.set_source(source);
			},
			Self::Mapper(worker) => {
				worker.set_source(source);
			},
		}
	}

	/// Asks the worker to render for these parameters. The distortion takes its
	/// input gain, the mapper the sample rate; both read them from the module.
	pub fn request(&self, params: [f32; 4], extra: f32) {
		match self {
			Self::Distortion(worker) => {
				worker.request(params, extra);
			},
			Self::Mapper(worker) => {
				worker.request(params, extra);
			},
		}
	}

	/// The JSON the card polls: the source, the last error, and whether the
	/// rendered table matches the source.
	pub fn json(&self) -> String {
		let (source, status) = match self {
			Self::Distortion(worker) => (worker.source(), worker.status()),
			Self::Mapper(worker) => (worker.source(), worker.status()),
		};
		let error = match status.error {
			Some(error) => serde_json::to_string(&error).unwrap_or_else(|_| "null".to_string()),
			None => "null".to_string(),
		};

		format!(
			"{{\"source\":{},\"error\":{error},\"applied\":{}}}",
			serde_json::to_string(&source).unwrap_or_else(|_| "\"\"".to_string()),
			status.applied
		)
	}

	/// The rendered curve, as the hero draws it.
	pub fn curve_json(&self) -> String {
		let points = match self {
			Self::Distortion(worker) => worker.curve(),
			Self::Mapper(worker) => worker.curve(),
		};

		let mut json = String::from("{\"points\":[");

		for (index, [x, y, _]) in points.iter().enumerate() {
			if index > 0 {
				json.push(',');
			}

			let _ = std::fmt::Write::write_fmt(&mut json, format_args!("[{x:.4},{y:.4}]"));
		}

		json.push_str("]}");
		json
	}
}
