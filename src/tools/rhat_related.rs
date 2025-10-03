//! Rhai related implementation

use crate::prelude::{FreqInfo, GenFreqInfo};

lazy_static::lazy_static! {
	static ref RHAI_ENGINE: rhai::Engine = {
		use rhai::packages::Package;

		let mut engine = rhai::Engine::new();
		engine.register_global_module(rhai_rand::RandomPackage::new().as_shared_module());
		engine
	};
}

/// Rhai frequency generator, can be used in [`crate::prelude::AdditiveOsc`]
/// 
/// Note: the rhai script can access the following variables:
/// - `a: f32`: the first parameter of the script
/// - `b: f32`: the second parameter of the script
/// - `c: f32`: the third parameter of the script
/// - `d: f32`: the fourth parameter of the script
/// - `sample_rate: f32`: the sample rate of the audio signal
/// - `index: i32`: the index of the current frequency
/// - `total_amount: i32`: the total amount of frequencies
/// 
/// The script should create two values:
/// - `ratio: f32`: the ratio of the frequency to the sample rate
/// - `amplitude: f32`: the amplitude of the frequency
///
/// Otherwise [`RhaiFreqGenError::MissingVariable`] will be returned.
///  
/// If the script fails to evaluate, the [`crate::tools::rhat_related::RhaiFreqGenError`] will be set.
pub struct RhaiFreqGen {
	/// The changable parameters that can be used in the script
	pub daw_values: [f32; 4],
	/// The sample rate of the audio signal
	pub sample_rate: usize,
	ast: Option<(String, rhai::AST)>,
	need_recaculate: bool,
	error: Option<RhaiFreqGenError>,
}

#[derive(Debug, thiserror::Error)]
/// Rhai frequency generator error
pub enum RhaiFreqGenError {
	/// Failed to parse script
	#[error("Failed to parse script: {0}")]
	ParseError(#[from] rhai::ParseError),
	/// Failed to read file
	#[error("Failed to read file: {0}")]
	Io(#[from] std::io::Error),
	/// Failed to evaluate script
	#[error("Failed to evaluate script: {0}")]
	Eval(#[from] Box<rhai::EvalAltResult>),
	/// Missing variable that is required to evaluate script
	/// 
	/// The variable name is included in the error message.
	/// 
	/// Can also be caused by mismatched variable types.
	#[error("Missing variable: {0}")]
	MissingVariable(String),
}

impl RhaiFreqGen {
	/// Create a new [`RhaiFreqGen`]
	pub fn new(sample_rate: usize) -> Self {
		Self {
			daw_values: [0.0; 4],
			sample_rate,
			ast: None,
			need_recaculate: true,
			error: None,
		}
	}

	/// Get the possible error from the last evaluation
	pub fn get_eval_error(&mut self) -> Result<(), RhaiFreqGenError> {
		if let Some(e) = self.error.take() {
			return Err(e);
		}
		Ok(())
	}

	/// Load the script from a file
	pub fn load_from_file(&mut self, path: impl AsRef<std::path::Path>) -> Result<(), RhaiFreqGenError> {
		let code = std::fs::read_to_string(path)?;
		self.load_script(code)?;
		Ok(())
	}

	/// Load the script from a string
	pub fn load_script(&mut self, code: impl AsRef<str>) -> Result<(), RhaiFreqGenError> {
		let code = code.as_ref();
		if code.is_empty() {
			self.ast = None;
			return Ok(());
		}

		let ast = RHAI_ENGINE.compile(code)?;
		self.need_recaculate = true;
		self.ast = Some((code.to_string(), ast));
		Ok(())
	}
}

impl GenFreqInfo for RhaiFreqGen {
	fn gen_info(&mut self, index: usize, total_amount: usize) -> FreqInfo {
		use std::f32::consts::PI;
		self.need_recaculate = false;
		let Some(ast) = &self.ast else {
			return FreqInfo {
				ratio: index as f32 + 1.0,
				amplitude: (2.0 / PI) * (- 1.0_f32).powi(index as i32) / (index as f32 + 1.0),
			};
		};

		let mut scope = rhai::Scope::new();
		scope.push("a", self.daw_values[0]);
		scope.push("b", self.daw_values[1]);
		scope.push("c", self.daw_values[2]);
		scope.push("d", self.daw_values[3]);
		scope.push("sample_rate", self.sample_rate as i32);
		scope.push("index", index as i32);
		scope.push("total_amount", total_amount as i32);

		if let Err(e) = RHAI_ENGINE.run_ast_with_scope(&mut scope, &ast.1) {
			self.error = Some(RhaiFreqGenError::Eval(Box::new(*e)));
			return Default::default();
		};

		let Some(ratio) = scope.get_value::<f32>("ratio") else {
			self.error = Some(RhaiFreqGenError::MissingVariable("ratio".to_string()));
			return Default::default();
		};
		let Some(amplitude) = scope.get_value::<f32>("amplitude") else {
			self.error = Some(RhaiFreqGenError::MissingVariable("amplitude".to_string()));
			return Default::default();
		};


		FreqInfo {
			ratio,
			amplitude,
		}
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) -> bool {
		use egui::*;

		let mut clear_error = false;

		if let Some(error) = self.error.as_ref() {
			ui.colored_label(Color32::RED, format!("ERR: {}", error));
			clear_error = ui.button("Clear Error").clicked();
		}

		if clear_error {
			self.error = None;
		}

		let mut params = self.daw_values;

		Grid::new(format!("{}_rhai_freq_gen", id_prefix))
			.num_columns(4)
			.show(ui, |ui| 
		{
			ui.label("a");
			ui.add(Slider::new(&mut params[0], -1.0..=1.0));
			ui.label("b");
			ui.add(Slider::new(&mut params[1], -1.0..=1.0));
			ui.end_row();
			
			ui.label("c");
			ui.add(Slider::new(&mut params[2], -1.0..=1.0));
			ui.label("d");
			ui.add(Slider::new(&mut params[3], -1.0..=1.0));
			ui.end_row();
		});

		if params != self.daw_values {
			self.daw_values = params;
			self.need_recaculate = true;
		}

		ui.horizontal(|ui| {
			if ui.button("Reload").clicked() {
				self.need_recaculate = true;
			}

			if ui.button("Load Code").clicked() {
				let dialog = rfd::FileDialog::new().add_filter("Rhai files", &["rhai"]);
				if let Some(path) = dialog.pick_file() {
					match self.load_from_file(path) {
						Ok(()) => {},
						Err(e) => self.error = Some(e),
					}
				}
			}

			if ui.button("Recaculate Ratio").clicked() {
				self.need_recaculate = true;
			}
		});
		
		CollapsingHeader::new("Show Code")
			.id_salt(format!("{}_rhai_freq_gen_code", id_prefix))
			.show(ui, |ui| {
				if let Some((code, _)) = &self.ast {
					ui.label(code)
				}else {
					ui.label("No code loaded")
				}
			});

		self.need_recaculate
	}
}