//! Rhai related implementation

use i_am_dsp_derive::Parameters;

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
#[derive(Parameters)]
#[default_float_range(min = -1.0, max = 1.0)]
pub struct RhaiFreqGen {
	/// The changable parameter a that can be used in the script
	pub a: f32,
	/// The changable parameter a that can be used in the script
	pub b: f32,
	/// The changable parameter a that can be used in the script
	pub c: f32,
	/// The changable parameter a that can be used in the script
	pub d: f32,
	#[range(min = 1, max = 192000)]
	/// The sample rate of the audio signal
	pub sample_rate: usize,
	#[persist(serialize = "format_ast", deserialize = "parse_ast")]
	ast: Option<(String, rhai::AST)>,
	#[skip]
	need_recaculate: bool,
	#[skip]
	error: Option<RhaiFreqGenError>,

	#[cfg(feature = "real_time_demo")]
	#[skip]
	allow_load_code: bool,
	#[cfg(feature = "real_time_demo")]
	#[skip]
	opened_file: Option<std::path::PathBuf>,
	#[cfg(feature = "real_time_demo")]
	#[skip]
	dialog: Option<egui_file::FileDialog>,
}

fn format_ast(input: &Option<(String, rhai::AST)>) -> Vec<u8> {
	if let Some((code, _)) = input {
		code.as_bytes().to_vec()
	}else {
		vec![]
	}
}

fn parse_ast(input: Vec<u8>) -> Option<(String, rhai::AST)> {
	if input.is_empty() {
		None
	}else {
		let code = String::from_utf8(input).unwrap();
		let ast = RHAI_ENGINE.compile(&code).unwrap();
		Some((code, ast))
	}
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
			a: 0.0,
			b: 0.0,
			c: 0.0,
			d: 0.0,
			sample_rate,
			ast: None,
			need_recaculate: true,
			error: None,

			#[cfg(feature = "real_time_demo")]
			allow_load_code: false,
			#[cfg(feature = "real_time_demo")]
			opened_file: None,
			#[cfg(feature = "real_time_demo")]
			dialog: None,
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
		scope.push("a", self.a);
		scope.push("b", self.b);
		scope.push("c", self.c);
		scope.push("d", self.d);
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

		let mut params = [self.a, self.b, self.c, self.d];

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

		if params[0] != self.a || params[1] != self.b || params[2] != self.c || params[3] != self.d {
			self.a = params[0];
			self.b = params[1];
			self.c = params[2];
			self.d = params[3];
			self.need_recaculate = true;
		}

		ui.horizontal(|ui| {
			if ui.button("Reload").clicked() {
				self.need_recaculate = true;
			}

			let mut path = None;

			if self.allow_load_code {
				ui.input(|input| {
					path = input.raw.dropped_files.first().map(|inner| {
						inner.path.clone()
					}).unwrap_or_default();
				});
			}

			if ui.button("Load Code").clicked() {
				use std::ffi::OsStr;
				use egui_file::FileDialog;

				let filter = Box::new({
					let ext = Some(OsStr::new("rhai"));
					move |path: &std::path::Path| -> bool {
						path.extension() == ext
					}
				});
				let mut dialog = FileDialog::open_file(self.opened_file.clone()).show_files_filter(filter);
				dialog.open();

				self.dialog = Some(dialog);
			}
			
			if let Some(dialog) = self.dialog.as_mut() {
				let dialog = dialog.show(ui.ctx());
				if dialog.selected() {
					path = dialog.path().map(|path| path.to_path_buf());
				}
			}

			if let Some(path) = path {
				if path.extension().map(|ext| ext.to_string_lossy().to_lowercase() != "rhai").unwrap_or(true) {
					return;
				}
				
				self.opened_file = Some(path.clone());

				match self.load_from_file(path) {
					Ok(()) => {},
					Err(e) => self.error = Some(e),
				}
				self.allow_load_code = false;
			}

			if ui.button("Recaculate Ratio").clicked() {
				self.need_recaculate = true;
			}
			if ui.selectable_label(self.allow_load_code, "Allow Load Code").clicked() {
				self.allow_load_code = !self.allow_load_code;
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