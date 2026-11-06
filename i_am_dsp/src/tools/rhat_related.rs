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

/// Rhai frequency generator, can be used in [`crate::prelude::AdditiveOsc`].
///
/// The script runs once per partial, with these variables in scope:
///
/// - `a`, `b`, `c`, `d`: `f32` in `[-1.0, 1.0]`, the product's four knobs
/// - `sample_rate`: `i32`, the host's rate in hertz
/// - `index`: `i32`, which partial is being asked for, from `0` to `total_amount - 1`
/// - `total_amount`: `i32`, how many partials the oscillator plays
///
/// It has to leave four variables in scope, each declared with `let`: Rhai
/// treats an assignment to an undeclared name as a runtime error, and the
/// values are read back from the script's scope afterwards. All four are
/// `f32`; an integer is not read back as one, so convert with `to_float`.
///
/// - `ratio`: how far above the note's fundamental the partial sits. `1.0` is
///   the fundamental itself. Only a whole-number ratio can use the wavetable
///   fast path, and only while nothing in the script is damped.
/// - `amplitude`: the partial's level, linear.
/// - `damping`: how fast it decays, as `exp(-damping * cycles)`; `0.0` keeps
///   the partial ringing.
/// - `phase`: where the partial starts within its cycle, in `0.0..=1.0`.
///
/// A partial whose output is missing (or is not an `f32`) is left silent, and
/// [`Self::get_eval_error`] reports it as [`RhaiFreqGenError::MissingVariable`].
///
/// ```rhai
/// let n = to_float(index + 1);
/// let ratio = n;
/// let amplitude = 1.0 / n;
/// let damping = 0.0;
/// let phase = 0.0;
/// ```
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
	#[skip]
	script_revision: u64,

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
	} else {
		vec![]
	}
}

fn parse_ast(input: Vec<u8>) -> Option<(String, rhai::AST)> {
	if input.is_empty() {
		None
	} else {
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
			script_revision: 0,

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
	pub fn load_from_file(
		&mut self,
		path: impl AsRef<std::path::Path>,
	) -> Result<(), RhaiFreqGenError> {
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
		self.script_revision = self.script_revision.wrapping_add(1);
		Ok(())
	}

	/// Installs a script that was compiled off the audio thread.
	pub fn load_compiled(&mut self, script: RhaiScript) {
		self.need_recaculate = true;
		self.error = None;
		self.ast = Some((script.code, script.ast));
		self.script_revision = self.script_revision.wrapping_add(1);
	}
}

/// A script compiled away from the audio thread, ready for [`RhaiFreqGen`].
///
/// Compiling a script allocates and can take milliseconds; a host must not pay
/// that inside a process callback, so the page compiles it and the audio thread
/// only swaps the result in.
#[derive(Clone)]
pub struct RhaiScript {
	code: String,
	ast: rhai::AST,
}

impl RhaiScript {
	/// Compiles a script body.
	pub fn compile(code: &str) -> Result<Self, String> {
		let ast = RHAI_ENGINE
			.compile(code)
			.map_err(|error| error.to_string())?;

		Ok(Self {
			code: code.to_string(),
			ast,
		})
	}

	/// The script's body.
	pub fn code(&self) -> &str {
		&self.code
	}

	/// Runs the script for a picture of the spectrum, away from the audio thread.
	///
	/// Each partial is asked for its ratio and amplitude only; the picture on the
	/// page does not need the damping or the phase.
	pub fn partials(
		&self,
		a: f32,
		b: f32,
		c: f32,
		d: f32,
		sample_rate: usize,
		total: usize,
	) -> Vec<[f32; 2]> {
		let mut out = Vec::with_capacity(total);

		for index in 0..total {
			let mut scope = rhai::Scope::new();
			scope.push("a", a);
			scope.push("b", b);
			scope.push("c", c);
			scope.push("d", d);
			scope.push("sample_rate", sample_rate as i32);
			scope.push("index", index as i32);
			scope.push("total_amount", total as i32);

			if RHAI_ENGINE
				.run_ast_with_scope(&mut scope, &self.ast)
				.is_err()
			{
				break;
			}

			let ratio = scope.get_value::<f32>("ratio").unwrap_or(1.0);
			let amplitude = scope.get_value::<f32>("amplitude").unwrap_or(0.0);

			out.push([ratio, amplitude]);
		}

		out
	}
}

impl GenFreqInfo for RhaiFreqGen {
	fn revision(&self) -> u64 {
		let mut hash = 0xcbf2_9ce4_8422_2325u64;

		for value in [self.a, self.b, self.c, self.d] {
			hash = (hash ^ value.to_bits() as u64).wrapping_mul(0x0000_0100_0000_01b3);
		}

		hash = (hash ^ self.sample_rate as u64).wrapping_mul(0x0000_0100_0000_01b3);
		hash = (hash ^ self.script_revision).wrapping_mul(0x0000_0100_0000_01b3);

		hash
	}

	fn gen_info(&mut self, index: usize, total_amount: usize) -> FreqInfo {
		use std::f32::consts::PI;
		self.need_recaculate = false;
		let Some(ast) = &self.ast else {
			return FreqInfo {
				ratio: index as f32 + 1.0,
				amplitude: (2.0 / PI) * (-1.0_f32).powi(index as i32) / (index as f32 + 1.0),
				damping: 0.0,
				phase: 0.0,
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
		let Some(damping) = scope.get_value::<f32>("damping") else {
			self.error = Some(RhaiFreqGenError::MissingVariable("damping".to_string()));
			return Default::default();
		};
		let Some(phase) = scope.get_value::<f32>("phase") else {
			self.error = Some(RhaiFreqGenError::MissingVariable("phase".to_string()));
			return Default::default();
		};

		FreqInfo {
			ratio,
			amplitude,
			damping,
			phase,
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
			.show(ui, |ui| {
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

		if params[0] != self.a || params[1] != self.b || params[2] != self.c || params[3] != self.d
		{
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
						// egui 0.36 made `DroppedFile` a trait: the path is a method now.
						inner.path().to_path_buf()
					});
				});
			}

			if ui.button("Load Code").clicked() {
				use egui_file::FileDialog;
				use std::ffi::OsStr;

				let filter = Box::new({
					let ext = Some(OsStr::new("rhai"));
					move |path: &std::path::Path| -> bool { path.extension() == ext }
				});
				let mut dialog = FileDialog::open_file().show_files_filter(filter);
				if let Some(opened) = &self.opened_file {
					dialog = dialog.initial_path(opened.parent().unwrap_or(opened));
				}
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
				if path
					.extension()
					.map(|ext| ext.to_string_lossy().to_lowercase() != "rhai")
					.unwrap_or(true)
				{
					return;
				}

				self.opened_file = Some(path.clone());

				match self.load_from_file(path) {
					Ok(()) => {}
					Err(e) => self.error = Some(e),
				}
				self.allow_load_code = false;
			}

			if ui.button("Recaculate Ratio").clicked() {
				self.need_recaculate = true;
			}
			if ui
				.selectable_label(self.allow_load_code, "Allow Load Code")
				.clicked()
			{
				self.allow_load_code = !self.allow_load_code;
			}
		});

		CollapsingHeader::new("Show Code")
			.id_salt(format!("{}_rhai_freq_gen_code", id_prefix))
			.show(ui, |ui| {
				if let Some((code, _)) = &self.ast {
					ui.label(code)
				} else {
					ui.label("No code loaded")
				}
			});

		self.need_recaculate
	}
}
