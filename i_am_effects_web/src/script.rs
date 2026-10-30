//! Rhai scripts for the scripted products.
//!
//! A script is compiled once and rendered into a lookup table by a worker thread.
//! The audio thread only reads those tables: Rhai allocates, can loop, and can take
//! an unbounded amount of time, none of which belongs in a process callback.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crossbeam_channel::{Receiver, Sender, bounded};
use rhai::{Array, Dynamic, Engine, Scope};

/// The most operations one evaluation of a script may run.
const MAX_OPERATIONS: u64 = 200_000;

/// A compiled script, with the engine that owns its AST.
pub struct CompiledScript {
	engine: Engine,
	ast: rhai::AST,
}

impl CompiledScript {
	/// Compiles a script body.
	///
	/// The body is wrapped in a block so its lets are local to one evaluation.
	/// Without the block Rhai keeps top-level variables in the scope: every later
	/// call then walks a longer scope, and the memory grows without bound.
	pub fn compile(source: &str) -> Result<Self, String> {
		let mut engine = Engine::new();
		engine.set_max_operations(MAX_OPERATIONS);
		engine.set_max_expr_depths(32, 64);

		let wrapped = format!("{{ {source} }}");
		let ast = engine.compile(&wrapped).map_err(|error| error.to_string())?;

		Ok(Self { engine, ast })
	}

	/// A scope with the four controls and the input variable.
	pub fn scope(params: [f32; 4]) -> Scope<'static> {
		let mut scope = Scope::new();
		scope.push("p1", params[0]);
		scope.push("p2", params[1]);
		scope.push("p3", params[2]);
		scope.push("p4", params[3]);
		scope.push("x", 0.0_f32);
		scope
	}

	/// Evaluates the script as one number.
	pub fn number(&self, scope: &mut Scope) -> Result<f32, String> {
		let value: Dynamic = self
			.engine
			.eval_ast_with_scope(scope, &self.ast)
			.map_err(|error| error.to_string())?;

		value
			.as_float()
			.map_err(|_| "the script must return a number".to_string())
	}

	/// Evaluates the script as `[frequency, amplitude]`.
	#[allow(dead_code)] // the frequency mapper uses it next
	pub fn pair(&self, scope: &mut Scope) -> Result<(f32, f32), String> {
		let value: Dynamic = self
			.engine
			.eval_ast_with_scope(scope, &self.ast)
			.map_err(|error| error.to_string())?;

		let array: Array = value.cast();

		if array.len() != 2 {
			return Err("the script must return [frequency, amplitude]".to_string());
		}

		let frequency = array[0]
			.clone()
			.try_cast::<f32>()
			.ok_or_else(|| "the first value must be a frequency".to_string())?;
		let amplitude = array[1]
			.clone()
			.try_cast::<f32>()
			.ok_or_else(|| "the second value must be an amplitude".to_string())?;

		Ok((frequency, amplitude))
	}
}

/// What the last render did.
#[derive(Clone, Debug, Default)]
pub struct ScriptStatus {
	/// Why the last render failed, if it did.
	pub error: Option<String>,
	/// How many tables were rendered successfully.
	pub applied: u64,
}

/// One request for a new table.
struct ScriptRequest {
	source: Arc<str>,
	params: [f32; 4],
	extra: f32,
}

/// The script, its status, and the table the audio thread reads.
///
/// The worker thread owns the Rhai engine; this struct only shares the values the
/// editor and the effect need. T is the product's table.
pub struct ScriptWorker<T: Send + Sync + 'static> {
	source: Arc<Mutex<String>>,
	source_generation: Arc<AtomicU64>,
	status: Arc<Mutex<ScriptStatus>>,
	curve: Arc<Mutex<Vec<[f32; 3]>>>,
	table: Arc<Mutex<Option<Arc<T>>>>,
	table_generation: Arc<AtomicU64>,
	compiled: Arc<Mutex<Option<Arc<CompiledScript>>>>,
	compiled_generation: Arc<AtomicU64>,
	requests: Sender<ScriptRequest>,
}

impl<T: Send + Sync + 'static> ScriptWorker<T> {
	/// Starts a worker.
	///
	/// render turns a compiled script, the four controls and the extra value
	/// (the distortion's input gain, or 0 for a frequency mapper) into a table.
	/// curve_of reduces a table to the points the page draws.
	pub fn spawn<R, C>(render: R, curve_of: C) -> Arc<Self>
	where
		R: Fn(&CompiledScript, [f32; 4], f32) -> Result<T, String> + Send + 'static,
		C: Fn(&T) -> Vec<[f32; 3]> + Send + Sync + 'static,
	{
		let source = Arc::new(Mutex::new(String::new()));
		let source_generation = Arc::new(AtomicU64::new(0));
		let status = Arc::new(Mutex::new(ScriptStatus::default()));
		let curve = Arc::new(Mutex::new(Vec::new()));
		let table = Arc::new(Mutex::new(None));
		let table_generation = Arc::new(AtomicU64::new(0));
		let compiled = Arc::new(Mutex::new(None));
		let compiled_generation = Arc::new(AtomicU64::new(0));

		let (requests, receiver) = bounded(8);

		let worker = Arc::new(Self {
			source: source.clone(),
			source_generation: source_generation.clone(),
			status: status.clone(),
			curve: curve.clone(),
			table: table.clone(),
			table_generation: table_generation.clone(),
			compiled: compiled.clone(),
			compiled_generation: compiled_generation.clone(),
			requests,
		});

		std::thread::spawn(move || {
			run(
				receiver,
				status,
				curve,
				table,
				table_generation,
				compiled,
				compiled_generation,
				render,
				curve_of,
			);
		});

		worker
	}

	/// The current script text.
	pub fn source(&self) -> String {
		self.source.lock().map(|source| source.clone()).unwrap_or_default()
	}

	/// Stores a new script and returns its generation.
	pub fn set_source(&self, source: String) -> u64 {
		if let Ok(mut current) = self.source.lock() {
			*current = source;
		}

		self.source_generation.fetch_add(1, Ordering::Release) + 1
	}

	/// The generation of the current script.
	pub fn source_generation(&self) -> u64 {
		self.source_generation.load(Ordering::Acquire)
	}

	/// Asks for a table for the current script and controls.
	///
	/// A full queue drops the request; the caller can simply ask again, and the
	/// last values it sent are the ones the worker renders.
	pub fn request(&self, params: [f32; 4], extra: f32) -> bool {
		let Ok(source) = self.source.lock() else {
			return false;
		};

		let request = ScriptRequest {
			source: Arc::from(source.as_str()),
			params,
			extra,
		};

		self.requests.try_send(request).is_ok()
	}

	/// The generation of the table, bumped after every successful render.
	pub fn table_generation(&self) -> u64 {
		self.table_generation.load(Ordering::Acquire)
	}

	/// The current table, if one has been rendered.
	pub fn table(&self) -> Option<Arc<T>> {
		self.table.lock().ok()?.clone()
	}

	/// The generation of the compiled script, bumped after every attempt.
	pub fn compiled_generation(&self) -> u64 {
		self.compiled_generation.load(Ordering::Acquire)
	}

	/// The compiled script, for a product that runs it on the audio thread.
	pub fn compiled(&self) -> Option<Arc<CompiledScript>> {
		self.compiled.lock().ok()?.clone()
	}

	/// The points the page draws for the current table.
	pub fn curve(&self) -> Vec<[f32; 3]> {
		self.curve.lock().map(|curve| curve.clone()).unwrap_or_default()
	}

	/// What the last render did.
	pub fn status(&self) -> ScriptStatus {
		self.status.lock().map(|status| status.clone()).unwrap_or_default()
	}
}

#[allow(clippy::too_many_arguments)]
fn run<T, R, C>(
	receiver: Receiver<ScriptRequest>,
	status: Arc<Mutex<ScriptStatus>>,
	curve: Arc<Mutex<Vec<[f32; 3]>>>,
	table: Arc<Mutex<Option<Arc<T>>>>,
	table_generation: Arc<AtomicU64>,
	compiled: Arc<Mutex<Option<Arc<CompiledScript>>>>,
	compiled_generation: Arc<AtomicU64>,
	render: R,
	curve_of: C,
)
where
	T: Send + Sync + 'static,
	R: Fn(&CompiledScript, [f32; 4], f32) -> Result<T, String>,
	C: Fn(&T) -> Vec<[f32; 3]>,
{
	let mut compiled_local: Option<(Arc<str>, Arc<CompiledScript>)> = None;

	while let Ok(first) = receiver.recv() {
		// Only the newest request matters: a knob that moved ten times should
		// render once, at where it ended up.
		let mut request = first;

		while let Ok(next) = receiver.try_recv() {
			request = next;
		}

		let stale = compiled_local
			.as_ref()
			.map(|(compiled_source, _)| compiled_source.as_ref() != request.source.as_ref())
			.unwrap_or(true);

		if stale {
			match CompiledScript::compile(request.source.as_ref()) {
				Ok(script) => {
					let script = Arc::new(script);
					compiled_local = Some((request.source.clone(), script.clone()));

					if let Ok(mut current) = compiled.lock() {
						*current = Some(script);
					}

					compiled_generation.fetch_add(1, Ordering::Release);
				},
				Err(error) => {
					if let Ok(mut current) = compiled.lock() {
						*current = None;
					}

					compiled_generation.fetch_add(1, Ordering::Release);
					set_error(&status, error);
					continue;
				},
			}
		}

		let Some((_, script)) = compiled_local.as_ref() else {
			continue;
		};

		match render(script.as_ref(), request.params, request.extra) {
			Ok(rendered) => {
				let rendered = Arc::new(rendered);
				let points = curve_of(&rendered);

				if let Ok(mut current) = table.lock() {
					*current = Some(rendered);
				}

				if let Ok(mut current) = curve.lock() {
					*current = points;
				}

				if let Ok(mut current) = status.lock() {
					current.error = None;
					current.applied = current.applied.saturating_add(1);
				}

				table_generation.fetch_add(1, Ordering::Release);
			},
			Err(error) => set_error(&status, error),
		}
	}
}

fn set_error(status: &Arc<Mutex<ScriptStatus>>, error: String) {
	if let Ok(mut status) = status.lock() {
		status.error = Some(error);
	}
}

/// Escapes a string for JSON.
pub(crate) fn json_escape(value: &str) -> String {
	let mut out = String::with_capacity(value.len());

	for character in value.chars() {
		match character {
			'"' => out.push_str("\\\""),
			'\\' => out.push_str("\\\\"),
			'\n' => out.push_str("\\n"),
			'\r' => out.push_str("\\r"),
			'\t' => out.push_str("\\t"),
			character if (character as u32) < 0x20 => {
				let _ = std::fmt::Write::write_fmt(&mut out, format_args!("\\u{:04x}", character as u32));
			},
			character => out.push(character),
		}
	}

	out
}
