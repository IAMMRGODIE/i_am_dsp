//! The convolver product: an impulse response loaded from a file, convolved by
//! the FFT engine in `i_am_dsp`.
//!
//! The IR is not a control. Clicking its picture asks the host side of the editor
//! for a native file dialog (on Windows, owned by the DAW window so it comes up in
//! front); a page without that path can still POST the bytes to the product's
//! `/ir` route. Either way the file is decoded and resampled off the audio thread.
//! A shared [IrSlot] is how the result reaches the audio thread and how the page
//! reads the waveform back; the IR is also serialized into the plug-in state, so a
//! project keeps the room it was saved with.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use i_am_dsp::prelude::{Effect, FftConvolver, ParamMap, Parameter, ProcessContext, SetValue};
use i_am_dsp::tools::pcm_data::{load_from_bytes, resample};
use i_am_dsp_derive::Parameters;
use i_am_plugin::processor::EffectMessage;
use i_am_plugin::processor::EffectProcessor;
use i_am_plugin::{AspectRatioStrategy, Descriptor, Editor, GuiResizeHints, Plugin, WindowOptions};
use i_am_web_gui::{ProtocolRequest, ProtocolResponse, Scope};
use serde::{Deserialize, Serialize};

/// The colour of the faceplate.
const ACCENT: &str = "#b08cff";
/// The design size of the page.
const SIZE: (f32, f32) = (520.0, 330.0);
/// How many points the waveform picture uses on each side of zero.
const PEAKS: usize = 512;
/// The longest impulse response the loader keeps, in seconds.
const MAX_SECONDS: usize = 30;

/// One loaded impulse response, with the picture the page draws.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IrData {
	/// The file name the page sent, for the hero badge.
	pub name: String,
	/// The rate the samples are at, which is the host's once they are loaded.
	pub rate: usize,
	/// The left channel, or the mono channel duplicated.
	pub left: Vec<f32>,
	/// The right channel.
	pub right: Vec<f32>,
	/// Minimum and maximum of both channels in each picture column.
	pub peaks: Vec<(f32, f32)>,
	/// A cheap identity, so the audio thread can notice a state load.
	pub fingerprint: u64,
}

impl IrData {
	/// An impulse response from raw samples, with the picture the card draws.
	pub fn from_samples(name: String, rate: usize, left: Vec<f32>, right: Vec<f32>) -> Self {
		let fingerprint = fingerprint(&name, rate, &left, &right);
		let peaks = peaks_of(&left, &right);

		Self {
			name,
			rate,
			left,
			right,
			peaks,
			fingerprint,
		}
	}

	/// The same response converted to another sample rate.
	pub fn resampled(&self, rate: usize) -> Self {
		if rate == self.rate {
			return self.clone();
		}

		let [left, right] = resample([self.left.clone(), self.right.clone()], self.rate, rate)
			.unwrap_or_else(|_| [self.left.clone(), self.right.clone()]);

		Self::from_samples(self.name.clone(), rate, left, right)
	}

	/// The length of the longer channel, in samples.
	pub fn samples(&self) -> usize {
		self.left.len().max(self.right.len())
	}

	/// Whether the response has anything to convolve with.
	pub fn is_loaded(&self) -> bool {
		self.samples() > 0
	}

	/// The length in seconds at the rate the samples are at.
	pub fn seconds(&self) -> f32 {
		if self.rate == 0 {
			0.0
		} else {
			self.samples() as f32 / self.rate as f32
		}
	}
}

/// What the page and the audio thread share: the current IR, its generation, the
/// host rate, and the last loading error.
pub struct IrSlot {
	current: Mutex<Option<Arc<IrData>>>,
	error: Mutex<Option<String>>,
	generation: AtomicU64,
	rate: AtomicUsize,
	/// Whether the native picker is open right now.
	pending: AtomicBool,
}

impl Default for IrSlot {
	fn default() -> Self {
		Self::new()
	}
}

impl IrSlot {
	/// An empty slot, with the rate at the effect crate's placeholder.
	pub fn new() -> Self {
		Self {
			current: Mutex::new(None),
			error: Mutex::new(None),
			generation: AtomicU64::new(0),
			rate: AtomicUsize::new(44_100),
			pending: AtomicBool::new(false),
		}
	}

	/// The rate the effect is currently built for.
	pub fn host_rate(&self) -> usize {
		self.rate.load(Ordering::Relaxed)
	}

	/// Record the rate of the effect that is being built.
	pub fn set_host_rate(&self, rate: usize) {
		self.rate.store(rate, Ordering::Relaxed);
	}

	/// The generation moves whenever an IR is stored, so the audio thread can
	/// skip the lock while nothing has happened.
	pub fn generation(&self) -> u64 {
		self.generation.load(Ordering::Acquire)
	}

	/// Whether a native picker is open right now.
	pub fn is_pending(&self) -> bool {
		self.pending.load(Ordering::Relaxed)
	}

	/// Marks the native picker as open or finished.
	pub fn set_pending(&self, pending: bool) {
		self.pending.store(pending, Ordering::Relaxed);
	}

	/// The current impulse response, if one is loaded.
	pub fn load(&self) -> Option<Arc<IrData>> {
		self.current.lock().ok()?.clone()
	}

	/// Stores a new impulse response and bumps the generation.
	pub fn set(&self, ir: Arc<IrData>) {
		if let Ok(mut current) = self.current.lock() {
			*current = Some(ir);
			if let Ok(mut error) = self.error.lock() {
				*error = None;
			}
			self.generation.fetch_add(1, Ordering::Release);
		}
	}

	/// Records why the last load failed, for the picture to show.
	pub fn fail(&self, message: String) {
		if let Ok(mut error) = self.error.lock() {
			*error = Some(message);
		}
		self.generation.fetch_add(1, Ordering::Release);
	}

	/// The last loading error, if the last attempt failed.
	pub fn error(&self) -> Option<String> {
		self.error.lock().ok()?.clone()
	}

	/// The JSON the page polls for the hero.
	pub fn json(&self) -> String {
		let Some(ir) = self.load() else {
			let error = self.error();
			return format!(
				"{{\"name\":null,\"rate\":0,\"samples\":0,\"peaks\":[],\"error\":{},\"pending\":{}}}",
				match error {
					Some(error) => format!("\"{}\"", json_escape(&error)),
					None => "null".to_string(),
				},
				self.is_pending()
			);
		};

		let mut peaks = String::new();
		for (index, (low, high)) in ir.peaks.iter().enumerate() {
			if index > 0 {
				peaks.push(',');
			}
			let _ = std::fmt::Write::write_fmt(&mut peaks, format_args!("[{low:.3},{high:.3}]"));
		}

		format!(
			"{{\"name\":\"{}\",\"rate\":{},\"samples\":{},\"peaks\":[{}],\"error\":null,\"pending\":{}}}",
			json_escape(&ir.name),
			ir.rate,
			ir.samples(),
			peaks,
			self.is_pending(),
		)
	}
}

/// One convolver instance.
#[derive(Parameters)]
pub struct ConvolverEffect {
	/// The level of the dry signal.
	#[range(min = 0.0, max = 1.0)]
	#[default(0.0)]
	#[unit(percent)]
	#[id(display = "Dry")]
	#[layout(group = "Convolver", order = 0, widget = knob)]
	pub dry_gain: f32,

	/// The level of the convolved signal.
	#[range(min = 0.0, max = 1.0)]
	#[default(1.0)]
	#[unit(percent)]
	#[id(display = "Wet")]
	#[layout(group = "Convolver", order = 1, widget = knob, size = large, hero = "impulse")]
	pub wet_gain: f32,

	/// The impulse response, saved with the plug-in state.
	#[serde]
	#[id(display = "Impulse Response")]
	pub ir: Option<Arc<IrData>>,

	#[skip]
	slot: Arc<IrSlot>,
	#[skip]
	convolver: Option<FftConvolver<2>>,
	#[skip]
	sample_rate: usize,
	/// The fingerprint of the IR the convolution was built from.
	#[skip]
	adopted: u64,
	/// The slot generation that was last looked at.
	#[skip]
	seen_slot: u64,
}

impl ConvolverEffect {
	/// A convolver with no impulse response yet, sharing `slot` with its editor.
	pub fn new(sample_rate: usize, slot: Arc<IrSlot>) -> Self {
		Self {
			dry_gain: 0.0,
			wet_gain: 1.0,
			ir: None,
			slot,
			convolver: None,
			sample_rate,
			adopted: 0,
			seen_slot: 0,
		}
	}

	/// Builds the FFT convolver for the current IR.
	fn build(&mut self) {
		let Some(ir) = self.ir.as_ref().filter(|ir| ir.is_loaded()) else {
			self.convolver = None;
			return;
		};

		let left = if ir.left.is_empty() {
			ir.right.clone()
		} else {
			ir.left.clone()
		};
		let right = if ir.right.is_empty() {
			left.clone()
		} else {
			ir.right.clone()
		};

		let mut convolver = FftConvolver::<2>::new([left, right], self.sample_rate);
		// The dry/wet balance is applied around the convolver, so its own gains are neutral.
		convolver.dry_gain = 0.0;
		convolver.wet_gain = 1.0;
		self.convolver = Some(convolver);
	}

	/// Takes a new IR from the slot or from the saved state, whichever changed.
	fn adopt(&mut self) {
		let generation = self.slot.generation();

		if generation != self.seen_slot {
			self.seen_slot = generation;

			if let Some(ir) = self.slot.load() {
				self.adopted = ir.fingerprint;
				self.ir = Some(ir);
				self.build();
			}
		}

		let stored = self.ir.as_ref().map(|ir| ir.fingerprint).unwrap_or(0);

		if stored != self.adopted {
			self.adopted = stored;

			if let Some(ir) = self.ir.clone() {
				// The slot is what a rebuild at another host rate starts from, so a
				// state load has to put the IR there too.
				self.slot.set(ir);
				self.seen_slot = self.slot.generation();
			}

			self.build();
		}
	}
}

impl Effect<2> for ConvolverEffect {
	fn delay(&self) -> usize {
		0
	}

	fn clear_tail(&mut self) {
		if let Some(convolver) = &mut self.convolver {
			convolver.clear();
		}
	}

	fn process(
		&mut self,
		samples: &mut [f32; 2],
		other: &[&[f32; 2]],
		context: &mut Box<dyn ProcessContext>,
	) {
		self.adopt();

		let Some(convolver) = &mut self.convolver else {
			// No impulse response: the dry signal passes, which is what a convolver
			// with nothing to convolve with should sound like.
			return;
		};

		let dry = *samples;
		let dry_gain = self.dry_gain.clamp(0.0, 1.0);
		let wet_gain = self.wet_gain.clamp(0.0, 1.0);

		convolver.process(samples, other, context);

		for (channel, sample) in samples.iter_mut().enumerate() {
			*sample = dry[channel] * dry_gain + *sample * wet_gain;
		}
	}
}

/// A window handle for the window that should own the picker.
#[cfg(windows)]
struct Owner(isize);

#[cfg(windows)]
impl raw_window_handle::HasWindowHandle for Owner {
	fn window_handle(
		&self,
	) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
		let hwnd = std::num::NonZeroIsize::new(self.0)
			.ok_or(raw_window_handle::HandleError::Unavailable)?;
		let handle = raw_window_handle::RawWindowHandle::Win32(
			raw_window_handle::Win32WindowHandle::new(hwnd),
		);

		// SAFETY: the window outlives the dialog, and the handle is only borrowed
		// for the duration of the call.
		Ok(unsafe { raw_window_handle::WindowHandle::borrow_raw(handle) })
	}
}

#[cfg(windows)]
impl raw_window_handle::HasDisplayHandle for Owner {
	fn display_handle(
		&self,
	) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
		// SAFETY: the Windows display handle carries no state.
		Ok(unsafe {
			raw_window_handle::DisplayHandle::borrow_raw(
				raw_window_handle::RawDisplayHandle::Windows(
					raw_window_handle::WindowsDisplayHandle::new(),
				),
			)
		})
	}
}

/// Opens the native picker, owned by the window the user clicked, on a thread of
/// its own.
///
/// A page's own file input has no owner of its own inside a host, so its picker
/// can open behind the DAW, or appear not to open at all. A dialog whose owner is
/// the foreground window always comes up in front of it.
///
/// The dialog also runs on a worker thread, not in the IPC callback: opening a
/// modal window from inside the click that WebView2 is still dispatching made it
/// appear only some of the time.
#[cfg(windows)]
fn pick_ir(slot: Arc<IrSlot>) {
	use windows_sys::Win32::UI::WindowsAndMessaging::{GA_ROOT, GetAncestor, GetForegroundWindow};

	if slot.is_pending() {
		return;
	}

	slot.set_pending(true);

	let mut window = unsafe { GetForegroundWindow() };

	if !window.is_null() {
		let root = unsafe { GetAncestor(window, GA_ROOT) };
		if !root.is_null() {
			window = root;
		}
	}

	let owner = window as isize;

	std::thread::spawn(move || {
		// Let the click that asked for the picker finish before a modal dialog
		// takes over: it is dispatched by WebView2 inside the same input event.
		std::thread::sleep(std::time::Duration::from_millis(50));

		let mut dialog =
			rfd::FileDialog::new().add_filter("Impulse response", &["wav", "flac", "mp3", "ogg"]);

		if owner != 0 {
			let owner = Owner(owner);
			dialog = dialog.set_parent(&owner);
		}

		if let Some(path) = dialog.pick_file() {
			load_ir(&slot, &path);
		}

		slot.set_pending(false);
	});
}

/// Reads and decodes one picked file into the slot.
#[cfg(windows)]
fn load_ir(slot: &IrSlot, path: &std::path::Path) {
	let name = path
		.file_name()
		.map(|name| name.to_string_lossy().into_owned())
		.unwrap_or_else(|| "impulse".to_string());

	match std::fs::read(path) {
		Ok(bytes) => match decode_ir(&bytes, slot.host_rate(), name) {
			Ok(ir) => slot.set(Arc::new(ir)),
			Err(error) => slot.fail(error),
		},
		Err(error) => slot.fail(error.to_string()),
	}
}

#[cfg(not(windows))]
fn pick_ir(_slot: Arc<IrSlot>) {}

/// Decodes an uploaded file into an impulse response at `sample_rate`.
/// Decodes an impulse response file into the samples the convolver runs at.
/// The page and the rack both load IRs through this, so a file means the same.
pub fn decode_ir(bytes: &[u8], sample_rate: usize, name: String) -> Result<IrData, String> {
	if bytes.is_empty() {
		return Err("the file is empty".to_string());
	}

	let mut pcm = load_from_bytes::<2>(bytes.to_vec()).map_err(|error| error.to_string())?;

	if pcm.sample_rate != sample_rate {
		pcm.pcm_data = resample(pcm.pcm_data, pcm.sample_rate, sample_rate)
			.map_err(|error| error.to_string())?;
		pcm.sample_rate = sample_rate;
	}

	let [mut left, mut right] = pcm.pcm_data;

	if left.is_empty() && right.is_empty() {
		return Err("the file has no audio".to_string());
	}

	if right.is_empty() {
		right = left.clone();
	}

	if left.is_empty() {
		left = right.clone();
	}

	let limit = sample_rate.saturating_mul(MAX_SECONDS);
	left.truncate(limit);
	right.truncate(limit);

	let fingerprint = fingerprint(&name, sample_rate, &left, &right);
	let peaks = peaks_of(&left, &right);

	Ok(IrData {
		name,
		rate: sample_rate,
		left,
		right,
		peaks,
		fingerprint,
	})
}

/// A cheap identity for an IR: the name, the rate and the sample bits.
fn fingerprint(name: &str, rate: usize, left: &[f32], right: &[f32]) -> u64 {
	const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
	const PRIME: u64 = 0x0000_0100_0000_01b3;

	let mut hash = OFFSET;
	let mut eat = |value: u64| {
		for byte in value.to_le_bytes() {
			hash ^= byte as u64;
			hash = hash.wrapping_mul(PRIME);
		}
	};

	for byte in name.bytes() {
		eat(byte as u64);
	}
	eat(rate as u64);
	eat(left.len() as u64);
	eat(right.len() as u64);

	for sample in left.iter().chain(right.iter()) {
		eat(sample.to_bits() as u64);
	}

	hash
}

/// The maximum extent of both channels in each picture column, normalized to 1.
fn peaks_of(left: &[f32], right: &[f32]) -> Vec<(f32, f32)> {
	let length = left.len().max(right.len());

	if length == 0 {
		return Vec::new();
	}

	let buckets = PEAKS.min(length);
	let mut peaks = Vec::with_capacity(buckets);
	let mut scale = 0.0_f32;

	for bucket in 0..buckets {
		let start = bucket * length / buckets;
		let end = (((bucket + 1) * length) / buckets)
			.max(start + 1)
			.min(length);
		let mut low = f32::INFINITY;
		let mut high = f32::NEG_INFINITY;

		for index in start..end {
			let sample = (left.get(index).copied().unwrap_or(0.0)
				+ right.get(index).copied().unwrap_or(0.0))
				* 0.5;

			low = low.min(sample);
			high = high.max(sample);
		}

		scale = scale.max(low.abs()).max(high.abs());
		peaks.push((low, high));
	}

	if scale > 0.0 {
		for (low, high) in &mut peaks {
			*low /= scale;
			*high /= scale;
		}
	}

	peaks
}

/// Answers `/ir`: a POST loads the response, a GET describes the current one.
fn ir_route(request: &ProtocolRequest<'_>, slot: &IrSlot) -> Option<ProtocolResponse> {
	if request.path != "/ir" {
		return None;
	}

	match request.method {
		"POST" => {
			let name = request
				.query
				.and_then(|query| query.strip_prefix("name="))
				.map(percent_decode)
				.filter(|name| !name.is_empty())
				.unwrap_or_else(|| "impulse".to_string());

			match decode_ir(request.body, slot.host_rate(), name) {
				Ok(ir) => {
					slot.set(Arc::new(ir));
					Some(ProtocolResponse::json("{\"ok\":true}"))
				}
				Err(error) => {
					slot.fail(error.clone());
					Some(ProtocolResponse::json(format!(
						"{{\"error\":\"{}\"}}",
						json_escape(&error)
					)))
				}
			}
		}
		"GET" => Some(ProtocolResponse::json(slot.json())),
		_ => Some(ProtocolResponse::not_found()),
	}
}

/// Decodes `%20`-style escapes; the page sends names with `encodeURIComponent`.
fn percent_decode(value: &str) -> String {
	let bytes = value.as_bytes();
	let mut out = Vec::with_capacity(bytes.len());
	let mut index = 0;

	while index < bytes.len() {
		if bytes[index] == b'%' && index + 2 < bytes.len() {
			// Decode from the bytes rather than a string slice: an unencoded
			// multibyte character could put a slice boundary inside a character.
			let high = (bytes[index + 1] as char).to_digit(16);
			let low = (bytes[index + 2] as char).to_digit(16);

			if let (Some(high), Some(low)) = (high, low) {
				out.push((high * 16 + low) as u8);
				index += 3;
				continue;
			}
		}

		out.push(bytes[index]);
		index += 1;
	}

	String::from_utf8_lossy(&out).to_string()
}

/// Escapes a string for the page's JSON.
fn json_escape(value: &str) -> String {
	let mut out = String::with_capacity(value.len());

	for character in value.chars() {
		match character {
			'"' => out.push_str("\\\""),
			'\\' => out.push_str("\\\\"),
			'\n' => out.push_str("\\n"),
			'\r' => out.push_str("\\r"),
			'\t' => out.push_str("\\t"),
			character if (character as u32) < 0x20 => {
				let _ = std::fmt::Write::write_fmt(
					&mut out,
					format_args!("\\u{:04x}", character as u32),
				);
			}
			character => out.push(character),
		}
	}

	out
}

/// The product.
pub struct ConvolverWeb {
	processor: EffectProcessor<ConvolverEffect>,
	bypass: Arc<AtomicBool>,
	scope: Arc<Scope>,
	ir: Arc<IrSlot>,
	clear: Arc<AtomicBool>,
}

impl Default for ConvolverWeb {
	fn default() -> Self {
		let ir = Arc::new(IrSlot::new());
		let build_slot = ir.clone();

		Self {
			processor: EffectProcessor::with_build(move |rate| {
				build_slot.set_host_rate(rate);
				ConvolverEffect::new(rate, build_slot.clone())
			}),
			bypass: Arc::new(AtomicBool::new(false)),
			scope: Arc::new(Scope::new()),
			ir,
			clear: Arc::new(AtomicBool::new(false)),
		}
	}
}

impl ConvolverWeb {
	/// The rate the effect inside is currently built for.
	pub fn sample_rate(&self) -> usize {
		self.processor.sample_rate()
	}

	/// The shared IR state, for a test.
	pub fn ir(&self) -> &Arc<IrSlot> {
		&self.ir
	}
}

impl i_am_dsp::prelude::Parameters for ConvolverWeb {
	fn get_parameters(&self) -> Vec<Parameter> {
		self.processor.parameters()
	}

	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		self.processor.set_parameter(identifier, value)
	}
}

impl i_am_dsp::Processor for ConvolverWeb {
	type Message = EffectMessage;

	fn delay(&self) -> usize {
		i_am_dsp::Processor::delay(&self.processor)
	}

	fn process(
		&mut self,
		samples: &mut [f32; 2],
		other: &[[f32; 2]],
		process_context: &mut Box<dyn ProcessContext>,
	) {
		if self.clear.swap(false, Ordering::Relaxed) {
			self.processor.clear_tail();
		}

		for event in process_context.events() {
			match event {
				i_am_dsp::NoteEvent::NoteOn { note, .. } => self.scope.set_note(*note as i32),
				i_am_dsp::NoteEvent::NoteOff { note, .. }
				| i_am_dsp::NoteEvent::Stop { note, .. } => {
					if self.scope.note() == *note as i32 {
						self.scope.set_note(-1);
					}
				}
				i_am_dsp::NoteEvent::ImmediateStop => self.scope.set_note(-1),
				_ => {}
			}
		}

		let input = *samples;
		let input_mono = (input[0] + input[1]) * 0.5;
		let sidechain = other
			.get(1)
			.map(|frame| (frame[0] + frame[1]) * 0.5)
			.unwrap_or(input_mono);

		if self.bypass.load(Ordering::Relaxed) {
			self.scope.push_stereo(input, input, sidechain);
			return;
		}

		i_am_dsp::Processor::process(&mut self.processor, samples, other, process_context);

		let output = *samples;
		self.scope.push_stereo(input, output, sidechain);
	}

	fn on_message(&self, message: Self::Message) {
		i_am_dsp::Processor::on_message(&self.processor, message);
	}
}

impl Plugin for ConvolverWeb {
	const DESCRIPTOR: Descriptor =
		Descriptor::new("iamdsp.effects.convolver.web", "I Am Convolver")
			.with_vendor("iamplugins")
			.with_version(env!("CARGO_PKG_VERSION"));

	const HAS_GUI: bool = true;

	fn new() -> Self {
		Self::default()
	}

	fn window_options() -> WindowOptions {
		const FACE_FRACTION: f64 = 0.85;

		WindowOptions::new()
			.with_size((SIZE.0 as f64 / FACE_FRACTION, SIZE.1 as f64 / FACE_FRACTION))
			.with_title("I Am Convolver")
			.with_resize_hints(GuiResizeHints {
				can_resize_horizontally: true,
				can_resize_vertically: true,
				strategy: AspectRatioStrategy::Disregard,
			})
	}

	fn param_map(&self) -> ParamMap {
		self.processor.param_map()
	}

	fn sync_params(&mut self) {
		self.processor.sync_params();
	}

	fn on_activate(&mut self, sample_rate: usize) {
		self.processor.activate(sample_rate);
	}

	fn editor(&self) -> Option<Box<dyn Editor<Self>>> {
		let face = i_am_web_gui::Faceplate::new("I Am Convolver", ACCENT, SIZE).with_clear();
		let route_slot = self.ir.clone();
		let clear_flag = self.clear.clone();
		let command_slot = self.ir.clone();

		i_am_web_gui::editor_with(
			self,
			face,
			self.bypass.clone(),
			self.scope.clone(),
			self.processor.sample_rate(),
			move |request: &ProtocolRequest<'_>| {
				if request.method == "POST" && request.path == "/clear" {
					clear_flag.store(true, Ordering::Relaxed);
					return Some(ProtocolResponse::json("{\"ok\":true}"));
				}

				ir_route(request, &route_slot)
			},
			move |command: &str| {
				if command != "__load_ir" {
					return false;
				}

				pick_ir(command_slot.clone());
				true
			},
		)
	}
}
#[cfg(test)]
mod tests {
	use super::*;
	use i_am_dsp::prelude::Parameters;
	use i_am_dsp::tools::pcm_data::save_pcm_data;
	use std::sync::atomic::AtomicUsize;

	/// A unique temporary WAV for each test, returned as its encoded bytes.
	fn wav_bytes(channels: &[Vec<f32>], rate: usize) -> Vec<u8> {
		static NEXT: AtomicUsize = AtomicUsize::new(0);
		// The system temp directory is not writable in every build environment, so
		// the workspace's own target directory holds the fixture files.
		let directory =
			std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../target/convolver-tests");
		std::fs::create_dir_all(&directory).expect("a temp directory");
		let path = directory.join(format!("ir-{}.wav", NEXT.fetch_add(1, Ordering::Relaxed)));
		save_pcm_data(&path, channels, rate).expect("a wav file");
		let bytes = std::fs::read(&path).expect("the encoded bytes");
		let _ = std::fs::remove_file(&path);
		bytes
	}

	fn delta_ir(name: &str, fingerprint: u64) -> Arc<IrData> {
		Arc::new(IrData {
			name: name.to_string(),
			rate: 48_000,
			left: vec![1.0],
			right: vec![1.0],
			peaks: vec![(1.0, 1.0)],
			fingerprint,
		})
	}

	#[test]
	fn an_uploaded_wav_becomes_an_impulse_response() {
		let bytes = wav_bytes(&[vec![1.0, 0.0, 0.5], vec![0.5, 0.0, -0.5]], 48_000);
		let ir = decode_ir(&bytes, 48_000, "click.wav".to_string()).expect("a decoded response");

		assert_eq!(ir.rate, 48_000);
		assert_eq!(ir.samples(), 3);
		assert_eq!(ir.left.len(), 3);
		assert_eq!(ir.right.len(), 3);
		assert_eq!(ir.name, "click.wav");
		assert!(!ir.peaks.is_empty());
		assert_eq!(
			ir.fingerprint,
			fingerprint(&ir.name, ir.rate, &ir.left, &ir.right)
		);
	}

	#[test]
	fn a_mono_response_fills_both_channels() {
		let bytes = wav_bytes(&[vec![1.0, -0.5, 0.25]], 48_000);
		let ir = decode_ir(&bytes, 48_000, "mono.wav".to_string()).expect("a decoded response");

		assert_eq!(ir.left, ir.right);
	}

	#[test]
	fn an_impulse_response_is_brought_to_the_host_rate() {
		let mut click = vec![0.0; 4096];
		click[0] = 1.0;
		let bytes = wav_bytes(&[click.clone(), click], 48_000);
		let ir = decode_ir(&bytes, 24_000, "click.wav".to_string()).expect("a resampled response");

		assert_eq!(ir.rate, 24_000);
		assert!(
			(ir.samples() as isize - 2048).abs() < 64,
			"a click of 4096 samples at 48 kHz should be about 2048 at 24 kHz, got {}",
			ir.samples(),
		);
	}

	#[test]
	fn the_ir_route_loads_and_describes_a_file() {
		let slot = IrSlot::new();
		slot.set_host_rate(48_000);
		let bytes = wav_bytes(&[vec![1.0, 0.0, 0.5]], 48_000);

		let post = ProtocolRequest {
			method: "POST",
			path: "/ir",
			query: Some("name=click.wav"),
			body: &bytes,
		};
		let response = ir_route(&post, &slot).expect("the route owns /ir");
		assert_eq!(response.status, 200);
		assert!(slot.load().is_some());

		let get = ProtocolRequest {
			method: "GET",
			path: "/ir",
			query: None,
			body: &[],
		};
		let description = ir_route(&get, &slot).expect("the route owns /ir");
		let body = String::from_utf8(description.body).expect("utf-8 json");
		assert!(body.contains("\"name\":\"click.wav\""), "{body}");
		assert!(body.contains("\"samples\":3"), "{body}");
	}

	#[test]
	fn an_empty_upload_reports_an_error() {
		let slot = IrSlot::new();
		let post = ProtocolRequest {
			method: "POST",
			path: "/ir",
			query: None,
			body: &[],
		};

		ir_route(&post, &slot).expect("the route owns /ir");
		assert!(slot.error().is_some());
		assert!(slot.load().is_none());
	}

	#[test]
	fn the_convolver_passes_a_click_through_a_delta_response() {
		let slot = Arc::new(IrSlot::new());
		slot.set(delta_ir("delta", 7));
		let mut effect = ConvolverEffect::new(48_000, slot);
		let mut context: Box<dyn ProcessContext> = Box::new(());
		let mut frame = [0.5, -0.25];

		effect.process(&mut frame, &[], &mut context);

		assert!((frame[0] - 0.5).abs() < 0.01, "{}", frame[0]);
		assert!((frame[1] + 0.25).abs() < 0.01, "{}", frame[1]);
	}

	#[test]
	fn a_fully_dry_balance_leaves_the_input_alone() {
		let slot = Arc::new(IrSlot::new());
		slot.set(delta_ir("delta", 9));
		let mut effect = ConvolverEffect::new(48_000, slot);
		effect.dry_gain = 1.0;
		effect.wet_gain = 0.0;
		let mut context: Box<dyn ProcessContext> = Box::new(());
		let mut frame = [0.25, 0.75];

		effect.process(&mut frame, &[], &mut context);

		assert_eq!(frame, [0.25, 0.75]);
	}

	#[test]
	fn the_dry_and_wet_gains_mix_the_two_paths() {
		let slot = Arc::new(IrSlot::new());
		slot.set(delta_ir("delta", 10));
		let mut effect = ConvolverEffect::new(48_000, slot);
		effect.dry_gain = 0.25;
		effect.wet_gain = 0.5;
		let mut context: Box<dyn ProcessContext> = Box::new(());
		let mut frame = [1.0, -1.0];

		effect.process(&mut frame, &[], &mut context);

		assert!((frame[0] - 0.75).abs() < 0.01, "{}", frame[0]);
		assert!((frame[1] + 0.75).abs() < 0.01, "{}", frame[1]);
	}

	#[test]
	fn a_saved_response_reaches_a_fresh_instance() {
		let slot = Arc::new(IrSlot::new());
		let mut effect = ConvolverEffect::new(48_000, slot);
		effect.ir = Some(delta_ir("saved", 11));

		let parameter = effect
			.get_parameters()
			.into_iter()
			.find(|parameter| parameter.identifier == "ir")
			.expect("the impulse response is part of the state");
		let SetValue::Serialized(bytes) = parameter.value.to_set_value() else {
			panic!("the response was not serialized");
		};

		let fresh_slot = Arc::new(IrSlot::new());
		let mut fresh = ConvolverEffect::new(48_000, fresh_slot.clone());
		assert!(fresh.set_parameter("ir", SetValue::Serialized(bytes)));
		assert!(fresh.ir.is_some());

		let mut context: Box<dyn ProcessContext> = Box::new(());
		let mut frame = [1.0, 1.0];
		fresh.process(&mut frame, &[], &mut context);

		assert!(fresh.convolver.is_some());
		assert!(
			fresh_slot.load().is_some(),
			"the state load also fills the slot"
		);
	}

	#[test]
	fn a_pending_picker_is_reported() {
		let slot = IrSlot::new();
		slot.set_pending(true);
		assert!(slot.json().contains("\"pending\":true"), "{}", slot.json());
		slot.set_pending(false);
		assert!(slot.json().contains("\"pending\":false"), "{}", slot.json());
	}

	#[test]
	fn the_page_name_is_unescaped() {
		assert_eq!(
			percent_decode("Vintage%20Plate%2B.wav"),
			"Vintage Plate+.wav"
		);
	}
}
