//! A spectrum drawn as partial levels and phases, rendered into band limited
//! tables on a worker thread and morphed at playback.
//!
//! The drawing never reaches the audio thread as a table build: an edit asks the
//! renderer for a new set of tables, the renderer builds them, and the oscillator
//! swaps the finished set in on its next sample. A page holds 512 partials, which
//! is what makes a drawn spectrum sound like a spectrum at the bottom of the
//! keyboard instead of a 64-line approximation; playback is still one lookup per
//! sample because the sum lives in a table.

use std::{
	f32::consts::PI,
	sync::{
		Arc, Condvar, Mutex,
		atomic::{AtomicBool, AtomicU64, Ordering},
	},
	thread::{self, JoinHandle},
};

use crate::prelude::{
	Parameter, ParameterMeta, Parameters, SetValue, Unit, Value, from_binary, to_binary,
};

use super::Wavetable;

/// The partials a drawn page holds.
pub const DRAWN_PARTIALS: usize = 512;

/// The points of the picture the renderer hands the editor for one page.
const CURVE_POINTS: usize = 1024;

/// One page of the editor: a level and a start phase for every partial.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SpectralFrame {
	/// The level of each partial, linear, 0 for silent.
	pub levels: Vec<f32>,
	/// Where each partial starts, in cycles.
	pub phases: Vec<f32>,
}

impl SpectralFrame {
	/// A page from a list of levels and the matching phases.
	pub fn new(levels: Vec<f32>, phases: Vec<f32>) -> Self {
		Self {
			levels: levels.into_iter().take(DRAWN_PARTIALS).collect(),
			phases: phases.into_iter().take(DRAWN_PARTIALS).collect(),
		}
	}

	/// A saw: every partial, at 1 / n.
	pub fn saw() -> Self {
		Self::new(
			(0..DRAWN_PARTIALS)
				.map(|index| 1.0 / (index + 1) as f32)
				.collect(),
			vec![0.0; DRAWN_PARTIALS],
		)
	}

	/// A square: the odd partials only, at 1 / n.
	pub fn square() -> Self {
		Self::new(
			(0..DRAWN_PARTIALS)
				.map(|index| {
					if index % 2 == 0 {
						1.0 / (index + 1) as f32
					} else {
						0.0
					}
				})
				.collect(),
			vec![0.0; DRAWN_PARTIALS],
		)
	}

	/// The level of a partial, or zero when the page is shorter than that.
	pub fn level(&self, index: usize) -> f32 {
		self.levels.get(index).copied().unwrap_or(0.0)
	}

	/// The start phase of a partial, in cycles.
	pub fn phase(&self, index: usize) -> f32 {
		self.phases.get(index).copied().unwrap_or(0.0)
	}

	/// The three arrays the table builder wants, padded to the full partial count.
	fn partials(
		&self,
	) -> (
		[f32; DRAWN_PARTIALS],
		[f32; DRAWN_PARTIALS],
		[f32; DRAWN_PARTIALS],
	) {
		let mut ratios = [0.0; DRAWN_PARTIALS];
		let mut levels = [0.0; DRAWN_PARTIALS];
		let mut phases = [0.0; DRAWN_PARTIALS];

		for index in 0..DRAWN_PARTIALS {
			ratios[index] = (index + 1) as f32;
			levels[index] = self.level(index).clamp(0.0, 1.0);
			phases[index] = self.phase(index).rem_euclid(1.0);
		}

		(ratios, levels, phases)
	}
}

impl Default for SpectralFrame {
	fn default() -> Self {
		Self::saw()
	}
}

/// The pages the editor draws and the audio thread plays.
pub struct SpectralFeed {
	frames: Mutex<Arc<Vec<SpectralFrame>>>,
	generation: AtomicU64,
}

impl SpectralFeed {
	/// A feed with the pages a fresh instance starts on.
	pub fn new(frames: Vec<SpectralFrame>) -> Arc<Self> {
		Arc::new(Self {
			frames: Mutex::new(Arc::new(frames)),
			generation: AtomicU64::new(0),
		})
	}

	/// The pages as they are now.
	pub fn frames(&self) -> Arc<Vec<SpectralFrame>> {
		self.frames
			.lock()
			.unwrap_or_else(|error| error.into_inner())
			.clone()
	}

	/// Replaces the pages.
	pub fn set(&self, frames: Vec<SpectralFrame>) {
		*self
			.frames
			.lock()
			.unwrap_or_else(|error| error.into_inner()) = Arc::new(frames);
		self.generation.fetch_add(1, Ordering::Release);
	}

	/// Moves whenever the pages change.
	pub fn generation(&self) -> u64 {
		self.generation.load(Ordering::Acquire)
	}
}

/// The tables of every page, and the picture the editor draws for each of them.
pub struct PageBank {
	frames: Vec<SpectralFrame>,
	pages: Vec<Arc<Wavetable>>,
	curves: Vec<Vec<f32>>,
}

impl PageBank {
	/// How many pages were built.
	pub fn pages(&self) -> usize {
		self.pages.len()
	}

	/// One period of a page, for the editor to draw.
	pub fn curve(&self, page: usize) -> Option<&[f32]> {
		self.curves.get(page).map(Vec::as_slice)
	}

	/// Reads one page at a phase of its fundamental.
	fn sample(&self, page: usize, phase: f32, max_harmonic: f32) -> (f32, f32) {
		self.pages[page].play(phase, max_harmonic)
	}
}

/// Builds the pages' band limited tables and pictures, reusing the pages that
/// did not change since the previous bank.
///
/// There is no page limit, so editing one page of a large set must not rebuild
/// the others: a page that is bit-for-bit what it was keeps its tables.
fn build_bank(frames: &[SpectralFrame], previous: Option<&PageBank>) -> PageBank {
	let mut pages: Vec<Arc<Wavetable>> = Vec::with_capacity(frames.len());
	let mut curves: Vec<Vec<f32>> = Vec::with_capacity(frames.len());

	for (index, frame) in frames.iter().enumerate() {
		if let Some(previous) = previous
			&& previous.frames.get(index) == Some(frame)
			&& let (Some(table), Some(curve)) =
				(previous.pages.get(index), previous.curves.get(index))
		{
			pages.push(table.clone());
			curves.push(curve.clone());
			continue;
		}

		let (ratios, levels, phases) = frame.partials();
		let table = Wavetable::build::<DRAWN_PARTIALS>(&ratios, &levels, &phases, DRAWN_PARTIALS)
			.unwrap_or_else(|| {
				// Not reachable with integer ratios, but a silent page is better
				// than a panic on the render thread if it ever is.
				let silence = [0.0; DRAWN_PARTIALS];
				Wavetable::build::<DRAWN_PARTIALS>(&ratios, &silence, &silence, DRAWN_PARTIALS)
					.expect("a silent page always builds")
			});

		let curve = (0..CURVE_POINTS)
			.map(|point| {
				let phase = point as f32 / CURVE_POINTS as f32;
				table.play(phase, f32::INFINITY).0
			})
			.collect();

		pages.push(Arc::new(table));
		curves.push(curve);
	}

	PageBank {
		frames: frames.to_vec(),
		pages,
		curves,
	}
}

struct RendererShared {
	pending: Mutex<Option<Vec<SpectralFrame>>>,
	ready: Mutex<Option<Arc<PageBank>>>,
	ready_generation: AtomicU64,
	wake: Condvar,
	alive: AtomicBool,
}

/// Renders the drawn pages into tables away from the audio thread.
pub struct SpectrumRenderer {
	shared: Arc<RendererShared>,
	thread: Mutex<Option<JoinHandle<()>>>,
}

impl SpectrumRenderer {
	/// A renderer over the pages a fresh instance starts on.
	///
	/// The first set is built here, on the thread that constructs the plug-in,
	/// because a voice that starts silent would be a bug the user can hear.
	pub fn new(frames: Vec<SpectralFrame>) -> Arc<Self> {
		let shared = Arc::new(RendererShared {
			pending: Mutex::new(None),
			ready: Mutex::new(Some(Arc::new(build_bank(&frames, None)))),
			ready_generation: AtomicU64::new(1),
			wake: Condvar::new(),
			alive: AtomicBool::new(true),
		});

		let worker = shared.clone();
		let thread = thread::Builder::new()
			.name("spectrum-renderer".to_string())
			.spawn(move || render(worker))
			.ok();

		Arc::new(Self {
			shared,
			thread: Mutex::new(thread),
		})
	}

	/// Asks for a new set of tables. A request that is already waiting is
	/// replaced, so a fast drag costs one build instead of one per pointer move.
	pub fn request(&self, frames: Vec<SpectralFrame>) {
		if let Ok(mut pending) = self.shared.pending.lock() {
			*pending = Some(frames);
		}

		self.shared.wake.notify_one();
	}

	/// How many banks have been published.
	pub fn generation(&self) -> u64 {
		self.shared.ready_generation.load(Ordering::Acquire)
	}

	/// The latest bank, if the audio thread has not taken it yet.
	pub fn current(&self) -> Option<(u64, Arc<PageBank>)> {
		let generation = self.generation();
		let ready = self.shared.ready.lock().ok()?;

		ready.clone().map(|bank| (generation, bank))
	}

	/// The latest bank, for a caller that must not wait.
	pub fn try_take(&self) -> Option<(u64, Arc<PageBank>)> {
		let generation = self.generation();
		let ready = self.shared.ready.try_lock().ok()?;

		ready.clone().map(|bank| (generation, bank))
	}
}

impl Drop for SpectrumRenderer {
	fn drop(&mut self) {
		self.shared.alive.store(false, Ordering::Release);
		self.shared.wake.notify_all();

		if let Ok(mut thread) = self.thread.lock()
			&& let Some(thread) = thread.take()
		{
			let _ = thread.join();
		}
	}
}

fn render(shared: Arc<RendererShared>) {
	let mut last: Option<Arc<PageBank>> = None;

	loop {
		let frames = {
			let mut pending = match shared.pending.lock() {
				Ok(pending) => pending,
				Err(_) => return,
			};

			while pending.is_none() && shared.alive.load(Ordering::Acquire) {
				pending = match shared.wake.wait(pending) {
					Ok(pending) => pending,
					Err(_) => return,
				};
			}

			if !shared.alive.load(Ordering::Acquire) {
				return;
			}

			let Some(frames) = pending.take() else {
				continue;
			};

			frames
		};

		let bank = Arc::new(build_bank(&frames, last.as_deref()));
		last = Some(bank.clone());

		if let Ok(mut ready) = shared.ready.lock() {
			*ready = Some(bank);
			shared.ready_generation.fetch_add(1, Ordering::Release);
		}
	}
}

/// The parameters of a drawn spectrum: the pages in a feed, and the morph
/// between them.
pub struct DrawnSpectrum {
	feed: Arc<SpectralFeed>,
	/// 0 is the first page, 1 the last.
	pub morph: f32,
}

impl DrawnSpectrum {
	/// A parameter block over the pages of a feed.
	pub fn new(feed: Arc<SpectralFeed>) -> Self {
		Self { feed, morph: 0.0 }
	}

	/// The pages, for an editor that draws them.
	pub fn feed(&self) -> &Arc<SpectralFeed> {
		&self.feed
	}
}

fn morph_parameter(value: f32) -> Parameter {
	let meta = ParameterMeta {
		default: Some(0.0),
		unit: Unit::None,
		display_name: Some("Morph".into()),
		choices: None,
		#[cfg(feature = "layout")]
		layout: crate::prelude::ParamLayout::new()
			.group("Shape")
			.order(0)
			.widget(crate::prelude::ParamWidget::Knob)
			.hero("spectrum"),
	};

	Parameter::new(
		"morph",
		Value::Float {
			value,
			range: 0.0..=1.0,
			logarithmic: false,
		},
	)
	.with_meta(meta)
}

impl Parameters for DrawnSpectrum {
	fn get_parameters(&self) -> Vec<Parameter> {
		let frames = self.feed.frames();
		let serialized = to_binary(&*frames).expect("a drawn spectrum serializes");

		vec![
			morph_parameter(self.morph),
			Parameter::new("frames", Value::Serialized(serialized)),
		]
	}

	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		match identifier {
			"morph" => {
				let SetValue::Float(value) = value else {
					return false;
				};

				self.morph = value.clamp(0.0, 1.0);
				true
			}
			"frames" => {
				let SetValue::Serialized(data) = value else {
					return false;
				};

				match from_binary::<Vec<SpectralFrame>>(data) {
					Ok(frames) => {
						self.feed.set(frames);
						true
					}
					Err(_) => false,
				}
			}
			_ => false,
		}
	}
}

/// The oscillator: the pages' finished tables, crossfaded by Morph.
pub struct TableBankOsc {
	params: DrawnSpectrum,
	renderer: Arc<SpectrumRenderer>,
	bank: Option<Arc<PageBank>>,
	seen: u64,
	feed_generation: u64,
	max_harmonic: f32,
}

impl TableBankOsc {
	/// A voice over the pages of a feed.
	pub fn new(feed: Arc<SpectralFeed>) -> Self {
		let params = DrawnSpectrum::new(feed);
		let feed_generation = params.feed().generation();
		let renderer = SpectrumRenderer::new((*params.feed().frames()).clone());
		let (seen, bank) = match renderer.current() {
			Some((generation, bank)) => (generation, Some(bank)),
			None => (0, None),
		};

		Self {
			params,
			renderer,
			bank,
			seen,
			feed_generation,
			max_harmonic: f32::INFINITY,
		}
	}

	/// The renderer, for an editor that wants the pages' pictures.
	pub fn renderer(&self) -> &Arc<SpectrumRenderer> {
		&self.renderer
	}

	/// The pages, in parameter form.
	pub fn params(&self) -> &DrawnSpectrum {
		&self.params
	}

	/// The tables in play, for a test.
	pub fn bank(&self) -> Option<&Arc<PageBank>> {
		self.bank.as_ref()
	}
}

impl Parameters for TableBankOsc {
	fn get_parameters(&self) -> Vec<Parameter> {
		self.params.get_parameters()
	}

	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		let changed = self.params.set_parameter(identifier, value);

		if changed && identifier == "frames" {
			// The voice would ask for the same rebuild on its next sample; take the
			// generation now so a load costs one render, not two.
			self.feed_generation = self.params.feed().generation();
			self.renderer
				.request((*self.params.feed().frames()).clone());
		}

		changed
	}

	fn stable_id_overrides(&self) -> Vec<(String, u32)> {
		self.params.stable_id_overrides()
	}
}

impl<const CHANNELS: usize> crate::prelude::Oscillator<CHANNELS> for TableBankOsc {
	fn play_at(&self, frequency: f32, time: f32, phase: [f32; CHANNELS]) -> [f32; CHANNELS] {
		let Some(bank) = self.bank.as_ref() else {
			return [0.0; CHANNELS];
		};

		if bank.pages.is_empty() {
			return [0.0; CHANNELS];
		}

		// Morph walks the pages: the whole number picks one, the fraction fades
		// towards the next, and a single page ignores it.
		let position = if bank.pages.len() == 1 {
			0.0
		} else {
			self.params.morph.clamp(0.0, 1.0) * (bank.pages.len() - 1) as f32
		};
		let from = position.floor() as usize;
		let to = (from + 1).min(bank.pages.len() - 1);
		let blend = position - from as f32;
		let (sine, cosine) = bank.sample(from, time * frequency, self.max_harmonic);
		let (sine, cosine) = if to == from {
			(sine, cosine)
		} else {
			let (next_sine, next_cosine) = bank.sample(to, time * frequency, self.max_harmonic);
			(
				sine + (next_sine - sine) * blend,
				cosine + (next_cosine - cosine) * blend,
			)
		};

		let mut output = [0.0; CHANNELS];

		for (output, phase) in output.iter_mut().zip(phase) {
			let angle = 2.0 * PI * phase;
			let (sin, cos) = angle.sin_cos();
			*output = sine * cos + cosine * sin;
		}

		output
	}

	fn set_pitch(&mut self, frequency: f32, sample_rate: usize) {
		// Everything above this ratio folds back into the audible band at this pitch.
		self.max_harmonic = if frequency > 0.0 && sample_rate > 0 {
			sample_rate as f32 / (2.0 * frequency)
		} else {
			f32::INFINITY
		};
	}

	fn sync(&mut self) {
		// An editor can replace the pages without touching the parameters; the
		// feed's own generation is what tells the voice to ask for a render.
		let feed_generation = self.params.feed().generation();

		if feed_generation != self.feed_generation {
			self.feed_generation = feed_generation;
			self.renderer
				.request((*self.params.feed().frames()).clone());
		}

		let generation = self.renderer.generation();

		if generation != self.seen
			&& let Some((generation, bank)) = self.renderer.try_take()
		{
			self.bank = Some(bank);
			self.seen = generation;
		}
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, _: &mut egui::Ui, _: String) {}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::prelude::Oscillator;

	fn page(fundamental: f32, second: f32) -> SpectralFrame {
		let mut levels = vec![0.0; DRAWN_PARTIALS];
		levels[0] = fundamental;
		levels[1] = second;

		SpectralFrame::new(levels, vec![0.0; DRAWN_PARTIALS])
	}

	fn oscillator(frames: Vec<SpectralFrame>) -> TableBankOsc {
		TableBankOsc::new(SpectralFeed::new(frames))
	}

	/// A page becomes a band limited table and a picture of one period.
	#[test]
	fn a_page_builds_a_table_and_its_picture() {
		let bank = build_bank(&[page(1.0, 0.0), page(0.0, 1.0)], None);

		assert_eq!(bank.pages(), 2);
		assert_eq!(bank.curve(0).map(<[f32]>::len), Some(CURVE_POINTS));

		let curve = bank.curve(0).expect("the first page");

		// A single partial at phase zero is a sine: a quarter of the way through
		// the period it is at its peak.
		assert!(curve[CURVE_POINTS / 4] > 0.9, "{}", curve[CURVE_POINTS / 4]);
		assert!((curve[0]).abs() < 1e-3);
	}

	/// Morph crossfades the tables of two pages at playback.
	#[test]
	fn the_morph_crossfades_the_pages() {
		let mut voice = oscillator(vec![page(1.0, 0.0), page(0.0, 1.0)]);
		Oscillator::<2>::set_pitch(&mut voice, 110.0, 48_000);

		// At a quarter of the fundamental, the first page's partial is at its
		// peak and the second page's partial sits on a zero crossing.
		let quarter = 0.25;

		voice.params.morph = 0.0;
		let first = voice.play_at(1.0, quarter, [0.0; 2])[0];
		voice.params.morph = 1.0;
		let second = voice.play_at(1.0, quarter, [0.0; 2])[0];
		voice.params.morph = 0.5;
		let mixed = voice.play_at(1.0, quarter, [0.0; 2])[0];

		assert!((first - 1.0).abs() < 1e-2, "{first}");
		assert!(second.abs() < 1e-2, "{second}");
		assert!((mixed - 0.5).abs() < 1e-2, "{mixed}");
	}

	/// An edit asks the renderer for new tables, and the oscillator picks them up
	/// on a later sample without ever building one itself.
	#[test]
	fn an_edit_reaches_the_audio_thread() {
		let mut voice = oscillator(vec![page(1.0, 0.0)]);
		assert_eq!(voice.bank().map(|bank| bank.pages()), Some(1));

		let saved = voice.get_parameters();
		let frames = saved
			.iter()
			.find(|parameter| parameter.identifier == "frames")
			.expect("the frames parameter")
			.value
			.clone();

		assert_eq!(voice.bank().map(|bank| bank.pages()), Some(1));

		// The same edit a load or the page makes: replace the pages.
		let mut frames_value = match frames {
			Value::Serialized(data) => from_binary::<Vec<SpectralFrame>>(data).expect("frames"),
			other => panic!("frames is not serialized: {other:?}"),
		};
		frames_value.push(page(0.0, 1.0));
		voice.set_parameter(
			"frames",
			SetValue::Serialized(to_binary(&frames_value).expect("frames serialize")),
		);

		let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);

		while std::time::Instant::now() < deadline {
			Oscillator::<2>::sync(&mut voice);

			if voice.bank().map(|bank| bank.pages()) == Some(2) {
				return;
			}

			thread::sleep(std::time::Duration::from_millis(2));
		}

		panic!("the renderer never published the new pages");
	}

	/// A project save reads the pages from the parameters, and a load writes them
	/// back into the feed: the editor shows what was saved.
	#[test]
	fn the_pages_survive_a_state_round_trip() {
		let source = oscillator(vec![SpectralFrame::saw(), SpectralFrame::square()]);
		let saved = source.get_parameters();
		let frames = saved
			.iter()
			.find(|parameter| parameter.identifier == "frames")
			.expect("the frames parameter");
		let data = match &frames.value {
			Value::Serialized(data) => data.clone(),
			other => panic!("frames is not serialized: {other:?}"),
		};

		let mut restored = oscillator(vec![]);
		assert!(restored.set_parameter("frames", SetValue::Serialized(data)));

		assert_eq!(
			*restored.params().feed().frames(),
			*source.params().feed().frames()
		);
	}

	/// An edit to one page leaves the other pages' tables alone: with no page
	/// limit, a big set must not pay for a rebuild it did not need.
	#[test]
	fn an_edit_only_rebuilds_the_page_that_changed() {
		let first = build_bank(&[page(1.0, 0.0), page(0.0, 1.0)], None);
		let second = build_bank(&[page(1.0, 0.0), page(0.0, 0.5)], Some(&first));

		assert!(Arc::ptr_eq(&first.pages[0], &second.pages[0]));
		assert!(!Arc::ptr_eq(&first.pages[1], &second.pages[1]));
	}

	/// Morph is a knob like any other; the pages are state the host saves but
	/// never draws as a control.
	#[test]
	fn the_morph_is_a_knob_and_the_pages_are_state() {
		let mut voice = oscillator(vec![SpectralFrame::saw()]);

		assert!(voice.set_parameter("morph", SetValue::Float(2.0)));
		assert_eq!(voice.params().morph, 1.0);
		assert!(!voice.set_parameter("missing", SetValue::Float(0.0)));

		let parameters = voice.get_parameters();
		let morph = parameters
			.iter()
			.find(|parameter| parameter.identifier == "morph")
			.expect("morph");

		assert!(matches!(morph.value, Value::Float { .. }));
		assert!(
			parameters.iter().any(
				|parameter| parameter.identifier == "frames" && parameter.value.is_serialized()
			)
		);
	}
}
