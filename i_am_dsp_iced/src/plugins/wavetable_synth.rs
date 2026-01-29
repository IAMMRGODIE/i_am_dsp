use std::{ops::RangeInclusive, sync::{Arc, atomic::Ordering}, time::Instant};

use crossbeam_channel::{Receiver, Sender, TryRecvError};
use i_am_dsp::{Generator, NoteEvent, prelude::{Adsr, EqualTemperament, PmTable, TableOsc, WaveTable, WaveTableSmoother}};
use iced::{Border, Element, Length, Theme, alignment::{Horizontal, Vertical}, widget::{button, canvas, column, container, row, text}};
use portable_atomic::{AtomicF32, AtomicUsize};

use crate::{Message, Processor, SyncedView, styles::{BORDER_WIDTH, ERROR_COLOR, PADDING, PRIMARY_COLOR}, tools::{adsr_editor::AdsrEditor, knob::knob, selector::selector, slider::slider, unison::UnisonEditor, waveform::{Waveform, WaveformBuf}}};

#[derive(Clone)]
pub enum WavetableSynthMessage {
	MidiEvent(NoteEvent),
	Tick(Instant),
	Empty,
	ClearError,
}

/// A function that returns a list of wave tables to use for the synth.
pub type TableBuildFn = Box<dyn Fn(usize) -> Vec<Box<dyn WaveTable + Send + Sync>> + Send + Sync>;

/// A shortcut for the PmTable type.
pub type SynthTable = PmTable<WaveTableSmoother, WaveTableSmoother>;

pub struct WavetableSynth {
	sample_rate: usize,
	table: Adsr<TableOsc<SynthTable>, EqualTemperament, 2>,
	adsr_params: Arc<AdsrEditor>,
	smooth_factor_carrier: Arc<AtomicF32>,
	smooth_factor_modulator: Arc<AtomicF32>,
	pm_factor: Arc<AtomicF32>,
	unison_editor: Arc<UnisonEditor>,
	senders: Vec<Sender<f32>>,
	table_builder: TableBuildFn,
	pitch_factor: Arc<AtomicF32>,
	gain: Arc<AtomicF32>,
}

impl WavetableSynth {
	pub fn new(sample_rate: usize, table_builder: impl Fn(usize) -> Vec<Box<dyn WaveTable + Send + Sync>> + Send + Sync +'static) -> Self {
		let carrier = WaveTableSmoother::new(table_builder(sample_rate), 0.0);
		let modulator = WaveTableSmoother::new(table_builder(sample_rate), 0.0);
		let table = TableOsc(PmTable::new(carrier, modulator));

		let table = Adsr::new(
			table, 
			EqualTemperament, 
			sample_rate
		);

		let adsr_params = Arc::new(AdsrEditor::new(&table));
		let unison_editor = Arc::new(UnisonEditor::new(&table));

		Self {
			pitch_factor: Arc::new(AtomicF32::new(table.pitch_factor)),
			gain: Arc::new(AtomicF32::new(table.gain)),
			sample_rate,
			table,
			adsr_params,
			smooth_factor_carrier: Arc::default(),
			smooth_factor_modulator: Arc::default(),
			pm_factor: Arc::default(),
			senders: vec![],
			table_builder: Box::new(table_builder),
			unison_editor,
		}
	}
}

pub struct WavetableSynthView{
	wavetable: Waveform<SynthTable>,
	waveform: WaveformBuf,

	carrier: Waveform<WaveTableSmoother>,
	modulator: Waveform<WaveTableSmoother>,

	error_msg: Option<String>,
	reciver: Receiver<f32>,
	adsr_editor: Arc<AdsrEditor>,
	unison_editor: Arc<UnisonEditor>,
	pitch_factor: Arc<AtomicF32>,
	gain: Arc<AtomicF32>,

	smooth_factor_carrier: Arc<AtomicF32>,
	smooth_factor_modulator: Arc<AtomicF32>,
	pm_factor: Arc<AtomicF32>,
	current_pos: Arc<AtomicUsize>,
	tables: Vec<Waveform<Box<dyn WaveTable + 'static>>>,
}

impl SyncedView for WavetableSynthView {
	type Message = WavetableSynthMessage;

	fn update(&mut self, message: &Self::Message) {
		loop {
			match self.reciver.try_recv() {
				Ok(value) => {
					self.waveform.update(&[value]);
				},
				Err(TryRecvError::Empty) => {
					break;
				}
				Err(TryRecvError::Disconnected) => {

				}
			}
		}

		match message {
			WavetableSynthMessage::MidiEvent(_) => {},
			WavetableSynthMessage::Tick(_) => {},
			WavetableSynthMessage::Empty => {},
			WavetableSynthMessage::ClearError => {
				self.error_msg = None;
			}
		}

		let smooth_factor_carrier = self.smooth_factor_carrier.load(Ordering::Relaxed);
		let smooth_factor_modulator = self.smooth_factor_modulator.load(Ordering::Relaxed);
		let pm_factor = self.pm_factor.load(Ordering::Relaxed);

		if smooth_factor_carrier != self.wavetable.table.carrier.smooth_factor {
			self.wavetable.table.carrier.smooth_factor = smooth_factor_carrier;
			self.carrier.table.smooth_factor = smooth_factor_carrier;
			self.wavetable.toggle_update();
			self.carrier.toggle_update();
		}

		if smooth_factor_modulator != self.wavetable.table.modulator.smooth_factor {
			self.wavetable.table.modulator.smooth_factor = smooth_factor_modulator;
			self.modulator.table.smooth_factor = smooth_factor_modulator;
			self.wavetable.toggle_update();
			self.modulator.toggle_update();
		}

		if pm_factor != self.wavetable.table.pm_factor {
			self.wavetable.table.pm_factor = pm_factor;
			self.wavetable.toggle_update();
		}
	}

	fn view(&self) -> Element<'_, Self::Message> {
		let current_pos = self.current_pos.load(Ordering::Relaxed);
		let smooth_factor_now = match current_pos {
			0 => &self.smooth_factor_carrier,
			1 => &self.smooth_factor_modulator,
			_ => &self.pm_factor,
		};

		let smooth_factor = smooth_factor_now.load(Ordering::Relaxed);
		let tables_len = self.tables.len();
		let tables = self.tables.iter().enumerate().map(|(i, table)| {
			let selected = if i + 1 == tables_len {
				(i as f32 / tables_len as f32..=(i + 1) as f32 / tables_len as f32).contains(&smooth_factor)
			}else {
				(i as f32 / tables_len as f32..(i + 1) as f32 / tables_len as f32).contains(&smooth_factor)
			};
			table.set_selected(selected);
			Element::from(canvas(table).width(32.0).height(32.0))
		}).collect::<Vec<_>>();


		let canvas_show = match current_pos {
			0 => Element::from(canvas(&self.carrier).width(Length::FillPortion(5)).height(Length::FillPortion(5))),
			1 => Element::from(canvas(&self.modulator).width(Length::FillPortion(5)).height(Length::FillPortion(5))),
			_ => Element::from(canvas(&self.wavetable).width(Length::FillPortion(5)).height(Length::FillPortion(5))),
		};

		let factor = match current_pos {
			0 => &self.smooth_factor_carrier,
			1 => &self.smooth_factor_modulator,
			_ => &self.pm_factor,
		};

		#[inline(always)]
		fn knob_wrapped(range: RangeInclusive<f32>, param: &AtomicF32) -> crate::tools::knob::Knob<'_, WavetableSynthMessage> {
			knob(range, param.load(Ordering::Relaxed), |value| {
				param.store(value, Ordering::Relaxed);
				WavetableSynthMessage::Empty
			}).width(32.0).height(32.0)
		}

		#[inline(always)]
		fn theme_func(theme: &Theme) -> iced::widget::container::Style {
			iced::widget::container::Style::default()
				.border(Border::default()
				.color(theme.extended_palette().background.strongest.color)
				.width(BORDER_WIDTH)
				.rounded(PADDING)
			)
			.background(theme.extended_palette().background.weakest.color)
		}

		column![
			self.error_msg.as_ref().map(|msg| text(msg).color(ERROR_COLOR)),
			self.error_msg.as_ref().map(|_| button("Clear Error").on_press(WavetableSynthMessage::ClearError)),
			row![
				canvas(self.unison_editor.as_ref()).height(36.0).width(Length::FillPortion(1)),
				selector(vec!["Carrier", "Module", "Output"], current_pos).on_change(|_, value| {
					self.current_pos.store(value, Ordering::Relaxed);
					WavetableSynthMessage::Empty
				}).height(36.0).width(Length::FillPortion(1)),
			].spacing(16.0).width(Length::Fill),
			row![
				container(
					column(tables).spacing(16.0).width(64.0).align_x(Horizontal::Center)
				).style(theme_func).height(Length::Fill).align_y(Vertical::Center),
				container(column![
					row![
						knob_wrapped(0.0..=2.0, &self.unison_editor.unison_detune),
						knob_wrapped(-10.0..=10.0, &self.unison_editor.unison_bend),
						knob_wrapped(0.0..=1.0, &self.unison_editor.unison_blend),
						knob_wrapped(0.0..=1.0, &self.unison_editor.random_phase),
						knob_wrapped(0.0..=1.0, &self.unison_editor.random_pan),
					].spacing(16.0).align_y(Vertical::Center),
					slider(1.0..=32.0, self.unison_editor.unisons.load(Ordering::Relaxed) as f32, |value| {
						self.unison_editor.unisons.store(value as usize, Ordering::Relaxed);
						WavetableSynthMessage::Empty
					}).step(1.0).text("Unison").formatter(|float| format!("{:.0}", float)),
					slider(0.0..=1.0, factor.load(Ordering::Relaxed), |value| {
						factor.store(value, Ordering::Relaxed);
						WavetableSynthMessage::Empty
					}).text("Factor"),
					slider(0.25..=4.0, self.pitch_factor.load(Ordering::Relaxed), |value| {
						self.pitch_factor.store(value, Ordering::Relaxed);
						WavetableSynthMessage::Empty
					}).text("Pitch").on_release(|_| {
						let value = self.pitch_factor.load(Ordering::Relaxed);
						let note = value.ln() / 2.0_f32.powf(1.0 / 12.0).ln();
						let note = note.round();
						let value = 2.0_f32.powf(1.0 / 12.0).powf(note);
						self.pitch_factor.store(value, Ordering::Relaxed);
						WavetableSynthMessage::Empty
					}).formatter(|float| format!("{:.0}semi", float.ln() / 2.0_f32.powf(1.0 / 12.0).ln()))
						.logarithmic().speed(0.3),
				].spacing(16.0).padding(16.0)).style(theme_func).width(Length::FillPortion(4)).height(Length::Fill).align_y(Vertical::Center),
				column![
					canvas_show,
					canvas(&self.waveform).width(Length::Fill).height(48.0),
				].spacing(16.0).height(Length::Fill),
			].spacing(16.0).width(Length::Fill).height(Length::FillPortion(1)),
			row![
				container(column![
					knob_wrapped(-10.0..=10.0, &self.adsr_editor.attack_bend),
					knob_wrapped(-10.0..=10.0, &self.adsr_editor.decay_bend),
					knob_wrapped(-10.0..=10.0, &self.adsr_editor.release_bend),
				].spacing(16.0).width(64.0).align_x(Horizontal::Center)).style(theme_func).height(Length::Fill).align_y(Vertical::Center),
				container(column![
					canvas(self.adsr_editor.as_ref()).width(Length::Fill).height(Length::FillPortion(4)),
					container(row![
						knob_wrapped(0.0001..=10000.0, &self.adsr_editor.delay_time).logarithmic(true).speed(0.01),
						knob_wrapped(0.0001..=10000.0, &self.adsr_editor.attack_time).logarithmic(true).speed(0.01),
						knob_wrapped(0.0001..=10000.0, &self.adsr_editor.hold_time).logarithmic(true).speed(0.01),
						knob_wrapped(0.0001..=10000.0, &self.adsr_editor.decay_time).logarithmic(true).speed(0.01),
						knob_wrapped(0.0001..=1.0, &self.adsr_editor.sustain_level).logarithmic(true).speed(0.01),
						knob_wrapped(0.0001..=10000.0, &self.adsr_editor.release_time).logarithmic(true).speed(0.01),
						slider(0.01..=4.0, self.gain.load(Ordering::Relaxed), |value| {
							self.gain.store(value, Ordering::Relaxed);
							WavetableSynthMessage::Empty
						}).text("Gain").formatter(|float| format!("{:.2}dB", float.log10() * 20.0)),
					]
						.spacing(16.0)
						.align_y(Vertical::Center)
						.height(Length::Fill)
					).style(theme_func).height(64.0).padding(16.0).width(Length::Fill),
				].spacing(16.0).align_x(Horizontal::Center)).width(Length::FillPortion(9))
			].align_y(Vertical::Center).spacing(16.0).height(Length::FillPortion(1)),
		].align_x(Horizontal::Center).padding(16.0).spacing(16.0).into()
	}
}

impl Message for WavetableSynthMessage {
	fn from_note_event(event: NoteEvent) -> Self {
		Self::MidiEvent(event)
	}

	fn note_event(&self) -> Option<NoteEvent> {
		if let Self::MidiEvent(event) = self {
			Some(event.clone())
		}else {
			None
		}
	}

	fn tick(instant: Instant) -> Self {
		Self::Tick(instant)
	}
}

impl Processor for WavetableSynth {
	type Message = WavetableSynthMessage;
	type SyncedView = WavetableSynthView;

	fn process(&mut self, samples: &mut [f32; 2], _: &[&[f32; 2]], process_context: &mut Box<dyn i_am_dsp::ProcessContext>) {
		*samples = self.table.generate(process_context);
		let to_send = (samples[0] + samples[1]) / 2.0;

		self.senders.retain(|inner| {
			inner.try_send(to_send).is_ok()
		});
	}

	fn delay(&self) -> usize {
		0
	}

	fn on_message(&mut self, _message: Self::Message) {
		self.adsr_params.adjust(&mut self.table);
		self.unison_editor.adjust(&mut self.table);
		self.table.oscillator.0.carrier.smooth_factor = self.smooth_factor_carrier.load(Ordering::Relaxed);
		self.table.oscillator.0.modulator.smooth_factor = self.smooth_factor_modulator.load(Ordering::Relaxed);
		self.table.oscillator.0.pm_factor = self.pm_factor.load(Ordering::Relaxed);
		self.table.pitch_factor = self.pitch_factor.load(Ordering::Relaxed);
		self.table.gain = self.gain.load(Ordering::Relaxed);
	}

	fn synced_view(&mut self) -> Self::SyncedView {
		let (sender, reciver) = crossbeam_channel::unbounded();
		self.senders.push(sender);

		let smooth_factor_carrier = self.smooth_factor_carrier.load(Ordering::Relaxed);
		let smooth_factor_modulator = self.smooth_factor_modulator.load(Ordering::Relaxed);
		let pm_factor = self.pm_factor.load(Ordering::Relaxed);

		let carrier = WaveTableSmoother::new((self.table_builder)(self.sample_rate), smooth_factor_carrier);
		let modulator = WaveTableSmoother::new((self.table_builder)(self.sample_rate), smooth_factor_modulator);
		let mut table = PmTable::new(carrier, modulator);
		let carrier = WaveTableSmoother::new((self.table_builder)(self.sample_rate), smooth_factor_carrier);
		let modulator = WaveTableSmoother::new((self.table_builder)(self.sample_rate), smooth_factor_modulator);
		let current_postion = Arc::new(AtomicUsize::new(0));

		table.pm_factor = pm_factor;

		let tables = (self.table_builder)(self.sample_rate);
		let tables_len = tables.len();

		let tables = tables.into_iter().enumerate().map(|(i, table)| {
			let current_postion = current_postion.clone();
			let smooth_factor_carrier = self.smooth_factor_carrier.clone();
			let smooth_factor_modulator = self.smooth_factor_modulator.clone();
			let pm_factor = self.pm_factor.clone();
			Waveform::new(table as Box<dyn WaveTable + 'static>, 256, false)
				.on_change(move |_| {
					let to_set = match current_postion.load(Ordering::Relaxed) {
						0 => &smooth_factor_carrier,
						1 => &smooth_factor_modulator,
						_ => &pm_factor,
					};

					if tables_len <= 1 {
						to_set.store(0.5, Ordering::Relaxed);
					}else {
						to_set.store(i as f32 / (tables_len - 1) as f32, Ordering::Relaxed);
					}
					false
				})
		}).collect();

		WavetableSynthView {
			waveform: WaveformBuf::new(self.sample_rate * 2, 300.0),
			wavetable: Waveform::new(table, 256, false)
				.on_change(|_| false)
				.disable_hover()
				.color(PRIMARY_COLOR),
			carrier: Waveform::new(carrier, 256, false)
				.on_change(|_| false)
				.disable_hover()
				.color(PRIMARY_COLOR),
			modulator: Waveform::new(modulator, 256, false)
				.on_change(|_| false)
				.disable_hover()
				.color(PRIMARY_COLOR),

			error_msg: None,
			reciver,
			adsr_editor: self.adsr_params.clone(),
			unison_editor: self.unison_editor.clone(),
			pitch_factor: self.pitch_factor.clone(),
			gain: self.gain.clone(),

			smooth_factor_carrier: self.smooth_factor_carrier.clone(),
			smooth_factor_modulator: self.smooth_factor_modulator.clone(),
			pm_factor: self.pm_factor.clone(),
			current_pos: current_postion,
			tables,
		}
	}
}