//! Every control in the kit, at every size and every scale.
//!
//! Run it with:
//!
//! cargo run --release -p i_am_dsp_iced --example widget_kit
//!
//! Drag a knob, hold Shift while dragging it, spin the wheel over one, double
//! click one, and right click one: that is the interaction contract the whole
//! bundle is meant to share. The scale buttons at the top re-render everything at
//! another UI scale, which is how a faceplate is meant to handle DPI.

use i_am_dsp::prelude::{SineWave, Unit};
use i_am_dsp_iced::{
	iced::{
		self, Element, Length, Subscription, alignment,
		widget::{canvas, column, container, row, scrollable, text},
	},
	styles::{self, KnobSize},
	tools::{
		button::push_button,
		knob::knob,
		meter::Meter,
		selector::selector,
		slider::slider,
		spectrum::Spectrum,
		transfer::{Transfer, transfer},
		waveform::{Waveform, WaveformBuf},
	},
};

/// The state of the gallery.
pub struct WidgetKit {
	cutoff: f32,
	gain: f32,
	mix: f32,
	drive: f32,
	sync: f32,
	modulated: f32,
	modulation: f32,
	slot: usize,
	pan: f32,
	scale: f32,

	/// A stereo level meter, fed a random walk.
	meter: Meter,
	/// A spectrum display, fed a shape that drifts.
	spectrum: Spectrum,
	/// The bins the spectrum is drawn from, kept so the shape can drift.
	spectrum_bins: Vec<f32>,
	/// Where the drifting resonance sits, from 0 to 1.
	resonance: f32,
	/// A scrolling scope, fed samples.
	scope: WaveformBuf,
	/// The phase the scope's own signal is at.
	phase: f32,
	/// A wavetable, drawn as it is.
	table: Waveform<SineWave>,
	/// The curve of a shaper, which follows the Drive knob.
	transfer: Transfer,
	/// The two channel levels the random walk is at.
	levels: [f32; 2],
}

impl Default for WidgetKit {
	fn default() -> Self {
		let drive = 0.4;

		// The displays start on something fixed, not on noise: a still frame of the
		// gallery is what the snapshot test compares, and a random one would differ
		// every run. The random walk only begins once the ticks do.
		let mut scope = WaveformBuf::new(2048, 240.0);

		for sample in 0..2048 {
			let t = sample as f32 * 0.02;
			scope.update(&[(t).sin() * 0.7 * (1.0 + (t * 0.3).sin() * 0.3)]);
		}

		let spectrum_bins: Vec<f32> = (0..48)
			.map(|bin| {
				let t = bin as f32 / 47.0;

				(1.0 - t).powf(1.6) * 0.7
			})
			.collect();

		let spectrum = Spectrum::new();
		spectrum.update(&spectrum_bins);

		let meter = Meter::new();
		meter.update([0.72, 0.45], [0.5, 0.3]);

		Self {
			cutoff: 1200.0,
			gain: -3.0,
			mix: 0.65,
			drive,
			sync: 0.5,
			modulated: 0.35,
			modulation: 0.72,
			slot: 0,
			pan: 0.0,
			scale: styles::ui_scale(),
			meter,
			spectrum,
			spectrum_bins,
			resonance: 0.25,
			scope,
			phase: 0.0,
			table: Waveform::new(SineWave, 256, false),
			transfer: transfer(move |input| (input * (1.0 + drive * 6.0)).tanh()),
			levels: [0.72, 0.45],
		}
	}
}

#[derive(Debug, Clone)]
pub enum Message {
	Cutoff(f32),
	Gain(f32),
	Mix(f32),
	Drive(f32),
	Sync(f32),
	Modulated(f32),
	Slot(usize),
	Pan(f32),
	Scale(f32),
	Toggle,
	/// Keeps the frame redrawing, which is what makes the hover and click
	/// animations advance; without it a control animates only while the mouse
	/// is actually moving.
	Tick,
}

impl WidgetKit {
	fn update(&mut self, message: Message) {
		match message {
			Message::Cutoff(value) => self.cutoff = value,
			Message::Gain(value) => self.gain = value,
			Message::Mix(value) => self.mix = value,
			Message::Drive(value) => {
				self.drive = value;
				// A shaper's curve follows its controls: this is the one line of wiring
				// a hero needs.
				self.transfer.set_curve(move |input| (input * (1.0 + value * 6.0)).tanh());
			},
			Message::Sync(value) => self.sync = value,
			Message::Modulated(value) => self.modulated = value,
			Message::Slot(slot) => self.slot = slot,
			Message::Pan(value) => self.pan = value,
			Message::Scale(scale) => {
				styles::set_ui_scale(scale);
				self.scale = scale;
			},
			Message::Toggle => {},
			Message::Tick => self.stir(),
		}
	}

	/// Feeds the displays something that moves.
	///
	/// A random walk rather than noise, so a meter reads like a signal and a
	/// spectrum drifts instead of flickering. The scope gets real samples.
	fn stir(&mut self) {
		for (channel, level) in self.levels.iter_mut().enumerate() {
			let step = if channel == 0 { 0.07 } else { 0.05 };

			*level = (*level + rand::random_range(-step..step)).clamp(0.04, 1.0);
		}

		self.meter.update(self.levels, [self.levels[0] * 0.62, self.levels[1] * 0.55]);

		// The shape drifts towards a tilt, with a resonance that wanders across it.
		self.resonance = (self.resonance + rand::random_range(-0.012..0.012)).clamp(0.08, 0.85);

		let bin_count = self.spectrum_bins.len();

		for (index, bin) in self.spectrum_bins.iter_mut().enumerate() {
			let t = index as f32 / (bin_count - 1) as f32;
			let target = (1.0 - t).powf(1.6) * 0.6 + rand::random_range(0.0..0.25);

			*bin = *bin * 0.85 + target * 0.15;
		}

		let mut bins = self.spectrum_bins.clone();
		let centre = (self.resonance * (bins.len() - 1) as f32) as usize;

		for index in centre.saturating_sub(2)..(centre + 2).min(bins.len()) {
			bins[index] = (bins[index] + 0.4).min(1.0);
		}

		self.spectrum.update(&bins);

		// The scope gets samples, the way an effect would hand them over.
		self.phase = (self.phase + 0.013) % 1.0;

		for step in 0..64 {
			let t = self.phase + step as f32 * 0.006;
			let signal = (t * std::f32::consts::TAU * 4.0).sin() * 0.65 * (1.0 + (t * 5.0).sin() * 0.35);

			self.scope.update(&[signal]);
		}
	}

	/// Keeps the modifier state the controls read, and nothing else.
	///
	/// A plug-in gets this from [i_am_dsp_iced::modifiers_subscription]; the
	/// gallery has no processor, so it subscribes by hand.
	fn subscription(&self) -> Subscription<Message> {
		Subscription::batch([
			// A plug-in gets this from i_am_dsp_iced::timer; the gallery has no
			// processor, so it asks for the same 16 ms heartbeat directly.
			iced::time::every(std::time::Duration::from_millis(16)).map(|_| Message::Tick),
			iced::event::listen_with::<Message>(|event, _status, _window| {
				if let iced::Event::Keyboard(iced::keyboard::Event::ModifiersChanged(modifiers)) = event {
					i_am_dsp_iced::tools::input::set_modifiers(modifiers);
				}

				None
			}),
		])
	}

	/// The whole gallery.
	pub fn view(&self) -> Element<'_, Message> {
		let header = row![
			text("Widget kit").size(styles::px(styles::font::TITLE)),
			text("Shift: fine, double click or Ctrl+click: reset, wheel: step, right click: menu")
				.size(styles::px(styles::font::CAPTION))
				.color(styles::SECONDARY_TEXT_COLOR),
		]
		.spacing(styles::u(styles::space::L))
		.align_y(alignment::Vertical::Center);

		let scales = row![
			text("Scale").size(styles::px(styles::font::LABEL)),
			self.scale_button(0.75),
			self.scale_button(1.0),
			self.scale_button(1.25),
			self.scale_button(1.5),
			self.scale_button(2.0),
		]
		.spacing(styles::u(styles::space::S))
		.align_y(alignment::Vertical::Center);

		// The knobs themselves: the same widget, told apart only by the metadata
		// a faceplate would carry anyway.
		// One common height for the row, so the labels line up on a baseline even
		// though the rings are three different sizes: the label sticks to the
		// bottom of the box and the ring to the top.
		let cell = Length::Fixed(KnobSize::Large.labeled_height());
		let sizes = self.section(
			"Sizes",
			row![
				knob(0.0..=1.0, self.mix, Message::Mix)
					.size(KnobSize::Small)
					.height(cell)
					.label("Mix")
					.unit(Unit::Percent)
					.default_value(1.0),
				knob(0.0..=1.0, self.mix, Message::Mix)
					.size(KnobSize::Medium)
					.height(cell)
					.label("Mix")
					.unit(Unit::Percent)
					.default_value(1.0),
				knob(0.0..=1.0, self.mix, Message::Mix)
					.size(KnobSize::Large)
					.height(cell)
					.label("Mix")
					.unit(Unit::Percent)
					.default_value(1.0),
			]
			.spacing(styles::u(styles::space::L))
			.align_y(alignment::Vertical::Top),
		);

		let readouts = self.section(
			"Units",
			row![
				knob(20.0..=20000.0, self.cutoff, Message::Cutoff)
					.logarithmic(true)
					.label("Cutoff")
					.unit(Unit::Hertz)
					.default_value(1000.0),
				knob(-24.0..=24.0, self.gain, Message::Gain)
					.bipolar(true)
					.label("Gain")
					.unit(Unit::Decibels)
					.step(0.1)
					.default_value(0.0),
				knob(0.03125..=4.0, self.sync, Message::Sync)
					.label("Time")
					.unit(Unit::Beats)
					.default_value(0.5),
				knob(0.0..=1.0, self.drive, Message::Drive)
					.label("Drive")
					.value_formatter(|value| format!("{:.0} %", value * 100.0)),
			]
			.spacing(styles::u(styles::space::L)),
		);

		let modulation = self.section(
			"Modulation",
			row![
				knob(0.0..=1.0, self.modulated, Message::Modulated)
					.label("Modulated")
					.unit(Unit::Percent)
					.modulation(self.modulation),
				knob(0.0..=1.0, self.modulated, Message::Modulated)
					.label("Unmodulated")
					.unit(Unit::Percent),
				knob(-24.0..=24.0, self.gain, Message::Gain)
					.bipolar(true)
					.label("Modulated gain")
					.unit(Unit::Decibels)
					.modulation(9.0)
					.default_value(0.0),
			]
			.spacing(styles::u(styles::space::L)),
		);

		let visuals = self.section(
			"Visualisers",
			row![
				canvas(&self.meter).width(styles::u(22.0)).height(styles::u(34.0)),
				column![
					canvas(&self.scope).width(styles::u(60.0)).height(styles::u(16.0)),
					canvas(&self.spectrum).width(styles::u(60.0)).height(styles::u(16.0)),
				]
				.spacing(styles::u(styles::space::S)),
				column![
					canvas(&self.table).width(styles::u(30.0)).height(styles::u(16.0)),
					canvas(&self.transfer).width(styles::u(30.0)).height(styles::u(16.0)),
				]
				.spacing(styles::u(styles::space::S)),
			]
			.spacing(styles::u(styles::space::L))
			.align_y(alignment::Vertical::Top),
		);

		let width = Length::Fixed(styles::u(72.0));
		let rows = self.section(
			"Row controls",
			column![
				selector(vec!["Low", "Band", "High"], self.slot)
					.on_change(|_slots, slot| Message::Slot(slot))
					.width(width),
				slider(0.0..=1.0_f32, self.mix, Message::Mix)
					.text("Mix")
					.formatter(|value| format!("{:.0} %", value * 100.0))
					.default_value(1.0)
					.width(width),
				slider(-1.0..=1.0_f32, self.pan, Message::Pan)
					.text("Pan")
					.bipolar(true)
					.default_value(0.0)
					.modulation(self.pan + 0.45)
					.width(width),
				push_button("A button").on_press(Message::Toggle),
			]
			.spacing(styles::u(styles::space::S)),
		);

		container(
			scrollable(
				column![header, scales, sizes, visuals, rows, readouts, modulation]
					.spacing(styles::u(styles::space::XL))
					.padding(styles::u(styles::space::XL)),
			)
			.width(Length::Fill),
		)
		.width(Length::Fill)
		.height(Length::Fill)
		.style(|_| container::Style {
			background: Some(styles::BACKGROUND_COLOR.into()),
			..Default::default()
		})
		.into()
	}

	fn scale_button(&self, scale: f32) -> Element<'_, Message> {
		let label = format!("{:.0} %", scale * 100.0);
		let active = (self.scale - scale).abs() < f32::EPSILON;
		push_button(label).accent(active).on_press(Message::Scale(scale)).into()
	}

	/// One titled card of controls.
	fn section<'a>(&'a self, title: &'a str, content: impl Into<Element<'a, Message>>) -> Element<'a, Message> {
		let content: Element<'a, Message> = content.into();
		container(
			column![
				text(title).size(styles::px(styles::font::SECTION)).color(styles::SECONDARY_TEXT_COLOR),
				content,
			]
			.spacing(styles::u(styles::space::M)),
		)
		.padding(styles::u(styles::space::L))
		.width(Length::Fill)
		.style(|_| container::Style {
			background: Some(styles::CARD_COLOR.into()),
			border: iced::Border {
				color: styles::CARD_BORDER_COLOR,
				width: 1.0,
				radius: styles::u(styles::size::RADIUS).into(),
			},
			..Default::default()
		})
		.into()
	}
}

#[cfg(feature = "standalone")]
fn main() -> iced::Result {
	// A plug-in reads its scale from the host; the gallery takes it from the
	// environment, so the same build can be compared at different scales.
	if let Ok(scale) = std::env::var("IAM_UI_SCALE")
		&& let Ok(scale) = scale.parse::<f32>()
	{
		styles::set_ui_scale(scale);
	}

	iced::application(WidgetKit::default, WidgetKit::update, WidgetKit::view)
		.subscription(WidgetKit::subscription)
		.theme(styles::theme())
		.window_size((940.0, 900.0))
		.run()
}

#[cfg(not(feature = "standalone"))]
fn main() {
	println!("the standalone feature is not enabled, nothing to run");
}
