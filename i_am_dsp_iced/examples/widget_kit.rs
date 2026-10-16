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

use i_am_dsp::prelude::{SawWave, SineWave, SquareWave, TriangleWave, Unit, WaveTable};
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
		spectrogram::Spectrogram,
		spectrum::Spectrum,
		stereo::StereoMeter,
		transfer::{Transfer, transfer},
		waveform::{Waveform, WaveformBuf},
	},
};

/// The amplitude a gain in decibels means.
fn amplitude_of(db: f32) -> f32 {
	10.0_f32.powf(db / 20.0)
}

/// Where a frequency sits between two others on a logarithmic axis, from 0 to 1.
fn across_log(frequency: f32, low: f32, high: f32) -> f32 {
	let frequency = frequency.max(low);

	((frequency.log10() - low.log10()) / (high.log10() - low.log10())).clamp(0.0, 1.0)
}

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
	/// A scrolling scope, fed samples.
	scope: WaveformBuf,
	/// The phase the scope's own signal is at.
	phase: f32,
	/// The curve of a shaper, which follows the Drive knob.
	transfer: Transfer,
	/// A stereo field display, fed the two channels a pan apart.
	stereo: StereoMeter,
	/// The frames the stereo display is given, kept so they are not rebuilt.
	stereo_frames: Vec<(f32, f32)>,
	/// A spectrogram, fed the same spectrum as the display above it.
	spectrogram: Spectrogram,
	/// The tables the wavetable display can show, chosen by the selector.
	tables: Vec<Waveform<Box<dyn WaveTable>>>,
	/// Which of them is on screen.
	table: usize,
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
		meter.update([0.55, 0.38], [0.34, 0.22]);

		// The stereo display and the spectrogram are fed on the tick, so they are
		// seeded here: a still frame of the gallery should not show two empty boxes
		// any more than it should show a random one.
		let stereo = StereoMeter::new(96);
		let mut stereo_frames = Vec::with_capacity(96);

		for step in 0..64 {
			let t = step as f32 / 64.0 * std::f32::consts::TAU;

			stereo_frames.push(((t * 3.0).sin() * 0.6, (t * 3.0 + 0.4).sin() * 0.5));
		}

		stereo.update(&stereo_frames);

		let spectrogram = Spectrogram::new(64).bands(64);

		for column in 0..64 {
			let decay = 1.0 - column as f32 / 80.0;
			let mut shaded: Vec<f32> = spectrum_bins.iter().map(|bin| bin * decay).collect();

			// A resonance that moves from column to column, so a still frame reads as
			// a waterfall rather than as one shape repeated.
			let count = shaded.len();
			let centre = (column * 3) % count;

			for band in &mut shaded[centre.saturating_sub(2)..(centre + 2).min(count)] {
				*band = (*band + 0.4).min(1.0);
			}

			spectrogram.update(&shaded);
		}

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
			scope,
			phase: 0.0,
			tables: vec![
				Waveform::new(Box::new(SineWave) as Box<dyn WaveTable>, 256, false),
				Waveform::new(Box::new(TriangleWave) as Box<dyn WaveTable>, 256, false),
				Waveform::new(Box::new(SawWave) as Box<dyn WaveTable>, 256, false),
				Waveform::new(Box::new(SquareWave) as Box<dyn WaveTable>, 256, false),
			],
			table: 0,
			transfer: transfer(move |input| (input * (1.0 + drive * 6.0)).tanh()),
			stereo,
			stereo_frames,
			spectrogram,
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
		// The knobs drive the displays -- Gain is the level, Cutoff is where the
		// spectrum peaks and how fast the scope runs, Pan opens the stereo field --
		// and a little wobble on top keeps them reading as a signal rather than as
		// a still picture of the settings.
		let amplitude = amplitude_of(self.gain).min(2.0);
		// Scaled so that the gain the gallery starts at sits in the middle of the
		// meter and turning it up is what reaches the top.
		let level = (amplitude * (0.07 + self.mix * 0.25)).min(1.0);
		let balance = self.pan.clamp(-1.0, 1.0);

		// The pan is a balance: at the centre the two channels are equal, so the
		// meter sits level and the stereo field stands up; turning it leans both.
		let left = level * (1.0 - balance * 0.5);
		let right = level * (1.0 + balance * 0.5);

		self.meter.update(
			[
				(left + rand::random_range(-0.02..0.02)).clamp(0.0, 1.0),
				(right + rand::random_range(-0.02..0.02)).clamp(0.0, 1.0),
			],
			[left * 0.62, right * 0.62],
		);

		// The spectrum: a tilt, a peak where the filter is, and the level the gain
		// puts it at.
		let resonance = across_log(self.cutoff, 20.0, 20_000.0);
		let bin_count = self.spectrum_bins.len();

		for (index, bin) in self.spectrum_bins.iter_mut().enumerate() {
			let t = index as f32 / (bin_count - 1) as f32;
			let target = (1.0 - t).powf(1.6) * 0.5 * amplitude.min(1.5) + rand::random_range(0.0..0.12);

			*bin = *bin * 0.85 + target.clamp(0.0, 1.0) * 0.15;
		}

		let mut bins = self.spectrum_bins.clone();
		let centre = (resonance * (bin_count - 1) as f32).round() as usize;

		for band in &mut bins[centre.saturating_sub(2)..(centre + 2).min(bin_count)] {
			*band = (*band + 0.45).min(1.0);
		}

		self.spectrum.update(&bins);
		self.spectrogram.update(&bins);

		// The scope draws the gain as amplitude and the cutoff as frequency.
		self.phase = (self.phase + 0.013) % 1.0;

		let cycles = 1.0 + across_log(self.cutoff, 20.0, 20_000.0) * 9.0;
		let scope_gain = amplitude.min(1.0) * 0.8;

		for step in 0..64 {
			let t = self.phase + step as f32 * 0.006;
			let signal = (t * std::f32::consts::TAU * cycles).sin()
				* scope_gain
				* (1.0 + (t * 5.0).sin() * 0.25);

			self.scope.update(&[signal]);
		}

		// The stereo field: the two channels are the same signal a pan apart, so the
		// scope stands up for mono and opens out as the pan widens.
		// A little width even at the centre, so the display shows a field rather than
		// a single line; the pan opens it from there.
		let offset = 0.18 + self.pan * std::f32::consts::FRAC_PI_2;
		self.stereo_frames.clear();

		for step in 0..64 {
			let t = self.phase * 6.0 + step as f32 * 0.11;

			self.stereo_frames.push((
				(t.sin() * left).clamp(-1.0, 1.0),
				((t + offset).sin() * right).clamp(-1.0, 1.0),
			));
		}

		self.stereo.update(&self.stereo_frames);

		// The wavetable display shows the table the selector picked.
		self.table = self.slot.min(self.tables.len().saturating_sub(1));
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
			column![
				row![
					canvas(&self.meter).width(styles::u(20.0)).height(styles::u(32.0)),
					column![
						canvas(&self.scope).width(styles::u(58.0)).height(styles::u(15.0)),
						canvas(&self.spectrum).width(styles::u(58.0)).height(styles::u(15.0)),
					]
					.spacing(styles::u(styles::space::S)),
					column![
						canvas(&self.tables[self.table]).width(styles::u(26.0)).height(styles::u(15.0)),
						canvas(&self.transfer).width(styles::u(26.0)).height(styles::u(15.0)),
					]
					.spacing(styles::u(styles::space::S)),
					canvas(&self.stereo).width(styles::u(30.0)).height(styles::u(32.0)),
				]
				.spacing(styles::u(styles::space::L))
				.align_y(alignment::Vertical::Top),
				canvas(&self.spectrogram).width(styles::u(158.0)).height(styles::u(18.0)),
			]
			.spacing(styles::u(styles::space::S)),
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
