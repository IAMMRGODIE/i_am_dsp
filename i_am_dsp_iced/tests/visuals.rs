//! The visualisers, drawn.
//!
//! Every one of these is a canvas program, so a frame of it is the whole of what
//! it does: this renders them side by side with real data and keeps the frame as
//! a reference, which is the only way to notice that a display has stopped
//! drawing anything.

use std::path::Path;

use i_am_dsp::prelude::SineWave;
use i_am_dsp_iced::{
	iced::{
		Element, Length,
		widget::{canvas, column, row, text},
	},
	styles,
	tools::{
		meter::Meter,
		spectrum::Spectrum,
		transfer::transfer,
		waveform::{Waveform, WaveformBuf},
	},
};

#[derive(Debug, Clone)]
struct Message;

/// One visualiser, under its name.
fn panel<'a>(name: &'a str, display: impl Into<Element<'a, Message>>, size: (f32, f32)) -> Element<'a, Message> {
	column![
		text(name.to_string())
			.size(styles::px(styles::font::SECTION))
			.color(styles::SECONDARY_TEXT_COLOR),
		display.into(),
	]
	.spacing(styles::u(styles::space::XS))
	.width(Length::Fixed(size.0))
	.into()
}

#[test]
fn the_visualisers_draw() {
	// A scrolled waveform, fed the way an effect feeds it: one sample at a time.
	let mut waveform = WaveformBuf::new(512, 320.0);

	for sample in 0..4096 {
		let t = sample as f32;
		let signal = (t * 0.08).sin() * 0.8 * (1.0 + (t * 0.001).sin() * 0.4);

		waveform.update(&[signal]);
	}

	// A wavetable, drawn from the table itself.
	let table = Waveform::new(SineWave, 128, false);

	// A meter with one channel loud and one quiet: the shape of a mix.
	let meter = Meter::new();
	meter.update([0.82, 0.34], [0.5, 0.18]);

	// A spectrum with a tilt and one resonance in it.
	let spectrum = Spectrum::new();
	let bins: Vec<f32> = (0..64)
		.map(|bin| {
			let t = bin as f32 / 63.0;
			let tilt = (1.0 - t).powf(1.6) * 0.65;
			let resonance = (1.0 - (t - 0.22).abs() * 11.0).max(0.0) * 0.35;

			(tilt + resonance).clamp(0.0, 1.0)
		})
		.collect();

	spectrum.update(&bins);

	// A soft clipper: the curve a drive effect is.
	let shaper = transfer(|input| (input * 2.0).tanh() * 0.9);

	let sheet: Element<'_, Message> = column![
		row![
			panel("Waveform (buffer)", canvas(waveform).width(320.0).height(120.0), (320.0, 150.0)),
			panel("Waveform (table)", canvas(table).width(320.0).height(120.0), (320.0, 150.0)),
		]
		.spacing(styles::u(styles::space::L)),
		row![
			panel("Meter", canvas(meter).width(96.0).height(160.0), (96.0, 190.0)),
			panel("Spectrum", canvas(spectrum).width(300.0).height(160.0), (300.0, 190.0)),
			panel("Transfer", canvas(shaper).width(150.0).height(160.0), (150.0, 190.0)),
		]
		.spacing(styles::u(styles::space::L)),
	]
	.spacing(styles::u(styles::space::L))
	.padding(styles::u(styles::space::L))
	.into();

	let mut ui = iced_test::simulator(sheet);
	let snapshot = match ui.snapshot(&styles::theme()) {
		Ok(snapshot) => snapshot,
		Err(error) => panic!("the visualisers render: {error:?}"),
	};

	let path = Path::new(env!("CARGO_MANIFEST_DIR"))
		.join("tests")
		.join("snapshots")
		.join("visuals.png");

	assert!(
		snapshot.matches_image(&path).expect("the frame is written"),
		"the visualisers changed; delete {} and run again to accept it",
		path.display()
	);
}
