//! A eq effect

#[cfg(feature = "standalone")]
fn main() {
	use i_am_dsp_iced::styles::theme;
	use i_am_dsp_iced::{demo::Demo, plugins::eq::Equalizer};

	iced::application(|| {
		Demo::new(|sample_rate| {
			Equalizer::new(sample_rate, 7)
		})
	}, Demo::update, Demo::view)
		.subscription(|_| { Demo::<Equalizer>::subscriber() })
		.theme(theme())
		.window_size((720.0, 560.0))
		.run().expect("cant run app")
}

#[cfg(not(feature = "standalone"))]
fn main() {
	println!("`standalone` feature not enabled, nothing to do.");
}
