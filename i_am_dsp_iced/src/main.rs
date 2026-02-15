#[cfg(feature = "standalone")]
fn main() {
	use i_am_dsp_iced::styles::theme;
	use i_am_dsp_iced::{demo::Demo, plugins::wavetable_synth::WavetableSynth};
	use i_am_dsp::prelude::{SawWave, SineWave, SquareWave, TriangleWave};
	use i_am_dsp::prelude::WaveTable;

	iced::application(|| {
		Demo::new(|sample_rate| {
			WavetableSynth::new(sample_rate, |_| {
				vec![
					Box::new(SineWave) as Box<dyn WaveTable + Send + Sync>,
					Box::new(TriangleWave) as Box<dyn WaveTable + Send + Sync>,
					Box::new(SawWave) as Box<dyn WaveTable + Send + Sync>,
					Box::new(SquareWave) as Box<dyn WaveTable + Send + Sync>,
				]
			})
		})
	}, Demo::update, Demo::view)
		.subscription(|_| { Demo::<WavetableSynth>::subscriber() })
		.theme(theme())
		.window_size((720.0, 560.0))
		.run().expect("cant run app")
}

#[cfg(not(feature = "standalone"))]
fn main() {
	println!("`standalone` feature not enabled, nothing to do.");
}
