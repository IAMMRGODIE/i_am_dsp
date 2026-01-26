// use crate::{prelude::{Biquad, Enveloper, MIN_FREQUENCY}, tools::audio_io_chooser::AudioIoChooser, Effect};

// pub struct Vocoder<Envelope: Enveloper<CHANNELS>, const CHANNELS: usize> {
// 	envelope: Vec<Envelope>,
// 	biquads: Vec<Biquad<CHANNELS>>,
// 	logarithmic: bool,
// 	env_history: Vec<f32>,
// 	create_envelope: Box<dyn Fn(usize) -> Envelope + Send + Sync>,
// 	pub attack_time: f32,
// 	pub release_time: f32,
// 	pub audio_io_chooser: AudioIoChooser,
// }

// impl<Envelope: Enveloper<CHANNELS>, const CHANNELS: usize> Vocoder<Envelope, CHANNELS> {
// 	/// Creates a new Vocoder with the given parameters.
// 	/// 
// 	/// will pass sample_rate to `create_envelope`.
// 	/// 
// 	/// Panics if `bands` is 0.
// 	pub fn new(
// 		create_envelope: impl Fn(usize) -> Envelope + Send + Sync + 'static,
// 		bands: usize,
// 		sample_rate: usize,
// 		logarithmic: bool,
// 		audio_io_chooser: AudioIoChooser,
// 	) -> Self {
// 		assert!(bands > 0);

// 		let mut biquads = Vec::with_capacity(bands);
// 		let mut envelope = Vec::with_capacity(bands);

// 		for _  in 0..bands {
// 			envelope.push(create_envelope(sample_rate));
// 		}

// 		if bands == 1 {
// 			biquads.push(Biquad::new(sample_rate));
// 		}else {
// 			for i in 0..bands {
// 				let freq = calc_freq(i, bands, sample_rate, logarithmic);
// 				let next_freq = calc_freq(i + 1, bands, sample_rate, logarithmic);
// 				let bandwidth = (next_freq - freq) / 2.0;
// 				let center_freq = (freq + next_freq) / 2.0;
// 				biquads.push(Biquad::bandpass(sample_rate, center_freq, bandwidth));
// 			}
// 		}

// 		Self { 
// 			envelope, 
// 			biquads, 
// 			logarithmic, 
// 			env_history: vec![1.0; bands], 
// 			attack_time: 46.0, 
// 			release_time: 200.0, 
// 			audio_io_chooser,
// 			create_envelope: Box::new(create_envelope),
// 		}
// 	}

// 	pub fn toggle_logarithmic(&mut self) {
// 		self.logarithmic = !self.logarithmic;
// 		let bands = self.biquads.len();
// 		if bands != 1 {
// 			for (i, biquad) in self.biquads.iter_mut().enumerate() {
// 				let freq = calc_freq(i, bands, biquad.sample_rate(), self.logarithmic);
// 				let next_freq = calc_freq(i + 1, bands, biquad.sample_rate(), self.logarithmic);
// 				let bandwidth = (next_freq - freq) / 2.0;
// 				let center_freq = (freq + next_freq) / 2.0;
// 				biquad.set_to_bandpass(center_freq, bandwidth);
// 			}
// 		}
// 	}

// 	/// Changes band parameters.
// 	pub fn reset_band(
// 		&mut self,
// 		bands: usize,
// 	) {
// 		assert!(bands > 0);
// 		let sample_rate: usize = self.biquads[0].sample_rate();
// 		let mut biquads = Vec::with_capacity(bands);
// 		let mut envelope = Vec::with_capacity(bands);

// 		for _  in 0..bands {
// 			envelope.push((self.create_envelope)(sample_rate));
// 		}

// 		if bands == 1 {
// 			biquads.push(Biquad::new(sample_rate));
// 		}else {
// 			for i in 0..bands {
// 				let freq = calc_freq(i, bands, sample_rate, self.logarithmic);
// 				let next_freq = calc_freq(i + 1, bands, sample_rate, self.logarithmic);
// 				let bandwidth = (next_freq - freq) / 2.0;
// 				let center_freq = (freq + next_freq) / 2.0;
// 				biquads.push(Biquad::bandpass(sample_rate, center_freq, bandwidth));
// 			}
// 		}

// 		self.envelope = envelope;
// 		self.biquads = biquads;
// 		self.env_history.resize(bands, 1.0);
// 	}
// }

// fn calc_freq(i: usize, bands: usize, sample_rate: usize, logarithmic: bool) -> f32 {
// 	let max_freq = sample_rate as f32 / 2.0 - MIN_FREQUENCY;
// 	if logarithmic {
// 		let lerped_freq = MIN_FREQUENCY.ln() + (max_freq.ln() - MIN_FREQUENCY.ln()) * i as f32 / bands as f32;
// 		lerped_freq.exp()
// 	}else {
// 		MIN_FREQUENCY + (max_freq - MIN_FREQUENCY) * i as f32 / bands as f32
// 	}
// }

// impl<Envelope: Enveloper<CHANNELS> + Send + Sync, const CHANNELS: usize> Effect<CHANNELS> for Vocoder<Envelope, CHANNELS> {
// 	fn delay(&self) -> usize {
// 		0
// 	}

// 	fn name(&self) -> &str {
// 		"Vocoder"
// 	}

// 	fn process(&mut self, samples: &mut [f32; CHANNELS], other: &[&[f32; CHANNELS]]) {
// 		let input = *self.audio_io_chooser.choose(samples, other);
// 		let sample_rate = self.biquads[0].sample_rate() as f32;
// 		let attack_factor = 1.0 - (-1.0 / (self.attack_time * sample_rate / 1000.0)).exp();
// 		let release_factor = 1.0 - (-1.0 / (self.release_time * sample_rate / 1000.0)).exp();

// 		*samples = [0.0; CHANNELS];

// 		for ((biquad, envelope), env_history) in self.biquads.iter_mut()
// 			.zip(self.envelope.iter_mut())
// 			.zip(self.env_history.iter_mut())
// 		{
// 			let mut input_value = input;
// 			biquad.process(&mut input_value, other);
// 			envelope.input_value(&mut input_value);
// 			let current_env = envelope
// 				.get_current_envelope()
// 				.into_iter()
// 				.map(|inner| inner * inner)
// 				.sum::<f32>().sqrt();

// 			if current_env < *env_history {
// 				*env_history = attack_factor * current_env + (1.0 - attack_factor) * *env_history;
// 			}else {
// 				*env_history = (1.0 - release_factor) * *env_history + release_factor * current_env;
// 			}

// 			for (i, sample) in input_value.into_iter().enumerate() {
// 				samples[i] += sample * *env_history;
// 			}
// 		}
// 	}

// 	#[cfg(feature = "real_time_demo")]
// 	fn demo_ui(&mut self, ui: &mut egui::Ui, _: String) {
// 		// self.envelope

// 		ui.add(egui::Slider::new(&mut self.attack_time, 0.0..=1000.0).text("Attack Time"));
// 		ui.add(egui::Slider::new(&mut self.release_time, 0.0..=1000.0).text("Release Time"));
// 		let mut bands = self.biquads.len();
// 		ui.add(egui::Slider::new(&mut bands, 1..=100).text("Bands"));
// 		if bands != self.biquads.len() {
// 			self.reset_band(bands);
// 		}
// 		if ui.selectable_label(self.logarithmic, "Logarithmic").clicked() {
// 			self.toggle_logarithmic();
// 		}
// 	}
// }