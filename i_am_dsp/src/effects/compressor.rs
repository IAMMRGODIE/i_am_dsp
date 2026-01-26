//! A basic compressor effect

// use i_am_dsp_derive::Parameters;

// use crate as i_am_dsp;

use i_am_dsp_derive::Parameters;

#[cfg(feature = "real_time_demo")]
use crate::tools::ring_buffer::RingBuffer;
use crate::{prelude::Enveloper, tools::{audio_io_chooser::AudioIoChooser, smoother::DoubleTimeConstant}, Effect, ProcessContext};

#[cfg(feature = "real_time_demo")]
const HISTORY_LEN: usize = 32768;

/// A basic compressor effect
#[derive(Parameters)]
pub struct Compressor<Envelope: Enveloper<CHANNELS>, const CHANNELS: usize = 2> {
	#[range(min = -60.0, max = 0.0)]
	/// The threshold for the compressor, saves in dB
	pub threshold: f32,
	#[range(min = 1.0, max = 10.0)]
	/// The ratio for the compressor, Should always be less than 1.0
	pub ratio: f32,
	#[sub_param]
	/// The audio input/output chooser
	pub audio_io_chooser: AudioIoChooser,
	/// The sample rate of the audio, saves in Hz
	#[skip]
	pub sample_rate: usize,
	/// The smoother for the gain factor
	#[sub_param]
	pub smoother: DoubleTimeConstant<1>,
	/// The output gain factor, saves in linear scale
	#[range(min = 0.01, max = 4.0)]
	#[logarithmic]
	pub gain_linear: f32,
	#[sub_param]
	/// The enveloper for the input signal
	pub enveloper: Envelope,
	#[range(min = 0.01, max = 1.0)]
	#[logarithmic]
	/// The gain factor for the output signal, saves in linear scale
	pub gain_factor: f32,

	#[cfg(feature = "real_time_demo")]
	#[skip]
	history: RingBuffer<f32>,
}

impl<Envelope: Enveloper<CHANNELS>, const CHANNELS: usize> Compressor<Envelope, CHANNELS> {
	/// Creates a new compressor effect with the given parameters.
	/// 
	/// # Panics
	/// 
	/// if `CHANNELS` is 0.
	pub fn new(
		enveloper: Envelope, 
		sample_rate: usize, 
		attack_time: f32, 
		release_time: f32, 
		threshold: f32,
		ratio: f32,
	) -> Self {
		assert!(CHANNELS > 0, "CHANNELS must be greater than 0");

		Self {
			enveloper,
			smoother: DoubleTimeConstant::new(attack_time, release_time, 0.0, sample_rate),
			threshold,
			ratio,
			sample_rate,
			audio_io_chooser: AudioIoChooser::Current,
			gain_factor: 1.0,
			gain_linear: 1.0,

			#[cfg(feature = "real_time_demo")]
			history: RingBuffer::new(HISTORY_LEN),
		}
	}
}

impl<Envelope: Enveloper<CHANNELS> + Send + Sync, const CHANNELS: usize> Effect<CHANNELS> for Compressor<Envelope, CHANNELS> {
	fn delay(&self) -> usize {
		self.enveloper.delay()
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"Compressor"
	}

	fn process(&mut self, samples: &mut [f32; CHANNELS], other: &[&[f32; CHANNELS]], _: &mut Box<dyn ProcessContext>) {
		self.ratio = self.ratio.abs();
		let mut io_to_envelop = *self.audio_io_chooser.choose(samples, other);
		self.enveloper.input_value(&mut io_to_envelop);
		let envelope = self.enveloper.get_current_envelope();
		let envelop_db = 20.0 * (envelope.iter().map(|x| x * x).sum::<f32>()).sqrt().max(1e-6).log10();
		let envelop_db_compressed = if envelop_db < self.threshold {
			envelop_db
		}else if self.ratio.is_infinite() {
			self.threshold
		}else {
			(envelop_db - self.threshold) / self.ratio + self.threshold
		};
		
		let target_db = envelop_db_compressed - envelop_db;

		self.smoother.input_value(&[target_db]);
		self.gain_factor = self.smoother.get_smoothed_result()[0];
		self.gain_factor = 10.0_f32.powf(self.gain_factor / 20.0);

		#[cfg(feature = "real_time_demo")]
		{
			self.history.push(self.gain_factor);
		}

		for sample in samples.iter_mut() {
			*sample *= self.gain_factor * self.gain_linear;
		}
			
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		use egui::*;
		use egui::emath::RectTransform;
		use crate::tools::ui_tools::{draw_envelope, gain_ui};

		ui.horizontal(|ui| {
			let avaliable_width = ui.available_width();

			Resize::default()
			.id_salt(format!("{}_compresser_resize", id_prefix))
			.max_width(avaliable_width * 0.35)
			.max_height(avaliable_width * 0.35)
			.show(ui, |ui| {Frame::canvas(ui.style()).show(ui, |ui| {
				let (_, rect) = ui.allocate_space(Vec2::splat(ui.available_width()));
				let to_screen = RectTransform::from_to(
					Rect::from_x_y_ranges(0.0..=1.0, -1.0..=0.0), 
					rect
				);
				let threshold_linear = 10.0_f32.powf(self.threshold / 20.0);
				let compressed = if self.ratio.is_infinite() {
					threshold_linear
				}else if self.ratio <= 0.0 {
					1.0
				}else {
					(1.0 - threshold_linear) / self.ratio + threshold_linear
				};
				let points = vec![
					to_screen * pos2(0.0, 0.0),
					to_screen * pos2(threshold_linear, -threshold_linear),
					to_screen * pos2(1.0, -compressed),
				];

				ui.painter().extend([
					Shape::line(points, (2.0, Color32::WHITE))
				]);
			})});

			ui.vertical(|ui| {
				let width = ui.available_width();
				Resize::default()
					.max_width(width)
					.min_width(width)
					.id_salt(format!("{}_compresser_env_in", id_prefix))
					.max_height(avaliable_width * 0.35 / 2.0)
					.show(ui, |ui| 
				{
					self.enveloper.demo_ui(ui, format!("{}_compresser_env_in", id_prefix))
				});

				Resize::default()
				.id_salt(format!("{}_compresser_env_out", id_prefix))
				.max_height(avaliable_width * 0.35 / 2.0)
				.max_width(width)
				.min_width(width)
				.show(ui, |ui| {
					let history = std::mem::take(&mut self.history);
					draw_envelope(ui, &[&history], false);
					self.history = history;
				});
			})
		});

		gain_ui(ui, &mut self.gain_linear, None, false);

		ui.add(Slider::new(&mut self.smoother.attack_time, 0.0..=1000.0).text("Attack time (ms)"));
		ui.add(Slider::new(&mut self.smoother.release_time, 0.0..=1000.0).text("Release time (ms)"));
		ui.add(Slider::new(&mut self.threshold, -60.0..=0.0).text("Threshold (dB)"));
		ui.add(Slider::new(&mut self.ratio, 1.0..=10.0).text("Ratio"));

		// self.enveloper.demo_ui(ui, format!("{}_compresser_env_gui", id_prefix));
	}
}