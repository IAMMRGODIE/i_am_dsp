//! Envelope Getter

use i_am_parameters_derive::Parameters;

use crate::{effects::prelude::{hilbert_transform, Convolver, HilbertTransform}, prelude::Parameters, tools::ring_buffer::RingBuffer, Effect, ProcessContext};

/// The main trait for envelopes
pub trait Enveloper<const CHANNELS: usize = 2>: Parameters {
	/// Get the delay of the envelope
	fn delay(&self) -> usize;
	/// Input a value to the envelope
	fn input_value(&mut self, input: &mut [f32; CHANNELS]);
	/// Get the current envelope value
	fn get_current_envelope(&self) -> [f32; CHANNELS];
	
	#[cfg(feature = "real_time_demo")]
	/// The name of the envelope, used to identify the envelope in the demo UI
	fn name(&self) -> &str {
		"Anonymous Enveloper"
	}

	#[cfg(feature = "real_time_demo")]
	/// The UI for the envelope, used in the demo UI
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		let _ = id_prefix;
		ui.label("Unimplemented");
	}
}

impl<const CHANNELS: usize, T: Enveloper<CHANNELS> + Sync + Send> Effect<CHANNELS> for T {
	fn process(&mut self, samples: &mut [f32; CHANNELS], _: &[&[f32; CHANNELS]], _: &mut Box<dyn ProcessContext>) {
		self.input_value(samples);
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		self.name()
	}

	fn delay(&self) -> usize {
		self.delay()
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		self.demo_ui(ui, id_prefix)
	}
}

/// An envelope with history
/// 
/// This is mainly used for the demo UI, to show the history of the envelope
#[derive(Parameters)]
pub struct EnvelopeWithHistory<T: Enveloper<CHANNELS>, const CHANNELS: usize = 2> {
	#[skip]
	env_history: [RingBuffer<f32>; CHANNELS],
	#[sub_param]
	envelope: T,
}

impl<const CHANNELS: usize, T: Enveloper<CHANNELS>> EnvelopeWithHistory<T, CHANNELS> {
	/// Create a new envelope with history
	/// 
	/// # Panics
	/// 
	/// Panics if `CHANNELS` is 0;
	pub fn new(env: T, history_len: usize) -> Self {
		assert!(CHANNELS > 0, "CHANNELS must be greater than 0");

		let env_history = core::array::from_fn(|_| RingBuffer::new(history_len));
		Self {
			env_history,
			envelope: env,
		}
	}

	/// Change the length of the history
	/// 
	/// This will clear the history
	pub fn change_length(&mut self, history_len: usize) {
		self.env_history = core::array::from_fn(|_| RingBuffer::new(history_len));
	}
}

impl<const CHANNELS: usize, T: Enveloper<CHANNELS>> Enveloper<CHANNELS> for EnvelopeWithHistory<T, CHANNELS> {
	fn input_value(&mut self, input: &mut [f32; CHANNELS]) {
		self.envelope.input_value(input);
		for (i, val) in self.envelope.get_current_envelope().into_iter().enumerate() {
			self.env_history[i].push(val);
		}
	}

	fn get_current_envelope(&self) -> [f32; CHANNELS] {
		let mut result = [0.0; CHANNELS];
		for (i, buffer) in self.env_history.iter().enumerate() {
			result[i] = buffer[-1]
		}
		result
	}

	fn delay(&self) -> usize {
		self.envelope.delay()
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		self.envelope.name()
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		use egui::*;
		use crate::tools::ui_tools::draw_envelope;

		egui::Resize::default().resizable([false, true])
			.min_width(ui.available_width())
			.max_width(ui.available_width())
			.id_salt(format!("{id_prefix}_envelope_with_history"))
			.show(ui, |ui| 
		{
			draw_envelope(ui, &self.env_history.iter().collect::<Vec<&RingBuffer<f32>>>(), false)
		});

		ScrollArea::neither().show(ui, |ui| {
			ui.allocate_space(Vec2::new(ui.available_width(), 1.0));
			self.envelope.demo_ui(ui, format!("{} with history", id_prefix))
		});
	}
}

/// Get the envelope using an IIR Hilbert Transform
#[derive(Parameters)]
pub struct IIRHilbertEnvelope<const ORDER: usize, const CHANNELS: usize = 2> {
	#[sub_param]
	hilbert_transformer: HilbertTransform<ORDER, CHANNELS>,
	#[skip]
	history: [f32; CHANNELS],
	/// The gain factor of the envelope, saves in linear scale.
	#[range(min = 0.01, max = 4.0)]
	#[logarithmic]
	pub gain_factor: f32,
}

impl<const ORDER: usize, const CHANNELS: usize> IIRHilbertEnvelope<ORDER, CHANNELS> {
	/// Create a new IIR Hilbert Envelope
	/// 
	/// Paniics if `CHANNELS` is 0;
	pub fn new(sample_rate: usize) -> Self {
		assert!(CHANNELS > 0, "CHANNELS must be greater than 0");

		Self {
			hilbert_transformer: HilbertTransform::new(sample_rate),
			history: [0.0; CHANNELS],
			gain_factor: 1.0,
		}
	}
}

impl<const ORDER: usize, const CHANNELS: usize> Enveloper<CHANNELS> for IIRHilbertEnvelope<ORDER, CHANNELS> {
	fn delay(&self) -> usize {
		0
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"IIR Hilbert Envelope"
	}

	fn get_current_envelope(&self) -> [f32; CHANNELS] {
		let mut out = self.history;
		for val in out.iter_mut() {
			*val *= self.gain_factor
		}
		out
	}

	fn input_value(&mut self, input: &mut [f32; CHANNELS]) {
		let mut transformed_input = *input;
		let mut ctx = Box::new(()) as Box<dyn ProcessContext>;
		self.hilbert_transformer.process(&mut transformed_input, &[], &mut ctx);

		for (i, val) in self.history.iter_mut().enumerate() {
			*val = input[i].hypot(transformed_input[i])
		}
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, _: String) {
		use crate::tools::ui_tools::gain_ui;

		gain_ui(ui, &mut self.gain_factor, None, false);
	}
}

/// Get the envelope by low-passing the rectified signal
#[derive(Parameters)]
pub struct RectifyEnvelope<const CHANNELS: usize = 2> {
	#[skip]
	envlope_history: [f32; CHANNELS],
	#[range(min = 10.0, max = 20000.0)]
	cutoff: f32,
	#[skip]
	sample_rate: usize,
	/// The gain factor of the envelope, saves in linear scale.
	#[range(min = 0.01, max = 4.0)]
	#[logarithmic]
	pub gain_factor: f32,
}

impl<const CHANNELS: usize> RectifyEnvelope<CHANNELS> {
	/// Create a new Rectify Envelope
	/// 
	/// Paniics if `CHANNELS` is 0;
	pub const fn new(sample_rate: usize, cutoff: f32) -> Self {
		assert!(CHANNELS > 0, "CHANNELS must be greater than 0");
		let envlope_history = [0.0; CHANNELS];
		Self {
			envlope_history,
			cutoff,
			sample_rate,
			gain_factor: 1.0,
		}
	}

	/// Set the cutoff frequency
	pub fn set_cutoff(&mut self, cutoff: f32) {
		self.cutoff = cutoff
	}

	/// Set the sample rate
	pub fn set_sample_rate(&mut self, sample_rate: usize) {
		self.sample_rate = sample_rate
	}
}

impl<const CHANNELS: usize> Enveloper<CHANNELS> for RectifyEnvelope<CHANNELS> {
	fn delay(&self) -> usize {
		0
	}

	fn get_current_envelope(&self) -> [f32; CHANNELS] {
		let mut out = self.envlope_history;
		for val in out.iter_mut() {
			*val *= self.gain_factor
		}
		out
	}

	fn input_value(&mut self, input: &mut [f32; CHANNELS]) {
		let alpha = 1.0 / (1.0 + self.sample_rate as f32 / self.cutoff);

		for (i, val) in self.envlope_history.iter_mut().enumerate() {
			*val = input[i].abs() * alpha + *val * (1.0 - alpha)
		}
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"Rectify Envelope"
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, _: String) {
		use crate::{effects::filter::MIN_FREQUENCY, tools::ui_tools::gain_ui};
		let max_freq = self.sample_rate as f32 / 2.0;

		ui.add(egui::Slider::new(&mut self.cutoff, MIN_FREQUENCY..=max_freq - MIN_FREQUENCY)
			.text("Cutoff Frequency (Hz)")
			.logarithmic(true)
		);
		gain_ui(ui, &mut self.gain_factor, None, false);
	}
}

/// Get the envelope by detact the peaks in the signal, and smooth it.
#[derive(Parameters)]
pub struct PeakDetector<const CHANNELS: usize = 2> {
	/// The attack time of the envelope in milliseconds.
	#[range(min = 0.0, max = 1000.0)]
	pub attack_time: f32,
	/// The release time of the envelope in milliseconds.
	#[range(min = 0.0, max = 1000.0)]
	pub release_time: f32,
	#[skip]
	/// The sample rate of the signal.
	pub sample_rate: usize,
	#[skip]
	envlope_history: [f32; CHANNELS],
	#[skip]
	last_history: [f32; CHANNELS],
	/// The gain factor of the envelope, saves in linear scale.
	#[range(min = 0.01, max = 4.0)]
	#[logarithmic]
	pub gain_factor: f32,
}

impl<const CHANNELS: usize> PeakDetector<CHANNELS> {
	/// Create a new Peak Detector
	/// 
	/// `attack_time` and `release_time` are in milliseconds.
	/// 
	/// Panics if `CHANNELS` is 0;
	pub const fn new(sample_rate: usize, attack_time: f32, release_time: f32) -> Self {
		assert!(CHANNELS > 0, "CHANNELS must be greater than 0");
		let envlope_history = [0.0; CHANNELS];
		let last_history = [0.0; CHANNELS];
		Self {
			attack_time,
			release_time,
			sample_rate,
			envlope_history,
			last_history,
			gain_factor: 1.0,
		}
	}
}

impl<const CHANNELS: usize> Enveloper<CHANNELS> for PeakDetector<CHANNELS> {
	fn delay(&self) -> usize {
		0
	}

	fn get_current_envelope(&self) -> [f32; CHANNELS] {
		let mut out = self.envlope_history;

		for val in out.iter_mut() {
			*val *= self.gain_factor
		}

		out
	}

	fn input_value(&mut self, input: &mut [f32; CHANNELS]) {
		let sample_rate = self.sample_rate as f32;
		let attack_factor = 1.0 - (-1.0 / (self.attack_time * sample_rate / 1000.0)).exp();
		let release_factor = 1.0 - (-1.0 / (self.release_time * sample_rate / 1000.0)).exp();

		let attack_factor = attack_factor.min(1.0);
		let release_factor = release_factor.min(1.0);

		for (i, val) in self.envlope_history.iter_mut().enumerate() {
			let is_attacking = input[i] > self.last_history[i];
			if is_attacking {
				*val = attack_factor * input[i].abs() + (1.0 - attack_factor) * *val;
			}else {
				*val *= 1.0 - release_factor;
			}
			self.last_history[i] = input[i].abs();
		}
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"Peak Detector"
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, _: String) {
		use crate::tools::ui_tools::gain_ui;

		ui.add(egui::Slider::new(&mut self.attack_time, 0.0..=1000.0)
			.text("Attack Time (ms)")
			.max_decimals(4)
		);
		ui.add(egui::Slider::new(&mut self.release_time, 0.0..=1000.0)
			.text("Release Time (ms)")
			.max_decimals(4)
		);
		gain_ui(ui, &mut self.gain_factor, None, false);
	}
}


/// Teager-Kaiser Energy Operator Envelope
#[derive(Parameters)]
pub struct TkeoEnvelope<const CHANNELS: usize = 2> {
	#[skip]
	data_history: [[f32; CHANNELS]; 3],
	#[skip]
	envlope_history: [f32; CHANNELS],
	/// The gain factor of the envelope, saves in linear scale.
	#[range(min = 0.01, max = 4.0)]
	#[logarithmic]
	pub gain_factor: f32,
}

impl<const CHANNELS: usize> Default for TkeoEnvelope<CHANNELS> {
	fn default() -> Self {
		Self::new()
	}
}

impl<const CHANNELS: usize> TkeoEnvelope<CHANNELS> {
	/// Create a new Tkeo Envelope
	/// 
	/// Panics if `CHANNELS` is 0;
	pub const fn new() -> Self {
		assert!(CHANNELS > 0, "CHANNELS must be greater than 0");
		let data_history = [[0.0; CHANNELS]; 3];
		let envlope_history = [0.0; CHANNELS];
		Self {
			data_history,
			envlope_history,
			gain_factor: 1.0,
		}
	}
}

impl<const CHANNELS: usize> Enveloper<CHANNELS> for TkeoEnvelope<CHANNELS> {
	fn delay(&self) -> usize {
		1
	}

	fn get_current_envelope(&self) -> [f32; CHANNELS] {
		let mut out = self.envlope_history;
		for val in out.iter_mut() {
			*val *= self.gain_factor
		}
		out
	}

	fn input_value(&mut self, input: &mut [f32; CHANNELS]) {
		self.data_history[0] = self.data_history[1];
		self.data_history[1] = self.data_history[2];
		self.data_history[2] = *input;
		
		*input = self.data_history[1];

		for (i, val) in self.envlope_history.iter_mut().enumerate() {
			*val = (self.data_history[1][i] * self.data_history[1][i] - self.data_history[0][i] * self.data_history[2][i]).abs().sqrt();
		}
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"Tkeo Envelope"
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, _: String) {
		use crate::tools::ui_tools::gain_ui;

		gain_ui(ui, &mut self.gain_factor, None, false);
	}
}

/// Get the envelope by using a FIR Hilbert Transform
#[derive(Parameters)]
pub struct FIRHilbertEnvelope<const CHANNELS: usize> {
	#[sub_param]
	convolver: Convolver<CHANNELS>,
	#[skip]
	ir_len: usize,
	#[skip]
	output_envlope: [f32; CHANNELS],
	/// The gain factor of the envelope, saves in linear scale.
	#[range(min = 0.01, max = 4.0)]
	#[logarithmic]
	pub gain_factor: f32,
}

impl<const CHANNELS: usize> FIRHilbertEnvelope<CHANNELS> {
	/// Create a new FIR Hilbert Envelope
	/// 
	/// Panic if `ir_len` is not odd or `CHANNELS` is zero.
	pub fn new(ir_len: usize) -> Self {
		assert!(CHANNELS > 0 && ir_len % 2 == 1);
		let convolver = Convolver::new(
			hilbert_transform(ir_len),
			&crate::effects::prelude::DelyaCaculateMode::Fir,
		);
		Self {
			convolver,
			ir_len,
			output_envlope: [0.0; CHANNELS],
			gain_factor: 1.0,
		}
	}

	/// Set the impulse response length.
	pub fn set_ir_len(&mut self, ir_len: usize) {
		assert!(ir_len % 2 == 1);
		self.ir_len = ir_len;
		self.convolver.replace_ir(
			hilbert_transform(ir_len), 
			&crate::effects::prelude::DelyaCaculateMode::Fir,
		);
	}
}

impl<const CHANNELS: usize> Enveloper<CHANNELS> for FIRHilbertEnvelope<CHANNELS> {
	fn delay(&self) -> usize {
		self.convolver.delay()
	}

	fn get_current_envelope(&self) -> [f32; CHANNELS] {
		let mut out = self.output_envlope;
		for val in out.iter_mut() {
			*val *= self.gain_factor
		}
		out
	}

	fn input_value(&mut self, input: &mut [f32; CHANNELS]) {
		let mut imag_part = *input;
		let mut ctx = Box::new(()) as Box<dyn ProcessContext>;
		self.convolver.process(&mut imag_part, &[], &mut ctx);
		let history = self.convolver.get_history();
		for (i, val) in imag_part.into_iter().enumerate() {
			let real_part = history[i][(self.convolver.delay() - 1) / 2];
			self.output_envlope[i] = real_part.hypot(val)
		}
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"FIR Hilbert Envelope"
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, _: String) {
		use crate::tools::ui_tools::gain_ui;

		let mut ir_len = self.ir_len;
		ui.add(egui::Slider::new(&mut ir_len, 15..=1023)
			.text("IR Length (samples)"));

		gain_ui(ui, &mut self.gain_factor, None, false);

		if ir_len != self.ir_len {
			if ir_len.is_multiple_of(2) {
				ir_len += 1;
			}
			self.set_ir_len(ir_len);
		}
	}
}