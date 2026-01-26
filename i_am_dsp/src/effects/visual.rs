//! Visual-related effects.

use std::array;

use i_am_dsp_derive::Parameters;

use crate::{tools::ring_buffer::RingBuffer, Effect};

/// A simple waveform effect that displays the input signal in a ring buffer.
#[derive(Parameters)]
pub struct Waveform<const CHANNELS: usize = 2> {
	#[skip]
	buffer: [RingBuffer<f32>; CHANNELS],
	/// Whether the waveform is frozen or not.
	pub frozen: bool,
}
impl<const CHANNELS: usize> Waveform<CHANNELS> {
	/// Creates a new `Waveform` effect with the given buffer capacity.
	/// 
	/// # Panics
	/// 
	/// Panics if `CHANNELS` is less than or equal to 0.
	pub fn new(capacity: usize) -> Self {
		assert!(CHANNELS > 0, "CHANNELS must be greater than 0");

		Self {
			buffer: array::from_fn(|_| RingBuffer::new(capacity)),
			frozen: false,
		}
	}

	/// Resizes the buffer capacity of the history buffer.
	pub fn resize(&mut self, capacity: usize) {
		for i in 0..CHANNELS {
			self.buffer[i].resize(capacity);
		}
	}
}

impl<const CHANNELS: usize> Effect for Waveform<CHANNELS> {
	fn delay(&self) -> usize {
		0
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"Waveform"
	}

	fn process(
		&mut self, 
		samples: &mut [f32; 2], 
		_: &[&[f32; 2]],
		_: &mut Box<dyn crate::ProcessContext>,
	) {
		if self.frozen {
			return;
		}

		for (i, sample) in samples.iter().enumerate() {
			self.buffer[i].push(*sample);
		}
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		use egui::*;
		use crate::tools::ui_tools::draw_envelope;

		Resize::default()
			.min_width(ui.available_width())
			.max_width(ui.available_width())
			.id_salt(format!("{}_waveform_resize", id_prefix))
			.show(ui, |ui| {
				let env = self.buffer.iter().collect::<Vec<_>>();
				if draw_envelope(ui, &env, true).clicked() &&
					ui.input(|input| input.modifiers.ctrl) 
				{
					self.frozen = !self.frozen;
				}
			});

		ui.horizontal(|ui| {
			let mut len = self.buffer[0].capacity();
			ui.add(Slider::new(&mut len, 1024..=65536).text("Capacity"));
			if len != self.buffer[0].capacity() {
				self.resize(len);
			}
			if ui.selectable_label(self.frozen, "Freeze").clicked() {
				self.frozen = !self.frozen;
			}
		});
	}
}
