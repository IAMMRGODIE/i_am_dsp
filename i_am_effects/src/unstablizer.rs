//! The note-anchored spectral shifter with a small product surface.

use i_am_dsp::prelude::{Effect, ProcessContext, Unstablizer, overlap_add_gain};
use i_am_dsp_derive::Parameters;

#[derive(Parameters)]
pub struct UnstablizerEffect {
	/// The A4 reference pitch.
	#[range(min = 400.0, max = 480.0)]
	#[unit(hertz)]
	#[default(440.0)]
	#[id(display = "Reference")]
	#[layout(group = "Unstablizer", order = 0, widget = knob)]
	pub reference_pitch: f32,

	/// The transpose of the anchor, in semitones.
	#[range(min = -24.0, max = 24.0)]
	#[default(0.0)]
	#[id(display = "Transpose")]
	#[layout(group = "Unstablizer", order = 1, widget = knob, hero = "spectra")]
	pub transpose: f32,

	/// How far the partials move around the anchor, as a ratio.
	#[range(min = 0.25, max = 4.0)]
	#[logarithmic]
	#[default(1.0)]
	#[unit(percent)]
	#[id(display = "Shift")]
	#[layout(group = "Unstablizer", order = 2, widget = knob)]
	pub shift: f32,

	/// The manual anchor when Follow Notes is off.
	#[range(min = 20.0, max = 20_000.0)]
	#[logarithmic]
	#[unit(hertz)]
	#[default(440.0)]
	#[id(display = "Anchor")]
	#[layout(group = "Unstablizer", order = 3, widget = knob)]
	pub anchor: f32,

	/// The phase-vocoder window shape.
	#[range(min = 0.0, max = 1.0)]
	#[default(0.6)]
	#[id(display = "Window")]
	#[layout(group = "Unstablizer", order = 4, widget = knob)]
	pub window_factor: f32,

	/// The window size, picked from the header.
	#[choices("512", "1024", "2048", "4096")]
	#[default(2)]
	#[id(display = "Window Size")]
	#[layout(group = "Unstablizer", order = 5, widget = choice)]
	pub window_size: i32,

	/// Whether played notes set the anchor.
	#[default(true)]
	#[id(display = "Follow")]
	#[layout(group = "Unstablizer", order = 6, widget = toggle)]
	pub follow_notes: bool,

	#[skip]
	inner: Unstablizer<2>,
	#[skip]
	last_window_factor: f32,
	#[skip]
	last_window_size: i32,
}

impl UnstablizerEffect {
	pub fn new(sample_rate: usize) -> Self {
		Self {
			reference_pitch: 440.0,
			transpose: 0.0,
			shift: 1.0,
			anchor: 440.0,
			window_factor: 0.6,
			window_size: 2,
			follow_notes: true,
			inner: Unstablizer::new(sample_rate),
			last_window_factor: 0.6,
			last_window_size: 2,
		}
	}

	fn compensate_overlap_gain(&mut self) {
		self.inner.vocoder.gain = 1.0 / overlap_add_gain(self.inner.vocoder.window_factor);
	}
}

impl Effect<2> for UnstablizerEffect {
	fn delay(&self) -> usize {
		self.inner.delay()
	}

	fn process(
		&mut self,
		samples: &mut [f32; 2],
		other: &[&[f32; 2]],
		context: &mut Box<dyn ProcessContext>,
	) {
		if (self.window_factor - self.last_window_factor).abs() > 1e-4 {
			self.inner.vocoder.window_factor = self.window_factor.clamp(0.0, 1.0);
			self.compensate_overlap_gain();
			self.last_window_factor = self.window_factor;
		}

		if self.window_size != self.last_window_size {
			let size = [512, 1024, 2048, 4096][self.window_size.clamp(0, 3) as usize];
			if self.inner.vocoder.renew_window_size(size).is_some() {
				self.compensate_overlap_gain();
			}
			self.last_window_size = self.window_size;
		}

		self.inner.reference_pitch = self.reference_pitch;
		self.inner.transpose = self.transpose;
		self.inner.vocoder.mapper.shift = self.shift;
		self.inner.follow_notes = self.follow_notes;

		// Follow Notes only overrides the anchor when it actually has a note.
		// Setting the manual anchor first makes it a fallback when no note has
		// arrived, instead of leaving the mapper at a stale/default frequency.
		self.inner.vocoder.mapper.anchor = self.anchor;

		self.inner.process(samples, other, context);
	}
}
