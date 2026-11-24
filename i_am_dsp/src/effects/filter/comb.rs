use super::*;

/// A classic combs filter
#[derive(Parameters)]
pub struct CombsFilter<const CHANNELS: usize> {
	#[range(min = 0.0, max = 1.0)]
	#[logarithmic]
	/// The feedback of the combs filter
	pub feedback: f32,
	#[skip]
	history: [RingBuffer<f32>; CHANNELS],
}

impl<const CHANNELS: usize> CombsFilter<CHANNELS> {
	/// Creates a new combs filter with the given sample rate and feedback value
	pub fn new(feedback: f32, delay_samples: usize) -> Self {
		Self {
			feedback,
			history: core::array::from_fn(|_| RingBuffer::new(delay_samples)),
		}
	}

	/// Returns the feedback delay in samples of the combs filter
	pub fn delay_samples(&self) -> usize {
		self.history[0].capacity()
	}

	/// Resizes the history buffer of the combs filter
	pub fn resize_history(&mut self, new_size: usize) {
		for buffer in self.history.iter_mut() {
			buffer.resize(new_size);
		}
	}

	/// Clears the comb's delay line, stopping its tail.
	pub fn clear_history(&mut self) {
		for buffer in self.history.iter_mut() {
			buffer.clear();
		}
	}
}

impl<const CHANNELS: usize> Effect<CHANNELS> for CombsFilter<CHANNELS> {
	fn delay(&self) -> usize {
		0
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"Combs Filter"
	}

	fn process(
		&mut self,
		samples: &mut [f32; CHANNELS],
		_: &[&[f32; CHANNELS]],
		_: &mut Box<dyn ProcessContext>,
	) {
		for (i, sample) in samples.iter_mut().enumerate() {
			*sample += self.feedback * self.history[i][0];
			// A feedback loop walks through the subnormal range on the way to
			// silence, where arithmetic is an order of magnitude slower.
			if sample.abs() <= DENORMAL_THRESHOLD {
				*sample = 0.0;
			}
			self.history[i].push(*sample);
		}
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, _: String) {
		use egui::Slider;

		ui.horizontal(|ui| {
			ui.add(
				Slider::new(&mut self.feedback, 0.01..=0.99)
					.text("Feedback")
					.logarithmic(true),
			);
			let mut delay_samples = self.delay_samples();
			ui.add(
				Slider::new(&mut delay_samples, 16..=1024)
					.logarithmic(true)
					.text("Delay (samples)"),
			);
			if delay_samples != self.delay_samples() {
				self.resize_history(delay_samples);
			}
		});
	}
}

/// A classic allpass filter
#[derive(Parameters)]
pub struct AllpassFilter<const CHANNELS: usize> {
	#[range(min = 0.0, max = 1.0)]
	#[logarithmic]
	/// The feedback factor of the allpass filter
	pub feedback: f32,
	#[skip]
	processed_history: [RingBuffer<f32>; CHANNELS],
	#[skip]
	unprocessed_history: [RingBuffer<f32>; CHANNELS],
}

impl<const CHANNELS: usize> AllpassFilter<CHANNELS> {
	/// Creates a new allpass filter with the given sample rate and feedback value
	pub fn new(feedback: f32, delay_samples: usize) -> Self {
		assert!(CHANNELS > 0, "CHANNELS must be greater than 0");

		Self {
			feedback,
			processed_history: core::array::from_fn(|_| RingBuffer::new(delay_samples)),
			unprocessed_history: core::array::from_fn(|_| RingBuffer::new(delay_samples)),
		}
	}

	/// Returns the feedback delay in samples of the allpass filter
	pub fn delay_samples(&self) -> usize {
		self.processed_history[0].capacity()
	}

	/// Resizes the history buffer of the allpass filter
	pub fn resize_history(&mut self, new_size: usize) {
		for buffer in self.processed_history.iter_mut() {
			buffer.resize(new_size);
		}
		for buffer in self.unprocessed_history.iter_mut() {
			buffer.resize(new_size);
		}
	}

	/// Clears both allpass delay lines, stopping its tail.
	pub fn clear_history(&mut self) {
		for buffer in self.processed_history.iter_mut() {
			buffer.clear();
		}
		for buffer in self.unprocessed_history.iter_mut() {
			buffer.clear();
		}
	}
}

impl<const CHANNELS: usize> Effect<CHANNELS> for AllpassFilter<CHANNELS> {
	fn delay(&self) -> usize {
		0
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"Allpass Filter"
	}

	fn process(
		&mut self,
		samples: &mut [f32; CHANNELS],
		_: &[&[f32; CHANNELS]],
		_: &mut Box<dyn ProcessContext>,
	) {
		for (i, sample) in samples.iter_mut().enumerate() {
			let processed = self.processed_history[i][0];
			let unprocessed = self.unprocessed_history[i][0];
			let sample_backup = *sample;
			*sample = -*sample * self.feedback + unprocessed + self.feedback * processed;
			// See the combs filter: keep the loop out of the subnormal range.
			if sample.abs() <= DENORMAL_THRESHOLD {
				*sample = 0.0;
			}
			self.processed_history[i].push(*sample);
			self.unprocessed_history[i].push(sample_backup);
		}
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, _: String) {
		use egui::Slider;

		ui.horizontal(|ui| {
			ui.add(
				Slider::new(&mut self.feedback, 0.01..=0.99)
					.text("Feedback")
					.logarithmic(true),
			);
			let mut delay_samples = self.delay_samples();
			ui.add(
				Slider::new(&mut delay_samples, 16..=1024)
					.logarithmic(true)
					.text("Delay (samples)"),
			);
			if delay_samples != self.delay_samples() {
				self.resize_history(delay_samples);
			}
		});
	}
}
