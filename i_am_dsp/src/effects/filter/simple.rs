use super::*;

/// A simple low pass filter based on [`Biquad`]
#[derive(Parameters)]
pub struct Lowpass<const CHANNELS: usize> {
	#[range(min = 10.0, max = 21000.0)]
	#[logarithmic]
	/// The cutoff of the lowpass filter
	#[unit(hertz)]
	#[id(display = "Cutoff")]
	#[layout(group = "Filter", order = 0, widget = knob, size = large, hero = "spectrum")]
	pub cutoff: f32,
	#[range(min = 0.01, max = 10.0)]
	/// The Q value of the lowpass filter
	#[id(display = "Resonance")]
	#[layout(group = "Filter", order = 1, widget = knob)]
	pub q: f32,
	#[sub_param]
	filter: Biquad<CHANNELS>,
	/// The values the filter coefficients were computed from, so an effect
	/// that re-applies its settings every sample does not recompute
	/// trigonometric functions.
	#[skip]
	cached: Option<(f32, f32, f32)>,
}

impl<const CHANNELS: usize> Lowpass<CHANNELS> {
	/// Creates a new lowpass filter with the given sample rate, cutoff, and Q value
	pub fn new(sample_rate: usize, cutoff: f32, q: f32) -> Self {
		Self {
			cutoff,
			q,
			filter: Biquad::lowpass(sample_rate, cutoff, q),
			cached: None,
		}
	}

	/// Clears the filter's stored input and output samples.
	pub fn clear_state(&mut self) {
		self.filter.clear_state();
	}
}

impl<const CHANNELS: usize> Effect<CHANNELS> for Lowpass<CHANNELS> {
	fn delay(&self) -> usize {
		0
	}

	fn clear_tail(&mut self) {
		self.clear_state();
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"Lowpass"
	}

	fn process(
		&mut self,
		samples: &mut [f32; CHANNELS],
		other: &[&[f32; CHANNELS]],
		process_context: &mut Box<dyn ProcessContext>,
	) {
		if self.cached != Some((self.cutoff, self.q, 0.0)) {
			self.filter.set_to_lowpass(self.cutoff, self.q);
			self.cached = Some((self.cutoff, self.q, 0.0));
		}
		self.filter.process(samples, other, process_context);
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		use crate::tools::ui_tools::draw_complex_response;
		use egui::Slider;

		egui::Resize::default()
			.resizable([false, true])
			.min_width(ui.available_width())
			.max_width(ui.available_width())
			.id_salt(format!("{id_prefix}_lowpass_resize"))
			.show(ui, |ui| {
				draw_complex_response(ui, self.filter.sample_rate, |freq| {
					self.filter.complex_response(freq)
				});
			});
		ui.horizontal(|ui| {
			ui.add(
				Slider::new(&mut self.cutoff, 10.0..=21000.0)
					.text("Cutoff")
					.logarithmic(true),
			);
			ui.add(Slider::new(&mut self.q, 0.01..=10.0).text("Q"));
		});
	}
}

/// A simple high pass filter based on [`Biquad`]
#[derive(Parameters)]
pub struct Highpass<const CHANNELS: usize> {
	#[range(min = 10.0, max = 21000.0)]
	#[logarithmic]
	/// The cutoff of the highpass filter
	#[unit(hertz)]
	pub cutoff: f32,
	#[range(min = 0.01, max = 10.0)]
	/// The Q value of the highpass filter
	pub q: f32,
	#[sub_param]
	filter: Biquad<CHANNELS>,
	/// The values the filter coefficients were computed from, so an effect
	/// that re-applies its settings every sample does not recompute
	/// trigonometric functions.
	#[skip]
	cached: Option<(f32, f32, f32)>,
}

impl<const CHANNELS: usize> Highpass<CHANNELS> {
	/// Creates a new highpass filter with the given sample rate, cutoff, and Q value
	pub fn new(sample_rate: usize, cutoff: f32, q: f32) -> Self {
		Self {
			cutoff,
			q,
			filter: Biquad::highpass(sample_rate, cutoff, q),
			cached: None,
		}
	}
}

impl<const CHANNELS: usize> Effect<CHANNELS> for Highpass<CHANNELS> {
	fn delay(&self) -> usize {
		0
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"Highpass"
	}

	fn process(
		&mut self,
		samples: &mut [f32; CHANNELS],
		other: &[&[f32; CHANNELS]],
		process_context: &mut Box<dyn ProcessContext>,
	) {
		if self.cached != Some((self.cutoff, self.q, 0.0)) {
			self.filter.set_to_highpass(self.cutoff, self.q);
			self.cached = Some((self.cutoff, self.q, 0.0));
		}
		self.filter.process(samples, other, process_context);
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		use crate::tools::ui_tools::draw_complex_response;
		use egui::Slider;

		egui::Resize::default()
			.resizable([false, true])
			.min_width(ui.available_width())
			.max_width(ui.available_width())
			.id_salt(format!("{id_prefix}_lowpass_resize"))
			.show(ui, |ui| {
				draw_complex_response(ui, self.filter.sample_rate, |freq| {
					self.filter.complex_response(freq)
				});
			});
		ui.horizontal(|ui| {
			ui.add(
				Slider::new(&mut self.cutoff, 10.0..=21000.0)
					.text("Cutoff")
					.logarithmic(true),
			);
			ui.add(Slider::new(&mut self.q, 0.01..=10.0).text("Q"));
		});
	}
}

/// A simple bandpass filter based on [`Biquad`]
#[derive(Parameters)]
pub struct Bandpass<const CHANNELS: usize> {
	#[range(min = 10.0, max = 21000.0)]
	#[logarithmic]
	/// The cutoff of the bandpass filter
	#[unit(hertz)]
	pub cutoff: f32,
	#[range(min = 10.0, max = 10000.0)]
	#[logarithmic]
	/// The bandwidth value of the bandpass filter
	#[unit(hertz)]
	pub bandwidth: f32,
	#[sub_param]
	filter: Biquad<CHANNELS>,
	/// The values the filter coefficients were computed from, so an effect
	/// that re-applies its settings every sample does not recompute
	/// trigonometric functions.
	#[skip]
	cached: Option<(f32, f32, f32)>,
}

impl<const CHANNELS: usize> Bandpass<CHANNELS> {
	/// Creates a new bandpass filter with the given sample rate, cutoff, and bandwidth value
	pub fn new(sample_rate: usize, cutoff: f32, bandwidth: f32) -> Self {
		Self {
			cutoff,
			bandwidth,
			filter: Biquad::bandpass(sample_rate, cutoff, bandwidth),
			cached: None,
		}
	}
}

impl<const CHANNELS: usize> Effect<CHANNELS> for Bandpass<CHANNELS> {
	fn delay(&self) -> usize {
		0
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"Bandpass"
	}

	fn process(
		&mut self,
		samples: &mut [f32; CHANNELS],
		other: &[&[f32; CHANNELS]],
		process_context: &mut Box<dyn ProcessContext>,
	) {
		if self.cached != Some((self.cutoff, self.bandwidth, 0.0)) {
			self.filter.set_to_bandpass(self.cutoff, self.bandwidth);
			self.cached = Some((self.cutoff, self.bandwidth, 0.0));
		}
		self.filter.process(samples, other, process_context);
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		use crate::tools::ui_tools::draw_complex_response;
		use egui::Slider;

		egui::Resize::default()
			.resizable([false, true])
			.min_width(ui.available_width())
			.max_width(ui.available_width())
			.id_salt(format!("{id_prefix}_lowpass_resize"))
			.show(ui, |ui| {
				draw_complex_response(ui, self.filter.sample_rate, |freq| {
					self.filter.complex_response(freq)
				});
			});
		ui.horizontal(|ui| {
			ui.add(
				Slider::new(&mut self.cutoff, 10.0..=21000.0)
					.text("Cutoff")
					.logarithmic(true),
			);
			ui.add(
				Slider::new(&mut self.bandwidth, 10.0..=10000.0)
					.text("Bandwidth")
					.logarithmic(true),
			);
		});
	}
}

/// A simple bandstop filter based on [`Biquad`]
#[derive(Parameters)]
pub struct Bandstop<const CHANNELS: usize> {
	#[range(min = 10.0, max = 21000.0)]
	#[logarithmic]
	/// The cutoff of the bandstop filter
	#[unit(hertz)]
	pub cutoff: f32,
	#[range(min = 10.0, max = 10000.0)]
	#[logarithmic]
	/// The bandwidth value of the bandstop filter
	#[unit(hertz)]
	pub bandwidth: f32,
	#[sub_param]
	filter: Biquad<CHANNELS>,
}

impl<const CHANNELS: usize> Bandstop<CHANNELS> {
	/// Creates a new bandstop filter with the given sample rate, cutoff, and bandwidth value
	pub fn new(sample_rate: usize, cutoff: f32, bandwidth: f32) -> Self {
		Self {
			cutoff,
			bandwidth,
			filter: Biquad::bandstop(sample_rate, cutoff, bandwidth),
		}
	}
}

impl<const CHANNELS: usize> Effect<CHANNELS> for Bandstop<CHANNELS> {
	fn delay(&self) -> usize {
		0
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"bandstop"
	}

	fn process(
		&mut self,
		samples: &mut [f32; CHANNELS],
		other: &[&[f32; CHANNELS]],
		process_context: &mut Box<dyn ProcessContext>,
	) {
		self.filter.set_to_bandstop(self.cutoff, self.bandwidth);
		self.filter.process(samples, other, process_context);
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		use crate::tools::ui_tools::draw_complex_response;
		use egui::Slider;

		egui::Resize::default()
			.resizable([false, true])
			.min_width(ui.available_width())
			.max_width(ui.available_width())
			.id_salt(format!("{id_prefix}_lowpass_resize"))
			.show(ui, |ui| {
				draw_complex_response(ui, self.filter.sample_rate, |freq| {
					self.filter.complex_response(freq)
				});
			});
		ui.horizontal(|ui| {
			ui.add(
				Slider::new(&mut self.cutoff, 10.0..=21000.0)
					.text("Cutoff")
					.logarithmic(true),
			);
			ui.add(
				Slider::new(&mut self.bandwidth, 10.0..=10000.0)
					.text("Bandwidth")
					.logarithmic(true),
			);
		});
	}
}

/// A simple peaking filter based on [`Biquad`]
#[derive(Parameters)]
pub struct Peak<const CHANNELS: usize> {
	#[range(min = 10.0, max = 21000.0)]
	#[logarithmic]
	/// The cutoff of the peaking filter
	#[unit(hertz)]
	pub cutoff: f32,
	#[range(min = 0.0, max = 4.0)]
	#[logarithmic]
	/// The gain of the peaking filter
	pub gain: f32,
	#[range(min = 10.0, max = 10000.0)]
	#[logarithmic]
	/// The bandwidth of the peaking filter
	#[unit(hertz)]
	pub bandwidth: f32,
	#[sub_param]
	filter: Biquad<CHANNELS>,
	/// The values the filter coefficients were computed from, so an effect
	/// that re-applies its settings every sample does not recompute
	/// trigonometric functions.
	#[skip]
	cached: Option<(f32, f32, f32)>,
}

impl<const CHANNELS: usize> Peak<CHANNELS> {
	/// Creates a new peaking filter with the given sample rate, cutoff, gain, and bandwidth value
	pub fn new(sample_rate: usize, cutoff: f32, gain: f32, bandwidth: f32) -> Self {
		Self {
			cutoff,
			gain,
			bandwidth,
			filter: Biquad::peak(sample_rate, cutoff, gain, bandwidth),
			cached: None,
		}
	}
}

impl<const CHANNELS: usize> Effect<CHANNELS> for Peak<CHANNELS> {
	fn delay(&self) -> usize {
		0
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"Peak"
	}

	fn process(
		&mut self,
		samples: &mut [f32; CHANNELS],
		other: &[&[f32; CHANNELS]],
		process_context: &mut Box<dyn ProcessContext>,
	) {
		if self.cached != Some((self.cutoff, self.gain, self.bandwidth)) {
			self.filter
				.set_to_peak(self.cutoff, self.gain, self.bandwidth);
			self.cached = Some((self.cutoff, self.gain, self.bandwidth));
		}
		self.filter.process(samples, other, process_context);
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		use crate::tools::ui_tools::draw_complex_response;
		use egui::Slider;

		egui::Resize::default()
			.resizable([false, true])
			.min_width(ui.available_width())
			.max_width(ui.available_width())
			.id_salt(format!("{id_prefix}_lowpass_resize"))
			.show(ui, |ui| {
				draw_complex_response(ui, self.filter.sample_rate, |freq| {
					self.filter.complex_response(freq)
				});
			});
		ui.horizontal(|ui| {
			ui.add(
				Slider::new(&mut self.cutoff, 10.0..=21000.0)
					.text("Cutoff")
					.logarithmic(true),
			);
			ui.add(
				Slider::new(&mut self.gain, 0.01..=4.0)
					.text("Gain")
					.logarithmic(true),
			);
			ui.add(
				Slider::new(&mut self.bandwidth, 10.0..=10000.0)
					.text("Bandwidth")
					.logarithmic(true),
			);
		});
	}
}

/// A simple high shelf filter based on [`Biquad`]
#[derive(Parameters)]
pub struct HighShelf<const CHANNELS: usize> {
	#[range(min = 10.0, max = 21000.0)]
	#[logarithmic]
	/// The cutoff of the high shelf filter
	#[unit(hertz)]
	pub cutoff: f32,
	#[range(min = 0.0, max = 4.0)]
	#[logarithmic]
	/// The gain of the high shelf filter in linear scale
	pub gain: f32,
	#[range(min = 0.5, max = 2.0)]
	/// The slope of the high shelf filter
	pub slope: f32,
	#[sub_param]
	filter: Biquad<CHANNELS>,
	/// The values the filter coefficients were computed from, so an effect
	/// that re-applies its settings every sample does not recompute
	/// trigonometric functions.
	#[skip]
	cached: Option<(f32, f32, f32)>,
}

impl<const CHANNELS: usize> HighShelf<CHANNELS> {
	/// Creates a new high shelf filter with the given sample rate, cutoff, gain, and slope value
	pub fn new(sample_rate: usize, cutoff: f32, gain: f32, slope: f32) -> Self {
		Self {
			cutoff,
			gain,
			slope,
			filter: Biquad::high_shelf(sample_rate, cutoff, gain, slope),
			cached: None,
		}
	}
}

impl<const CHANNELS: usize> Effect<CHANNELS> for HighShelf<CHANNELS> {
	fn delay(&self) -> usize {
		0
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"HighShelf"
	}

	fn process(
		&mut self,
		samples: &mut [f32; CHANNELS],
		other: &[&[f32; CHANNELS]],
		process_context: &mut Box<dyn ProcessContext>,
	) {
		if self.cached != Some((self.cutoff, self.gain, self.slope)) {
			self.filter
				.set_to_high_shelf(self.cutoff, self.gain, self.slope);
			self.cached = Some((self.cutoff, self.gain, self.slope));
		}
		self.filter.process(samples, other, process_context);
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		use crate::tools::ui_tools::draw_complex_response;
		use egui::Slider;

		egui::Resize::default()
			.resizable([false, true])
			.min_width(ui.available_width())
			.max_width(ui.available_width())
			.id_salt(format!("{id_prefix}_lowpass_resize"))
			.show(ui, |ui| {
				draw_complex_response(ui, self.filter.sample_rate, |freq| {
					self.filter.complex_response(freq)
				});
			});
		ui.horizontal(|ui| {
			ui.add(
				Slider::new(&mut self.cutoff, 10.0..=21000.0)
					.text("Cutoff")
					.logarithmic(true),
			);
			ui.add(
				Slider::new(&mut self.gain, 0.01..=4.0)
					.text("Gain")
					.logarithmic(true),
			);
			ui.add(Slider::new(&mut self.slope, 0.5..=2.0).text("Slope"));
		});
	}
}

/// A simple low shelf filter based on [`Biquad`]
#[derive(Parameters)]
pub struct LowShelf<const CHANNELS: usize> {
	#[range(min = 10.0, max = 21000.0)]
	#[logarithmic]
	/// The cutoff of the low shelf filter
	#[unit(hertz)]
	pub cutoff: f32,
	#[range(min = 0.0, max = 4.0)]
	#[logarithmic]
	/// The gain of the low shelf filter in linear scale
	pub gain: f32,
	#[range(min = 0.5, max = 2.0)]
	/// The slope of the low shelf filter
	pub slope: f32,
	#[sub_param]
	filter: Biquad<CHANNELS>,
	/// The values the filter coefficients were computed from, so an effect
	/// that re-applies its settings every sample does not recompute
	/// trigonometric functions.
	#[skip]
	cached: Option<(f32, f32, f32)>,
}

impl<const CHANNELS: usize> LowShelf<CHANNELS> {
	/// Creates a new low shelf filter with the given sample rate, cutoff, gain, and slope value
	pub fn new(sample_rate: usize, cutoff: f32, gain: f32, slope: f32) -> Self {
		Self {
			cutoff,
			gain,
			slope,
			filter: Biquad::low_shelf(sample_rate, cutoff, gain, slope),
			cached: None,
		}
	}
}

impl<const CHANNELS: usize> Effect<CHANNELS> for LowShelf<CHANNELS> {
	fn delay(&self) -> usize {
		0
	}

	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"LowShelf"
	}

	fn process(
		&mut self,
		samples: &mut [f32; CHANNELS],
		other: &[&[f32; CHANNELS]],
		process_context: &mut Box<dyn ProcessContext>,
	) {
		if self.cached != Some((self.cutoff, self.gain, self.slope)) {
			self.filter
				.set_to_low_shelf(self.cutoff, self.gain, self.slope);
			self.cached = Some((self.cutoff, self.gain, self.slope));
		}
		self.filter.process(samples, other, process_context);
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		use crate::tools::ui_tools::draw_complex_response;
		use egui::Slider;

		egui::Resize::default()
			.resizable([false, true])
			.min_width(ui.available_width())
			.max_width(ui.available_width())
			.id_salt(format!("{id_prefix}_lowpass_resize"))
			.show(ui, |ui| {
				draw_complex_response(ui, self.filter.sample_rate, |freq| {
					self.filter.complex_response(freq)
				});
			});
		ui.horizontal(|ui| {
			ui.add(
				Slider::new(&mut self.cutoff, 10.0..=21000.0)
					.text("Cutoff")
					.logarithmic(true),
			);
			ui.add(
				Slider::new(&mut self.gain, 0.01..=4.0)
					.text("Gain")
					.logarithmic(true),
			);
			ui.add(Slider::new(&mut self.slope, 0.5..=2.0).text("Slope"));
		});
	}
}
