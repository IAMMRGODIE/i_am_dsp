use super::*;

impl<const CHANNELS: usize> Effect<CHANNELS> for Biquad<CHANNELS> {
	#[cfg(feature = "real_time_demo")]
	fn name(&self) -> &str {
		"Biquad Filter"
	}

	fn delay(&self) -> usize {
		0
	}

	fn process(
		&mut self,
		samples: &mut [f32; CHANNELS],
		_: &[&[f32; CHANNELS]],
		_: &mut Box<dyn ProcessContext>,
	) {
		for i in (0..CHANNELS).step_by(4) {
			let sample = f32x4::from(&samples[i..(i + 4).min(CHANNELS)]);
			let x_0 = f32x4::from(&self.x[0][i..(i + 4).min(CHANNELS)]);
			let x_1 = f32x4::from(&self.x[1][i..(i + 4).min(CHANNELS)]);
			let y_0 = f32x4::from(&self.y[0][i..(i + 4).min(CHANNELS)]);
			let y_1 = f32x4::from(&self.y[1][i..(i + 4).min(CHANNELS)]);

			let output = self.b[0] * sample + self.b[1] * x_0 + self.b[2] * x_1
				- self.a[0] * y_0
				- self.a[1] * y_1;

			let output = output.to_array();
			for j in 0..4.min(CHANNELS - i) {
				self.x[1][i + j] = self.x[0][i + j];
				self.x[0][i + j] = samples[i + j];
				self.y[1][i + j] = self.y[0][i + j];
				self.y[0][i + j] = output[j];

				samples[i + j] = output[j];
			}
		}

		// // TODO: implement SIMD version of this filter
		// for (i, sample) in samples.iter_mut().enumerate() {
		// 	let output =
		// 		self.b[0] * *sample +
		// 		self.b[1] * self.x[0][i] +
		// 		self.b[2] * self.x[1][i] -
		// 		self.a[0] * self.y[0][i] -
		// 		self.a[1] * self.y[1][i];

		// 	let output = if output.is_nan() { 0.0 } else { output };

		// 	self.x[1][i] = self.x[0][i];
		// 	self.x[0][i] = *sample;
		// 	self.y[1][i] = self.y[0][i];
		// 	self.y[0][i] = output;

		// 	*sample = output;
		// }
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		use crate::tools::ui_tools::draw_complex_response;
		use egui::*;
		use std::ops::RangeInclusive;

		type ChangeFunc<const CHANNELS: usize> = Box<dyn Fn(&mut Biquad<CHANNELS>, f32)>;
		type ChangeFunc2<const CHANNELS: usize> = Box<dyn Fn(&mut Biquad<CHANNELS>, f32, f32)>;
		type ChangeFunc3<const CHANNELS: usize> = Box<dyn Fn(&mut Biquad<CHANNELS>, f32, f32, f32)>;

		fn cutoff_ui<const CHANNELS: usize>(
			ui: &mut egui::Ui,
			cutoff: &mut f32,
			max_freq: f32,
			change_func: ChangeFunc<CHANNELS>,
		) -> Option<(ChangeFunc<CHANNELS>, f32)> {
			let mut cutoff_backup = *cutoff;
			let slider = Slider::new(&mut cutoff_backup, 10.0..=max_freq - MIN_FREQUENCY)
				.text("Cutoff")
				.logarithmic(true);
			ui.add(slider);
			if cutoff_backup != *cutoff {
				*cutoff = cutoff_backup;
				Some((change_func, *cutoff))
			} else {
				None
			}
		}

		fn cutoff_q_ui<const CHANNELS: usize>(
			ui: &mut egui::Ui,
			cutoff: &mut f32,
			q: &mut f32,
			q_range: RangeInclusive<f32>,
			q_text: impl Into<String>,
			max_freq: f32,
			change_func: ChangeFunc2<CHANNELS>,
		) -> Option<(ChangeFunc2<CHANNELS>, f32, f32)> {
			let mut cutoff_backup = *cutoff;
			let slider = Slider::new(&mut cutoff_backup, 10.0..=max_freq - MIN_FREQUENCY)
				.text("Cutoff")
				.logarithmic(true);
			ui.add(slider);
			let mut q_backup = *q;
			let q_log = *q_range.end() > 1000.0;
			let q_slider = Slider::new(&mut q_backup, q_range)
				.text(q_text.into())
				.logarithmic(q_log);
			ui.add(q_slider);

			if cutoff_backup != *cutoff || q_backup != *q {
				*cutoff = cutoff_backup;
				*q = q_backup;
				Some((change_func, *cutoff, *q))
			} else {
				None
			}
		}

		fn simulate_ui<const CHANNELS: usize>(
			ui: &mut egui::Ui,
			params: (&mut f32, &mut f32, &mut f32),
			max_freq: f32,
			factor_range: RangeInclusive<f32>,
			mut texts: Vec<impl Into<String>>,
			change_func: ChangeFunc3<CHANNELS>,
		) -> Option<(ChangeFunc3<CHANNELS>, f32, f32, f32)> {
			let (cutoff, gain, factor) = params;

			let mut cutoff_backup = *cutoff;
			let mut gain_backup = *gain;
			let mut factor_backup = *factor;
			let factor_text = texts.pop().unwrap().into();
			let gain_text = texts.pop().unwrap().into();
			let cutoff_text = texts.pop().unwrap().into();

			let slider_cutoff = Slider::new(&mut cutoff_backup, 10.0..=max_freq - MIN_FREQUENCY)
				.text(cutoff_text)
				.logarithmic(true);
			let slider_gain = Slider::new(&mut gain_backup, -12.0..=6.0).text(gain_text);
			let factor_log = *factor_range.end() > 1000.0;
			let slider_factor = Slider::new(&mut factor_backup, factor_range)
				.text(factor_text)
				.logarithmic(factor_log);

			ui.add(slider_cutoff);
			ui.add(slider_gain);
			ui.add(slider_factor);

			if cutoff_backup != *cutoff || gain_backup != *gain || factor_backup != *factor {
				*cutoff = cutoff_backup;
				*gain = gain_backup;
				*factor = factor_backup;
				Some((change_func, *cutoff, *gain, *factor))
			} else {
				None
			}
		}

		egui::Resize::default()
			.resizable([false, true])
			.min_width(ui.available_width())
			.max_width(ui.available_width())
			.id_salt(format!("{id_prefix}_filter"))
			.show(ui, |ui| {
				draw_complex_response(ui, self.sample_rate, |freq| self.complex_response(freq));
			});

		ScrollArea::horizontal().show(ui, |ui| {
			ui.horizontal(|ui| {
				if ui
					.selectable_label(
						matches!(self.gui_state.filter_type, FilterType::ByPass),
						"Bypass",
					)
					.clicked()
				{
					self.gui_state.filter_type = FilterType::ByPass;
					self.b = [1.0, 0.0, 0.0];
					self.a = [0.0, 0.0];
				}
				if ui
					.selectable_label(
						matches!(self.gui_state.filter_type, FilterType::Butterworth(_)),
						"Butterworth",
					)
					.clicked()
				{
					self.gui_state.filter_type =
						FilterType::Butterworth(Butterworth::Lowpass { cutoff: 1000.0 });
					self.set_to_butterworth_lowpass(1000.0);
				}

				if ui
					.selectable_label(
						matches!(self.gui_state.filter_type, FilterType::Rbj(_)),
						"Rbj",
					)
					.clicked()
				{
					self.gui_state.filter_type = FilterType::Rbj(RBJFilter::Lowpass {
						cutoff: 1000.0,
						q: Self::Q1,
					});
					self.set_to_lowpass(1000.0, Self::Q1);
				}

				if ui
					.selectable_label(self.gui_state.allpass_mode, "Allpass")
					.clicked()
				{
					self.gui_state.allpass_mode = !self.gui_state.allpass_mode;
					if self.gui_state.allpass_mode {
						self.transform_to_allpass();
					} else {
						self.gui_state.filter_type = FilterType::ByPass;
						self.b = [1.0, 0.0, 0.0];
						self.a = [0.0, 0.0];
					}
				}
			});

			fn butterworth_gui<const CHANNELS: usize>(
				ui: &mut egui::Ui,
				butterworth: &mut Butterworth,
				sample_rate: usize,
			) -> Option<Biquad<CHANNELS>> {
				let mut output = None;
				if ui
					.selectable_label(
						matches!(butterworth, Butterworth::Lowpass { .. }),
						"Lowpass",
					)
					.clicked()
				{
					*butterworth = Butterworth::Lowpass { cutoff: 1000.0 };
					output = Some(
						Biquad::butterworth_lowpass(sample_rate, 1000.0, 1)
							.pop()
							.unwrap(),
					)
				}
				if ui
					.selectable_label(
						matches!(butterworth, Butterworth::Highpass { .. }),
						"Highpass",
					)
					.clicked()
				{
					*butterworth = Butterworth::Highpass { cutoff: 1000.0 };
					output = Some(
						Biquad::butterworth_highpass(sample_rate, 1000.0, 1)
							.pop()
							.unwrap(),
					)
				}
				if ui
					.selectable_label(
						matches!(butterworth, Butterworth::Bandpass { .. }),
						"Bandpass",
					)
					.clicked()
				{
					*butterworth = Butterworth::Bandpass { cutoff: 1000.0 };
					output = Some(
						Biquad::butterworth_bandpass(sample_rate, 1000.0, 1)
							.pop()
							.unwrap(),
					)
				}
				if ui
					.selectable_label(
						matches!(butterworth, Butterworth::Bandstop { .. }),
						"Bandstop",
					)
					.clicked()
				{
					*butterworth = Butterworth::Bandstop { cutoff: 1000.0 };
					output = Some(
						Biquad::butterworth_bandstop(sample_rate, 1000.0, 1)
							.pop()
							.unwrap(),
					)
				}

				output
			}

			fn rbj_gui<const CHANNELS: usize>(
				ui: &mut egui::Ui,
				rbj: &mut RBJFilter,
				sample_rate: usize,
			) -> Option<Biquad<CHANNELS>> {
				let mut output = None;
				if ui
					.selectable_label(matches!(rbj, RBJFilter::Lowpass { .. }), "Lowpass")
					.clicked()
				{
					*rbj = RBJFilter::Lowpass {
						cutoff: 1000.0,
						q: Biquad::<CHANNELS>::Q1,
					};
					output = Some(Biquad::lowpass(sample_rate, 1000.0, Biquad::<CHANNELS>::Q1))
				}
				if ui
					.selectable_label(matches!(rbj, RBJFilter::Highpass { .. }), "Highpass")
					.clicked()
				{
					*rbj = RBJFilter::Highpass {
						cutoff: 1000.0,
						q: Biquad::<CHANNELS>::Q1,
					};
					output = Some(Biquad::highpass(
						sample_rate,
						1000.0,
						Biquad::<CHANNELS>::Q1,
					))
				}
				if ui
					.selectable_label(matches!(rbj, RBJFilter::Bandpass { .. }), "Bandpass")
					.clicked()
				{
					*rbj = RBJFilter::Bandpass {
						cutoff: 1000.0,
						q: Biquad::<CHANNELS>::Q1,
					};
					output = Some(Biquad::bandpass(sample_rate, 1000.0, 100.0))
				}
				if ui
					.selectable_label(matches!(rbj, RBJFilter::Bandstop { .. }), "Bandstop")
					.clicked()
				{
					*rbj = RBJFilter::Bandstop {
						cutoff: 1000.0,
						q: Biquad::<CHANNELS>::Q1,
					};
					output = Some(Biquad::bandstop(sample_rate, 1000.0, 100.0))
				}
				if ui
					.selectable_label(matches!(rbj, RBJFilter::Peak { .. }), "Peak")
					.clicked()
				{
					*rbj = RBJFilter::Peak {
						cutoff: 1000.0,
						gain: -3.01,
						bandwidth: 100.0,
					};
					output = Some(Biquad::peak(sample_rate, 1000.0, -3.01, 100.0))
				}
				if ui
					.selectable_label(matches!(rbj, RBJFilter::LowShelf { .. }), "LowShelf")
					.clicked()
				{
					*rbj = RBJFilter::LowShelf {
						cutoff: 1000.0,
						gain: -3.01,
						slope: 1.5,
					};
					output = Some(Biquad::low_shelf(sample_rate, 1000.0, -3.01, 1.5))
				}
				if ui
					.selectable_label(matches!(rbj, RBJFilter::HighShelf { .. }), "HighShelf")
					.clicked()
				{
					*rbj = RBJFilter::HighShelf {
						cutoff: 1000.0,
						gain: -3.01,
						slope: 1.5,
					};
					output = Some(Biquad::high_shelf(sample_rate, 1000.0, -3.01, 1.5))
				}

				output
			}

			let mut need_change: Option<Biquad<CHANNELS>> = None;

			ui.horizontal(|ui| match &mut self.gui_state.filter_type {
				FilterType::Butterworth(inner) => {
					need_change =
						need_change
							.take()
							.or(butterworth_gui(ui, inner, self.sample_rate));
				}
				FilterType::Rbj(inner) => {
					need_change = need_change.take().or(rbj_gui(ui, inner, self.sample_rate))
				}
				FilterType::ByPass => {}
			});

			if let Some(biquad) = need_change {
				self.a = biquad.a;
				self.b = biquad.b;
				if self.gui_state.allpass_mode {
					self.transform_to_allpass();
				}
			}

			let sample_rate = self.sample_rate as f32;
			let max_freq = sample_rate / 2.0;

			let mut change_cutoff: Option<(ChangeFunc<CHANNELS>, f32)> = None;
			let mut change_cutoff_q: Option<(ChangeFunc2<CHANNELS>, f32, f32)> = None;
			let mut change_simulated: Option<(ChangeFunc3<CHANNELS>, f32, f32, f32)> = None;

			match &mut self.gui_state.filter_type {
				FilterType::ByPass => {
					ui.label("Bypassed");
				}
				FilterType::Butterworth(Butterworth::Lowpass { cutoff }) => {
					change_cutoff = change_cutoff.take().or(cutoff_ui(
						ui,
						cutoff,
						max_freq,
						Box::new(Self::set_to_butterworth_lowpass),
					));
				}
				FilterType::Butterworth(Butterworth::Highpass { cutoff }) => {
					change_cutoff = change_cutoff.take().or(cutoff_ui(
						ui,
						cutoff,
						max_freq,
						Box::new(Self::set_to_butterworth_highpass),
					));
				}
				FilterType::Butterworth(Butterworth::Bandpass { cutoff }) => {
					change_cutoff = change_cutoff.take().or(cutoff_ui(
						ui,
						cutoff,
						max_freq,
						Box::new(Self::set_to_butterworth_bandpass),
					));
				}
				FilterType::Butterworth(Butterworth::Bandstop { cutoff }) => {
					change_cutoff = change_cutoff.take().or(cutoff_ui(
						ui,
						cutoff,
						max_freq,
						Box::new(Self::set_to_butterworth_bandstop),
					));
				}
				FilterType::Rbj(RBJFilter::Lowpass { cutoff, q }) => {
					change_cutoff_q = change_cutoff_q.take().or(cutoff_q_ui(
						ui,
						cutoff,
						q,
						0.01..=10.0,
						"Q",
						max_freq,
						Box::new(Self::set_to_lowpass),
					));
				}
				FilterType::Rbj(RBJFilter::Highpass { cutoff, q }) => {
					change_cutoff_q = change_cutoff_q.take().or(cutoff_q_ui(
						ui,
						cutoff,
						q,
						0.01..=10.0,
						"Q",
						max_freq,
						Box::new(Self::set_to_highpass),
					));
				}
				FilterType::Rbj(RBJFilter::Bandpass { cutoff, q }) => {
					change_cutoff_q = change_cutoff_q.take().or(cutoff_q_ui(
						ui,
						cutoff,
						q,
						MIN_FREQUENCY..=10000.0,
						"BandWidth",
						max_freq,
						Box::new(Self::set_to_bandpass),
					));
				}
				FilterType::Rbj(RBJFilter::Bandstop { cutoff, q }) => {
					change_cutoff_q = change_cutoff_q.take().or(cutoff_q_ui(
						ui,
						cutoff,
						q,
						MIN_FREQUENCY..=10000.0,
						"BandWidth",
						max_freq,
						Box::new(Self::set_to_bandstop),
					));
				}
				FilterType::Rbj(RBJFilter::Peak {
					cutoff,
					gain,
					bandwidth,
				}) => {
					change_simulated = change_simulated.take().or(simulate_ui(
						ui,
						(cutoff, gain, bandwidth),
						max_freq,
						MIN_FREQUENCY..=10000.0,
						vec!["Cutoff", "Gain(dB)", "Bandwidth"],
						Box::new(Self::set_to_peak),
					));
				}
				FilterType::Rbj(RBJFilter::LowShelf {
					cutoff,
					gain,
					slope,
				}) => {
					change_simulated = change_simulated.take().or(simulate_ui(
						ui,
						(cutoff, gain, slope),
						max_freq,
						0.5..=2.0,
						vec!["Cutoff", "Gain(dB)", "Slope"],
						Box::new(Self::set_to_low_shelf),
					));
				}
				FilterType::Rbj(RBJFilter::HighShelf {
					cutoff,
					gain,
					slope,
				}) => {
					change_simulated = change_simulated.take().or(simulate_ui(
						ui,
						(cutoff, gain, slope),
						max_freq,
						0.5..=2.0,
						vec!["Cutoff", "Gain(dB)", "Slope"],
						Box::new(Self::set_to_high_shelf),
					));
				}
			}

			if let Some((change_func, cutoff)) = change_cutoff {
				change_func(self, cutoff);
				if self.gui_state.allpass_mode {
					self.transform_to_allpass();
				}
			}

			if let Some((change_func, cutoff, q)) = change_cutoff_q {
				change_func(self, cutoff, q);
				if self.gui_state.allpass_mode {
					self.transform_to_allpass();
				}
			}

			if let Some((change_func, cutoff, gain, factor)) = change_simulated {
				change_func(self, cutoff, gain, factor);
				if self.gui_state.allpass_mode {
					self.transform_to_allpass();
				}
			}
		});
	}
}
