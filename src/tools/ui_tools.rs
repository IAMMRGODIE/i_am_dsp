use std::f32::consts::PI;

use egui::*;

use crate::{effects::filter::MIN_FREQUENCY, tools::ring_buffer::RingBuffer};

pub(crate) fn draw_complex_response(
	ui: &mut egui::Ui,
	sample_rate: usize,
	complex_response: impl Fn(f32) -> (f32, f32),
) -> egui::Response {
	Frame::canvas(ui.style()).show(ui, |ui| {
		const SAMPLE_POINTS: usize = 128;
		// const LOW_DB: f32 = -60.0;
		// const HIGH_DB: f32 = 0.0;

		let desired_size = ui.available_size();
		let response = ui.allocate_response(desired_size, Sense::click_and_drag());
		let rect = response.rect;

		let to_screen =
			emath::RectTransform::from_to(Rect::from_x_y_ranges(0.0..=1.0, -1.0..=1.0), rect);

		let mut amplitude_positions = vec![Pos2::ZERO; SAMPLE_POINTS];
		let mut phase_positions = vec![Pos2::ZERO; SAMPLE_POINTS];

		for (i, (amp_pos, phase_pos)) in amplitude_positions
			.iter_mut()
			.zip(phase_positions.iter_mut())
			.enumerate() 
		{
			let t = i as f32 / (SAMPLE_POINTS - 1) as f32;
			amp_pos.x = t;
			phase_pos.x = t;

			let logarithmic_lerp = MIN_FREQUENCY.ln() + ((sample_rate as f32 / 2.0 - MIN_FREQUENCY).ln() - MIN_FREQUENCY.ln()) * t;
			let freq  = logarithmic_lerp.exp();
			let (amplitude, phase) = complex_response(freq);
			let amplitude = amplitude * 0.5;
			// let amp_db = amplitude.log10() * 20.0;
			amp_pos.y = amplitude * 2.0 - 1.0;
			amp_pos.y = - amp_pos.y;
			phase_pos.y = if phase <= PI {
				phase
			}else {
				phase - 2.0 * PI
			} / PI;

			*amp_pos = to_screen * *amp_pos;
			*phase_pos = to_screen * *phase_pos;
		}

		ui.painter_at(rect).extend([
			Shape::line(amplitude_positions, (2.0, Color32::WHITE)),
			Shape::line(phase_positions, (2.0, Color32::BLUE)),
		]);

		response
	}).inner
}

pub(crate) fn draw_waveform(
	ui: &mut egui::Ui,
	current_sample_in: Option<f32>,
	pcm_data: &[&[f32]],
	loop_range: &Option<(usize, usize)>,
	reverse: bool,
	mask_reverse: bool,
	// speed: f32,
) -> egui::Response {
	Frame::canvas(ui.style()).show(ui, |ui| {
		const SELECT_COLOR: Color32 = Color32::from_rgb(240, 180, 17);
		const POINT_WIDTH: f32 = 3.0;
		const PADDING: f32 = 1.0;

		let desired_size = ui.available_size();
		let response = ui.allocate_response(desired_size, Sense::click_and_drag());
		let rect = response.rect;
		let total_samples = pcm_data[0].len();

		if total_samples == 0 {
			return response;
		}

		// println!("rect: {:?}, desired_size: {:?}", rect, desired_size);

		let to_screen =
			emath::RectTransform::from_to(Rect::from_x_y_ranges(0.0..=1.0, -1.0..=1.0), rect);

		let total_shapes = (rect.width() / (POINT_WIDTH + PADDING)).ceil() as usize;
		let identity = 1.0 / total_shapes as f32;
		let sample_per_shape = total_samples / total_shapes;
		// let total_samples = sample_per_shape * total_shapes;

		let mut shapes = vec![];

		let current_time = current_sample_in.unwrap_or(total_samples as f32) / total_samples as f32;
		let mut current_sample: usize = 0;

		for i in 0..=total_shapes {
			// let mut positions = [Pos2::ZERO; CHANNELS];
			let sample = pcm_data
				.iter()
				.map(|vals| {
					if vals.len() <= 32767 {
						let idx = i * (vals.len() - 1) / total_shapes;
						let idx_next = (i + 1) * (vals.len() - 1) / total_shapes;
						let idx_next = idx_next.min(vals.len() - 1);

						return vals[idx..idx_next]
							.iter()
							.max_by_key(|val| (val.abs() * 1000.0) as usize)
							.unwrap_or(&0.0)
							.powi(2);
					}

					const STEP: usize = 10;
					const WINDOW_SIZE: usize = 1000;

					let total = (current_sample + WINDOW_SIZE).min(total_samples) - current_sample.saturating_sub(WINDOW_SIZE) ;
					let total = total / STEP;
					vals.iter()
						.skip(current_sample.saturating_sub(WINDOW_SIZE))
						.step_by(STEP)
						.take(WINDOW_SIZE)
						.map(|val| val * val)
						.sum::<f32>() / total as f32
						// .unwrap_or(&0.0)
				})
				.sum::<f32>().sqrt();
			
			let position_x = i as f32 * identity;
			let position_x = if reverse {
				1.0 - position_x
			}else {
				position_x
			};
			let position = pos2(position_x, sample.abs().clamp(0.0, 1.0) * 0.95);
			let position_upside_down = pos2(position.x, - position.y);
			let position = to_screen * position;
			let position_upside_down = to_screen * position_upside_down;
			
			let should_color_selected = if let Some((from, to)) = loop_range {
				let from_x = *from as f32 / total_samples as f32;
				let to_x = *to as f32 / total_samples as f32;
				position_x >= from_x && position_x <= to_x
			}else {
				false
			};

			let high_light = if mask_reverse {
				position_x <= current_time
			}else {
				position_x >= current_time
			};

			shapes.push(epaint::Shape::line(
				vec![position, position_upside_down], 
				(POINT_WIDTH, if should_color_selected {
					lerp_to_gamma(SELECT_COLOR, Color32::from_gray(225), 0.7)
				}else if high_light {
					epaint::Color32::from_gray(225)
				}else {
					epaint::Color32::from_gray(125)
				}),
			));

			current_sample += sample_per_shape;
		}

		if let Some((from, to)) = loop_range {
			let from_x = *from as f32 / total_samples as f32;
			let to_x = *to as f32 / total_samples as f32;
			let from_positions = vec![
				to_screen * pos2(from_x, -1.0), 
				to_screen * pos2(from_x, 1.0)
			];
			let to_positions = vec![
				to_screen * pos2(to_x, -1.0), 
				to_screen * pos2(to_x, 1.0)
			];
			shapes.push(epaint::Shape::line(
				from_positions,
				(2.0, SELECT_COLOR),
			));
			shapes.push(epaint::Shape::line(
				to_positions,
				(2.0, SELECT_COLOR),
			));

			shapes.push(epaint::Shape::rect_filled(
				Rect::from_min_max(
					to_screen * pos2(from_x, -1.0),
					to_screen * pos2(to_x, 1.0),
				),
				0.0,
				Color32::from_rgba_unmultiplied(
					SELECT_COLOR.r(), 
					SELECT_COLOR.g(), 
					SELECT_COLOR.b(), 
					32,
				),
			));
		}

		if let Some(current_sample) = current_sample_in {
			let current_time = current_sample / total_samples as f32;
			shapes.push(epaint::Shape::line(
				vec![
					to_screen * pos2(current_time, -1.0), 
					to_screen * pos2(current_time, 1.0)
				],
				(2.0, epaint::Color32::from_gray(255)),
			));
		}

		ui.painter_at(rect).extend(shapes);

		response
	}).inner
}

pub(crate) fn draw_envelope(
	ui: &mut egui::Ui,
	envelope: &[&RingBuffer<f32>],
	has_negative: bool,
) -> egui::Response {
	Frame::canvas(ui.style()).show(ui, |ui| {
		const MAX_POINTS: usize = 1024;

		let desired_size = ui.available_size();
		let response = ui.allocate_response(desired_size, Sense::click_and_drag());
		let rect = response.rect;

		let to_screen =
			emath::RectTransform::from_to(Rect::from_x_y_ranges(0.0..=1.0, -1.0..=0.0), rect);

		let len = envelope[0].capacity();
		let channels = envelope.len();
		let step = len / MAX_POINTS + 1;
		let mut positions = vec![];
		for i in (0..len).step_by(step) {
			let env = envelope.iter().map(|x| x[i]).sum::<f32>() / channels as f32;
			let env = if has_negative {
				(env + 1.0) / 2.0
			}else {
				env
			};
			let pos = pos2(
				i as f32 / len as f32,
				- env * if has_negative {
					1.0
				}else {
					0.85
				}
			);
			positions.push(to_screen * pos);
		}

		ui.painter_at(rect).extend([
			Shape::line(positions, (1.0, Color32::WHITE))
		]);

		response
	}).inner
}

#[inline(always)]
pub(crate) fn gain_ui(
	ui: &mut egui::Ui,
	gain: &mut f32,
	name: Option<String>,
	under_1: bool,
) {
	let range = if under_1 {
		0.01..=0.99
	}else {
		0.01..=4.0
	};

	ui.add(Slider::new(gain, range)
		.custom_formatter(|val, _| {
			let db = val.log10() * 20.0;
			if db.is_nan() {
				"-inf dB".to_string()
			}else {
				format!("{:.2} dB", db)
			}
		})
		.custom_parser(|text| {
			let db = text.parse::<f64>().ok()?;
			let val = 10.0f64.powf(db / 20.0);
			Some(val)
		})
		.text(name.unwrap_or("Gain".to_string())));
}

pub(crate) fn draw_wavetable(
	ui: &mut egui::Ui,
	mut wavetable: impl FnMut(f32) -> f32,
) -> egui::Response {
	Frame::canvas(ui.style()).show(ui, |ui| {
		const POINTS: usize = 256;
		let response = ui.allocate_response(ui.available_size(), Sense::click_and_drag());
		let rect = response.rect;
		let to_screen = emath::RectTransform::from_to(Rect::from_x_y_ranges(0.0..=1.0, -1.0..=1.0), rect);

		let positions = (0..POINTS).map(|i| {
			let t = i as f32 / (POINTS - 1) as f32;
			to_screen * pos2(t, - wavetable(t))
		}).collect::<Vec<_>>();
		ui.painter_at(rect).extend([
			Shape::line(positions, (1.0, Color32::WHITE))
		]);

		response
	}).inner
}

fn fast_round(r: f32) -> u8 {
	(r + 0.5) as _ // rust does a saturating cast since 1.45
}

pub fn lerp_to_gamma(this: Color32, other: Color32, t: f32) -> Color32 {
	use emath::lerp;

	Color32::from_rgba_premultiplied(
		fast_round(lerp((this[0] as f32)..=(other[0] as f32), t)),
		fast_round(lerp((this[1] as f32)..=(other[1] as f32), t)),
		fast_round(lerp((this[2] as f32)..=(other[2] as f32), t)),
		fast_round(lerp((this[3] as f32)..=(other[3] as f32), t)),
	)
}