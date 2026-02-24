//! A helper to display and edit the EQ cruve.

use std::{collections::HashMap, f32::consts::PI, ops::RangeInclusive, sync::{Arc, RwLock}};

use i_am_dsp::prelude::{AtomicValue, Biquad, ParamMap, Paramed};
use iced::{Point, Rectangle, Renderer, Size, Theme, mouse::{self, Event, ScrollDelta}, widget::{Action, canvas::{Frame, Path, Program, Stroke}}};
use portable_atomic::{AtomicBool, AtomicF32, AtomicUsize};

use crate::{styles::{ALPHA_FACTOR, BORDER_WIDTH, PADDING}, tools::utils::Animator};

const MIN_HZ: f32 = 10.0;

/// A helper to display and edit the EQ cruve.
#[derive(Debug, Clone)]
pub struct EqualizerEditor {
	/// The sample rate of the audio signal.
	pub sample_rate: usize,
	/// The current editing filter.
	pub current_editing: Arc<AtomicUsize>,
	biquad_filter_map: ParamMap,
	sample_points: usize,
	// canvas_cache: iced::widget::canvas::Cache,
	biquad_settings: Arc<RwLock<Vec<BiquadFilterEditor>>>,
	total_biquads: usize,
}

impl EqualizerEditor {
	/// Create a new [`EqualizerEditor`]
	/// 
	/// # Panics
	/// 
	/// panics if sample_rate <= 30.
	pub fn new(biquads: usize, sample_rate: usize) -> (Self, Paramed<Vec<Biquad>>) {
		assert!(sample_rate > 30, "Sample rate must be greater than 60");
		
		let max_hz = sample_rate as f32 / 2.0 - MIN_HZ;
		
		let mut filters = Vec::with_capacity(biquads);
		let mut additional_keys = HashMap::new();

		let biquad_settings = (0..biquads)
			.map(|i| {
				let t = (i + 1) as f32 / (biquads + 1) as f32;
				let freq = (MIN_HZ.ln() + (max_hz.ln() - MIN_HZ.ln()) * t).exp();

				let filter = Biquad::<2>::peak(sample_rate, freq, 0.0, (max_hz - MIN_HZ) / 10.0);

				let (a, b) = (filter.a, filter.b);

				additional_keys.insert(format!("{i}.a1"), Arc::new(AtomicValue::Float { 
					value: AtomicF32::new(a[0]), 
					range: -16384.0..=16384.0, 
					logarithmic: false, 
					changed: AtomicBool::new(false)
				}));

				additional_keys.insert(format!("{i}.a2"), Arc::new(AtomicValue::Float { 
					value: AtomicF32::new(a[1]), 
					range: -16384.0..=16384.0, 
					logarithmic: false, 
					changed: AtomicBool::new(false)
				}));

				additional_keys.insert(format!("{i}.b0"), Arc::new(AtomicValue::Float { 
					value: AtomicF32::new(b[0]), 
					range: -16384.0..=16384.0, 
					logarithmic: false, 
					changed: AtomicBool::new(false)
				}));

				additional_keys.insert(format!("{i}.b1"), Arc::new(AtomicValue::Float { 
					value: AtomicF32::new(b[1]), 
					range: -16384.0..=16384.0, 
					logarithmic: false, 
					changed: AtomicBool::new(false)
				}));

				additional_keys.insert(format!("{i}.b2"), Arc::new(AtomicValue::Float { 
					value: AtomicF32::new(b[2]), 
					range: -16384.0..=16384.0, 
					logarithmic: false, 
					changed: AtomicBool::new(false)
				}));

				filters.push(filter);

				BiquadFilterEditor::Peak { 
					cutoff: freq, 
					// slope: 1.0,
					bandwidth: (max_hz - MIN_HZ) / 10.0, 
					gain_db: 0.0, 
				}
			}).collect::<Vec<_>>();
		
		let output = Paramed::new_with_additional(filters, additional_keys);

		let biquad_filter_map = output.param_map();

		(Self {
			sample_rate,
			current_editing: Default::default(),
			biquad_filter_map,
			biquad_settings: Arc::new(RwLock::new(biquad_settings)),
			sample_points: 1024,
			total_biquads: biquads,
			// canvas_cache: iced::widget::canvas::Cache::default(),
		}, output)
	}

	// #[inline(always)]
	// /// refresh the canvas cache.
	// pub fn refresh_cache(&mut self) {
	// 	self.canvas_cache.clear();
	// }

	/// Edit the filter at the given index, will not call the given function if the index is out of bounds.
	/// 
	/// Returns None if the index is out of bounds.
	pub fn edit_filter<F: FnOnce(&mut BiquadFilterEditor, usize) -> R, R>(&self, index: usize, f: F) -> Option<R> {
		let mut biquad_settings = if let Ok(inner) = self.biquad_settings.write() {
			inner
		}else {
			return None;
		};

		if index < biquad_settings.len() {
			let out = f(&mut biquad_settings[index], self.sample_rate);
			biquad_settings[index].modify(&self.biquad_filter_map, index, self.sample_rate);
			// self.canvas_cache.clear();
			Some(out)
		}else {
			None
		}
	}

	/// Edit the current filter
	pub fn edit_current<F: FnOnce(&mut BiquadFilterEditor, usize) -> R, R>(&self, f: F) -> R {
		let mut biquad_settings = self.biquad_settings.write().expect("cannot get lock");
		let index = self.current_editing.load(std::sync::atomic::Ordering::SeqCst);

		let out = f(&mut biquad_settings[index], self.sample_rate);
		biquad_settings[index].modify(&self.biquad_filter_map, index, self.sample_rate);
		out
	}

	fn draw_path(&self, size: Size, padding: f32, max_gain_db: f32) -> (Path, Path) {
		let (mut a_vec, mut b_vec) = (vec![], vec![]);
		
		let usable_width = size.width - 2.0 * padding;
		let usable_height = size.height - 2.0 * padding;

		let max_hz = self.sample_rate as f32 / 2.0 - MIN_HZ;

		for i in 0..self.total_biquads {
			let a1 = self.biquad_filter_map.get(&format!("{i}.a1")).unwrap()
				.load(std::sync::atomic::Ordering::SeqCst).float().unwrap();
			let a2 = self.biquad_filter_map.get(&format!("{i}.a2")).unwrap()
				.load(std::sync::atomic::Ordering::SeqCst).float().unwrap();
			let b0 = self.biquad_filter_map.get(&format!("{i}.b0")).unwrap()
				.load(std::sync::atomic::Ordering::SeqCst).float().unwrap();
			let b1 = self.biquad_filter_map.get(&format!("{i}.b1")).unwrap()
				.load(std::sync::atomic::Ordering::SeqCst).float().unwrap();
			let b2 = self.biquad_filter_map.get(&format!("{i}.b2")).unwrap()
				.load(std::sync::atomic::Ordering::SeqCst).float().unwrap();

			a_vec.push([a1, a2]);
			b_vec.push([b0, b1, b2]);
		}

		#[inline]
		fn complex_response(a: &[f32; 2], b: &[f32; 3], frequency: f32, sample_rate: usize) -> (f32, f32) {
			let frequency = 2.0 * PI * frequency / sample_rate as f32;

			let cos_f = frequency.cos();
			let sin_f = frequency.sin();
			let cos_2f = 2.0 * cos_f * cos_f - 1.0;
			let sin_2f = 2.0 * cos_f * sin_f;

			let real_n = b[0] + b[1] * cos_f + b[2] * cos_2f;
			let imag_n = - (b[1] * sin_f + b[2] * sin_2f);
			
			let real_d = 1.0 + a[0] * cos_f + a[1] * cos_2f;
			let imag_d = - (a[0] * sin_f + a[1] * sin_2f);

			let amplitude = real_n.hypot(imag_n) / real_d.hypot(imag_d);
			let phase = (imag_n.atan2(real_n) - imag_d.atan2(real_d)).rem_euclid(2.0 * PI);

			(amplitude, phase)
		}

		#[inline]
		fn complex_mul(l: (f32, f32), r: (f32, f32)) -> (f32, f32) {
			let (l_a, l_p) = l;
			let (r_a, r_p) = r;
			(l_a * r_a, (l_p + r_p) % (2.0 * PI))
		}

		// complex_response_vec
		let mut cr_vec = vec![];

		for i in 0..self.sample_points {
			let t = i as f32 / (self.sample_points - 1) as f32;
			let freq = (MIN_HZ.ln() + (max_hz.ln() - MIN_HZ.ln()) * t).exp();
			let mut value = (1.0, 0.0);

			for (a, b) in a_vec.iter().zip(b_vec.iter()) {
				let cr = complex_response(a, b, freq, self.sample_rate);
				value = complex_mul(value, cr);
			}

			cr_vec.push(value);
		}

		(
			Path::new(|builder| {
				builder.move_to(Point::new(2.0, padding + usable_height / 2.0));
				for (i, (amp, _)) in cr_vec.iter().enumerate() {
					let amp_db = 20.0 * amp.log10();
					let height = (1.0 - (amp_db + max_gain_db) / (2.0 * max_gain_db)) * usable_height;
					let width = (i as f32 / (self.sample_points - 1) as f32) * usable_width;
					let point = Point::new(padding + width, padding + height);
					builder.line_to(point);
				}
				builder.line_to(Point::new(
					padding + padding + padding + padding + usable_width, 
					padding + usable_height / 2.0
				));
				builder.line_to(Point::new(
					padding + padding + padding + padding + usable_width,
					padding + padding + padding + padding + usable_height
				));
				builder.line_to(Point::new(2.0, padding + padding + padding + padding + usable_height));
				builder.close();
			}), 
			Path::new(|builder| {
				for (i, (_, phase)) in cr_vec.iter().enumerate() {
					let phase = if *phase <= PI {
						phase / PI
					}else {
						(phase - 2.0 * PI) / PI
					};
					let phase = (phase + 1.0) / 2.0;

					let height = (1.0 - phase) * usable_height;
					let width = (i as f32 / (self.sample_points - 1) as f32) * usable_width;
					let point = Point::new(padding + width, padding + height);
					if i == 0 {
						builder.move_to(point);
					}else {
						builder.line_to(point);
					}
				}
			})
		)
	}

	fn filter_positions(&self, size: Size, padding: f32, max_gain_db: f32) -> Option<Vec<Point>> {
		let usable_width = size.width - 2.0 * padding;
		let usable_height = size.height - 2.0 * padding;
		let max_hz = self.sample_rate as f32 / 2.0 - MIN_HZ;

		let settings = self.biquad_settings.read().ok()?;
		Some(settings.iter().map(|settings| {
			let freq = settings.get_cutoff();
			let gain = settings.get_gain_db().unwrap_or(0.0);
			let height = (1.0 - (gain + max_gain_db) / (2.0 * max_gain_db)) * usable_height;
			let width = (freq.ln() - MIN_HZ.ln()) / (max_hz.ln() - MIN_HZ.ln()) * usable_width;
			let x = padding + width;
			let y = padding + height;
			Point::new(x, y)
		}).collect())
	}
}

#[derive(Clone, Debug)]
/// A helper to edit the EQ cruve.
#[allow(missing_docs)]
pub enum BiquadFilterEditor {
	LowPass {
		cutoff: f32,
		q: f32,
	},
	HighPass {
		cutoff: f32,
		q: f32,
	},
	BandPass {
		cutoff: f32,
		bandwidth: f32,
	},
	BandStop {
		cutoff: f32,
		bandwidth: f32,
	},
	Peak {
		cutoff: f32,
		bandwidth: f32,
		gain_db: f32,
	},
	LowShelf {
		cutoff: f32,
		gain_db: f32,
		slope: f32,
	},
	HighShelf {
		cutoff: f32,
		gain_db: f32,
		slope: f32,
	},
}

impl BiquadFilterEditor {
	/// set the q value for the filter.
	pub fn set_q(&mut self, new_q: f32) {
		match self {
			BiquadFilterEditor::LowPass { q, .. } => *q = new_q.clamp(0.01, 10.0),
			BiquadFilterEditor::HighPass { q, .. } => *q = new_q.clamp(0.01, 10.0),
			BiquadFilterEditor::BandPass { bandwidth, cutoff } |
			BiquadFilterEditor::BandStop { bandwidth, cutoff } |
			BiquadFilterEditor::Peak { bandwidth, cutoff, .. } => {
				let max_q = *cutoff / MIN_HZ;
				let min_q = *cutoff / 10000.0;
				let q = new_q.clamp(min_q, max_q);

				let new_band_width = *cutoff / q;
				*bandwidth = new_band_width.clamp(MIN_HZ, 10000.0);
			},
			BiquadFilterEditor::LowShelf { slope, .. } => *slope = new_q.clamp(0.5, 2.0),
			BiquadFilterEditor::HighShelf { slope, .. } => *slope = new_q.clamp(0.5, 2.0),
		}
	}

	/// get the q value for the filter.
	pub fn get_q(&self) -> f32 {
		match self {
			BiquadFilterEditor::LowPass { q, .. } => *q,
			BiquadFilterEditor::HighPass { q, .. } => *q,
			BiquadFilterEditor::BandPass { bandwidth, cutoff } | 
			BiquadFilterEditor::BandStop { bandwidth, cutoff } | 
			BiquadFilterEditor::Peak { bandwidth, cutoff, .. } => {
				let out = *cutoff / *bandwidth;
				let max_q = cutoff / MIN_HZ;
				let min_q = cutoff / 10000.0;
				out.clamp(min_q, max_q)
			},
			BiquadFilterEditor::LowShelf { slope, .. } => *slope,
			BiquadFilterEditor::HighShelf { slope, .. } => *slope,
		}
	}

	/// Get the range of q values for the filter.
	pub fn get_range_q(&self) -> RangeInclusive<f32> {
		match self {
			BiquadFilterEditor::LowPass { .. } => 0.01..=10.0,
			BiquadFilterEditor::HighPass { .. } => 0.01..=10.0,
			BiquadFilterEditor::BandPass { .. } => 0.01..=5.0,
			BiquadFilterEditor::BandStop { .. } => 0.01..=5.0,
			BiquadFilterEditor::Peak { .. } => 0.01..=5.0,
			BiquadFilterEditor::LowShelf { .. } => 0.5..=2.0,
			BiquadFilterEditor::HighShelf { .. } => 0.5..=2.0,
		}
	}

	/// set the gain_db value for the filter.
	pub fn set_gain_db(&mut self, gain_db_new: f32) {
		match self {
			BiquadFilterEditor::LowPass { .. } => {},
			BiquadFilterEditor::HighPass { .. } => {},
			BiquadFilterEditor::BandPass { .. } => {},
			BiquadFilterEditor::BandStop { .. } => {},
			BiquadFilterEditor::Peak { gain_db, .. } => *gain_db = gain_db_new.clamp(-12.0, 12.0),
			BiquadFilterEditor::LowShelf { gain_db, .. } => *gain_db = gain_db_new.clamp(-12.0, 12.0),
			BiquadFilterEditor::HighShelf { gain_db, .. } => *gain_db = gain_db_new.clamp(-12.0, 12.0),
		}
	}

	/// get the gain_db value for the filter.
	pub fn get_gain_db(&self) -> Option<f32> {
		match self {
			BiquadFilterEditor::LowPass { .. } => None,
			BiquadFilterEditor::HighPass { .. } => None,
			BiquadFilterEditor::BandPass { .. } => None,
			BiquadFilterEditor::BandStop { .. } => None,
			BiquadFilterEditor::Peak { gain_db, .. } => Some(*gain_db),
			BiquadFilterEditor::LowShelf { gain_db, .. } => Some(*gain_db),
			BiquadFilterEditor::HighShelf { gain_db, .. } => Some(*gain_db),
		}
	}

	/// get the cutoff value for the filter.
	pub fn get_cutoff(&self) -> f32 {
		match self {
			BiquadFilterEditor::LowPass { cutoff, .. } => *cutoff,
			BiquadFilterEditor::HighPass { cutoff, .. } => *cutoff,
			BiquadFilterEditor::BandPass { cutoff, .. } => *cutoff,
			BiquadFilterEditor::BandStop { cutoff, .. } => *cutoff,
			BiquadFilterEditor::Peak { cutoff, .. } => *cutoff,
			BiquadFilterEditor::LowShelf { cutoff, .. } => *cutoff,
			BiquadFilterEditor::HighShelf { cutoff, .. } => *cutoff,
		}
	}

	/// set the cutoff value for the filter.
	pub fn set_cutoff(&mut self, cutoff: f32, sample_rate: usize) {
		let cutoff_new = cutoff.clamp(MIN_HZ, sample_rate as f32 / 2.0 - MIN_HZ);

		match self {
			BiquadFilterEditor::LowPass { cutoff , .. } => *cutoff = cutoff_new,
			BiquadFilterEditor::HighPass { cutoff, .. } => *cutoff = cutoff_new,
			BiquadFilterEditor::BandPass { cutoff, .. } => *cutoff = cutoff_new,
			BiquadFilterEditor::BandStop { cutoff, .. } => *cutoff = cutoff_new,
			BiquadFilterEditor::Peak { cutoff, .. } => *cutoff = cutoff_new,
			BiquadFilterEditor::LowShelf { cutoff, .. } => *cutoff = cutoff_new,
			BiquadFilterEditor::HighShelf { cutoff, .. } => *cutoff = cutoff_new,
		}
	}

	/// set current filter to the next filter type.
	pub fn set_to_next(&mut self) {
		match self {
			BiquadFilterEditor::LowPass { cutoff, q } => *self = BiquadFilterEditor::HighPass { 
				cutoff: *cutoff, 
				q: q.clamp(0.01, 10.0)
			},
			BiquadFilterEditor::HighPass { cutoff, q } => *self = BiquadFilterEditor::BandPass { 
				cutoff: *cutoff, 
				bandwidth: (*cutoff / *q).clamp(10.0, 10000.0),
			},
			BiquadFilterEditor::BandPass { cutoff, bandwidth } => *self = BiquadFilterEditor::BandStop { 
				cutoff: *cutoff, 
				bandwidth: bandwidth.clamp(10.0, 10000.0),
			},
			BiquadFilterEditor::BandStop { cutoff, bandwidth } => *self = BiquadFilterEditor::Peak {
				cutoff: *cutoff, 
				bandwidth: bandwidth.clamp(10.0, 10000.0),
				gain_db: 0.0,
			},
			BiquadFilterEditor::Peak { cutoff, bandwidth, gain_db } => *self = BiquadFilterEditor::LowShelf {
				cutoff: *cutoff, 
				gain_db: *gain_db,
				slope: (*cutoff / *bandwidth).clamp(0.5, 2.0),
			},
			BiquadFilterEditor::LowShelf { cutoff, gain_db, slope } => *self = BiquadFilterEditor::HighShelf {
				cutoff: *cutoff, 
				gain_db: *gain_db,
				slope: slope.clamp(0.5, 2.0),
			},
			BiquadFilterEditor::HighShelf { cutoff, slope, .. } => *self = BiquadFilterEditor::LowPass {
				cutoff: *cutoff, 
				q: slope.clamp(0.5, 2.0)
			},
		}
	}

	/// Set the filter to the previous filter type.
	pub fn set_to_prev(&mut self) {
		match self {
			BiquadFilterEditor::LowPass { cutoff, q } => *self = BiquadFilterEditor::LowShelf { 
				cutoff: *cutoff, 
				gain_db: 0.0,
				slope: q.clamp(0.5, 2.0),
			},
			BiquadFilterEditor::HighPass { cutoff, q } => *self = BiquadFilterEditor::Peak { 
				cutoff: *cutoff, 
				bandwidth: (*cutoff / *q).clamp(10.0, 10000.0),
				gain_db: 0.0,
			},
			BiquadFilterEditor::BandPass { cutoff, bandwidth } => *self = BiquadFilterEditor::HighPass { 
				cutoff: *cutoff, 
				q: (*cutoff / *bandwidth).clamp(0.01, 10.0),
			},
			BiquadFilterEditor::BandStop { cutoff, bandwidth } => *self = BiquadFilterEditor::BandPass { 
				cutoff: *cutoff, 
				bandwidth: bandwidth.clamp(10.0, 10000.0),
			},
			BiquadFilterEditor::Peak { cutoff, bandwidth, .. } => *self = BiquadFilterEditor::BandStop {
				cutoff: *cutoff, 
				bandwidth: *bandwidth,
			},
			BiquadFilterEditor::LowShelf { cutoff, gain_db, slope } => *self = BiquadFilterEditor::Peak {
				cutoff: *cutoff, 
				bandwidth: (*cutoff / *slope).clamp(10.0, 10000.0),
				gain_db: *gain_db,
			},
			BiquadFilterEditor::HighShelf { cutoff, gain_db, slope } => *self = BiquadFilterEditor::LowShelf {
				cutoff: *cutoff, 
				gain_db: *gain_db,
				slope: *slope,
			},
		}
	}

	fn modify(&self, param_map: &ParamMap, index: usize, sample_rate: usize) {
		let (a, b) = match self {
			BiquadFilterEditor::LowPass { cutoff, q } => {
				let filter = Biquad::<1>::lowpass(sample_rate, *cutoff, *q);
				(filter.a, filter.b)
			},
			BiquadFilterEditor::HighPass { cutoff, q } => {
				let filter = Biquad::<1>::highpass(sample_rate, *cutoff, *q);
				(filter.a, filter.b)
			},
			BiquadFilterEditor::BandPass { cutoff, bandwidth } => {
				let filter = Biquad::<1>::bandpass(sample_rate, *cutoff, *bandwidth);
				(filter.a, filter.b)
			},
			BiquadFilterEditor::BandStop { cutoff, bandwidth } => {
				let filter = Biquad::<1>::bandstop(sample_rate, *cutoff, *bandwidth);
				(filter.a, filter.b)
			},
			BiquadFilterEditor::Peak { cutoff, bandwidth, gain_db } => {
				let filter = Biquad::<1>::peak(sample_rate, *cutoff,  *gain_db, *bandwidth);
				(filter.a, filter.b)
			},
			BiquadFilterEditor::LowShelf { cutoff, gain_db, slope } => {
				let filter = Biquad::<1>::low_shelf(sample_rate, *cutoff, *gain_db, *slope);
				(filter.a, filter.b)
			},
			BiquadFilterEditor::HighShelf { cutoff, gain_db, slope } => {
				let filter = Biquad::<1>::high_shelf(sample_rate, *cutoff, *gain_db, *slope);
				(filter.a, filter.b)
			},
		};

		param_map.set(&format!("{index}.a1"), a[0], std::sync::atomic::Ordering::SeqCst);
		param_map.set(&format!("{index}.a2"), a[1], std::sync::atomic::Ordering::SeqCst);
		param_map.set(&format!("{index}.b0"), b[0], std::sync::atomic::Ordering::SeqCst);
		param_map.set(&format!("{index}.b1"), b[1], std::sync::atomic::Ordering::SeqCst);
		param_map.set(&format!("{index}.b2"), b[2], std::sync::atomic::Ordering::SeqCst);
	}
}

#[derive(Default)]
/// The internal state of the [`EqualizerEditor`]
/// 
/// Normally, you don't need to use this directly.
pub struct EqEditorState {
	current_mouse_pos: Point,
	hovering: HashMap<usize, Animator>,
	dragging: Option<DraggingInfo>,
	dragging_animator: Animator,
	last_drag_id: Option<usize>,
}

struct DraggingInfo {
	start_node_id: usize,
	last_pos: Point,
}

impl<Message> Program<Message> for EqualizerEditor {
	type State = EqEditorState;

	fn draw(
		&self,
		state: &Self::State,
		renderer: &Renderer,
		theme: &Theme,
		bounds: Rectangle,
		_cursor: mouse::Cursor,
	) -> Vec<iced::widget::canvas::Geometry<Renderer>> {
		let mut frame = Frame::new(renderer, bounds.size());
		let color = theme.extended_palette().primary.base.color;

		let rect = Path::rounded_rectangle(
			Point::new(BORDER_WIDTH, BORDER_WIDTH), 
			bounds.size().expand([- BORDER_WIDTH * 2.0, - BORDER_WIDTH * 2.0]), 
			PADDING.into()
		);

		frame.fill(&rect, theme.extended_palette().background.weakest.color);

		let (amp_res, phase_res) = self.draw_path(bounds.size(), PADDING, 12.0);
		
		frame.stroke(&phase_res, Stroke::default().with_color(theme.extended_palette().background.weakest.text).with_width(1.0));
		frame.stroke(&amp_res, Stroke::default().with_color(color).with_width(1.5));
		frame.fill(&amp_res, color.scale_alpha(ALPHA_FACTOR * ALPHA_FACTOR));

		let Some(positions) = self.filter_positions(bounds.size(), PADDING, 12.0) else {
			return vec![frame.into_geometry()];
		};

		let node_circles = Path::new(|builder| {
			const RADIUS: f32 = 10.0;

			for (i, point) in positions.into_iter().enumerate() {
				builder.circle(point, RADIUS);

				let animator = state.hovering.get(&i).map(|inner| inner.calc()).unwrap_or_default();

				if let Some(info) = &state.last_drag_id && *info == i {
					let inter = 1.75 * animator + (1.5 - 1.75 * animator) * state.dragging_animator.calc();  

					builder.circle(point, RADIUS * inter);
				}else {
					builder.circle(point, RADIUS * 1.75 * animator);
				}
			}
		});

		frame.stroke(&node_circles, Stroke::default().with_color(color).with_width(1.0));
		frame.fill(&node_circles, color.scale_alpha(ALPHA_FACTOR * ALPHA_FACTOR));

		let border_color = theme.extended_palette().background.strongest.color;
		frame.stroke(&rect, Stroke::default().with_color(border_color).with_width(BORDER_WIDTH));

		vec![frame.into_geometry(),]
	}

	fn update(
		&self,
		state: &mut Self::State,
		event: &iced::Event,
		bounds: iced::Rectangle,
		_cursor: mouse::Cursor,
	) -> Option<iced::widget::Action<Message>> {
		if let Some(info) = &mut state.dragging {
			let mut biquad_settings = if let Ok(inner) = self.biquad_settings.write() {
				inner
			}else {
				return None;
			};
			let delta_x = state.current_mouse_pos.x - info.last_pos.x;
			let delta_y = state.current_mouse_pos.y - info.last_pos.y;

			let max_hz = self.sample_rate as f32 / 2.0 - MIN_HZ;
			
			let change_x = delta_x / bounds.width * (max_hz.ln() - MIN_HZ.ln());
			let change_y = - delta_y / bounds.height * 24.0;
			
			let cutoff_ori = biquad_settings[info.start_node_id].get_cutoff();
			let gain_ori = biquad_settings[info.start_node_id].get_gain_db().unwrap_or(0.0);
			
			biquad_settings[info.start_node_id].set_cutoff(
				(cutoff_ori.ln() + change_x).exp(),
				self.sample_rate,
			);
			biquad_settings[info.start_node_id].set_gain_db(gain_ori + change_y);

			biquad_settings[info.start_node_id].modify(&self.biquad_filter_map, info.start_node_id, self.sample_rate);
			
			info.last_pos = state.current_mouse_pos;
		}

		let positions = self.filter_positions(bounds.size(), PADDING, 12.0)?;

		for (i, point) in positions.iter().enumerate() {
			if state.current_mouse_pos.distance(Point::new(point.x + bounds.x, point.y + bounds.y)) < 10.0 && state.dragging.is_none() {
				state.hovering.entry(i).or_default().in_if_out();
			}else {
				state.hovering.entry(i).or_default().out_if_in();
			}
		}

		let action = if state.hovering.values().any(|inner| inner.is_animating()) || 
			state.dragging_animator.is_animating() 
		{
			Some(Action::request_redraw())
		}else {
			None
		};

		let iced::Event::Mouse(mouse_event) = event else {
			return action;
		};

		match mouse_event {
			Event::CursorMoved { position } => {
				state.current_mouse_pos = *position;
			},
			Event::ButtonPressed(_) if state.dragging.is_none() => {
				for (i, point) in positions.into_iter().enumerate() {
					if state.current_mouse_pos.distance(Point::new(point.x + bounds.x, point.y + bounds.y)) < 10.0 {
						state.dragging = Some(DraggingInfo { 
							start_node_id: i, 
							last_pos: state.current_mouse_pos, 
						});
						state.dragging_animator.in_if_out();
						state.last_drag_id = Some(i);
						self.current_editing.store(i, std::sync::atomic::Ordering::SeqCst);
						break;
					}
				}
			},
			Event::ButtonReleased(_) => {
				state.dragging = None;
				state.dragging_animator.out_if_in();
			},
			Event::WheelScrolled { delta } => {
				let mut biquad_settings = if let Ok(inner) = self.biquad_settings.write() {
					inner
				}else {
					return None;
				};

				let delta = match delta {
					ScrollDelta::Lines { y, .. } => *y * 16.0,
					ScrollDelta::Pixels { y, .. } => *y,
				};

				if let Some(info) = &mut state.dragging {
					let q = biquad_settings[info.start_node_id].get_q();
					biquad_settings[info.start_node_id].set_q(q + delta / 100.0);
					biquad_settings[info.start_node_id].modify(&self.biquad_filter_map, info.start_node_id, self.sample_rate);
				}
			}
			_ => {}
		}

		action
	}
}