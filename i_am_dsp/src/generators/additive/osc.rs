//! The additive oscillator itself.

use std::f32::consts::PI;

use i_am_dsp_derive::Parameters;
use wide::f32x4;

use crate::prelude::Oscillator;

use super::{FreqInfo, GenFreqInfo, UNDAMPED_FACTOR, Wavetable};

/// A type of oscillator that adds up multiple sine waves
#[derive(Parameters)]
pub struct AdditiveOsc<Freq: GenFreqInfo, const MAX_SINES: usize = 64, const CHANNELS: usize = 2> {
	#[sub_param]
	/// The generator that says where the partials sit and how loud they are.
	#[id(display = "")]
	freq_gen: Freq,
	#[skip]
	pub(crate) calculated_ratios: [f32; MAX_SINES],
	#[skip]
	calculated_amplitudes: [f32; MAX_SINES],
	#[skip]
	calculated_dampings: [f32; MAX_SINES],
	#[skip]
	calculated_phases: [f32; MAX_SINES],
	#[skip]
	max_ratio: f32,
	#[skip]
	min_ratio: f32,
	#[skip]
	max_amplitude: f32,
	/// The highest partial ratio that still fits below the Nyquist frequency of the played note.
	///
	/// It stays infinite until a driver tells us the pitch through set_pitch, which leaves the
	/// oscillator unbandlimited for callers that only want the raw sum of the partials.
	#[skip]
	max_harmonic: f32,
	/// One period of the partials, used when every ratio is an integer and nothing is damped.
	///
	/// It is rebuilt by new, recaculate_ratio and rebuild_table, and ignored while num_sines does
	/// not match the number of partials it was built from.
	#[skip]
	table: Option<Wavetable>,
	/// Whether any partial decays over time, which rules out the wavetable and the `powf` shortcut.
	#[skip]
	has_damping: bool,
	/// How long a rebuilt table crossfades from the one it replaces, in seconds.
	///
	/// Zero keeps the old behaviour: a new table is played from the next sample.
	#[skip]
	rebuild_fade: f32,
	/// The table the current one replaced, while the crossfade lasts.
	#[skip]
	previous_table: Option<Wavetable>,
	/// When the crossfade started, in seconds; NaN until the next sample sets it.
	#[skip]
	fade_start: std::sync::atomic::AtomicU32,
	/// The number of sine waves to add up
	#[range(min = 1, max = 64)]
	#[default(32)]
	#[id(display = "Partials")]
	#[layout(widget = knob, hero = "additive")]
	pub num_sines: usize,
	#[range(min = 0.0, max = 4.0)]
	#[logarithmic]
	#[default(1.0)]
	#[id(display = "Level")]
	/// The gain of the oscillator, which saves in linear scale
	pub gain: f32,
	/// The generator revision the partial table was built from.
	#[skip]
	last_revision: u64,
	/// The partial count the table was built for.
	#[skip]
	last_num_sines: usize,
}

impl<Freq: GenFreqInfo, const MAX_SINES: usize, const CHANNELS: usize>
	AdditiveOsc<Freq, MAX_SINES, CHANNELS>
{
	/// Creates a new AdditiveOsc with the given frequency generator and number of sine waves
	///
	/// # Panics
	///
	/// 1. If the number of sine waves is greater than `MAX_SINES`
	/// 2. If the number of sine waves is 0
	/// 3. If the number of channels is 0
	pub fn new(mut freq_gen: Freq, num_sines: usize) -> Self {
		assert!(num_sines <= MAX_SINES);
		assert!(MAX_SINES > 0);
		assert!(CHANNELS > 0);
		let index_0 = freq_gen.gen_info(0, MAX_SINES);
		let mut max_ratio = index_0.ratio;
		let mut min_ratio = max_ratio;
		let mut max_amplitude = index_0.amplitude;
		let calculated_ratios: [FreqInfo; MAX_SINES] = core::array::from_fn(|i| {
			let output = freq_gen.gen_info(i, MAX_SINES);
			max_ratio = max_ratio.max(output.ratio);
			min_ratio = min_ratio.min(output.ratio);
			max_amplitude = max_amplitude.max(output.amplitude);
			output
		});

		let calculated_amplitudes = core::array::from_fn(|i| calculated_ratios[i].amplitude);

		let calculated_dampings = core::array::from_fn(|i| (-calculated_ratios[i].damping).exp());

		let calculated_phases = core::array::from_fn(|i| calculated_ratios[i].phase);

		let calculated_ratios = core::array::from_fn(|i| calculated_ratios[i].ratio);

		let last_revision = freq_gen.revision();
		let mut output = Self {
			freq_gen,
			calculated_ratios,
			calculated_amplitudes,
			calculated_dampings,
			calculated_phases,
			max_amplitude,
			max_ratio,
			min_ratio: min_ratio.max(1.0),
			table: None,
			has_damping: false,
			rebuild_fade: 0.0,
			previous_table: None,
			fade_start: std::sync::atomic::AtomicU32::new(f32::NAN.to_bits()),
			max_harmonic: f32::INFINITY,
			num_sines,
			gain: 1.0,
			last_revision,
			last_num_sines: num_sines,
		};
		output.update_damping_cache();
		output.rebuild_table();
		output
	}

	/// The generator that says where the partials sit and how loud they are.
	pub fn freq_gen(&self) -> &Freq {
		&self.freq_gen
	}

	/// The generator, mutably.
	///
	/// A caller that changes it outside the parameter surface (a script arriving
	/// from the page, say) has to call [`Self::refresh`] or let the oscillator's
	/// own sync see the generator's revision change.
	pub fn freq_gen_mut(&mut self) -> &mut Freq {
		&mut self.freq_gen
	}

	/// Changes the frequency generator of the oscillator
	pub fn change_freq_gen(&mut self, freq_gen: Freq) {
		self.freq_gen = freq_gen;
		self.last_revision = self.freq_gen.revision();
		self.recaculate_ratio();
	}

	/// Rebuilds whatever the current controls imply, if they moved.
	///
	/// The generator's revision is a cheap hash of its controls, so calling this
	/// every sample costs a few compares and no work at all while nothing moves.
	pub fn refresh(&mut self) {
		let revision = self.freq_gen.revision();

		if revision != self.last_revision {
			self.last_revision = revision;
			self.recaculate_ratio();
		}

		if self.num_sines != self.last_num_sines {
			self.rebuild_table();
		}
	}

	/// Recalculates the ratio and amplitude of the oscillator
	pub fn recaculate_ratio(&mut self) {
		let index_0 = self.freq_gen.gen_info(0, MAX_SINES);
		let mut max_ratio = index_0.ratio;
		let mut min_ratio = max_ratio;
		let mut max_amplitude = index_0.amplitude;

		let calculated_ratios: [FreqInfo; MAX_SINES] = core::array::from_fn(|i| {
			let output = self.freq_gen.gen_info(i, MAX_SINES);
			max_ratio = max_ratio.max(output.ratio);
			min_ratio = min_ratio.min(output.ratio);
			max_amplitude = max_amplitude.max(output.amplitude);
			output
		});

		let calculated_amplitudes = core::array::from_fn(|i| calculated_ratios[i].amplitude);

		let calculated_dampings = core::array::from_fn(|i| (-calculated_ratios[i].damping).exp());

		let calculated_phases = core::array::from_fn(|i| calculated_ratios[i].phase);

		let calculated_ratios = core::array::from_fn(|i| calculated_ratios[i].ratio);

		self.max_ratio = max_ratio;
		self.min_ratio = min_ratio.max(1.0);
		self.calculated_phases = calculated_phases;
		self.calculated_dampings = calculated_dampings;
		self.calculated_ratios = calculated_ratios;
		self.calculated_amplitudes = calculated_amplitudes;
		self.max_amplitude = max_amplitude;
		self.update_damping_cache();
		self.rebuild_table();
	}

	/// Crossfades a rebuilt table from the one it replaces, for the given seconds.
	///
	/// The default is zero: a rebuilt table is played from the next sample. A script
	/// whose four knobs the host automates can set this so a knob move glides
	/// instead of swapping the waveform between two samples.
	pub fn set_rebuild_fade(&mut self, seconds: f32) {
		self.rebuild_fade = seconds.max(0.0);
	}

	/// Rebuilds the wavetable the fast path plays from.
	///
	/// Call it after changing num_sines directly. Until it is called the oscillator sums its
	/// partials one by one, which is correct but slower.
	pub fn rebuild_table(&mut self) {
		self.last_num_sines = self.num_sines;
		let previous = self.table.take();
		let table = if self.has_damping {
			None
		} else {
			Wavetable::build(
				&self.calculated_ratios,
				&self.calculated_amplitudes,
				&self.calculated_phases,
				self.num_sines.min(MAX_SINES),
			)
		};

		self.table = table;

		if self.rebuild_fade > 0.0 && previous.is_some() && self.table.is_some() {
			self.previous_table = previous;
			self.fade_start
				.store(f32::NAN.to_bits(), std::sync::atomic::Ordering::Relaxed);
		} else {
			self.previous_table = None;
		}
	}

	/// Reads a group of amplitudes, with the partials above the pitch limit set to zero.
	#[inline]
	fn band_limit(&self, amplitudes: &[f32], ratios: f32x4, max_harmonic: f32x4) -> f32x4 {
		ratios
			.simd_gt(max_harmonic)
			.select(f32x4::ZERO, f32x4::from(amplitudes))
	}

	/// Caches whether any partial decays over time, which costs a `powf` per partial per sample.
	fn update_damping_cache(&mut self) {
		self.has_damping = self
			.calculated_dampings
			.iter()
			.any(|damping| *damping != UNDAMPED_FACTOR);
	}

	/// Applies a per channel phase offset to the summed sine and cosine of all partials.
	#[inline]
	fn apply_channel_phase(
		&self,
		sine: f32,
		cosine: f32,
		phase: [f32; CHANNELS],
	) -> [f32; CHANNELS] {
		let mut output = [0.0; CHANNELS];

		for (output, phase) in output.iter_mut().zip(phase) {
			let angle = 2.0 * PI * phase;
			let (sin, cos) = angle.sin_cos();
			*output = (sine * cos + cosine * sin) * self.gain;
		}

		output
	}

	/// Plays every partial from the wavetable, which already holds their sum.
	#[inline]
	fn play_table(
		&self,
		table: &Wavetable,
		frequency: f32,
		time: f32,
		phase: [f32; CHANNELS],
	) -> [f32; CHANNELS] {
		let (sine, cosine) = table.play(time * frequency, self.max_harmonic);

		self.apply_channel_phase(sine, cosine, phase)
	}

	/// Sums the partials one sine at a time.
	///
	/// This is the fallback for partial sets that are not one period long. The sine and cosine of a
	/// partial are only computed once for all channels: a per channel phase is an angle offset, so
	/// it can be applied to the two sums afterwards.
	fn play_sines(&self, frequency: f32, time: f32, phase: [f32; CHANNELS]) -> [f32; CHANNELS] {
		let len = self
			.num_sines
			.min(MAX_SINES)
			.min(self.calculated_ratios.len());
		let max_harmonic = f32x4::splat(self.max_harmonic);

		if self.has_damping {
			let mut output = [0.0; CHANNELS];

			for (i, output) in output.iter_mut().enumerate() {
				for j in (0..len).step_by(4) {
					let end = (j + 4).min(len);
					let ratio = f32x4::from(&self.calculated_ratios[j..end]);
					let freq = ratio * frequency;
					let phase_freq = f32x4::from(&self.calculated_phases[j..end]);
					let t = time * freq + phase[i] + phase_freq;
					let amp =
						self.band_limit(&self.calculated_amplitudes[j..end], ratio, max_harmonic)
							* self.gain;
					let damping = f32x4::from(&self.calculated_dampings[j..end]).powf_simd(t);

					*output += ((t * 2.0 * PI).sin() * amp * damping).reduce_add();
				}
			}

			return output;
		}

		let mut sum_sine = f32x4::ZERO;
		let mut sum_cosine = f32x4::ZERO;

		for j in (0..len).step_by(4) {
			let end = (j + 4).min(len);
			let ratio = f32x4::from(&self.calculated_ratios[j..end]);
			let freq = ratio * frequency;
			let phase_freq = f32x4::from(&self.calculated_phases[j..end]);
			let t = (time * freq + phase_freq) * f32x4::splat(2.0 * PI);
			let amp = self.band_limit(&self.calculated_amplitudes[j..end], ratio, max_harmonic);
			let (sin, cos) = t.sin_cos();

			sum_sine = sin.mul_add(amp, sum_sine);
			sum_cosine = cos.mul_add(amp, sum_cosine);
		}

		self.apply_channel_phase(sum_sine.reduce_add(), sum_cosine.reduce_add(), phase)
	}
}

impl<Freq: GenFreqInfo, const MAX_SINES: usize, const CHANNELS: usize> Oscillator<CHANNELS>
	for AdditiveOsc<Freq, MAX_SINES, CHANNELS>
{
	fn play_at(&self, frequency: f32, time: f32, phase: [f32; CHANNELS]) -> [f32; CHANNELS] {
		let num_sines = self.num_sines.min(MAX_SINES);

		if let Some(table) = self.table.as_ref()
			&& table.num_sines == num_sines
		{
			if let Some(previous) = self
				.previous_table
				.as_ref()
				.filter(|previous| previous.num_sines == num_sines)
			{
				let start =
					f32::from_bits(self.fade_start.load(std::sync::atomic::Ordering::Relaxed));
				let weight = if start.is_nan() {
					self.fade_start
						.store(time.to_bits(), std::sync::atomic::Ordering::Relaxed);
					0.0
				} else if time < start {
					// A new note: the table the previous one played is not this note's.
					1.0
				} else {
					((time - start) / self.rebuild_fade).clamp(0.0, 1.0)
				};

				if weight < 1.0 {
					let fresh = self.play_table(table, frequency, time, phase);
					let old = self.play_table(previous, frequency, time, phase);

					return core::array::from_fn(|channel| {
						old[channel] + (fresh[channel] - old[channel]) * weight
					});
				}
			}

			return self.play_table(table, frequency, time, phase);
		}

		self.play_sines(frequency, time, phase)
	}

	fn set_pitch(&mut self, frequency: f32, sample_rate: usize) {
		// Everything above this ratio folds back into the audible band at this pitch.
		self.max_harmonic = if frequency > 0.0 && sample_rate > 0 {
			sample_rate as f32 / (2.0 * frequency)
		} else {
			f32::INFINITY
		};
	}

	fn sync(&mut self) {
		self.refresh();
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		use crate::tools::ui_tools::gain_ui;
		use egui::emath::RectTransform;
		use egui::*;

		Resize::default()
			.max_width(ui.available_width())
			.min_width(ui.available_width())
			.id_salt(format!("{}_additive_osc_ratio", id_prefix))
			.show(ui, |ui| {
				Frame::canvas(ui.style()).show(ui, |ui| {
					let (_, rect) = ui.allocate_space(ui.available_size());
					let to_screen =
						RectTransform::from_to(Rect::from_x_y_ranges(0.0..=1.0, -1.0..=1.0), rect);

					if MAX_SINES == 1 {
						let y = self.calculated_amplitudes[0];
						let pos_1 = to_screen * pos2(0.5, y);
						let pos_2 = to_screen * pos2(0.5, 0.0);
						ui.painter()
							.line_segment([pos_1, pos_2], (3.0, Color32::WHITE));
						return;
					}

					let mid_ratio = (self.max_ratio * self.min_ratio).sqrt();
					let min_ratio = self.min_ratio.min(mid_ratio / 2.0);
					let max_ratio = self.max_ratio.max(mid_ratio * 2.0);
					let max_amp = self.max_amplitude;

					ui.painter().extend(
						self.calculated_ratios
							.iter()
							.zip(self.calculated_amplitudes.iter())
							.take(self.num_sines)
							.map(|(ratio, amplitude)| {
								let lerped_ratio = (ratio.ln() - min_ratio.ln())
									/ (max_ratio.ln() - min_ratio.ln());
								let x = lerped_ratio * 0.99 + 0.005;
								let y = -amplitude / max_amp;
								Shape::line_segment(
									[to_screen * pos2(x, y), to_screen * pos2(x, 0.0)],
									(3.0, Color32::WHITE),
								)
							}),
					);
				});
			});

		if self.freq_gen.demo_ui(ui, format!("{}_freq_gen", id_prefix)) {
			self.recaculate_ratio();
		}

		ui.horizontal(|ui| {
			ui.label(format!("Max Ratio: {:.2}", self.max_ratio));
			ui.label(format!("Min Ratio: {:.2}", self.min_ratio));
			let num_sines = self.num_sines;
			ui.add(Slider::new(&mut self.num_sines, 1..=MAX_SINES).text("Num Sines"));
			if self.num_sines != num_sines {
				self.rebuild_table();
			}
			gain_ui(ui, &mut self.gain, None, false);
		});
	}
}

#[cfg(test)]
mod tests;
