//! The oscillators and tables built on other ones.

use i_am_dsp_derive::Parameters;

use crate::prelude::{Parameter, Parameters, SetValue, bend, page_position};

use super::{Oscillator, TableOsc, Warp, WaveTable};

/// A wave table that smoothly transitions between multiple wave tables.
#[derive(Parameters)]
pub struct OscillatorSmoother<const CHANNELS: usize> {
	/// The oscillators to smooth between.
	#[sub_param]
	pub oscillators: Vec<Box<dyn Oscillator<CHANNELS> + Send + Sync>>,
	/// should be between 0 and 1, where 0 is first table and 1 is last table.
	pub smooth_factor: f32,

	#[cfg(feature = "real_time_demo")]
	current_oscillator: usize,
}

impl<const CHANNELS: usize> OscillatorSmoother<CHANNELS> {
	/// Sample the wave table at time t, t in the range [0, 1]
	pub fn sample(&self, t: f32, phase: [f32; CHANNELS]) -> [f32; CHANNELS] {
		if self.oscillators.is_empty() {
			return [0.0; CHANNELS];
		} else if self.oscillators.len() == 1 {
			return self.oscillators[0].play_at(1.0, t, phase);
		}

		let (from, to, fraction) = page_position(self.smooth_factor, self.oscillators.len());
		let from_sample = self.oscillators[from].play_at(1.0, t, phase);

		if fraction <= f32::EPSILON || from == to {
			return from_sample;
		}

		let to_sample = self.oscillators[to].play_at(1.0, t, phase);
		let mut output = [0.0; CHANNELS];

		for i in 0..CHANNELS {
			output[i] = from_sample[i] + (to_sample[i] - from_sample[i]) * fraction;
		}

		output
	}
}

impl<const CHANNELS: usize> Oscillator<CHANNELS> for OscillatorSmoother<CHANNELS> {
	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		use crate::tools::ui_tools::draw_wavetable;
		egui::Resize::default()
			.resizable([false, true])
			.min_width(ui.available_width())
			.max_width(ui.available_width())
			.id_salt(format!("{id_prefix}_wavtable_smoother"))
			.show(ui, |ui| {
				draw_wavetable(ui, |t| {
					self.sample(t, [0.0; CHANNELS])
						.iter()
						.map(|inner| *inner * *inner)
						.sum::<f32>()
						.sqrt()
				});
			});

		ui.horizontal(|ui| {
			ui.add(egui::Slider::new(&mut self.smooth_factor, 0.0..=1.0).text("Smooth Factor"));
			if self.oscillators.is_empty() {
				return;
			}
			ui.label(format!("current oscillator: {}", self.current_oscillator));
			ui.menu_button("Select Osc", |ui| {
				for i in 0..self.oscillators.len() {
					if ui.button(format!("Oscillator {}", i)).clicked() {
						self.current_oscillator = i;
						ui.close();
					}
				}
			});
		});

		if let Some(osc) = self.oscillators.get_mut(self.current_oscillator) {
			osc.demo_ui(ui, format!("{id_prefix}_oscillator_smoother_oscillator"));
		} else {
			self.current_oscillator = 0;
		}
	}

	fn play_at(&self, frequency: f32, time: f32, phase: [f32; CHANNELS]) -> [f32; CHANNELS] {
		let t = (time * frequency) % 1.0;
		self.sample(t, phase)
	}
}

impl<const CHANNELS: usize> OscillatorSmoother<CHANNELS> {
	/// Create a new [`OscillatorSmoother`] from a vector of oscillators and a smooth factor.
	pub fn new(
		oscillators: Vec<Box<dyn Oscillator<CHANNELS> + Send + Sync>>,
		smooth_factor: f32,
	) -> Self {
		Self {
			oscillators,
			smooth_factor,

			#[cfg(feature = "real_time_demo")]
			current_oscillator: 0,
		}
	}
}

/// A phase-modulation oscillator: the modulator bends the carrier's phase.
#[derive(Parameters)]
pub struct PmOsc<Carrier: WaveTable, Modulator: WaveTable> {
	/// The carrier wave.
	#[sub_param]
	pub carrier: TableOsc<Carrier>,
	/// The modulator wave.
	#[sub_param]
	pub modulator: TableOsc<Modulator>,
	/// How far the modulator pushes the carrier's phase, in cycles.
	#[range(min = 0.0, max = 8.0)]
	#[layout(group = "Modulation", order = 0, widget = knob)]
	pub index: f32,
	/// How fast the modulator runs against the carrier.
	#[range(min = 1.0, max = 16.0)]
	#[layout(group = "Modulation", order = 1, widget = knob, step = 1.0)]
	pub ratio: f32,
}

impl<Carrier: WaveTable, Modulator: WaveTable> PmOsc<Carrier, Modulator> {
	/// Create a new [`PmOsc`] from a carrier and modulator wave.
	pub fn new(carrier: Carrier, modulator: Modulator) -> Self {
		Self {
			carrier: TableOsc(carrier),
			modulator: TableOsc(modulator),
			index: 0.0,
			ratio: 1.0,
		}
	}
}

impl<Carrier: WaveTable, Modulator: WaveTable, const CHANNELS: usize> Oscillator<CHANNELS>
	for PmOsc<Carrier, Modulator>
{
	fn play_at(&self, frequency: f32, time: f32, phase: [f32; CHANNELS]) -> [f32; CHANNELS] {
		if self.index <= 0.0005 {
			return self.carrier.play_at(frequency, time, phase);
		}

		// The modulator is a phase offset in cycles, so the carrier is played
		// with the offset added rather than with a shifted frequency.
		let modulator = self.modulator.play_at(frequency * self.ratio, time, phase);
		let mut shifted = phase;

		for (channel, value) in shifted.iter_mut().enumerate() {
			*value += modulator[channel] * self.index;
		}

		self.carrier.play_at(frequency, time, shifted)
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		Oscillator::<CHANNELS>::demo_ui(
			&mut self.carrier,
			ui,
			format!("{id_prefix}_pm_osc_carrier"),
		);
		Oscillator::<CHANNELS>::demo_ui(
			&mut self.modulator,
			ui,
			format!("{id_prefix}_pm_osc_modulator"),
		);

		ui.add(egui::Slider::new(&mut self.index, 0.0..=8.0).text("PM Index"));
		ui.add(
			egui::Slider::new(&mut self.ratio, 1.0..=16.0)
				.text("Modulator Ratio")
				.integer(),
		);
	}
}

/// An amplitude-modulation oscillator: the modulator scales the carrier's level.
#[derive(Parameters)]
pub struct AmOsc<Carrier: WaveTable, Modulator: WaveTable> {
	/// The carrier wave.
	#[sub_param]
	pub carrier: TableOsc<Carrier>,
	/// The modulator wave.
	#[sub_param]
	pub modulator: TableOsc<Modulator>,
	/// How much of the modulator reaches the carrier's level.
	#[range(min = 0.0, max = 1.0)]
	#[layout(group = "Modulation", order = 0, widget = knob)]
	pub depth: f32,
	/// How fast the modulator runs against the carrier.
	#[range(min = 1.0, max = 16.0)]
	#[layout(group = "Modulation", order = 1, widget = knob, step = 1.0)]
	pub ratio: f32,
}

impl<Carrier: WaveTable, Modulator: WaveTable> AmOsc<Carrier, Modulator> {
	/// Create a new [`AmOsc`] from a carrier and modulator wave.
	pub fn new(carrier: Carrier, modulator: Modulator) -> Self {
		Self {
			carrier: TableOsc(carrier),
			modulator: TableOsc(modulator),
			depth: 0.0,
			ratio: 1.0,
		}
	}
}

impl<Carrier: WaveTable, Modulator: WaveTable, const CHANNELS: usize> Oscillator<CHANNELS>
	for AmOsc<Carrier, Modulator>
{
	fn play_at(&self, frequency: f32, time: f32, phase: [f32; CHANNELS]) -> [f32; CHANNELS] {
		if self.depth <= 0.0005 {
			return self.carrier.play_at(frequency, time, phase);
		}

		// A bipolar modulator at full depth reaches silence, which is what makes
		// this modulation rather than a tremolo around a fixed level.
		let modulator = self.modulator.play_at(frequency * self.ratio, time, phase);
		let mut output = self.carrier.play_at(frequency, time, phase);

		for (channel, value) in output.iter_mut().enumerate() {
			*value *= 1.0 + modulator[channel] * self.depth;
		}

		output
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		Oscillator::<CHANNELS>::demo_ui(
			&mut self.carrier,
			ui,
			format!("{id_prefix}_am_osc_carrier"),
		);
		Oscillator::<CHANNELS>::demo_ui(
			&mut self.modulator,
			ui,
			format!("{id_prefix}_am_osc_modulator"),
		);

		ui.add(egui::Slider::new(&mut self.depth, 0.0..=1.0).text("AM Depth"));
		ui.add(
			egui::Slider::new(&mut self.ratio, 1.0..=16.0)
				.text("Modulator Ratio")
				.integer(),
		);
	}
}

/// A ring-modulation oscillator: the modulator multiplies the carrier.
#[derive(Parameters)]
pub struct RmOsc<Carrier: WaveTable, Modulator: WaveTable> {
	/// The carrier wave.
	#[sub_param]
	pub carrier: TableOsc<Carrier>,
	/// The modulator wave.
	#[sub_param]
	pub modulator: TableOsc<Modulator>,
	/// The blend from the carrier to the product of the two waves.
	#[range(min = 0.0, max = 1.0)]
	#[layout(group = "Modulation", order = 0, widget = knob)]
	pub depth: f32,
	/// How fast the modulator runs against the carrier.
	#[range(min = 1.0, max = 16.0)]
	#[layout(group = "Modulation", order = 1, widget = knob, step = 1.0)]
	pub ratio: f32,
}

impl<Carrier: WaveTable, Modulator: WaveTable> RmOsc<Carrier, Modulator> {
	/// Create a new [`RmOsc`] from a carrier and modulator wave.
	pub fn new(carrier: Carrier, modulator: Modulator) -> Self {
		Self {
			carrier: TableOsc(carrier),
			modulator: TableOsc(modulator),
			depth: 0.0,
			ratio: 1.0,
		}
	}
}

impl<Carrier: WaveTable, Modulator: WaveTable, const CHANNELS: usize> Oscillator<CHANNELS>
	for RmOsc<Carrier, Modulator>
{
	fn play_at(&self, frequency: f32, time: f32, phase: [f32; CHANNELS]) -> [f32; CHANNELS] {
		if self.depth <= 0.0005 {
			return self.carrier.play_at(frequency, time, phase);
		}

		let modulator = self.modulator.play_at(frequency * self.ratio, time, phase);
		let mut output = self.carrier.play_at(frequency, time, phase);

		for (channel, value) in output.iter_mut().enumerate() {
			*value *= 1.0 - self.depth + self.depth * modulator[channel];
		}

		output
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		Oscillator::<CHANNELS>::demo_ui(
			&mut self.carrier,
			ui,
			format!("{id_prefix}_rm_osc_carrier"),
		);
		Oscillator::<CHANNELS>::demo_ui(
			&mut self.modulator,
			ui,
			format!("{id_prefix}_rm_osc_modulator"),
		);

		ui.add(egui::Slider::new(&mut self.depth, 0.0..=1.0).text("Ring Depth"));
		ui.add(
			egui::Slider::new(&mut self.ratio, 1.0..=16.0)
				.text("Modulator Ratio")
				.integer(),
		);
	}
}

/// A wave table that smoothly transitions between 4 multiple wave tables.
#[derive(Parameters)]
pub struct BilinearSmoother<A: WaveTable, B: WaveTable, C: WaveTable, D: WaveTable> {
	/// The first wave table.
	#[sub_param]
	pub a: A,
	/// The second wave table.
	#[sub_param]
	pub b: B,
	/// The third wave table.
	#[sub_param]
	pub c: C,
	/// The fourth wave table.
	#[sub_param]
	pub d: D,
	/// The smoothing factor, should be between 0 and 1.
	pub smooth_factor_1: f32,
	/// The smoothing factor, should be between 0 and 1.
	pub smooth_factor_2: f32,
}

impl<A: WaveTable, B: WaveTable, C: WaveTable, D: WaveTable> BilinearSmoother<A, B, C, D> {
	/// Create a new [`BilinearSmoother`] from four wave tables and two smoothing factors.
	pub fn new(a: A, b: B, c: C, d: D) -> Self {
		Self {
			a,
			b,
			c,
			d,
			smooth_factor_1: 0.0,
			smooth_factor_2: 0.0,
		}
	}
}

impl<const CHANNELS: usize, A: WaveTable, B: WaveTable, C: WaveTable, D: WaveTable>
	Oscillator<CHANNELS> for BilinearSmoother<A, B, C, D>
{
	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		use crate::tools::ui_tools::draw_wavetable;

		egui::Resize::default()
			.resizable([false, true])
			.min_width(ui.available_width())
			.max_width(ui.available_width())
			.id_salt(format!("{id_prefix}_wavtable_smoother"))
			.show(ui, |ui| {
				draw_wavetable(ui, |t| self.sample(t, 0));
			});

		ui.horizontal(|ui| {
			ui.add(egui::Slider::new(&mut self.smooth_factor_1, 0.0..=1.0).text("Smooth Factor 1"));
			ui.add(egui::Slider::new(&mut self.smooth_factor_2, 0.0..=1.0).text("Smooth Factor 2"));
		});
	}

	fn play_at(&self, frequency: f32, time: f32, phase: [f32; CHANNELS]) -> [f32; CHANNELS] {
		let mut output = [0.0; CHANNELS];

		for i in 0..CHANNELS {
			let t = (time * frequency + phase[i]).rem_euclid(1.0);
			output[i] = self.sample(t, i);
		}

		output
	}
}

impl<A: WaveTable, B: WaveTable, C: WaveTable, D: WaveTable> WaveTable
	for BilinearSmoother<A, B, C, D>
{
	fn sample(&self, t: f32, channel: usize) -> f32 {
		let a = self.a.sample(t, channel);
		let b = self.b.sample(t, channel);
		let c = self.c.sample(t, channel);
		let d = self.d.sample(t, channel);

		let p = self.smooth_factor_1.clamp(0.0, 1.0);
		let q = self.smooth_factor_2.clamp(0.0, 1.0);

		(1.0 - p) * (1.0 - q) * a + (1.0 - p) * q * b + p * (1.0 - q) * c + p * q * d
	}
}

/// How many points one cycle of the integrated modulator keeps.
const FM_POINTS: usize = 2_048;
/// How often the modulator's shape is checked for a change.
const FM_RECHECK: usize = 256;

/// An FM oscillator: the integral of the modulator bends the carrier's phase.
///
/// The modulator is a table, so its integral over one cycle is a table too:
/// build it once, scale it by the depth, and add it to the carrier's phase. That
/// is frequency modulation rather than the phase modulation `PmOsc` does, and
/// it needs no history because the integral of a table is another table.
#[derive(Parameters)]
pub struct FmOsc<Carrier: WaveTable, Modulator: WaveTable> {
	/// The carrier wave.
	#[sub_param]
	pub carrier: TableOsc<Carrier>,
	/// The modulator wave.
	#[sub_param]
	pub modulator: TableOsc<Modulator>,
	/// How far the integrated modulator pushes the carrier, in cycles.
	#[range(min = 0.0, max = 16.0)]
	#[layout(group = "Modulation", order = 0, widget = knob)]
	pub depth: f32,
	/// How fast the modulator runs against the carrier.
	#[range(min = 1.0, max = 16.0)]
	#[layout(group = "Modulation", order = 1, widget = knob, step = 1.0)]
	pub ratio: f32,
	/// One cycle of the modulator's integral, in cycles of the carrier.
	#[skip]
	integral: [f32; FM_POINTS],
	/// The modulator shape the integral was built from.
	#[skip]
	seen: u64,
	#[skip]
	age: usize,
}

impl<Carrier: WaveTable, Modulator: WaveTable> FmOsc<Carrier, Modulator> {
	/// Create a new [`FmOsc`] from a carrier and modulator wave.
	pub fn new(carrier: Carrier, modulator: Modulator) -> Self {
		let mut fm = Self {
			carrier: TableOsc(carrier),
			modulator: TableOsc(modulator),
			depth: 0.0,
			ratio: 1.0,
			integral: [0.0; FM_POINTS],
			seen: 0,
			age: 0,
		};

		fm.seen = fm.fingerprint();
		fm.rebuild();

		fm
	}

	/// A cheap sample of the modulator's shape, to notice a parameter move.
	///
	/// The integral is only valid for the shape it was built from, and the
	/// modulator may be a table with its own controls (a drawn LFO, an imported
	/// wave). Sixty-four points are enough for that shape check and cost nothing
	/// per sample when spread over the recheck interval.
	fn fingerprint(&self) -> u64 {
		let mut hash = 0xcbf2_9ce4_8422_2325u64;

		for index in 0..64 {
			let value = self.modulator.0.sample(index as f32 / 64.0, 0);

			hash ^= value.to_bits() as u64;
			hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
		}

		hash
	}

	/// Rebuilds the integral table, in place, from the modulator's current shape.
	///
	/// The mean is removed first: an integral that does not return to zero would
	/// shift the carrier's pitch by a constant and leave a jump at the cycle end.
	fn rebuild(&mut self) {
		let mut running = 0.0f32;
		let mut total = 0.0f32;

		for index in 0..FM_POINTS {
			let value = self.modulator.0.sample(index as f32 / FM_POINTS as f32, 0);

			running += value;
			total += value;
			self.integral[index] = running;
		}

		let mean = total / FM_POINTS as f32;
		let scale = 1.0 / FM_POINTS as f32;

		for (index, value) in self.integral.iter_mut().enumerate() {
			let unit = (index + 1) as f32 / FM_POINTS as f32;

			*value = *value * scale - mean * unit;
		}
	}

	/// The integral at a position in the cycle, linearly interpolated.
	fn integral_at(&self, position: f32) -> f32 {
		let scaled = position * FM_POINTS as f32;
		let index = scaled as usize % FM_POINTS;
		let next = (index + 1) % FM_POINTS;
		let fraction = scaled - scaled.floor();

		self.integral[index] + (self.integral[next] - self.integral[index]) * fraction
	}
}

impl<Carrier: WaveTable, Modulator: WaveTable, const CHANNELS: usize> Oscillator<CHANNELS>
	for FmOsc<Carrier, Modulator>
{
	fn play_at(&self, frequency: f32, time: f32, phase: [f32; CHANNELS]) -> [f32; CHANNELS] {
		if self.depth <= 0.0 {
			return self.carrier.play_at(frequency, time, phase);
		}

		let ratio = self.ratio.max(1.0);
		let mut shifted = phase;

		for (channel, value) in shifted.iter_mut().enumerate() {
			let position = (time * frequency * ratio + phase[channel]).rem_euclid(1.0);

			*value += self.depth * self.integral_at(position);
		}

		self.carrier.play_at(frequency, time, shifted)
	}

	fn sync(&mut self) {
		self.age = self.age.wrapping_add(1);

		if !self.age.is_multiple_of(FM_RECHECK) {
			return;
		}

		let fingerprint = self.fingerprint();

		if fingerprint != self.seen {
			self.seen = fingerprint;
			self.rebuild();
		}
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		Oscillator::<CHANNELS>::demo_ui(
			&mut self.carrier,
			ui,
			format!("{id_prefix}_fm_osc_carrier"),
		);
		Oscillator::<CHANNELS>::demo_ui(
			&mut self.modulator,
			ui,
			format!("{id_prefix}_fm_osc_modulator"),
		);

		ui.add(egui::Slider::new(&mut self.depth, 0.0..=16.0).text("FM Depth"));
		ui.add(
			egui::Slider::new(&mut self.ratio, 1.0..=16.0)
				.text("Modulator Ratio")
				.integer(),
		);
	}
}

/// A warp oscillator: the warp decides where in the table each position reads.
///
/// The table is read twice per output sample and decimated, so the edges a warp
/// makes (a fold, a repeated cycle) do not alias as hard as they would at the
/// output rate.
#[derive(Parameters)]
pub struct WarpOsc<Carrier: WaveTable, Warper: Warp> {
	/// The carrier wave.
	#[sub_param]
	pub carrier: TableOsc<Carrier>,
	/// The phase warp.
	#[sub_param]
	pub warper: Warper,
	#[skip]
	sample_rate: usize,
}

impl<Carrier: WaveTable, Warper: Warp> WarpOsc<Carrier, Warper> {
	/// Create a new [`WarpOsc`] from a carrier and a warp.
	pub fn new(carrier: Carrier, warper: Warper) -> Self {
		Self {
			carrier: TableOsc(carrier),
			warper,
			sample_rate: 48_000,
		}
	}

	/// One sample of the warped table at an offset in seconds.
	fn sample_at(&self, frequency: f32, time: f32, phase: f32, channel: usize) -> f32 {
		let t = (time * frequency + phase).rem_euclid(1.0);
		let warped = self.warper.apply(t).rem_euclid(1.0);

		self.carrier.0.sample(warped, channel)
	}
}

impl<Carrier: WaveTable, Warper: Warp, const CHANNELS: usize> Oscillator<CHANNELS>
	for WarpOsc<Carrier, Warper>
{
	fn play_at(&self, frequency: f32, time: f32, phase: [f32; CHANNELS]) -> [f32; CHANNELS] {
		let mut output = [0.0; CHANNELS];

		// A warp can fold or repeat the table, and those edges alias at the
		// output rate. Sampling twice and decimating with a binomial kernel is
		// stateless -- the oscillator is read, not stepped -- so it costs five
		// reads and no history.
		let half = 0.5 / self.sample_rate as f32;
		let kernel = [(-2.0, 1.0), (-1.0, 4.0), (0.0, 6.0), (1.0, 4.0), (2.0, 1.0)];

		for (channel, value) in output.iter_mut().enumerate() {
			let mut total = 0.0;

			for (offset, weight) in kernel {
				total += weight
					* self.sample_at(frequency, time + offset * half, phase[channel], channel);
			}

			*value = total / 16.0;
		}

		output
	}

	fn set_pitch(&mut self, frequency: f32, sample_rate: usize) {
		self.sample_rate = sample_rate.max(1);
		<TableOsc<Carrier> as Oscillator<CHANNELS>>::set_pitch(
			&mut self.carrier,
			frequency,
			sample_rate,
		);
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		use crate::tools::ui_tools::draw_wavetable;

		Oscillator::<CHANNELS>::demo_ui(
			&mut self.carrier,
			ui,
			format!("{id_prefix}_warp_osc_carrier"),
		);
		draw_wavetable(ui, |t| self.warper.apply(t));
	}
}

impl<const CHANNELS: usize> Oscillator<CHANNELS> for Box<dyn Oscillator<CHANNELS> + Send + Sync> {
	fn play_at(&self, frequency: f32, time: f32, phase: [f32; CHANNELS]) -> [f32; CHANNELS] {
		self.as_ref().play_at(frequency, time, phase)
	}

	fn set_pitch(&mut self, frequency: f32, sample_rate: usize) {
		self.as_mut().set_pitch(frequency, sample_rate);
	}

	fn absolute_time(&self) -> bool {
		self.as_ref().absolute_time()
	}

	fn sync(&mut self) {
		self.as_mut().sync();
	}

	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		self.as_mut().demo_ui(ui, id_prefix);
	}
}

impl<const CHANNELS: usize> Parameters for Box<dyn Oscillator<CHANNELS> + Send + Sync> {
	fn get_parameters(&self) -> Vec<Parameter> {
		self.as_ref().get_parameters()
	}

	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		self.as_mut().set_parameter(identifier, value)
	}
}

impl WaveTable for Box<dyn WaveTable + Send + Sync> {
	fn sample(&self, t: f32, channel: usize) -> f32 {
		self.as_ref().sample(t, channel)
	}
}

impl Parameters for Box<dyn WaveTable + Send + Sync> {
	fn get_parameters(&self) -> Vec<Parameter> {
		self.as_ref().get_parameters()
	}

	fn set_parameter(&mut self, identifier: &str, value: crate::prelude::SetValue) -> bool {
		self.as_mut().set_parameter(identifier, value)
	}
}

impl WaveTable for Box<dyn WaveTable> {
	fn sample(&self, t: f32, channel: usize) -> f32 {
		self.as_ref().sample(t, channel)
	}
}

impl Parameters for Box<dyn WaveTable> {
	fn get_parameters(&self) -> Vec<Parameter> {
		self.as_ref().get_parameters()
	}

	fn set_parameter(&mut self, identifier: &str, value: crate::prelude::SetValue) -> bool {
		self.as_mut().set_parameter(identifier, value)
	}
}

/// A editable LFO.
#[derive(Parameters)]
pub struct EditableLfo {
	/// The inner data of current lfo.
	///
	/// the format is (time, value, bend)
	#[serde]
	pub data: Vec<(f32, f32, f32)>,
	#[range(min = 0.01, max = 100.0)]
	#[logarithmic]
	/// The frequency of the LFO.
	#[unit(hertz)]
	pub lfo_frequency: f32,

	/// If true, the LFO will clamp the time to the range [0, 1] rather than wrapping around.
	pub one_shot: bool,

	#[skip]
	#[cfg(feature = "real_time_demo")]
	dragging: Option<(usize, bool)>,

	#[skip]
	#[cfg(feature = "real_time_demo")]
	grid: Option<(usize, usize)>,
}

impl Default for EditableLfo {
	fn default() -> Self {
		Self::new()
	}
}

impl EditableLfo {
	/// Create a new editable LFO.
	pub fn new() -> Self {
		Self {
			data: vec![],
			lfo_frequency: 1.0,
			one_shot: false,

			#[cfg(feature = "real_time_demo")]
			dragging: None,
			#[cfg(feature = "real_time_demo")]
			grid: None,
		}
	}

	/// Get the value of the LFO at the given time.
	///
	/// Will clamp the time and value to the range [0, 1].
	pub fn add_point(&mut self, time: f32, value: f32, bend: f32) {
		let time = time.clamp(0.0, 1.0);
		let value = value.clamp(0.0, 1.0);

		self.data.push((time, value, bend));
		self.data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
	}

	/// Remove the point at the given time.
	pub fn remove_point(&mut self, time: f32) {
		self.data.retain(|p| p.0 != time);
	}

	/// Change the value of the point at the given time.
	///
	/// Will clamp the value to the range [0, 1].
	pub fn change_point_value(&mut self, time: f32, value: f32) {
		let value = value.clamp(0.0, 1.0);
		if let Some(p) = self.data.iter_mut().find(|p| p.0 == time) {
			p.1 = value;
		}
	}

	/// Change the bend of the point at the given time.
	pub fn change_point_bend(&mut self, time: f32, bend: f32) {
		if let Some(p) = self.data.iter_mut().find(|p| p.0 == time) {
			p.2 = bend;
		}
	}

	/// Change the time of the point at the given time.
	///
	/// Will clamp the new time to the range [0, 1].
	pub fn change_point_time(&mut self, old_time: f32, new_time: f32) {
		let new_time = new_time.clamp(0.0, 1.0);
		// assert!((0.0..=1.0).contains(&new_time), "Time must be in the range [0, 1]");

		if old_time == new_time {
			return;
		}

		if let Some(p) = self.data.iter_mut().find(|p| p.0 == old_time) {
			p.0 = new_time;
		}
		self.data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
	}

	#[cfg(feature = "real_time_demo")]
	/// Draws the editable LFO in the demo UI.
	pub fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		use crate::tools::ui_tools::draw_wavetable;
		use egui::*;

		const HEIGHT: f32 = 100.0;
		const RADIUS: f32 = 3.0;

		Resize::default()
			.min_height(HEIGHT)
			.max_height(HEIGHT)
			.id_salt(format!("{id_prefix}_editable_lfo"))
			.show(ui, |ui| {
				let response =
					draw_wavetable(ui, |t| self.sample(t / self.lfo_frequency, 0) * 2.0 - 1.0);

				let rect = response.rect;

				let pointer_position = ui.input(|input| input.pointer.latest_pos());

				let to_screen = emath::RectTransform::from_to(
					Rect::from_x_y_ranges(0.0..=1.0, -1.0..=0.0),
					rect,
				);

				let dragging = response.is_pointer_button_down_on();

				if let Some((row, column)) = &self.grid {
					for i in 0..=*row {
						for j in 0..=*column {
							let x = (j as f32) / *column as f32;
							let y = (i as f32) / *row as f32;
							let row_pos_l = to_screen * pos2(0.0, -y);
							let row_pos_r = to_screen * pos2(1.0, -y);
							ui.painter().line_segment(
								[row_pos_l, row_pos_r],
								(1.0, Color32::from_rgba_unmultiplied(50, 50, 50, 10)),
							);
							let column_pos_t = to_screen * pos2(x, 0.0);
							let column_pos_b = to_screen * pos2(x, -1.0);
							ui.painter().line_segment(
								[column_pos_t, column_pos_b],
								(1.0, Color32::from_rgba_unmultiplied(50, 50, 50, 10)),
							);
						}
					}
				}

				ui.painter_at(rect)
					.extend((0..self.data.len()).flat_map(|i| {
						let (current_time, value, _) = self.data[i];
						let next_time = if i == self.data.len() - 1 {
							1.0
						} else {
							self.data[i + 1].0
						};
						let current_x = current_time;
						let bend_x = (current_x + next_time) / 2.0;
						let current_y = value;
						let bend_y = self.sample(bend_x / self.lfo_frequency, 0);

						let bend_pos = to_screen * pos2(bend_x, -bend_y);
						let current_pos = to_screen * pos2(current_x, -current_y);

						if dragging && let Some(pos) = pointer_position {
							if pos.distance(bend_pos) <= RADIUS * 2.0 && self.dragging.is_none() {
								self.dragging = Some((i, true));
							} else if pos.distance(current_pos) <= RADIUS * 2.0
								&& self.dragging.is_none()
							{
								self.dragging = Some((i, false));
							}
						}

						[
							Shape::circle_stroke(current_pos, RADIUS, (1.0, Color32::WHITE)),
							Shape::circle_stroke(bend_pos, RADIUS, (1.0, Color32::WHITE)),
						]
					}));

				let Some(pointer_position) = pointer_position else {
					return;
				};

				if response.double_clicked() {
					let mut time = (pointer_position.x - rect.left()) / rect.width();
					let mut value = (rect.bottom() - pointer_position.y) / rect.height();
					if let Some((row, column)) = &self.grid
						&& *row != 0
						&& *column != 0
					{
						time = (time * *column as f32).round() / *column as f32;
						value = (value * *row as f32).round() / *row as f32;
					}

					self.add_point(time, value, 0.0);
				}

				if response.clicked_by(PointerButton::Secondary) {
					for (current_time, current_value, _) in self.data.iter() {
						let point_time = *current_time;
						let current_time = (current_time * rect.width()) + rect.left();
						let current_value = (current_value * rect.height()) + rect.top();
						if (pointer_position.x - current_time)
							.hypot(pointer_position.y - current_value)
							<= RADIUS * 2.0
						{
							self.remove_point(point_time);
							break;
						}
					}
				}

				if let Some((i, is_bend)) = self.dragging {
					let time = (pointer_position.x - rect.left()) / rect.width();
					let value = (rect.bottom() - pointer_position.y) / rect.height();
					if is_bend {
						self.change_point_bend(self.data[i].0, value * 20.0 - 10.0);
					} else {
						self.change_point_value(self.data[i].0, value);
						self.change_point_time(self.data[i].0, time);
					}
				}

				if !dragging {
					if let Some((row, column)) = self.grid.take() {
						if let Some((i, is_bend)) = self.dragging.take() {
							if !is_bend {
								let (time, mut value, _) = self.data[i];
								value = (value * row as f32).round() / row as f32;
								self.change_point_value(time, value);
								let new_time = (time * column as f32).round() / column as f32;
								self.change_point_time(time, new_time);
							}
							self.dragging = Some((i, is_bend));
						}
						self.grid = Some((row, column));
					}

					self.dragging = None;
				}
			});

		if ui.selectable_label(self.one_shot, "One Shot").clicked() {
			self.one_shot = !self.one_shot;
		}
		ui.add(
			Slider::new(&mut self.lfo_frequency, 0.01..=100.0)
				.text("LFO Frequency")
				.logarithmic(true),
		);

		if ui.selectable_label(self.grid.is_some(), "Grid").clicked() {
			if self.grid.is_none() {
				self.grid = Some((8, 8));
			} else {
				self.grid = None;
			}
		}

		if let Some((row, column)) = &mut self.grid {
			ui.horizontal(|ui| {
				ui.add(Slider::new(row, 1..=16).text("Grid Rows"));
				ui.add(Slider::new(column, 1..=16).text("Grid Columns"));
			});
		}
	}
}

impl WaveTable for EditableLfo {
	fn sample(&self, t: f32, _: usize) -> f32 {
		let t = t * self.lfo_frequency;
		let t = if self.one_shot {
			t.clamp(0.0, 1.0)
		} else {
			t % 1.0
		};

		let mut value = 0.0;
		let mut bend_amount = 0.0;
		let mut last_time = 0.0;
		let mut current_time = None;
		for (time, v, b) in &self.data {
			if *time > t {
				current_time = Some((*time, *v));
				break;
			}
			value = *v;
			bend_amount = *b;
			last_time = *time;
		}

		let (current_time, current_value) = current_time.unwrap_or((1.0, 0.0));

		// A time past the last point lands on a zero-length segment when the
		// curve ends at 1.0; dividing by it would hand the caller a NaN, which
		// one NaN away from the audio thread is one too many.
		let span = current_time - last_time;
		let t = if span > 0.0 {
			(t - last_time) / span
		} else {
			1.0
		};
		let bended_t = bend(t, bend_amount);
		(current_value - value) * bended_t + value
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// Records the driver calls the boxed forwarding test checks.
	struct Recorder {
		pitches: std::sync::Arc<std::sync::atomic::AtomicUsize>,
		syncs: std::sync::Arc<std::sync::atomic::AtomicUsize>,
	}

	impl Parameters for Recorder {
		fn get_parameters(&self) -> Vec<Parameter> {
			Vec::new()
		}

		fn set_parameter(&mut self, _: &str, _: SetValue) -> bool {
			false
		}
	}

	impl Oscillator<1> for Recorder {
		fn play_at(&self, _: f32, _: f32, phase: [f32; 1]) -> [f32; 1] {
			phase
		}

		fn set_pitch(&mut self, _: f32, _: usize) {
			self.pitches
				.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
		}

		fn sync(&mut self) {
			self.syncs
				.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
		}

		fn absolute_time(&self) -> bool {
			true
		}
	}

	/// A time past the last point lands on a zero-length segment when the curve
	/// ends at 1.0; the sampler must not divide by it, because a NaN here is a
	/// NaN in the modulation.
	#[test]
	fn sampling_the_end_of_a_curve_is_finite() {
		let mut lfo = EditableLfo::new();
		lfo.add_point(0.0, 0.0, 0.0);
		lfo.add_point(1.0, 1.0, 0.0);

		for time in [0.0, 0.5, 0.999_999, 1.0, 1.5] {
			let value = lfo.sample(time, 0);
			assert!(value.is_finite(), "t = {time} gave {value}");
		}
	}

	/// A curve that ends before 1.0 ramps to zero, which is what the sampler's
	/// fallback segment is for. Only a one-shot reaches past the end; a looping
	/// LFO wraps back to the start.
	#[test]
	fn a_curve_that_ends_early_falls_to_zero() {
		let mut lfo = EditableLfo::new();
		lfo.one_shot = true;
		lfo.add_point(0.0, 1.0, 0.0);
		lfo.add_point(0.5, 1.0, 0.0);

		assert!(lfo.sample(0.75, 0) > 0.4 && lfo.sample(0.75, 0) < 0.6);
		assert!(lfo.sample(1.0, 0).abs() < 1e-6);
	}

	/// A negative phase must wrap, not saturate: `%` on a negative float stays
	/// negative, and the table reader turns that into garbage.
	#[test]
	fn a_negative_phase_does_not_fall_to_zero() {
		use crate::generators::wavetable::{Oscillator, TableOsc};

		let osc = TableOsc(vec![1.0f32; 8]);
		let output = Oscillator::<1>::play_at(&osc, 440.0, 0.0, [-0.25]);

		assert!((output[0] - 1.0).abs() < 1e-3, "{output:?}");
	}

	/// The index is a phase offset in cycles, so a quarter cycle lands on the
	/// ramp quarter way up.
	#[test]
	fn the_pm_index_moves_the_carrier_phase() {
		use crate::generators::wavetable::Oscillator;

		let ramp: Vec<f32> = (0..1_024).map(|index| index as f32 / 1_023.0).collect();
		let mut pm = PmOsc::new(ramp, vec![1.0f32; 4]);
		let plain = Oscillator::<1>::play_at(&pm, 440.0, 0.0, [0.0])[0];

		pm.index = 0.25;

		let shifted = Oscillator::<1>::play_at(&pm, 440.0, 0.0, [0.0])[0];

		assert!(plain.abs() < 1e-3, "the carrier starts at zero: {plain}");
		assert!(
			(shifted - 0.25).abs() < 1e-2,
			"a quarter cycle offset should land on 0.25: {shifted}"
		);
	}

	/// AM at depth 1 with a full modulator doubles the carrier; at depth 0 it is
	/// the carrier itself.
	#[test]
	fn am_scales_the_carrier_with_the_modulator() {
		use crate::generators::wavetable::Oscillator;

		let mut am = AmOsc::new(vec![0.5f32; 4], vec![1.0f32; 4]);
		let plain = Oscillator::<1>::play_at(&am, 440.0, 0.0, [0.0])[0];

		am.depth = 1.0;

		let modulated = Oscillator::<1>::play_at(&am, 440.0, 0.0, [0.0])[0];

		assert!((plain - 0.5).abs() < 1e-3, "{plain}");
		assert!((modulated - 1.0).abs() < 1e-3, "{modulated}");
	}

	/// The decimator's kernel sums to one, so the fundamental passes unchanged.
	#[test]
	fn the_oversampler_passes_the_fundamental() {
		use crate::generators::wavetable::{BendWarp, Oscillator, SineWave};

		let mut osc = WarpOsc::new(SineWave, BendWarp::new());
		Oscillator::<1>::set_pitch(&mut osc, 440.0, 48_000);

		let value = Oscillator::<1>::play_at(&osc, 440.0, 0.0, [0.25])[0];

		assert!((value - 1.0).abs() < 2e-2, "{value}");
	}

	/// A hard warp still has to stay inside the table's range.
	#[test]
	fn a_hard_warp_stays_bounded() {
		use crate::generators::wavetable::{Oscillator, QuantizeWarp, SineWave};

		let mut warp = QuantizeWarp::new();
		warp.steps = 8.0;

		let mut osc = WarpOsc::new(SineWave, warp);
		Oscillator::<1>::set_pitch(&mut osc, 1_000.0, 48_000);

		for index in 0..512 {
			let time = index as f32 / 48_000.0;
			let value = Oscillator::<1>::play_at(&osc, 1_000.0, time, [0.0])[0];

			assert!(value.is_finite() && value.abs() <= 1.01, "{time}: {value}");
		}
	}

	/// The FM integral closes at the end of the cycle, so the phase does not
	/// jump when the modulator wraps.
	#[test]
	fn the_fm_integral_closes_at_the_end_of_the_cycle() {
		use crate::generators::wavetable::SineWave;

		let fm = FmOsc::new(SineWave, SineWave);

		assert!(
			fm.integral[FM_POINTS - 1].abs() < 1e-3,
			"{}",
			fm.integral[FM_POINTS - 1]
		);
	}

	/// At zero depth the carrier plays untouched.
	#[test]
	fn fm_at_zero_depth_is_the_carrier() {
		use crate::generators::wavetable::{Oscillator, SineWave, TableOsc};

		let fm = FmOsc::new(SineWave, SineWave);
		let carrier = TableOsc(SineWave);

		for index in 0..32 {
			let phase = [index as f32 / 32.0];

			assert_eq!(
				Oscillator::<1>::play_at(&fm, 440.0, 0.0, phase),
				Oscillator::<1>::play_at(&carrier, 440.0, 0.0, phase)
			);
		}
	}

	/// The phase moves by the depth times the integral of the modulator.
	#[test]
	fn fm_shifts_the_phase_by_the_integral() {
		use crate::generators::wavetable::{Oscillator, SineWave};

		let mut fm = FmOsc::new(SineWave, SineWave);
		fm.depth = 1.0;
		fm.ratio = 1.0;

		// The integral of a unit sine at a half cycle is 1/pi cycles, so the
		// carrier reads at 0.5 + 1/pi and a sine carrier gives sin(pi + 2).
		let value = Oscillator::<1>::play_at(&fm, 440.0, 0.0, [0.5])[0];
		let expected = (std::f32::consts::PI + 2.0).sin();

		assert!(
			(value - expected).abs() < 1e-3,
			"{value} against {expected}"
		);
	}

	/// FM is periodic with the note: a whole carrier cycle later is the same
	/// sample, because the integral returns to zero.
	#[test]
	fn fm_is_periodic_with_the_note() {
		use crate::generators::wavetable::{Oscillator, SineWave};

		let mut fm = FmOsc::new(SineWave, SineWave);
		fm.depth = 2.0;
		fm.ratio = 1.0;

		let start = Oscillator::<1>::play_at(&fm, 480.0, 0.0, [0.0])[0];
		let later = Oscillator::<1>::play_at(&fm, 480.0, 1.0 / 480.0, [0.0])[0];

		assert!((start - later).abs() < 1e-4, "{start} against {later}");
	}

	/// A modulator with its own controls has to reach the integral: the shape is
	/// checked every few samples and rebuilt when it moved.
	#[test]
	fn a_modulator_change_rebuilds_the_integral() {
		use crate::generators::wavetable::{Oscillator, SineWave};

		let sine: Vec<f32> = (0..1_024)
			.map(|index| (std::f32::consts::TAU * index as f32 / 1_024.0).sin())
			.collect();
		let mut fm = FmOsc::new(SineWave, sine);
		fm.depth = 1.0;
		fm.ratio = 1.0;

		let before = Oscillator::<1>::play_at(&fm, 440.0, 0.0, [0.5])[0];

		fm.modulator.0 = (0..1_024)
			.map(|index| -(std::f32::consts::TAU * index as f32 / 1_024.0).sin())
			.collect();

		for _ in 0..FM_RECHECK {
			Oscillator::<1>::sync(&mut fm);
		}

		let after = Oscillator::<1>::play_at(&fm, 440.0, 0.0, [0.5])[0];

		assert!(
			(after - before).abs() > 0.5,
			"{before} did not become {after}"
		);
	}

	/// A boxed oscillator forwards the driver calls to the oscillator inside:
	/// losing `set_pitch` would silently remove the band limiting.
	#[test]
	fn a_boxed_oscillator_forwards_the_driver_calls() {
		use std::sync::Arc;
		use std::sync::atomic::{AtomicUsize, Ordering};

		let pitches = Arc::new(AtomicUsize::new(0));
		let syncs = Arc::new(AtomicUsize::new(0));
		let mut boxed: Box<dyn Oscillator<1> + Send + Sync> = Box::new(Recorder {
			pitches: pitches.clone(),
			syncs: syncs.clone(),
		});

		Oscillator::<1>::set_pitch(&mut boxed, 440.0, 48_000);
		Oscillator::<1>::sync(&mut boxed);

		assert_eq!(pitches.load(Ordering::Relaxed), 1);
		assert_eq!(syncs.load(Ordering::Relaxed), 1);
		assert!(Oscillator::<1>::absolute_time(&boxed));
	}

	/// A boxed table reads exactly like the table it holds.
	#[test]
	fn a_boxed_table_reads_the_same() {
		use crate::generators::wavetable::TableOsc;

		let boxed: Box<dyn Oscillator<1> + Send + Sync> = Box::new(TableOsc(vec![1.0f32; 8]));

		assert_eq!(Oscillator::<1>::play_at(&boxed, 440.0, 0.0, [0.25]), [1.0]);
	}

	/// The warp moves where the carrier reads, without changing its shape.
	#[test]
	fn the_warp_moves_the_read_position() {
		use crate::generators::wavetable::{BendWarp, Oscillator};

		let ramp: Vec<f32> = (0..1_024).map(|index| index as f32 / 1_023.0).collect();
		let mut osc = WarpOsc::new(ramp, BendWarp::new());
		let plain = Oscillator::<1>::play_at(&osc, 440.0, 0.0, [0.5])[0];

		osc.warper.power = 2.0;

		let warped = Oscillator::<1>::play_at(&osc, 440.0, 0.0, [0.5])[0];

		assert!((plain - 0.5).abs() < 1e-2, "{plain}");
		assert!((warped - 0.25).abs() < 1e-2, "{warped}");
	}

	/// Ring modulation at full depth is the product of the two waves.
	#[test]
	fn ring_at_full_depth_multiplies_the_two_waves() {
		use crate::generators::wavetable::Oscillator;

		let mut rm = RmOsc::new(vec![0.5f32; 4], vec![0.5f32; 4]);
		let plain = Oscillator::<1>::play_at(&rm, 440.0, 0.0, [0.0])[0];

		rm.depth = 1.0;

		let modulated = Oscillator::<1>::play_at(&rm, 440.0, 0.0, [0.0])[0];

		assert!((plain - 0.5).abs() < 1e-3, "{plain}");
		assert!((modulated - 0.25).abs() < 1e-3, "{modulated}");
	}
}
