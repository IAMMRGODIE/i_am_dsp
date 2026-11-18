//! Wave table and Oscillator implementations.

use crate::prelude::{Parameter, Parameters};

#[cfg(feature = "sin_table")]
use std::f32::consts::PI;

#[cfg(feature = "sin_table")]
lazy_static::lazy_static! {
	static ref SINE_TABLE: Vec<f32> = (0..256).map(|i| (i as f32 / 256.0 * 2.0 * PI).sin()).collect();
}

/// Wraps a phase into `0..1`.
///
/// `f32::rem_euclid` goes through `fmod`, a library call; an oscillator does this
/// once or twice per sample per voice, so a floor and a subtract is worth it.
pub(crate) fn wrap01(value: f32) -> f32 {
	value - value.floor()
}

/// A trait for wave tables that can be used by oscillators or be used as LFO.
///
/// Note that `Vec<f32>` and `RingBuffer<f32>` are also implemented for this trait.
pub trait WaveTable: Parameters + Send {
	/// Returns the sample value at time t.
	///
	/// t in the range [0, 1]
	fn sample(&self, t: f32, channel: usize) -> f32;

	/// Whether both channels read the same sample.
	///
	/// True by default: no table in this crate uses the channel, so a voice can
	/// read it once for a copy instead of once per channel. A table that really
	/// differs per channel has to say so, or one side would be wrong.
	fn is_mono(&self) -> bool {
		true
	}

	/// Lets a table that renders off the audio thread pick up new pages.
	///
	/// The default does nothing: a table built from its own parameters is
	/// already current. The drawn spectrum renders on a worker, so it has to be
	/// told, and every adapter between it and the voice has to pass it on.
	fn sync(&mut self) {}
}

/// An oscillator that can be used to generate audio signals.
///
/// For midi support, use [`crate::prelude::Adsr`]
pub trait Oscillator<const CHANNELS: usize>: Parameters {
	/// Plays the oscillator at the given frequency and time, and returns the output for each channel.
	///
	/// frequency in Hz
	/// time in milliseconds
	/// phase in [0.0, 1.0]
	fn play_at(&self, frequency: f32, time: f32, phase: [f32; CHANNELS]) -> [f32; CHANNELS];

	/// Tells the oscillator the pitch of the note it is about to play, and the sample rate.
	///
	/// A driver that knows both calls this before playing a note, so that an oscillator built from
	/// a fixed set of partials can drop the ones that would alias. The default implementation
	/// ignores it, which leaves the oscillator unbandlimited.
	fn set_pitch(&mut self, frequency: f32, sample_rate: usize) {
		let _ = (frequency, sample_rate);
	}

	/// Whether `play_at` reads absolute time instead of a wrapped phase.
	///
	/// A table oscillator only needs the fractional part, so the driver can wrap
	/// the phase every sample and keep f32 resolution on a long note; a sampler
	/// reads its playback position from the time, so it must stay absolute.
	fn absolute_time(&self) -> bool {
		false
	}

	/// Called once per sample by a driver, before any [`play_at`](Self::play_at).
	///
	/// An oscillator that derives something from its parameters (an additive
	/// oscillator's partial table, say) can notice a control change here; the
	/// default does nothing.
	fn sync(&mut self) {}

	#[cfg(feature = "real_time_demo")]
	/// Draws the wave table for the oscillator in the demo UI.
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String);
}

/// A wrapper for a wavetable to use it as an oscillator with default demo UI.
pub struct TableOsc<T: WaveTable>(pub T);

impl<T: WaveTable> Parameters for TableOsc<T> {
	fn get_parameters(&self) -> Vec<Parameter> {
		self.0.get_parameters()
	}

	fn set_parameter(&mut self, identifier: &str, value: crate::prelude::SetValue) -> bool {
		self.0.set_parameter(identifier, value)
	}
}

impl<T: WaveTable, const CHANNELS: usize> Oscillator<CHANNELS> for TableOsc<T> {
	#[cfg(feature = "real_time_demo")]
	fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
		use crate::tools::ui_tools::draw_wavetable;

		let width = ui.available_width();
		egui::Resize::default()
			.resizable([false, true])
			// .auto_sized()
			.min_width(width)
			.max_width(width)
			.id_salt(format!("{id_prefix}_wavtable"))
			.show(ui, |ui| {
				draw_wavetable(ui, |t| self.0.sample(t, 0));
			});
	}

	fn play_at(&self, frequency: f32, time: f32, phase: [f32; CHANNELS]) -> [f32; CHANNELS] {
		let mut output = [0.0; CHANNELS];

		// A mono table read at one phase is the same on both sides: one read
		// then feeds them all.
		if CHANNELS > 0 && phase.iter().all(|value| value.to_bits() == phase[0].to_bits()) && self.0.is_mono() {
			let value = self.0.sample(wrap01(time * frequency + phase[0]), 0);
			output = [value; CHANNELS];

			return output;
		}

		for i in 0..CHANNELS {
			let t = wrap01(time * frequency + phase[i]);
			output[i] = self.0.sample(t, i);
		}

		output
	}

	fn sync(&mut self) {
		self.0.sync();
	}
}

/// A wrapper for an ossillator to use it as a wavetable.
pub struct OscTable<const CHANNELS: usize, O: Oscillator<CHANNELS>>(pub O);

impl<const CHANNELS: usize, O: Oscillator<CHANNELS>> Parameters for OscTable<CHANNELS, O> {
	fn get_parameters(&self) -> Vec<Parameter> {
		self.0.get_parameters()
	}

	fn set_parameter(&mut self, identifier: &str, value: crate::prelude::SetValue) -> bool {
		self.0.set_parameter(identifier, value)
	}
}

impl<const CHANNELS: usize, O: Oscillator<CHANNELS> + Send> WaveTable for OscTable<CHANNELS, O> {
	fn sync(&mut self) {
		self.0.sync();
	}

	fn sample(&self, t: f32, _: usize) -> f32 {
		if CHANNELS == 0 {
			return 0.0;
		}

		self.0.play_at(1.0, t, [0.0; CHANNELS]).iter().sum::<f32>() / CHANNELS as f32
	}
}

#[cfg(test)]
mod adapter_tests {
	use super::*;
	use std::sync::Arc;
	use std::sync::atomic::{AtomicUsize, Ordering};

	struct Counter(Arc<AtomicUsize>);

	impl Parameters for Counter {
		fn get_parameters(&self) -> Vec<Parameter> {
			Vec::new()
		}

		fn set_parameter(&mut self, _: &str, _: crate::prelude::SetValue) -> bool {
			false
		}
	}

	impl WaveTable for Counter {
		fn sample(&self, _: f32, _: usize) -> f32 {
			0.0
		}

		fn sync(&mut self) {
			self.0.fetch_add(1, Ordering::Relaxed);
		}
	}

	impl<const CHANNELS: usize> Oscillator<CHANNELS> for Counter {
		fn play_at(&self, _: f32, _: f32, _: [f32; CHANNELS]) -> [f32; CHANNELS] {
			[0.0; CHANNELS]
		}

		fn sync(&mut self) {
			self.0.fetch_add(1, Ordering::Relaxed);
		}
	}

	/// A table under an oscillator, and an oscillator under a table: the drawn
	/// spectrum sits at the bottom of one of these and only notices new pages
	/// through its sync, so neither adapter may swallow it.
	#[test]
	fn both_adapters_pass_the_sync_on() {
		let table = Arc::new(AtomicUsize::new(0));
		let mut oscillator: TableOsc<Counter> = TableOsc(Counter(table.clone()));

		Oscillator::<2>::sync(&mut oscillator);
		assert_eq!(table.load(Ordering::Relaxed), 1);

		let wrapped = Arc::new(AtomicUsize::new(0));
		let mut table = OscTable::<2, _>(Counter(wrapped.clone()));

		table.sync();
		assert_eq!(wrapped.load(Ordering::Relaxed), 1);
	}
}

mod frames;
mod oscillators;
mod smoothers;
mod tables;
mod warp;

pub use frames::*;
pub use oscillators::*;
pub use smoothers::*;
pub use tables::*;
pub use warp::*;
