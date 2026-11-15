//! A wider that only adds a synthesized side.
//!
//! Widening usually means turning up the side signal, which does nothing for a
//! mono source: its side is silent. This effect synthesizes the side instead, by
//! taking the mid, detuning and dispersing it, and mixing the result into the
//! side only. The mid is never touched, so folding the output to mono gives back
//! the input exactly, whatever the controls say.
//!
//! Everything in the side path is zero latency: an allpass Hilbert shifter, an
//! allpass disperser and a biquad. There is no delay line, which is what makes
//! this different from a Haas widener and from the phase-vocoder Particle Wider.

use i_am_dsp_derive::Parameters;

use crate::{
	Effect, ProcessContext,
	effects::{disperser::Disperser, filter::Biquad, freq_shifter::IIRFreqShifter},
};

/// The most allpass sections the disperser may stack.
const MAX_DISPERSER_SECTIONS: usize = 16;

/// A zero-latency wider that synthesizes its side from the mid.
#[derive(Parameters)]
pub struct OuterWider {
	/// How much of the synthesized side is mixed in.
	#[range(min = 0.0, max = 1.0)]
	#[unit(percent)]
	#[default(0.3)]
	#[id(display = "Amount")]
	#[layout(group = "Width", order = 0, widget = knob, hero = "stereo")]
	pub amount: f32,

	/// The existing side gain; 1 is neutral.
	#[range(min = 0.0, max = 2.0)]
	#[default(1.0)]
	#[id(display = "Side")]
	#[layout(group = "Width", order = 1, widget = knob)]
	pub side: f32,

	/// How far the synthesized side is detuned from the mid.
	#[range(min = -20.0, max = 20.0)]
	#[unit(hertz)]
	#[default(1.5)]
	#[id(display = "Shift")]
	#[layout(group = "Width", order = 2, widget = knob)]
	pub shift: f32,

	/// How much the allpass chain scatters the phase of the side.
	#[range(min = 0.0, max = 1.0)]
	#[unit(percent)]
	#[default(0.4)]
	#[id(display = "Disperse")]
	#[layout(group = "Width", order = 3, widget = knob)]
	pub disperse: f32,

	/// The lower edge of the synthesized side, where the bass stays centred.
	#[range(min = 20.0, max = 2000.0)]
	#[unit(hertz)]
	#[logarithmic]
	#[default(150.0)]
	#[id(display = "Low Cut")]
	#[layout(group = "Width", order = 4, widget = knob)]
	pub low_cut: f32,

	/// Where the allpass chain scatters the phase most.
	#[range(min = 20.0, max = 20_000.0)]
	#[unit(hertz)]
	#[logarithmic]
	#[default(1_000.0)]
	#[id(display = "Cutoff")]
	#[layout(group = "Width", order = 5, widget = knob)]
	pub cutoff: f32,

	/// How wide a band around the cutoff the scattering covers.
	#[range(min = 20.0, max = 10_000.0)]
	#[unit(hertz)]
	#[logarithmic]
	#[default(1_600.0)]
	#[id(display = "Bandwidth")]
	#[layout(group = "Width", order = 6, widget = knob)]
	pub bandwidth: f32,

	#[skip]
	sample_rate: usize,
	#[skip]
	lows: Biquad<1>,
	#[skip]
	shifter: IIRFreqShifter<8, 1>,
	#[skip]
	disperser: Disperser<1>,
	#[skip]
	seen_low_cut: f32,
}

impl OuterWider {
	/// A wider at its default settings.
	pub fn new(sample_rate: usize) -> Self {
		let mut lows = Biquad::new(sample_rate);
		lows.set_to_highpass(150.0, 0.707);

		let mut disperser = Disperser::<1>::new(sample_rate);
		disperser.set_biquad_count(6);
		disperser.set_filter_parameters(1_000.0, 1_600.0);

		Self {
			amount: 0.3,
			side: 1.0,
			shift: 1.5,
			disperse: 0.4,
			low_cut: 150.0,
			cutoff: 1_000.0,
			bandwidth: 1_600.0,
			sample_rate,
			lows,
			shifter: IIRFreqShifter::new(sample_rate, 1.5),
			disperser,
			seen_low_cut: 150.0,
		}
	}

	/// The allpass sections a disperse amount asks for.
	fn sections(disperse: f32) -> usize {
		(disperse.clamp(0.0, 1.0) * MAX_DISPERSER_SECTIONS as f32).round() as usize
	}

	/// Pushes the controls into the stage that owns each of them.
	///
	/// The shifter and the disperser notice a moved input themselves when they
	/// run, and both setters skip a rebuild that would change nothing, so this
	/// only has to handle the low cut: the biquad has no such check of its own.
	fn sync(&mut self) {
		if self.low_cut != self.seen_low_cut {
			self.lows.set_to_highpass(self.low_cut, 0.707);
			self.seen_low_cut = self.low_cut;
		}
	}
}

impl Effect<2> for OuterWider {
	fn delay(&self) -> usize {
		// Every stage of the side path is a filter: no sample is held back.
		0
	}

	fn process(
		&mut self,
		samples: &mut [f32; 2],
		_other: &[&[f32; 2]],
		context: &mut Box<dyn ProcessContext>,
	) {
		self.sync();

		// Both of these settle against what they were last built from, so the
		// call costs a comparison unless the control moved.
		self.shifter.set_shift_freq(self.shift);
		self.disperser
			.set_biquad_count(Self::sections(self.disperse));
		self.disperser
			.set_filter_parameters(self.cutoff, self.bandwidth);

		let mid = (samples[0] + samples[1]) * 0.5;
		let side = (samples[0] - samples[1]) * 0.5;

		// The synthesized side is a mono signal on purpose: it has to arrive in
		// both channels with the same magnitude and opposite sign, or folding to
		// mono would not cancel it.
		let mut synthesized = [self.lows.process_sample([mid])[0]];
		self.shifter.process(&mut synthesized, &[], context);
		self.disperser.process(&mut synthesized, &[], context);

		let out = side * self.side + synthesized[0] * self.amount;
		samples[0] = mid + out;
		samples[1] = mid - out;
	}

	fn clear_tail(&mut self) {
		self.lows.set_to_default_state();
		self.shifter = IIRFreqShifter::new(self.sample_rate, self.shift);
		self.disperser = Disperser::<1>::new(self.sample_rate);
		self.disperser
			.set_biquad_count(Self::sections(self.disperse));
		self.disperser
			.set_filter_parameters(self.cutoff, self.bandwidth);
		self.seen_low_cut = self.low_cut;
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// Renders a stereo pair through the wider and returns every output frame.
	fn render(
		wider: &mut OuterWider,
		input: impl Fn(usize) -> [f32; 2],
		frames: usize,
	) -> Vec<[f32; 2]> {
		let mut context: Box<dyn ProcessContext> = Box::new(());
		let mut output = Vec::with_capacity(frames);

		for index in 0..frames {
			let mut frame = input(index);
			wider.process(&mut frame, &[], &mut context);
			output.push(frame);
		}

		output
	}

	/// A sine of the given frequency and amplitude in both channels.
	fn mono_sine(frequency: f32, amplitude: f32) -> impl Fn(usize) -> [f32; 2] {
		move |index| {
			let phase = std::f32::consts::TAU * frequency * index as f32 / 48_000.0;
			let sample = phase.sin() * amplitude;
			[sample, sample]
		}
	}

	/// The side path is filters only: nothing is held back a sample.
	#[test]
	fn the_wider_has_no_latency() {
		assert_eq!(OuterWider::new(48_000).delay(), 0);
	}

	/// The mid is never touched, so folding to mono gives the input back at any
	/// setting: this is the invariant the whole effect exists to keep.
	#[test]
	fn folding_to_mono_gives_the_input_back() {
		let mut wider = OuterWider::new(48_000);
		wider.amount = 1.0;
		wider.side = 2.0;
		wider.shift = 10.0;
		wider.disperse = 1.0;
		wider.low_cut = 60.0;
		wider.cutoff = 400.0;
		wider.bandwidth = 300.0;

		let input = |index: usize| {
			let phase = std::f32::consts::TAU * 440.0 * index as f32 / 48_000.0;
			[phase.sin() * 0.5, (phase * 1.3).cos() * 0.4]
		};
		let output = render(&mut wider, input, 4_096);

		for (index, wet) in output.iter().enumerate().take(4_096) {
			let dry = input(index);

			assert!(
				((wet[0] + wet[1]) - (dry[0] + dry[1])).abs() < 1e-5,
				"frame {index} folded to {} instead of {}",
				wet[0] + wet[1],
				dry[0] + dry[1]
			);
		}
	}

	/// A source with no side at all has to come out with one.
	#[test]
	fn a_mono_source_comes_out_stereo() {
		let mut wider = OuterWider::new(48_000);
		wider.amount = 0.5;
		wider.shift = 3.0;
		wider.disperse = 0.6;

		let output = render(&mut wider, mono_sine(1_000.0, 0.5), 8_192);
		let settled = &output[4_096..];
		let side = settled
			.iter()
			.map(|frame| ((frame[0] - frame[1]) * 0.5).powi(2))
			.sum::<f32>()
			.sqrt();

		assert!(side > 0.01, "the synthesized side is only {side}");
	}

	/// The low cut is what keeps the bass out of the synthesized side.
	#[test]
	fn the_low_cut_keeps_the_bass_centred() {
		let mut wider = OuterWider::new(48_000);
		wider.amount = 1.0;
		wider.low_cut = 400.0;

		let output = render(&mut wider, mono_sine(60.0, 0.5), 8_192);
		let settled = &output[4_096..];
		let mid = settled
			.iter()
			.map(|frame| ((frame[0] + frame[1]) * 0.5).powi(2))
			.sum::<f32>()
			.sqrt();
		let side = settled
			.iter()
			.map(|frame| ((frame[0] - frame[1]) * 0.5).powi(2))
			.sum::<f32>()
			.sqrt();

		assert!(side < mid * 0.05, "side {side} against mid {mid}");
	}

	/// With nothing asked of it the effect returns the input. The mid/side
	/// round trip is float arithmetic, so this is a pass through to rounding,
	/// not bit for bit.
	#[test]
	fn a_neutral_setting_passes_the_signal_through() {
		let mut wider = OuterWider::new(48_000);
		wider.amount = 0.0;
		wider.side = 1.0;

		let input = |index: usize| {
			let phase = std::f32::consts::TAU * 220.0 * index as f32 / 48_000.0;
			[phase.sin() * 0.3, (phase * 0.7).sin() * 0.2]
		};
		let output = render(&mut wider, input, 1_024);

		for (index, wet) in output.iter().enumerate().take(1_024) {
			let dry = input(index);

			assert!(
				(wet[0] - dry[0]).abs() < 1e-6 && (wet[1] - dry[1]).abs() < 1e-6,
				"frame {index}: {wet:?} against {dry:?}"
			);
		}
	}
}
