//! The envelope node.

use super::*;
use super::table::write_param;

/// The envelope that turns an oscillator and the block's notes into audio.
pub(super) struct AdsrNode {
	rate: usize,
	pub(super) inner: Option<Adsr<Box<dyn Oscillator<2> + Send + Sync>, EqualTemperament, 2>>,
}

impl AdsrNode {
	pub(super) fn new(rate: usize) -> Self {
		Self { rate, inner: None }
	}
}

impl BlockNode for AdsrNode {
	fn install(&mut self, oscillator: Box<dyn Oscillator<2> + Send + Sync>) -> bool {
		self.inner = Some(Adsr::new(oscillator, EqualTemperament, self.rate));

		true
	}

	fn set_param(&mut self, name: &str, value: f32) {
		if let Some(inner) = &mut self.inner {
			// The Voice page mixes floats, an integer (unisons) and a bool
			// (phase per channel); a float-only write silently dropped two of them.
			write_param(inner, name, value);
		}
	}

	/// The note the next one should glide from.
	///
	/// A voice group is rebuilt for every note, so this envelope has never played
	/// anything and knows no pitch to leave behind: without this the glide control would
	/// do nothing at all in the patcher.
	fn glide_from(&mut self, channel: u8, note: usize) {
		if let Some(inner) = &mut self.inner {
			i_am_dsp::Generator::glide_from(inner, channel, note);
		}
	}

	fn process(&mut self, context: &mut BlockContext, _: &[Buffer], outputs: &mut [Buffer]) {
		let Some(inner) = &mut self.inner else {
			if let Some(Buffer::Audio(out)) = outputs.first_mut() {
				out.clear();
				out.frames = context.frames;
			}

			return;
		};

		if let Some(Buffer::Audio(out)) = outputs.first_mut() {
			out.clear();
			out.frames = context.frames;

			for frame in 0..context.frames {
				context.context.set_sample_offset(frame);
				let samples = inner.generate(&mut context.context);

				out.channels[0][frame] = samples[0];
				out.channels[1][frame] = samples[1];
			}
		}
	}
}
