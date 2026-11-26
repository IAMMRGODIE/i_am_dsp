//! The synth node, over any of the products' generators.

use super::*;

/// A synth product: its notes come from the block's events, its audio out the
/// node's first port.
pub(super) struct SynthNode<P: Default + Plugin> {
	inner: P,
	/// The value type each parameter declares.
	kinds: BTreeMap<String, ParamKind>,
	/// Whether the map moved since the last block.
	dirty: bool,
	/// The script the product runs, when it is a scripted one.
	script: Option<std::sync::Arc<dyn i_am_effects::ScriptSource>>,
}

impl<P: Default + Plugin> SynthNode<P> {
	pub(super) fn new(rate: usize) -> Self {
		Self::with_plugin(P::default(), rate)
	}

	/// A synth built with state of its own, like the table a Modal node plays.
	pub(super) fn with_plugin(mut inner: P, rate: usize) -> Self {
		inner.on_activate(rate);
		let kinds = parameter_kinds(&inner.get_parameters());

		Self {
			inner,
			kinds,
			dirty: false,
			script: None,
		}
	}

	/// A synth whose page edits the script it runs.
	pub(super) fn with_script(
		mut inner: P,
		script: std::sync::Arc<dyn i_am_effects::ScriptSource>,
		rate: usize,
	) -> Self {
		inner.on_activate(rate);
		let kinds = parameter_kinds(&inner.get_parameters());

		Self {
			inner,
			kinds,
			dirty: false,
			script: Some(script),
		}
	}
}

impl<P: Default + Plugin> BlockNode for SynthNode<P> {
	fn script_source(&self) -> Option<std::sync::Arc<dyn i_am_effects::ScriptSource>> {
		self.script.clone()
	}

	fn glide_from(&mut self, channel: u8, note: usize) {
		self.inner.glide_from(channel, note);
	}

	fn set_param(&mut self, name: &str, value: f32) {
		// A synth keeps its parameters in a shared map and reads them back on the
		// next block: write the map, then push it into the effect. The map refuses
		// a value of the wrong type, so the parameter's declared type decides it.
		let kind = self.kinds.get(name).copied().unwrap_or(ParamKind::Float);
		let map = self.inner.param_map();
		map.set(name, set_value(kind, value), Ordering::Relaxed);
		self.dirty = true;
	}

	fn process(&mut self, context: &mut BlockContext, _: &[Buffer], outputs: &mut [Buffer]) {
		// Every parameter was written into the map before this block; one sync
		// carries them all into the DSP.
		if self.dirty {
			self.inner.sync_params();
			self.dirty = false;
		}

		for event in &context.events {
			self.inner
				.on_message(<P::Message as Message>::from_note_event(event.clone()));
		}

		if let Some(Buffer::Audio(out)) = outputs.first_mut() {
			out.clear();
			out.frames = context.frames;

			for frame in 0..context.frames {
				context.context.set_sample_offset(frame);
				let mut samples = [0.0f32; 2];
				self.inner.process(&mut samples, &[], &mut context.context);
				out.channels[0][frame] = samples[0];
				out.channels[1][frame] = samples[1];
			}
		}
	}
}
