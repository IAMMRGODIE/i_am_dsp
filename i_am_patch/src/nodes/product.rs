//! One product module, wrapped as a node.

use super::*;

/// A rack module, driven by its own product schema.
pub(super) struct ProductNode {
	effect: Box<dyn Effect<2>>,
	/// Parameters that move across the current block: name, start, end.
	ramps: Vec<(&'static str, f32, f32)>,
	/// The value type each parameter declares.
	kinds: BTreeMap<String, ParamKind>,
	/// The worker a scripted product runs, so the page can edit its source.
	script: Option<std::sync::Arc<dyn i_am_effects::ScriptSource>>,
}

impl ProductNode {
	pub(super) fn new(effect: Box<dyn Effect<2>>) -> Self {
		let kinds = parameter_kinds(&effect.get_parameters());

		Self {
			effect,
			ramps: Vec::with_capacity(8),
			kinds,
			script: None,
		}
	}

	/// A product whose page edits the script it runs.
	pub(super) fn with_script(
		effect: Box<dyn Effect<2>>,
		script: std::sync::Arc<dyn i_am_effects::ScriptSource>,
	) -> Self {
		let mut node = Self::new(effect);
		node.script = Some(script);
		node
	}
}

impl BlockNode for ProductNode {
	fn script_source(&self) -> Option<std::sync::Arc<dyn i_am_effects::ScriptSource>> {
		self.script.clone()
	}

	fn clear_tail(&mut self) {
		self.effect.clear_tail();
	}

	fn set_param(&mut self, name: &str, value: f32) {
		let kind = self.kinds.get(name).copied().unwrap_or(ParamKind::Float);
		self.effect.set_parameter(name, set_value(kind, value));
	}

	fn set_param_block(&mut self, name: &'static str, start: f32, end: f32, _frames: usize) {
		// The buffer is allocated once, so a modulated block does not allocate.
		self.ramps.push((name, start, end));
	}

	fn process(&mut self, context: &mut BlockContext, inputs: &[Buffer], outputs: &mut [Buffer]) {
		let input = audio(inputs, 0);
		let sidechain = inputs.get(1).and_then(Buffer::audio).copied();
		let Self {
			effect,
			ramps,
			kinds,
			..
		} = self;

		if let Some(Buffer::Audio(out)) = outputs.first_mut() {
			out.clear();
			out.frames = context.frames;
			let frames = context.frames.max(1) as f32;

			for frame in 0..context.frames {
				let unit = frame as f32 / frames;

				for &(name, start, end) in ramps.iter() {
					let kind = kinds.get(name).copied().unwrap_or(ParamKind::Float);
					let value = match kind {
						ParamKind::Float => start + (end - start) * unit,
						_ => end,
					};

					effect.set_parameter(name, set_value(kind, value));
				}

				let mut samples = [input.channels[0][frame], input.channels[1][frame]];
				// The product's chooser counts the main input as port 0, so the
				// sidechain frame sits at index 1; index 0 mirrors it.
				let side =
					sidechain.map(|block| [block.channels[0][frame], block.channels[1][frame]]);
				let other = side.map(|frame| [frame, frame]);
				let other: &[&[f32; 2]] = match &other {
					Some(frames) => &[&frames[0], &frames[1]],
					None => &[],
				};

				context.context.set_sample_offset(frame);
				effect.process(&mut samples, other, &mut context.context);
				out.channels[0][frame] = samples[0];
				out.channels[1][frame] = samples[1];
			}

			ramps.clear();
		}
	}
}
