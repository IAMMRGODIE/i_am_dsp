use super::*;

/// The product.
pub struct ConvolverWeb {
	processor: EffectProcessor<ConvolverEffect>,
	bypass: Arc<AtomicBool>,
	scope: Arc<Scope>,
	ir: Arc<IrSlot>,
	clear: Arc<AtomicBool>,
}

impl Default for ConvolverWeb {
	fn default() -> Self {
		let ir = Arc::new(IrSlot::new());
		let build_slot = ir.clone();

		Self {
			processor: EffectProcessor::with_build(move |rate| {
				build_slot.set_host_rate(rate);
				ConvolverEffect::new(rate, build_slot.clone())
			}),
			bypass: Arc::new(AtomicBool::new(false)),
			scope: Arc::new(Scope::new()),
			ir,
			clear: Arc::new(AtomicBool::new(false)),
		}
	}
}

impl ConvolverWeb {
	/// The faceplate size this product declares to a host.
	pub const fn size() -> (f32, f32) {
		SIZE
	}

	/// The rate the effect inside is currently built for.
	pub fn sample_rate(&self) -> usize {
		self.processor.sample_rate()
	}

	/// The shared IR state, for a test.
	pub fn ir(&self) -> &Arc<IrSlot> {
		&self.ir
	}
}

impl i_am_dsp::prelude::Parameters for ConvolverWeb {
	fn get_parameters(&self) -> Vec<Parameter> {
		self.processor.parameters()
	}

	fn set_parameter(&mut self, identifier: &str, value: SetValue) -> bool {
		self.processor.set_parameter(identifier, value)
	}
}

impl i_am_dsp::Processor for ConvolverWeb {
	type Message = EffectMessage;

	fn delay(&self) -> usize {
		i_am_dsp::Processor::delay(&self.processor)
	}

	fn process(
		&mut self,
		samples: &mut [f32; 2],
		other: &[[f32; 2]],
		process_context: &mut Box<dyn ProcessContext>,
	) {
		if self.clear.swap(false, Ordering::Relaxed) {
			self.processor.clear_tail();
		}

		for event in process_context.events() {
			match event {
				i_am_dsp::NoteEvent::NoteOn { note, .. } => self.scope.set_note(*note as i32),
				i_am_dsp::NoteEvent::NoteOff { note, .. }
				| i_am_dsp::NoteEvent::Stop { note, .. } => {
					if self.scope.note() == *note as i32 {
						self.scope.set_note(-1);
					}
				}
				i_am_dsp::NoteEvent::ImmediateStop => self.scope.set_note(-1),
				_ => {}
			}
		}

		let input = *samples;
		let input_mono = (input[0] + input[1]) * 0.5;
		let sidechain = other
			.get(1)
			.map(|frame| (frame[0] + frame[1]) * 0.5)
			.unwrap_or(input_mono);

		if self.bypass.load(Ordering::Relaxed) {
			self.scope.push_stereo(input, input, sidechain);
			return;
		}

		i_am_dsp::Processor::process(&mut self.processor, samples, other, process_context);

		let output = *samples;
		self.scope.push_stereo(input, output, sidechain);
	}

	fn on_message(&self, message: Self::Message) {
		i_am_dsp::Processor::on_message(&self.processor, message);
	}
}

impl Plugin for ConvolverWeb {
	const DESCRIPTOR: Descriptor =
		Descriptor::new("iamdsp.effects.convolver.web", "I Am Convolver")
			.with_vendor("iamplugins")
			.with_version(env!("CARGO_PKG_VERSION"));

	const HAS_GUI: bool = true;

	fn new() -> Self {
		Self::default()
	}

	fn window_options() -> WindowOptions {
		i_am_web_gui::window_options((SIZE.0, SIZE.1), "I Am Convolver")
	}

	fn param_map(&self) -> ParamMap {
		self.processor.param_map()
	}

	fn sync_params(&mut self) {
		self.processor.sync_params();
	}

	fn on_activate(&mut self, sample_rate: usize) {
		self.processor.activate(sample_rate);
	}

	fn editor(&self) -> Option<Box<dyn Editor<Self>>> {
		let face = i_am_web_gui::Faceplate::new("I Am Convolver", ACCENT, SIZE).with_clear();
		let route_slot = self.ir.clone();
		let clear_flag = self.clear.clone();
		let command_slot = self.ir.clone();

		i_am_web_gui::editor_with(
			self,
			face,
			self.bypass.clone(),
			self.scope.clone(),
			self.processor.sample_rate(),
			move |request: &ProtocolRequest<'_>| {
				if request.method == "POST" && request.path == "/clear" {
					clear_flag.store(true, Ordering::Relaxed);
					return Some(ProtocolResponse::json("{\"ok\":true}"));
				}

				ir_route(request, &route_slot)
			},
			move |command: &str| {
				if command != "__load_ir" {
					return false;
				}

				pick_ir(command_slot.clone());
				true
			},
		)
	}
}
