/// Declares a plug-in with a web faceplate and nothing else.
///
/// The DSP, the parameter metadata and this one line are the whole product. The
/// window is resizable; the page reflows rather than scaling a picture of itself.
///
/// A parameter named by `mode = "id"` is drawn as a switch in the header instead of
/// a control in the body, which is how an effect with several algorithms changes
/// one.
///
/// An instrument is declared with the ports and tags it has rather than the effect
/// defaults: `ports = &[]` says there is no audio to process, `midi = &[...]` says
/// where the notes come in, and `tags = &[...]` puts it in the host's instrument
/// menu. The generator itself is played through
/// [`GeneratorEffect`](i_am_dsp::GeneratorEffect).
///
/// A product with more controls than one face shows comfortably can hand its
/// sections to tabs with `pages = &[...]`, and `hidden = &[...]` keeps a
/// parameter off the face while the host still sees it.
///
/// ```ignore
/// web_plugin!(
///     FilterWeb,
///     Multimode,
///     |rate| Multimode::new(rate, 1_000.0, 0.707),
///     "iamdsp.effects.filter.web",
///     "I Am Filter",
///     (520.0, 330.0),
///     "#4c8dff",
///     mode = "mode"
/// );
/// ```
#[macro_export]
macro_rules! web_plugin {
	($name:ident, $effect:ty, $build:expr, $id:literal, $title:literal, $size:expr, $accent:literal $(, mode = $mode:literal)? $(, visible = $visible:expr)? $(, header = $header:expr)? $(, switches = $switches:expr)? $(, pages = $pages:expr)? $(, hidden = $hidden:expr)? $(, ports = $ports:expr)? $(, midi = $midi:expr)? $(, tags = $tags:expr)? $(, telemetry = $telemetry:expr)? $(, clear = $clear:literal)?) => {
		#[doc = concat!("The ", $title, " plug-in, with a web faceplate.")]
		pub struct $name {
			processor: i_am_plugin::processor::EffectProcessor<$effect>,
			bypass: std::sync::Arc<std::sync::atomic::AtomicBool>,
			scope: std::sync::Arc<$crate::Scope>,
			clear: std::sync::Arc<std::sync::atomic::AtomicBool>,
		}

		impl Default for $name {
			fn default() -> Self {
				Self {
					processor: i_am_plugin::processor::EffectProcessor::new($build),
					bypass: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
					scope: std::sync::Arc::new($crate::Scope::new()),
					clear: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
				}
			}
		}

		impl $name {
			/// The faceplate size this product declares to a host.
			pub const fn size() -> (f32, f32) {
				$size
			}

			/// The rate the effect inside is currently built for.
			pub fn sample_rate(&self) -> usize {
				self.processor.sample_rate()
			}

			/// Whether the editor's bypass switch is on.
			pub fn is_bypassed(&self) -> bool {
				self.bypass.load(std::sync::atomic::Ordering::Relaxed)
			}

			/// Turn the bypass on or off from code, for a test or a host that
			/// grows its own bypass later.
			pub fn set_bypassed(&self, bypassed: bool) {
				self.bypass
					.store(bypassed, std::sync::atomic::Ordering::Relaxed);
			}
		}

		impl i_am_dsp::prelude::Parameters for $name {
			fn get_parameters(&self) -> Vec<i_am_dsp::prelude::Parameter> {
				self.processor.parameters()
			}

			fn set_parameter(
				&mut self,
				identifier: &str,
				value: i_am_dsp::prelude::SetValue,
			) -> bool {
				self.processor.set_parameter(identifier, value)
			}
		}

		impl i_am_dsp::Processor for $name {
			type Message = i_am_plugin::processor::EffectMessage;

			fn delay(&self) -> usize {
				i_am_dsp::Processor::delay(&self.processor)
			}

			fn process(
				&mut self,
				samples: &mut [f32; 2],
				other: &[[f32; 2]],
				process_context: &mut Box<dyn i_am_dsp::ProcessContext>,
			) {
				// The scope keeps the last 4096 frames at the host's rate: about
				// 85 ms at 48 kHz, long enough to read a waveform and full-rate
				// enough for the page to run an FFT over it.
				for event in process_context.events() {
					match event {
						i_am_dsp::NoteEvent::NoteOn { note, .. } => {
							self.scope.set_note(*note as i32)
						}
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

				if self.clear.swap(false, std::sync::atomic::Ordering::Relaxed) {
					self.processor.clear_tail();
				}

				let input = *samples;
				let input_mono = (input[0] + input[1]) * 0.5;
				// A compressor with a sidechain port reads other[1]; without one the
				// input stands in for it, which is also what the DSP does.
				let sidechain = other
					.get(1)
					.map(|frame| (frame[0] + frame[1]) * 0.5)
					.unwrap_or(input_mono);

				// Bypassed is the input, untouched: an effect has nothing to add.
				if self.bypass.load(std::sync::atomic::Ordering::Relaxed) {
					self.scope.push_stereo(input, input, sidechain);
					return;
				}

				i_am_dsp::Processor::process(&mut self.processor, samples, other, process_context);

				$(
					($telemetry)(self.processor.effect(), &self.scope);
				)?

				let output = *samples;
				self.scope.push_stereo(input, output, sidechain);
			}

			fn on_message(&self, message: Self::Message) {
				i_am_dsp::Processor::on_message(&self.processor, message);
			}
		}

		impl i_am_plugin::Plugin for $name {
			const DESCRIPTOR: i_am_plugin::Descriptor = i_am_plugin::Descriptor::new($id, $title)
				.with_vendor("iamplugins")
				.with_version(env!("CARGO_PKG_VERSION"))
				$(.with_tags($tags))?;

			const HAS_GUI: bool = true;

			$(const INPUT_PORTS: &'static [i_am_plugin::AudioPort] = $ports;)?
			$(const INPUT_MIDI_PORTS: &'static [i_am_plugin::MidiPort] = $midi;)?

			fn new() -> Self {
				Self::default()
			}

			fn window_options() -> i_am_plugin::WindowOptions {
				$crate::window_options($size, $title)
			}

			fn param_map(&self) -> i_am_dsp::prelude::ParamMap {
				self.processor.param_map()
			}

			fn sync_params(&mut self) {
				self.processor.sync_params();
			}

			fn on_activate(&mut self, sample_rate: usize) {
				self.processor.activate(sample_rate);
			}

			fn editor(&self) -> Option<Box<dyn i_am_plugin::Editor<Self>>> {
				let face = $crate::Faceplate::new($title, $accent, $size) $(.with_mode($mode))? $(.with_visible($visible))? $(.with_header($header))? $(.with_switches($switches))? $(.with_pages($pages))? $(.with_hidden($hidden))?;
				$(let face = if $clear { face.with_clear() } else { face };)?

				let clear = self.clear.clone();

				$crate::editor_with(
					self,
					face,
					self.bypass.clone(),
					self.scope.clone(),
					self.processor.sample_rate(),
					move |request: &$crate::ProtocolRequest<'_>| {
						if request.method == "POST" && request.path == "/clear" {
							clear.store(true, std::sync::atomic::Ordering::Relaxed);
							return Some($crate::ProtocolResponse::json("{\"ok\":true}"));
						}

						None
					},
					|_: &str| false,
				)
			}
		}
	};
}
