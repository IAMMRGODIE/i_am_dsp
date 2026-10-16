//! events

use crate::shared::*;
use crate::plugin::{Plugin};
use crate::main_thread::{PluginMain};
use crate::audio::{AudioProcessor};

pub(crate) fn note_event_to_clap<'a>(to_send: &mut OutputEvents<'a>, offset: usize, note_event: i_am_dsp::NoteEvent) -> Result<(), PluginError> {
	match note_event {
		i_am_dsp::NoteEvent::NoteOn { channel, note, velocity } => {
			to_send.try_push(NoteOnEvent::new(
				offset as u32,
				Pckn::new(Match::All, channel as u16, note as u16, Match::All),
				velocity as f64,
			))?;
		},
		i_am_dsp::NoteEvent::ImmediateStop => {
			to_send.try_push(NoteChokeEvent::new(
				offset as u32,
				Pckn::new(Match::All, Match::All, Match::All, Match::All),
			))?;
		}
		i_am_dsp::NoteEvent::MidiCC { .. } => {},
		i_am_dsp::NoteEvent::NoteOff { channel, note, velocity } => {
			to_send.try_push(NoteOffEvent::new(
				offset as u32,
				Pckn::new(Match::All, channel as u16, note as u16, Match::All),
				velocity as f64,
			))?;
		},
		i_am_dsp::NoteEvent::Stop { channel, note } => {
			to_send.try_push(NoteOffEvent::new(
				offset as u32,
				Pckn::new(Match::All, channel as u16, note as u16, Match::All),
				0.0,
			))?;
		},
		i_am_dsp::NoteEvent::NoteEnd { channel, note } => {
			to_send.try_push(NoteOffEvent::new(
				offset as u32,
				Pckn::new(Match::All, channel as u16, note as u16, Match::All),
				0.0,
			))?;
		}
	}

	Ok(())
}

impl<P: Plugin> PluginStateImpl for PluginMain<P> {
	fn load(&mut self, input: &mut clack_plugin::stream::InputStream) -> Result<(), PluginError> {
		let mut buf = vec![];
		input.read_to_end(&mut buf)?;
		let params: Vec<i_am_dsp::prelude::Parameter> = from_binary(buf)?;
		// SAFETY: the processor is pinned inside the `PluginMain` box for as long as the
		// plugin lives, and we never move `P` out of it, so unwrapping the pin is sound.
		let processor = unsafe {
			self.processor.as_mut().get_unchecked_mut()
		};

		for param in params {
			if param.value.is_serialized() {
				processor.set_parameter(&param.identifier, param.value.to_set_value());
			}else {
				self.param_map.set(&param.identifier, param.value.to_set_value(), std::sync::atomic::Ordering::SeqCst);
			}
		}
		Ok(())
	}
	
	fn save(&mut self, output: &mut clack_plugin::stream::OutputStream) -> Result<(), PluginError> {
		let params = self.processor.get_parameters();
		let binary = to_binary(&params)?;
		output.write_all(&binary)?;
		Ok(())
	}
}

impl<P: Plugin> PluginMainThreadParams for PluginMain<P> {
	fn count(&mut self) -> u32 {
		self.param_map.len() as u32
	}

	fn get_info(&mut self, param_index: u32, info: &mut clack_extensions::params::ParamInfoWriter) {
		let index = param_index as usize;
		let Some(param) = self.param_map.get_by_index(index) else {
			return;
		};
		// The id is derived from the parameter's identifier rather than from its position, so
		// that adding a parameter no longer renumbers every parameter declared after it.
		let Some(stable_id) = self.param_map.query_stable_id(index) else {
			return;
		};

		let flags = if param.is_float() {
			ParamInfoFlags::IS_AUTOMATABLE
		}else {
			ParamInfoFlags::IS_AUTOMATABLE | ParamInfoFlags::IS_ENUM
		};

		let (min_value, max_value) = match &*param {
			AtomicValue::Bool { .. } => (0.0, 1.0),
			AtomicValue::Nothing => (0.0, 0.0),
			AtomicValue::Float { range, .. } => (*range.start() as f64, *range.end() as f64),
			AtomicValue::Int { range, .. } => (*range.start() as f64, *range.end() as f64),
			_ => (0.0, 0.0),
		};

		// A parameter that declares what it starts at says so; the ones that do not — a delay
		// time set by whoever constructs the effect, say — leave it to the host.
		let meta = self.param_map.query_meta(index).unwrap_or_default();
		let default_value = meta
			.default
			.map(|default| default.clamp(min_value, max_value))
			.unwrap_or(min_value);

		info.set(&ParamInfo {
			id: stable_id.into(),
			flags,
			cookie: Default::default(),
			name: meta.display_name.as_deref().unwrap_or_else(|| self.param_map.query_param_id(index).unwrap()).as_bytes(),
			module: Default::default(),
			min_value,
			max_value,
			default_value,
		});
	}

	fn get_value(&mut self, param_id: clack_plugin::prelude::ClapId) -> Option<f64> {
		let param = self.param_map.get_by_stable_id(param_id.get())?;
		let current = param.load(std::sync::atomic::Ordering::SeqCst);
		match current {
			SetValue::Bool(val) => if val { Some(1.0) } else { Some(0.0) },
			SetValue::Int(val) => Some(val as f64),
			SetValue::Float(val) => Some(val as f64),
			SetValue::Nothing => None,
			SetValue::Serialized(_) => None,
		}
	}

	fn value_to_text(
		&mut self,
		param_id: clack_plugin::prelude::ClapId,
		value: f64,
		writer: &mut clack_extensions::params::ParamDisplayWriter,
	) -> core::fmt::Result {
		use std::fmt::Write;

		let Some(index) = self.param_map.query_param_index_by_stable_id(param_id.get()) else {
			return Err(std::fmt::Error);
		};

		// The unit is what turns `0.5` into `1/8` for a synced time, or adds `ms` to a length.
		// A parameter that declares nothing keeps the plain number it has always shown.
		let unit = self.param_map.query_meta(index).unwrap_or_default().unit;
		write!(writer, "{}", unit.format(value))
	}

	fn text_to_value(&mut self, param_id: clack_plugin::prelude::ClapId, text: &std::ffi::CStr) -> Option<f64> {
		let index = self.param_map.query_param_index_by_stable_id(param_id.get())?;

		let text = text.to_str().ok()?;
		// What the unit wrote, it can read back: `1/8 dotted`, `250 ms`, or a plain number.
		let unit = self.param_map.query_meta(index).unwrap_or_default().unit;
		unit.parse(text).or_else(|| text.parse::<f64>().ok())
	}

	fn flush(
		&mut self,
		input_parameter_changes: &clack_plugin::prelude::InputEvents,
		output_parameter_changes: &mut OutputEvents,
	) {
		flush(input_parameter_changes, output_parameter_changes, &self.param_map);
	}
}

pub(crate) fn handle_param_value_event(
	param: &ParamValueEvent,
	param_map: &ParamMap,
) {
	if let Some(param_id) = param.param_id() {
		let value = param.value() as f32;
		if let Some(inner) = param_map.get_by_stable_id(param_id.get()) {
			if inner.is_int() {
				inner.store(value as i32, std::sync::atomic::Ordering::SeqCst);
				inner.set_changed(false, std::sync::atomic::Ordering::SeqCst);
			}else if inner.is_float() {
				inner.store(value, std::sync::atomic::Ordering::SeqCst);
				inner.set_changed(false, std::sync::atomic::Ordering::SeqCst);
			}else if inner.is_bool() {
				inner.store(value > 0.5, std::sync::atomic::Ordering::SeqCst);
				inner.set_changed(false, std::sync::atomic::Ordering::SeqCst);
			}
		}
	}
}

pub(crate) fn handle_param_mod_event(
	param: &ParamModEvent,
	param_map: &ParamMap,
) {
	if let Some(param_id) = param.param_id() {
		let value = param.amount() as f32;
		if let Some(inner) = param_map.get_by_stable_id(param_id.get()) {
			if inner.is_int() {
				let amount = inner.load(std::sync::atomic::Ordering::SeqCst).int().unwrap() + value as i32;
				inner.store(amount, std::sync::atomic::Ordering::SeqCst);
				inner.set_changed(false, std::sync::atomic::Ordering::SeqCst);
			}else if inner.is_float() {
				let amount = inner.load(std::sync::atomic::Ordering::SeqCst).float().unwrap() + value;
				inner.store(amount, std::sync::atomic::Ordering::SeqCst);
				inner.set_changed(false, std::sync::atomic::Ordering::SeqCst);
			}
		}
	}
}

pub(crate) fn sync_params(
	output_parameter_changes: &mut OutputEvents, 
	param_map: &ParamMap
) {
	for (index, val) in param_map.iter().enumerate() {
		if val.is_chanegd() {
			let Some(stable_id) = param_map.query_stable_id(index) else {
				continue;
			};

			let value = match val.load(std::sync::atomic::Ordering::SeqCst) {
				SetValue::Bool(v) => if v { 1.0 } else { 0.0 },
				SetValue::Int(v) => v as f64,
				SetValue::Float(v) => v as f64,
				_ => continue
			};

			let _ = output_parameter_changes.try_push(ParamValueEvent::new(
				0,
				stable_id.into(),
				Pckn::match_all(),
				value,
				Default::default(),
			));

			val.set_changed(false, std::sync::atomic::Ordering::SeqCst);
		}
	}
}

fn flush(
	input_parameter_changes: &clack_plugin::prelude::InputEvents,
	output_parameter_changes: &mut OutputEvents, 
	param_map: &ParamMap
) {
	for event in input_parameter_changes {
		match event.as_core_event() {
			Some(CoreEventSpace::ParamValue(param)) => {
				handle_param_value_event(param, param_map);
			},
			Some(CoreEventSpace::ParamMod(param)) => {
				handle_param_mod_event(param, param_map);
			},
			_ => {}
		}
	}

	sync_params(output_parameter_changes, param_map);
}

impl<P: Plugin> PluginAudioPortsImpl for PluginMain<P> {
	fn count(&mut self, is_input: bool) -> u32 {
		if is_input {
			P::INPUT_PORTS.len() as u32
		}else {
			P::OUTPUT_PORTS.len() as u32
		}
	}

	fn get(&mut self, index: u32, is_input: bool, writer: &mut clack_extensions::audio_ports::AudioPortInfoWriter) {
		let ports = if is_input {
			P::INPUT_PORTS
		}else {
			P::OUTPUT_PORTS
		};

		if index >= ports.len() as u32 {
			return;
		}

		writer.set(&AudioPortInfo {
			id: index.into(),
			name: ports[index as usize].name.as_bytes(),
			channel_count: 2,
			flags: if index == 0 {
				AudioPortFlags::IS_MAIN
			}else {
				AudioPortFlags::empty()
			},
			port_type: Some(AudioPortType::STEREO),
			in_place_pair: None,
		});
	}
}

impl<'a, P: Plugin> PluginAudioProcessorParams for AudioProcessor<'a, P> {
	fn flush(
		&mut self,
		input_parameter_changes: &clack_plugin::prelude::InputEvents,
		output_parameter_changes: &mut OutputEvents,
	) {
		flush(input_parameter_changes, output_parameter_changes, &self.param_map);
	}
}
