//! The add menus: the generators and the effects grouped by what they are.

use super::data::*;

use super::*;

pub(super) fn generator_add_menu(
	ui: &mut egui::Ui,
	sample_rate: usize,
	output_track: usize,
) -> Option<GeneratorWarpper> {
	let mut output = None;
	let mut path = None;
	for (name, generator) in GENERATOR_LIST.iter() {
		if ui.button(name).clicked() {
			output = Some(generator(sample_rate));
			path = Some(name.clone());
			ui.close();
			break;
		}
	}

	if ui.button("Close").clicked() {
		ui.close();
	}

	if let (Some(generator), Some(path)) = (output, path) {
		Some(GeneratorWarpper {
			path,
			generator,
			output_track,
			gain: 1.0,
		})
	} else {
		None
	}
}

pub(super) fn effect_add_menu(ui: &mut egui::Ui, sample_rate: usize) -> Option<EffectWarpper> {
	let mut output = None;
	let mut path = None;

	for (name_1, effect) in EFFECT_LIST.iter() {
		ui.menu_button(name_1, |ui| {
			for (name, effect) in effect.iter() {
				if ui.button(name).clicked() {
					output = Some(effect(sample_rate));
					path = Some((name_1.clone(), name.clone()));
					ui.close();
					break;
				}
			}
		});
	}

	if ui.button("Close").clicked() {
		ui.close();
	}

	if let (Some(effect), Some((path_1, path_2))) = (output, path) {
		Some(EffectWarpper {
			path: (path_1, path_2),
			effect,
			mix: 1.0,
			dry_delay: std::array::from_fn(|_| RingBuffer::new(0)),
		})
	} else {
		None
	}
}

#[cfg(feature = "standalone")]
lazy_static::lazy_static! {
	static ref NoteMap: HashMap<Key, usize> = {
		const C5: usize = 60;
		const C4: usize = 60 - 12;

		HashMap::from([
			(Key::Q, C5),
			(Key::Num2, C5 + 1),
			(Key::W, C5 + 2),
			(Key::Num3, C5 + 3),
			(Key::E, C5 + 4),
			(Key::R, C5 + 5),
			(Key::Num5, C5 + 6),
			(Key::T, C5 + 7),
			(Key::Num6, C5 + 8),
			(Key::Y, C5 + 9),
			(Key::Num7, C5 + 10),
			(Key::U, C5 + 11),
			(Key::I, C5 + 12),
			(Key::Num9, C5 + 13),
			(Key::O, C5 + 14),
			(Key::Num0, C5 + 15),
			(Key::P, C5 + 16),

			(Key::Z, C4),
			(Key::S, C4 + 1),
			(Key::X, C4 + 2),
			(Key::D, C4 + 3),
			(Key::C, C4 + 4),
			(Key::V, C4 + 5),
			(Key::G, C4 + 6),
			(Key::B, C4 + 7),
			(Key::H, C4 + 8),
			(Key::N, C4 + 9),
			(Key::J, C4 + 10),
			(Key::M, C4 + 11),
		])
	};
}

#[cfg(feature = "standalone")]
pub(super) fn simulate_midi(
	ui: &mut egui::Ui,
	ctx: &mut SimpleContext,
	key_holding: &mut HashSet<usize>,
	clear_all: bool,
	note_shift: isize,
) {
	ui.input(|input| {
		for (key, note) in NoteMap.iter() {
			if input.key_down(*key) {
				let note = if note_shift >= 0 {
					note + note_shift as usize
				} else {
					note.saturating_sub(note_shift.unsigned_abs())
				};
				if key_holding.insert(note) {
					// println!("semi-note on: {}", note);
					ctx.send_event(NoteEvent::NoteOn {
						channel: 0,
						note,
						velocity: 1.0,
					});
					// dbg!("q");
				}
			} else if input.key_released(*key) {
				let note = if note_shift >= 0 {
					note + note_shift as usize
				} else {
					note.saturating_sub(note_shift.unsigned_abs())
				};
				if key_holding.remove(&note) {
					// println!("semi-note off: {}", note);
					ctx.send_event(NoteEvent::NoteOff {
						channel: 0,
						note,
						velocity: 1.0,
					});
				}
			}
		}
	});
	if clear_all {
		key_holding.clear();
		ctx.send_event(NoteEvent::ImmediateStop);
	}
}

pub(super) fn get_range(value: &Value) -> Option<(FloatOrInt, FloatOrInt, bool)> {
	match value {
		Value::Float {
			range, logarithmic, ..
		} => Some((
			FloatOrInt::Float(*range.start()),
			FloatOrInt::Float(*range.end()),
			*logarithmic,
		)),
		Value::Int {
			range, logarithmic, ..
		} => Some((
			FloatOrInt::Int(*range.start()),
			FloatOrInt::Int(*range.end()),
			*logarithmic,
		)),
		_ => None,
	}
}

pub(super) fn draw_change_history(
	ui: &mut egui::Ui,
	history: &RingBuffer<Option<ListenedValue>>,
) -> Option<LfoedParam> {
	let mut output = None;
	for history in history.range(0..CHANGE_HISTORY_LEN) {
		let Some(history) = history else {
			continue;
		};
		let Some(range) = history.range else {
			continue;
		};

		match &history.history {
			ChangedValue::Generator {
				index, identifier, ..
			} => {
				if ui
					.button(format!("Generator {}: {}", index, identifier))
					.clicked()
				{
					output = Some(LfoedParam {
						track: None,
						index: *index,
						identifier: identifier.clone(),
						min: range.0,
						max: range.1,
						start: range.0,
						end: range.1,
						logarithmic: range.2,
					});
					ui.close();
					break;
				}
			}
			ChangedValue::Effect {
				track,
				index,
				identifier,
				..
			} => {
				if ui
					.button(format!("Effect {}: {}", index, identifier))
					.clicked()
				{
					output = Some(LfoedParam {
						track: Some(*track),
						index: *index,
						identifier: identifier.clone(),
						min: range.0,
						max: range.1,
						start: range.0,
						end: range.1,
						logarithmic: range.2,
					});
					ui.close();
					break;
				}
			}
		}
	}
	output
}
