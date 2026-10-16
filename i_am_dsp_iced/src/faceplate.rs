//! Laying an effect out from what its parameters say about themselves.
//!
//! A faceplate is the part of an interface a person meets: the controls, their
//! names, the sections they are in. None of it has to be written twice if the
//! parameters carry a [layout](i_am_dsp::prelude::ParamLayout): an effect that
//! says where each control belongs gets a window without anyone laying one out,
//! and the same description drives the module card it becomes inside a rack.
//!
//! Nothing here is required. A faceplate that wants to place every control by
//! hand still can; this is for the twenty effects that have no reason to
//! differ.

use i_am_dsp::prelude::{ParamSize, ParamWidget, Parameter, SetValue, Value};
use iced::{
	Element, Length,
	widget::{column, container, row, text},
};

use crate::{
	styles::{self, KnobSize},
	tools::{button::push_button, knob::knob, slider::slider},
};

/// How wide a control of each size is asked to be, in design units.
///
/// A slider carries a label and a value beside its track, so it asks for more
/// room than a knob, which only has to fit its own name underneath.
fn width_of(size: ParamSize, widget: ParamWidget) -> Length {
	let units = match (widget, size) {
		(ParamWidget::Slider, ParamSize::Small) => 24.0,
		(ParamWidget::Slider, ParamSize::Large) => 40.0,
		(ParamWidget::Slider, _) => 32.0,
		(_, ParamSize::Small) => 14.0,
		(_, ParamSize::Large) => 24.0,
		// A size this build does not know is drawn at the default one.
		_ => 18.0,
	};

	Length::Fixed(styles::u(units))
}

/// How big a knob of each size is drawn.
fn knob_size(size: ParamSize) -> KnobSize {
	match size {
		ParamSize::Small => KnobSize::Small,
		ParamSize::Medium => KnobSize::Medium,
		ParamSize::Large => KnobSize::Large,
		_ => KnobSize::Medium,
	}
}

/// Arranges parameters into the sections a faceplate draws.
///
/// Sections appear in the order they were first declared in, and the members of a
/// section are sorted by the order they declared; members that declared none keep
/// the order they were declared in, behind the ones that did. A parameter with no
/// group at all gets a section of its own, which is drawn without a title.
pub fn arrange<'a>(parameters: &'a [Parameter]) -> Vec<(Option<&'a str>, Vec<&'a Parameter>)> {
	let mut groups: Vec<(Option<&'a str>, Vec<&'a Parameter>)> = Vec::new();

	for parameter in parameters {
		let group = parameter.meta.layout.group.as_deref();

		match groups.iter_mut().find(|(name, _)| *name == group) {
			Some((_, members)) => members.push(parameter),
			None => groups.push((group, vec![parameter])),
		}
	}

	for (_, members) in &mut groups {
		// A stable sort, so everything without a declared order keeps the order it
		// was declared in.
		members.sort_by_key(|parameter| parameter.meta.layout.order.unwrap_or(u16::MAX));
	}

	groups
}

/// Lays a set of parameters out as one faceplate.
///
/// The controls, their names, their units, their sections and the size they are
/// drawn at all come from the parameters themselves. A parameter that declares
/// nothing still gets a control: the layout decides where it goes, not whether it
/// exists.
///
/// Changes come back through one callback, keyed by the parameter's identifier,
/// which is what a host would hand back as well.
pub fn faceplate<'a, Message: Clone + 'a>(
	parameters: &'a [Parameter],
	on_change: impl Fn(&'a str, SetValue) -> Message + Copy + 'a,
) -> Element<'a, Message> {
	let sections = arrange(parameters).into_iter().filter_map(|(group, members)| {
		let controls: Vec<_> = members
			.into_iter()
			.filter_map(|parameter| control(parameter, on_change))
			.collect();

		// A section whose every parameter turned out not to be a control -- a
		// filter that only carries its coefficients, say -- is not drawn at all.
		if controls.is_empty() {
			return None;
		}

		let content = row(controls).spacing(styles::u(styles::space::L));

		let mut card = column![].spacing(styles::u(styles::space::M));

		if let Some(group) = group {
			card = card.push(
				text(group.to_string())
					.size(styles::px(styles::font::SECTION))
					.color(styles::SECONDARY_TEXT_COLOR),
			);
		}

		// The card hugs its controls rather than filling the window: a faceplate is
		// a row of sections, not a wall of empty boxes.
		Some(
			container(card.push(content))
			.padding(styles::u(styles::space::L))
			.style(|_| container::Style {
				background: Some(styles::CARD_COLOR.into()),
				border: iced::Border {
					color: styles::CARD_BORDER_COLOR,
					width: styles::px(1.0),
					radius: styles::u(styles::size::RADIUS).into(),
				},
				..Default::default()
			})
			.into(),
		)
	});

	column(sections).spacing(styles::u(styles::space::M)).into()
}

/// Draws one parameter, when it is one a person can move.
///
/// A serialized value is state that travels with a preset rather than a control
/// -- a filter's coefficients, say -- so a faceplate leaves it alone; the host is
/// told it is not editable and this agrees with the host.
fn control<'a, Message: Clone + 'a>(
	parameter: &'a Parameter,
	on_change: impl Fn(&'a str, SetValue) -> Message + Copy + 'a,
) -> Option<Element<'a, Message>> {
	let identifier = parameter.identifier.as_str();
	let name = parameter
		.meta
		.display_name
		.clone()
		.map(|name| name.to_string())
		.unwrap_or_else(|| parameter.identifier.clone());
	let layout = parameter.meta.layout.clone();
	let unit = parameter.meta.unit;
	let default = parameter.meta.default;
	let bipolar = layout.bipolar;
	let step = layout.step;

	match &parameter.value {
		Value::Float { value, range, logarithmic } => {
			let value = *value;
			let logarithmic = *logarithmic;
			let range = range.clone();

			if layout.widget == ParamWidget::Slider {
				let mut control = slider(range, value, move |value| on_change(identifier, SetValue::Float(value)))
					.text(name)
					.formatter(move |value| unit.format(value as f64))
					.bipolar(bipolar)
					.width(width_of(layout.size, ParamWidget::Slider));

				if logarithmic {
					control = control.logarithmic();
				}
				if let Some(step) = step {
					control = control.step(step as f32);
				}
				if let Some(default) = default {
					control = control.default_value(default as f32);
				}

				return Some(control.into());
			}

			let mut control = knob(range, value, move |value| on_change(identifier, SetValue::Float(value)))
				.label(name)
				.unit(unit)
				.logarithmic(logarithmic)
				.bipolar(bipolar)
				.size(knob_size(layout.size));

			if let Some(step) = step {
				control = control.step(step as f32);
			}
			if let Some(default) = default {
				control = control.default_value(default as f32);
			}

			Some(control.into())
		},
		Value::Int { value, range, .. } => {
			let value = *value;
			let range = range.clone();

			if layout.widget == ParamWidget::Slider {
				// A whole number is still a slider if it asked to be one; it snaps
				// to its own steps because it has no fractions to offer.
				return Some(
					slider(range, value, move |value| on_change(identifier, SetValue::Int(value)))
						.text(name)
						.formatter(|value| format!("{value:.0}"))
						.step(step.unwrap_or(1.0) as f32)
						.width(width_of(layout.size, ParamWidget::Slider))
						.into(),
				);
			}

			let mut control = knob(range, value, move |value| on_change(identifier, SetValue::Int(value)))
				.label(name)
				.bipolar(bipolar)
				.size(knob_size(layout.size));

			if let Some(step) = step {
				control = control.step(step as f32);
			}
			if let Some(default) = default {
				control = control.default_value(default as f32);
			}

			Some(control.into())
		},
		Value::Bool(value) => {
			// A latching button, which is the same shape as everything else here.
			Some(
				push_button(name)
					.accent(*value)
					.on_press(on_change(identifier, SetValue::Bool(!value)))
					.into(),
			)
		},
		// Serialized state and placeholders are not controls.
		_ => None,
	}
}
