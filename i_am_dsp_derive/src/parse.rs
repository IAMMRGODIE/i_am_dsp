use syn::{Attribute, Error, Ident, Lit, LitInt, Member, Result, Type};

/// The kind of a numeric/bool field, inferred from its type.
pub(crate) enum FieldKind {
	Float,
	Signed,
	Unsigned,
	Bool,
}

/// Classify a field type by its last path segment, so path-qualified forms like
/// `std::primitive::f32` (or plain `f32`) are recognised uniformly. Alias types are
/// not resolved (that would require name resolution); unknown types return `None`.
pub(crate) fn classify_type(ty: &Type) -> Option<FieldKind> {
	let path = match ty {
		Type::Path(path) => path,
		_ => return None,
	};
	let last = path.path.segments.last()?.ident.to_string();
	match last.as_str() {
		"f32" | "f64" => Some(FieldKind::Float),
		"i8" | "i16" | "i32" | "i64" | "isize" => Some(FieldKind::Signed),
		"u8" | "u16" | "u32" | "u64" | "usize" => Some(FieldKind::Unsigned),
		"bool" => Some(FieldKind::Bool),
		_ => None,
	}
}

/// The parsed min..=max bounds of a `#[range(...)]` / `#[default_*_range(...)]` attribute.
#[derive(Debug)]
pub(crate) enum ParamRange {
	Float(f32, f32),
	Int(i64, i64),
}

/// The unit a parameter declares with `#[unit(...)]`, before it becomes tokens.
#[derive(Debug, Clone, Copy)]
pub(crate) enum ParameterUnit {
	None,
	Milliseconds,
	Hertz,
	Beats,
	Seconds,
	Samples,
	Decibels,
	Percent,
}

#[derive(Debug, Default)]
pub(crate) struct FieldConfig {
	pub(crate) range: Option<ParamRange>,
	pub(crate) id_name: String,
	pub(crate) id_stable: Option<u32>,
	/// What a host should call the parameter, from `#[id(display = "...")]`.
	pub(crate) id_display: Option<String>,
	/// The value the parameter starts at, from `#[default(...)]`.
	pub(crate) param_default: Option<f64>,
	pub(crate) unit: Option<ParameterUnit>,
	/// The names of an integer parameter's modes, from `#[choices("Low", ..)]`.
	pub(crate) choices: Option<Vec<String>>,
	/// Where the parameter wants to sit in a faceplate, from `#[layout(..)]`.
	#[cfg(feature = "layout")]
	pub(crate) layout: ParamLayoutConfig,
	pub(crate) persist_serialize: Option<String>,
	pub(crate) persist_deserialize: Option<String>,
	pub(crate) skip: bool,
	pub(crate) is_sub_param: bool,
	pub(crate) is_bool: bool,
	pub(crate) logarithmic: bool,
	pub(crate) is_serde: bool,
}

pub(crate) fn handle_attribute(
	member: &Member,
	field_ty: &Type,
	attrs: &[Attribute],
	default_float_range: (f32, f32),
	default_int_range: (i64, i64),
	default_uint_range: (i64, i64),
) -> Result<FieldConfig> {
	let field_name = match member {
		Member::Named(ident) => ident.to_string(),
		Member::Unnamed(index) => index.index.to_string(),
	};

	let mut config = FieldConfig::default();
	for attr in attrs {
		let Some(path) = attr.path().get_ident() else {
			continue;
		};
		let path = path.to_string();
		let path = path.trim().to_string();
		match path.as_str() {
			"range" => {
				config.range = Some(parse_range_attribute(attr)?);
			}
			"id" => {
				let (prefix, name, stable, display) = parse_id_attribute(attr)?;
				config.id_stable = stable;
				config.id_display = display;
				let prefix = prefix.unwrap_or_default();

				let name = if let Some(name) = name {
					name
				} else {
					field_name.clone()
				};

				config.id_name = if config.id_display.is_some()
					&& name == field_name
					&& prefix.is_empty()
					&& config.id_stable.is_none()
				{
					String::new()
				} else {
					format!("{}{}", prefix, name)
				};

				if config.id_name.contains('.') {
					return Err(Error::new_spanned(attr, "id attribute cannot contain `.`"));
				}

				if config.id_name.is_empty() && config.id_display.is_none() {
					return Err(Error::new_spanned(attr, "id attribute cannot be empty"));
				}

				if !config.id_name.is_empty() && syn::parse_str::<Ident>(&config.id_name).is_err() {
					return Err(Error::new_spanned(attr, "Name must be a valid rust ident"));
				}
			}
			"persist" => {
				let (serialize, deserialize) = parse_persist_attribute(attr)?;
				config.persist_serialize = Some(serialize);
				config.persist_deserialize = Some(deserialize);
			}
			"default" => config.param_default = Some(parse_default_attribute(attr)?),
			"unit" => config.unit = Some(parse_unit_attribute(attr)?),
			"choices" => config.choices = Some(parse_choices_attribute(attr)?),
			#[cfg(feature = "layout")]
			"layout" => config.layout = parse_layout_attribute(attr)?,
			"logarithmic" => config.logarithmic = true,
			"skip" => config.skip = true,
			"sub_param" => config.is_sub_param = true,
			"serde" => config.is_serde = true,
			_ => {}
		}
	}

	if config.skip {
		let should_be_none = config.range.is_none()
			&& config.persist_deserialize.is_none()
			&& config.id_name.is_empty()
			&& config.id_stable.is_none()
			&& config.param_default.is_none()
			&& config.unit.is_none()
			&& !config.is_sub_param
			&& !config.is_serde;

		if !should_be_none {
			return Err(Error::new_spanned(
				member,
				"`skip` attribute cannot be used with other attributes",
			));
		}
	}

	if config.persist_serialize.is_some() {
		let should_be_none =
			config.range.is_none() && !config.skip && !config.is_sub_param && !config.is_serde;

		if !should_be_none {
			return Err(Error::new_spanned(
				member,
				"`persist` serialize attribute cannot be used with other attributes",
			));
		}
	}

	if config.range.is_some() {
		let should_be_none = config.persist_serialize.is_none()
			&& !config.skip
			&& !config.is_sub_param
			&& !config.is_serde;

		if !should_be_none {
			return Err(Error::new_spanned(
				member,
				"`range` attribute cannot be used with other parameter attributes",
			));
		}
	}

	if config.is_sub_param {
		let should_be_none = config.range.is_none()
			&& config.persist_serialize.is_none()
			&& config.persist_deserialize.is_none()
			&& config.id_name.is_empty()
			&& config.id_stable.is_none()
			&& config.param_default.is_none()
			&& config.unit.is_none()
			&& !config.skip
			&& !config.is_serde;

		if !should_be_none {
			return Err(Error::new_spanned(
				member,
				"`sub_param` attribute cannot be used with other attributes (including `id`)",
			));
		}
	}

	// Infer the parameter kind from the field type when no explicit attribute is given.
	if config.range.is_none()
		&& config.persist_serialize.is_none()
		&& !config.skip
		&& !config.is_sub_param
		&& !config.is_serde
	{
		match classify_type(field_ty) {
			Some(FieldKind::Float) => {
				config.range = Some(ParamRange::Float(
					default_float_range.0,
					default_float_range.1,
				))
			}
			Some(FieldKind::Signed) => {
				config.range = Some(ParamRange::Int(default_int_range.0, default_int_range.1))
			}
			Some(FieldKind::Unsigned) => {
				config.range = Some(ParamRange::Int(default_uint_range.0, default_uint_range.1))
			}
			Some(FieldKind::Bool) => config.is_bool = true,
			None => {
				return Err(Error::new_spanned(
					member,
					format!(
						"Unsupported type for parameter field `{field_name}`, consider adding `range`, `skip`, `sub_param`, `persist` or `serde` to it"
					),
				));
			}
		}
	}

	Ok(config)
}

pub(crate) fn parse_range_attribute(attr: &Attribute) -> Result<ParamRange> {
	let mut min_float: Option<f32> = None;
	let mut min_int: Option<i64> = None;
	let mut max_float: Option<f32> = None;
	let mut max_int: Option<i64> = None;
	attr.parse_nested_meta(|meta| {
		if meta.path.is_ident("min") {
			let val: Lit = meta.value()?.parse()?;
			match val {
				Lit::Float(inner) => match inner.base10_parse::<f32>() {
					Ok(t) => min_float = Some(t),
					Err(e) => return Err(Error::new_spanned(attr, format!("Invalid float, {e}"))),
				},
				Lit::Int(inner) => match inner.base10_parse::<i64>() {
					Ok(t) => min_int = Some(t),
					Err(e) => return Err(Error::new_spanned(attr, format!("Invalid int, {e}"))),
				},
				_ => {
					return Err(Error::new_spanned(
						attr,
						"Invalid literal for min, expected float or int",
					));
				}
			}
		} else if meta.path.is_ident("max") {
			let val: Lit = meta.value()?.parse()?;
			match val {
				Lit::Float(inner) => match inner.base10_parse::<f32>() {
					Ok(t) => max_float = Some(t),
					Err(e) => return Err(Error::new_spanned(attr, format!("Invalid float, {e}"))),
				},
				Lit::Int(inner) => match inner.base10_parse::<i64>() {
					Ok(t) => max_int = Some(t),
					Err(e) => return Err(Error::new_spanned(attr, format!("Invalid int, {e}"))),
				},
				_ => {
					return Err(Error::new_spanned(
						attr,
						"Invalid literal for max, expected float or int",
					));
				}
			}
		}
		Ok(())
	})?;

	let is_float = min_float.is_some() || max_float.is_some();
	let is_int = min_int.is_some() || max_int.is_some();
	if is_float && is_int {
		return Err(Error::new_spanned(
			attr,
			"min and max must be of the same type",
		));
	}

	let range = if is_float {
		ParamRange::Float(
			min_float.or(max_float).unwrap_or(0.0),
			max_float.or(min_float).unwrap_or(0.0),
		)
	} else {
		ParamRange::Int(
			min_int.or(max_int).unwrap_or(0),
			max_int.or(min_int).unwrap_or(0),
		)
	};

	match &range {
		ParamRange::Float(min, max) if min > max => {
			return Err(Error::new_spanned(
				attr,
				"min must be less than or equal to max",
			));
		}
		ParamRange::Int(min, max) if min > max => {
			return Err(Error::new_spanned(
				attr,
				"min must be less than or equal to max",
			));
		}
		_ => {}
	}

	Ok(range)
}

/// Reads `#[default(1.0)]`: a number, or a bool for a bool parameter.
///
/// The value is kept as `f64` because that is what a host is handed.
pub(crate) fn parse_default_attribute(attr: &Attribute) -> Result<f64> {
	let val: Lit = attr.parse_args()?;
	match val {
		Lit::Float(inner) => inner.base10_parse::<f64>(),
		Lit::Int(inner) => inner.base10_parse::<f64>(),
		Lit::Bool(inner) => Ok(if inner.value { 1.0 } else { 0.0 }),
		other => Err(Error::new_spanned(
			other,
			"default expects a number, or a bool",
		)),
	}
}

/// Reads `#[unit(beats)]` and friends: how a host should show the value.
pub(crate) fn parse_choices_attribute(attr: &Attribute) -> Result<Vec<String>> {
	use syn::punctuated::Punctuated;

	let names =
		attr.parse_args_with(Punctuated::<syn::LitStr, syn::Token![,]>::parse_terminated)?;
	let choices: Vec<String> = names.iter().map(|name| name.value()).collect();

	if choices.is_empty() {
		return Err(Error::new_spanned(
			attr,
			"A parameter needs at least one choice",
		));
	}

	for choice in &choices {
		if choice.trim().is_empty() {
			return Err(Error::new_spanned(attr, "A choice name must not be empty"));
		}
	}

	Ok(choices)
}

pub(crate) fn parse_unit_attribute(attr: &Attribute) -> Result<ParameterUnit> {
	let ident: Ident = attr.parse_args()?;

	match ident.to_string().as_str() {
		"none" => Ok(ParameterUnit::None),
		"milliseconds" | "ms" => Ok(ParameterUnit::Milliseconds),
		"hertz" | "hz" => Ok(ParameterUnit::Hertz),
		"beats" => Ok(ParameterUnit::Beats),
		"seconds" | "s" => Ok(ParameterUnit::Seconds),
		"samples" => Ok(ParameterUnit::Samples),
		"decibels" | "db" => Ok(ParameterUnit::Decibels),
		"percent" => Ok(ParameterUnit::Percent),
		other => Err(Error::new_spanned(
			&ident,
			format!("unknown unit `{other}`: expected one of none, milliseconds, hertz, beats"),
		)),
	}
}

#[cfg(feature = "layout")]
/// How a parameter asked to be drawn, from the widget option of a layout attribute.
#[derive(Debug, Clone, Copy)]
pub(crate) enum LayoutWidget {
	Auto,
	Knob,
	Slider,
	Toggle,
	Choice,
	Notes,
}

#[cfg(feature = "layout")]
/// How much room a parameter asked for, from the size option of a layout attribute.
#[derive(Debug, Clone, Copy)]
pub(crate) enum LayoutSize {
	Small,
	Medium,
	Large,
}

#[cfg(feature = "layout")]
/// What a field's layout attribute asked for.
#[derive(Debug, Default)]
pub(crate) struct ParamLayoutConfig {
	pub(crate) group: Option<String>,
	pub(crate) order: Option<u16>,
	pub(crate) widget: Option<LayoutWidget>,
	pub(crate) size: Option<LayoutSize>,
	pub(crate) bipolar: bool,
	pub(crate) step: Option<f64>,
	pub(crate) hero: Option<String>,
}

#[cfg(feature = "layout")]
/// Reads the layout attribute: where a parameter wants to sit in a faceplate.
///
/// Every option is optional, and every one of them is a hint. The attribute
/// exists so that an effect can say how it wants to be met without a faceplate
/// having to be written for it.
pub(crate) fn parse_layout_attribute(attr: &Attribute) -> Result<ParamLayoutConfig> {
	let mut layout = ParamLayoutConfig::default();

	attr.parse_nested_meta(|meta| {
		if meta.path.is_ident("group") {
			let val: Lit = meta.value()?.parse()?;
			match val {
				Lit::Str(inner) => {
					let value = inner.value();
					if value.trim().is_empty() {
						return Err(Error::new_spanned(attr, "A group name must not be empty"));
					}
					layout.group = Some(value);
				}
				_ => {
					return Err(Error::new_spanned(
						attr,
						"Invalid literal for group, expected a string",
					));
				}
			}
		} else if meta.path.is_ident("order") {
			let val: LitInt = meta.value()?.parse()?;
			let order: u32 = val.base10_parse()?;
			layout.order = Some(
				u16::try_from(order)
					.map_err(|_| Error::new_spanned(attr, "An order must fit in 16 bits"))?,
			);
		} else if meta.path.is_ident("widget") {
			let ident: Ident = meta.value()?.parse()?;
			layout.widget = Some(match ident.to_string().as_str() {
				"auto" => LayoutWidget::Auto,
				"knob" => LayoutWidget::Knob,
				"slider" => LayoutWidget::Slider,
				"toggle" => LayoutWidget::Toggle,
				"choice" => LayoutWidget::Choice,
				"notes" => LayoutWidget::Notes,
				other => {
					return Err(Error::new_spanned(
						&ident,
						format!(
							"unknown widget {other}: expected auto, knob, slider, toggle, choice or notes"
						),
					));
				}
			});
		} else if meta.path.is_ident("size") {
			let ident: Ident = meta.value()?.parse()?;
			layout.size = Some(match ident.to_string().as_str() {
				"small" => LayoutSize::Small,
				"medium" => LayoutSize::Medium,
				"large" => LayoutSize::Large,
				other => {
					return Err(Error::new_spanned(
						&ident,
						format!("unknown size {other}: expected small, medium or large"),
					));
				}
			});
		} else if meta.path.is_ident("hero") {
			let val: Lit = meta.value()?.parse()?;
			match val {
				Lit::Str(inner) => {
					let value = inner.value();
					if value.trim().is_empty() {
						return Err(Error::new_spanned(attr, "A hero name must not be empty"));
					}
					layout.hero = Some(value);
				}
				_ => {
					return Err(Error::new_spanned(
						attr,
						"Invalid literal for hero, expected a string",
					));
				}
			}
		} else if meta.path.is_ident("bipolar") {
			layout.bipolar = true;
		} else if meta.path.is_ident("step") {
			let val: Lit = meta.value()?.parse()?;
			let step = match val {
				Lit::Float(inner) => inner.base10_parse::<f64>()?,
				Lit::Int(inner) => inner.base10_parse::<f64>()?,
				other => return Err(Error::new_spanned(other, "step expects a number")),
			};

			if step <= 0.0 {
				return Err(Error::new_spanned(attr, "A step must be greater than zero"));
			}

			layout.step = Some(step);
		} else {
			return Err(Error::new_spanned(
				attr,
				"unknown layout option: expected group, order, widget, size, bipolar or step",
			));
		}

		Ok(())
	})?;

	Ok(layout)
}

/// What a field's `#[id(..)]` attribute asked for: the identifier prefix, the identifier
/// name, a pinned host id and the display name.
type IdAttribute = (Option<String>, Option<String>, Option<u32>, Option<String>);

pub(crate) fn parse_id_attribute(attr: &Attribute) -> Result<IdAttribute> {
	let mut prefix = None;
	let mut name = None;
	let mut stable = None;
	let mut display = None;
	attr.parse_nested_meta(|meta| {
		if meta.path.is_ident("stable") {
			let val: LitInt = meta.value()?.parse()?;
			let value: u32 = val.base10_parse()?;
			// A VST3 wrapper strips the top bit of the id before handing it to the host, so an id
			// above 31 bits would address a different parameter there.
			if value > 0x7FFF_FFFF {
				return Err(Error::new_spanned(
					attr,
					"Stable id must fit in 31 bits (a VST3 host reserves the top bit)",
				));
			}
			stable = Some(value);
		} else if meta.path.is_ident("prefix") {
			let val: Lit = meta.value()?.parse()?;
			match val {
				Lit::Str(inner) => prefix = Some(inner.value()),
				_ => {
					return Err(Error::new_spanned(
						attr,
						"Invalid literal for prefix, expected string",
					));
				}
			}
		} else if meta.path.is_ident("name") {
			let val: Lit = meta.value()?.parse()?;
			match val {
				Lit::Str(inner) => {
					let value = inner.value();
					let value = value.trim().to_string();
					if syn::parse_str::<Ident>(&value).is_err() {
						return Err(Error::new_spanned(attr, "Name must be a valid rust ident"));
					}
					name = Some(inner.value());
				}
				_ => {
					return Err(Error::new_spanned(
						attr,
						"Invalid literal for name, expected string",
					));
				}
			}
		} else if meta.path.is_ident("display") {
			// What a host calls the parameter. Unlike `name` this never touches the identifier,
			// so it is free-form and may be combined with `sub_param`.
			let val: Lit = meta.value()?.parse()?;
			match val {
				Lit::Str(inner) => {
					// An empty display is a level that contributes no word: the
					// sub-parameters under it sit flat under the product's name.
					display = Some(inner.value());
				}
				_ => {
					return Err(Error::new_spanned(
						attr,
						"Invalid literal for display, expected string",
					));
				}
			}
		}
		Ok(())
	})?;
	Ok((prefix, name, stable, display))
}

pub(crate) fn parse_persist_attribute(attr: &Attribute) -> Result<(String, String)> {
	let mut serialize = None;
	let mut deserialize = None;

	attr.parse_nested_meta(|meta| {
		if meta.path.is_ident("serialize") {
			let val: Lit = meta.value()?.parse()?;
			match val {
				Lit::Str(inner) => serialize = Some(inner.value()),
				_ => {
					return Err(Error::new_spanned(
						attr,
						"Invalid literal for serialize, expected string",
					));
				}
			}
		} else if meta.path.is_ident("deserialize") {
			let val: Lit = meta.value()?.parse()?;
			match val {
				Lit::Str(inner) => deserialize = Some(inner.value()),
				_ => {
					return Err(Error::new_spanned(
						attr,
						"Invalid literal for deserialize, expected string",
					));
				}
			}
		}
		Ok(())
	})?;

	if let (Some(serialize), Some(deserialize)) = (serialize, deserialize) {
		Ok((serialize, deserialize))
	} else {
		Err(Error::new_spanned(
			attr,
			"Missing serialize or deserialize attribute",
		))
	}
}
