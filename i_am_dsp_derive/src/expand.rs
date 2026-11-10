use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DataStruct, DeriveInput, Error, Field, Fields, Ident, Member, Result};

use crate::parse::*;

pub(crate) fn impl_config(input: &DeriveInput) -> Result<TokenStream> {
	let struct_name = &input.ident;
	let generics = &input.generics;
	let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

	let mut default_float_range = (0.0f32, 1.0f32);
	let mut default_int_range: (i64, i64) = (0, 256);
	let mut default_uint_range: (i64, i64) = (0, 256);
	let mut struct_id_prefix = String::new();

	for attr in &input.attrs {
		let Some(path) = attr.path().get_ident() else {
			continue;
		};
		let path = path.to_string();
		let path = path.trim().to_string();
		match path.as_str() {
			"default_float_range" => {
				let range = parse_range_attribute(attr)?;
				if let ParamRange::Float(min, max) = range {
					default_float_range = (min, max);
				} else {
					return Err(Error::new_spanned(
						attr,
						"default_float_range expects float bounds",
					));
				}
			}
			"default_int_range" => {
				let range = parse_range_attribute(attr)?;
				if let ParamRange::Int(min, max) = range {
					default_int_range = (min, max);
				} else {
					return Err(Error::new_spanned(
						attr,
						"default_int_range expects integer bounds",
					));
				}
			}
			"default_uint_range" => {
				let range = parse_range_attribute(attr)?;
				if let ParamRange::Int(min, max) = range {
					if min < 0 || max < 0 {
						return Err(Error::new_spanned(
							attr,
							"default_uint_range must be non-negative",
						));
					}
					default_uint_range = (min, max);
				} else {
					return Err(Error::new_spanned(
						attr,
						"default_uint_range expects integer bounds",
					));
				}
			}
			"id" => {
				let (prefix, name, stable, display) = parse_id_attribute(attr)?;
				if name.is_some() || stable.is_some() || display.is_some() {
					return Err(Error::new_spanned(
						attr,
						"Struct-level `id` attribute only supports `prefix`",
					));
				}
				if let Some(prefix) = prefix {
					struct_id_prefix = prefix;
				}
			}
			_ => continue,
		}
	}

	let fields: Vec<(Member, String, &Field)> = match &input.data {
		Data::Struct(DataStruct { fields, .. }) => match fields {
			Fields::Named(fields) => fields
				.named
				.iter()
				.map(|field| {
					let ident = field.ident.as_ref().expect("named field has ident");
					(Member::Named(ident.clone()), ident.to_string(), field)
				})
				.collect(),
			Fields::Unnamed(fields) => fields
				.unnamed
				.iter()
				.enumerate()
				.map(|(index, field)| (Member::Unnamed(index.into()), index.to_string(), field))
				.collect(),
			Fields::Unit => vec![],
		},
		Data::Enum(_) => {
			return Err(Error::new_spanned(
				input,
				"`Parameters` derive only supports structs; enums are not supported. Consider restructuring the enum into a struct (e.g. with a tag/index field) or implementing `Parameters` manually",
			));
		}
		Data::Union(_) => {
			return Err(Error::new_spanned(
				input,
				"`Parameters` derive only supports structs; unions are not supported. Implement `Parameters` manually instead",
			));
		}
	};

	if fields.is_empty() {
		return Ok(quote! {
			impl #impl_generics i_am_dsp::prelude::Parameters for #struct_name #ty_generics #where_clause {
				fn get_parameters(&self) -> Vec<i_am_dsp::prelude::Parameter> {
					vec![]
				}

				fn set_parameter(&mut self, _: &str, _: i_am_dsp::prelude::SetValue) -> bool {
					false
				}
			}
		}
		.into());
	}

	let mut field_configs = Vec::new();

	for (member, name, field) in &fields {
		let field_type = &field.ty;

		let config = handle_attribute(
			member,
			field_type,
			&field.attrs,
			default_float_range,
			default_int_range,
			default_uint_range,
		)?;

		// Validate that any explicit `range` matches the field type.
		if let Some(range) = &config.range {
			let kind = classify_type(field_type);
			match (range, kind) {
				(ParamRange::Float(..), Some(FieldKind::Float)) => {}
				(ParamRange::Int(min, max), Some(kind))
					if matches!(kind, FieldKind::Signed | FieldKind::Unsigned) =>
				{
					let min = *min;
					let max = *max;
					if matches!(kind, FieldKind::Unsigned) && (min < 0 || max < 0) {
						return Err(Error::new_spanned(
							member,
							"Unsigned integer range attribute must be non-negative",
						));
					}
				}
				(ParamRange::Float(..), _) => {
					return Err(Error::new_spanned(
						member,
						"Cannot use float range attribute with non-float type",
					));
				}
				(ParamRange::Int(..), _) => {
					return Err(Error::new_spanned(
						member,
						"Cannot use int range attribute with non-integer type",
					));
				}
			}
		}

		field_configs.push((config, member.to_owned(), name.clone(), field_type));
	}

	let mut parameter_imply = vec![];
	let mut setter_imply = vec![];
	let mut paramters_map_create = vec![];
	let mut override_imply = vec![];
	// Stable ids pinned inside one struct have to differ, or two parameters would share the id
	// a host keys its automation on. Ids pinned in *other* structs cannot be seen here;
	// `ParamMap` catches those at run time.
	let mut pinned_ids: std::collections::HashMap<u32, String> = std::collections::HashMap::new();
	let field_count = field_configs.len();
	let mut param_index: usize = 0;
	for (config, member, field_name, field_type) in field_configs {
		if config.skip {
			continue;
		}

		// `#[sub_param]` nesting relies on the raw field name / index to build dotted
		// identifiers, so renaming attributes cannot be combined with it.
		let is_sub_param = config.is_sub_param;
		// For positional (tuple) fields the default identifier is the running parameter
		// ordinal, so it stays aligned with `get_parameters()` order and with
		// `set_parameter_by_index` even when `#[skip]` fields create holes.
		let default_id = match &member {
			Member::Named(_) => field_name.clone(),
			Member::Unnamed(_) => param_index.to_string(),
		};
		let id_string = if is_sub_param {
			field_name.clone()
		} else if config.id_name.is_empty() {
			format!("{}{}", struct_id_prefix, default_id)
		} else {
			config.id_name
		};
		let logarithmic = config.logarithmic;
		let meta = {
			let default = match config.param_default {
				Some(value) => quote!(Some(#value)),
				None => quote!(None),
			};
			let unit = match config.unit {
				Some(ParameterUnit::Milliseconds) => quote!(i_am_dsp::prelude::Unit::Milliseconds),
				Some(ParameterUnit::Hertz) => quote!(i_am_dsp::prelude::Unit::Hertz),
				Some(ParameterUnit::Beats) => quote!(i_am_dsp::prelude::Unit::Beats),
				Some(ParameterUnit::Seconds) => quote!(i_am_dsp::prelude::Unit::Seconds),
				Some(ParameterUnit::Samples) => quote!(i_am_dsp::prelude::Unit::Samples),
				Some(ParameterUnit::Decibels) => quote!(i_am_dsp::prelude::Unit::Decibels),
				Some(ParameterUnit::Percent) => quote!(i_am_dsp::prelude::Unit::Percent),
				Some(ParameterUnit::None) | None => quote!(i_am_dsp::prelude::Unit::None),
			};
			let display = match &config.id_display {
				Some(name) => quote!(Some(std::borrow::Cow::Borrowed(#name))),
				None => quote!(None),
			};
			let choices = match &config.choices {
				Some(choices) => quote!(Some(&[#(#choices),*])),
				None => quote!(None),
			};

			// The layout half is behind a feature on both sides of the boundary: the
			// derive only emits the field when it is asked to, and i_am_dsp only has
			// the field to emit into under the same feature, because one forwards the
			// other. A crate that brings its own interface turns it off and pays
			// nothing for it.
			#[cfg(feature = "layout")]
			let meta = {
				let group = match &config.layout.group {
					Some(group) => quote!(Some(std::borrow::Cow::Borrowed(#group))),
					None => quote!(None),
				};
				let order = match config.layout.order {
					Some(order) => quote!(Some(#order)),
					None => quote!(None),
				};
				let widget = match config.layout.widget.unwrap_or(LayoutWidget::Auto) {
					LayoutWidget::Auto => quote!(i_am_dsp::prelude::ParamWidget::Auto),
					LayoutWidget::Knob => quote!(i_am_dsp::prelude::ParamWidget::Knob),
					LayoutWidget::Slider => quote!(i_am_dsp::prelude::ParamWidget::Slider),
					LayoutWidget::Toggle => quote!(i_am_dsp::prelude::ParamWidget::Toggle),
					LayoutWidget::Choice => quote!(i_am_dsp::prelude::ParamWidget::Choice),
					LayoutWidget::Notes => quote!(i_am_dsp::prelude::ParamWidget::Notes),
				};
				let size = match config.layout.size.unwrap_or(LayoutSize::Medium) {
					LayoutSize::Small => quote!(i_am_dsp::prelude::ParamSize::Small),
					LayoutSize::Medium => quote!(i_am_dsp::prelude::ParamSize::Medium),
					LayoutSize::Large => quote!(i_am_dsp::prelude::ParamSize::Large),
				};
				let bipolar = config.layout.bipolar;
				let step = match config.layout.step {
					Some(step) => quote!(Some(#step)),
					None => quote!(None),
				};
				let hero = match &config.layout.hero {
					Some(hero) => quote!(Some(std::borrow::Cow::Borrowed(#hero))),
					None => quote!(None),
				};
				let layout = quote! {
					i_am_dsp::prelude::ParamLayout {
						group: #group,
						order: #order,
						widget: #widget,
						size: #size,
						bipolar: #bipolar,
						step: #step,
						hero: #hero,
					}
				};

				quote! {
					i_am_dsp::prelude::ParameterMeta { default: #default, unit: #unit, display_name: #display, choices: #choices, layout: #layout }
				}
			};

			#[cfg(not(feature = "layout"))]
			let meta = quote! {
				i_am_dsp::prelude::ParameterMeta { default: #default, unit: #unit, display_name: #display, choices: #choices }
			};

			meta
		};

		if let Some(stable_id) = config.id_stable {
			if let Some(previous) = pinned_ids.insert(stable_id, id_string.clone()) {
				return Err(Error::new_spanned(
					&member,
					format!("Stable id {stable_id} is already pinned by `{previous}`"),
				));
			}

			override_imply.push(quote! {
				overrides.push((#id_string.to_string(), #stable_id));
			});
		} else if is_sub_param {
			// A nested struct pins its own ids; its identifiers are prefixed with the field name,
			// so the pins have to be prefixed the same way to keep matching.
			override_imply.push(quote! {
				for (identifier, id) in i_am_dsp::prelude::Parameters::stable_id_overrides(&self.#member) {
					overrides.push((format!("{}.{}", #field_name, identifier), id));
				}
			});
		}

		paramters_map_create.push(quote! {
			#id_string => #param_index,
		});

		if let Some(range) = config.range {
			match range {
				ParamRange::Float(min, max) => {
					parameter_imply.push(quote! {
						parameters.push(i_am_dsp::prelude::Parameter {
							identifier: #id_string.to_string(),
							value: i_am_dsp::prelude::Value::Float {
								value: self.#member as f32,
								range: #min..=#max,
								logarithmic: #logarithmic,
							},
						meta: #meta,
						});
					});
					setter_imply.push(quote! {
						#param_index => {
							if let i_am_dsp::prelude::SetValue::Float(value) = &value {
								let value = *value;
								self.#member = value as #field_type;
								return true;
							}
						},
					});
				}
				ParamRange::Int(min, max) => {
					let min_i32 = i32::try_from(min).map_err(|_| {
						Error::new_spanned(
							&member,
							format!(
								"Range bound {min} does not fit in i32 (the host-facing integer parameter type)"
							),
						)
					})?;
					let max_i32 = i32::try_from(max).map_err(|_| {
						Error::new_spanned(
							&member,
							format!(
								"Range bound {max} does not fit in i32 (the host-facing integer parameter type)"
							),
						)
					})?;
					parameter_imply.push(quote! {
						parameters.push(i_am_dsp::prelude::Parameter {
							identifier: #id_string.to_string(),
							value: i_am_dsp::prelude::Value::Int {
								value: self.#member as i32,
								range: #min_i32..=#max_i32,
								logarithmic: #logarithmic,
							},
						meta: #meta,
						});
					});
					setter_imply.push(quote! {
						#param_index => {
							if let i_am_dsp::prelude::SetValue::Int(value) = &value {
								let value = *value;
								self.#member = value as #field_type;
								return true;
							}
						},
					});
				}
			}
			param_index += 1;
			continue;
		}

		if config.is_serde {
			parameter_imply.push(quote! {
				let parsed = i_am_dsp::prelude::to_binary(&self.#member).expect("Failed to serialize value");
				parameters.push(i_am_dsp::prelude::Parameter {
					identifier: #id_string.to_string(),
					value: i_am_dsp::prelude::Value::Serialized(parsed),
					meta: #meta,
				});
			});
			setter_imply.push(quote! {
				#param_index => {
					let owned_value = std::mem::take(&mut value);
					match owned_value {
						i_am_dsp::prelude::SetValue::Serialized(parsed) => {
							let value = i_am_dsp::prelude::from_binary(parsed).expect("Failed to deserialize value");
							self.#member = value;
							return true;
						},
						other => value = other,
					}
				},
			});
		} else if let (Some(serialize), Some(deserialize)) =
			(config.persist_serialize, config.persist_deserialize)
		{
			let serialize = Ident::new(&serialize, proc_macro2::Span::call_site());
			let deserialize = Ident::new(&deserialize, proc_macro2::Span::call_site());
			parameter_imply.push(quote! {
				let parsed = #serialize(&self.#member);
				parameters.push(i_am_dsp::prelude::Parameter {
					identifier: #id_string.to_string(),
					value: i_am_dsp::prelude::Value::Serialized(parsed),
						meta: #meta,
				});
			});
			setter_imply.push(quote! {
				#param_index => {
					let owned_value = std::mem::take(&mut value);
					match owned_value {
						i_am_dsp::prelude::SetValue::Serialized(parsed) => {
							let value = #deserialize(parsed);
							self.#member = value;
							return true;
						},
						other => value = other,
					}
				},
			});
		} else if is_sub_param {
			let display_prefix = config
				.id_display
				.clone()
				.unwrap_or_else(|| field_name.clone());
			let sub_head = format!("{}.", field_name);
			let sub_head_len = sub_head.len();
			// Where the level itself asked to sit, for the children that did not say.
			// Without the layout feature there is no such place to inherit.
			#[cfg(feature = "layout")]
			let inherit_layout = {
				let parent_group = match &config.layout.group {
					Some(group) => quote!(Some(std::borrow::Cow::Borrowed(#group))),
					None => quote!(None),
				};
				let parent_order = match config.layout.order {
					Some(order) => quote!(Some(#order)),
					None => quote!(None),
				};

				quote! {
					// A nested level sits wherever its parent sits, unless the child
					// asked for somewhere of its own: a Tempo Beats control belongs
					// with the rest of the timing controls.
					if p.meta.layout.group.is_none() {
						p.meta.layout.group = #parent_group;
					}
					if p.meta.layout.order.is_none() {
						p.meta.layout.order = #parent_order;
					}
				}
			};

			#[cfg(not(feature = "layout"))]
			let inherit_layout = quote! {};
			parameter_imply.push(quote! {
				let mut sub_parameters = i_am_dsp::prelude::Parameters::get_parameters(&self.#member);
				sub_parameters.iter_mut().for_each(|p| {
					p.identifier = format!("{}{}", #sub_head, p.identifier);
					// A host reads a nested parameter by name, so the level that contains it names
					// it: a `Beats` field under a `Tempo` sub-parameter shows as `Tempo Beats`.
					// A level that declares nothing contributes its field name.
					let child = match &p.meta.display_name {
						Some(child) => child.clone(),
						None => p.identifier.rsplit('.').next().unwrap_or_default().into(),
					};
					// A level that names itself with an empty display contributes no word, so a
					// sub-parameter can sit flat under its product (Oscillator > Bend) instead of
					// carrying its internal field name (Oscillator Freq Gen Bend).
					let joined = if #display_prefix.is_empty() {
						child.into_owned()
					} else {
						format!("{} {}", #display_prefix, child)
					};
					p.meta.display_name = Some(joined.into());
					#inherit_layout
				});
				parameters.extend(sub_parameters);
			});
			setter_imply.push(quote! {
				#param_index => {
					let param_id = &identifier[#sub_head_len..];
					let value = std::mem::take(&mut value);
					return i_am_dsp::prelude::Parameters::set_parameter(&mut self.#member, param_id, value);
				},
			});
		} else if config.is_bool {
			parameter_imply.push(quote! {
				parameters.push(i_am_dsp::prelude::Parameter {
					identifier: #id_string.to_string(),
					value: i_am_dsp::prelude::Value::Bool(self.#member),
					meta: #meta,
				});
			});
			setter_imply.push(quote! {
				#param_index => {
					if let i_am_dsp::prelude::SetValue::Bool(value) = &value {
						self.#member = *value;
						return true;
					}
				},
			});
		}

		param_index += 1;
	}

	let expanded = quote! {
		impl #impl_generics i_am_dsp::prelude::Parameters for #struct_name #ty_generics #where_clause {
			fn get_parameters(&self) -> Vec<i_am_dsp::prelude::Parameter> {
				use i_am_dsp::prelude::Parameters;

				#[allow(dead_code)]
				let mut parameters = Vec::with_capacity(#field_count);
				#(#parameter_imply)*
				parameters
			}

			fn stable_id_overrides(&self) -> Vec<(String, u32)> {
				let _ = self;
				#[allow(unused_mut)]
				let mut overrides = Vec::new();
				#(#override_imply)*
				overrides
			}

			fn set_parameter(&mut self, identifier: &str, mut value: i_am_dsp::prelude::SetValue) -> bool {
				let identifier_head = identifier.split('.').next().unwrap_or("");

				// Inline match on the identifier head: no module-level items are generated,
				// so multiple derives can never collide on helper names.
				let index = match identifier_head {
					#(#paramters_map_create)*
					_ => return false,
				};

				match index {
					#(#setter_imply)*
					_ => {}
				}

				false
			}
		}
	};

	Ok(expanded.into())
}
