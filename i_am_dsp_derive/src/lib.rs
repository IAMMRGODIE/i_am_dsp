//! A helper proc macro that derives the `Parameters` trait for structs.
//!
//! Supports named structs, tuple (positional) structs and empty / unit structs.

#![warn(missing_docs)]

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Attribute, Data, DataStruct, DeriveInput, Error, Field, Fields, Ident, Lit, LitInt, Member, Result, Type};

#[proc_macro_derive(Parameters, attributes(
	range,
	id,
	persist,
	serde,
	skip,
	sub_param,
	logarithmic,
	default,
	unit,
	layout,
	default_float_range,
	default_int_range,
	default_uint_range
))]
/// A helper macro to derive the `Parameters` trait for a struct (named fields, tuple/positional fields, or empty).
///
/// # Syntax
///
/// ```ignore
/// #[derive(Parameters)]
/// #[default_float_range(min = 0.0, max = 1.0)]
/// #[default_int_range(min = 0, max = 10)]
/// #[default_uint_range(min = 0, max = 10)]
/// struct MyParameters {
///     #[range(min = 0.0, max = 1.0)]
///     a: f32,
///     #[range(min = 1, max = 10)]
///     b: i32,
///     #[id(name = "Foo")]
///     #[persist(serialize = "string_to_binary", deserialize = "binary_to_string")]
///     c: String,
///
///     #[sub_param]
///     d: SubParameters,
///
///     boo: bool,
///
///     #[skip]
///     _e: Vec<f32>,
/// }
///
/// #[derive(Parameters)]
/// struct SubParameters {
///     #[range(min = 0.0, max = 1.0)]
///     #[id(prefix = "sub_")]
///     f: f32,
///     #[range(min = 0, max = 1)]
///     g: i32,
///     #[logarithmic]
///     h: f32,
///     #[serde]
///     i: (f32, f32),
/// }
///
/// fn string_to_binary(s: &String) -> Vec<u8> {
///     s.as_bytes().to_vec()
/// }
///
/// fn binary_to_string(b: Vec<u8>) -> String {
///     String::from_utf8_lossy(&b).to_string()
/// }
/// ```
///
/// ## Tuple structs
///
/// Tuple structs (positional / "anonymous" fields) are supported too; fields are
/// addressed by their parameter position (`"0"`, `"1"`, ...) unless renamed with `#[id(name = "...")]`:
///
/// ```ignore
/// #[derive(Parameters)]
/// #[default_float_range(min = 0.0, max = 1.0)]
/// struct MyTuple(
///     #[range(min = -1.0, max = 1.0)] f32,
///     #[id(name = "mix")] f32,
///     #[skip] Vec<f32>,
///     #[sub_param] SubParameters,
///     bool,
/// );
/// ```
///
/// Unit structs (`struct Foo;`), empty tuple structs (`struct Foo();`) and empty
/// named structs (`struct Foo {}`) are also supported and simply expose no parameters.
///
/// ## Identifiers and indices
///
/// Every parameter has a string `identifier`. For plain fields the identifier is the
/// field name (or the parameter position for tuple fields), optionally prefixed by a
/// struct-level `#[id(prefix = "...")]` or renamed by a field-level `#[id(name = "...")]`.
/// `#[sub_param]` fields contribute multiple parameters with dotted identifiers like
/// `"sub.a"`, `"sub.b"`, ....
///
/// `Parameters::set_parameter_by_index` addresses parameters by their position in the
/// flattened list returned by `get_parameters()` (sub-parameters count as individual
/// entries). The derive never relies on those positions for its own dispatch; identifier
/// lookup is the single source of truth, so the two are always consistent.
///
/// ## Attributes
///
/// - `range`: A range attribute that specifies the minimum and maximum values of the parameter.
/// - `id`: An id attribute that specifies the name of the parameter. Can also be used on the struct itself as `#[id(prefix = "...")]` to prefix every parameter identifier. A field-level `#[id(stable = 4021)]` additionally pins the id the host knows the parameter by, so it survives the field being renamed.
/// - `persist`: A persist attribute that specifies how to serialize and deserialize the parameter.
/// - `serde`: Use serde to serialize and deserialize the parameter.
/// - `skip`: A skip attribute that specifies that the field should be skipped when deriving the `Parameters` trait.
/// - `sub_param`: A sub_param attribute that specifies that the field is a sub-parameter struct.
/// - `logarithmic`: A logarithmic attribute that specifies that the parameter should be displayed in logarithmic scale. Parameters with `logarithmic` must be positive.
/// - `default`: The value the parameter starts at, so that a host can offer "reset to default" instead of guessing the minimum: `#[default(0.8)]`, `#[default(false)]`.
///
///   **Declare it wherever there is an answer to give.** A host that is not told falls back to the bottom of the range, which is rarely where an effect starts: a synced time comes back as a thirty-second note instead of a beat, a gain at its quietest instead of at unity. The one case that should be left alone is a parameter whose starting value is an argument to the constructor — a delay time, a sample rate — because then there is no single answer to write down.
/// - `unit`: How a host should show the value as text: `#[unit(beats)]` renders `0.5` as `1/8`, and `#[unit(milliseconds)]` appends `ms`. One of `none`, `milliseconds` (or `ms`), `hertz` (or `hz`), `beats`, `seconds` (or `s`), `samples`, `decibels` (or `db`) and `percent`. A percentage is a *fraction of one*: `0.5` reads back and forth as `50.0 %`.
///
///   Declaring a unit is what keeps a host from having to guess: a value with one is shown with it, and a value that declares none stays a plain number. Not every parameter has one to declare — a gain, a ratio, a mix — and those are fine as they are.
/// - layout: Where the parameter wants to sit in a faceplate, and how it wants to be
///   drawn: which group it belongs to, where it sits inside it, whether it wants a knob or
///   a slider, how much room it wants, whether the middle of its range is its neutral
///   point, and the interval its values snap to. Every one of them is a hint, and a
///   faceplate is free to ignore all of them:
///   `#[layout(group = "Time", order = 0, widget = knob, size = large, bipolar, step = 0.25)]`.
///   The options are group, order, widget (auto, knob, slider, toggle, choice), size
///   (small, medium, large), bipolar and step.
///
/// - `id(display = "...")`: What a host should *call* the parameter, when the identifier is not what a person should read: `#[id(display = "BPM Sync")]`. It never touches the identifier, so it can sit on a `sub_param`, where it names the level those parameters sit under: a `Beats` field under a `Tempo` sub-parameter is shown as `Tempo Beats`. A level that declares nothing contributes its field name, so the worst a host sees is `tempo beats` rather than `tempo.beats`.
/// - `default_float_range`: A default range attribute that specifies the default minimum and maximum values of the parameter for float type.
/// - `default_int_range`: A default range attribute that specifies the default minimum and maximum values of the parameter for signed integer type.
/// - `default_uint_range`: A default range attribute that specifies the default minimum and maximum values of the parameter for unsigned integer type.
///
/// by default, default_float_range is (0.0, 1.0), default_int_range is (0, 256), default_uint_range is (0, 256)
pub fn derive_parameters(input: TokenStream) -> TokenStream {
	let input = parse_macro_input!(input as DeriveInput);

	match impl_config(&input) {
		Ok(tokens) => tokens,
		Err(err) => err.to_compile_error().into(),
	}
}

fn impl_config(input: &DeriveInput) -> Result<TokenStream> {
	let struct_name = &input.ident;
	let generics = &input.generics;
	let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

	let mut default_float_range = (0.0f32, 1.0f32);
	let mut default_int_range: (i64, i64) = (0, 256);
	let mut default_uint_range: (i64, i64) = (0, 256);
	let mut struct_id_prefix = String::new();

	for attr in &input.attrs {
		let Some(path) = attr.path().get_ident() else { continue; };
		let path = path.to_string();
		let path = path.trim().to_string();
		match path.as_str() {
			"default_float_range" => {
				let range = parse_range_attribute(attr)?;
				if let ParamRange::Float(min, max) = range {
					default_float_range = (min, max);
				} else {
					return Err(Error::new_spanned(attr, "default_float_range expects float bounds"));
				}
			},
			"default_int_range" => {
				let range = parse_range_attribute(attr)?;
				if let ParamRange::Int(min, max) = range {
					default_int_range = (min, max);
				} else {
					return Err(Error::new_spanned(attr, "default_int_range expects integer bounds"));
				}
			},
			"default_uint_range" => {
				let range = parse_range_attribute(attr)?;
				if let ParamRange::Int(min, max) = range {
					if min < 0 || max < 0 {
						return Err(Error::new_spanned(attr, "default_uint_range must be non-negative"));
					}
					default_uint_range = (min, max);
				} else {
					return Err(Error::new_spanned(attr, "default_uint_range expects integer bounds"));
				}
			},
			"id" => {
				let (prefix, name, stable, display) = parse_id_attribute(attr)?;
				if name.is_some() || stable.is_some() || display.is_some() {
					return Err(Error::new_spanned(attr, "Struct-level `id` attribute only supports `prefix`"));
				}
				if let Some(prefix) = prefix {
					struct_id_prefix = prefix;
				}
			},
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
		}.into());
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
				(ParamRange::Float(min, max), Some(FieldKind::Float)) => {
					if config.logarithmic && (*min <= 0.0 || *max <= 0.0) {
						return Err(Error::new_spanned(member, "Logarithmic attribute must be positive"));
					}
				}
				(ParamRange::Int(min, max), Some(kind)) if matches!(kind, FieldKind::Signed | FieldKind::Unsigned) => {
					let min = *min;
					let max = *max;
					if matches!(kind, FieldKind::Unsigned) && (min < 0 || max < 0) {
						return Err(Error::new_spanned(member, "Unsigned integer range attribute must be non-negative"));
					}
					if config.logarithmic && (min <= 0 || max <= 0) {
						return Err(Error::new_spanned(member, "Logarithmic attribute must be positive"));
					}
				}
				(ParamRange::Float(..), _) => {
					return Err(Error::new_spanned(member, "Cannot use float range attribute with non-float type"));
				}
				(ParamRange::Int(..), _) => {
					return Err(Error::new_spanned(member, "Cannot use int range attribute with non-integer type"));
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
				let layout = quote! {
					i_am_dsp::prelude::ParamLayout {
						group: #group,
						order: #order,
						widget: #widget,
						size: #size,
						bipolar: #bipolar,
						step: #step,
					}
				};

				quote! {
					i_am_dsp::prelude::ParameterMeta { default: #default, unit: #unit, display_name: #display, layout: #layout }
				}
			};

			#[cfg(not(feature = "layout"))]
			let meta = quote! {
				i_am_dsp::prelude::ParameterMeta { default: #default, unit: #unit, display_name: #display }
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
						Error::new_spanned(&member, format!("Range bound {min} does not fit in i32 (the host-facing integer parameter type)"))
					})?;
					let max_i32 = i32::try_from(max).map_err(|_| {
						Error::new_spanned(&member, format!("Range bound {max} does not fit in i32 (the host-facing integer parameter type)"))
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
		} else if let (Some(serialize), Some(deserialize)) = (config.persist_serialize, config.persist_deserialize) {
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
			let display_prefix = config.id_display.clone().unwrap_or_else(|| field_name.clone());
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
					p.meta.display_name = Some(format!("{} {}", #display_prefix, child).into());
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

/// The kind of a numeric/bool field, inferred from its type.
enum FieldKind {
	Float,
	Signed,
	Unsigned,
	Bool,
}

/// Classify a field type by its last path segment, so path-qualified forms like
/// `std::primitive::f32` (or plain `f32`) are recognised uniformly. Alias types are
/// not resolved (that would require name resolution); unknown types return `None`.
fn classify_type(ty: &Type) -> Option<FieldKind> {
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
enum ParamRange {
	Float(f32, f32),
	Int(i64, i64),
}

/// The unit a parameter declares with `#[unit(...)]`, before it becomes tokens.
#[derive(Debug, Clone, Copy)]
enum ParameterUnit {
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
struct FieldConfig {
	range: Option<ParamRange>,
	id_name: String,
	id_stable: Option<u32>,
	/// What a host should call the parameter, from `#[id(display = "...")]`.
	id_display: Option<String>,
	/// The value the parameter starts at, from `#[default(...)]`.
	param_default: Option<f64>,
	unit: Option<ParameterUnit>,
	/// Where the parameter wants to sit in a faceplate, from `#[layout(..)]`.
	#[cfg(feature = "layout")]
	layout: ParamLayoutConfig,
	persist_serialize: Option<String>,
	persist_deserialize: Option<String>,
	skip: bool,
	is_sub_param: bool,
	is_bool: bool,
	logarithmic: bool,
	is_serde: bool,
}

fn handle_attribute(
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
		let Some(path) = attr.path().get_ident() else { continue; };
		let path = path.to_string();
		let path = path.trim().to_string();
		match path.as_str() {
			"range" => {
				config.range = Some(parse_range_attribute(attr)?);
			},
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

				config.id_name = if config.id_display.is_some() && name == field_name && prefix.is_empty() && config.id_stable.is_none() { String::new() } else { format!("{}{}", prefix, name) };

				if config.id_name.contains('.') {
					return Err(Error::new_spanned(attr, "id attribute cannot contain `.`"));
				}

				if config.id_name.is_empty() && config.id_display.is_none() {
					return Err(Error::new_spanned(attr, "id attribute cannot be empty"));
				}

				if !config.id_name.is_empty() && syn::parse_str::<Ident>(&config.id_name).is_err() {
					return Err(Error::new_spanned(attr, "Name must be a valid rust ident"));
				}
			},
			"persist" => {
				let (serialize, deserialize) = parse_persist_attribute(attr)?;
				config.persist_serialize = Some(serialize);
				config.persist_deserialize = Some(deserialize);
			},
			"default" => config.param_default = Some(parse_default_attribute(attr)?),
			"unit" => config.unit = Some(parse_unit_attribute(attr)?),
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
		let should_be_none = config.range.is_none() &&
			config.persist_deserialize.is_none() &&
			config.id_name.is_empty() &&
			config.id_stable.is_none() &&
			config.param_default.is_none() &&
			config.unit.is_none() &&
			!config.is_sub_param &&
			!config.is_serde;

		if !should_be_none {
			return Err(Error::new_spanned(member, "`skip` attribute cannot be used with other attributes"));
		}
	}

	if config.persist_serialize.is_some() {
		let should_be_none = config.range.is_none() &&
			!config.skip &&
			!config.is_sub_param &&
			!config.is_serde;

		if !should_be_none {
			return Err(Error::new_spanned(member, "`persist` serialize attribute cannot be used with other attributes"));
		}
	}

	if config.range.is_some() {
		let should_be_none = config.persist_serialize.is_none() &&
			!config.skip &&
			!config.is_sub_param &&
			!config.is_serde;

		if !should_be_none {
			return Err(Error::new_spanned(member, "`range` attribute cannot be used with other parameter attributes"));
		}
	}

	if config.is_sub_param {
		let should_be_none = config.range.is_none() &&
			config.persist_serialize.is_none() &&
			config.persist_deserialize.is_none() &&
			config.id_name.is_empty() &&
			config.id_stable.is_none() &&
			config.param_default.is_none() &&
			config.unit.is_none() &&
			!config.skip &&
			!config.is_serde;

		if !should_be_none {
			return Err(Error::new_spanned(member, "`sub_param` attribute cannot be used with other attributes (including `id`)"));
		}
	}

	// Infer the parameter kind from the field type when no explicit attribute is given.
	if config.range.is_none() &&
		config.persist_serialize.is_none() &&
		!config.skip &&
		!config.is_sub_param &&
		!config.is_serde
	{
		match classify_type(field_ty) {
			Some(FieldKind::Float) => config.range = Some(ParamRange::Float(default_float_range.0, default_float_range.1)),
			Some(FieldKind::Signed) => config.range = Some(ParamRange::Int(default_int_range.0, default_int_range.1)),
			Some(FieldKind::Unsigned) => config.range = Some(ParamRange::Int(default_uint_range.0, default_uint_range.1)),
			Some(FieldKind::Bool) => config.is_bool = true,
			None => {
				return Err(Error::new_spanned(
					member,
					format!("Unsupported type for parameter field `{field_name}`, consider adding `range`, `skip`, `sub_param`, `persist` or `serde` to it"),
				));
			}
		}
	}

	Ok(config)
}

fn parse_range_attribute(attr: &Attribute) -> Result<ParamRange> {
	let mut min_float: Option<f32> = None;
	let mut min_int: Option<i64> = None;
	let mut max_float: Option<f32> = None;
	let mut max_int: Option<i64> = None;
	attr.parse_nested_meta(|meta| {
		if meta.path.is_ident("min") {
			let val: Lit = meta.value()?.parse()?;
			match val {
				Lit::Float(inner) => {
					match inner.base10_parse::<f32>() {
						Ok(t) => min_float = Some(t),
						Err(e) => return Err(Error::new_spanned(attr, format!("Invalid float, {e}"))),
					}
				},
				Lit::Int(inner) => {
					match inner.base10_parse::<i64>() {
						Ok(t) => min_int = Some(t),
						Err(e) => return Err(Error::new_spanned(attr, format!("Invalid int, {e}"))),
					}
				},
				_ => return Err(Error::new_spanned(attr, "Invalid literal for min, expected float or int")),
			}
		} else if meta.path.is_ident("max") {
			let val: Lit = meta.value()?.parse()?;
			match val {
				Lit::Float(inner) => {
					match inner.base10_parse::<f32>() {
						Ok(t) => max_float = Some(t),
						Err(e) => return Err(Error::new_spanned(attr, format!("Invalid float, {e}"))),
					}
				},
				Lit::Int(inner) => {
					match inner.base10_parse::<i64>() {
						Ok(t) => max_int = Some(t),
						Err(e) => return Err(Error::new_spanned(attr, format!("Invalid int, {e}"))),
					}
				},
				_ => return Err(Error::new_spanned(attr, "Invalid literal for max, expected float or int")),
			}
		}
		Ok(())
	})?;

	let is_float = min_float.is_some() || max_float.is_some();
	let is_int = min_int.is_some() || max_int.is_some();
	if is_float && is_int {
		return Err(Error::new_spanned(attr, "min and max must be of the same type"));
	}

	let range = if is_float {
		ParamRange::Float(min_float.or(max_float).unwrap_or(0.0), max_float.or(min_float).unwrap_or(0.0))
	} else {
		ParamRange::Int(min_int.or(max_int).unwrap_or(0), max_int.or(min_int).unwrap_or(0))
	};

	match &range {
		ParamRange::Float(min, max) if min > max => return Err(Error::new_spanned(attr, "min must be less than or equal to max")),
		ParamRange::Int(min, max) if min > max => return Err(Error::new_spanned(attr, "min must be less than or equal to max")),
		_ => {}
	}

	Ok(range)
}

/// Reads `#[default(1.0)]`: a number, or a bool for a bool parameter.
///
/// The value is kept as `f64` because that is what a host is handed.
fn parse_default_attribute(attr: &Attribute) -> Result<f64> {
	let val: Lit = attr.parse_args()?;
	match val {
		Lit::Float(inner) => inner.base10_parse::<f64>(),
		Lit::Int(inner) => inner.base10_parse::<f64>(),
		Lit::Bool(inner) => Ok(if inner.value { 1.0 } else { 0.0 }),
		other => Err(Error::new_spanned(other, "default expects a number, or a bool")),
	}
}

/// Reads `#[unit(beats)]` and friends: how a host should show the value.
fn parse_unit_attribute(attr: &Attribute) -> Result<ParameterUnit> {
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
enum LayoutWidget {
	Auto,
	Knob,
	Slider,
	Toggle,
	Choice,
}

#[cfg(feature = "layout")]
/// How much room a parameter asked for, from the size option of a layout attribute.
#[derive(Debug, Clone, Copy)]
enum LayoutSize {
	Small,
	Medium,
	Large,
}

#[cfg(feature = "layout")]
/// What a field's layout attribute asked for.
#[derive(Debug, Default)]
struct ParamLayoutConfig {
	group: Option<String>,
	order: Option<u16>,
	widget: Option<LayoutWidget>,
	size: Option<LayoutSize>,
	bipolar: bool,
	step: Option<f64>,
}

#[cfg(feature = "layout")]
/// Reads the layout attribute: where a parameter wants to sit in a faceplate.
///
/// Every option is optional, and every one of them is a hint. The attribute
/// exists so that an effect can say how it wants to be met without a faceplate
/// having to be written for it.
fn parse_layout_attribute(attr: &Attribute) -> Result<ParamLayoutConfig> {
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
				},
				_ => return Err(Error::new_spanned(attr, "Invalid literal for group, expected a string")),
			}
		} else if meta.path.is_ident("order") {
			let val: LitInt = meta.value()?.parse()?;
			let order: u32 = val.base10_parse()?;
			layout.order = Some(u16::try_from(order).map_err(|_| Error::new_spanned(attr, "An order must fit in 16 bits"))?);
		} else if meta.path.is_ident("widget") {
			let ident: Ident = meta.value()?.parse()?;
			layout.widget = Some(match ident.to_string().as_str() {
				"auto" => LayoutWidget::Auto,
				"knob" => LayoutWidget::Knob,
				"slider" => LayoutWidget::Slider,
				"toggle" => LayoutWidget::Toggle,
				"choice" => LayoutWidget::Choice,
				other => return Err(Error::new_spanned(&ident, format!("unknown widget {other}: expected auto, knob, slider, toggle or choice"))),
			});
		} else if meta.path.is_ident("size") {
			let ident: Ident = meta.value()?.parse()?;
			layout.size = Some(match ident.to_string().as_str() {
				"small" => LayoutSize::Small,
				"medium" => LayoutSize::Medium,
				"large" => LayoutSize::Large,
				other => return Err(Error::new_spanned(&ident, format!("unknown size {other}: expected small, medium or large"))),
			});
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
			return Err(Error::new_spanned(attr, "unknown layout option: expected group, order, widget, size, bipolar or step"));
		}

		Ok(())
	})?;

	Ok(layout)
}

/// What a field's `#[id(..)]` attribute asked for: the identifier prefix, the identifier
/// name, a pinned host id and the display name.
type IdAttribute = (Option<String>, Option<String>, Option<u32>, Option<String>);

fn parse_id_attribute(attr: &Attribute) -> Result<IdAttribute> {
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
				return Err(Error::new_spanned(attr, "Stable id must fit in 31 bits (a VST3 host reserves the top bit)"));
			}
			stable = Some(value);
		} else if meta.path.is_ident("prefix") {
			let val: Lit = meta.value()?.parse()?;
			match val {
				Lit::Str(inner) => prefix = Some(inner.value()),
				_ => return Err(Error::new_spanned(attr, "Invalid literal for prefix, expected string")),
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
				},
				_ => return Err(Error::new_spanned(attr, "Invalid literal for name, expected string")),
			}
		} else if meta.path.is_ident("display") {
			// What a host calls the parameter. Unlike `name` this never touches the identifier,
			// so it is free-form and may be combined with `sub_param`.
			let val: Lit = meta.value()?.parse()?;
			match val {
				Lit::Str(inner) => {
					let value = inner.value();
					if value.trim().is_empty() {
						return Err(Error::new_spanned(attr, "Display name must not be empty"));
					}
					display = Some(value);
				},
				_ => return Err(Error::new_spanned(attr, "Invalid literal for display, expected string")),
			}
		}
		Ok(())
	})?;
	Ok((prefix, name, stable, display))
}

fn parse_persist_attribute(attr: &Attribute) -> Result<(String, String)> {
	let mut serialize = None;
	let mut deserialize = None;

	attr.parse_nested_meta(|meta| {
		if meta.path.is_ident("serialize") {
			let val: Lit = meta.value()?.parse()?;
			match val {
				Lit::Str(inner) => serialize = Some(inner.value()),
				_ => return Err(Error::new_spanned(attr, "Invalid literal for serialize, expected string")),
			}
		} else if meta.path.is_ident("deserialize") {
			let val: Lit = meta.value()?.parse()?;
			match val {
				Lit::Str(inner) => deserialize = Some(inner.value()),
				_ => return Err(Error::new_spanned(attr, "Invalid literal for deserialize, expected string")),
			}
		}
		Ok(())
	})?;

	if let (Some(serialize), Some(deserialize)) = (serialize, deserialize) {
		Ok((serialize, deserialize))
	} else {
		Err(Error::new_spanned(attr, "Missing serialize or deserialize attribute"))
	}
}