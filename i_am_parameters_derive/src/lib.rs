//! A Helper macro to derive the `Parameters` trait for a struct with named fields.

#![warn(missing_docs)]

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Attribute, Data, DataStruct, DeriveInput, Error, Fields, Ident, Lit, Result, Type};

#[proc_macro_derive(Parameters, attributes(
	range, 
	id, 
	persist,
	serde,
	skip, 
	sub_param, 
	logarithmic,
	default_float_range, 
	default_int_range,
	default_uint_range
))]
/// A helper macro to derive the `Parameters` trait for a struct with named fields.
/// 
/// # Syntax
/// 
/// ```no_run
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
///     _e: Vec<f32>
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
///     h:f32,
///     #[serde]
///     i: (f32, f32),
/// }
/// 
/// fn string_to_binary(s: &String) -> Vec<u8> {
///     s.as_bytes().to_vec()
/// }
/// 
/// fn binary_to_string(b: Vec<u8>) -> String {
///     String::from_utf8_lossy(b).to_string()
/// }
/// ```
/// 
/// - `range`: A range attribute that specifies the minimum and maximum values of the parameter.
/// - `id`: An id attribute that specifies the name of the parameter.
/// - `persist`: A persist attribute that specifies how to serialize and deserialize the parameter.
/// - `serde`: Use serde to serialize and deserialize the parameter.
/// - `skip`: A skip attribute that specifies that the field should be skipped when deriving the `Parameters` trait.
/// - `sub_param`: A sub_param attribute that specifies that the field is a sub-parameter struct.
/// - `logarithmic`: A logarithmic attribute that specifies that the parameter should be displayed in logarithmic scale. Parameter with logarithmic must be positive.
/// - `default_float_range`: A default range attribute that specifies the default minimum and maximum values of the parameter for float type.
/// - `default_int_range`: A default range attribute that specifies the default minimum and maximum values of the parameter for signed integer type.
/// - `default_uint_range`: A default range attribute that specifies the default minimum and maximum values of the parameter for usigned integer type.
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
	let mut default_float_range = (0.0, 1.0);
	let mut default_int_range = (f32::from_bits(0), f32::from_bits(256));
	let mut default_uint_range = (f32::from_bits(0), f32::from_bits(256));

	for attr in &input.attrs {
		let Some(path) = attr.path().get_ident() else { continue; };
		let path = path.to_string();
		let path = path.trim().to_string();
		match path.as_str() {
			"default_float_range" => {
				let range = parse_range_attribute(attr)?;
				default_float_range = (range.0, range.1);
			},
			"default_int_range" => {
				let range = parse_range_attribute(attr)?;
				default_int_range = (range.0, range.1);
			},
			"default_uint_range" => {
				let range = parse_range_attribute(attr)?;
				default_uint_range = (range.0, range.1);
			},
			_ => continue,
		}
	}

	let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

	let fields = if let Data::Struct(DataStruct {
		fields,
		..
	}) = &input.data {
		match fields {
			Fields::Named(fields) => &fields.named,
			Fields::Unit => {
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
			},
			_ => {
				return Err(Error::new_spanned(
					input,
					"Parameters derive macro only supports structs with named fields",
				));
			}
		}
	}else {
		return Err(Error::new_spanned(
			input,
			"Parameters derive macro only supports structs with named fields",
		));
	};

	let mut field_configs = Vec::new();

	for field in fields {
		let field_name = field.ident.as_ref().unwrap();
		let field_type = &field.ty;
		
		let config = handle_attribute(
			field_name, 
			field_type, 
			&field.attrs,
			default_float_range,
			default_int_range,
			default_uint_range,
		)?;

		if let Some((min, max)) = &config.range {
			if let Type::Path(path) = field_type {
				if let Some(ident) = path.path.get_ident() {
					let ident = ident.to_string();

					if &ident == "f32" || &ident == "f64" {
						if config.is_int {
							return Err(Error::new_spanned(field, "Cant use int range attribute with float type"));
						}
						if config.logarithmic && (*min <= 0.0 || *max <= 0.0) {
							return Err(Error::new_spanned(field, "Logarithmic attribute must be positive"));
						}
					}else if &ident == "i32" || 
						&ident == "i64" || 
						&ident == "isize" ||
						&ident == "u32" || 
						&ident == "u64" ||
						&ident == "usize"
					{
						let min = min.to_bits() as i32;
						let max = max.to_bits() as i32;
						if !config.is_int {
							return Err(Error::new_spanned(field, "Cant use float range attribute with integer type"));
						}
						if ident.starts_with("u") {
							let inner = min < 0 || max < 0;
							if inner {
								return Err(Error::new_spanned(field, "Unsigned integer range attribute must be non-negative"));
							}
						}

						if config.logarithmic && (min <= 0 || max <= 0) {
							return Err(Error::new_spanned(field, "Logarithmic attribute must be positive"));
						}
					}else {
						return Err(Error::new_spanned(field, "Cant use range attribute with non-numeric type"));
					}
				}
			}
		}

		field_configs.push((config, field_name, field_type));
	}

	let mut parameter_imply = vec![];
	let mut setter_imply = vec![];
	for (config, field_name, field_type) in field_configs {
		if config.skip {
			continue;
		}

		let id = if config.id_name.is_empty() {
			format!("{}", field_name)
		}else {
			config.id_name
		};
		let id = Ident::new(&id, proc_macro2::Span::call_site());
		let logarithmic = config.logarithmic;


		if let Some((min, max)) = config.range {
			if config.is_int {
				let min = min.to_bits() as i32;
				let max = max.to_bits() as i32;
				parameter_imply.push(quote! {
					parameters.push(i_am_dsp::prelude::Parameter {
						identifier: stringify!(#id).to_string(),
						value: i_am_dsp::prelude::Value::Int {
							value: self.#field_name as i32,
							range: #min..=#max,
							logarithmic: #logarithmic,
						},
					});
				});
				setter_imply.push(quote! {
					if identifier == stringify!(#id) {
						if let i_am_dsp::prelude::SetValue::Int(value) = &value {
							let value = *value;
							self.#field_name = value as #field_type;
							return true;
						}
					}
				});
			}else {
				parameter_imply.push(quote! {
					parameters.push(i_am_dsp::prelude::Parameter {
						identifier: stringify!(#id).to_string(),
						value: i_am_dsp::prelude::Value::Float {
							value: self.#field_name as f32,
							range: #min..=#max,
							logarithmic: #logarithmic,
						},
					});
				});
				setter_imply.push(quote! {
					if identifier == stringify!(#id) {
						if let i_am_dsp::prelude::SetValue::Float(value) = &value {
							let value = *value;
							self.#field_name = value as #field_type;
							return true;
						}
					}
				});
			}
		}else if config.is_serde {
			parameter_imply.push(quote! {
				let parsed = i_am_dsp::prelude::to_binary(&self.#field_name);
				parameters.push(i_am_dsp::prelude::Parameter {
					identifier: stringify!(#id).to_string(),
					value: i_am_dsp::prelude::Value::Serialized(parsed),
				});
			});
			setter_imply.push(quote! {
				if identifier == stringify!(#id) {
					let owned_value = std::mem::take(&mut value);
					match owned_value {
						i_am_dsp::prelude::SetValue::Serialized(parsed) => {
							let value = i_am_dsp::prelude::from_binary(parsed);
							self.#field_name = value;
							return true;
						},
						other => value = other,
					}
				}
			});
		}else if let (Some(serialize), Some(deserialize)) = (config.persist_serialize, config.persist_deserialize) {
			let serialize = Ident::new(&serialize, proc_macro2::Span::call_site());
			let deserialize = Ident::new(&deserialize, proc_macro2::Span::call_site());
			parameter_imply.push(quote! {
				let parsed = #serialize(&self.#field_name);
				parameters.push(i_am_dsp::prelude::Parameter {
					identifier: stringify!(#id).to_string(),
					value: i_am_dsp::prelude::Value::Serialized(parsed),
				});
			});
			setter_imply.push(quote! {
				if identifier == stringify!(#id) {
					let owned_value = std::mem::take(&mut value);
					match owned_value {
						i_am_dsp::prelude::SetValue::Serialized(parsed) => {
							let value = #deserialize(parsed);
							self.#field_name = value;
							return true;
						},
						other => value = other,
					}
				}
			});
		}else if config.is_sub_param {
			parameter_imply.push(quote! {
				let mut sub_parameters = i_am_dsp::prelude::Parameters::get_parameters(&self.#field_name);
				sub_parameters.iter_mut().for_each(|p| {
					p.identifier = format!("{}.{}", stringify!(#field_name), p.identifier);
				});
				parameters.extend(sub_parameters);
			});
			setter_imply.push(quote! {
				if identifier.starts_with(&format!("{}.", stringify!(#field_name))) {
					let param_id = identifier.split(".").last().unwrap();
					let value = std::mem::take(&mut value);
					i_am_dsp::prelude::Parameters::set_parameter(&mut self.#field_name, param_id, value);
					// .set_parameter(&param_id, value);
					return true;
				}
			});
		}else if config.is_bool {
			parameter_imply.push(quote! {
				parameters.push(i_am_dsp::prelude::Parameter {
					identifier: stringify!(#id).to_string(),
					value: i_am_dsp::prelude::Value::Bool(self.#field_name),
				});
			});
			setter_imply.push(quote! {
				if identifier == stringify!(#id) {
					if let i_am_dsp::prelude::SetValue::Bool(value) = &value {
						self.#field_name = *value;
						return true;
					}
				}
			});
		}
	}

	let expanded = quote! {
		impl #impl_generics i_am_dsp::prelude::Parameters for #struct_name #ty_generics #where_clause {
			fn get_parameters(&self) -> Vec<i_am_dsp::prelude::Parameter> {
				use i_am_dsp::prelude::Parameters;

				let mut parameters = Vec::new();
				#(#parameter_imply)*
				parameters
			}

			fn set_parameter(&mut self, identifier: &str, mut value: i_am_dsp::prelude::SetValue) -> bool {
				#(#setter_imply)*
				return false;
			}
		}
	};

	Ok(expanded.into())
}

#[derive(Debug, Default)]
struct FieldConfig {
	range: Option<(f32, f32)>,
	is_int: bool,
	id_name: String,
	persist_serialize: Option<String>,
	persist_deserialize: Option<String>,
	skip: bool,
	is_sub_param: bool,
	is_bool: bool,
	logarithmic: bool,
	is_serde: bool,
}

fn handle_attribute(
	field_name: &Ident, 
	field_ty: &Type, 
	attrs: &[Attribute],
	default_float_range: (f32, f32),
	default_int_range: (f32, f32),
	default_uint_range: (f32, f32),
) -> Result<FieldConfig> {
	let mut config = FieldConfig::default();
	for attr in attrs {
		let Some(path) = attr.path().get_ident() else { continue; };
		let path = path.to_string();
		let path = path.trim().to_string();
		match path.as_str() {
			"range" => {
				let (min, max, is_int) = parse_range_attribute(attr)?;
				config.range = Some((min, max));
				config.is_int = is_int;
			},
			"id" => {
				let (prefix, name) = parse_id_attribute(attr)?;
				let predix = prefix.unwrap_or_default();

				let name = if let Some(name) = name {
					name
				}else {
					field_name.to_string()
				};

				config.id_name = format!("{}{}", predix, name);

				if config.id_name.contains(".") {
					return Err(Error::new_spanned(attr, "id attribute cannot contain `.`"));
				}

				if config.id_name.is_empty() {
					return Err(Error::new_spanned(attr, "id attribute cannot be empty"));
				}

				if syn::parse_str::<Ident>(&config.id_name).is_err() {
					return Err(Error::new_spanned(attr, "Name must be a vaild rust ident"));
				}
			},
			"persist" => {
				let (serialize, deserialize) = parse_persist_attribute(attr)?;
				config.persist_serialize = Some(serialize);
				config.persist_deserialize = Some(deserialize);
			},
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
			!config.is_sub_param &&
			!config.is_serde;
		
		if !should_be_none {
			return Err(Error::new_spanned(field_name, "`skip` attribute cannot be used with other attributes"));
		}
	}

	if config.persist_serialize.is_some() {
		let should_be_none = config.range.is_none() && 
			!config.skip && 
			!config.is_sub_param &&
			!config.is_serde;

		if !should_be_none {
			return Err(Error::new_spanned(field_name, "`persist` serialize attribute cannot be used with other attributes"));
		}
	}

	if config.range.is_some() {
		let should_be_none = config.persist_serialize.is_none() && 
			!config.skip && 
			!config.is_sub_param &&
			!config.is_serde;

		if !should_be_none {
			return Err(Error::new_spanned(field_name, "`range` attribute cannot be used with persist serialize attribute"));
		}
	}

	if config.is_sub_param {
		let should_be_none = config.range.is_none() && 
			config.persist_serialize.is_none() &&
			config.persist_deserialize.is_none() &&
			!config.skip &&
			!config.is_serde;

		if !should_be_none {
			return Err(Error::new_spanned(field_name, "`sub_param` attribute cannot be used with other attributes"));
		}
	}

	if config.range.is_none() && 
		config.persist_serialize.is_none() && 
		!config.skip && 
		!config.is_sub_param && 
		!config.is_serde 
	{
		if let Type::Path(path) = field_ty {
			if let Some(ident) = path.path.get_ident() {
				let ident = ident.to_string();
				if ident == "f32" || ident == "f64" {
					config.range = Some(default_float_range);
					config.is_int = false;
				}else if ident == "i32" || 
					ident == "i64" || 
					ident == "isize" 
				{
					config.range = Some(default_int_range);
					config.is_int = true;
				}else if ident == "u32" || 
					ident == "u64" ||
					ident == "usize" 
				{
					config.range = Some(default_uint_range);
					config.is_int = true;
				}else if ident == "bool" {
					config.is_bool = true;
				}else {
					return Err(Error::new_spanned(field_name, "Unsupported type, consider using `persist`, `skip`, or `sub_param`, `serde`"));
				}
			}else {
				return Err(Error::new_spanned(field_name, "Unsupported type, consider using `persist`, `skip`, or `sub_param`, `serde`"));
			}
		}else {
			return Err(Error::new_spanned(field_name, "Unsupported type, consider using `persist`, `skip`, or `sub_param`, `serde`"));
		}
	}

	Ok(config)
}

fn parse_range_attribute(attr: &Attribute) -> Result<(f32, f32, bool)> {
	let mut min = None;
	let mut min_is_int = false;
	let mut max = None;
	let mut max_is_int = false;
	attr.parse_nested_meta(|meta| {
		if meta.path.is_ident("min") {
			let value = meta.value()?;
			let val: Lit = value.parse()?;
			match val {
				Lit::Float(inner) => {
					match inner.base10_parse::<f32>() {
						Ok(t) => min = Some(t),
						Err(e) => return Err(Error::new_spanned(attr, format!("Invalid float, {}", e))),
					}
				},
				Lit::Int(inner) => {
					match inner.base10_parse::<i32>() {
						Ok(t) => min = Some(f32::from_bits(t as u32)),
						Err(e) => return Err(Error::new_spanned(attr, format!("Invalid int, {}", e))),
					}
					min_is_int = true;
				},
				_ => return Err(Error::new_spanned(attr, "Invalid literal for min, expected float or int")),
			}
		}else if meta.path.is_ident("max") {
			let value = meta.value()?;
			let val: Lit = value.parse()?;
			match val {
				Lit::Float(inner) => {
					match inner.base10_parse::<f32>() {
						Ok(t) => max = Some(t),
						Err(e) => return Err(Error::new_spanned(attr, format!("Invalid float, {}", e))),
					}
				},
				Lit::Int(inner) => {
					match inner.base10_parse::<i32>() {
						Ok(t) => max = Some(f32::from_bits(t as u32)),
						Err(e) => return Err(Error::new_spanned(attr, format!("Invalid int, {}", e))),
					}
					max_is_int = true;
				},
				_ => return Err(Error::new_spanned(attr, "Invalid literal for max, expected float or int")),
			}
		}

		Ok(())
	})?;
	if let (Some(min), Some(max)) = (min, max) {
		if min > max {
			return Err(Error::new_spanned(attr, "min must be less than or equal to max"));
		}
		if min_is_int != max_is_int {
			return Err(Error::new_spanned(attr, "min and max must be of the same type"));
		}
		Ok((min, max, min_is_int))
	}else {
		Err(Error::new_spanned(attr, "Missing min or max attribute"))
	}
}

fn parse_id_attribute(attr: &Attribute) -> Result<(Option<String>, Option<String>)> {
	let mut prefix = None;
	let mut name = None;
	attr.parse_nested_meta(|meta| {
		if meta.path.is_ident("prefix") {
			let value = meta.value()?;
			let val: Lit = value.parse()?;
			match val {
				Lit::Str(inner) => prefix = Some(inner.value()),
				_ => return Err(Error::new_spanned(attr, "Invalid literal for prefix, expected string")),
			}
		}else if meta.path.is_ident("name") {
			let value = meta.value()?;
			let val: Lit = value.parse()?;
			match val {
				Lit::Str(inner) => {
					let value = inner.value();
					let value = value.trim().to_string();
					if syn::parse_str::<Ident>(&value).is_err() {
						return Err(Error::new_spanned(attr, "Name must be a vaild rust ident"));
					}
					name = Some(inner.value());
				},
				_ => return Err(Error::new_spanned(attr, "Invalid literal for name, expected string")),
			}
		}
		Ok(())
	})?;
	Ok((prefix, name))
}

fn parse_persist_attribute(attr: &Attribute) -> Result<(String, String)> {
	let mut serialize = None;
	let mut deserialize = None;

	attr.parse_nested_meta(|meta| {
		if meta.path.is_ident("serialize") {
			let value = meta.value()?;
			let val: Lit = value.parse()?;
			match val {
				Lit::Str(inner) => serialize = Some(inner.value()),
				_ => return Err(Error::new_spanned(attr, "Invalid literal for serialize, expected string")),
			}
		}else if meta.path.is_ident("deserialize") {
			let value = meta.value()?;
			let val: Lit = value.parse()?;
			match val {
				Lit::Str(inner) => deserialize = Some(inner.value()),
				_ => return Err(Error::new_spanned(attr, "Invalid literal for deserialize, expected string")),
			}
		}
		Ok(())
	})?;

	if let (Some(serialize), Some(deserialize)) = (serialize, deserialize) {
		Ok((serialize, deserialize))
	}else {
		Err(Error::new_spanned(attr, "Missing serialize or deserialize attribute"))
	}
}