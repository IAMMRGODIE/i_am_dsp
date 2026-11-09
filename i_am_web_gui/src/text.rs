/// The host window a faceplate asks for.
///
/// The page draws the faceplate at 85% of the window, so the window asks for the
/// design size over that: the first open shows the same faceplate as the design,
/// with a frame around it. The page is a flex layout, not a picture of one, so
/// any size it is given is one it can lay itself out in, and both axes resize.
pub fn window_options(size: (f32, f32), title: &str) -> i_am_plugin::WindowOptions {
	/// The fraction of the window the page draws the faceplate at.
	const FACE_FRACTION: f32 = 0.85;

	i_am_plugin::WindowOptions::new()
		.with_size((
			size.0 as f64 / FACE_FRACTION as f64,
			size.1 as f64 / FACE_FRACTION as f64,
		))
		.with_title(title)
		.with_resize_hints(i_am_plugin::GuiResizeHints {
			can_resize_horizontally: true,
			can_resize_vertically: true,
			strategy: i_am_plugin::AspectRatioStrategy::Disregard,
		})
}

/// Escapes a string so it can sit inside a JSON string literal.
///
/// The pages build their JSON by hand -- a route answers a string, not a
/// serializer -- so every error message and file name that reaches one has to
/// come through here first.
pub fn json_escape(value: &str) -> String {
	let mut out = String::with_capacity(value.len());

	for character in value.chars() {
		match character {
			'"' => out.push_str("\\\""),
			'\\' => out.push_str("\\\\"),
			'\n' => out.push_str("\\n"),
			'\r' => out.push_str("\\r"),
			'\t' => out.push_str("\\t"),
			character if (character as u32) < 0x20 => {
				let _ = std::fmt::Write::write_fmt(
					&mut out,
					format_args!("\\u{:04x}", character as u32),
				);
			}
			character => out.push(character),
		}
	}

	out
}

/// Decodes the `%XX` escapes of a URL-encoded value.
///
/// A data URL and a query value both arrive encoded; this is what turns
/// `Vintage%20Plate.wav` back into a name a person can read.
pub fn percent_decode(value: &str) -> String {
	let bytes = value.as_bytes();
	let mut out = Vec::with_capacity(bytes.len());
	let mut index = 0;

	while index < bytes.len() {
		if bytes[index] == b'%' && index + 2 < bytes.len() {
			// Decode from the bytes rather than a string slice: an unencoded
			// multibyte character could put a slice boundary inside a character.
			let high = (bytes[index + 1] as char).to_digit(16);
			let low = (bytes[index + 2] as char).to_digit(16);

			if let (Some(high), Some(low)) = (high, low) {
				out.push((high * 16 + low) as u8);
				index += 3;
				continue;
			}
		}

		out.push(bytes[index]);
		index += 1;
	}

	String::from_utf8_lossy(&out).to_string()
}
