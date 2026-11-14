//! Mapping a morph control onto a pair of pages.

/// The pages a `0..1` morph sits between.
///
/// Returns `(from, to, fraction)`. An empty list is treated as a single page,
/// and a non-finite morph is treated as the first page.
pub fn page_position(morph: f32, count: usize) -> (usize, usize, f32) {
	if count <= 1 {
		return (0, 0, 0.0);
	}

	let morph = if morph.is_finite() {
		morph.clamp(0.0, 1.0)
	} else {
		0.0
	};
	let position = morph * (count - 1) as f32;
	let from = position.floor() as usize;
	let to = (from + 1).min(count - 1);

	(from, to, position - from as f32)
}
