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

/// The value a fraction of the way from one page's value to another's.
///
/// A morph is only ever this: the two pages it sits between, and how far across
/// it is. Keeping the arithmetic here means the smoothed wavetables and the
/// dense tables a spectral effect blends cannot disagree about what half way
/// between two pages sounds like.
pub fn blend(from: f32, to: f32, fraction: f32) -> f32 {
	from + (to - from) * fraction
}

/// The shape of a segment between two points: `unit` 0..1 reshaped by `factor`.
///
/// A factor of zero is the straight line, positive pulls the curve below the
/// diagonal and negative above it. The editor draws this exact function, so a
/// bend a page shows is the bend the audio side reads.
pub fn segment_bend(unit: f32, factor: f32) -> f32 {
	if factor.abs() < 1e-4 {
		return unit;
	}

	((unit * factor).exp() - 1.0) / (factor.exp() - 1.0)
}

/// Blends one dense page into another, entry by entry.
///
/// `from` and `to` are two pages sampled onto the same grid, which is what a
/// caller that has already densified both pages has. The result goes into
/// `out`, so the audio thread gets one multiply-add per entry and no
/// allocation; the shortest of the three slices decides how far it goes.
pub fn blend_slice(out: &mut [f32], from: &[f32], to: &[f32], fraction: f32) {
	for ((value, from), to) in out.iter_mut().zip(from).zip(to) {
		*value = blend(*from, *to, fraction);
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// One page is always its own neighbour, whatever the morph says.
	#[test]
	fn a_single_page_is_its_own_neighbour() {
		assert_eq!(page_position(0.7, 1), (0, 0, 0.0));
		assert_eq!(page_position(0.7, 0), (0, 0, 0.0));
	}

	/// Three pages put a quarter-way morph half way between the first two.
	#[test]
	fn a_morph_lands_between_the_pages_it_names() {
		assert_eq!(page_position(0.25, 3), (0, 1, 0.5));
		assert_eq!(page_position(0.5, 3), (1, 2, 0.0));
		assert_eq!(page_position(1.0, 3), (2, 2, 0.0));
	}

	/// A morph outside the unit range, or a non-finite one, still names real pages.
	#[test]
	fn a_morph_outside_the_range_is_clamped() {
		assert_eq!(page_position(-4.0, 4), (0, 1, 0.0));
		assert_eq!(page_position(9.0, 4), (3, 3, 0.0));
		assert_eq!(page_position(f32::NAN, 4), (0, 1, 0.0));
	}

	/// A bend pins both ends and moves the middle: zero is the straight line.
	#[test]
	fn a_bend_pins_the_ends_and_moves_the_middle() {
		assert!(segment_bend(0.0, 3.0).abs() < 1e-6);
		assert!((segment_bend(1.0, 3.0) - 1.0).abs() < 1e-6);
		assert!((segment_bend(0.5, 0.0) - 0.5).abs() < 1e-6);
		assert!(segment_bend(0.5, 3.0) < 0.5);
		assert!(segment_bend(0.5, -3.0) > 0.5);
	}

	/// A blend of zero is the from page and one is the to page, bit for bit.
	#[test]
	fn the_ends_of_a_blend_are_the_pages_themselves() {
		assert_eq!(blend(0.25, 0.75, 0.0), 0.25);
		assert_eq!(blend(0.25, 0.75, 1.0), 0.75);
	}

	/// A dense table blends entry by entry, which is the morph a page bank ends up doing.
	#[test]
	fn a_blend_slice_mixes_the_two_pages_entry_by_entry() {
		let from = [0.0, 1.0, -1.0];
		let to = [1.0, 0.5, 1.0];
		let mut out = [0.0; 3];

		blend_slice(&mut out, &from, &to, 0.5);

		assert_eq!(out, [0.5, 0.75, 0.0]);
	}
}
