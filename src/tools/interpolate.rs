pub(crate) fn cubic_interpolate(
	t: f32,
	values: [f32; 4]
) -> f32 {
	let c0 = values[1];
	let c1 = 0.5 * (values[2] - values[0]);
	let c2 = values[0] - 2.5 * values[1] + 2.0 * values[2] - 0.5 * values[3];
	let c3 = 0.5 * (values[3] - values[0]) + 1.5 * (values[1] - values[2]);

	c0 + t * (c1 + t * (c2 + t * c3))
}