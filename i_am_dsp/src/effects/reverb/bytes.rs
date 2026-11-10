use super::*;

pub(super) fn format_feedback_matrix<const DELAY_LINES: usize, const CHANNELS: usize>(
	mat: &[FeedbackMatrix<DELAY_LINES>; CHANNELS],
) -> Vec<u8> {
	let mut result = vec![];
	for channel in mat {
		for row in channel.to_dense() {
			for elem in row {
				result.extend_from_slice(&f32::to_le_bytes(elem));
			}
		}
	}
	result
}

pub(super) fn parse_feedback_matrix<const DELAY_LINES: usize, const CHANNELS: usize>(
	data: Vec<u8>,
) -> [FeedbackMatrix<DELAY_LINES>; CHANNELS] {
	let stride = DELAY_LINES * DELAY_LINES * size_of::<f32>();
	core::array::from_fn(|channel| {
		let mut dense = [[0.0; DELAY_LINES]; DELAY_LINES];
		for (i, row) in dense.iter_mut().enumerate() {
			for (j, elem) in row.iter_mut().enumerate() {
				let at = channel * stride + (i * DELAY_LINES + j) * size_of::<f32>();
				let bits = [data[at], data[at + 1], data[at + 2], data[at + 3]];
				*elem = f32::from_le_bytes(bits);
			}
		}
		FeedbackMatrix::from_dense(dense)
	})
}

pub(super) fn format_delay_time<const DELAY_LINES: usize>(
	delay_time: &[usize; DELAY_LINES],
) -> Vec<u8> {
	let mut result = vec![];
	for delay in delay_time {
		result.extend_from_slice(&(*delay as u32).to_le_bytes());
	}
	result
}

pub(super) fn parse_delay_time<const DELAY_LINES: usize>(data: Vec<u8>) -> [usize; DELAY_LINES] {
	core::array::from_fn(|i| {
		let at = i * size_of::<u32>();
		u32::from_le_bytes([data[at], data[at + 1], data[at + 2], data[at + 3]]) as usize
	})
}

pub(super) fn format_output_factors<const DELAY_LINES: usize>(
	output_factors: &[f32; DELAY_LINES],
) -> Vec<u8> {
	let mut result = vec![];
	for factor in output_factors {
		result.extend_from_slice(&f32::to_le_bytes(*factor));
	}
	result
}

pub(super) fn parse_output_factors<const DELAY_LINES: usize>(data: Vec<u8>) -> [f32; DELAY_LINES] {
	core::array::from_fn(|i| {
		let at = i * size_of::<f32>();
		f32::from_le_bytes([data[at], data[at + 1], data[at + 2], data[at + 3]])
	})
}
