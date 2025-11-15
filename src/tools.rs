//! Useful tools for DSP.

pub mod ring_buffer;
pub mod pcm_data;
pub mod audio_io_chooser;
pub mod smoother;
pub mod wsola;
pub(crate) mod interpolate;
pub(crate) mod matrix;

#[cfg(feature = "real_time_demo")]
pub(crate) mod ui_tools;

#[cfg(feature = "rhai")]
pub mod rhat_related;

pub(crate) fn format_pcm_data<const CHANNELS: usize>(data: &Option<[Vec<f32>; CHANNELS]>) -> Vec<u8> {
	let Some(data) = data else {
		return vec![];
	};
	
	assert_eq!(std::mem::size_of::<f32>(), 4);

	let mut samples = vec![];
	
	for channel in data {
		samples.extend_from_slice(channel);
	}
	
	for channel in data.iter().rev() {
		samples.push(f32::from_bits(channel.len() as u32))
	}

	let ptr = samples.as_mut_ptr();
	let len = samples.len() * std::mem::size_of::<f32>();
	let cap = samples.capacity() * std::mem::size_of::<f32>();

	std::mem::forget(samples);

	unsafe {
		Vec::from_raw_parts(ptr as *mut u8, len, cap)
	}
}

pub(crate) fn parse_pcm_data<const CHANNELS: usize>(mut data: Vec<u8>) -> Option<[Vec<f32>; CHANNELS]> {
	if data.is_empty() {
		return None;
	}

	assert_eq!(std::mem::size_of::<f32>(), 4);

	if !data.len().is_multiple_of(std::mem::size_of::<f32>()) {
		panic!("Invalid data length");
	}

	let ptr = data.as_mut_ptr() as *mut f32;
	let len = data.len() / std::mem::size_of::<f32>();
	let cap = data.capacity() / std::mem::size_of::<f32>();

	std::mem::forget(data);

	let mut samples = unsafe {
		Vec::from_raw_parts(ptr, len, cap)
	};

	// let channels = samples.pop().expect("Invalid PCM data: missing channel count").to_bits() as usize;
	let mut data_len = Vec::with_capacity(CHANNELS);

	for _ in 0..CHANNELS {
		data_len.push(samples.pop().expect("Invalid PCM data: missing channel length").to_bits() as usize);
	}

	let outputs: [Vec<f32>; CHANNELS] = std::array::from_fn(|i| {
		let mut channel = samples.split_off(data_len[i]);
		std::mem::swap(&mut channel, &mut samples);
		channel
	});

	Some(outputs)
}

pub(crate) fn format_vec_f32(data: &[f32]) -> Vec<u8> {
	assert_eq!(std::mem::size_of::<f32>(), 4);

	let mut samples = vec![];
	
	samples.extend_from_slice(data);

	let ptr = samples.as_mut_ptr();
	let len = samples.len() * std::mem::size_of::<f32>();
	let cap = samples.capacity() * std::mem::size_of::<f32>();

	std::mem::forget(samples);

	unsafe {
		Vec::from_raw_parts(ptr as *mut u8, len, cap)
	}
}

pub(crate) fn parse_vec_f32(mut data: Vec<u8>) -> Vec<f32> {
	assert_eq!(std::mem::size_of::<f32>(), 4);

	if !data.len().is_multiple_of(std::mem::size_of::<f32>()) {
		panic!("Invalid data length");
	}

	let ptr = data.as_mut_ptr() as *mut f32;
	let len = data.len() / std::mem::size_of::<f32>();
	let cap = data.capacity() / std::mem::size_of::<f32>();

	std::mem::forget(data);

	unsafe {
		Vec::from_raw_parts(ptr, len, cap)
	}
}