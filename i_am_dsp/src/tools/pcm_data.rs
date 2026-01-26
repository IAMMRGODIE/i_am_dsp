//! A helper function to load PCM data from a WAV file.

use std::path::Path;

use symphonia::core::{audio::AudioBuffer, codecs::CODEC_TYPE_NULL, io::MediaSourceStream, probe::Hint};

#[derive(Debug, thiserror::Error)]
/// Error type for the `load_from_file` function.
pub enum ReadFileError {
	/// IO error.
	#[error("IO error: {0}")]
	Io(#[from] std::io::Error),
	/// Format error.
	#[error("Format error: {0}")]
	Format(#[from] symphonia::core::errors::Error),
	#[error("No audio track found in file")]
	/// No audio track found in file.
	NoAudioTrack,
	/// Missing sample rate.
	#[error("Missing sample rate")]
	MissingSampleRate,
	/// Resampler construction error.
	#[error("Resampler construction error: {0}")]
	ResamplerConstructionError(#[from] rubato::ResamplerConstructionError),
	/// Resample error.
	#[error("Resample error: {0}")]
	ResampleError(#[from] rubato::ResampleError),
}

/// Output of the `load_from_file` function.
#[non_exhaustive]
pub struct PcmOutput<const CHANNELS: usize> {
	/// Sample rate of the audio data.
	pub sample_rate: usize,
	/// PCM data for each channel.
	pub pcm_data: [Vec<f32>; CHANNELS],
}

/// Load PCM data from a WAV file.
pub fn load_from_file<const CHANNELS: usize>(path: impl AsRef<Path>) -> Result<PcmOutput<CHANNELS>, ReadFileError> {
	let path = path.as_ref();
	if path.extension().map(|ext| ext != "wav").unwrap_or(true) {
		return Err(ReadFileError::Io(std::io::Error::new(std::io::ErrorKind::InvalidInput, "Only Wav files are supported for now")));
	}

	let file = std::fs::File::open(path)?;
	let stream = MediaSourceStream::new(Box::new(file), Default::default());
	let mut binding = Hint::new();
	let hint = binding.with_extension("wav");
	let probed = symphonia::default::get_probe().format(
		hint,
		stream,
		&Default::default(),
		&Default::default(),
	)?;

	let mut format = probed.format;
	let codec_params = &format.tracks()
		.iter()
		.find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
		.ok_or(ReadFileError::NoAudioTrack)?
		.codec_params;

	let audio_sample_rate = codec_params.sample_rate.ok_or(ReadFileError::MissingSampleRate)? as usize;
	
	let mut decoder = symphonia::default::get_codecs()
		.make(codec_params, &Default::default())?;

	let mut pcm_data = [const { Vec::new() }; CHANNELS];

	loop {
		let packet = match format.next_packet() {
			Ok(packet) => packet,
			Err(symphonia::core::errors::Error::IoError(_)) => break,
			Err(e) => return Err(ReadFileError::Format(e)),
		};
		let buf = decoder.decode(&packet)?;
		let mut float_buffer = AudioBuffer::<f32>::new(
			buf.capacity() as u64, 
			*buf.spec()
		);
		buf.convert(&mut float_buffer);
		for (id, plane) in float_buffer.planes().planes().iter().enumerate() {
			if id >= CHANNELS {
				break
			}
			pcm_data[id].extend_from_slice(plane);
		}
	}

	Ok(PcmOutput { 
		sample_rate: audio_sample_rate, 
		pcm_data 
	})
}

/// Save PCM data to a WAV file.
/// 
/// # Panics
/// 
/// Panics if pcm_data is empty.
pub fn save_pcm_data(
	path: impl AsRef<Path>, 
	pcm_data: &[Vec<f32>], 
	sample_rate: usize
) -> Result<(), hound::Error> {
	assert!(!pcm_data.is_empty(), "PCM data must not be empty");
	let channels = pcm_data.len();

	let spec = hound::WavSpec {
		channels: channels as u16,
		sample_rate: sample_rate as u32,
		bits_per_sample: 32,
		sample_format: hound::SampleFormat::Float,
	};

	let max_channel_len = pcm_data.iter().map(|channel| channel.len()).max().unwrap_or(0);

	let mut writer = hound::WavWriter::create(path, spec)?;

	for i in 0..max_channel_len {
		for frame in pcm_data.iter() {
			if i < frame.len() {
				writer.write_sample(frame[i])?;
			}else {
				writer.write_sample(0.0)?;
			}
		}
	}
	
	writer.finalize()?;

	Ok(())
}