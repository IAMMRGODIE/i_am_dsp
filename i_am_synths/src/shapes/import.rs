use super::*;

/// How an imported file is cut into frames.
#[derive(Clone, Copy, PartialEq)]
pub(super) enum SliceMode {
	/// Decide from the file: metadata, whole frames, one cycle, or pitch.
	Auto,
	/// Cut at a fixed frame size: the metadata's, or 2048.
	Fixed,
	/// Find the recording's pitch and cut at its cycle.
	Pitch,
	/// The whole file is one cycle.
	Single,
}

impl SliceMode {
	pub(super) fn parse(value: Option<&str>) -> Self {
		match value.unwrap_or("") {
			"fixed" => Self::Fixed,
			"pitch" => Self::Pitch,
			"single" => Self::Single,
			_ => Self::Auto,
		}
	}
}

/// Reads a WAV into frames: the frame size comes from the file's metadata when
/// it has any, and otherwise from the audio itself.
pub(super) fn decode_table(
	bytes: &[u8],
	name: &str,
	mode: SliceMode,
) -> Result<ImportedWave, String> {
	let mut reader =
		WavReader::new(Cursor::new(bytes)).map_err(|error| format!("read wav: {error}"))?;
	let spec = reader.spec();
	let channels = spec.channels.max(1) as usize;
	let mut interleaved: Vec<f32> = Vec::new();

	match spec.sample_format {
		SampleFormat::Float => {
			for sample in reader.samples::<f32>() {
				interleaved.push(sample.map_err(|error| format!("read wav: {error}"))?);
			}
		}
		SampleFormat::Int => {
			let scale = (1u32 << (spec.bits_per_sample.saturating_sub(1)).min(31)) as f32;

			for sample in reader.samples::<i32>() {
				interleaved
					.push(sample.map_err(|error| format!("read wav: {error}"))? as f32 / scale);
			}
		}
	}

	if interleaved.is_empty() {
		return Err("the file holds no samples".to_string());
	}

	let mono: Vec<f32> = if channels == 1 {
		interleaved
	} else {
		interleaved
			.chunks(channels)
			.map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
			.collect()
	};

	let frames = slice_table(
		&mono,
		spec.sample_rate as usize,
		serum_frame_len(bytes),
		mode,
	);

	if frames.is_empty() {
		return Err("the file holds no cycles".to_string());
	}

	// Peak normalize the whole table, so the frames keep their relative levels.
	let peak = frames
		.iter()
		.flat_map(|frame| frame.iter())
		.fold(0.0_f32, |peak, sample| peak.max(sample.abs()));
	let gain = if peak > 1e-6 { 0.99 / peak } else { 1.0 };

	Ok(ImportedWave::new(
		name,
		frames
			.into_iter()
			.map(|frame| frame.into_iter().map(|sample| sample * gain).collect())
			.collect(),
	))
}

/// Cuts the samples into frames and resamples each one to [WAVE_FRAME_LEN].
pub(super) fn slice_table(
	samples: &[f32],
	sample_rate: usize,
	metadata: Option<usize>,
	mode: SliceMode,
) -> Vec<Vec<f32>> {
	if samples.is_empty() {
		return Vec::new();
	}

	let mut raw: Vec<&[f32]> = Vec::new();

	match mode {
		SliceMode::Single => raw.push(samples),
		SliceMode::Fixed => {
			let frame_len = metadata.unwrap_or(WAVE_FRAME_LEN);

			raw = samples.chunks(frame_len).collect();
		}
		SliceMode::Pitch => {
			raw = match detect_cycle(samples, sample_rate) {
				Some(cycle) => samples.chunks(cycle).collect(),
				None => samples.chunks(WAVE_FRAME_LEN).collect(),
			}
		}
		SliceMode::Auto => {
			if let Some(frame_len) = metadata {
				raw = samples.chunks(frame_len).collect();
			}

			if raw.is_empty() {
				let single_cycle = samples.len() < WAVE_FRAME_LEN + WAVE_FRAME_LEN / 2;

				if single_cycle {
					// A file shorter than a frame and a half is one cycle: this is
					// the Adventure Kid case, which a fixed 2048 cut would ruin.
					raw.push(samples);
				} else if samples.len().is_multiple_of(WAVE_FRAME_LEN) {
					raw = samples.chunks(WAVE_FRAME_LEN).collect();
				} else if let Some(cycle) = detect_cycle(samples, sample_rate) {
					// The file is longer than a table and not a whole number of
					// frames: find its pitch and cut at the cycle, which is what
					// Serum and Vital call a pitch splice.
					raw = samples.chunks(cycle).collect();
				} else {
					raw = samples.chunks(WAVE_FRAME_LEN).collect();
				}
			}
		}
	}

	// A trailing chunk shorter than half a frame is a cut cycle, not a frame.
	let frame_hint = raw.first().map(|frame| frame.len()).unwrap_or(0);

	if raw.len() > 1
		&& raw
			.last()
			.map(|frame| frame.len() * 2 < frame_hint)
			.unwrap_or(false)
	{
		raw.pop();
	}

	let mut frames: Vec<Vec<f32>> = raw
		.into_iter()
		.map(|frame| resample(frame, WAVE_FRAME_LEN))
		.collect();

	if frames.len() > MAX_WAVE_FRAMES {
		let step = frames.len() as f32 / MAX_WAVE_FRAMES as f32;

		frames = (0..MAX_WAVE_FRAMES)
			.map(|index| frames[(index as f32 * step) as usize].clone())
			.collect();
	}

	frames
}

/// The cycle length a recording's pitch implies, or nothing when the recording
/// has no stable pitch.
///
/// A complex or wandering source (a chord, noise, a drum loop) has no single
/// cycle length; cutting it at one would make hundreds of incommensurate
/// fragments, so the fixed cut is the honest answer there. The estimate is the
/// median over a long analysis window, and the estimates have to agree: one
/// stray reading must not decide the cut.
pub(super) fn detect_cycle(samples: &[f32], sample_rate: usize) -> Option<usize> {
	// Only the head of a long file: a table source is short, and the analysis is
	// the expensive part of an import.
	let analysis = &samples[..samples.len().min(sample_rate.saturating_mul(4))];
	let mut tracker = PitchTracker::new(sample_rate, 4_096);
	let mut estimates = Vec::new();

	for sample in analysis {
		if let Some(estimate) = tracker.process_sample(*sample)
			&& estimate.clarity >= 0.5
			&& estimate.frequency.is_finite()
			&& estimate.frequency > 20.0
		{
			estimates.push(estimate.frequency);
		}
	}

	if estimates.is_empty() {
		return None;
	}

	let pitch = median(&estimates);

	// A source whose pitch moves has no one cycle length.
	if estimates.len() >= 3 {
		let deviations: Vec<f32> = estimates
			.iter()
			.map(|frequency| (frequency - pitch).abs())
			.collect();

		if median(&deviations) / pitch > 0.05 {
			return None;
		}
	}

	let cycle = (sample_rate as f32 / pitch).round() as usize;

	// A cycle of a few dozen samples resamples into a smear; the fixed cut does
	// better on anything that high.
	if cycle < 64 || cycle > samples.len() / 2 {
		return None;
	}

	Some(cycle)
}

pub(super) fn median(values: &[f32]) -> f32 {
	let mut sorted = values.to_vec();

	sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
	sorted[sorted.len() / 2]
}

/// Reads a frame at a new length with linear interpolation.
pub(super) fn resample(frame: &[f32], len: usize) -> Vec<f32> {
	let mut output = vec![0.0; len];

	if frame.is_empty() || len == 0 {
		return output;
	}

	if frame.len() == 1 {
		output.fill(frame[0]);
		return output;
	}

	for (index, value) in output.iter_mut().enumerate() {
		let position = index as f32 / len as f32 * frame.len() as f32;
		let start = position.floor() as usize;
		let blend = position - start as f32;
		let before = frame[start.min(frame.len() - 1)];
		let after = frame[(start + 1).min(frame.len() - 1)];

		*value = before + (after - before) * blend;
	}

	output
}

/// The frame size in a Serum style file: its clm chunk records the number of
/// samples in a frame. A plain WAV has none, and the audio decides instead.
pub(super) fn serum_frame_len(bytes: &[u8]) -> Option<usize> {
	if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
		return None;
	}

	let mut at = 12;

	while at + 8 <= bytes.len() {
		let id = &bytes[at..at + 4];
		let size = u32::from_le_bytes([bytes[at + 4], bytes[at + 5], bytes[at + 6], bytes[at + 7]])
			as usize;
		let data = at + 8;

		if id == b"clm " && size >= 4 && data + 4 <= bytes.len() {
			let frame_len = u32::from_le_bytes([
				bytes[data],
				bytes[data + 1],
				bytes[data + 2],
				bytes[data + 3],
			]) as usize;

			if (64..=8192).contains(&frame_len) {
				return Some(frame_len);
			}
		}

		at = data.saturating_add(size).saturating_add(size & 1);

		if at == data {
			break;
		}
	}

	None
}

/// One value out of a query string like name=a%20b&mode=pitch.
pub(super) fn query_value<'a>(query: Option<&'a str>, key: &str) -> Option<&'a str> {
	query?.split('&').find_map(|part| {
		let (name, value) = part.split_once('=')?;

		(name == key).then_some(value)
	})
}
