use crate::wav;

use i_am_dsp::prelude::{Parameters, SetValue};

use super::*;
use i_am_dsp::prelude::{Value, to_binary};

/// A 16 bit mono WAV of the given samples, for the import tests.
/// A file of whole frames keeps them: 4096 samples are two frames of 2048.
#[test]
fn a_table_of_frames_keeps_them() {
	let mut samples = Vec::new();

	for index in 0..4096 {
		samples.push((index as f32 / 2048.0 * std::f32::consts::TAU).sin());
	}

	let table = decode_table(&wav(&samples, 44_100), "two", SliceMode::Auto).expect("a table");

	assert_eq!(table.frames(), 2);
	assert!(
		table
			.frames
			.iter()
			.all(|frame| frame.len() == WAVE_FRAME_LEN)
	);
}

/// A short file is one cycle, not a broken frame.
#[test]
fn a_short_file_is_one_cycle() {
	let samples: Vec<f32> = (0..1000)
		.map(|index| (index as f32 / 1000.0 * std::f32::consts::TAU).sin())
		.collect();
	let table = decode_table(&wav(&samples, 44_100), "cycle", SliceMode::Auto).expect("a table");

	assert_eq!(table.frames(), 1);
	assert_eq!(table.frames[0].len(), WAVE_FRAME_LEN);
}

/// A recording that is not a whole number of frames is cut at its pitch, the
/// pitch splice Serum and Vital offer.
#[test]
fn a_recording_is_cut_at_its_pitch() {
	let samples: Vec<f32> = (0..12_000)
		.map(|index| (index as f32 / 100.0 * std::f32::consts::TAU).sin())
		.collect();
	let frames = slice_table(&samples, 44_100, None, SliceMode::Auto);

	assert!(frames.len() > 20, "{} frames", frames.len());
	assert!(frames.iter().all(|frame| frame.len() == WAVE_FRAME_LEN));
}

/// A source whose pitch moves has no one cycle: the import falls back to the
/// fixed cut instead of shattering it into stray fragments.
#[test]
fn an_unsteady_recording_falls_back_to_fixed_frames() {
	let rate = 44_100;
	let samples: Vec<f32> = (0..rate / 2)
		.map(|index| {
			let time = index as f32 / rate as f32;
			let phase = 2.0 * std::f32::consts::PI * (200.0 * time + 600.0 * time * time);

			phase.sin()
		})
		.collect();

	assert_eq!(detect_cycle(&samples, rate), None);

	let frames = slice_table(&samples, rate, None, SliceMode::Pitch);

	assert!(!frames.is_empty());
	assert!(frames.iter().all(|frame| frame.len() == WAVE_FRAME_LEN));
}

/// The import can be told how to cut instead of guessing.
#[test]
fn the_import_modes_cut_differently() {
	let samples: Vec<f32> = (0..5000)
		.map(|index| ((index as f32 * 12.9898).sin() * 43_758.55).fract())
		.collect();

	let fixed = slice_table(&samples, 44_100, None, SliceMode::Fixed);
	assert_eq!(fixed.len(), 2);

	let single = slice_table(&samples, 44_100, None, SliceMode::Single);
	assert_eq!(single.len(), 1);
}

/// The clm chunk Serum writes says how long a frame is.
#[test]
fn the_clm_chunk_sets_the_frame_size() {
	let mut bytes = Vec::new();
	bytes.extend_from_slice(b"RIFF");
	bytes.extend_from_slice(&0u32.to_le_bytes());
	bytes.extend_from_slice(b"WAVE");
	bytes.extend_from_slice(b"clm ");
	bytes.extend_from_slice(&4u32.to_le_bytes());
	bytes.extend_from_slice(&1024u32.to_le_bytes());

	assert_eq!(serum_frame_len(&bytes), Some(1024));
}

/// The table can be flipped: the route rebuilds it and describes the new one.
#[test]
fn the_route_flips_a_table() {
	let slot = Arc::new(Mutex::new(WaveSlot::default()));
	let preview = Arc::new(Mutex::new(WavePreview::default()));
	let samples: Vec<f32> = (0..WAVE_FRAME_LEN)
		.map(|index| index as f32 / WAVE_FRAME_LEN as f32 - 0.5)
		.collect();
	let body = wav(&samples, 44_100);

	let post = ProtocolRequest {
		method: "POST",
		path: "/waves",
		query: Some("name=ramp"),
		body: &body,
	};
	waves_route(&post, &slot, &preview).expect("the import");

	let first = preview.lock().unwrap().previews[0][0];
	assert!(first < -0.4, "{first}");

	let reverse = ProtocolRequest {
		method: "POST",
		path: "/waves",
		query: Some("action=reverse"),
		body: &[],
	};
	let response = waves_route(&reverse, &slot, &preview).expect("the reverse");

	assert!(
		String::from_utf8(response.body)
			.unwrap()
			.contains("\"version\""),
		"the route did not describe the flipped table"
	);

	let reversed = preview.lock().unwrap().previews[0][0];
	assert!(reversed > 0.4, "{reversed}");

	let invert = ProtocolRequest {
		method: "POST",
		path: "/waves",
		query: Some("action=invert"),
		body: &[],
	};
	waves_route(&invert, &slot, &preview).expect("the invert");

	let inverted = preview.lock().unwrap().previews[0][0];
	assert!(inverted < -0.4, "{inverted}");
}

/// DC and normalize even the frames out: one click zeroes each mean, the other
/// brings every frame to the same peak.
#[test]
fn the_route_removes_dc_and_normalizes_frames() {
	let slot = Arc::new(Mutex::new(WaveSlot::default()));
	let preview = Arc::new(Mutex::new(WavePreview::default()));
	let mut samples = Vec::new();

	for frame in 0..2 {
		let (offset, level) = if frame == 0 { (0.5, 0.4) } else { (0.2, 0.1) };

		for index in 0..WAVE_FRAME_LEN {
			samples.push(
				offset
					+ level * (index as f32 / WAVE_FRAME_LEN as f32 * std::f32::consts::TAU).sin(),
			);
		}
	}

	let body = wav(&samples, 44_100);
	let post = ProtocolRequest {
		method: "POST",
		path: "/waves",
		query: Some("name=dc"),
		body: &body,
	};
	waves_route(&post, &slot, &preview).expect("the import");

	let mean = |frame: &[f32]| frame.iter().sum::<f32>() / frame.len() as f32;
	let frames = slot
		.lock()
		.unwrap()
		.bank
		.as_ref()
		.unwrap()
		.frames()
		.to_vec();

	assert!(mean(&frames[0]).abs() > 0.2);

	let dc = ProtocolRequest {
		method: "POST",
		path: "/waves",
		query: Some("action=dc"),
		body: &[],
	};
	waves_route(&dc, &slot, &preview).expect("the dc removal");

	let frames = slot
		.lock()
		.unwrap()
		.bank
		.as_ref()
		.unwrap()
		.frames()
		.to_vec();

	assert!(frames.iter().all(|frame| mean(frame).abs() < 1e-4));

	let normalize = ProtocolRequest {
		method: "POST",
		path: "/waves",
		query: Some("action=normalize"),
		body: &[],
	};
	waves_route(&normalize, &slot, &preview).expect("the normalize");

	let frames = slot
		.lock()
		.unwrap()
		.bank
		.as_ref()
		.unwrap()
		.frames()
		.to_vec();

	for frame in &frames {
		let peak = frame
			.iter()
			.fold(0.0_f32, |peak, value| peak.max(value.abs()));

		assert!((peak - 0.99).abs() < 2e-3, "{peak}");
	}
}

/// The route takes the slicing mode from the query string.
#[test]
fn the_route_takes_the_slice_mode() {
	let slot = Arc::new(Mutex::new(WaveSlot::default()));
	let preview = Arc::new(Mutex::new(WavePreview::default()));
	let samples: Vec<f32> = (0..5000)
		.map(|index| ((index as f32 * 12.9898).sin() * 43_758.55).fract())
		.collect();
	let body = wav(&samples, 44_100);
	let post = ProtocolRequest {
		method: "POST",
		path: "/waves",
		query: Some("name=odd&mode=fixed"),
		body: &body,
	};
	waves_route(&post, &slot, &preview).expect("the import");

	assert_eq!(preview.lock().unwrap().count, 2);
	assert_eq!(preview.lock().unwrap().name.as_deref(), Some("odd"));
}

/// Deleting a frame shortens the table; the last frame cannot be deleted.
#[test]
fn the_route_deletes_a_frame() {
	let slot = Arc::new(Mutex::new(WaveSlot::default()));
	let preview = Arc::new(Mutex::new(WavePreview::default()));
	let samples: Vec<f32> = (0..4096)
		.map(|index| (index as f32 / 2048.0 * std::f32::consts::TAU).sin())
		.collect();
	let body = wav(&samples, 44_100);
	let post = ProtocolRequest {
		method: "POST",
		path: "/waves",
		query: Some("name=two"),
		body: &body,
	};
	waves_route(&post, &slot, &preview).expect("the import");
	assert_eq!(preview.lock().unwrap().count, 2);

	let delete = ProtocolRequest {
		method: "POST",
		path: "/waves",
		query: Some("action=delete&frame=0"),
		body: &[],
	};
	waves_route(&delete, &slot, &preview).expect("the delete");
	assert_eq!(preview.lock().unwrap().count, 1);

	// The last frame stays: there is nothing to morph across without it.
	let response = waves_route(&delete, &slot, &preview).expect("the second delete");
	let text = String::from_utf8(response.body).expect("utf-8");

	assert!(text.contains("the last frame stays"), "{text}");
	assert_eq!(preview.lock().unwrap().count, 1);
}

/// The imported table travels as the voice's serialized parameter, so it
/// survives a save and a load.
#[test]
fn an_imported_table_is_saved_with_the_state() {
	let mut plugin = ShapesWeb::new();
	let wave = ImportedWave::new("saved", vec![vec![0.2; WAVE_FRAME_LEN]]);
	let binary = to_binary(&Some(wave)).expect("a table serializes");

	assert!(plugin.set_parameter("oscillator.imported", SetValue::Serialized(binary.clone())));

	let parameter = plugin
		.get_parameters()
		.into_iter()
		.find(|parameter| parameter.identifier == "oscillator.imported")
		.expect("the serialized table");
	let Value::Serialized(data) = parameter.value else {
		panic!("the table is not serialized")
	};

	assert_eq!(data, binary);
	assert_eq!(
		plugin.imported().map(|wave| wave.name),
		Some("saved".into())
	);
}

/// The route stores what it decodes and describes it back, which is what the
/// page draws and what a save picks up.
#[test]
fn the_route_imports_and_describes_a_table() {
	let slot = Arc::new(Mutex::new(WaveSlot::default()));
	let preview = Arc::new(Mutex::new(WavePreview::default()));
	let samples: Vec<f32> = (0..4096)
		.map(|index| (index as f32 / 2048.0 * std::f32::consts::TAU).sin())
		.collect();
	let body = wav(&samples, 44_100);
	let post = ProtocolRequest {
		method: "POST",
		path: "/waves",
		query: Some("name=two"),
		body: &body,
	};

	let response = waves_route(&post, &slot, &preview).expect("the route");
	let text = String::from_utf8(response.body).expect("utf-8");

	assert!(text.contains("\"name\":\"two\""), "{text}");
	assert!(text.contains("\"count\":2"), "{text}");
	assert!(text.contains("\"previews\":[["), "{text}");
	assert!(text.contains("\"version\":"), "{text}");
	assert_eq!(
		slot.lock().unwrap().bank.as_ref().map(|bank| bank.len()),
		Some(2)
	);

	// The page asks again with the version it holds, and gets a short answer.
	let version = slot
		.lock()
		.unwrap()
		.bank
		.as_ref()
		.map(|bank| bank.id())
		.unwrap();
	let get = ProtocolRequest {
		method: "GET",
		path: "/waves",
		query: Some("since="),
		body: &[],
	};

	assert!(waves_route(&get, &slot, &preview).is_some());

	let query = format!("since={version}");
	let get = ProtocolRequest {
		method: "GET",
		path: "/waves",
		query: Some(&query),
		body: &[],
	};
	let response = waves_route(&get, &slot, &preview).expect("the route");
	let text = String::from_utf8(response.body).expect("utf-8");

	assert!(text.contains("\"unchanged\":true"), "{text}");
}
