//! What a host pays to save one synth's state.

use std::sync::Arc;
use std::time::Instant;

use i_am_dsp::prelude::{ImportedWave, Parameters, SampleData, SetValue, to_binary};
use i_am_plugin::Plugin;
use i_am_synths::{DrummerHandle, DrummerWeb, SamplerWeb, ShapesWeb};

fn tone(seconds: f32, offset: f32) -> Vec<f32> {
	let count = (48_000.0 * seconds) as usize;

	(0..count)
		.map(|index| (index as f32 * 0.01 + offset).sin() * 0.5)
		.collect()
}

fn sample(seconds: f32, offset: f32) -> SampleData {
	SampleData {
		name: "probe.wav".to_string(),
		rate: 48_000,
		pcm: tone(seconds, offset),
		right: Some(tone(seconds, offset + 0.5)),
	}
}

fn report(name: &str, mut measure: impl FnMut() -> Vec<u8>) {
	let _ = measure();

	let runs = 10;
	let started = Instant::now();
	let mut bytes = 0;

	for _ in 0..runs {
		bytes += measure().len();
	}

	let each = started.elapsed().as_secs_f64() / runs as f64;

	println!(
		"{name:<30} {:>8.2} ms {:>8} KB",
		each * 1000.0,
		bytes / runs / 1024
	);
}

fn main() {
	// A sampler holding two seconds of stereo audio.
	let mut sampler = SamplerWeb::new();
	let binary = to_binary(&Some(sample(2.0, 0.0))).expect("the sample");
	assert!(sampler.set_parameter("oscillator.sample", SetValue::Serialized(binary)));
	report("sampler, 2 s", || {
		to_binary(&sampler.get_parameters()).expect("the state")
	});

	// Four seconds, the sort of one-shot a kit or a pad holds.
	let mut longer = SamplerWeb::new();
	let binary = to_binary(&Some(sample(4.0, 1.0))).expect("the sample");
	assert!(longer.set_parameter("oscillator.sample", SetValue::Serialized(binary)));
	report("sampler, 4 s", || {
		to_binary(&longer.get_parameters()).expect("the state")
	});

	// A drummer with a kit: six one second one-shots and a click.
	let handle = DrummerHandle::new();
	handle.restore_kit(
		Some(sample(1.0, 2.0)),
		std::array::from_fn(|index| Some(sample(1.0, index as f32))),
	);
	let drummer = DrummerWeb::with_handle(handle);
	report("drummer, kit", || {
		to_binary(&drummer.get_parameters()).expect("the state")
	});

	// A shapes oscillator with ten frames of wavetable.
	let mut shapes = ShapesWeb::new();
	let frames: Vec<Vec<f32>> = (0..10)
		.map(|frame| {
			(0..2_048)
				.map(|index| (index as f32 * 0.01 + frame as f32).sin())
				.collect()
		})
		.collect();
	let bank = ImportedWave {
		name: "probe.wav".to_string(),
		frames,
	};
	let binary = to_binary(&bank).expect("the bank");
	assert!(shapes.set_parameter("oscillator.imported", SetValue::Serialized(binary)));
	report("shapes, 10 frames", || {
		to_binary(&shapes.get_parameters()).expect("the state")
	});

	let _ = Arc::new(());
}
