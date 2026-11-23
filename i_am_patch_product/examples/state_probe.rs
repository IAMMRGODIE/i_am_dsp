//! What a host pays to save one product's state.

use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

use i_am_dsp::prelude::{ImportedTable, Parameters, SampleData, SetValue, to_binary};
use i_am_effects::{ConvolverWeb, IrData};
use i_am_patch::state::SlotState;
use i_am_patch_product::patcher::PatcherEffect;
use i_am_plugin::Plugin;

const TEXT: &str = "node 0 input:midi 0 0\nnode 1 synth:additive 100 0\nnode 2 out:audio 200 0\nconnect 0.midi -> 1.notes\nconnect 1.audio -> 2.audio\n";

fn tone(seconds: usize) -> (Vec<f32>, Vec<f32>) {
	let left = (0..48_000 * seconds)
		.map(|index| (index as f32 * 0.01).sin() * 0.5)
		.collect();
	let right = (0..48_000 * seconds)
		.map(|index| (index as f32 * 0.011).cos() * 0.5)
		.collect();

	(left, right)
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

fn time_ms(name: &str, mut work: impl FnMut()) {
	work();

	let runs = 10;
	let started = Instant::now();

	for _ in 0..runs {
		work();
	}

	println!(
		"{name:<30} {:>8.2} ms",
		started.elapsed().as_secs_f64() * 1000.0 / runs as f64
	);
}

fn slots_with(seconds: usize) -> Vec<u8> {
	let mut state = SlotState::default();
	let (left, right) = tone(seconds);

	state.samples.insert(
		1,
		i_am_patch::state::SampleState {
			sample: Some(SampleData {
				name: "probe.wav".to_string(),
				rate: 48_000,
				pcm: left.clone(),
				right: Some(right.clone()),
			}),
			slices: Vec::new(),
		},
	);
	state.irs.insert(
		2,
		i_am_patch::state::IrState {
			ir: IrData::from_samples("probe.wav".to_string(), 48_000, left, right),
			fades: Default::default(),
		},
	);
	state.fitted.insert(
		3,
		ImportedTable {
			name: "probe".to_string(),
			bytes: vec![0; 120_000],
		},
	);

	to_binary(&state).expect("the slots")
}

fn main() {
	let mut plain = PatcherEffect::default();
	plain.set_parameter("patch", SetValue::Serialized(TEXT.as_bytes().to_vec()));
	report("patch, no payload", || {
		to_binary(&plain.get_parameters()).expect("the state")
	});

	for seconds in [2, 30] {
		let mut heavy = PatcherEffect::default();
		heavy.set_parameter("patch", SetValue::Serialized(TEXT.as_bytes().to_vec()));
		heavy.set_parameter("patch_slots", SetValue::Serialized(slots_with(seconds)));
		report(&format!("patch, {seconds} s audio"), || {
			let parameters = heavy.get_parameters();
			to_binary(&parameters).expect("the state")
		});
	}

	// Where the time goes: the capture clones every payload, the encode writes it.
	let slots = i_am_patch::Slots::default();
	{
		let (left, right) = tone(2);
		let sample = slots.sample(1);
		let mut sample = sample.lock().expect("the sample slot");
		sample.sample = Some(Arc::new(SampleData {
			name: "probe.wav".to_string(),
			rate: 48_000,
			pcm: left.clone(),
			right: Some(right.clone()),
		}));
		slots.ir(2).set(Arc::new(IrData::from_samples(
			"probe.wav".to_string(),
			48_000,
			left,
			right,
		)));
	}

	let state = slots.capture();

	time_ms("slots.capture()", || {
		black_box(slots.capture());
	});
	time_ms("encode SlotState", || {
		black_box(to_binary(&state).expect("x"));
	});

	let convolver = ConvolverWeb::new();
	let (left, right) = tone(2);
	convolver.ir().set(Arc::new(IrData::from_samples(
		"probe.wav".to_string(),
		48_000,
		left,
		right,
	)));
	report("convolver, never played", || {
		to_binary(&convolver.get_parameters()).expect("the state")
	});
}
