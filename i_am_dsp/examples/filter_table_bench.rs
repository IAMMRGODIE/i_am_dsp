//! Measures the Filter Table direct FIR on this machine.
//!
//! ```text
//! cargo run --release -p i_am_dsp --example filter_table_bench
//! ```

use std::hint::black_box;
use std::time::Instant;

use i_am_dsp::prelude::{FilterFrame, FilterTableEffect};
use i_am_dsp::{Effect, NoteEvent, ProcessContext, ProcessInfos};

struct Context(ProcessInfos);

impl ProcessContext for Context {
	fn infos(&self) -> &ProcessInfos {
		&self.0
	}

	fn next_event(&mut self) -> Option<NoteEvent> {
		None
	}

	fn send_event(&mut self, _: NoteEvent) {}

	fn events(&self) -> &[NoteEvent] {
		&[]
	}

	fn clear_events(&mut self) {}
}

fn main() {
	const RATE: usize = 48_000;
	const SAMPLES: usize = 1_000_000;
	const TAPS: usize = 1024;

	let frame = FilterFrame::new();
	let mut effect = FilterTableEffect::with_frame(RATE, frame);
	effect.mix = 1.0;
	let mut infos = ProcessInfos::default();
	infos.sample_rate = RATE;
	let mut context: Box<dyn ProcessContext> = Box::new(Context(infos));
	let mut phase = 0.0f32;
	let mut block = [0.0f32; 2];
	black_box(block);

	// Warm up the kernel and the branch predictor before the clock starts.
	for _ in 0..RATE {
		phase += 440.0 / RATE as f32;
		block = [(phase * std::f32::consts::TAU).sin(), 0.0];
		effect.process(&mut block, &[], &mut context);
	}

	let start = Instant::now();
	for _ in 0..SAMPLES {
		phase += 440.0 / RATE as f32;
		if phase >= 1.0 {
			phase -= 1.0;
		}
		block = [(phase * std::f32::consts::TAU).sin(), 0.0];
		effect.process(&mut block, &[], &mut context);
		black_box(block);
	}
	let elapsed = start.elapsed();
	let samples = SAMPLES as f64;
	let seconds = elapsed.as_secs_f64();
	let ns = seconds / samples * 1e9;
	let realtime = seconds / (samples / RATE as f64);

	println!("Filter Table direct FIR");
	println!("  taps:          {TAPS} (SIMD f32x4)");
	println!("  rate:          {RATE} Hz stereo");
	println!("  samples:       {SAMPLES}");
	println!("  elapsed:       {seconds:.3} s");
	println!("  per sample:    {ns:.2} ns");
	println!("  realtime:      {:.2}x", 1.0 / realtime);
	println!("  one voice CPU: {:.1}%", realtime * 100.0);

	// A cutoff change rebuilds the kernel once; the next process picks it up.
	let rebuilds = 1_000;
	let start = Instant::now();
	for index in 0..rebuilds {
		effect.cutoff = 100.0 + (index % 100) as f32 * 100.0;
		effect.process(&mut block, &[], &mut context);
	}
	let rebuild_seconds = start.elapsed().as_secs_f64();
	println!(
		"  kernel rebuild: {:.1} us",
		rebuild_seconds / rebuilds as f64 * 1e6
	);

	// The first samples after a rebuild run both kernels during the crossfade.
	let start = Instant::now();
	for _ in 0..100 {
		effect.cutoff = 1000.0;
		effect.process(&mut block, &[], &mut context);
		for _ in 0..128 {
			effect.process(&mut block, &[], &mut context);
		}
	}
	println!(
		"  crossfade block: {:.1} us / 128 samples",
		start.elapsed().as_secs_f64() / 100.0 * 1e6
	);

	// The low cutoff path decimates by eight.
	effect.cutoff = 80.0;
	effect.process(&mut block, &[], &mut context);
	let start = Instant::now();
	for _ in 0..100_000 {
		effect.process(&mut block, &[], &mut context);
	}
	let low_seconds = start.elapsed().as_secs_f64();
	println!(
		"  low cutoff:    {:.2} ns/sample ({:.0}x realtime, decimated)",
		low_seconds / 100_000.0 * 1e9,
		1.0 / (low_seconds / (100_000.0 / RATE as f64))
	);

	// Eight voices in one process, the way a polyphonic instrument would run
	// them. Each voice has its own frame and filter state.
	let mut voices: Vec<FilterTableEffect> = (0..8)
		.map(|_| FilterTableEffect::with_frame(RATE, FilterFrame::new()))
		.collect();
	let samples = 200_000;
	let start = Instant::now();
	for _ in 0..samples {
		for voice in &mut voices {
			voice.process(&mut block, &[], &mut context);
		}
	}
	let voice_seconds = start.elapsed().as_secs_f64();
	let voice_realtime = voice_seconds / (samples as f64 / RATE as f64);
	println!(
		"  eight voices:  {:.2} ns/sample/voice, {:.1}% CPU",
		voice_seconds / samples as f64 * 1e9,
		voice_realtime * 100.0
	);
}
