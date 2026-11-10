//! Offline check: the mean of the demo graph's output.
use i_am_dsp::prelude::NoteEvent;
use i_am_patch::poly::Poly;

fn main() {
	let graph = i_am_patch_product::shared::demo();
	let mut poly = Poly::new(&graph, 48_000).expect("poly");
	let mut frames = Vec::new();
	let silence = vec![[0.0f32; 2]; 64];
	for block in 0..200u64 {
		let events = if block == 0 {
			vec![NoteEvent::NoteOn {
				channel: 0,
				note: 60,
				velocity: 1.0,
			}]
		} else {
			Vec::new()
		};
		let mut out = vec![[0.0f32; 2]; 64];
		poly.process(&silence, &events, &mut out);
		frames.extend(out);
	}
	let n = frames.len() as f64;
	let mean_l: f64 = frames.iter().map(|f| f[0] as f64).sum::<f64>() / n;
	let mean_r: f64 = frames.iter().map(|f| f[1] as f64).sum::<f64>() / n;
	let peak = frames
		.iter()
		.flat_map(|f| f.iter())
		.fold(0.0f32, |p, s| p.max(s.abs()));
	let tail: f64 = frames[(frames.len() - 6400)..]
		.iter()
		.map(|f| f[0] as f64)
		.sum::<f64>()
		/ 6400.0;
	println!("mean L {mean_l:.6} R {mean_r:.6} tail {tail:.6} peak {peak:.4}");
}
