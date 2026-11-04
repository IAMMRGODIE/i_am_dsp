use super::super::*;
use super::*;

/// The IR energy scale keeps the wet path from growing with a peak-normalised
/// IR: this four-sample IR has L2 2, so the first wet sample is halved.
#[test]
fn the_energy_normalization_scales_the_wet() {
	let mut convolver = FftConvolver::<2>::new([vec![1.0, -1.0], vec![1.0, -1.0]], 48_000);
	convolver.normalize_energy();
	convolver.dry_gain = 0.0;
	convolver.wet_gain = 1.0;

	let mut samples = [1.0, 1.0];
	let mut context: Box<dyn ProcessContext> = Box::new(());
	convolver.process(&mut samples, &[], &mut context);

	assert!((samples[0] - 0.5).abs() < 1e-4, "{samples:?}");
}

fn direct_conv(x: &[f32], h: &[f32]) -> Vec<f32> {
	let mut y = vec![0.0; x.len()];
	for n in 0..x.len() {
		for k in 0..h.len() {
			if k <= n {
				y[n] += x[n - k] * h[k];
			}
		}
	}
	y
}

/// delta IR: the FFT convolver must reproduce the input sample for sample.
/// This is the zero latency check.
#[test]
fn delta_ir_is_passthrough() {
	let ir = convolve_identity::<1>(8);
	let mut conv = FftConvolver::<1, 8>::new(ir, 48_000);
	conv.dry_gain = 0.0;
	conv.wet_gain = 1.0;

	let x: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.1).sin()).collect();
	let mut out = vec![];
	let mut ctx: Box<dyn ProcessContext> = Box::new(());
	for &s in &x {
		let mut block = [s];
		conv.process(&mut block, &[], &mut ctx);
		out.push(block[0]);
	}

	for n in 0..x.len() {
		let diff = (out[n] - x[n]).abs();
		assert!(diff < 1e-4, "delta passthrough mismatch at {n}: {}", diff);
	}
}

/// For a short IR that spans the direct head and several partitions, the FFT
/// convolver must agree with a direct (!) convolution.
#[test]
fn matches_direct_convolution() {
	// IR longer than a few blocks, spanning multiple partitions
	let ir_len = 257usize;
	let ir: Vec<f32> = (0..ir_len)
		.map(|i| (i as f32 * 0.3).sin() * (-(i as f32) / 40.0).exp())
		.collect();
	let ir_channels = [ir.clone()];

	let mut fft_conv = FftConvolver::<1, 8>::new(ir_channels.clone(), 48_000);
	fft_conv.dry_gain = 0.0;
	fft_conv.wet_gain = 1.0;

	let x: Vec<f32> = (0..2048)
		.map(|i| (i as f32 * 0.05).sin() + 0.3 * (i as f32 * 0.9).cos())
		.collect();

	let mut out = vec![];
	let mut ctx: Box<dyn ProcessContext> = Box::new(());
	for &s in &x {
		let mut block = [s];
		fft_conv.process(&mut block, &[], &mut ctx);
		out.push(block[0]);
	}

	let reference = direct_conv(&x, &ir);

	let mut max_err = 0.0f32;
	for n in 0..x.len() {
		let err = (out[n] - reference[n]).abs();
		max_err = max_err.max(err);
	}
	assert!(
		max_err < 1e-3,
		"FFT convolver disagrees with direct: max err = {max_err}"
	);
}

/// No IR (empty) must not panic and should behave as a passthrough.
#[test]
fn empty_ir_does_not_panic() {
	let mut conv = FftConvolver::<1, 8>::new([vec![]], 48_000);
	conv.wet_gain = 0.0;

	let mut ctx: Box<dyn ProcessContext> = Box::new(());
	for i in 0..128usize {
		let mut block = [i as f32 * 0.01];
		conv.process(&mut block, &[], &mut ctx);
	}
}

/// dry_gain and wet_gain must both be applied and stay time-aligned: the wet
/// path is zero latency, so with a delta IR and gains of 1.0/1.0 the output is
/// twice the input, sample for sample.
#[test]
fn dry_wet_mix_is_time_aligned() {
	let ir = convolve_identity::<1>(8);
	let mut conv = FftConvolver::<1, 8>::new(ir, 48_000);
	conv.dry_gain = 1.0;
	conv.wet_gain = 1.0;

	let x: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.1).sin()).collect();
	let mut out = vec![];
	let mut ctx: Box<dyn ProcessContext> = Box::new(());
	for &s in &x {
		let mut block = [s];
		conv.process(&mut block, &[], &mut ctx);
		out.push(block[0]);
	}

	for n in 0..x.len() {
		let diff = (out[n] - 2.0 * x[n]).abs();
		assert!(diff < 1e-4, "dry/wet mix mismatch at {n}: {}", diff);
	}
}
/// An impulse response that spans several partitions, including equal sized
/// and staggered ones, must still match a direct convolution.
#[test]
fn staggered_partitions_match_direct_convolution() {
	let ir_len = 1500usize;
	let ir: Vec<f32> = (0..ir_len)
		.map(|i| (i as f32 * 0.11).sin() * (-(i as f32) / 300.0).exp())
		.collect();

	let mut conv = FftConvolver::<1, 8>::new([ir.clone()], 48_000);
	conv.dry_gain = 0.0;
	conv.wet_gain = 1.0;

	let x: Vec<f32> = (0..4096)
		.map(|i| (i as f32 * 0.017).sin() + 0.5 * (i as f32 * 0.31).cos())
		.collect();

	let mut out = vec![];
	let mut ctx: Box<dyn ProcessContext> = Box::new(());
	for &s in &x {
		let mut block = [s];
		conv.process(&mut block, &[], &mut ctx);
		out.push(block[0]);
	}

	let reference = direct_conv(&x, &ir);

	let mut max_err = 0.0f32;
	for n in 0..x.len() {
		max_err = max_err.max((out[n] - reference[n]).abs());
	}
	assert!(
		max_err < 1e-3,
		"the partitions disagree with direct: max err = {max_err}"
	);
}

/// The production configuration (FFT_SIZE = 256 with the one second impulse
/// responses the reverb presets use) must run without tripping the alignment
/// and accumulator bounds assertions, and stay finite.
#[test]
fn long_ir_smoke() {
	let ir: Vec<f32> = (0..48_000)
		.map(|i| (i as f32 * 0.003).sin() * (-(i as f32) / 8000.0).exp())
		.collect();

	let mut conv = FftConvolver::<1, 256>::new([ir], 48_000);
	conv.dry_gain = 0.0;
	conv.wet_gain = 1.0;

	let mut ctx: Box<dyn ProcessContext> = Box::new(());
	for i in 0..20_000usize {
		let mut block = [(i as f32 * 0.01).sin()];
		conv.process(&mut block, &[], &mut ctx);
		assert!(block[0].is_finite());
	}
}

/// Run 'samples' samples through a convolver with the given block size and
/// check the result against a direct convolution.
fn check_block_size<const FS: usize>(ir: &[f32], samples: usize) {
	let mut conv = FftConvolver::<1, FS>::new([ir.to_vec()], 48_000);
	conv.dry_gain = 0.0;
	conv.wet_gain = 1.0;

	let x: Vec<f32> = (0..samples)
		.map(|i| (i as f32 * 0.023).sin() + 0.4 * (i as f32 * 0.7).cos())
		.collect();

	let mut out = vec![];
	let mut ctx: Box<dyn ProcessContext> = Box::new(());
	for &s in &x {
		let mut block = [s];
		conv.process(&mut block, &[], &mut ctx);
		out.push(block[0]);
	}

	let reference = direct_conv(&x, ir);
	let mut max_err = 0.0f32;
	for n in 0..x.len() {
		max_err = max_err.max((out[n] - reference[n]).abs());
	}
	assert!(
		max_err < 1e-3,
		"FFT_SIZE = {FS} disagrees with direct: max err = {max_err}"
	);
}

/// Block sizes that are not powers of two have to work as well: the block
/// ring and the output accumulator are rounded up to powers of two
/// independently of FFT_SIZE, and a partition cycle is only ever a multiple
/// of the block size.
#[test]
fn non_power_of_two_block_size() {
	let ir: Vec<f32> = (0..100)
		.map(|i| (i as f32 * 0.19).sin() * (-(i as f32) / 40.0).exp())
		.collect();
	check_block_size::<12>(&ir, 512);
	check_block_size::<13>(&ir, 512);
}

/// A single deep tap must come out at exactly the tap index, with no extra
/// block of latency. This pins down the alignment of the staggered deep
/// partitions.
#[test]
fn deep_tap_keeps_its_offset() {
	let mut ir = vec![0.0f32; 1500];
	ir[900] = 1.0;

	let mut conv = FftConvolver::<1, 8>::new([ir], 48_000);
	conv.dry_gain = 0.0;
	conv.wet_gain = 1.0;

	let x: Vec<f32> = (0..3000)
		.map(|i| ((i * 37 % 101) as f32) / 101.0 - 0.5)
		.collect();

	let mut out = vec![];
	let mut ctx: Box<dyn ProcessContext> = Box::new(());
	for &s in &x {
		let mut block = [s];
		conv.process(&mut block, &[], &mut ctx);
		out.push(block[0]);
	}

	let tap = 900;
	for n in tap..x.len() {
		let diff = (out[n] - x[n - tap]).abs();
		assert!(diff < 1e-4, "deep tap misaligned at {n}: {}", diff);
	}
}

/// The partition cap has to bound the largest transform even for impulse
/// responses much longer than it, the split has to cover the impulse response
/// without gaps, and no partition may start before its own hop: that last
/// invariant is what makes the convolver zero latency.
#[test]
fn partition_layout_is_bounded_causal_and_complete() {
	let ir = vec![0.1f32; MAX_PARTITION_HOP * 4];
	let buffer = FftBuffer::<1, 256>::new([ir], 48_000);
	let partitions = &buffer.partitions[0];

	assert_eq!(buffer.head[0].len(), 256, "the head covers one block");
	assert!(
		partitions.len() < 16,
		"the split should stay logarithmic, got {} partitions",
		partitions.len()
	);
	assert_eq!(
		partitions[0].offset, 256,
		"the tail starts where the head stops"
	);

	let mut covered = buffer.head[0].len();
	for partition in partitions {
		assert_eq!(
			partition.offset, covered,
			"the split has a gap or an overlap"
		);
		assert!(
			partition.offset >= partition.hop,
			"offset {} below hop {}",
			partition.offset,
			partition.hop
		);
		assert!(
			partition.hop <= MAX_PARTITION_HOP,
			"hop {} above the cap",
			partition.hop
		);
		assert!(
			partition.fft_size <= 2 * MAX_PARTITION_HOP,
			"fft {} above the bound",
			partition.fft_size
		);
		covered += partition.segment_len;
	}
	assert_eq!(covered, MAX_PARTITION_HOP * 4, "the split is incomplete");
}
