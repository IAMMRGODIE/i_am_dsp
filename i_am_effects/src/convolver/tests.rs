use super::*;
use i_am_dsp::prelude::Parameters;
use i_am_dsp::tools::pcm_data::save_pcm_data;
use std::sync::atomic::AtomicUsize;

/// A unique temporary WAV for each test, returned as its encoded bytes.
fn wav_bytes(channels: &[Vec<f32>], rate: usize) -> Vec<u8> {
	static NEXT: AtomicUsize = AtomicUsize::new(0);
	// The system temp directory is not writable in every build environment, so
	// the workspace's own target directory holds the fixture files.
	let directory =
		std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../target/convolver-tests");
	std::fs::create_dir_all(&directory).expect("a temp directory");
	let path = directory.join(format!("ir-{}.wav", NEXT.fetch_add(1, Ordering::Relaxed)));
	save_pcm_data(&path, channels, rate).expect("a wav file");
	let bytes = std::fs::read(&path).expect("the encoded bytes");
	let _ = std::fs::remove_file(&path);
	bytes
}

fn delta_ir(name: &str, fingerprint: u64) -> Arc<IrData> {
	Arc::new(IrData {
		name: name.to_string(),
		rate: 48_000,
		left: vec![1.0],
		right: vec![1.0],
		peaks: vec![(1.0, 1.0)],
		fingerprint,
	})
}

#[test]
fn an_uploaded_wav_becomes_an_impulse_response() {
	let bytes = wav_bytes(&[vec![1.0, 0.0, 0.5], vec![0.5, 0.0, -0.5]], 48_000);
	let ir = decode_ir(&bytes, 48_000, "click.wav".to_string()).expect("a decoded response");

	assert_eq!(ir.rate, 48_000);
	assert_eq!(ir.samples(), 3);
	assert_eq!(ir.left.len(), 3);
	assert_eq!(ir.right.len(), 3);
	assert_eq!(ir.name, "click.wav");
	assert!(!ir.peaks.is_empty());
	assert_eq!(
		ir.fingerprint,
		fingerprint(&ir.name, ir.rate, &ir.left, &ir.right)
	);
}

#[test]
fn a_mono_response_fills_both_channels() {
	let bytes = wav_bytes(&[vec![1.0, -0.5, 0.25]], 48_000);
	let ir = decode_ir(&bytes, 48_000, "mono.wav".to_string()).expect("a decoded response");

	assert_eq!(ir.left, ir.right);
}

#[test]
fn an_impulse_response_is_brought_to_the_host_rate() {
	let mut click = vec![0.0; 4096];
	click[0] = 1.0;
	let bytes = wav_bytes(&[click.clone(), click], 48_000);
	let ir = decode_ir(&bytes, 24_000, "click.wav".to_string()).expect("a resampled response");

	assert_eq!(ir.rate, 24_000);
	assert!(
		(ir.samples() as isize - 2048).abs() < 64,
		"a click of 4096 samples at 48 kHz should be about 2048 at 24 kHz, got {}",
		ir.samples(),
	);
}

#[test]
fn the_ir_route_loads_and_describes_a_file() {
	let slot = IrSlot::new();
	slot.set_host_rate(48_000);
	let bytes = wav_bytes(&[vec![1.0, 0.0, 0.5]], 48_000);

	let post = ProtocolRequest {
		method: "POST",
		path: "/ir",
		query: Some("name=click.wav"),
		body: &bytes,
	};
	let response = ir_route(&post, &slot).expect("the route owns /ir");
	assert_eq!(response.status, 200);
	assert!(slot.load().is_some());

	let get = ProtocolRequest {
		method: "GET",
		path: "/ir",
		query: None,
		body: &[],
	};
	let description = ir_route(&get, &slot).expect("the route owns /ir");
	let body = String::from_utf8(description.body).expect("utf-8 json");
	assert!(body.contains("\"name\":\"click.wav\""), "{body}");
	assert!(body.contains("\"samples\":3"), "{body}");
}

#[test]
fn an_empty_upload_reports_an_error() {
	let slot = IrSlot::new();
	let post = ProtocolRequest {
		method: "POST",
		path: "/ir",
		query: None,
		body: &[],
	};

	ir_route(&post, &slot).expect("the route owns /ir");
	assert!(slot.error().is_some());
	assert!(slot.load().is_none());
}

#[test]
fn the_convolver_passes_a_click_through_a_delta_response() {
	let slot = Arc::new(IrSlot::new());
	slot.set(delta_ir("delta", 7));
	let mut effect = ConvolverEffect::new(48_000, slot);
	let mut context: Box<dyn ProcessContext> = Box::new(());
	let mut frame = [0.5, -0.25];

	effect.process(&mut frame, &[], &mut context);

	assert!((frame[0] - 0.5).abs() < 0.01, "{}", frame[0]);
	assert!((frame[1] + 0.25).abs() < 0.01, "{}", frame[1]);
}

#[test]
fn a_fully_dry_balance_leaves_the_input_alone() {
	let slot = Arc::new(IrSlot::new());
	slot.set(delta_ir("delta", 9));
	let mut effect = ConvolverEffect::new(48_000, slot);
	effect.dry_gain = 1.0;
	effect.wet_gain = 0.0;
	let mut context: Box<dyn ProcessContext> = Box::new(());
	let mut frame = [0.25, 0.75];

	effect.process(&mut frame, &[], &mut context);

	assert_eq!(frame, [0.25, 0.75]);
}

#[test]
fn the_dry_and_wet_gains_mix_the_two_paths() {
	let slot = Arc::new(IrSlot::new());
	slot.set(delta_ir("delta", 10));
	let mut effect = ConvolverEffect::new(48_000, slot);
	effect.dry_gain = 0.25;
	effect.wet_gain = 0.5;
	let mut context: Box<dyn ProcessContext> = Box::new(());
	let mut frame = [1.0, -1.0];

	effect.process(&mut frame, &[], &mut context);

	assert!((frame[0] - 0.75).abs() < 0.01, "{}", frame[0]);
	assert!((frame[1] + 0.75).abs() < 0.01, "{}", frame[1]);
}

/// The fades are state, not controls: a host cannot automate them, and a
/// project carries them.
#[test]
fn the_fades_are_state_not_a_control() {
	let mut effect = ConvolverEffect::new(48_000, Arc::new(IrSlot::new()));
	effect.fades = IrFades {
		fade_in: 0.2,
		fade_out: 0.1,
	};

	let parameter = effect
		.get_parameters()
		.into_iter()
		.find(|parameter| parameter.identifier == "fades")
		.expect("the fades are part of the state");
	let SetValue::Serialized(bytes) = parameter.value.to_set_value() else {
		panic!("the fades were not serialized");
	};

	let mut fresh = ConvolverEffect::new(48_000, Arc::new(IrSlot::new()));
	assert!(fresh.set_parameter("fades", SetValue::Serialized(bytes)));
	assert_eq!(
		fresh.fades,
		IrFades {
			fade_in: 0.2,
			fade_out: 0.1
		}
	);
}

/// A state load hands the fades back to the picture, which is what the hero
/// draws and what a drag starts from.
#[test]
fn a_state_load_publishes_the_fades_to_the_picture() {
	let slot = Arc::new(IrSlot::new());
	let mut effect = ConvolverEffect::new(48_000, slot.clone());
	effect.fades = IrFades {
		fade_in: 0.25,
		fade_out: 0.5,
	};

	let mut context: Box<dyn ProcessContext> = Box::new(());
	let mut frame = [0.0, 0.0];
	effect.process(&mut frame, &[], &mut context);

	assert_eq!(
		slot.fades(),
		IrFades {
			fade_in: 0.25,
			fade_out: 0.5
		}
	);
}

#[test]
fn the_fades_soften_both_ends_of_the_response() {
	let mut samples = vec![1.0_f32; 100];

	fade_edges(&mut samples, 0.5, 0.5);

	assert!(samples[0].abs() < 1e-3, "{}", samples[0]);
	assert!(samples[99].abs() < 1e-3, "{}", samples[99]);
	assert!(samples[50] > 0.99, "{}", samples[50]);
	assert!(
		samples[25] > 0.1 && samples[25] < 0.9,
		"the inside of the fade is not a curve: {}",
		samples[25]
	);
}

#[test]
fn a_saved_response_reaches_a_fresh_instance() {
	let slot = Arc::new(IrSlot::new());
	let mut effect = ConvolverEffect::new(48_000, slot);
	effect.ir = Some(delta_ir("saved", 11));

	let parameter = effect
		.get_parameters()
		.into_iter()
		.find(|parameter| parameter.identifier == "ir")
		.expect("the impulse response is part of the state");
	let SetValue::Serialized(bytes) = parameter.value.to_set_value() else {
		panic!("the response was not serialized");
	};

	let fresh_slot = Arc::new(IrSlot::new());
	let mut fresh = ConvolverEffect::new(48_000, fresh_slot.clone());
	assert!(fresh.set_parameter("ir", SetValue::Serialized(bytes)));
	assert!(fresh.ir.is_some());

	let mut context: Box<dyn ProcessContext> = Box::new(());
	let mut frame = [1.0, 1.0];
	fresh.process(&mut frame, &[], &mut context);

	assert!(fresh.convolver.is_some());
	assert!(
		fresh_slot.load().is_some(),
		"the state load also fills the slot"
	);
}

#[test]
fn a_pending_picker_is_reported() {
	let slot = IrSlot::new();
	slot.set_pending(true);
	assert!(slot.json().contains("\"pending\":true"), "{}", slot.json());
	slot.set_pending(false);
	assert!(slot.json().contains("\"pending\":false"), "{}", slot.json());
}

#[test]
fn the_page_name_is_unescaped() {
	assert_eq!(
		percent_decode("Vintage%20Plate%2B.wav"),
		"Vintage Plate+.wav"
	);
}
