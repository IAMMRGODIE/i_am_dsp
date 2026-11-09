use super::*;

fn render(kit: &mut DrumInstrument, count: usize) -> Vec<f32> {
	(0..count).map(|_| kit.next_sample()).collect()
}

fn rms(samples: &[f32]) -> f32 {
	(samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len().max(1) as f32).sqrt()
}

#[test]
fn the_kick_sounds_and_decays() {
	let mut kit = DrumInstrument::new(48_000);
	assert!(kit.start(KICK, 1.0));

	let out = render(&mut kit, 48_000);
	let peak = out
		.iter()
		.fold(0.0f32, |peak, sample| peak.max(sample.abs()));

	assert!(peak > 0.05, "the kick was silent: {peak}");
	assert!(rms(&out[..2_400]) > rms(&out[45_600..]) * 100.0);
}

#[test]
fn the_snare_sounds_and_is_not_the_kick() {
	let mut snare = DrumInstrument::new(48_000);
	assert!(snare.start(SNARE, 1.0));
	let snare = render(&mut snare, 9_600);

	let mut kick = DrumInstrument::new(48_000);
	kick.start(KICK, 1.0);
	let kick = render(&mut kick, 9_600);

	let peak = snare
		.iter()
		.fold(0.0f32, |peak, sample| peak.max(sample.abs()));
	assert!(peak > 0.05, "the snare was silent: {peak}");

	// The wires make the snare cross zero far more often than a 52 Hz shell.
	let crossings = |samples: &[f32]| {
		samples
			.windows(2)
			.filter(|pair| (pair[0] >= 0.0) != (pair[1] >= 0.0))
			.count()
	};

	assert!(
		crossings(&snare) > crossings(&kick) * 3,
		"the snare sounds like a kick: {} vs {}",
		crossings(&snare),
		crossings(&kick)
	);
}

#[test]
fn the_hats_sound_and_the_open_one_rings_longer() {
	let mut closed = DrumInstrument::new(48_000);
	assert!(closed.start(HAT_CLOSED, 1.0));
	let closed = render(&mut closed, 24_000);

	let mut open = DrumInstrument::new(48_000);
	assert!(open.start(HAT_OPEN, 1.0));
	let open = render(&mut open, 24_000);

	let peak = closed
		.iter()
		.fold(0.0f32, |peak, sample| peak.max(sample.abs()));
	assert!(peak > 0.02, "the closed hat was silent: {peak}");
	assert!(
		rms(&open[12_000..]) > rms(&closed[12_000..]) * 5.0,
		"the open hat died like a closed one: {} vs {}",
		rms(&open[12_000..]),
		rms(&closed[12_000..])
	);
}

#[test]
fn a_closed_hat_chokes_an_open_one() {
	let mut free = DrumInstrument::new(48_000);
	free.start(HAT_OPEN, 1.0);
	let free = render(&mut free, 24_000);

	let mut choked = DrumInstrument::new(48_000);
	choked.start(HAT_OPEN, 1.0);
	render(&mut choked, 4_800);
	// The choke is the closed hat's gesture, not its sound: a silent one
	// still stops the open hat, and then the tail is the only thing left.
	choked.level_hat = 0.0;
	choked.start(HAT_CLOSED, 1.0);
	let after = render(&mut choked, 4_800);

	let open_tail = rms(&free[4_800..6_240]);
	let choked_tail = rms(&after[0..1_440]);

	assert!(
		choked_tail < open_tail * 0.7,
		"the closed hat did not choke the open one: {choked_tail} vs {open_tail}"
	);
}

#[test]
fn the_clap_retriggers() {
	let mut kit = DrumInstrument::new(48_000);
	assert!(kit.start(CLAP, 1.0));
	let out = render(&mut kit, 12_000);

	// A burst, a near silent gap, another burst: the four hands.
	let early = rms(&out[0..240]);
	let gap = rms(&out[432..480]);

	assert!(early > 0.01, "the clap was silent: {early}");
	assert!(
		gap < early * 0.25,
		"the clap did not retrigger: {early} then {gap}"
	);
}

#[test]
fn the_toms_follow_the_note() {
	let crossings = |note: u8| {
		let mut kit = DrumInstrument::new(48_000);
		assert!(kit.start(note, 1.0));
		let out = render(&mut kit, 4_800);

		out.windows(2)
			.filter(|pair| (pair[0] >= 0.0) != (pair[1] >= 0.0))
			.count()
	};

	assert!(
		crossings(TOM_HIGH) > crossings(TOM_LOW) * 13 / 10,
		"the toms do not follow the note: {} vs {}",
		crossings(TOM_HIGH),
		crossings(TOM_LOW)
	);
}

#[test]
fn the_cymbals_are_not_just_sines() {
	let crossings = |fm: f32| {
		let mut kit = DrumInstrument::new(48_000);
		kit.cymbal_fm = fm;
		kit.start(CRASH, 1.0);
		let out = render(&mut kit, 2_400);

		out.windows(2)
			.filter(|pair| (pair[0] >= 0.0) != (pair[1] >= 0.0))
			.count()
	};

	let plain = crossings(0.0);
	let fm = crossings(1.0);

	// The bank itself is dense already (and now reaches the top octave), so
	// the sidebands show as a smaller jump in zero crossings than before.
	assert!(
		fm as f32 > plain as f32 * 1.2,
		"the FM did not thicken the cymbal: {plain} then {fm}"
	);
}

#[test]
fn the_cymbals_ring_longer_than_a_hat() {
	let tail = |note: u8| {
		let mut kit = DrumInstrument::new(48_000);
		kit.start(note, 1.0);
		let out = render(&mut kit, 96_000);

		rms(&out[48_000..])
	};

	let crash = tail(CRASH);
	let closed = tail(HAT_CLOSED);
	let ride = tail(RIDE);

	assert!(crash > 0.001, "the crash died: {crash}");
	assert!(
		crash > closed * 20.0,
		"the crash rang like a hat: {crash} vs {closed}"
	);
	assert!(ride < crash, "the ride outlasts the crash");
}

#[test]
fn the_kit_answers_the_general_midi_map() {
	for note in [
		35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55, 57, 59,
	] {
		let mut kit = DrumInstrument::new(48_000);
		assert!(kit.start(note, 0.9), "note {note} is not mapped");
		let peak = render(&mut kit, 480)
			.iter()
			.fold(0.0f32, |peak, sample| peak.max(sample.abs()));

		assert!(peak > 0.0, "note {note} made no sound");
	}

	for note in [34u8, 54, 56, 60, 127] {
		let mut kit = DrumInstrument::new(48_000);
		assert!(!kit.start(note, 0.9), "note {note} should be silent");
	}
}

#[test]
fn a_harder_hit_hits_harder() {
	// The cymbal is the convolver's output now; the IR owns the timbre, and
	// what velocity still moves in the wet is the level.
	let rms_of = |velocity: f32| {
		let mut kit = DrumInstrument::new(48_000);
		kit.start(CRASH, velocity);
		rms(&render(&mut kit, 2_400))
	};

	let soft = rms_of(0.2);
	let hard = rms_of(1.0);

	assert!(
		hard > soft * 1.5,
		"a harder hit was not louder: {soft} then {hard}"
	);
}

#[test]
fn an_imported_click_is_layered_over_the_kick() {
	let sample = Arc::new(SampleData {
		right: None,
		name: "click".to_string(),
		rate: 48_000,
		pcm: vec![0.4; 480],
	});

	let mean = |with: bool| {
		let mut kit = DrumInstrument::new(48_000);
		kit.level_kick = 1.0;

		if with {
			kit.click_sample = Some((*sample).clone());
			kit.click_sample_level = 0.8;
		}

		kit.start(KICK, 0.5);
		let out = render(&mut kit, 480);

		out.iter().sum::<f32>() / out.len() as f32
	};

	let plain = mean(false);
	let layered = mean(true);

	assert!(
		layered > plain + 0.02,
		"the imported click was not heard: {layered} vs {plain}"
	);
}

#[test]
fn a_slot_reports_its_shape() {
	let slots = Samples::new();
	slots.set(
		slot_drum("kick").expect("the kick's slot"),
		Some(SampleData {
			right: None,
			name: "kick".to_string(),
			rate: 48_000,
			pcm: vec![1.0; 320],
		}),
	);

	let json = slots.json();

	assert!(json.contains("\"peaks\""));
	assert!(json.contains("1.000"), "{json}");
}

#[test]
fn a_loaded_slot_replaces_the_drum() {
	let slots = Samples::new();
	let mut kit = DrumInstrument::new(48_000).with_samples(slots.clone());

	slots.set(
		slot_drum("snare").expect("the snare's slot"),
		Some(SampleData {
			right: None,
			name: "snare".to_string(),
			rate: 48_000,
			pcm: vec![0.5; 2_400],
		}),
	);

	assert!(kit.start(SNARE, 0.5));

	let out = render(&mut kit, 2_400);
	let mean = out.iter().sum::<f32>() / out.len() as f32;

	assert!(
		mean > 0.1,
		"the sample slot did not replace the snare: {mean}"
	);
}

#[test]
fn a_fitted_bank_replaces_the_cymbal() {
	let mut kit = DrumInstrument::new(48_000);
	kit.set_fitted(Some(DrumPreset {
		name: "fit".to_string(),
		decay: 0.4,
		partials: vec![PresetPartial {
			frequency: 500.0,
			amplitude: 1.0,
			t60: 0.4,
		}],
		noise: Vec::new(),
	}));

	assert!(kit.start(CRASH, 1.0));
	let out = render(&mut kit, 4_800);
	let peak = out
		.iter()
		.fold(0.0f32, |peak, sample| peak.max(sample.abs()));
	let crossings = out
		.windows(2)
		.filter(|pair| (pair[0] >= 0.0) != (pair[1] >= 0.0))
		.count();

	assert!(peak > 0.01, "the fitted bank was silent: {peak}");
	assert!(
		crossings > 40 && crossings < 200,
		"the fitted bank is not one 500 Hz partial: {crossings} crossings"
	);
}

#[test]
fn the_snare_splits_its_shell_and_wires() {
	let rms_of = |noise: f32| {
		let curves = SnareCurves {
			base: Curve::from_triples([(0.0, 1.0, 0.0), (0.15, 0.0, 0.0)]),
			noise: Curve::from_triples([(0.0, noise, 0.0), (0.15, 0.0, 0.0)]),
			pitch: Curve::from_triples([(0.0, 0.0, 0.0), (0.01, 0.0, 0.0)]),
		};
		let mut kit = DrumInstrument::new(48_000).with_snare(Arc::new(Mutex::new(curves)));
		kit.snare_wires = 1.0;

		assert!(kit.start(SNARE, 1.0));
		rms(&render(&mut kit, 4_800))
	};

	let shell = rms_of(0.0);
	let both = rms_of(1.0);

	assert!(shell > 1e-4, "the shell was silent: {shell}");
	assert!(
		(both - shell).abs() > shell * 0.02,
		"the wires changed nothing: {both} vs {shell}"
	);
}

#[test]
fn the_hats_can_be_wide() {
	let spread = |width: f32| {
		let mut kit = DrumInstrument::new(48_000);
		kit.width = width;
		kit.pan_hat = 0.0;
		kit.start(HAT_CLOSED, 1.0);

		let mut total = 0.0;
		for _ in 0..4_800 {
			let frame = kit.next_frame();
			total += (frame[0] - frame[1]).abs();
		}

		total
	};

	assert!(spread(1.0) > 0.01, "the wide hat was mono: {}", spread(1.0));
	assert!(spread(0.0) < 1e-4, "the mono hat was wide: {}", spread(0.0));
}

#[test]
fn a_pan_moves_the_hat() {
	let energy = |pan: f32| {
		let mut kit = DrumInstrument::new(48_000);
		kit.pan_hat = pan;
		kit.start(HAT_CLOSED, 1.0);

		let mut left = 0.0;
		let mut right = 0.0;
		for _ in 0..4_800 {
			let frame = kit.next_frame();
			left += frame[0] * frame[0];
			right += frame[1] * frame[1];
		}

		(left, right)
	};

	let (left, right) = energy(-1.0);
	assert!(
		left > right * 10.0,
		"the pan did not move the hat: {left} vs {right}"
	);
}

#[test]
fn the_mix_can_silence_a_voice() {
	let mut kit = DrumInstrument::new(48_000);
	kit.level_snare = 0.0;
	kit.start(SNARE, 0.9);
	assert!(
		render(&mut kit, 4_800)
			.iter()
			.all(|sample| sample.abs() < 1e-9)
	);

	let mut kit = DrumInstrument::new(48_000);
	kit.start(SNARE, 0.9);
	assert!(
		render(&mut kit, 4_800)
			.iter()
			.any(|sample| sample.abs() > 1e-6)
	);
}

#[test]
fn the_mix_level_scales_a_voice() {
	let rms_of = |level: f32| {
		let mut kit = DrumInstrument::new(48_000);
		kit.level_kick = level;
		kit.start(KICK, 0.8);
		rms(&render(&mut kit, 4_800))
	};

	let full = rms_of(1.0);
	let half = rms_of(0.5);

	assert!(half < full * 0.6 && half > full * 0.4, "{full} vs {half}");
}

#[test]
fn the_amp_templates_are_in_seconds() {
	for template in default_amp_templates() {
		let first = template.points().first().expect("a template has points");

		assert!(first.time <= 1e-4);
		assert!(
			template.end() > 0.01 && template.end() < 10.0,
			"{}",
			template.end()
		);
	}

	// The kick's default ends at the shape's 450 ms.
	assert!((default_amp_templates()[0].end() - 0.45).abs() < 1e-4);
}

#[test]
fn the_amp_curve_bounds_the_voice() {
	let mut amplitudes = default_amp_templates();
	amplitudes[0] = Curve::from_triples([(0.0, 1.0, 0.0), (0.05, 0.0, 0.0)]);
	let mut kit = DrumInstrument::new(48_000).with_amps(Arc::new(Mutex::new(amplitudes)));
	kit.decay = 1.0;

	assert!(kit.start(KICK, 1.0));
	let out = render(&mut kit, 4_800);
	let tail = out[3_600..]
		.iter()
		.fold(0.0f32, |peak, sample| peak.max(sample.abs()));

	assert!(tail < 1e-3, "the voice outlived its amp curve: {tail}");
}

#[test]
fn a_ten_octave_start_stays_finite() {
	let curve = Curve::from_triples([(0.0, 1_000.0, 0.0), (0.02, 1.0, 0.0)]);
	let mut kit = DrumInstrument::new(48_000).with_shared(Arc::new(Mutex::new(curve)));

	assert!(kit.start(KICK, 1.0));
	let out = render(&mut kit, 4_800);

	assert!(
		out.iter()
			.all(|sample| sample.is_finite() && sample.abs() < 4.0),
		"a ten octave start left the rails"
	);
}

#[test]
fn the_kit_presets_write_the_controls() {
	let mut kit = DrumInstrument::new(48_000);

	kit.apply_kit(1);
	assert_eq!(kit.kit, 1);
	assert!((kit.tune - 48.0).abs() < 1e-4, "{}", kit.tune);
	assert!((kit.cymbal_fm - 0.45).abs() < 1e-4);

	kit.apply_kit(0);
	assert!((kit.tune - 52.0).abs() < 1e-4, "{}", kit.tune);
}

#[test]
fn a_note_the_kick_does_not_answer_is_silent() {
	let mut kit = DrumInstrument::new(48_000);

	assert!(!kit.start(60, 1.0));
	assert!(
		render(&mut kit, 480)
			.iter()
			.all(|sample| sample.abs() < 1e-12)
	);
}

#[test]
fn a_harder_hit_is_louder() {
	let mut soft = DrumInstrument::new(48_000);
	let mut hard = DrumInstrument::new(48_000);
	soft.start(KICK, 0.25);
	hard.start(KICK, 1.0);

	let soft = rms(&render(&mut soft, 4_800));
	let hard = rms(&render(&mut hard, 4_800));

	assert!(hard > soft * 2.0, "{hard} vs {soft}");
}

#[test]
fn a_clickless_kick_has_no_click() {
	let mut kit = DrumInstrument::new(48_000);
	kit.click = 0.0;

	let recipe = kit.kick_recipe();
	let material = recipe.layers.iter().filter(|layer| layer.keytrack == 0.0);
	let mut any = false;

	for layer in material {
		match &layer.source {
			Source::Partials(partials) => any |= !partials.is_empty(),
			Source::Noise(bands) => any |= !bands.is_empty(),
			Source::Metal { .. } => any = true,
			Source::Fm { .. } => any = true,
			Source::NoiseWide(_) => any = true,
			Source::Sample { .. } => any = true,
		}
	}

	assert!(!any, "a clickless kick still has material layers");
}

#[test]
fn the_editor_curve_is_heard() {
	let kit = DrumInstrument::new(48_000);
	let shared = Arc::new(Mutex::new(Curve::from_triples([
		(0.0, 7.0, 0.0),
		(0.1, 1.0, 0.0),
	])));

	let kit = kit.with_shared(shared);

	assert!((kit.kick_recipe().pitch.value(0.0) - 7.0).abs() < 1e-5);
}
