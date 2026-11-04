//! Every catalog product, the scope rings and the note input.

use super::*;

/// Every product in the catalog can sit in a slot and run. The rack offers the
/// whole line, not a subset it special-cases.
#[test]
fn every_catalog_product_can_run_in_a_slot() {
	for kind in ModuleKind::ALL {
		let mut state = RackState::initial();
		state.bands[0].slots[0] = Some(ModuleState::new(0, kind));

		let mut engine = crate::engine::Engine::new(48_000, &state);
		let mut context: Box<dyn ProcessContext> = Box::new(());
		let mut frame = [0.0_f32; 2];

		for i in 0..256 {
			let time = i as f32 / 48_000.0;
			let sample = (2.0 * std::f32::consts::PI * 220.0 * time).sin() * 0.1;
			frame = [sample, sample];
			engine.process(&mut frame, &[], &mut context);
		}

		assert!(
			frame[0].is_finite() && frame[1].is_finite(),
			"{} produced a non-finite frame",
			kind.name()
		);
	}
}

/// The rack page reads the same /scope document the products do: the heroes
/// that draw audio are useless without it.
#[test]
fn the_page_reads_the_audio_scope() {
	let js = PAGE_JS;

	assert!(
		js.contains("fetch(\"/module-scope?id=\""),
		"the rack does not read a module's own audio"
	);
	assert!(
		js.contains("fetch(\"/module-stereo?id=\""),
		"the rack does not read a module's stereo pair"
	);
	assert!(
		js.contains("card.repaintHero"),
		"a live hero is never redrawn"
	);
	assert!(js.contains("cardContext()"), "the cards share one context");
	assert!(
		js.contains("fetch(\"/module-note?id=\""),
		"the cards never read the note their module hears"
	);
	assert!(
		js.contains("fetch(\"/module-bands?id=\""),
		"the cards never read their band meter"
	);
	assert!(
		js.contains("matrix-filter"),
		"the matrix has no source filter"
	);
	assert!(
		js.contains("attachEqBands("),
		"the EQ picture has no draggable bands in the rack"
	);
	assert!(
		js.contains("applyPanelCollapsed("),
		"the control panel cannot be folded away"
	);
	assert!(
		js.contains("liveModulation("),
		"the cards never see the modulated values"
	);
	assert!(
		js.contains("indexOf(\"source:\") === 0"),
		"a source dropped on a slot would be treated as a module"
	);
	assert!(
		js.contains("spectra: { input: null, output: null }"),
		"no per-card spectra"
	);
	assert!(
		js.contains("channels: module.scope"),
		"the page polls cards that have no audio reader"
	);
}

/// The scope the page polls is fed by the audio thread, not by the UI thread.
#[test]
fn the_scope_receives_the_signal() {
	let mut rack = Rack::default();
	rack.on_activate(48_000);

	for i in 0..512 {
		let sample = if i % 2 == 0 { 0.5 } else { -0.5 };
		let mut frame = [sample, sample];
		let mut context: Box<dyn ProcessContext> = Box::new(());
		rack.process(&mut frame, &[], &mut context);
	}

	let snapshot = rack.scope.snapshot();
	assert_eq!(
		snapshot.len(),
		4096 * 12,
		"the scope document has the wrong size"
	);

	// The document is zero-padded up to POINTS, so the newest frame is the one
	// just before the padding starts.
	let newest = 511 * 4;
	let value = f32::from_le_bytes([
		snapshot[newest],
		snapshot[newest + 1],
		snapshot[newest + 2],
		snapshot[newest + 3],
	]);
	assert!(value.abs() > 0.01, "the scope has no signal: {value}");
}

/// A card shows what entered and left its own module: the ring behind the
/// second module holds exactly what the first one produced.
#[test]
fn a_module_scope_holds_what_that_module_heard() {
	let mut rack = Rack::default();
	rack.on_activate(48_000);

	{
		let mut state = rack.shared.lock();
		state.bands[0].slots[0] = Some(ModuleState::new(10, ModuleKind::Gain));
		state.bands[0].slots[1] = Some(ModuleState::new(11, ModuleKind::Gain));
	}
	rack.shared.dirty.store(true, Ordering::Release);

	for i in 0..512 {
		let time = i as f32 / 48_000.0;
		let sample = (2.0 * std::f32::consts::PI * 220.0 * time).sin() * 0.25;
		let mut frame = [sample, sample];
		let mut context: Box<dyn ProcessContext> = Box::new(());
		rack.process(&mut frame, &[], &mut context);
	}

	let first = rack.shared.scope(10, Channels::None);
	let second = rack.shared.scope(11, Channels::None);
	assert!(
		!std::sync::Arc::ptr_eq(&first, &second),
		"two modules share one scope"
	);

	let produced = first.snapshot();
	let heard = second.snapshot();
	let points = 4096 * 4;

	for frame in 0..512 {
		let offset = frame * 4;
		let out = &produced[points + offset..points + offset + 4];
		let into = &heard[offset..offset + 4];
		assert_eq!(out, into, "frame {frame} differs between the modules");
	}
}

/// The rings are allocated from what the card draws: a plot of parameters keeps
/// none, a waveform keeps the mono ones, a goniometer keeps the pair.
#[test]
fn a_card_without_an_audio_plot_keeps_no_rings() {
	let mut rack = Rack::default();
	rack.on_activate(48_000);

	{
		let mut state = rack.shared.lock();
		state.bands[0].slots[0] = Some(ModuleState::new(20, ModuleKind::Delay));
		state.bands[0].slots[1] = Some(ModuleState::new(21, ModuleKind::Filter));
		state.parallel.slots[0] = Some(ModuleState::new(22, ModuleKind::Width));
	}
	rack.shared.dirty.store(true, Ordering::Release);

	for i in 0..64 {
		let time = i as f32 / 48_000.0;
		let sample = (2.0 * std::f32::consts::PI * 220.0 * time).sin() * 0.1;
		let mut frame = [sample, sample];
		let mut context: Box<dyn ProcessContext> = Box::new(());
		rack.process(&mut frame, &[], &mut context);
	}

	assert_eq!(
		rack.shared.scope(20, Channels::None).channels(),
		Channels::None,
		"a delay card reads no audio"
	);
	assert_eq!(
		rack.shared.scope(21, Channels::None).channels(),
		Channels::Mono,
		"a filter card draws its spectrum"
	);
	assert_eq!(
		rack.shared.scope(22, Channels::None).channels(),
		Channels::Stereo,
		"a width card draws the stereo pair"
	);
}

/// The rack offers a note input, which is what the pitch-tracking modules read
/// from the process context; a host that never sends notes still loads it.
#[test]
fn the_rack_declares_a_note_input() {
	use i_am_plugin::Plugin;

	assert_eq!(Rack::INPUT_MIDI_PORTS.len(), 1);
	assert_eq!(Rack::INPUT_MIDI_PORTS[0].name, "Notes");
	assert!(
		ModuleKind::ALL.iter().any(|kind| kind.name() == "Autotune"),
		"the autotune is not offered"
	);
}

/// The note a pitch-tracking module hears lands on its scope, which is what the
/// card's badge reads, and the mask it tunes to comes with it.
#[test]
fn a_note_reaches_the_module_that_follows_it() {
	let mut rack = Rack::default();
	rack.on_activate(48_000);

	{
		let mut state = rack.shared.lock();
		state.bands[0].slots[0] = Some(ModuleState::new(30, ModuleKind::Autotune));
	}
	rack.shared.dirty.store(true, Ordering::Release);

	process_with(
		&mut rack,
		[0.0, 0.0],
		vec![NoteEvent::NoteOn {
			channel: 0,
			note: 60,
			velocity: 0.8,
		}],
	);

	let scope = rack.shared.existing_scope(30).expect("the module scope");
	assert_eq!(scope.note(), 60, "the card badge has no note");
	assert!(
		scope.allowed_notes() >= 0,
		"the autotune did not publish its note mask"
	);
}

/// A pitch-tracking module has notes but no audio document, so the page has to
/// poll one without waiting for the other.
#[test]
fn a_note_module_reads_notes_without_audio() {
	assert!(module::shows_notes(ModuleKind::Autotune));
	assert_eq!(
		module::audio_scope(ModuleKind::Autotune),
		Channels::None,
		"the autotune card draws no audio"
	);

	let js = PAGE_JS;
	assert!(
		js.contains("if (entry.channels) {"),
		"the note poll is still gated behind the audio one"
	);
	assert!(
		!js.contains("if (!entry.channels) return;"),
		"a card without audio never gets its notes"
	);
}

/// A chord lands on the module's scope note by note, which is what the card's
/// badge draws.
#[test]
fn a_chord_reaches_the_module_card() {
	let mut rack = Rack::default();
	rack.on_activate(48_000);

	{
		let mut state = rack.shared.lock();
		state.bands[0].slots[0] = Some(ModuleState::new(31, ModuleKind::Resonator));
	}
	rack.shared.dirty.store(true, Ordering::Release);

	process_with(
		&mut rack,
		[0.0, 0.0],
		[48usize, 55, 64]
			.iter()
			.map(|note| NoteEvent::NoteOn {
				channel: 0,
				note: *note,
				velocity: 0.9,
			})
			.collect(),
	);

	let scope = rack.shared.existing_scope(31).expect("the module scope");
	let held = scope.notes();

	for pitch in [48usize, 55, 64] {
		assert!(
			held & (1u128 << pitch) != 0,
			"note {pitch} is missing from the card"
		);
	}
}

/// The OTT card draws gain reduction, and the vocoder card draws bands: both
/// publish through the module meter, not through the audio scope.
#[test]
fn a_metered_module_publishes_its_bands() {
	let mut rack = Rack::default();
	rack.on_activate(48_000);

	{
		let mut state = rack.shared.lock();
		state.bands[0].slots[0] = Some(ModuleState::new(40, ModuleKind::Ott));
		state.bands[1].slots[0] = Some(ModuleState::new(41, ModuleKind::Vocoder));
	}
	rack.shared.dirty.store(true, Ordering::Release);

	for i in 0..512 {
		let time = i as f32 / 48_000.0;
		let sample = (2.0 * std::f32::consts::PI * 220.0 * time).sin() * 0.3;
		let mut frame = [sample, sample];
		let mut context: Box<dyn ProcessContext> = Box::new(());
		rack.process(&mut frame, &[], &mut context);
	}

	let ott = rack.shared.existing_meters(40).expect("the ott meter");
	let (bands, gains, input, _) = ott.snapshot();
	assert_eq!(bands, 3);
	assert!(gains, "the OTT reports gain reduction");
	assert!(input.iter().all(|value| value.is_finite()));

	let vocoder = rack.shared.existing_meters(41).expect("the vocoder meter");
	let (bands, gains, input, output) = vocoder.snapshot();
	assert!(!gains, "the vocoder reports levels");
	assert!(bands >= 4, "the vocoder runs a band bank");
	assert_eq!(input.len(), bands);
	assert_eq!(output.len(), bands);
}
