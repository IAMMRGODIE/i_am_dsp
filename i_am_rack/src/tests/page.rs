//! The page assets and the switches they rely on.

use super::*;

/// The rack page loads the same card assets the product pages use.
#[test]
fn the_rack_page_loads_the_shared_controls() {
	let html = include_str!("../page/rack.html");

	assert!(html.contains("/card.css"));
	assert!(html.contains("/card.js"));
	assert!(html.contains("/kit.css"));
	assert!(html.contains("/kit.js"));
	assert!(
		html.find("/card.js").unwrap() < html.find("/kit.js").unwrap(),
		"card.js must load before the page script"
	);
}

/// The band header keeps its trims on the same line as the band name instead of
/// taking a full knob column of vertical space.
#[test]
fn the_band_header_keeps_its_trim_controls_compact() {
	let css = include_str!("../page/rack.css");
	let js = PAGE_JS;

	assert!(
		css.contains("min-height:52px"),
		"band header height changed"
	);
	assert!(
		css.contains(".band-controls .knob-wrap{"),
		"band control wrapper missing"
	);
	assert!(
		css.contains("flex-direction:row"),
		"band controls are not inline"
	);
	assert!(
		css.contains(".band-controls .knob{width:36px"),
		"band knob size changed"
	);
	assert!(
		js.contains("format: function (value)"),
		"band trims lost their readout"
	);
	assert!(
		js.contains("name.title"),
		"the hidden narrow range needs a tooltip"
	);
}

/// The LFO appends points deliberately on double-click, leaving a single click
/// for dragging an existing point or bend handle.
#[test]
fn the_lfo_adds_points_on_double_click() {
	let js = PAGE_JS;

	assert!(
		js.contains("svg.ondblclick"),
		"the LFO needs a double-click handler"
	);
	assert!(
		js.contains("Double-click to add a point"),
		"the LFO hint is stale"
	);
	assert!(
		js.contains("if (!drag) return;"),
		"an empty single click should not capture the pointer"
	);
}

/// Bypass dims the editable surfaces, while the top bar stays available so the
/// user can turn it back on.
#[test]
fn bypass_dims_the_rack_surfaces() {
	let css = include_str!("../page/rack.css");
	let js = PAGE_JS;

	assert!(
		css.contains("body.bypassed .rack"),
		"the rack is not dimmed"
	);
	assert!(
		css.contains("body.bypassed .bottom"),
		"the bottom panel is not dimmed"
	);
	assert!(
		js.contains("document.body.classList.toggle(\"bypassed\""),
		"bypass does not reach the page"
	);
}

/// The compressor can key off the rack's second input instead of the signal it
/// is processing: a silent sidechain leaves a loud band alone, while the same
/// band keyed off itself is turned down.
#[test]
fn the_compressor_can_key_off_the_sidechain_input() {
	fn render(keyed: bool) -> f32 {
		let mut rack = Rack::default();

		{
			let mut state = rack.shared.lock();
			// Only the parallel lane is audible, so the measurement is the
			// compressor's own output and not the dry bands beside it.
			for band in &mut state.bands {
				band.mute = true;
			}

			let mut compressor = ModuleState::new(0, ModuleKind::Compressor);
			compressor.params.insert("threshold".to_string(), -40.0);
			compressor.params.insert("ratio".to_string(), 10.0);
			compressor.params.insert("attack".to_string(), 1.0);
			compressor.params.insert("release".to_string(), 10.0);
			compressor
				.params
				.insert("sidechain".to_string(), if keyed { 1.0 } else { 0.0 });
			state.parallel.slots[0] = Some(compressor);
		}

		rack.shared.dirty.store(true, Ordering::Release);

		let mut energy = 0.0;

		for i in 0..8_192 {
			let time = i as f32 / 44_100.0;
			let sample = (2.0 * std::f32::consts::PI * 220.0 * time).sin() * 0.5;
			let mut frame = [sample, sample];
			// Port 1 is the sidechain and nobody is feeding it.
			let other = [[0.0_f32; 2], [0.0_f32; 2]];
			let mut context: Box<dyn ProcessContext> = Box::new(());
			rack.process(&mut frame, &other, &mut context);

			if i >= 4_096 {
				energy += frame[0] * frame[0];
			}
		}

		energy
	}

	let unkeyed = render(false);
	let keyed = render(true);

	assert!(
		keyed > unkeyed * 1.5,
		"a silent sidechain left {keyed} of the signal, the keyed input gave {unkeyed}"
	);
}

/// The tempo-controlled modules carry the product's switches, so the free-time
/// knob gives way to the beats knob in the card.
#[test]
fn the_tempo_modules_carry_the_product_switches() {
	assert_eq!(ModuleKind::Delay.switches().len(), 2);
	assert_eq!(ModuleKind::Chorus.switches().len(), 1);
	assert_eq!(ModuleKind::Phaser.switches().len(), 1);
	assert!(ModuleKind::Gain.switches().is_empty());

	let delay = ModuleKind::Delay.switches()[0];
	assert_eq!(delay.control, "tempo.sync");
	assert!(delay.on.contains(&"tempo.beats"));
	assert!(delay.off.contains(&"delay_time"));
}
