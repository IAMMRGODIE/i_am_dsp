//! What the node builders do to the sound: the voices, the sources and the meters.

use super::*;
use super::oscillator::{ShapeTable, VoiceOsc};
use super::table::{WarpedTable, apply_params};

#[cfg(test)]
mod voice_tests {
	use super::*;

	fn sine() -> ShapeTable {
		ShapeTable {
			shape: Arc::new(AtomicU32::new(0)),
		}
	}

	fn shared(value: f32) -> Arc<AtomicU32> {
		Arc::new(AtomicU32::new(value.to_bits()))
	}

	/// The envelope's Voice page mixes value types; each has to reach the DSP.
	#[test]
	fn an_envelope_takes_its_voice_parameters() {
		use i_am_dsp::prelude::Value as ParamValue;

		let mut node = AdsrNode::new(48_000);

		assert!(node.install(Box::new(TableOsc(sine()))));

		node.set_param("unisons", 4.0);
		node.set_param("random_phase_by_channel", 1.0);

		let inner = node.inner.as_ref().expect("installed");
		let parameters = inner.get_parameters();
		let unisons = parameters
			.iter()
			.find(|parameter| parameter.identifier == "unisons")
			.expect("unisons");
		let channel = parameters
			.iter()
			.find(|parameter| parameter.identifier == "random_phase_by_channel")
			.expect("phase per channel");

		match unisons.value {
			ParamValue::Int { value, .. } => assert_eq!(value, 4),
			_ => panic!("unisons is not an int"),
		}

		match channel.value {
			ParamValue::Bool(value) => assert!(value),
			_ => panic!("phase per channel is not a bool"),
		}
	}

	fn play(kind: ModKind, depth: f32) -> f32 {
		let oscillator = VoiceOsc::new(
			kind,
			Box::new(sine()),
			Box::new(sine()),
			shared(depth),
			shared(1.0),
		);

		oscillator.play_at(440.0, 0.0, [0.25, 0.25])[0]
	}

	/// Ring at zero depth is the carrier; at full depth it is the product.
	#[test]
	fn the_ring_modulation_multiplies_both_tables() {
		let carrier = sine().sample(0.25, 0);

		assert!((play(ModKind::Rm, 0.0) - carrier).abs() < 1e-5);
		assert!((play(ModKind::Rm, 1.0) - carrier * carrier).abs() < 1e-5);
	}

	/// AM keeps the carrier and scales it by the modulator.
	#[test]
	fn the_amplitude_modulation_scales_the_carrier() {
		let carrier = sine().sample(0.25, 0);

		assert!((play(ModKind::Am, 0.0) - carrier).abs() < 1e-5);
		assert!((play(ModKind::Am, 1.0) - carrier * (1.0 + carrier)).abs() < 1e-5);
	}

	/// A product table is a snapshot of the product's fields, so the node keeps the
	/// values and the table follows them through sync: a knob move reaches the note
	/// that is already sounding.
	#[test]
	fn a_product_table_follows_its_node() {
		let kind = crate::graph::node("table:additive").expect("the kind");
		let params: BTreeMap<&'static str, f32> = kind
			.params
			.iter()
			.map(|param| (param.name, param.default))
			.collect();

		let mut node = ProductTableNode::new(
			|| {
				Box::new(OscTable::<2, _>(AdditiveOsc::<BendedSawGen, 64, 2>::new(
					BendedSawGen::default(),
					32,
				)))
			},
			&params,
		);

		let Some(ConstructorSpec::Table(mut table)) = node.constructor_spec() else {
			panic!("a product table is a table");
		};

		let snapshot = |table: &dyn WaveTable| -> Vec<f32> {
			table
				.get_parameters()
				.iter()
				.map(|parameter| match parameter.value.clone().to_set_value() {
					SetValue::Float(value) => value,
					SetValue::Int(value) => value as f32,
					SetValue::Bool(value) => value as u8 as f32,
					_ => f32::NAN,
				})
				.collect()
		};

		let before = snapshot(table.as_ref());

		// Every parameter the card shows, moved somewhere else.
		for param in kind.params.iter() {
			node.set_param(param.name, param.default + 0.25);
		}

		table.sync();

		let after = snapshot(table.as_ref());

		assert_ne!(
			before,
			after,
			"the table kept the values it was built with: {} parameters, none moved",
			kind.params.len()
		);
	}

	/// The card's ids carry the product's path; the table is the oscillator
	/// alone, so the path has to come off before the write.
	#[test]
	fn a_table_strips_the_processor_path() {
		use i_am_dsp::prelude::{Value as ParamValue, WaveSlot};

		let slot = Arc::new(std::sync::Mutex::new(WaveSlot::default()));
		let mut table = OscTable::<2, _>(WaveFramesOsc::with_slot(slot));
		let mut params = BTreeMap::new();

		params.insert("oscillator.smooth_factor", 1.0);
		apply_params(&mut table, &params);

		let found = table
			.get_parameters()
			.into_iter()
			.find(|parameter| parameter.identifier == "smooth_factor")
			.expect("the parameter");

		match found.value {
			ParamValue::Float { value, .. } => assert!((value - 1.0).abs() < 1e-6),
			_ => panic!("the smooth factor is not a float"),
		}
	}

	/// A warp reads the carrier where the phase function points.
	#[test]
	fn a_warp_reads_where_the_phase_points() {
		let table = WarpedTable {
			wave: Box::new(sine()),
			mode: WarpMode::Mirror,
			amount: shared(1.0),
			curve: None,
		};

		// A full fold maps a quarter cycle to a half.
		assert!((table.sample(0.25, 0) - sine().sample(0.5, 0)).abs() < 1e-5);
	}

	/// A remap with no curve is the table itself.
	#[test]
	fn a_remap_without_a_curve_reads_straight_through() {
		let table = WarpedTable {
			wave: Box::new(sine()),
			mode: WarpMode::Remap,
			amount: shared(1.0),
			curve: None,
		};

		assert!((table.sample(0.3, 0) - sine().sample(0.3, 0)).abs() < 1e-6);
	}

	/// The remap warp lerps towards the curve table it is given.
	#[test]
	fn a_remap_warp_follows_its_curve() {
		let curve = ShapeTable {
			shape: Arc::new(AtomicU32::new(0)),
		};
		let table = WarpedTable {
			wave: Box::new(sine()),
			mode: WarpMode::Remap,
			amount: shared(1.0),
			curve: Some(Box::new(ShapeTable {
				shape: Arc::new(AtomicU32::new(0)),
			})),
		};

		// At full amount the read follows the curve table.
		let expected = sine().sample(curve.sample(0.3, 0), 0);

		assert!((table.sample(0.3, 0) - expected).abs() < 1e-5);
	}

	/// PM and FM at zero depth are the carrier too.
	#[test]
	fn the_phase_modulations_at_zero_depth_are_the_carrier() {
		let carrier = sine().sample(0.25, 0);

		assert!((play(ModKind::Pm, 0.0) - carrier).abs() < 1e-5);
		assert!((play(ModKind::Fm, 0.0) - carrier).abs() < 1e-5);
	}
}

#[cfg(test)]
mod source_tests {
	use super::*;

	fn infos() -> i_am_dsp::ProcessInfos {
		i_am_dsp::ProcessInfos::default()
	}

	fn block(rate: usize) -> BlockContext {
		BlockContext {
			frames: BLOCK,
			rate,
			block: 0,
			events: Vec::new(),
			context: Box::new(()),
		}
	}

	fn run(node: &mut SourceNode, context: &mut BlockContext, input: f32) -> f32 {
		let mut outputs = [Buffer::Control(0.0)];
		let audio = AudioBlock {
			frames: BLOCK,
			channels: [[input; BLOCK]; 2],
		};
		let inputs = [Buffer::Audio(audio)];
		node.process(context, &inputs, &mut outputs);

		match outputs[0] {
			Buffer::Control(value) => value,
			_ => panic!("a control source writes a control buffer"),
		}
	}

	#[test]
	fn a_midi_source_says_where_its_input_sits() {
		let rate = 48_000;
		let mut node = SourceNode::new(SourceKind::Midi, rate, &BTreeMap::new(), &[]);
		let mut context = block(rate);

		context.events.push(NoteEvent::NoteOn {
			channel: 0,
			note: 60,
			velocity: 0.8,
		});
		run(&mut node, &mut context, 0.0);

		assert!((node.midi_position().unwrap() - 60.0 / 127.0).abs() < 1e-6);

		let mut params = BTreeMap::new();
		params.insert("mode", 1.0);

		let mut node = SourceNode::new(SourceKind::Midi, rate, &params, &[]);
		let mut context = block(rate);

		context.events.push(NoteEvent::NoteOn {
			channel: 0,
			note: 60,
			velocity: 0.8,
		});
		run(&mut node, &mut context, 0.0);

		assert!((node.midi_position().unwrap() - 0.8).abs() < 1e-6);
	}

	#[test]
	fn a_midi_note_bend_moves_the_output() {
		let rate = 48_000;
		let points = [
			crate::graph::Point {
				time: 0.0,
				value: 0.0,
				bend: 4.0,
			},
			crate::graph::Point {
				time: 1.0,
				value: 1.0,
				bend: 0.0,
			},
		];
		let mut node = SourceNode::new(SourceKind::Midi, rate, &BTreeMap::new(), &points);
		let mut context = block(rate);

		context.events.push(NoteEvent::NoteOn {
			channel: 0,
			note: 60,
			velocity: 0.8,
		});
		let out = run(&mut node, &mut context, 0.0);

		assert!(out > 0.0 && out < 60.0 / 127.0, "{out}");
	}

	#[test]
	fn a_midi_velocity_bend_moves_the_output() {
		let rate = 48_000;
		let mut params = BTreeMap::new();
		params.insert("mode", 1.0);

		let points = [
			crate::graph::Point {
				time: 0.0,
				value: 0.0,
				bend: 4.0,
			},
			crate::graph::Point {
				time: 1.0,
				value: 1.0,
				bend: 0.0,
			},
		];
		let mut node = SourceNode::new(SourceKind::Midi, rate, &params, &[]);

		node.set_velocity_points(&points);

		let mut context = block(rate);
		context.events.push(NoteEvent::NoteOn {
			channel: 0,
			note: 60,
			velocity: 0.8,
		});
		let out = run(&mut node, &mut context, 0.0);

		assert!(out > 0.0 && out < 0.8, "{out}");
	}

	#[test]
	fn a_parameter_takes_the_value_type_its_dsp_declares() {
		assert_eq!(
			super::set_value(super::ParamKind::Int, 2.6),
			SetValue::Int(3)
		);
		assert_eq!(
			super::set_value(super::ParamKind::Bool, 0.0),
			SetValue::Bool(false)
		);
		assert_eq!(
			super::set_value(super::ParamKind::Float, 0.5),
			SetValue::Float(0.5)
		);
	}

	#[test]
	fn an_lfo_node_runs_the_same_lfo_as_a_rack_slot() {
		let rate = 48_000;
		let mut params = BTreeMap::new();
		params.insert("rate", 1.0);
		params.insert("shape", 1.0);

		let mut node = SourceNode::new(SourceKind::Lfo, rate, &params, &[]);
		let settings = node.settings.clone();

		let mut runtime = SourceRuntime::from_settings(SourceKind::Lfo, false, settings, rate, 0);
		let from_node = run(&mut node, &mut block(rate), 0.0);

		for _ in 0..BLOCK {
			runtime.advance(0.0, &infos(), rate, &[]);
		}

		assert!(
			(from_node - runtime.value()).abs() < 1e-6,
			"{from_node} vs {}",
			runtime.value()
		);
	}

	#[test]
	fn a_random_node_runs_the_same_stream_as_a_rack_slot() {
		let rate = 48_000;
		let mut params = BTreeMap::new();
		params.insert("rate", 4.0);
		params.insert("smooth", 20.0);

		let mut node = SourceNode::new(SourceKind::Random, rate, &params, &[]);
		let settings = node.settings.clone();

		let mut runtime =
			SourceRuntime::from_settings(SourceKind::Random, false, settings, rate, 0);
		let from_node = run(&mut node, &mut block(rate), 0.0);

		for _ in 0..BLOCK {
			runtime.advance(0.0, &infos(), rate, &[]);
		}

		assert!(
			(from_node - runtime.value()).abs() < 1e-6,
			"{from_node} vs {}",
			runtime.value()
		);
	}

	#[test]
	fn an_envelope_node_follows_the_same_follower_as_a_rack_slot() {
		let rate = 48_000;
		let mut params = BTreeMap::new();
		params.insert("attack", 10.0);
		params.insert("release", 100.0);

		let mut node = SourceNode::new(SourceKind::Envelope, rate, &params, &[]);
		let settings = node.settings.clone();

		let mut runtime =
			SourceRuntime::from_settings(SourceKind::Envelope, false, settings, rate, 0);
		let from_node = run(&mut node, &mut block(rate), 1.0);

		for _ in 0..BLOCK {
			runtime.advance(1.0, &infos(), rate, &[]);
		}

		assert!(
			(from_node - runtime.value()).abs() < 1e-6,
			"{from_node} vs {}",
			runtime.value()
		);
	}
}

#[cfg(test)]
mod meter_tests {
	use super::*;

	fn block(rate: usize) -> BlockContext {
		BlockContext {
			frames: BLOCK,
			rate,
			block: 0,
			events: Vec::new(),
			context: Box::new(()),
		}
	}

	/// The OTT's card draws the bands its effect measures, so the node has to be
	/// built over the same handle the page route reads. A meter made inside the
	/// build is a picture nothing could ever see, which is what the card showed.
	#[test]
	fn an_ott_node_publishes_the_bands_its_card_draws() {
		let rate = 48_000;
		let slots = crate::Slots::default();
		let mut node =
			make_with("mod:ott", 5, rate, &BTreeMap::new(), &[], &[], &slots).expect("an ott node");
		let mut context = block(rate);
		let audio = AudioBlock {
			frames: BLOCK,
			channels: [[0.5; BLOCK]; 2],
		};

		for _ in 0..4 {
			let inputs = [Buffer::Audio(audio)];
			let mut outputs = [Buffer::Audio(AudioBlock::silent(BLOCK))];

			node.process(&mut context, &inputs, &mut outputs);
		}

		let Some(crate::BandMeter::Ott(meters)) = slots.bands(5) else {
			panic!("the ott node published no meter for its card");
		};
		let bands = meters.get();

		assert!(
			bands.iter().any(|band| *band != 0.0),
			"the meter never left zero: {bands:?}"
		);
	}
}
