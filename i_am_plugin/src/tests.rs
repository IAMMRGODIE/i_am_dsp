//! tests

use crate::{context::ClapContext, latency::LatencyWatch};
	use i_am_dsp::prelude::BeatClock;
	use clack_plugin::events::event_types::{TransportEvent, TransportFlags};
	use clack_plugin::events::{EventFlags, EventHeader};
	use clack_plugin::utils::{BeatTime, SecondsTime};
	use i_am_dsp::{NoteEvent, ProcessContext};

	impl ClapContext {
		/// Builds a context with no transport information and an open output channel.
		fn for_test() -> Self {
			let (event_sender, _receiver) = crossbeam_channel::unbounded();
			Self {
				current_event: 0,
				events: Vec::new(),
				events_sample: 0,
				info: None,
				current_sample: 0,
				clock: BeatClock::new(48_000),
				block_start: 0,
				event_sender,
			}
		}
	}

	fn note_on(note: usize) -> NoteEvent {
		NoteEvent::NoteOn {
			channel: 0,
			note,
			velocity: 1.0,
		}
	}

	/// A host transport report with a beat timeline, as CLAP delivers it.
	fn transport(beats: f64, tempo: f64, playing: bool) -> TransportEvent {
		let mut flags = TransportFlags::HAS_TEMPO | TransportFlags::HAS_BEATS_TIMELINE;
		if playing {
			flags |= TransportFlags::IS_PLAYING;
		}

		TransportEvent {
			header: EventHeader::new_core(0, EventFlags::empty()),
			flags,
			song_pos_beats: BeatTime::from_float(beats),
			song_pos_seconds: SecondsTime::from_float(beats * 0.5),
			tempo,
			tempo_inc: 0.0,
			loop_start_beats: BeatTime::from_float(0.0),
			loop_end_beats: BeatTime::from_float(0.0),
			loop_start_seconds: SecondsTime::from_float(0.0),
			loop_end_seconds: SecondsTime::from_float(0.0),
			bar_start: BeatTime::from_float(0.0),
			bar_number: 0,
			time_signature_numerator: 4,
			time_signature_denominator: 4,
		}
	}

	/// 120 bpm at 48 kHz makes a beat exactly 24_000 samples.
	#[test]
	// The deprecated field is read on purpose: it has to stay in step with the precise one.
	#[allow(deprecated)]
	fn the_position_advances_with_every_sample() {
		let mut context = ClapContext::for_test();
		context.apply_transport(&transport(8.0, 120.0, true), 0);

		context.set_sample_offset(0);
		assert_eq!(context.infos().position_beats, Some(8.0));
		// The legacy `f32` view is filled from the same clock, so it cannot disagree.
		assert_eq!(context.infos().current_beat_number, Some(8.0));

		context.set_sample_offset(12_000);
		assert_eq!(context.infos().position_beats, Some(8.5));
		context.set_sample_offset(24_000);
		assert_eq!(context.infos().position_beats, Some(9.0));
	}

	/// A transport event inside the block is the sample-accurate version of the same report,
	/// which is how a host asks for a seek or a loop wrap to land mid-block.
	#[test]
	fn a_mid_block_transport_moves_the_playhead_at_its_own_sample() {
		let mut context = ClapContext::for_test();
		context.apply_transport(&transport(8.0, 120.0, true), 0);
		context.set_sample_offset(24_000);
		assert_eq!(context.infos().position_beats, Some(9.0));

		// The host seeks back to beat 2, at sample 30_000 of this block.
		context.apply_transport(&transport(2.0, 120.0, true), 30_000);
		context.set_sample_offset(30_000);
		assert_eq!(context.infos().position_beats, Some(2.0));

		context.set_sample_offset(54_000);
		assert_eq!(context.infos().position_beats, Some(3.0));
	}

	/// A stopped transport holds the position: a sync'd effect must not glide forward while
	/// the user is stopped, it sits on the position the host reports.
	#[test]
	fn a_stopped_transport_holds_the_position() {
		let mut context = ClapContext::for_test();
		context.apply_transport(&transport(8.0, 120.0, false), 0);

		context.set_sample_offset(0);
		assert_eq!(context.infos().position_beats, Some(8.0));
		assert!(!context.infos().playing);

		context.set_sample_offset(24_000);
		assert_eq!(context.infos().position_beats, Some(8.0));
	}

	/// A tempo with no beat timeline still says how fast time passes, but there is no
	/// position to be in phase with.
	#[test]
	#[allow(deprecated)]
	fn a_tempo_without_a_beat_timeline_reports_no_position() {
		let mut context = ClapContext::for_test();
		let mut report = transport(0.0, 100.0, true);
		report.flags.remove(TransportFlags::HAS_BEATS_TIMELINE);
		context.apply_transport(&report, 0);

		context.set_sample_offset(0);
		assert_eq!(context.infos().tempo, Some(100.0));
		assert_eq!(context.infos().position_beats, None);
		assert_eq!(context.infos().current_beat_number, None);
	}

	/// A host that ramps its tempo reports how much it changes per sample; the clock has to
	/// integrate that, not treat the tempo as a constant across the block.
	#[test]
	fn a_tempo_ramp_inside_a_block_is_integrated() {
		let mut context = ClapContext::for_test();
		let mut report = transport(0.0, 120.0, true);
		report.tempo_inc = 1.0; // one bpm per sample, so the ramp is easy to sum by hand.
		context.apply_transport(&report, 0);

		// At 48 kHz the beats a ramp covers are the sum of the rates it passed through.
		let expected = (0..1_000)
			.map(|k| (120.0 + k as f64) / (60.0 * 48_000.0))
			.sum::<f64>();

		context.set_sample_offset(1_000);
		let position = context.infos().position_beats.expect("the block has a timeline");
		assert!((position - expected).abs() < 1e-12, "{position} != {expected}");
	}

	/// The clock counts absolute samples, so a block does not restart the position at its own
	/// first sample.
	#[test]
	fn a_later_block_keeps_the_timeline_running() {
		let mut context = ClapContext::for_test();
		context.block_start = 48_000;
		context.apply_transport(&transport(10.0, 120.0, true), 48_000);

		context.set_sample_offset(0);
		assert_eq!(context.infos().position_beats, Some(10.0));
		context.set_sample_offset(24_000);
		assert_eq!(context.infos().position_beats, Some(11.0));
	}

	/// A batch is timestamped at one sample, and a generator may only see it there.
	///
	/// Regression test for the events leaking forward to every remaining sample of the
	/// block, which made [`Adsr`](i_am_dsp::prelude::Adsr) pile up one voice per sample
	/// for a single note-on.
	#[test]
	fn a_batch_is_only_visible_at_its_own_sample() {
		let mut context = ClapContext::for_test();
		context.events_sample = 4;
		context.events = vec![note_on(60)];

		let mut visible = Vec::new();
		for sample in 0..512usize {
			context.current_sample = sample;
			if !context.events().is_empty() {
				visible.push(sample);
			}
		}

		assert_eq!(visible, vec![4]);
	}

	/// The consuming accessor has to close on the same sample boundary as
	/// [`ProcessContext::events`].
	#[test]
	fn next_event_stops_at_the_sample_boundary() {
		let mut context = ClapContext::for_test();
		context.events_sample = 7;
		context.events = vec![note_on(60), note_on(64)];

		context.current_sample = 7;
		assert_eq!(context.next_event(), Some(note_on(60)));
		assert_eq!(context.next_event(), Some(note_on(64)));
		assert_eq!(context.next_event(), None);

		context.current_sample = 8;
		assert_eq!(context.next_event(), None);
	}

	/// A generator that clears the events itself still sees an empty context afterwards.
	#[test]
	fn clear_events_hides_the_batch_at_once() {
		let mut context = ClapContext::for_test();
		context.events_sample = 3;
		context.current_sample = 3;
		context.events = vec![note_on(60)];

		context.clear_events();

		assert!(context.events().is_empty());
		assert_eq!(context.next_event(), None);
	}
	/// The audio thread and the main thread share the latency: the audio thread knows it, and
	/// the main thread is the one that may tell the host.
	#[test]
	fn a_latency_change_is_reported_once() {
		let mut watch = LatencyWatch::new(480);
		let publisher = watch.publisher();

		assert_eq!(watch.current(), 480, "the latency before anything is processed");
		assert_eq!(watch.take_change(), None, "the host has just been told");

		publisher.publish(960);
		assert_eq!(watch.current(), 960);
		assert_eq!(watch.take_change(), Some(960));
		// Told once: the host re-queries when it is told, so repeating it is noise.
		assert_eq!(watch.take_change(), None);

		// Publishing the same latency again is not a change either.
		publisher.publish(960);
		assert_eq!(watch.take_change(), None);

		publisher.publish(240);
		assert_eq!(watch.take_change(), Some(240));
	}
