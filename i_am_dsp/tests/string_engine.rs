//! End to end tests for the physical string instrument.

use i_am_dsp::generators::string_engine::{PRESETS, StringInstrument};
use i_am_dsp::{NoteEvent, ProcessContext, ProcessInfos, prelude::*};

const FS: usize = 48_000;

struct TestContext {
    info: ProcessInfos,
    events: Vec<NoteEvent>,
}

impl TestContext {
    fn new() -> Self {
        let mut info = ProcessInfos::default();
        info.sample_rate = FS;
        info.playing = true;
        Self {
            info,
            events: Vec::new(),
        }
    }
}

impl ProcessContext for TestContext {
    fn infos(&self) -> &ProcessInfos {
        &self.info
    }

    fn next_event(&mut self) -> Option<NoteEvent> {
        self.events.pop()
    }

    fn send_event(&mut self, event: NoteEvent) {
        self.events.push(event);
    }

    fn clear_events(&mut self) {
        self.events.clear();
    }

    fn events(&self) -> &[NoteEvent] {
        &self.events
    }
}

fn render(instrument: &mut StringInstrument, context: &mut Box<dyn ProcessContext>, samples: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(samples);
    for _ in 0..samples {
        out.push(instrument.generate(context)[0]);
    }
    out
}

fn rms(samples: &[f32]) -> f32 {
    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len().max(1) as f32).sqrt()
}

/// Estimate a fundamental by autocorrelation, on the mean removed signal.
fn estimate_frequency(samples: &[f32], low: f32, high: f32) -> f32 {
    let fs = FS as f32;
    let mean = samples.iter().sum::<f32>() / samples.len() as f32;
    let min_lag = (fs / high).floor() as usize;
    let max_lag = (fs / low).ceil() as usize;
    let mut best = (min_lag, f32::NEG_INFINITY);
    for lag in min_lag..=max_lag {
        let count = samples.len() - lag;
        let mut sum = 0.0f64;
        for i in 0..count {
            sum += ((samples[i] - mean) as f64) * ((samples[i + lag] - mean) as f64);
        }
        let value = (sum / count as f64) as f32;
        if value > best.1 {
            best = (lag, value);
        }
    }
    fs / best.0 as f32
}

#[test]
fn every_preset_makes_a_bounded_sound() {
    for (index, preset) in PRESETS.iter().enumerate() {
        let mut instrument = StringInstrument::new(FS);
        instrument.select_preset(index as i32);
        let mut context: Box<dyn ProcessContext> = Box::new(TestContext::new());
        context.send_event(NoteEvent::NoteOn {
            channel: 0,
            note: 57,
            velocity: 1.0,
        });
        context.send_event(NoteEvent::NoteOn {
            channel: 0,
            note: 64,
            velocity: 0.8,
        });
        let audio = render(&mut instrument, &mut context, FS * 2);
        assert!(
            audio.iter().all(|s| s.is_finite()),
            "{} produced a non finite sample",
            preset.name()
        );
        assert!(
            audio.iter().all(|s| s.abs() < 4.0),
            "{} produced a runaway sample",
            preset.name()
        );
        let level = rms(&audio[FS / 4..]);
        assert!(
            level > 1e-4,
            "{} was silent (rms {level})",
            preset.name()
        );
    }
}

#[test]
fn a_released_note_decays_away() {
    let mut instrument = StringInstrument::new(FS);
    instrument.select_preset(0);
    let mut context: Box<dyn ProcessContext> = Box::new(TestContext::new());
    context.send_event(NoteEvent::NoteOn {
        channel: 0,
        note: 57,
        velocity: 1.0,
    });
    let held = render(&mut instrument, &mut context, FS);
    context.send_event(NoteEvent::NoteOff {
        channel: 0,
        note: 57,
        velocity: 0.0,
    });
    let released = render(&mut instrument, &mut context, FS * 3);
    let before = rms(&held[held.len() / 2..]);
    let after = rms(&released[released.len() * 3 / 4..]);
    assert!(before > 1e-4, "nothing was heard before the release: {before}");
    assert!(
        after < 0.15 * before,
        "the damper did not stop the note: {before} then {after}"
    );
}

#[test]
fn the_library_note_numbering_is_an_octave_below_midi() {
    // The library puts A4 at note 57; the engine speaks standard MIDI, where
    // A4 is 69. An octave of error here would be very audible, so it is worth
    // one test.
    let mut instrument = StringInstrument::new(FS);
    instrument.select_preset(0);
    let mut context: Box<dyn ProcessContext> = Box::new(TestContext::new());
    context.send_event(NoteEvent::NoteOn {
        channel: 0,
        note: 57,
        velocity: 1.0,
    });
    let audio = render(&mut instrument, &mut context, FS * 3);
    let segment = &audio[FS / 2..];
    let fundamental = estimate_frequency(segment, 380.0, 520.0);
    assert!(
        (fundamental - 440.0).abs() < 12.0,
        "library note 57 sounded at {fundamental} Hz, expected 440 Hz"
    );
}

#[test]
fn fitting_the_filter_for_a_new_note_stays_within_an_audio_budget() {
    // The first note at a given pitch has to fit a dispersion allpass filter,
    // and that happens on the audio thread. It is cached afterwards, but the
    // first one still has to be short enough not to be heard as a dropout.
    let mut instrument = StringInstrument::new(FS);
    instrument.select_preset(0);
    let mut context: Box<dyn ProcessContext> = Box::new(TestContext::new());
    // Warm the engine up so the measurement is the fit and not the allocation.
    context.send_event(NoteEvent::NoteOn {
        channel: 0,
        note: 57,
        velocity: 1.0,
    });
    render(&mut instrument, &mut context, 1024);

    let mut worst = std::time::Duration::ZERO;
    for note in [40usize, 52, 64, 76, 88] {
        context.send_event(NoteEvent::NoteOn {
            channel: 0,
            note,
            velocity: 1.0,
        });
        let start = std::time::Instant::now();
        render(&mut instrument, &mut context, 64);
        worst = worst.max(start.elapsed());
    }
    println!("worst cold note onset: {worst:?}");
    assert!(
        worst < std::time::Duration::from_millis(30),
        "a cold note onset took {worst:?}"
    );
}

#[test]
fn switching_preset_reloads_the_controls() {
    let mut instrument = StringInstrument::new(FS);
    instrument.select_preset(0);
    assert_eq!(instrument.preset_name(), "Piano");
    let piano_strings = instrument.strings;
    instrument.select_preset(1);
    assert_eq!(instrument.preset_name(), "Guitar");
    assert_ne!(piano_strings, instrument.strings);
    assert_eq!(instrument.excitation, 0, "the guitar is plucked");
}#[test]
fn the_sustain_pedal_follows_midi_cc64() {
    let pedal_tail = |cc: f32| {
        let mut instrument = StringInstrument::new(FS);
        instrument.select_preset(0);
        let mut context: Box<dyn ProcessContext> = Box::new(TestContext::new());
        context.send_event(NoteEvent::MidiCC {
            channel: 0,
            cc: 64,
            value: cc,
        });
        context.send_event(NoteEvent::NoteOn {
            channel: 0,
            note: 57,
            velocity: 0.9,
        });
        let held = render(&mut instrument, &mut context, FS / 2);
        context.send_event(NoteEvent::NoteOff {
            channel: 0,
            note: 57,
            velocity: 0.0,
        });
        let released = render(&mut instrument, &mut context, FS * 2);
        (rms(&held), rms(&released[FS..]))
    };

    let (_, up) = pedal_tail(0.0);
    let (_, down) = pedal_tail(1.0);
    assert!(
        down > 4.0 * up,
        "CC64 did not hold the note: {up} up, {down} down"
    );
}

#[test]
fn the_sympathetic_strings_can_be_turned_off() {
    let tail = |sympathetic: f32| {
        let mut instrument = StringInstrument::new(FS);
        instrument.select_preset(0);
        instrument.sympathetic = sympathetic;
        let mut context: Box<dyn ProcessContext> = Box::new(TestContext::new());
        context.send_event(NoteEvent::NoteOn {
            channel: 0,
            note: 60,
            velocity: 1.0,
        });
        let audio = render(&mut instrument, &mut context, FS * 3);
        rms(&audio[FS * 2..])
    };
    let without = tail(0.0);
    let with = tail(0.8);
    assert!(
        with > 1.5 * without,
        "the sympathetic strings made no difference: {without} then {with}"
    );
}
