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

/// The energy of `samples` at `frequency`, by the Goertzel recurrence.
fn goertzel(samples: &[f32], frequency: f32) -> f32 {
    let omega = std::f32::consts::TAU * frequency / FS as f32;
    let coefficient = 2.0 * omega.cos();
    let (mut s1, mut s2) = (0.0f32, 0.0f32);
    for sample in samples {
        let s0 = sample + coefficient * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    (s1 * s1 + s2 * s2 - coefficient * s1 * s2).max(0.0)
}

/// Where in `low..high` the sound actually sits.
///
/// This used to be an autocorrelation, and on this instrument that was simply
/// wrong. A piano note is three detuned strings driving a body with resonances
/// of its own, and on a decaying inharmonic tone the correlation picked a lag
/// that was not the period - measured, it called a perfectly correct 659 Hz note
/// 585 Hz, and the failure looked exactly like a transpose bug. Scanning the
/// spectrum asks the question directly and needs no assumptions about the
/// waveform.
fn estimate_frequency(samples: &[f32], low: f32, high: f32) -> f32 {
    let mut windowed = samples.to_vec();
    let last = (windowed.len() - 1) as f32;
    for (index, sample) in windowed.iter_mut().enumerate() {
        let t = index as f32 / last;
        *sample *= 0.5 - 0.5 * (std::f32::consts::TAU * t).cos();
    }
    let mut best = low;
    let mut best_energy = 0.0f32;
    let steps = 400;
    for step in 0..steps {
        let probe = low * (high / low).powf(step as f32 / (steps - 1) as f32);
        let energy = goertzel(&windowed, probe);
        if energy > best_energy {
            best_energy = energy;
            best = probe;
        }
    }
    best
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

/// The pitch library note 57 sounds at, with the panel transposed by `shift`.
fn sounding_pitch(shift: f32) -> f32 {
    let mut instrument = StringInstrument::new(FS);
    instrument.select_preset(0);
    instrument.transpose = shift;
    let mut context: Box<dyn ProcessContext> = Box::new(TestContext::new());
    context.send_event(NoteEvent::NoteOn {
        channel: 0,
        note: 57,
        velocity: 1.0,
    });
    let audio = render(&mut instrument, &mut context, FS * 3);
    // Two semitones either side is wide enough to see a transposition error and
    // narrow enough that only the fundamental is in range.
    let low = 440.0 * 2f32.powf((shift - 2.0) / 12.0);
    let high = 440.0 * 2f32.powf((shift + 2.0) / 12.0);
    estimate_frequency(&audio[FS / 2..], low, high)
}

#[test]
fn the_library_note_numbering_is_an_octave_below_midi() {
    // The library puts A4 at note 57; the engine speaks standard MIDI, where
    // A4 is 69. An octave of error here would be very audible, so it is worth
    // one test.
    //
    // At concert pitch, which is where the presets sit. A transpose control that
    // moved the note numbering itself would make the whole library disagree with
    // itself about what a note number means, so the presets leave it alone and
    // the panel is the only thing that ever moves it.
    let fundamental = sounding_pitch(0.0);
    assert!(
        (fundamental - 440.0).abs() < 12.0,
        "library note 57 sounded at {fundamental} Hz, expected 440 Hz"
    );
}

/// And the panel really does ship an octave down, which is the whole point of
/// the control: the model is only convincing in the middle of its range.
#[test]
fn the_transpose_control_moves_the_pitch() {
    let down = sounding_pitch(-12.0);
    assert!(
        (down - 220.0).abs() < 6.0,
        "transposed down an octave, note 57 sounded at {down} Hz, expected 220"
    );
    let up = sounding_pitch(7.0);
    let wanted = 440.0 * 2f32.powf(7.0 / 12.0);
    assert!(
        (up - wanted).abs() < 12.0,
        "transposed up a fifth, note 57 sounded at {up} Hz, expected {wanted}"
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
