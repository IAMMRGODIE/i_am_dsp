//! A physically modelled string instrument.
//!
//! This is the thin adapter that makes `i_am_string` playable from the rest of
//! the library: it turns [NoteEvent]s into string excitations and exposes the
//! player controls as library parameters.

use i_am_dsp_derive::Parameters;

use i_am_string::prelude as strings;

use crate::{Generator, NoteEvent};

/// The presets, in menu order.
pub const PRESETS: [strings::Preset; 4] = strings::Preset::ALL;

/// The contact laws, in menu order.
pub const EXCITATIONS: [strings::Excitation; 3] = [
    strings::Excitation::Pluck,
    strings::Excitation::Hammer,
    strings::Excitation::Bow,
];

/// The string models, in menu order.
pub const CORES: [strings::Core; 2] = strings::Core::ALL;

/// The library's note numbering puts A4 at 57, the MIDI standard puts it at 69.
const NOTE_OFFSET: i32 = 12;

fn to_midi(note: usize) -> u8 {
    (note as i32 + NOTE_OFFSET).clamp(0, 127) as u8
}

fn excitation_index(excitation: strings::Excitation) -> i32 {
    EXCITATIONS
        .iter()
        .position(|candidate| *candidate == excitation)
        .unwrap_or(0) as i32
}

fn core_index(core: strings::Core) -> i32 {
    CORES
        .iter()
        .position(|candidate| *candidate == core)
        .unwrap_or(0) as i32
}

/// A polyphonic physical string instrument.
#[derive(Parameters)]
pub struct StringInstrument {
    #[skip]
    engine: strings::StringEngine,
    #[skip]
    sample_rate: usize,
    /// Which instrument is being played.
    #[range(min = 0, max = 3)]
    pub preset: i32,
    /// How the string is excited: pluck, hammer or bow.
    ///
    /// This is a separate choice from the instrument, and every combination
    /// works. A guitar played with a hammer is a piano; a cello plucked is a
    /// pizzicato; a piano played with a bow is a curiosity. Some of them need
    /// the controls moved before they sound like anything - a light string
    /// struck by a hard hammer is a click - but none of them is silent.
    #[range(min = 0, max = 2)]
    pub excitation: i32,
    /// How the string itself is modelled. Not a quality setting: the waveguide
    /// is cheap and its low partials are exact, the modal bank is exactly in
    /// tune at any stiffness. Which one sounds like the instrument differs.
    #[range(min = 0, max = 1)]
    pub core: i32,
    /// The output gain.
    #[range(min = 0.0, max = 2.0)]
    pub gain: f32,
    /// Scales the frequency independent decay. Larger damps faster.
    #[range(min = 0.1, max = 4.0)]
    pub damping: f32,
    /// Scales the frequency dependent decay. Larger is darker.
    #[range(min = 0.1, max = 4.0)]
    pub brightness: f32,
    /// Scales the string stiffness, and so its inharmonicity.
    #[range(min = 0.0, max = 4.0)]
    pub inharmonicity: f32,
    /// Where the string is excited, as a fraction of its length.
    #[range(min = 0.01, max = 0.5)]
    #[logarithmic]
    pub position: f32,
    /// How many strings sound per note.
    #[range(min = 1, max = 3)]
    pub strings: i32,
    /// The unison detune, in cents.
    #[range(min = 0.0, max = 20.0)]
    pub detune_cents: f32,
    /// Hammer felt hardness, or plectrum hardness.
    #[range(min = 0.0, max = 1.0)]
    pub hardness: f32,
    /// Bow force.
    #[range(min = 0.0, max = 1.0)]
    pub bow_force: f32,
    /// Bow speed.
    #[range(min = 0.0, max = 1.0)]
    pub bow_speed: f32,
    /// How much soundboard colouring is mixed in.
    #[range(min = 0.0, max = 1.0)]
    pub body: f32,
    /// How loudly the undamped strings answer the played ones.
    #[range(min = 0.0, max = 1.0)]
    pub sympathetic: f32,
    /// The sustain pedal, from 0 (up) to 1 (fully down).
    ///
    /// Anything in between is a half pedal. On a keyboard this follows MIDI
    /// CC64; the slider is here for the computer keyboard.
    #[range(min = 0.0, max = 1.0)]
    pub pedal: f32,
    /// How far the instrument is shifted, in semitones.
    ///
    /// The model is only convincing over about two octaves in the middle of its
    /// range, so the presets start an octave down and the middle of the keyboard
    /// lands where the model is strongest.
    #[range(min = -24.0, max = 12.0)]
    pub transpose: f32,
    /// How fast the left hand shakes the string, in hertz.
    #[range(min = 0.0, max = 12.0)]
    pub vibrato_rate: f32,
    /// How far either side of the note the hand bends it, in cents.
    #[range(min = 0.0, max = 120.0)]
    pub vibrato_depth: f32,
    /// How long the note sounds before the hand starts to shake.
    #[range(min = 0.0, max = 1.0)]
    pub vibrato_delay: f32,
}

impl StringInstrument {
    /// Create an instrument that renders at `sample_rate` hertz.
    pub fn new(sample_rate: usize) -> Self {
        let engine = strings::StringEngine::new(sample_rate as f32);
        let mut instrument = Self {
            engine,
            sample_rate,
            preset: 0,
            excitation: 0,
            core: 0,
            gain: 1.0,
            damping: 1.0,
            brightness: 1.0,
            inharmonicity: 1.0,
            position: 0.16,
            strings: 1,
            detune_cents: 0.0,
            hardness: 0.5,
            bow_force: 0.4,
            bow_speed: 0.35,
            body: 0.6,
            sympathetic: 0.6,
            pedal: 0.0,
            transpose: -12.0,
            vibrato_rate: 5.4,
            vibrato_depth: 12.0,
            vibrato_delay: 0.12,
        };
        instrument.load_preset(0);
        instrument
    }

    /// The underlying engine, for analysis and for tests.
    pub fn engine(&self) -> &strings::StringEngine {
        &self.engine
    }

    /// Select a preset and load its defaults into the controls.
    pub fn select_preset(&mut self, index: i32) {
        self.load_preset(index);
    }

    /// The name of the currently selected preset.
    pub fn preset_name(&self) -> &'static str {
        PRESETS[self.preset.clamp(0, PRESETS.len() as i32 - 1) as usize].name()
    }

    /// Copy a preset's defaults into both the engine and the controls.
    fn load_preset(&mut self, index: i32) {
        let index = index.clamp(0, PRESETS.len() as i32 - 1);
        self.preset = index;
        self.engine.set_preset(PRESETS[index as usize]);
        let defaults = self.engine.params().clone();
        self.excitation = excitation_index(defaults.excitation);
        self.core = core_index(defaults.core);
        self.gain = defaults.gain;
        self.damping = defaults.damping;
        self.brightness = defaults.brightness;
        self.inharmonicity = defaults.inharmonicity;
        self.position = defaults.position;
        self.strings = defaults.strings as i32;
        self.detune_cents = defaults.detune_cents;
        self.hardness = defaults.hardness;
        self.bow_force = defaults.bow_force;
        self.bow_speed = defaults.bow_speed;
        self.body = defaults.body;
        self.sympathetic = defaults.sympathetic;
        self.pedal = 0.0;
        // The transpose is a choice about where the instrument sits, like the
        // core, so it survives a change of preset.
        let transpose = self.transpose;
        self.transpose = if transpose == -12.0 { defaults.transpose } else { transpose };
        self.vibrato_rate = defaults.vibrato_rate;
        self.vibrato_depth = defaults.vibrato_depth;
        self.vibrato_delay = defaults.vibrato_delay;
    }

    /// Push the controls into the engine, rebuilding only what changed.
    fn sync(&mut self) {
        let wanted = strings::EngineParams {
            core: CORES[self.core.clamp(0, 1) as usize],
            excitation: EXCITATIONS[self.excitation.clamp(0, 2) as usize],
            gain: self.gain,
            damping: self.damping,
            brightness: self.brightness,
            inharmonicity: self.inharmonicity,
            position: self.position,
            strings: self.strings.clamp(1, 3) as usize,
            detune_cents: self.detune_cents,
            hardness: self.hardness,
            bow_force: self.bow_force,
            bow_speed: self.bow_speed,
            body: self.body,
            sympathetic: self.sympathetic,
            transpose: self.transpose,
            vibrato_rate: self.vibrato_rate,
            vibrato_depth: self.vibrato_depth,
            vibrato_delay: self.vibrato_delay,
            vibrato_onset: 0.14,
        };
        if self.engine.params() != &wanted {
            // This also drops the cached dispersion filters, which is why it is
            // only done when something really changed.
            self.engine.apply_params(wanted);
        }
        self.engine.set_pedal(self.pedal);
    }
}

impl Generator<2> for StringInstrument {
    fn generate(&mut self, process_context: &mut Box<dyn crate::ProcessContext>) -> [f32; 2] {
        let sample_rate = process_context.infos().sample_rate;
        if sample_rate > 0 && sample_rate != self.sample_rate {
            self.sample_rate = sample_rate;
            self.engine = strings::StringEngine::new(sample_rate as f32);
            let preset = self.preset;
            self.load_preset(preset);
        }

        // The host hands the same event list to every sample of a block, so
        // the events have to be taken and cleared; processing them in place
        // would restart the note on every single sample.
        let events = process_context.events().to_vec();
        process_context.clear_events();
        for event in &events {
            match event {
                NoteEvent::NoteOn { note, velocity, .. } => {
                    self.engine.note_on(to_midi(*note), *velocity);
                }
                NoteEvent::NoteOff { note, .. } => {
                    self.engine.note_off(to_midi(*note));
                }
                NoteEvent::Stop { note, .. } => {
                    self.engine.note_off(to_midi(*note));
                }
                NoteEvent::ImmediateStop => {
                    self.engine.all_notes_off();
                }
                NoteEvent::MidiCC { cc: 64, value, .. } => {
                    // The sustain pedal, as every keyboard sends it.
                    self.pedal = value.clamp(0.0, 1.0);
                }
                _ => {},
            }
        }

        self.sync();
        self.engine.next_frame()
    }

    #[cfg(feature = "real_time_demo")]
    fn name(&self) -> &str {
        "Modelled Strings"
    }

    #[cfg(feature = "real_time_demo")]
    fn demo_ui(&mut self, ui: &mut egui::Ui, id_prefix: String) {
        let previous_preset = self.preset;

        ui.horizontal_wrapped(|ui| {
            for (index, preset) in PRESETS.iter().enumerate() {
                ui.selectable_value(&mut self.preset, index as i32, preset.name());
            }
        });

        ui.horizontal_wrapped(|ui| {
            ui.label("Excite:");
            for (index, excitation) in EXCITATIONS.iter().enumerate() {
                ui.selectable_value(&mut self.excitation, index as i32, excitation.name());
            }
            ui.separator();
            ui.label("String:");
            for (index, core) in CORES.iter().enumerate() {
                ui.selectable_value(&mut self.core, index as i32, core.name());
            }
        });

        egui::Grid::new(format!("{id_prefix}_string_grid"))
            .num_columns(2)
            .show(ui, |ui| {
                ui.add(egui::Slider::new(&mut self.gain, 0.0..=2.0).text("Gain"));
                ui.end_row();
                ui.add(egui::Slider::new(&mut self.damping, 0.1..=4.0).text("Damping"));
                ui.end_row();
                ui.add(egui::Slider::new(&mut self.brightness, 0.1..=4.0).text("Brightness"));
                ui.end_row();
                ui.add(
                    egui::Slider::new(&mut self.inharmonicity, 0.0..=4.0).text("Inharmonicity"),
                );
                ui.end_row();
                ui.add(
                    egui::Slider::new(&mut self.position, 0.01..=0.5)
                        .text("Excitation point")
                        .logarithmic(true),
                );
                ui.end_row();
                ui.add(egui::Slider::new(&mut self.strings, 1..=3).text("Strings"));
                ui.end_row();
                ui.add(egui::Slider::new(&mut self.detune_cents, 0.0..=20.0).text("Detune (cents)"));
                ui.end_row();
                ui.add(egui::Slider::new(&mut self.hardness, 0.0..=1.0).text("Hardness"));
                ui.end_row();
                ui.add(egui::Slider::new(&mut self.bow_force, 0.0..=1.0).text("Bow force"));
                ui.end_row();
                ui.add(egui::Slider::new(&mut self.bow_speed, 0.0..=1.0).text("Bow speed"));
                ui.end_row();
                ui.add(egui::Slider::new(&mut self.body, 0.0..=1.0).text("Soundboard"));
                ui.end_row();
                ui.add(
                    egui::Slider::new(&mut self.sympathetic, 0.0..=1.0).text("Sympathetic"),
                );
                ui.end_row();
                ui.add(egui::Slider::new(&mut self.pedal, 0.0..=1.0).text("Sustain pedal"));
                ui.end_row();
                ui.add(
                    egui::Slider::new(&mut self.transpose, -24.0..=12.0)
                        .text("Transpose (semitones)")
                        .step_by(1.0),
                );
                ui.end_row();
                ui.add(
                    egui::Slider::new(&mut self.vibrato_rate, 0.0..=12.0).text("Vibrato rate"),
                );
                ui.end_row();
                ui.add(
                    egui::Slider::new(&mut self.vibrato_depth, 0.0..=120.0)
                        .text("Vibrato depth (cents)"),
                );
                ui.end_row();
                ui.add(
                    egui::Slider::new(&mut self.vibrato_delay, 0.0..=1.0).text("Vibrato delay"),
                );
                ui.end_row();
            });

        ui.label(format!("{} voices sounding", self.engine.active_voices()));

        if self.preset != previous_preset {
            self.load_preset(self.preset);
        }
    }
}
