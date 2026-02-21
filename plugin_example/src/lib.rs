use i_am_dsp::prelude::{Parameters, SawWave, SineWave, SquareWave, TriangleWave, WaveTable};
use i_am_dsp_iced::{Processor, plugins::wavetable_synth::{WavetableSynth, WavetableSynthMessage, WavetableSynthView}};
use i_am_plugin::{Descriptor, MidiPort, Plugin, Tag, WindowOptions, export_clap};

pub struct SynthWrapper(WavetableSynth);

impl Parameters for SynthWrapper {
    fn get_parameters(&self) -> Vec<i_am_dsp::prelude::Parameter> {
        self.0.get_parameters()
    }

    fn set_parameter(&mut self, identifier: &str, value: i_am_dsp::prelude::SetValue) -> bool {
        self.0.set_parameter(identifier, value)
    }
}

impl Processor for SynthWrapper {
    type Message = WavetableSynthMessage;
    type SyncedView = WavetableSynthView;

    fn delay(&self) -> usize {
        self.0.delay()
    }

    fn on_message(&self, message: Self::Message) {
        self.0.on_message(message)
    }

    fn process(&mut self, samples: &mut [f32; 2], other: &[[f32; 2]], process_context: &mut Box<dyn i_am_dsp::ProcessContext>) {
        self.0.process(samples, other, process_context)
    }

    fn synced_view(&self) -> Self::SyncedView {
        self.0.synced_view()
    }
}

impl Plugin for SynthWrapper {
    const DESCRIPTOR: i_am_plugin::Descriptor = 
        Descriptor::new("iamdsp.example.wavetable.synth", "I Am Table Synth")
            .with_tags(&[Tag::Instrument, Tag::Synthesizer])
            .with_vendor("iamplugins")
            .with_version(env!("CARGO_PKG_VERSION"))
    ;

    const INPUT_PORTS: &'static [i_am_plugin::AudioPort] = &[];

    const INPUT_MIDI_PORTS: &'static [MidiPort] = &[MidiPort {
        name: "Input",
    }];

    fn new() -> Self {
        SynthWrapper(WavetableSynth::new(44100, |_| {
            vec![
                Box::new(SineWave) as Box<dyn WaveTable + Send + Sync>,
                Box::new(TriangleWave) as Box<dyn WaveTable + Send + Sync>,
                Box::new(SawWave) as Box<dyn WaveTable + Send + Sync>,
                Box::new(SquareWave) as Box<dyn WaveTable + Send + Sync>,
            ]
        }))
    }

    fn window_options() -> WindowOptions {
        WindowOptions::new().with_size((720.0, 560.0)).with_scale_factor(1.0)
    }

    fn param_map(&self) -> i_am_dsp::prelude::ParamMap {
        self.0.param_map()
    }
}

export_clap!(SynthWrapper);