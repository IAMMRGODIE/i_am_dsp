//! What a host sees when it loads a library.
//!
//! This is the other end of [`export_clap`](i_am_plugin::export_clap): it loads a
//! built plug-in library through clack's host API and asks its factory what it has.
//! Nothing here knows what is inside the library -- which is the point, because that
//! is also true of the host that will load it in the real world.
//!
//! Run it against a built library:
//!
//! ```text
//! cargo run -p i_am_plugin --example host_probe -- path/to/library.clap
//! cargo run -p i_am_plugin --example host_probe -- path/to/library.clap --activate
//! ```
//!
//! With `--activate` it goes further than scanning: it instantiates the first
//! plug-in, activates it (which is where a host says what sample rate it will run at)
//! and starts processing. That is the only way to find out whether the activation
//! path works, because everything on it is invisible from the outside.
//!
//! There is a library to point it at in `tests/set_library`: a fixture crate that
//! exports two plug-ins from one binary, with its own manifest so that it can be a
//! cdylib without being a member of this workspace. From the workspace root:
//!
//! ```text
//! cargo build --manifest-path i_am_plugin/tests/set_library/Cargo.toml
//! cargo run -p i_am_plugin --example host_probe -- \
//!     i_am_plugin/tests/set_library/target/debug/set_library.dll
//! ```
//!
//! ...and it prints both of them, which is the whole of what "a bundle exports more
//! than one plug-in" means to a host.

use clack_host::host::HostInfo;
use clack_host::plugin::PluginInstance;
use clack_host::prelude::{
	AudioPortBuffer, AudioPortBufferType, AudioPorts, EventBuffer, HostHandlers, InputChannel, InputEvents,
	OutputEvents, PluginAudioConfiguration, PluginEntry,
};

/// The sample rate the probe asks a plug-in to run at.
///
/// Deliberately not 44.1 kHz: a plug-in that takes its rate from its constructor
/// rather than from activation would be built for the wrong one, and the point of
/// asking for this number is that a person can see the difference.
const SAMPLE_RATE: f64 = 96_000.0;

/// A host with nothing to say.
///
/// clack implements the three handler traits for `()`, which is all a probe needs:
/// no GUI, no timers, no callbacks the plug-in has to be answered on.
struct ProbeHost;

impl HostHandlers for ProbeHost {
	type Shared<'a> = ();
	type MainThread<'a> = ();
	type AudioProcessor<'a> = ();
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
	let mut args = std::env::args().skip(1);
	let path = args.next().expect("a path to a plug-in library");
	let rest: Vec<String> = args.collect();
	let activate = rest.iter().any(|argument| argument == "--activate");
	// --process asks for the plug-in at the given index to be run, not just scanned.
	let process_index = rest
		.iter()
		.position(|argument| argument == "--process")
		.and_then(|position| rest.get(position + 1))
		.map(|index| index.parse::<u32>().unwrap_or(0));

	// SAFETY: loading a library runs its code. That is what a host does when it
	// scans a plug-in folder, and it is why a plug-in is trusted software.
	let entry = unsafe { PluginEntry::load(path) }?;
	let factory = entry.get_plugin_factory().expect("a CLAP factory");

	println!("the library exports {} plug-in(s)", factory.plugin_count());

	for index in 0..factory.plugin_count() {
		let descriptor = factory.plugin_descriptor(index).expect("a descriptor");

		println!(
			"  {index}: {} ({})",
			descriptor.name().unwrap_or_default().to_string_lossy(),
			descriptor.id().unwrap_or_default().to_string_lossy(),
		);
	}

	if activate {
		run_one(&entry)?;
	}

	if let Some(index) = process_index {
		measure(&entry, index)?;
	}

	Ok(())
}

/// Processes a block of a tone through one of the plug-ins, so that what the host asked
/// for can be seen in what comes out.
///
/// A host cannot look inside a plug-in, and the sample rate is the clearest example of
/// why that matters: nothing about a plug-in's interface changes when it is activated
/// at 96 kHz, so the only place the difference shows up is the audio.
///
/// `I Am Filter` is the one to point this at. Its corner is 1 kHz, so a 3 kHz tone
/// comes out about 19 dB down -- unless the filter was built for 44.1 kHz, when its
/// corner sits at 2.2 kHz and the same tone passes nearly untouched. The numbers this
/// prints are the difference: roughly 0.04 out for a filter built for the rate the host
/// asked for, and roughly 0.24 for one built for the default.
fn measure(entry: &PluginEntry, index: u32) -> Result<(), Box<dyn std::error::Error>> {
	const FRAMES: usize = 512;
	const TONE: f32 = 3_000.0;
	const LEVEL: f32 = 0.5;

	let factory = entry.get_plugin_factory().expect("a CLAP factory");
	let descriptor = factory.plugin_descriptor(index).expect("a descriptor");
	let id = descriptor.id().expect("an id");
	let name = descriptor.name().unwrap_or_default().to_string_lossy().into_owned();

	let host_info = HostInfo::new(
		"i_am_dsp probe",
		"i_am_dsp",
		"https://example.com/i-am-dsp",
		env!("CARGO_PKG_VERSION"),
	)?;
	let mut instance = PluginInstance::<ProbeHost>::new(|_| (), |_| (), entry, id, &host_info)?;

	let configuration = PluginAudioConfiguration {
		sample_rate: SAMPLE_RATE,
		min_frames_count: 32,
		max_frames_count: FRAMES as u32,
	};
	let mut started = instance.activate(|_, _| (), configuration)?.start_processing()?;

	let mut input = [[0.0f32; FRAMES]; 2];
	let mut output = [[0.0f32; FRAMES]; 2];

	let tone: [f32; FRAMES] = std::array::from_fn(|frame| {
		let time = frame as f32 / SAMPLE_RATE as f32;

		LEVEL * (std::f32::consts::TAU * TONE * time).sin()
	});

	// Both channels carry the same thing, which is what a mono source in a stereo host
	// looks like.
	input[0] = tone;
	input[1] = tone;

	let mut input_ports = AudioPorts::with_capacity(2, 1);
	let mut output_ports = AudioPorts::with_capacity(2, 1);
	// Nothing plays into these effects, so the input has no events; the output buffer
	// is where a plug-in would put what it has to say.
	let input_events_buffer: [clack_plugin::events::event_types::NoteOnEvent; 0] = [];
	let input_events = InputEvents::from_buffer(&input_events_buffer);
	let mut output_events_buffer = EventBuffer::new();
	let mut output_events = OutputEvents::from_buffer(&mut output_events_buffer);

	let input_audio = input_ports.with_input_buffers([AudioPortBuffer {
		latency: 0,
		channels: AudioPortBufferType::f32_input_only(
			input.iter_mut().map(InputChannel::constant),
		),
	}]);
	let mut output_audio = output_ports.with_output_buffers([AudioPortBuffer {
		latency: 0,
		channels: AudioPortBufferType::f32_output_only(output.iter_mut().map(|channel| channel.as_mut_slice())),
	}]);

	let status = started.process(
		&input_audio,
		&mut output_audio,
		&input_events,
		&mut output_events,
		None,
		None,
	)?;

	// What came out, measured the way a meter measures: the level of the block.
	let rms = (output[0].iter().map(|sample| sample * sample).sum::<f32>() / FRAMES as f32).sqrt();

	println!(
		"{name} at {} Hz: {TONE} Hz in at {LEVEL}, {rms:.4} rms out ({status:?})",
		SAMPLE_RATE as usize,
	);

	instance.deactivate(started.stop_processing());

	Ok(())
}

/// Instantiates the first plug-in and takes it through activation.
fn run_one(entry: &PluginEntry) -> Result<(), Box<dyn std::error::Error>> {
	let factory = entry.get_plugin_factory().expect("a CLAP factory");
	let descriptor = factory.plugin_descriptor(0).expect("a descriptor");
	let id = descriptor.id().expect("an id");
	let name = descriptor.name().unwrap_or_default().to_string_lossy().into_owned();

	let host_info = HostInfo::new(
		"i_am_dsp probe",
		"i_am_dsp",
		"https://example.com/i-am-dsp",
		env!("CARGO_PKG_VERSION"),
	)?;

	let mut instance = PluginInstance::<ProbeHost>::new(
		|_| (),
		|_| (),
		entry,
		id,
		&host_info,
	)?;

	// Activating is where the host says what it runs at, and the plugin is handed
	// that configuration -- whatever it does with it happens here.
	let configuration = PluginAudioConfiguration {
		sample_rate: SAMPLE_RATE,
		min_frames_count: 32,
		max_frames_count: 512,
	};
	let stopped = instance.activate(|_, _| (), configuration)?;

	println!("activated {name} at {} Hz", SAMPLE_RATE as usize);

	// Starting is the other half of the same path: a plug-in that allocates on
	// activation and expects to be started before processing does it here.
	let started = stopped.start_processing()?;

	println!("started processing");

	// And back down the same path, which is where a plug-in frees what it took.
	instance.deactivate(started.stop_processing());

	println!("deactivated, and the host still has its plugin");

	Ok(())
}
