//! The effects, in a window, with something in front of them to hear.
//!
//! An effect with nothing to process is silent, so this puts a test source in front of
//! one of them: a saw at 110 Hz, which has enough harmonics that a filter, a crusher
//! and a drive all have something to do to it.
//!
//! ```text
//! cargo run -p i_am_effects --example effects_demo --features standalone
//! cargo run -p i_am_effects --example effects_demo --features standalone -- drive
//! cargo run -p i_am_effects --example effects_demo --features standalone -- crusher
//! ```
//!
//! The window is the effect's own faceplate -- the knobs, the sections, and the picture
//! each effect asked for -- because that is what a product shows a person. Nothing in
//! this file draws a control.

#[cfg(feature = "standalone")]
fn main() {
	let which = std::env::args().nth(1).unwrap_or_else(|| "filter".to_string());

	match which.as_str() {
		"crusher" => run::<i_am_effects::Crusher>(),
		"drive" => run::<i_am_effects::Drive>(),
		"filter" => run::<i_am_effects::Filter>(),
		other => {
			println!("no effect called {other}: running the filter");

			run::<i_am_effects::Filter>()
		},
	}
}

#[cfg(not(feature = "standalone"))]
fn main() {
	println!("the standalone feature is off: nothing to run");
}

#[cfg(feature = "standalone")]
use i_am_dsp_iced::iced;

#[cfg(feature = "standalone")]
fn run<P>()
where
	P: i_am_dsp_iced::Processor + i_am_plugin::Plugin + Default + 'static,
{
	use i_am_dsp_iced::demo::Demo;
	use i_am_dsp_iced::styles::theme;

	// The window is the size the effect itself asked for, in the options it hands a
	// host.
	let options = P::window_options();
	let size = options.window_size;
	let title = options.title.clone();

	iced::application(
		move || Demo::new(|sample_rate| Source::new(P::default(), sample_rate)),
		Demo::<Source<P>>::update,
		Demo::<Source<P>>::view,
	)
	.subscription(|_| Demo::<Source<P>>::subscriber())
	.theme(theme())
	.window_size(iced::Size::new(size.width as f32, size.height as f32))
	// Annotated because iced asks for a title for *any* borrow of the state, and an
	// unannotated closure is only inferred for one of them.
	.title(move |_: &Demo<Source<P>>| title.clone())
	.run()
	.expect("the demo window opens");
}

/// The test source, in front of the effect.
///
/// It is built for the rate the sound card runs at, which is what a host does: the
/// plug-in is made first and told what it will run at afterwards, through the same
/// hook a CLAP host uses.
struct Source<P> {
	effect: P,
	phase: f32,
	sample_rate: usize,
}

impl<P: i_am_plugin::Plugin> Source<P> {
	fn new(mut effect: P, sample_rate: usize) -> Self {
		effect.on_activate(sample_rate);

		Self {
			effect,
			phase: 0.0,
			sample_rate,
		}
	}
}

impl<P: i_am_dsp_iced::Processor> i_am_dsp::prelude::Parameters for Source<P> {
	fn get_parameters(&self) -> Vec<i_am_dsp::prelude::Parameter> {
		self.effect.get_parameters()
	}

	fn set_parameter(&mut self, identifier: &str, value: i_am_dsp::prelude::SetValue) -> bool {
		self.effect.set_parameter(identifier, value)
	}
}

impl<P: i_am_dsp_iced::Processor> i_am_dsp_iced::Processor for Source<P> {
	type Message = P::Message;
	type SyncedView = P::SyncedView;

	fn delay(&self) -> usize {
		self.effect.delay()
	}

	fn process(
		&mut self,
		samples: &mut [f32; 2],
		other: &[[f32; 2]],
		process_context: &mut Box<dyn i_am_dsp::ProcessContext>,
	) {
		// A saw: every harmonic of 110 Hz, so a filter has something to take away, a
		// crusher something to fold, and a drive something to bend.
		//
		// One sample per call, because that is how a processor is handed audio here:
		// the demo host calls this once per frame and the source keeps its phase.
		let value = (self.phase * 2.0 - 1.0) * 0.25;

		self.phase = (self.phase + 110.0 / self.sample_rate as f32) % 1.0;
		samples[0] = value;
		samples[1] = value;

		// And then whatever the window is showing, done to it.
		self.effect.process(samples, other, process_context);
	}

	fn on_message(&self, message: Self::Message) {
		self.effect.on_message(message);
	}

	fn synced_view(&self) -> Self::SyncedView {
		self.effect.synced_view()
	}
}
